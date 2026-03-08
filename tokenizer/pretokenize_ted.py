#!/usr/bin/env python3
# Pretokenize TED2025 multi-way JSONL into packed causal-LM chunks on disk
#
# pip install -U "transformers>=4.41" "datasets>=2.19"
#
# Output:
#   HF dataset on disk with columns:
#     - input_ids
#     - attention_mask
#     - labels
#
# Then load with:
#   from datasets import load_from_disk
#   ds = load_from_disk(OUTDIR)

import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Iterator

from datasets import Dataset
from transformers import AutoTokenizer


# -----------------
# Config
# -----------------
JSONL_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/TED2025/multi_way.jsonl"

BASE_MODEL_LLAMA = "meta-llama/Llama-2-7b-hf"
BASE_MODEL_OLMO = "allenai/OLMo-7B-hf"
OLMO_3_7B = "allenai/Olmo-3-7B-Instruct"
OLMO_3_7B_BASE = "allenai/Olmo-3-1025-7B"
OLMO_2_7B_BASE = "allenai/OLMo-2-1124-7B"

BASE_MODEL = OLMO_2_7B_BASE  # <-- change this

REQ_LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh-cn"]
MIN_LANGS_PER_ROW = 1
MIN_LANGS_PER_TALK = 2

USE_TAGS = False
CHUNK_TOKENS = 1024

OUTDIR = (
    f"/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/"
    f"{BASE_MODEL.split('/')[-1]}-ted2025-pretokenized-num-langs-{len(REQ_LANGS)}"
    f"-chunk-{CHUNK_TOKENS}"
    + ("-tags" if USE_TAGS else "-notags")
)


# -----------------
# Tokenizer
# -----------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# -----------------
# Helpers
# -----------------
def nonempty_str(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def prune_to_selected_langs(para_data: Dict, selected_langs: List[str]) -> Dict:
    pd = para_data or {}
    return {l: pd[l] for l in selected_langs if nonempty_str(pd.get(l))}


def n_selected_langs_in_row(obj: Dict, selected_langs: List[str]) -> int:
    return len(prune_to_selected_langs(obj.get("para_data", {}), selected_langs))


def has_at_least_k_langs(obj: Dict, selected_langs: List[str], k: int = 2) -> bool:
    return n_selected_langs_in_row(obj, selected_langs) >= k


def eligible_talk_ids(path: str, selected_langs: List[str], k: int = 2) -> Set[str]:
    talk_to_langs = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = obj.get("talk_id")
            if not tid:
                continue
            pd = prune_to_selected_langs(obj.get("para_data", {}), selected_langs)
            talk_to_langs[tid].update(pd.keys())
    return {tid for tid, langs in talk_to_langs.items() if len(langs) >= k}


def format_segment(para_data, use_tags: bool, lang_order=None):
    """
    Turn one JSON row's para_data into text for next-token prediction.
    - use_tags=False: just values concatenated
    - use_tags=True: "<en>\\ntext" style tags
    """
    pd = prune_to_selected_langs(
        para_data,
        lang_order if lang_order is not None else REQ_LANGS,
    )
    if not pd:
        return ""

    if lang_order is None:
        langs = list(pd.keys())
    else:
        langs = [l for l in lang_order if l in pd]

    if use_tags:
        parts = [f"<{l}>\n{pd[l]}" for l in langs]
    else:
        parts = [pd[l] for l in langs]

    if tok.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; please set one explicitly.")

    return "\n\n".join(parts) + tok.eos_token


def talk_chunk_generator(
    jsonl_path: str,
    tokenizer,
    use_tags: bool,
    chunk_tokens: int,
    lang_order=None,
    eligible_talks: Set[str] = None,
    min_langs_per_row: int = 2,
) -> Iterator[Dict]:
    """
    Same logic as your training pipeline:
    - keep only eligible talks
    - keep only rows with >= min_langs_per_row selected langs
    - format each row into a segment
    - accumulate segments within a talk
    - flush on talk boundary
    - don't exceed chunk_tokens
    - if a single segment is already >= chunk_tokens, emit it alone
    """
    buffer_text = ""
    buffer_tok = 0
    current_talk = None

    def flush():
        nonlocal buffer_text, buffer_tok
        if buffer_text:
            encoded = tokenizer(
                buffer_text,
                add_special_tokens=False,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.copy(),
                "text_len_chars": len(buffer_text),
                "text_len_tokens": len(input_ids),
                "talk_id": current_talk,
            }

        buffer_text, buffer_tok = "", 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            obj = json.loads(line)

            talk_id = obj.get("talk_id")
            if not talk_id:
                continue

            if eligible_talks is not None and talk_id not in eligible_talks:
                continue

            pd_sel = prune_to_selected_langs(obj.get("para_data", {}), REQ_LANGS)
            if len(pd_sel) < min_langs_per_row:
                continue

            seg = format_segment(pd_sel, use_tags=use_tags, lang_order=lang_order)
            if not seg:
                continue

            seg_ids = tokenizer.encode(seg, add_special_tokens=False)
            seg_tok = len(seg_ids)

            if current_talk is None:
                current_talk = talk_id
            elif talk_id != current_talk:
                yield from flush()
                current_talk = talk_id

            # single row already too large -> emit on its own
            if seg_tok >= chunk_tokens:
                yield from flush()
                yield {
                    "input_ids": seg_ids,
                    "attention_mask": [1] * len(seg_ids),
                    "labels": seg_ids.copy(),
                    "text_len_chars": len(seg),
                    "text_len_tokens": len(seg_ids),
                    "talk_id": talk_id,
                }
                continue

            if buffer_tok + seg_tok > chunk_tokens:
                yield from flush()
                buffer_text = seg
                buffer_tok = seg_tok
            else:
                buffer_text += seg
                buffer_tok += seg_tok

    yield from flush()


def main():
    os.makedirs(os.path.dirname(OUTDIR), exist_ok=True)

    print("=" * 80)
    print("STEP 1: FIND ELIGIBLE TALKS")
    print("=" * 80)
    eligible_talks = eligible_talk_ids(
        JSONL_PATH,
        REQ_LANGS,
        k=MIN_LANGS_PER_TALK,
    )
    print(f"Eligible talks: {len(eligible_talks):,}")

    print("\n" + "=" * 80)
    print("STEP 2: BUILD TOKENIZED CHUNKS")
    print("=" * 80)

    rows = []
    total_tokens = 0

    for i, ex in enumerate(
        talk_chunk_generator(
            JSONL_PATH,
            tokenizer=tok,
            use_tags=USE_TAGS,
            chunk_tokens=CHUNK_TOKENS,
            lang_order=REQ_LANGS,
            eligible_talks=eligible_talks,
            min_langs_per_row=MIN_LANGS_PER_ROW,
        )
    ):
        rows.append(ex)
        total_tokens += ex["text_len_tokens"]

        if (i + 1) % 1000 == 0:
            print(
                f"Built {i+1:,} chunks | total tokens so far: {total_tokens:,}",
                flush=True,
            )

    print(f"\nFinal number of chunks: {len(rows):,}")
    print(f"Final total tokens: {total_tokens:,}")

    if len(rows) == 0:
        raise RuntimeError("No chunks were produced. Check your filtering settings.")

    print("\n" + "=" * 80)
    print("STEP 3: SAVE DATASET")
    print("=" * 80)

    ds = Dataset.from_list(rows)
    ds.save_to_disk(OUTDIR)

    print(f"Saved tokenized dataset to:\n{OUTDIR}")

    print("\nSample example:")
    ex = ds[0]
    print("talk_id:", ex["talk_id"])
    print("len(input_ids):", len(ex["input_ids"]))
    print("first 50 tokens:", ex["input_ids"][:50])


if __name__ == "__main__":
    main()