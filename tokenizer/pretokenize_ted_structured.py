#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pretokenize TED data using the exact same filtering / collection / formatting /
packing logic as in LoRA_training_olmo2.ipynb, then save as DatasetDict with:
  - train
  - validation (tiny split)

Important:
- This mirrors the notebook logic:
    * para_data-based language extraction
    * eligible talk pre-scan
    * randomized pair direction using --seed
    * talk-boundary-aware packing
    * char-based token estimate: len(seg) // 4
    * tokenization only after chunk creation
    * truncation=True, max_length=MAX_LENGTH
- The notebook used an IterableDataset and no validation split.
  Here we materialize the chunks into a normal DatasetDict and carve out a tiny
  validation split from the final packed chunks.
  
  
python3 tokenizer/pretokenize_ted_structured.py \
  --model_id qwen2.5-7b \
  --include_samanantar_bn \
  --include_rogendo_sw \
  --hf_max_examples_per_source 1000000 
  


Default output:
    /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/ted_single_target_12_langs
"""

import os
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


MODEL_ID = "allenai/OLMo-2-1124-7B"
OUT_ROOT = "/data/jonathan/Lost-in-Mistranslation/datasets/tokenized"
OUT_PATH = ""
JSONL_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/TED2025/multi_way.jsonl"

MODEL_ALIASES = {
    "olmo2-7b": "allenai/OLMo-2-1124-7B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "gemma-2-9b": "google/gemma-2-9b",
}

REQ_LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh-cn", "ja", "ar", "sw", "bn"]
LANG_ORDER = REQ_LANGS

USE_TAGS = False
CHUNK_TOKENS = 512
MAX_LENGTH = 1024
MIN_LANGS_PER_ROW = 2
MIN_LANGS_PER_TALK = 2


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def resolve_model_id(model_id_or_alias: str) -> str:
    return MODEL_ALIASES.get(model_id_or_alias, model_id_or_alias)


def default_out_path(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    return os.path.join(OUT_ROOT, f"{model_name}_ted_12_langs_extra_bn_sw")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--jsonl_path", type=str, default=JSONL_PATH)
    parser.add_argument("--out_path", type=str, default=OUT_PATH)
    parser.add_argument("--chunk_tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--min_langs_per_row", type=int, default=MIN_LANGS_PER_ROW)
    parser.add_argument("--min_langs_per_talk", type=int, default=MIN_LANGS_PER_TALK)
    parser.add_argument("--val_ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_rogendo_sw", action="store_true")
    parser.add_argument("--include_samanantar_bn", action="store_true")
    parser.add_argument("--hf_max_examples_per_source", type=int, default=200000)
    return parser.parse_args()


def nonempty_str(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def prune_to_selected_langs(para_data, selected_langs):
    pd = para_data or {}
    return {l: pd[l] for l in selected_langs if nonempty_str(pd.get(l))}


def eligible_talk_ids(path, selected_langs, k=2):
    """Pre-scan: keep only talks that have >= k selected langs across all rows."""
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


def make_format_segment(tokenizer, rng):
    def format_segment(para_data):
        """
        All available languages once per row, with the language order randomly
        shuffled per row:

          en: {en_sent}
          de: {de_sent}
          es: {es_sent}
          ...

        Each language string appears exactly once, so the model can't shortcut
        x->en translation by attending to a duplicated `en: ...` line. Random
        order ensures both en->x and x->en directions are learned (when en
        lands after a non-English language, predicting the English line
        requires translating from the preceding non-English context).
        """
        pd = prune_to_selected_langs(para_data, LANG_ORDER)
        if not pd:
            return ""

        langs = [l for l in LANG_ORDER if l in pd]
        rng.shuffle(langs)

        lines = [f"{l}: {pd[l]}" for l in langs]
        return "\n".join(lines) + tokenizer.eos_token

    return format_segment


def format_pair(src_lang: str, src_text: str, tgt_lang: str, tgt_text: str, eos_token: str, rng) -> str:
    if rng.random() < 0.5:
        return f"{src_lang}: {src_text}\n{tgt_lang}: {tgt_text}" + eos_token
    return f"{tgt_lang}: {tgt_text}\n{src_lang}: {src_text}" + eos_token


def collect_parallel_hf_chunks(
    dataset_name: str,
    config_name: str,
    split: str,
    src_col: str,
    tgt_col: str,
    tgt_lang: str,
    chunk_tokens: int,
    eos_token: str,
    rng,
    max_examples: int,
) -> List[Dict[str, str]]:
    """
    Stream one Hugging Face parallel corpus and pack sentence pairs into the
    same {"text": ...} chunk shape used by the TED collector.
    """
    load_kwargs = {
        "split": split,
        "streaming": True,
    }
    if config_name:
        load_kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **load_kwargs)

    chunks = []
    buffer_text = ""
    buffer_tok = 0
    seen = 0

    def flush():
        nonlocal buffer_text, buffer_tok
        if buffer_text:
            chunks.append({"text": buffer_text, "langs": ["en", tgt_lang], "source": dataset_name})
        buffer_text, buffer_tok = "", 0

    for ex in ds:
        src_text = ex.get(src_col)
        tgt_text = ex.get(tgt_col)
        if not nonempty_str(src_text) or not nonempty_str(tgt_text):
            continue

        seen += 1
        if max_examples is not None and seen > max_examples:
            break

        seg = format_pair("en", src_text.strip(), tgt_lang, tgt_text.strip(), eos_token, rng)
        seg_tok = len(seg) // 4

        if seg_tok >= chunk_tokens:
            flush()
            chunks.append({"text": seg, "langs": ["en", tgt_lang], "source": dataset_name})
        elif buffer_tok + seg_tok > chunk_tokens:
            flush()
            buffer_text = seg
            buffer_tok = seg_tok
        else:
            buffer_text += seg
            buffer_tok += seg_tok

    flush()
    return chunks


def collect_chunks(
    jsonl_path: str,
    eligible_talks,
    format_segment,
    chunk_tokens: int,
    min_langs_per_row: int,
) -> List[Dict[str, str]]:
    """
    Exact notebook packing logic:
    - iterate rows in file order
    - only keep eligible talks
    - filter by min langs per row
    - build segment string via format_segment
    - estimate token count with len(seg) // 4
    - pack within talk boundaries
    """
    chunks = []

    buffer_text = ""
    buffer_tok = 0
    buffer_langs = set()
    current_talk = None

    def flush():
        nonlocal buffer_text, buffer_tok, buffer_langs
        if buffer_text:
            chunks.append({"text": buffer_text, "langs": sorted(buffer_langs), "source": "ted"})
        buffer_text, buffer_tok, buffer_langs = "", 0, set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:

            if not line.strip():
                continue

            obj = json.loads(line)
            talk_id = obj.get("talk_id")

            if not talk_id or talk_id not in eligible_talks:
                continue

            pd_sel = prune_to_selected_langs(obj.get("para_data", {}), REQ_LANGS)
            if len(pd_sel) < min_langs_per_row:
                continue

            seg = format_segment(pd_sel)
            if not seg:
                continue

            seg_tok = len(seg) // 4  # exact notebook estimate

            if current_talk is None:
                current_talk = talk_id
            elif talk_id != current_talk:
                flush()
                current_talk = talk_id

            seg_langs = set(pd_sel.keys())

            if seg_tok >= chunk_tokens:
                flush()
                chunks.append({"text": seg, "langs": sorted(seg_langs), "source": "ted"})
            elif buffer_tok + seg_tok > chunk_tokens:
                flush()
                buffer_text = seg
                buffer_tok = seg_tok
                buffer_langs = set(seg_langs)
            else:
                buffer_text += seg
                buffer_tok += seg_tok
                buffer_langs.update(seg_langs)

    flush()



    return chunks


def split_balanced_validation(chunks: List[Dict[str, str]], val_ratio: float, seed: int):
    """
    Build a validation split with equal primary-bucket coverage per non-English
    language. Chunks may contain multiple languages, especially TED chunks, so
    each chunk is assigned to the least-filled language bucket among the
    non-English languages it contains. English is omitted from balancing because
    nearly every parallel chunk contains it.
    """
    rng = random.Random(seed)
    buckets = {lang: [] for lang in LANG_ORDER if lang != "en"}
    other = []
    train_chunks = []
    val_chunks = []

    bucket_sizes = Counter()
    for chunk in chunks:
        chunk_langs = [lang for lang in chunk.get("langs", []) if lang in buckets]
        if not chunk_langs:
            other.append(chunk)
            continue

        bucket_lang = min(chunk_langs, key=lambda lang: bucket_sizes[lang])
        chunk["bucket_lang"] = bucket_lang
        buckets[bucket_lang].append(chunk)
        bucket_sizes[bucket_lang] += 1

    eligible_buckets = [bucket for bucket in buckets.values() if len(bucket) >= 2]
    if eligible_buckets:
        target_total_val = max(1, int(len(chunks) * val_ratio))
        per_lang_val = max(1, target_total_val // len(eligible_buckets))
        per_lang_val = min(per_lang_val, min(len(bucket) - 1 for bucket in eligible_buckets))
    else:
        per_lang_val = 0

    print(f"Balanced validation chunks per language bucket: {per_lang_val:,}")

    for lang, bucket in buckets.items():
        rng.shuffle(bucket)
        if len(bucket) < 2 or per_lang_val == 0:
            train_chunks.extend(bucket)
            continue

        n_val = min(per_lang_val, len(bucket) - 1)
        val_chunks.extend(bucket[:n_val])
        train_chunks.extend(bucket[n_val:])

    rng.shuffle(other)
    if len(other) >= 2:
        n_val = max(1, int(len(other) * val_ratio))
        n_val = min(n_val, len(other) - 1)
        val_chunks.extend(other[:n_val])
        train_chunks.extend(other[n_val:])
    else:
        train_chunks.extend(other)

    rng.shuffle(train_chunks)
    rng.shuffle(val_chunks)
    return train_chunks, val_chunks


def count_langs(chunks: List[Dict[str, str]]) -> Counter:
    counts = Counter()
    for chunk in chunks:
        for lang in chunk.get("langs", []):
            if lang != "en":
                counts[lang] += 1
    return counts


def count_bucket_langs(chunks: List[Dict[str, str]]) -> Counter:
    counts = Counter()
    for chunk in chunks:
        lang = chunk.get("bucket_lang")
        if lang:
            counts[lang] += 1
    return counts


def print_bucket_balance(chunks: List[Dict[str, str]], name: str):
    counts = count_bucket_langs(chunks)
    print(f"\n===== {name.upper()} PRIMARY LANGUAGE BUCKETS =====")
    for lang in LANG_ORDER:
        if lang == "en":
            continue
        print(f"{lang:5s} {counts.get(lang, 0):,}")


def print_lang_balance(chunks: List[Dict[str, str]], name: str):
    counts = count_langs(chunks)
    print(f"\n===== {name.upper()} LANGUAGE CHUNK COVERAGE =====")
    for lang in LANG_ORDER:
        if lang == "en":
            continue
        print(f"{lang:5s} {counts.get(lang, 0):,}")


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(element):
        return tokenizer(
            element["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def print_stats(ds: Dataset, name: str):
    lengths = [len(x) for x in ds["input_ids"]] if len(ds) > 0 else []
    total_tokens = sum(lengths)
    mean_len = total_tokens / len(lengths) if lengths else 0.0

    print(f"\n===== {name.upper()} =====")
    print(f"num_examples: {len(ds):,}")
    print(f"total_tokens: {total_tokens:,}")
    print(f"mean_len:     {mean_len:.2f}")
    if lengths:
        print(f"min_len:      {min(lengths)}")
        print(f"max_len:      {max(lengths)}")


def main():
    args = parse_args()
    args.model_id = resolve_model_id(args.model_id)
    if not args.out_path:
        args.out_path = default_out_path(args.model_id)
    os.makedirs(args.out_path, exist_ok=True)

    print(f"Loading tokenizer for {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    direction_rng = random.Random(args.seed)
    format_segment = make_format_segment(tokenizer, direction_rng)

    print("Pre-scanning JSONL for eligible talks...")
    eligible = eligible_talk_ids(
        args.jsonl_path,
        REQ_LANGS,
        k=args.min_langs_per_talk,
    )
    print(f"Eligible talks: {len(eligible)}")

    print("Collecting packed chunks with exact notebook logic...")
    chunks = collect_chunks(
        jsonl_path=args.jsonl_path,
        eligible_talks=eligible,
        format_segment=format_segment,
        chunk_tokens=args.chunk_tokens,
        min_langs_per_row=args.min_langs_per_row,
    )
    print(f"Collected TED chunks: {len(chunks):,}")

    eos_token = tokenizer.eos_token or ""

    if args.include_rogendo_sw:
        print("Collecting Rogendo English-Swahili sentence pairs...")
        sw_chunks = collect_parallel_hf_chunks(
            dataset_name="Rogendo/English-Swahili-Sentence-Pairs",
            config_name=None,
            split="train",
            src_col="English sentence",
            tgt_col="Swahili Translation",
            tgt_lang="sw",
            chunk_tokens=args.chunk_tokens,
            eos_token=eos_token,
            rng=direction_rng,
            max_examples=args.hf_max_examples_per_source,
        )
        chunks.extend(sw_chunks)
        print(f"Collected Rogendo sw chunks: {len(sw_chunks):,}")

    if args.include_samanantar_bn:
        print("Collecting Samanantar Bengali sentence pairs...")
        bn_chunks = collect_parallel_hf_chunks(
            dataset_name="ai4bharat/samanantar",
            config_name="bn",
            split="train",
            src_col="src",
            tgt_col="tgt",
            tgt_lang="bn",
            chunk_tokens=args.chunk_tokens,
            eos_token=eos_token,
            rng=direction_rng,
            max_examples=args.hf_max_examples_per_source,
        )
        chunks.extend(bn_chunks)
        print(f"Collected Samanantar bn chunks: {len(bn_chunks):,}")

    print(f"Collected total chunks: {len(chunks):,}")

    if len(chunks) < 2:
        raise ValueError("Not enough chunks collected to create train/validation split.")

    train_chunks, val_chunks = split_balanced_validation(chunks, args.val_ratio, args.seed)
    print_bucket_balance(train_chunks, "train")
    print_bucket_balance(val_chunks, "validation")
    print_lang_balance(train_chunks, "train raw")
    print_lang_balance(val_chunks, "validation raw")

    raw = DatasetDict({
        "train": Dataset.from_list(train_chunks),
        "validation": Dataset.from_list(val_chunks),
    })

    print("\nTokenizing train split...")
    train_tok = tokenize_dataset(raw["train"], tokenizer, args.max_length)

    print("Tokenizing validation split...")
    val_tok = tokenize_dataset(raw["validation"], tokenizer, args.max_length)

    final_ds = DatasetDict({
        "train": train_tok,
        "validation": val_tok,
    })

    print_stats(final_ds["train"], "train")
    print_stats(final_ds["validation"], "validation")

    print(f"\nSaving dataset to:\n{args.out_path}")
    final_ds.save_to_disk(args.out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()