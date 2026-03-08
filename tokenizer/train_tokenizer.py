#!/usr/bin/env python3
"""
Fast tokenizer extension for OLMo-2 using CulturaX (streaming) by *mining frequent script tokens*
and adding them via tokenizer.add_tokens().

This avoids expensive BPE merge training (which isn't truly incremental in HF tokenizers anyway).

Example:
  export HF_TOKEN=...
  python train_tokenizer.py \
    --base_model allenai/OLMo-2-1124-7B \
    --out_dir /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo2_tok_ext_mined_30k \
    --langs ar bn ru ja zh \
    --max_docs_per_lang 200000 \
    --min_chars 200 \
    --num_new_tokens 30000 \
    --ngram_max 3 \
    --max_script_chars_per_lang 30000000

Then for CPT:
  tok = AutoTokenizer.from_pretrained(OUT_DIR, use_fast=True)
  model.resize_token_embeddings(len(tok))
"""

import argparse
import os
from collections import Counter
from typing import Iterator, List, Optional, Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


# -------------------------
# CulturaX language mapping
# -------------------------
def map_to_culturax_config(lang: str) -> str:
    lang = lang.lower()
    if lang in ["zh-cn", "zh-hans", "zh_simplified"]:
        return "zh"
    return lang


# -------------------------
# Script ranges (Unicode blocks)
# -------------------------
SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    # Arabic
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    # Bengali
    "bn": [(0x0980, 0x09FF)],
    # Cyrillic (Russian)
    "ru": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    # Japanese: Kana + CJK ideographs (Kanji live in CJK range)
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
    # Chinese: CJK ideographs
    "zh": [(0x4E00, 0x9FFF)],
}


def in_ranges(ch: str, ranges: List[Tuple[int, int]]) -> bool:
    o = ord(ch)
    for a, b in ranges:
        if a <= o <= b:
            return True
    return False


# -------------------------
# CulturaX streaming iterator (balanced)
# -------------------------
def iter_culturax_texts_balanced(
    langs: List[str],
    split: str = "train",
    max_docs_per_lang: int = 200_000,
    min_chars: int = 200,
    seed_skip: int = 0,
    hf_token: Optional[str] = None,
) -> Iterator[Tuple[str, str]]:
    """
    Yields (lang_cfg, text) in a round-robin fashion for roughly balanced language sampling.
    """
    cfgs = [map_to_culturax_config(l) for l in langs]

    iters = []
    for cfg in cfgs:
        ds = load_dataset(
            "uonlp/CulturaX",
            cfg,
            split=split,
            streaming=True,
            token=hf_token,  # or rely on HF_TOKEN env / huggingface-cli login
        )
        it = iter(ds)
        for _ in range(seed_skip):
            try:
                next(it)
            except StopIteration:
                break
        iters.append((cfg, it))

    yielded = {cfg: 0 for cfg in cfgs}
    alive = [True] * len(iters)

    while any(alive):
        for i, (cfg, it) in enumerate(iters):
            if not alive[i]:
                continue
            if yielded[cfg] >= max_docs_per_lang:
                alive[i] = False
                continue
            try:
                ex = next(it)
            except StopIteration:
                alive[i] = False
                continue

            txt = ex.get("text")
            if not isinstance(txt, str):
                continue
            txt = txt.strip()
            if len(txt) < min_chars:
                continue

            yielded[cfg] += 1
            yield cfg, txt


# -------------------------
# Mining frequent script ngrams
# -------------------------
def mine_script_ngrams(
    stream: Iterator[Tuple[str, str]],
    target_langs: List[str],
    ngram_max: int = 3,
    max_script_chars_per_lang: int = 30_000_000,
) -> Dict[str, Counter]:
    """
    For each lang, keep only characters in that script range, then count ngrams (1..ngram_max).
    This is a fast way to get useful tokens to add without learning BPE merges.
    """
    counters = {l: Counter() for l in target_langs}
    seen_chars = {l: 0 for l in target_langs}

    for cfg, text in stream:
        l = cfg  # already mapped
        if l not in counters:
            continue
        if seen_chars[l] >= max_script_chars_per_lang:
            continue

        ranges = SCRIPT_RANGES.get(l)
        if not ranges:
            continue

        # Filter to script chars only
        chars = [ch for ch in text if in_ranges(ch, ranges)]
        if not chars:
            continue

        if seen_chars[l] + len(chars) > max_script_chars_per_lang:
            chars = chars[: max(0, max_script_chars_per_lang - seen_chars[l])]
        seen_chars[l] += len(chars)

        s = "".join(chars)
        # 1-grams
        counters[l].update(s)

        # 2..ngram_max grams
        for n in range(2, ngram_max + 1):
            if len(s) >= n:
                counters[l].update(s[i : i + n] for i in range(len(s) - n + 1))

    return counters


def choose_tokens_to_add(
    counters: Dict[str, Counter],
    base_vocab: Dict[str, int],
    total_new_tokens: int,
) -> List[str]:
    """
    Pick tokens from per-language counters.
    Simple heuristic: allocate budget proportional to script “needs”.
    """
    # You can tune these weights
    weights = {"zh": 0.35, "ja": 0.20, "ar": 0.20, "bn": 0.15, "ru": 0.10}
    langs = list(counters.keys())
    denom = sum(weights.get(l, 1.0 / len(langs)) for l in langs)

    # Candidate pool (oversample per language then global dedupe)
    picked: List[str] = []
    seen = set()

    for l in langs:
        frac = weights.get(l, 1.0 / len(langs)) / denom
        budget = max(1, int(total_new_tokens * frac))

        # Oversample then filter
        for tok, _ in counters[l].most_common(budget * 5):
            if not tok:
                continue
            if tok.isspace():
                continue
            if tok in base_vocab:
                continue
            if tok in seen:
                continue
            # avoid super long ngrams that won't generalize
            if len(tok) > 12:
                continue

            seen.add(tok)
            picked.append(tok)
            if len(picked) >= total_new_tokens:
                return picked

    return picked[:total_new_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--langs", type=str, nargs="+", required=True,
                    help="Languages to mine tokens for (use CulturaX config codes; zh-cn will map to zh).")

    ap.add_argument("--max_docs_per_lang", type=int, default=200_000)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--seed_skip", type=int, default=0)

    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--num_new_tokens", type=int, default=30_000)
    ap.add_argument("--ngram_max", type=int, default=3)
    ap.add_argument("--max_script_chars_per_lang", type=int, default=30_000_000)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load base tokenizer (keep OLMo format!)
    base_tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if base_tok.pad_token is None and base_tok.eos_token is not None:
        base_tok.pad_token = base_tok.eos_token

    base_size = len(base_tok)
    print(f"[info] base_model={args.base_model}")
    print(f"[info] base vocab size={base_size}")

    # Map langs to CulturaX configs
    target_langs = [map_to_culturax_config(l) for l in args.langs]

    # Ensure we have script ranges for all requested langs
    missing = [l for l in target_langs if l not in SCRIPT_RANGES]
    if missing:
        raise ValueError(f"No SCRIPT_RANGES defined for: {missing}. Add them to SCRIPT_RANGES.")

    stream = iter_culturax_texts_balanced(
        langs=target_langs,
        max_docs_per_lang=args.max_docs_per_lang,
        min_chars=args.min_chars,
        seed_skip=args.seed_skip,
        hf_token=args.hf_token,
    )

    print("[info] mining frequent ngrams...")
    counters = mine_script_ngrams(
        stream=stream,
        target_langs=target_langs,
        ngram_max=args.ngram_max,
        max_script_chars_per_lang=args.max_script_chars_per_lang,
    )

    base_vocab = base_tok.get_vocab()
    to_add = choose_tokens_to_add(
        counters=counters,
        base_vocab=base_vocab,
        total_new_tokens=args.num_new_tokens,
    )

    print(f"[info] selected {len(to_add)} tokens to add (requested {args.num_new_tokens})")
    from transformers import AddedToken
    added = base_tok.add_tokens([AddedToken(t, normalized=False) for t in to_add])
    print(f"[info] tokenizer.add_tokens actually added: {added}")

    # Save (this will keep the tokenizer class compatible with OLMo)
    base_tok.save_pretrained(args.out_dir)
    print(f"[done] saved tokenizer to: {args.out_dir}")
    print(f"[done] final vocab size: {len(base_tok)}")


if __name__ == "__main__":
    main()