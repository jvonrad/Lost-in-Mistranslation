#!/usr/bin/env python3

# '''
# python training/train_tokenizer.py \
#   --base_model allenai/OLMo-2-1124-7B \
#   --out_dir /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo2_tok_ext_ar_bn_ru \
#   --langs ar bn ru \
#   --num_new_tokens 10000
# '''

import argparse
import os
import re
import unicodedata
from collections import Counter
from typing import Iterator, List, Optional, Dict, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, AddedToken


# -------------------------
# CulturaX language mapping
# -------------------------
def map_to_culturax_config(lang: str) -> str:
    lang = lang.lower()
    if lang in ["zh-cn", "zh-hans"]:
        return "zh"
    return lang


# -------------------------
# Word extraction patterns
# -------------------------
WORD_PATTERNS = {
    "ar": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+"),
    "bn": re.compile(r"[\u0980-\u09FF]+"),
    "ru": re.compile(r"[\u0400-\u04FF\u0500-\u052F]+"),
}


# -------------------------
# Filter punctuation / junk
# -------------------------
def is_clean_word(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("Z"):
            return False
    return True


# -------------------------
# CulturaX streaming
# -------------------------
def iter_culturax_texts_balanced(
    langs: List[str],
    split: str = "train",
    max_docs_per_lang: int = 200_000,
    min_chars: int = 200,
    seed_skip: int = 0,
    hf_token: Optional[str] = None,
) -> Iterator[Tuple[str, str]]:

    cfgs = [map_to_culturax_config(l) for l in langs]

    iters = []
    for cfg in cfgs:
        ds = load_dataset(
            "uonlp/CulturaX",
            cfg,
            split=split,
            streaming=True,
            token=hf_token,
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
# Mine word candidates
# -------------------------
def mine_word_candidates(
    stream: Iterator[Tuple[str, str]],
    target_langs: List[str],
    max_script_chars_per_lang: int = 30_000_000,
    min_word_len: int = 2,
    max_word_len: int = 24,
) -> Dict[str, Counter]:

    counters = {l: Counter() for l in target_langs}
    seen_chars = {l: 0 for l in target_langs}

    for lang, text in stream:

        if lang not in counters:
            continue

        if seen_chars[lang] >= max_script_chars_per_lang:
            continue

        words = WORD_PATTERNS[lang].findall(text)

        if not words:
            continue

        total_chars = sum(len(w) for w in words)

        if seen_chars[lang] + total_chars > max_script_chars_per_lang:

            budget = max_script_chars_per_lang - seen_chars[lang]

            kept = []
            so_far = 0

            for w in words:
                if so_far + len(w) > budget:
                    break
                kept.append(w)
                so_far += len(w)

            words = kept
            total_chars = so_far

        seen_chars[lang] += total_chars

        for w in words:

            if not (min_word_len <= len(w) <= max_word_len):
                continue

            if not is_clean_word(w):
                continue

            counters[lang][w] += 1

    return counters


# -------------------------
# Candidate ranking
# -------------------------
def choose_tokens_to_add(
    counters: Dict[str, Counter],
    base_tok,
    total_new_tokens: int,
) -> List[str]:

    weights = {"ar": 0.35, "bn": 0.35, "ru": 0.30}

    langs = list(counters.keys())
    denom = sum(weights.get(l, 1.0 / len(langs)) for l in langs)

    base_vocab = base_tok.get_vocab()

    picked = []
    seen = set()

    for l in langs:

        frac = weights.get(l, 1.0 / len(langs)) / denom
        budget = max(1, int(total_new_tokens * frac))

        scored = []

        for tok, freq in counters[l].most_common(budget * 10):

            if not tok or tok.isspace():
                continue

            if tok in base_vocab:
                continue

            if tok in seen:
                continue

            if len(tok) > 24:
                continue

            tok_cost = len(base_tok(tok, add_special_tokens=False)["input_ids"])

            if tok_cost <= 1:
                continue

            if l == "ru" and tok_cost <= 2 and len(tok) <= 3:
                continue

            score = freq * (tok_cost - 1)

            scored.append((score, freq, tok_cost, tok))

        scored.sort(reverse=True)

        added = 0

        for score, freq, tok_cost, tok in scored:

            if tok in seen:
                continue

            seen.add(tok)
            picked.append(tok)

            added += 1

            if added >= budget:
                break

            if len(picked) >= total_new_tokens:
                return picked

    return picked[:total_new_tokens]


# -------------------------
# Main
# -------------------------
def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--langs", nargs="+", default=["ar", "bn", "ru"])

    ap.add_argument("--max_docs_per_lang", type=int, default=200_000)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--seed_skip", type=int, default=0)

    ap.add_argument("--hf_token", type=str, default=None)

    ap.add_argument("--num_new_tokens", type=int, default=10_000)
    ap.add_argument("--max_script_chars_per_lang", type=int, default=30_000_000)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if base_tok.pad_token is None and base_tok.eos_token is not None:
        base_tok.pad_token = base_tok.eos_token

    print(f"[info] base vocab size: {len(base_tok)}")

    target_langs = [map_to_culturax_config(l) for l in args.langs]

    stream = iter_culturax_texts_balanced(
        langs=target_langs,
        max_docs_per_lang=args.max_docs_per_lang,
        min_chars=args.min_chars,
        seed_skip=args.seed_skip,
        hf_token=args.hf_token,
    )

    print("[info] mining words...")

    counters = mine_word_candidates(
        stream=stream,
        target_langs=target_langs,
        max_script_chars_per_lang=args.max_script_chars_per_lang,
    )

    to_add = choose_tokens_to_add(
        counters=counters,
        base_tok=base_tok,
        total_new_tokens=args.num_new_tokens,
    )

    print(f"[info] selected {len(to_add)} tokens")

    added = base_tok.add_tokens(
        [AddedToken(t, normalized=False) for t in to_add]
    )

    print(f"[info] actually added {added}")

    base_tok.save_pretrained(args.out_dir)

    print(f"[done] saved tokenizer to {args.out_dir}")
    print(f"[done] final vocab size {len(base_tok)}")


if __name__ == "__main__":
    main()