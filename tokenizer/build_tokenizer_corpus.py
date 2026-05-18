#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a tokenizer-training corpus from:
  1) UW/olmo-mix-1124-subset-p99   (40%)
  2) CulturaX                     (60%, balanced across 11 non-English target languages)

Design choices:
- Preserve OLMo-style English/general/code/math/science behavior via OLMo subset
- Add balanced multilingual coverage via CulturaX
- Do NOT add extra English from CulturaX
- Slightly oversample Swahili and Bengali
- Sample by BYTE BUDGET, not doc count
- No aggressive filtering on OLMo subset
- Only very light filtering on CulturaX

Output rows:
    {"text": "...", "source": "olmo_mix" or "culturax", "lang": "en"/..., "bytes": N}
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from typing import Dict, Optional

from datasets import load_dataset


REQ_LANGS = ["es", "fr", "de", "id", "pt", "ru", "zh", "ja", "ar", "sw", "bn"]

CULTURAX_WEIGHTS = {
    "es": 1.00,
    "fr": 1.00,
    "de": 1.00,
    "id": 1.00,
    "pt": 1.00,
    "ru": 1.00,
    "zh": 1.00,
    "ja": 1.00,
    "ar": 1.00,
    "sw": 1.35,
    "bn": 1.35,
}

TEXT_FIELD_CANDIDATES = [
    "text",
    "content",
    "raw_content",
    "doc",
    "document",
    "body",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--total_gb", type=float, default=20.0)
    p.add_argument("--olmo_fraction", type=float, default=0.4,
                   help="Fraction of bytes from UW/olmo-mix-1124-subset-p99.")
    p.add_argument("--olmo_min_chars", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer_size", type=int, default=5000)

    # Only used for CulturaX light filtering
    p.add_argument("--culturax_min_chars", type=int, default=20)
    p.add_argument("--culturax_max_repeat_char_frac", type=float, default=0.60)

    p.add_argument("--culturax_dataset", type=str, default="uonlp/CulturaX")
    p.add_argument("--olmo_dataset", type=str, default="UW/olmo-mix-1124-subset-p99")
    p.add_argument("--culturax_split", type=str, default="train")
    p.add_argument("--olmo_split", type=str, default="train")
    return p.parse_args()


def gb_to_bytes(gb: float) -> int:
    return int(gb * (1024 ** 3))


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dominant_char_fraction(text: str) -> float:
    if not text:
        return 1.0
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return max(counts.values()) / len(text)


def get_text(example: dict) -> Optional[str]:
    for key in TEXT_FIELD_CANDIDATES:
        if key in example and isinstance(example[key], str):
            return example[key]
    for _, v in example.items():
        if isinstance(v, str) and len(v) > 0:
            return v
    return None


def keep_olmo_text(text: str, min_chars: int = 30) -> bool:
    return bool(text) and len(text) >= min_chars

def keep_culturax_text(text: str, min_chars: int, max_repeat_char_frac: float) -> bool:
    if not text:
        return False
    if len(text) < min_chars:
        return False
    if dominant_char_fraction(text) > max_repeat_char_frac:
        return False
    return True

def try_get_lang(example: dict) -> Optional[str]:
    for key in ["lang", "language", "iso_639_1", "iso_639_3"]:
        if key in example and isinstance(example[key], str):
            return example[key]
    return None


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def weighted_lang_budgets(total_bytes_for_culturax: int) -> Dict[str, int]:
    total_weight = sum(CULTURAX_WEIGHTS[l] for l in REQ_LANGS)
    budgets = {
        lang: int(total_bytes_for_culturax * CULTURAX_WEIGHTS[lang] / total_weight)
        for lang in REQ_LANGS
    }
    drift = total_bytes_for_culturax - sum(budgets.values())
    budgets["es"] += drift
    return budgets


def write_jsonl_row(fout, text: str, source: str, lang: str) -> int:
    n_bytes = utf8_bytes(text)
    row = {
        "text": text,
        "source": source,
        "lang": lang,
        "bytes": n_bytes,
    }
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return n_bytes


def load_streaming_dataset(name: str, split: str, config_name: Optional[str] = None):
    if config_name is None:
        return load_dataset(name, split=split, streaming=True)
    return load_dataset(name, config_name, split=split, streaming=True)






def sample_olmo_mix(
    fout,
    target_bytes: int,
    dataset_name: str,
    split: str,
    seed: int,
    buffer_size: int,
    min_chars: int
) -> int:
    print(f"[OLMO] Loading {dataset_name} ({split}) in streaming mode...", file=sys.stderr)
    ds = load_streaming_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    written = 0
    seen = 0
    kept = 0

    for ex in ds:
        if written >= target_bytes:
            break

        seen += 1
        text = get_text(ex)
        if text is None:
            continue

        text = normalize_text(text)
        if not keep_olmo_text(text, min_chars):
            continue

        written += write_jsonl_row(fout, text=text, source="olmo_mix", lang="en")
        kept += 1

        if kept % 10000 == 0:
            print(
                f"[OLMO] kept={kept:,} seen={seen:,} written={written / 1e9:.2f} GB",
                file=sys.stderr,
            )

    print(f"[OLMO] DONE kept={kept:,} seen={seen:,} written={written / 1e9:.2f} GB", file=sys.stderr)
    return written


def load_culturax_lang_stream(dataset_name: str, lang: str, split: str, seed: int, buffer_size: int):
    tried = []

    for config_name in [lang, f"{lang}", f"{lang}_Latin"]:
        try:
            ds = load_streaming_dataset(dataset_name, split=split, config_name=config_name)
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
            return ds
        except Exception as e:
            tried.append((config_name, str(e)))

    try:
        ds = load_streaming_dataset(dataset_name, split=split)
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

        def gen():
            for ex in ds:
                ex_lang = try_get_lang(ex)
                if ex_lang == lang:
                    yield ex

        return gen()
    except Exception as e:
        tried.append(("__no_config__", str(e)))

    msg = "\n".join([f"  - config={cfg}: {err}" for cfg, err in tried[:5]])
    raise RuntimeError(f"Could not load CulturaX for lang={lang}. Tried:\n{msg}")


def sample_culturax_lang(
    fout,
    dataset_name: str,
    lang: str,
    split: str,
    target_bytes: int,
    seed: int,
    buffer_size: int,
    min_chars: int,
    max_repeat_char_frac: float,
) -> int:
    print(f"[CULTURAX:{lang}] Loading stream...", file=sys.stderr)
    ds = load_culturax_lang_stream(dataset_name, lang, split, seed, buffer_size)

    written = 0
    seen = 0
    kept = 0

    for ex in ds:
        if written >= target_bytes:
            break

        seen += 1
        text = get_text(ex)
        if text is None:
            continue

        text = normalize_text(text)
        if not keep_culturax_text(text, min_chars=min_chars, max_repeat_char_frac=max_repeat_char_frac):
            continue

        written += write_jsonl_row(fout, text=text, source="culturax", lang=lang)
        kept += 1

        if kept % 5000 == 0:
            print(
                f"[CULTURAX:{lang}] kept={kept:,} seen={seen:,} written={written / 1e9:.2f} GB",
                file=sys.stderr,
            )

    print(f"[CULTURAX:{lang}] DONE kept={kept:,} seen={seen:,} written={written / 1e9:.2f} GB", file=sys.stderr)
    return written


def main():
    args = parse_args()
    ensure_parent_dir(args.output_jsonl)

    total_bytes = gb_to_bytes(args.total_gb)
    olmo_target = int(total_bytes * args.olmo_fraction)
    culturax_target = total_bytes - olmo_target
    lang_budgets = weighted_lang_budgets(culturax_target)

    print("=" * 80, file=sys.stderr)
    print("Building tokenizer corpus", file=sys.stderr)
    print(f"Total target:      {total_bytes / 1e9:.2f} GB", file=sys.stderr)
    print(f"OLMo target:       {olmo_target / 1e9:.2f} GB", file=sys.stderr)
    print(f"CulturaX target:   {culturax_target / 1e9:.2f} GB", file=sys.stderr)
    print("CulturaX lang budgets:", file=sys.stderr)
    for lang in REQ_LANGS:
        print(f"  {lang}: {lang_budgets[lang] / 1e9:.2f} GB", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    total_written = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        total_written += sample_olmo_mix(
            fout=fout,
            target_bytes=olmo_target,
            dataset_name=args.olmo_dataset,
            split=args.olmo_split,
            seed=args.seed,
            buffer_size=args.buffer_size,
            min_chars=args.olmo_min_chars
        )

        for i, lang in enumerate(REQ_LANGS):
            lang_seed = args.seed + 1000 + i
            total_written += sample_culturax_lang(
                fout=fout,
                dataset_name=args.culturax_dataset,
                lang=lang,
                split=args.culturax_split,
                target_bytes=lang_budgets[lang],
                seed=lang_seed,
                buffer_size=args.buffer_size,
                min_chars=args.culturax_min_chars,
                max_repeat_char_frac=args.culturax_max_repeat_char_frac,
            )

    print("=" * 80, file=sys.stderr)
    print(f"FINAL written: {total_written / 1e9:.2f} GB", file=sys.stderr)
    print(f"Saved to: {args.output_jsonl}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)


if __name__ == "__main__":
    main()