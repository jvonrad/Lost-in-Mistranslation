#!/usr/bin/env python3
"""
Sample 100 random QA pairs per language from jvonrad/PolyFact (test split)
and emit one JSON file per language with empty llm_eval / human_eval fields,
to be filled in by an LLM judge and a human annotator.

Labels (in both fields):
    A = Correct
    B = Ambiguous / underspecified
    C = Incorrect

Usage:
    python sample_for_verification.py
    python sample_for_verification.py --n 100 --seed 42 --split test
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

DATASET = "jvonrad/PolyFact"
LANGS = ["en", "fr", "es", "ar", "zh", "de", "ru"]
DEFAULT_OUT = Path(__file__).parent / "samples"


def sample_language(lang: str, n: int, seed: int, split: str) -> list[dict]:
    ds = load_dataset(DATASET, lang, split=split)
    if n > len(ds):
        raise ValueError(f"Requested {n} samples but {lang}/{split} only has {len(ds)}")
    ds = ds.shuffle(seed=seed).select(range(n))

    out = []
    for row in ds:
        out.append(
            {
                "fact_id": row["fact_id"],
                "language": row["language"],
                "subject": row["subject"],
                "relation": row["relation"],
                "object": row["object"],
                "question": row["question"],
                "options": {
                    "A": row["option_a"],
                    "B": row["option_b"],
                    "C": row["option_c"],
                    "D": row["option_d"],
                },
                "answer_text": row["answer_text"],
                "answer_index": row["answer_index"],
                "llm_eval": {
                    "label": None,        # "A" | "B" | "C"
                    "rationale": None,    # short justification
                    "model": None,        # filled in when judging
                },
                "human_eval": {
                    "label": None,        # "A" | "B" | "C"
                    "notes": None,
                    "annotator": None,
                },
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        items = sample_language(lang, args.n, args.seed, args.split)
        out_path = args.out_dir / f"polyfact_eval_{lang}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": DATASET,
                    "language": lang,
                    "split": args.split,
                    "n": args.n,
                    "seed": args.seed,
                    "label_schema": {
                        "A": "Correct",
                        "B": "Ambiguous / underspecified",
                        "C": "Incorrect",
                    },
                    "items": items,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {out_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
