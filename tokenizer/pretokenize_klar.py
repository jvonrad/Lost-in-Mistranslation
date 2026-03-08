#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pretokenize KLAR-CLC for causal LM factual retrieval training.

Creates train/val splits by FACT INDEX (so the same fact across languages
does not leak into both splits), formats prompts, tokenizes them, and stores:

- input_ids
- attention_mask
- labels   (prompt masked with -100; answer tokens supervised)
- language
- relation
- index
- input_text
- target_text

Example:
python pretokenize_klar.py \
  --klar_root /data/jonathan/KLAR-CLC \
  --model_name allenai/OLMo-2-1124-7B \
  --output_dir /data/jonathan/KLAR_tokenized/olmo2 \
  --template_mode random \
  --val_ratio 0.05 \
  --max_length 256 \
  --seed 42
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def parse_csv_arg(x: Optional[str]) -> Optional[List[str]]:
    if x is None or x.strip() == "":
        return None
    return [s.strip() for s in x.split(",") if s.strip()]


def discover_languages(klar_root: str) -> List[str]:
    return sorted([p.name for p in Path(klar_root).iterdir() if p.is_dir()])


def load_klar_examples(
    klar_root: str,
    languages: Optional[List[str]] = None,
    relations: Optional[List[str]] = None,
    template_mode: str = "random",   # first | random | all
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    klar_root = Path(klar_root)

    if languages is None:
        languages = discover_languages(str(klar_root))

    examples = []

    for lang in languages:
        lang_dir = klar_root / lang
        if not lang_dir.exists():
            print(f"[WARN] Missing language dir: {lang_dir}")
            continue

        for rel_file in sorted(lang_dir.glob("*.json")):
            rel = rel_file.stem
            if relations is not None and rel not in relations:
                continue

            with open(rel_file, "r", encoding="utf-8") as f:
                obj = json.load(f)

            templates = obj["prompt_templates"]
            samples = obj["samples"]

            for sample in samples:
                if template_mode == "first":
                    chosen_templates = [templates[0]]
                elif template_mode == "random":
                    chosen_templates = [rng.choice(templates)]
                elif template_mode == "all":
                    chosen_templates = templates
                else:
                    raise ValueError(f"Unknown template_mode: {template_mode}")

                for template in chosen_templates:
                    prompt = template.replace("<subject>", sample["subject"]).replace("<mask>", "").strip()
                    target = " " + sample["object"]

                    examples.append(
                        {
                            "language": lang,
                            "relation": rel,
                            "index": int(sample["index"]),
                            "subject": sample["subject"],
                            "object": sample["object"],
                            "template": template,
                            "input_text": prompt,
                            "target_text": target,
                        }
                    )

    return examples


def split_by_fact_index(
    examples: List[Dict],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by fact index, not by row.
    """
    all_indices = sorted({ex["index"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    n_val = max(1, int(len(all_indices) * val_ratio))
    val_indices = set(all_indices[:n_val])

    train_examples = [ex for ex in examples if ex["index"] not in val_indices]
    val_examples = [ex for ex in examples if ex["index"] in val_indices]
    return train_examples, val_examples


def tokenize_example(example, tokenizer, max_length: int):
    prompt = example["input_text"]
    answer = example["target_text"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "language": example["language"],
        "relation": example["relation"],
        "index": example["index"],
        "input_text": example["input_text"],
        "target_text": example["target_text"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--klar_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--languages", type=str, default="ar,en,es,fr,ja,ru,zh",
                        help="Comma-separated languages. Default: all.")
    parser.add_argument("--relations", type=str, default=None,
                        help="Comma-separated relations. Default: all.")
    parser.add_argument("--template_mode", type=str, default="random",
                        choices=["first", "random", "all"])
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    languages = parse_csv_arg(args.languages)
    relations = parse_csv_arg(args.relations)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading KLAR examples...")
    raw_examples = load_klar_examples(
        klar_root=args.klar_root,
        languages=languages,
        relations=relations,
        template_mode=args.template_mode,
        seed=args.seed,
    )

    train_examples, val_examples = split_by_fact_index(
        raw_examples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Total examples: {len(raw_examples):,}")
    print(f"Train examples: {len(train_examples):,}")
    print(f"Val examples:   {len(val_examples):,}")
    print(f"Languages:      {sorted({ex['language'] for ex in raw_examples})}")
    print(f"Relations:      {sorted({ex['relation'] for ex in raw_examples})}")

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    print("Tokenizing train...")
    train_ds = train_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )

    print("Tokenizing val...")
    val_ds = val_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length),
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    dsd = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })

    os.makedirs(args.output_dir, exist_ok=True)
    dsd.save_to_disk(args.output_dir)

    # Save metadata for reproducibility
    meta = {
        "klar_root": args.klar_root,
        "model_name": args.model_name,
        "languages": languages,
        "relations": relations,
        "template_mode": args.template_mode,
        "val_ratio": args.val_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
        "num_train": len(train_ds),
        "num_validation": len(val_ds),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # small sanity check
    ex = train_ds[0]
    print("\nSample decoded:")
    print(tokenizer.decode(ex["input_ids"], skip_special_tokens=True))
    answer_ids = [tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100]
    print("Supervised answer:", tokenizer.decode(answer_ids, skip_special_tokens=True))

    print(f"\nSaved tokenized dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()