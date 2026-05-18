#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from typing import Dict, List

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def format_mcq_example(ex: Dict) -> str:
    return (
        f"Question: {ex['question']}\n"
        f"A. {ex['option_a']}\n"
        f"B. {ex['option_b']}\n"
        f"C. {ex['option_c']}\n"
        f"D. {ex['option_d']}\n"
        f"Answer:"
    )


def make_rows(bundle: Dict) -> List[Dict]:
    rows = []
    for lang, obj in bundle["langs"].items():
        gold_letter = obj["answer"].strip()
        gold_idx = {"A": 0, "B": 1, "C": 2, "D": 3}[gold_letter]

        rows.append({
            "fact_id": bundle["fact_id"],
            "lang": lang,
            "question": obj["question"],
            "option_a": obj["option_a"],
            "option_b": obj["option_b"],
            "option_c": obj["option_c"],
            "option_d": obj["option_d"],
            "answer": gold_letter,
            "gold_letter_idx": gold_idx,
        })
    return rows


def tokenize_rows(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(ex):
        prompt = format_mcq_example(ex)
        target = f" {ex['answer']}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        assert len(target_ids) == 1, f"Expected one token for answer, got {target_ids}"

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "fact_id": ex["fact_id"],
            "lang": ex["lang"],
            "gold_letter_idx": ex["gold_letter_idx"],
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            bundle = json.loads(line)
            rows.extend(make_rows(bundle))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_ratio))
    n_val = min(n_val, len(rows) - 1)

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    raw = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
    })

    train_tok = tokenize_rows(raw["train"], tokenizer, args.max_length)
    val_tok = tokenize_rows(raw["validation"], tokenizer, args.max_length)

    final_ds = DatasetDict({
        "train": train_tok,
        "validation": val_tok,
    })

    final_ds.save_to_disk(args.output_dir)
    print(final_ds)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()