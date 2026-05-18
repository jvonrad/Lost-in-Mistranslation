#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python train_tokenizer.py \
  --input_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/olmo_tokenizer_mix.jsonl \
  --output_dir /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k \
  --vocab_size 151552 \
  --min_frequency 2
'''

import os
import json
import argparse
from typing import Iterator

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to tokenizer training JSONL with {'text': ...}",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save tokenizer files",
    )
    ap.add_argument(
        "--vocab_size",
        type=int,
        default=131072,
        help="Tokenizer vocab size",
    )
    ap.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum pair frequency",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to use",
    )
    return ap.parse_args()


def iter_text(jsonl_path: str, limit: int | None = None) -> Iterator[str]:
    n = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            yield text
            n += 1
            if limit is not None and n >= limit:
                break


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Byte-level BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Good default special tokens for causal LM adaptation
    special_tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training tokenizer on {args.input_jsonl}")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Min frequency: {args.min_frequency}")

    tokenizer.train_from_iterator(
        iter_text(args.input_jsonl, limit=args.limit),
        trainer=trainer,
    )

    # Optional post-processor: add BOS/EOS if you want
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", bos_id),
            ("</s>", eos_id),
        ],
    )

    tokenizer_json = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json)

    # Also save vocab/merges in a reusable format
    tokenizer.model.save(args.output_dir)

    # Save a minimal tokenizer config for later HF loading
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 32768,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }

    with open(os.path.join(args.output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    print(f"Saved tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()