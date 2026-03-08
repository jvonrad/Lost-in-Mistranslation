#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a causal LM on a pretokenized KLAR dataset saved with DatasetDict.save_to_disk()
using LoRA.

Example:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_klar_lora.py \
  --model_name allenai/OLMo-2-1124-7B \
  --tokenized_data_dir /data/jonathan/KLAR_tokenized/olmo2 \
  --output_dir /data/jonathan/klar_runs/olmo2_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --save_final \
  --save_merged
"""

import os
import argparse
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel

##############
# Helpers
##############

def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def rank0_print(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
        
def get_lora_target_modules_from_start_layer(model, base_targets, start_layer: int):
    """
    Return exact module names for LoRA injection, restricted to layers >= start_layer.

    Example matched names:
      model.layers.20.self_attn.q_proj
      model.layers.20.mlp.up_proj
      ...
    """
    matched = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        parts = name.split(".")
        layer_idx = None

        # find "...layers.<idx>..."
        for i in range(len(parts) - 1):
            if parts[i] == "layers":
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    pass

        if layer_idx is None or layer_idx < start_layer:
            continue

        if any(name.endswith(t) for t in base_targets):
            matched.append(name)

    return sorted(matched)


class DataCollatorForPretokenizedCausalLM:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            n = len(f["input_ids"])
            pad_len = max_len - n

            batch_input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_exact_match_accuracy(eval_preds, tokenizer):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    total = 0
    correct = 0

    for pred_row, label_row in zip(preds, labels):
        pred_ids = [int(p) for p, l in zip(pred_row, label_row) if l != -100]
        gold_ids = [int(l) for l in label_row if l != -100]

        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip().lower()
        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True).strip().lower()

        total += 1
        correct += int(pred_text == gold_text)

    return {"answer_em": correct / max(total, 1)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenized_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
    "--lora_start_layer",
    type=int,
    default=0,
    help="Only apply LoRA to transformer layers with index >= this value."
)
    parser.add_argument(
        "--print_target_modules",
        action="store_true",
        help="Print the final exact LoRA target modules and exit."
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_final", action="store_true")
    parser.add_argument("--save_merged", action="store_true")

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Module names to target with LoRA."
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if get_rank() != 0:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    rank0_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank0_print("Loading tokenized dataset...")
    dsd = load_from_disk(args.tokenized_data_dir)
    train_ds = dsd["train"]
    val_ds = dsd["validation"]

    rank0_print(f"Train size: {len(train_ds):,}")
    rank0_print(f"Val size:   {len(val_ds):,}")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    rank0_print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # Important for some models when using gradient checkpointing + PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    rank0_print("Wrapping model with LoRA...")
    
    exact_target_modules = get_lora_target_modules_from_start_layer(
        model=model,
        base_targets=args.lora_target_modules,
        start_layer=args.lora_start_layer,
    )

    rank0_print(f"LoRA start layer: {args.lora_start_layer}")
    rank0_print(f"Found {len(exact_target_modules)} target modules for LoRA")

    if len(exact_target_modules) == 0:
        raise ValueError(
            f"No LoRA target modules found for start_layer={args.lora_start_layer} "
            f"and base targets={args.lora_target_modules}"
        )

    if get_rank() == 0:
        for m in exact_target_modules[:50]:
            print("  ", m)
        if len(exact_target_modules) > 50:
            print(f"  ... and {len(exact_target_modules) - 50} more")

    if args.print_target_modules:
        return

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=exact_target_modules,
    )

    model = get_peft_model(model, peft_config)

    if args.torch_compile:
        model = torch.compile(model)

    total_params, trainable_params = count_parameters(model)
    rank0_print(f"Total params:     {total_params:,}")
    rank0_print(f"Trainable params: {trainable_params:,}")
    rank0_print(f"Trainable %:      {100.0 * trainable_params / total_params:.4f}%")

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    collator = DataCollatorForPretokenizedCausalLM(tokenizer)
    report_to = [] if args.report_to == "none" else [args.report_to]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=report_to,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        # compute_metrics=lambda eval_preds: compute_exact_match_accuracy(eval_preds, tokenizer),
    )

    if get_rank() == 0 and len(train_ds) > 0:
        ex = train_ds[0]
        print("\nSample decoded:")
        print(tokenizer.decode(ex["input_ids"], skip_special_tokens=True))
        answer_ids = [tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100]
        print("Supervised answer:", tokenizer.decode(answer_ids, skip_special_tokens=True))

    trainer.train()

    rank0_print("\nFinal eval...")
    metrics = trainer.evaluate()
    rank0_print(metrics)

    if args.save_final:
        final_dir = os.path.join(args.output_dir, "final_adapter")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        with open(os.path.join(final_dir, "BASE_MODEL.txt"), "w") as f:
            f.write(args.model_name + "\n")

        rank0_print(f"Saved final LoRA adapter to: {final_dir}")
        rank0_print(f"Use tokenizer from base model: {args.model_name}")

    if args.save_merged:
        if get_rank() == 0:
            rank0_print("Merging LoRA adapter into base model...")
            merged_dir = os.path.join(args.output_dir, "final_merged")

            # unwrap compile if needed is tricky; assume no torch_compile for merge path
            if isinstance(trainer.model, PeftModel):
                merged_model = trainer.model.merge_and_unload()
            else:
                merged_model = trainer.model

            merged_model.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)

            with open(os.path.join(merged_dir, "MERGED_FROM_BASE_MODEL.txt"), "w") as f:
                f.write(args.model_name + "\n")

            rank0_print(f"Saved merged model to: {merged_dir}")


if __name__ == "__main__":
    main()