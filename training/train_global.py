#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train allenai/OLMo-2-1124-7B on tokenized Global-MMLU.

Default mode: LoRA
Optional mode: full FT

Expected input:
  /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/global-mmlu-{num_langs}

Examples:

# Default LoRA
python train_global.py

# LoRA, 4 languages tokenized under global-mmlu-4
python train_global.py --num_langs 4

# Full parameter finetuning
python train_global.py --mode full

# Custom output dir
python train_global.py --output_dir /data/jonathan/Lost-in-Mistranslation/models/olmo2_7b_global_mmlu_lora
"""

import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForSeq2Seq,
	Trainer,
	TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType


DEFAULT_MODEL = "allenai/OLMo-2-1124-7B"
DEFAULT_DATA_ROOT = "/data/jonathan/Lost-in-Mistranslation/datasets/tokenized"
DEFAULT_OUT_ROOT = "/data/jonathan/Lost-in-Mistranslation/models"

import os

def is_main_process():
	return int(os.environ.get("RANK", "0")) == 0


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--num_langs", type=int, default=1)
	parser.add_argument(
		"--data_path",
		type=str,
		default="/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/global-mmlu-train-en-val-multi",
		help="Optional explicit tokenized dataset path",
	)
	parser.add_argument(
		"--mode",
		type=str,
		choices=["lora", "full"],
		default="lora",
		help="Training mode. Default: lora",
	)
	parser.add_argument("--output_dir", type=str, default=None)

	# optimization
	parser.add_argument("--epochs", type=float, default=5.0)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--weight_decay", type=float, default=0.0)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--per_device_train_batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=100)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
	parser.add_argument("--eval_steps", type=int, default=50)

	# LoRA
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)

	# misc
	parser.add_argument("--bf16", action="store_true", default=True)
	parser.add_argument("--no_bf16", action="store_true")
	parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
	parser.add_argument("--no_gradient_checkpointing", action="store_true")
	parser.add_argument("--save_final_merged", action="store_true")
	return parser.parse_args()


def infer_output_dir(args):
	if args.output_dir is not None:
		return args.output_dir

	mode_name = "lora" if args.mode == "lora" else "fullft"
	return os.path.join(
		DEFAULT_OUT_ROOT,
		f"olmo2_7b_global_mmlu_{args.num_langs}lang_{mode_name}",
	)


def count_trainable_params(model):
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	total = sum(p.numel() for p in model.parameters())
	pct = 100 * trainable / total
	return trainable, total, pct


def main():
	args = parse_args()

	if args.no_bf16:
		bf16 = False
	else:
		bf16 = True

	if args.no_gradient_checkpointing:
		gradient_checkpointing = False
	else:
		gradient_checkpointing = True

	data_path = args.data_path or os.path.join(
		DEFAULT_DATA_ROOT, f"global-mmlu-{args.num_langs}"
	)
	output_dir = infer_output_dir(args)
	os.makedirs(output_dir, exist_ok=True)

	if is_main_process():
		print(f"Loading tokenized dataset from: {data_path}")
	 
	ds = load_from_disk(data_path)
	train_ds = ds["train"]
	eval_ds = ds["validation"]
 
	if is_main_process():
		print(f"Train examples: {len(train_ds)}")
		print(f"Validation examples: {len(eval_ds)}")

	tokenizer = AutoTokenizer.from_pretrained(data_path, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	torch_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16

	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		torch_dtype=torch_dtype,
	)

	model.config.use_cache = False
	if gradient_checkpointing:
		model.gradient_checkpointing_enable()

	if args.mode == "lora":
		target_modules = [
			"q_proj", "k_proj", "v_proj", "o_proj",
			"gate_proj", "up_proj", "down_proj",
		]

		lora_config = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			inference_mode=False,
			r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			target_modules=target_modules,
		)
		model = get_peft_model(model, lora_config)
		if is_main_process():
			model.print_trainable_parameters()
	else:
		for p in model.parameters():
			p.requires_grad = True

	trainable, total, pct = count_trainable_params(model)
	if is_main_process():
		print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

	collator = DataCollatorForSeq2Seq(
		tokenizer=tokenizer,
		model=model,
		padding=True,
		pad_to_multiple_of=8,
		return_tensors="pt",
	)

	train_args = TrainingArguments(
		output_dir=output_dir,
		overwrite_output_dir=True,
		num_train_epochs=args.epochs,
		learning_rate=args.lr,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		eval_steps=args.eval_steps,
		eval_strategy="steps",
		save_strategy="steps",
		save_total_limit=2,
  		ddp_find_unused_parameters=False,
		bf16=bf16,
		fp16=not bf16,
		dataloader_num_workers=4,
		report_to="wandb",
		remove_unused_columns=False,
		seed=args.seed,
		gradient_checkpointing=gradient_checkpointing,
		lr_scheduler_type="cosine",
	)

	trainer = Trainer(
		model=model,
		args=train_args,
		train_dataset=train_ds,
		eval_dataset=eval_ds,
		data_collator=collator,
		tokenizer=tokenizer,
	)

	trainer.train()
 
	final_metrics = trainer.evaluate()
	if is_main_process():
		print("Final validation metrics:", final_metrics)

	final_dir = os.path.join(output_dir, "final")
	trainer.save_model(final_dir)
	tokenizer.save_pretrained(final_dir)
	if is_main_process():
		print(f"Saved final model to: {final_dir}")

	if args.mode == "lora" and args.save_final_merged:
		from peft import PeftModel

		if is_main_process():
			print("Merging LoRA adapters into base model...")
			merged = model.merge_and_unload()
			merged_dir = os.path.join(output_dir, "final_merged")
			merged.save_pretrained(merged_dir)
			tokenizer.save_pretrained(merged_dir)
			print(f"Saved merged model to: {merged_dir}")


if __name__ == "__main__":
	main()