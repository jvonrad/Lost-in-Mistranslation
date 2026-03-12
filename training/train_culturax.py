#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python training/train_culturax.py \
  --base_model allenai/OLMo-2-1124-7B \
  --tokenizer_path /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo2_tok_ext_bn_ru \
  --output_dir /data/jonathan/Lost-in-Mistranslation/models/olmo2-culturax-bn-ru-cpt \
  --langs bn ru \
  --max_steps 3000
'''

import os
import argparse
from typing import Dict, List, Optional, Iterator

import torch
import wandb
from datasets import load_dataset, IterableDataset, concatenate_datasets
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	Trainer,
	TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


from transformers import TrainerCallback


class NewTokenEmbeddingMonitor(TrainerCallback):
    def __init__(self, old_vocab_size: int):
        self.old_vocab_size = old_vocab_size
        self.initial_input_emb = None
        self.initial_output_emb = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        with torch.no_grad():
            self.initial_input_emb = model.get_input_embeddings().weight.detach().clone()
            out = model.get_output_embeddings()
            if out is not None:
                self.initial_output_emb = out.weight.detach().clone()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or logs is None:
            return

        with torch.no_grad():
            inp = model.get_input_embeddings().weight.detach()

            old_delta = (inp[:self.old_vocab_size] - self.initial_input_emb[:self.old_vocab_size]).norm().item()
            new_delta = (inp[self.old_vocab_size:] - self.initial_input_emb[self.old_vocab_size:]).norm().item()

            logs["emb/input_old_rows_delta_norm"] = old_delta
            logs["emb/input_new_rows_delta_norm"] = new_delta

            out = model.get_output_embeddings()
            if out is not None and self.initial_output_emb is not None:
                out_w = out.weight.detach()
                old_out_delta = (out_w[:self.old_vocab_size] - self.initial_output_emb[:self.old_vocab_size]).norm().item()
                new_out_delta = (out_w[self.old_vocab_size:] - self.initial_output_emb[self.old_vocab_size:]).norm().item()

                logs["emb/output_old_rows_delta_norm"] = old_out_delta
                logs["emb/output_new_rows_delta_norm"] = new_out_delta
                
def get_rank() -> int:
	return int(os.environ.get("RANK", 0))


rank = get_rank()

# ---------- EVAL SET -----------

LANGS = ["bn", "ru"]
MAX_EVAL_SAMPLES_PER_LANG = 200

def format_global_mmlu_example(ex, tokenizer):
	question = ex["question"].strip()
	a = ex["option_a"].strip()
	b = ex["option_b"].strip()
	c = ex["option_c"].strip()
	d = ex["option_d"].strip()
	gold = ex["answer"].strip()

	prompt = (
		f"Question: {question}\n"
		f"A. {a}\n"
		f"B. {b}\n"
		f"C. {c}\n"
		f"D. {d}\n"
		f"Answer:"
	)

	target = f" {gold}"

	prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
	target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
 
	assert len(target_ids) == 1, f"Expected 1 answer token, got {target_ids} for gold={gold}"

	input_ids = prompt_ids + target_ids
	attention_mask = [1] * len(input_ids)
	labels = [-100] * len(prompt_ids) + target_ids

	return {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"labels": labels,
	}




def load_global_mmlu_dev_eval_by_lang(langs, tokenizer):
    eval_sets = {}

    for lang in langs:
        ds = load_dataset("CohereLabs/Global-MMLU", lang, split="dev")

        if MAX_EVAL_SAMPLES_PER_LANG is not None:
            ds = ds.select(range(min(MAX_EVAL_SAMPLES_PER_LANG, len(ds))))

        ds = ds.map(
            lambda ex: format_global_mmlu_example(ex, tokenizer),
            remove_columns=ds.column_names,
        )
        eval_sets[lang] = ds

    return eval_sets

import numpy as np


import numpy as np

import numpy as np


def make_mmlu_accuracy_fn(tokenizer):
    choice_ids = [
        tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
    ]

    def compute_mmlu_accuracy(eval_pred):
        logits, labels = eval_pred  # logits: [N, seq_len, 4]

        preds = []
        golds = []

        for i in range(labels.shape[0]):
            pos = np.where(labels[i] != -100)[0]
            if len(pos) == 0:
                continue

            j = int(pos[0])   # answer token position in labels
            if j == 0:
                continue

            gold_token = int(labels[i, j])

            if gold_token not in choice_ids:
                continue
            gold_idx = choice_ids.index(gold_token)

            # causal LM predicts token at j from logits at j-1
            pred_idx = int(np.argmax(logits[i, j - 1]))

            preds.append(pred_idx)
            golds.append(gold_idx)

        acc = float(np.mean(np.array(preds) == np.array(golds))) if golds else 0.0
        return {"accuracy": acc}

    return compute_mmlu_accuracy

def print0(*args, **kwargs):
	if rank == 0:
		print(*args, **kwargs, flush=True)


def map_to_culturax_config(lang: str) -> str:
	lang = lang.lower()
	if lang in ["zh-cn", "zh-hans", "zh_simplified"]:
		return "zh"
	return lang


def iter_culturax_texts_balanced(
	langs: List[str],
	split: str,
	max_docs_per_lang: int,
	min_chars: int,
	seed_skip: int,
	hf_token: Optional[str],
) -> Iterator[Dict[str, str]]:
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
			yield {"lang": cfg, "text": txt}


def build_iterable_dataset(
	langs: List[str],
	split: str,
	max_docs_per_lang: int,
	min_chars: int,
	seed_skip: int,
	hf_token: Optional[str],
) -> IterableDataset:
	return IterableDataset.from_generator(
		lambda: iter_culturax_texts_balanced(
			langs=langs,
			split=split,
			max_docs_per_lang=max_docs_per_lang,
			min_chars=min_chars,
			seed_skip=seed_skip,
			hf_token=hf_token,
		)
	)


def tokenize_and_chunk_dataset(ds, tokenizer, seq_len: int):
	buffer = {
		"input_ids": [],
		"attention_mask": [],
	}

	def generator():
		for ex in ds:
			text = ex["text"]

			toks = tokenizer(
				text,
				add_special_tokens=False,
				return_attention_mask=True,
				truncation=False,
			)

			ids = toks["input_ids"] + [tokenizer.eos_token_id]
			mask = toks["attention_mask"] + [1]

			buffer["input_ids"].extend(ids)
			buffer["attention_mask"].extend(mask)

			while len(buffer["input_ids"]) >= seq_len:
				chunk_ids = buffer["input_ids"][:seq_len]
				chunk_mask = buffer["attention_mask"][:seq_len]

				buffer["input_ids"] = buffer["input_ids"][seq_len:]
				buffer["attention_mask"] = buffer["attention_mask"][seq_len:]

				yield {
					"input_ids": chunk_ids,
					"attention_mask": chunk_mask,
					"labels": chunk_ids.copy(),
				}

	return IterableDataset.from_generator(generator)


CHOICE_TOKEN_IDS = None  # set after tokenizer is loaded

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # logits: [B, seq_len, vocab_size]
    # Return only the 4 choice columns to save memory
    return logits[:, :, CHOICE_TOKEN_IDS]  # [B, seq_len, 4]

class CausalLMPaddingCollator:
	def __init__(self, tokenizer, label_pad_token_id: int = -100, pad_to_multiple_of: int = 8):
		self.tokenizer = tokenizer
		self.label_pad_token_id = label_pad_token_id
		self.pad_to_multiple_of = pad_to_multiple_of

	def __call__(self, features):
		input_ids = [f["input_ids"] for f in features]
		attention_mask = [f["attention_mask"] for f in features]
		labels = [f["labels"] for f in features]

		max_len = max(len(x) for x in input_ids)
		if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
			max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

		padded_input_ids = []
		padded_attention_mask = []
		padded_labels = []

		for ids, mask, labs in zip(input_ids, attention_mask, labels):
			pad_len = max_len - len(ids)
			padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
			padded_attention_mask.append(mask + [0] * pad_len)
			padded_labels.append(labs + [self.label_pad_token_id] * pad_len)

		return {
			"input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
			"attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
			"labels": torch.tensor(padded_labels, dtype=torch.long),
		}


def enable_only_new_token_rows(model, old_vocab_size: int, new_vocab_size: int):
	"""
	Keep full embedding matrix trainable as a parameter, but zero out gradients
	for all old rows so only newly added token rows are updated.
	"""
	if new_vocab_size <= old_vocab_size:
		raise ValueError(f"new_vocab_size={new_vocab_size} must be > old_vocab_size={old_vocab_size}")

	input_emb = model.get_input_embeddings()
	input_emb.weight.requires_grad_(True)

	def grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
		grad = grad.clone()
		grad[:old_vocab_size] = 0
		return grad

	input_emb.weight.register_hook(grad_mask_hook)

	# If LM head is untied, do the same there.
	output_emb = model.get_output_embeddings()
	if output_emb is not None and output_emb.weight is not input_emb.weight:
		output_emb.weight.requires_grad_(True)

		def lm_head_grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
			grad = grad.clone()
			grad[:old_vocab_size] = 0
			return grad

		output_emb.weight.register_hook(lm_head_grad_mask_hook)

	return input_emb, output_emb


def parse_args():
	ap = argparse.ArgumentParser()

	ap.add_argument("--base_model", type=str, default="allenai/OLMo-2-1124-7B")
	ap.add_argument("--tokenizer_path", type=str, required=True)
	ap.add_argument("--output_dir", type=str, required=True)

	ap.add_argument("--langs", nargs="+", default=["bn", "ru"])
	ap.add_argument("--split", type=str, default="train")
	ap.add_argument("--hf_token", type=str, default=None)

	ap.add_argument("--max_docs_per_lang", type=int, default=200_000)
	ap.add_argument("--min_chars", type=int, default=200)
	ap.add_argument("--seed_skip", type=int, default=0)

	ap.add_argument("--seq_len", type=int, default=1024)

	ap.add_argument("--max_steps", type=int, default=3000)
	ap.add_argument("--learning_rate", type=float, default=2e-5)
	ap.add_argument("--weight_decay", type=float, default=0.01)
	ap.add_argument("--warmup_ratio", type=float, default=0.03)

	ap.add_argument("--per_device_train_batch_size", type=int, default=2)
	ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
	ap.add_argument("--logging_steps", type=int, default=10)
	ap.add_argument("--eval_steps", type=int, default=100)
	ap.add_argument("--save_total_limit", type=int, default=3)
	ap.add_argument("--dataloader_num_workers", type=int, default=2)

	ap.add_argument("--lora_r", type=int, default=16)
	ap.add_argument("--lora_alpha", type=int, default=32)
	ap.add_argument("--lora_dropout", type=float, default=0.05)

	ap.add_argument("--bf16", action="store_true", default=True)
	ap.add_argument("--no_bf16", action="store_true")
	ap.add_argument("--report_to", type=str, default="wandb")

	return ap.parse_args()


def main():
	args = parse_args()
	if args.no_bf16:
		args.bf16 = False

	os.makedirs(args.output_dir, exist_ok=True)

	# Base tokenizer size before extension
	base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
	old_vocab_size = len(base_tokenizer)

	print0(f"[info] loading extended tokenizer from {args.tokenizer_path}")
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
  
	# Set global for preprocess_logits_for_metrics

	global CHOICE_TOKEN_IDS
	CHOICE_TOKEN_IDS = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]
  
	compute_metrics = make_mmlu_accuracy_fn(tokenizer)

	new_vocab_size = len(tokenizer)
	print0(f"[info] base vocab size:     {old_vocab_size}")
	print0(f"[info] extended vocab size: {new_vocab_size}")
	print0(f"[info] new token rows:      {new_vocab_size - old_vocab_size}")
	
	print0("[info] loading Global-MMLU dev eval set")
 
	eval_ds = load_global_mmlu_dev_eval_by_lang(LANGS, tokenizer)
	print0(f"[info] eval languages: {list(eval_ds.keys())}")
	for lang, ds in eval_ds.items():
		print0(f"[info] {lang} eval examples: {len(ds)}")
		print0(f"[info] {lang} eval sample length: {len(ds[0]['input_ids'])}")

	print0("[info] loading base model")
	model = AutoModelForCausalLM.from_pretrained(
		args.base_model,
		torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
	)

	print0("[info] resizing embeddings to extended tokenizer")
	model.resize_token_embeddings(new_vocab_size)

	model.config.pad_token_id = tokenizer.pad_token_id
	model.config.eos_token_id = tokenizer.eos_token_id
	if tokenizer.bos_token_id is not None:
		model.config.bos_token_id = tokenizer.bos_token_id

	# LoRA on attention layers
	peft_config = LoraConfig(
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
		lora_dropout=args.lora_dropout,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
	)
	model = get_peft_model(model, peft_config)

	# Train only newly added token rows in embeddings
	if new_vocab_size > old_vocab_size:
		print0("[info] training only newly added embedding rows")
		enable_only_new_token_rows(
			model=model,
			old_vocab_size=old_vocab_size,
			new_vocab_size=new_vocab_size,
		)
	else:
		print0("[info] no new tokens, embeddings frozen")

	if rank == 0:
		model.print_trainable_parameters()

	print0("[info] building streaming CulturaX dataset")
	raw_ds = build_iterable_dataset(
		langs=args.langs,
		split=args.split,
		max_docs_per_lang=args.max_docs_per_lang,
		min_chars=args.min_chars,
		seed_skip=args.seed_skip,
		hf_token=args.hf_token,
	)

	train_ds = tokenize_and_chunk_dataset(
		ds=raw_ds,
		tokenizer=tokenizer,
		seq_len=args.seq_len,
	)

	collator = CausalLMPaddingCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

	if args.report_to == "wandb" and rank == 0:
		wandb.init(
			project=os.getenv("WANDB_PROJECT", "UnLock"),
			config={
				"base_model": args.base_model,
				"tokenizer_path": args.tokenizer_path,
				"langs": args.langs,
				"max_docs_per_lang": args.max_docs_per_lang,
				"seq_len": args.seq_len,
				"max_steps": args.max_steps,
				"learning_rate": args.learning_rate,
				"weight_decay": args.weight_decay,
				"batch_size": args.per_device_train_batch_size,
				"grad_accum": args.gradient_accumulation_steps,
				"lora_r": args.lora_r,
				"lora_alpha": args.lora_alpha,
				"lora_dropout": args.lora_dropout,
				"old_vocab_size": old_vocab_size,
				"new_vocab_size": new_vocab_size,
				"num_new_tokens": new_vocab_size - old_vocab_size,
			},
		)

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		max_steps=args.max_steps,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		warmup_ratio=args.warmup_ratio,
		lr_scheduler_type="cosine",
		logging_steps=args.logging_steps,
		save_strategy="steps",
		save_steps=args.eval_steps,
		eval_strategy="steps",
		eval_steps=args.eval_steps,
  		eval_accumulation_steps=1,
		per_device_eval_batch_size=1,
		save_total_limit=args.save_total_limit,
		bf16=args.bf16,
		optim="adamw_torch_fused",
		report_to=(args.report_to if rank == 0 else "none"),
		remove_unused_columns=False,
		dataloader_num_workers=args.dataloader_num_workers,
		dataloader_pin_memory=True,
		ddp_find_unused_parameters=False,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		eval_dataset=eval_ds,
		data_collator=collator,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
  		preprocess_logits_for_metrics=preprocess_logits_for_metrics,
		callbacks=[NewTokenEmbeddingMonitor(old_vocab_size=old_vocab_size)],
	)

	print0("[info] starting training")
	trainer.train()

	if rank == 0:
		lora_out = os.path.join(args.output_dir, "final_lora")
		model.save_pretrained(lora_out)
		tokenizer.save_pretrained(lora_out)

		print0("[info] saving merged model")
		merged_model = model.merge_and_unload()
		merged_out = os.path.join(args.output_dir, "merged")
		
		os.makedirs(merged_out, exist_ok=True)
		merged_model.save_pretrained(merged_out)
		tokenizer.save_pretrained(merged_out)

		print0(f"[done] saved LoRA to {lora_out}")
		print0(f"[done] saved merged model to {merged_out}")

		if args.report_to == "wandb":
			wandb.finish()


if __name__ == "__main__":
	main()