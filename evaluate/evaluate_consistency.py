#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate allenai/OLMo-2-1124-7B on the VALIDATION split of your multilingual MCQ dataset
using conditional logprob scoring over answer options.

Example:
python evaluate_consistency.py \
  --input_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text_filtered_val.jsonl \
  --model allenai/OLMo-2-1124-7B \
  --batch_size 8 \
  --score_mode avg

score_mode:
  - sum : total logprob of option tokens
  - avg : length-normalized average logprob per token (usually better)
"""

import os
import json
import math
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]


def load_rows(path):
	rows = []
	with open(path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f, 1):
			line = line.strip()
			if not line:
				continue
			try:
				rows.append(json.loads(line))
			except Exception as e:
				print(f"[WARN] skipping bad json line {i}: {e}")
	return rows


def build_prompt(question: str) -> str:
	# Keep this simple and stable.
	return f"Question: {question}\nAnswer:"


def score_candidates_batch(
	model,
	tokenizer,
	examples,
	device,
	score_mode="avg",
):
	"""
	examples: list of dicts with keys:
	  - prompt
	  - options: list[str]
	returns:
	  list[list[float]] : scores per option
	"""
	flat_texts = []
	meta = []  # (example_idx, option_idx, prompt_len_chars)

	for ex_idx, ex in enumerate(examples):
		prompt = ex["prompt"]
		for opt_idx, opt in enumerate(ex["options"]):
			# leading space matters for tokenization consistency
			full_text = prompt + " " + opt
			flat_texts.append(full_text)
			meta.append((ex_idx, opt_idx, prompt, opt))

	enc = tokenizer(
		flat_texts,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=2048,
	)
	input_ids = enc["input_ids"].to(device)
	attention_mask = enc["attention_mask"].to(device)

	with torch.no_grad():
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = outputs.logits  # [B, T, V]
		logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)  # next-token probs
		target_ids = input_ids[:, 1:]  # tokens being predicted
		target_mask = attention_mask[:, 1:]

		token_logprobs = torch.gather(
			logprobs, 2, target_ids.unsqueeze(-1)
		).squeeze(-1)  # [B, T-1]

	# Need prompt lengths in token space, not char space
	prompt_token_lists = tokenizer(
		[m[2] for m in meta],
		padding=False,
		truncation=True,
		max_length=2048,
	)["input_ids"]

	prompt_lens = [len(x) for x in prompt_token_lists]

	# For sequence: full = prompt + option
	# The predicted tokens for the option start at index prompt_len-1 in token_logprobs
	# because token_logprobs[t] predicts input_ids[t+1]
	scores = [[None] * len(ex["options"]) for ex in examples]

	for row_idx, (ex_idx, opt_idx, _prompt, _opt) in enumerate(meta):
		plen = prompt_lens[row_idx]
		seq_len = int(attention_mask[row_idx].sum().item())

		start = max(plen - 1, 0)
		end = seq_len - 1  # token_logprobs length is seq_len-1

		opt_lp = token_logprobs[row_idx, start:end]
		opt_mask = target_mask[row_idx, start:end]

		opt_lp = opt_lp[opt_mask.bool()]

		if opt_lp.numel() == 0:
			score = -1e9
		else:
			if score_mode == "sum":
				score = float(opt_lp.sum().item())
			elif score_mode == "avg":
				score = float(opt_lp.mean().item())
			else:
				raise ValueError(f"Unknown score_mode: {score_mode}")

		scores[ex_idx][opt_idx] = score

	return scores


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--input_jsonl", required=True, help="Use your validation split here, not train.")
	ap.add_argument("--model", default="allenai/OLMo-2-1124-7B")
	ap.add_argument("--batch_size", type=int, default=8)
	ap.add_argument("--max_examples_per_lang", type=int, default=0, help="0 = all")
	ap.add_argument("--score_mode", choices=["sum", "avg"], default="avg")
	args = ap.parse_args()

	print(f"Loading validation rows from: {args.input_jsonl}")
	rows = load_rows(args.input_jsonl)
	print(f"Loaded {len(rows):,} rows")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Device: {device}")

	print(f"Loading tokenizer: {args.model}")
	tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	print(f"Loading model: {args.model}")
	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		dtype=torch.bfloat16 if device == "cuda" else torch.float32,
		trust_remote_code=True,
		device_map="auto" if device == "cuda" else None,
	)
	if device != "cuda":
		model = model.to(device)
	model.eval()

	per_lang_correct = defaultdict(int)
	per_lang_total = defaultdict(int)

	overall_correct = 0
	overall_total = 0

	for lang in LANGS:
		lang_examples = []

		for row in rows:
			if "langs" not in row or lang not in row["langs"]:
				continue
			item = row["langs"][lang]

			question = item.get("question", "").strip()
			options = item.get("options", [])
			gold = item.get("answer_text", "").strip()

			if not question or not isinstance(options, list) or len(options) != 4 or gold not in options:
				continue

			lang_examples.append({
				"fact_id": row.get("fact_id"),
				"lang": lang,
				"prompt": build_prompt(question),
				"options": options,
				"gold": gold,
			})

		if args.max_examples_per_lang > 0:
			lang_examples = lang_examples[:args.max_examples_per_lang]

		print(f"\n[{lang}] evaluating {len(lang_examples):,} examples")

		for i in range(0, len(lang_examples), args.batch_size):
			batch = lang_examples[i:i + args.batch_size]
			score_lists = score_candidates_batch(
				model=model,
				tokenizer=tokenizer,
				examples=batch,
				device=device,
				score_mode=args.score_mode,
			)

			for ex, scores in zip(batch, score_lists):
				pred_idx = max(range(len(scores)), key=lambda k: scores[k])
				pred = ex["options"][pred_idx]
				correct = int(pred == ex["gold"])

				per_lang_correct[lang] += correct
				per_lang_total[lang] += 1
				overall_correct += correct
				overall_total += 1

		acc = per_lang_correct[lang] / max(per_lang_total[lang], 1)
		print(f"[{lang}] acc = {acc:.4f} ({per_lang_correct[lang]}/{per_lang_total[lang]})")

	print("\n" + "=" * 80)
	print("FINAL RESULTS")
	print("=" * 80)
	for lang in LANGS:
		total = per_lang_total[lang]
		acc = per_lang_correct[lang] / max(total, 1)
		print(f"{lang:>2}  acc={acc:.4f}  n={total}")

	overall_acc = overall_correct / max(overall_total, 1)
	print("-" * 80)
	print(f"overall_acc={overall_acc:.4f}  total={overall_total}")


if __name__ == "__main__":
	main()