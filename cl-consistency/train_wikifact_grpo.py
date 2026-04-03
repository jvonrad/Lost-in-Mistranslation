#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom grouped-rollout multilingual RL trainer.

One training item = one fact.
For each fact:
  - create 12 separate single-language prompts
  - sample num_generations grouped rollouts
  - each grouped rollout contains 12 independent generations (one per language)
  - compute one joint reward over the 12 answers
  - normalize rewards across grouped rollouts for the same fact
  - apply that advantage to all 12 outputs in the rollout


## LORA WITH HF INFERENCE

  
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python cl-consistency/train_wikifact_grpo.py  \
	--model_id /data/jonathan/Lost-in-Mistranslation/models/olmo2-finetranslations-structured-lora-checkpoints/checkpoint-12400-merged \
	--dataset_id jonny-vr/WIKI-FACT \
	--output_dir /data/jonathan/Lost-in-Mistranslation/models/aligned-finetranslations-wikifact-grpo \
	--per_device_train_batch_size 1 \
	--num_train_epochs 1 \
	--learning_rate 5e-6 \
	--num_generations 8  \
	--max_completion_length 48 \
	--run_name ted-finetranslations-wikifact-grpo-new \
	--eval_steps 200  \
	--max_eval_wikifact 500 \
	--bf16 \
	--use_lora  \
	--kl_coef 0.0 \
	--max_train_samples 20000 

/data/jonathan/Lost-in-Mistranslation/models/olmo2-ted-cultura-sw-bn-structured-lora-final/merged

## LORA WITH VLLM

CUDA_VISIBLE_DEVICES=5 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve allenai/OLMo-2-1124-7B-Instruct \
  --enable-lora \
  --max-lora-rank 32 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85
  
CUDA_VISIBLE_DEVICES=6 python train_wikifact_grpo.py  \
	--model_id allenai/OLMo-2-1124-7B-Instruct \
	--dataset_id jonny-vr/WIKI-FACT \
	--output_dir /data/jonathan/Lost-in-Mistranslation/models/wikifact_grouped_rollout \
	--per_device_train_batch_size 1 \
	--num_train_epochs 1 \
	--learning_rate 5e-6 \
	--num_generations 8  \
	--max_completion_length 48 \
	--run_name grouped_rollout_v1_grpo_vllm \
	--eval_steps 200  \
	--max_eval_wikifact 500 \
	--bf16 \
	--use_lora  \
	--use_vllm \
	--vllm_base_url http://localhost:8000  \
	--vllm_sync_steps 10 \
	--lora_sync_dir /tmp/lora_checkpoint \
	--kl_coef 0.0 \
	--max_train_samples 20000 


## FULL PARAMETER FT Example:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,7 python train_wikifact_grpo.py \
  --model_id allenai/OLMo-2-1124-7B-Instruct \
  --dataset_id jonny-vr/WIKI-FACT \
  --output_dir /data/jonathan/Lost-in-Mistranslation/models/wikifact_grpo_vllm \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --num_generations 4 \
  --max_completion_length 48 \
  --run_name grouped_rollout_v1_grpo_full \
  --eval_steps 5 \
  --max_eval_wikifact 500 \
  --bf16 \
  --max_train_samples 20000 
"""

import os
import re
import json
import math
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple
import shutil                      # FIX 1: removed duplicate `import shutil`
import numpy as np
import evaluate
from datasets import load_dataset, Dataset   # FIX 2: consolidated duplicate `from datasets import ...`

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader



MODEL_ID = "allenai/OLMo-2-1124-7B-Instruct"
HF_DATASET_ID = "jonny-vr/WIKI-FACT"
OUTPUT_DIR = "/data/jonathan/Lost-in-Mistranslation/models/wikifact_grouped_rollout_grpo"

WANDB_PROJECT = "UnLock"

LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh", "ja", "ar", "sw", "bn"]

LANG_TO_NAME = {
	"en": "English",
	"de": "German",
	"id": "Indonesian",
	"pt": "Portuguese",
	"ar": "Arabic",
	"bn": "Bengali",
	"sw": "Swahili",
	"es": "Spanish",
	"ru": "Russian",
	"fr": "French",
	"ja": "Japanese",
	"zh": "Chinese",
}

LANG_NAME_MAP = {
	"ar": "Arabic",
	"bn": "Bengali",
	"de": "German",
	"es": "Spanish",
	"fr": "French",
	"id": "Indonesian",
	"ja": "Japanese",
	"pt": "Portuguese",
	"ru": "Russian",
	"sw": "Swahili",
	"zh": "Chinese",
}

FLORES_LANG_MAP = {
	"ar": "arb_Arab", "bn": "ben_Beng", "de": "deu_Latn",
	"es": "spa_Latn", "fr": "fra_Latn", "id": "ind_Latn",
	"ja": "jpn_Jpan", "pt": "por_Latn", "ru": "rus_Cyrl",
	"sw": "swh_Latn", "zh": "zho_Hans", "en": "eng_Latn",
}

# global-mmlu
MAX_EVAL_SAMPLES_PER_LANG = 267

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
VALID_LETTERS = {"A", "B", "C", "D"}

LORA_R = 32
LORA_ALPHA = 64


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--model_id", type=str, default=MODEL_ID)
	ap.add_argument("--dataset_id", type=str, default=HF_DATASET_ID)
	ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
	ap.add_argument("--run_name", type=str, default=None)
	ap.add_argument("--logprob_micro_batch_size", type=int, default=4)
	# In parse_args(), add:
	ap.add_argument("--use_lora", action="store_true", default=False)

	ap.add_argument("--use_vllm", action="store_true", default=False)
	ap.add_argument("--vllm_base_url", type=str, default="http://localhost:8000")
	ap.add_argument("--vllm_sync_steps", type=int, default=10)
	ap.add_argument("--lora_sync_dir", type=str, default="/tmp/lora_checkpoint")

	ap.add_argument("--learning_rate", type=float, default=5e-6)
	# In parse_args():
	ap.add_argument("--max_train_samples", type=int, default=None)
	ap.add_argument("--weight_decay", type=float, default=0.01)
	ap.add_argument("--warmup_ratio", type=float, default=0.03)
	ap.add_argument("--num_train_epochs", type=float, default=1.0)

	ap.add_argument("--per_device_train_batch_size", type=int, default=1)
	ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

	ap.add_argument("--max_prompt_length", type=int, default=512)
	ap.add_argument("--max_completion_length", type=int, default=48)
	ap.add_argument("--num_generations", type=int, default=4)
	ap.add_argument("--temperature", type=float, default=0.7)
	ap.add_argument("--top_p", type=float, default=0.95)

	ap.add_argument("--logging_steps", type=int, default=10)
	ap.add_argument("--eval_steps", type=int, default=200)
	ap.add_argument("--max_eval_wikifact", type=int, default=250)

	ap.add_argument("--min_languages", type=int, default=12)

	ap.add_argument("--coverage_reward_weight", type=float, default=0.05)
	ap.add_argument("--valid_option_reward_weight", type=float, default=0.15)
	ap.add_argument("--all_correct_bonus", type=float, default=0.25)
	ap.add_argument("--max_eval_flores", type=int, default=64)
	ap.add_argument("--kl_coef", type=float, default=0.05)

	ap.add_argument("--bf16", action="store_true", default=True)
	ap.add_argument("--no_bf16", action="store_true")
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--report_to", type=str, default="wandb")
	return ap.parse_args()




#############
# VLLM Setup
#############

import requests


def sync_lora_to_vllm(model, tokenizer, sync_dir, vllm_base_url, adapter_name="active-lora"):
	"""Save LoRA weights and tell vLLM to reload them."""
	model.save_pretrained(sync_dir)
	tokenizer.save_pretrained(sync_dir)
	resp = requests.post(
		f"{vllm_base_url}/v1/load_lora_adapter",
		json={
			"lora_name": adapter_name,
			"lora_path": sync_dir,
			"load_inplace": True,
		},
	)
	if resp.status_code != 200:
		print(f"WARNING: LoRA sync failed: {resp.text}", flush=True)


def generate_via_vllm(
	flat_prompts,
	vllm_base_url,
	max_completion_length,
	temperature,
	top_p,
	adapter_name="active-lora",
):
	"""Batch generate via the vLLM OpenAI-compatible API."""
	resp = requests.post(
		f"{vllm_base_url}/v1/completions",
		json={
			"model": adapter_name,
			"prompt": flat_prompts,          # send ALL prompts in one call
			"max_tokens": max_completion_length,
			"temperature": temperature,
			"top_p": top_p,
			"repetition_penalty": 1.3,
		},
	)
	data = resp.json()
	if "choices" not in data:
		raise RuntimeError(f"vLLM error: {data}")
	# Sort by index since vLLM may return out of order
	choices = sorted(data["choices"], key=lambda x: x["index"])
	return [c["text"] for c in choices]

def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def safe_strip(x: Any) -> str:
	return x.strip() if isinstance(x, str) else ""


def answer_text_to_letter(options: List[str], answer_text: str) -> Optional[str]:
	answer_text = safe_strip(answer_text)
	options = [safe_strip(x) for x in options]
	for i, opt in enumerate(options):
		if opt == answer_text:
			return INDEX_TO_LETTER[i]
	return None


def log_sample_rollout_to_file(
	output_dir: str,
	global_step: int,
	grouped_preds: Dict,
	group_rewards: Dict,
	batch: Dict,
):
	log_path = os.path.join(output_dir, "finetranslations.txt")
	sample_key = sorted(grouped_preds.keys())[0]
	sample_fact_idx, sample_gen_idx = sample_key

	# get meta for gold answers
	meta_by_lang = json.loads(batch["meta_by_lang_json"][sample_fact_idx])
	fact_id = batch["fact_id"][sample_fact_idx]

	with open(log_path, "a", encoding="utf-8") as f:
		f.write(f"\n{'='*60}\n")
		f.write(f"step={global_step} | fact_id={fact_id} | gen_idx={sample_gen_idx}\n")
		f.write(f"{'='*60}\n")
		for lang in LANGS:
			if lang in grouped_preds[sample_key]:
				pred = grouped_preds[sample_key][lang]
				gold = meta_by_lang.get(lang, {}).get("gold_text", "?")
				correct = "✓" if pred.strip() == gold.strip() or (
					resolve_prediction_to_letter(pred, meta_by_lang[lang]["options"])[0]
					== meta_by_lang[lang]["gold_letter"]
				) else "✗"
				f.write(f"  [{lang}] {correct} pred: {pred}\n")
				f.write(f"       gold: {gold}\n")
		reward = group_rewards[sample_key]["reward"]
		advantage = group_rewards[sample_key]["advantage"]
		f.write(f"reward={round(reward, 4)} advantage={round(advantage, 4)}\n")
##################
# Global MMLU
##################



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


def add_global_mmlu_avg(metrics_dict):
	lang_accs = []
	for lang in LANGS:
		key = f"eval_{lang}_accuracy"
		if key in metrics_dict:
			lang_accs.append(metrics_dict[key])

	if lang_accs:
		metrics_dict["eval_accuracy_avg"] = float(np.mean(lang_accs))

	return metrics_dict

def preprocess_logits_for_metrics(logits, labels):
	if isinstance(logits, tuple):
		logits = logits[0]
	# logits: [B, seq_len, vocab_size]
	# keep only answer-token columns A/B/C/D
	return logits[:, :, CHOICE_TOKEN_IDS]   # [B, seq_len, 4]


def make_mmlu_accuracy_fn(tokenizer):
	choice_ids = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]

	def compute_mmlu_accuracy(eval_pred):
		logits, labels = eval_pred   # logits: [N, seq_len, 4]

		preds = []
		golds = []

		for i in range(labels.shape[0]):
			pos = np.where(labels[i] != -100)[0]
			if len(pos) == 0:
				continue

			j = int(pos[0])   # answer token position
			if j == 0:
				continue

			gold_token = int(labels[i, j])
			if gold_token not in choice_ids:
				continue
			gold_idx = choice_ids.index(gold_token)

			# causal LM predicts token at position j using logits at j-1
			pred_idx = int(np.argmax(logits[i, j - 1]))

			preds.append(pred_idx)
			golds.append(gold_idx)

		acc = float(np.mean(np.array(preds) == np.array(golds))) if golds else 0.0
		return {"accuracy": acc}

	return compute_mmlu_accuracy

##########################
# FLORES BLEU evaluation
##########################

def load_flores_parallel_subset(target_langs, split="dev", max_samples=64):
	flores = {}
	for lang in target_langs:
		if lang == "en":
			continue

		pair_cfg = f"{FLORES_LANG_MAP['en']}-{FLORES_LANG_MAP[lang]}"
		ds = load_dataset("Muennighoff/flores200", pair_cfg, split=split)
		ds = ds.select(range(min(max_samples, len(ds))))

		flores[lang] = {
			"src_texts": ds[f"sentence_{FLORES_LANG_MAP['en']}"],
			"tgt_texts": ds[f"sentence_{FLORES_LANG_MAP[lang]}"],
		}
	return flores

@torch.no_grad()
def compute_flores_bleu(
	model,
	tokenizer,
	flores_sets,
	device,
	max_new_tokens=128,
	batch_size=4,
):
	bleu = evaluate.load("sacrebleu")
	metrics = {}

	model.eval()

	for lang, data in flores_sets.items():
		src_texts = data["src_texts"]
		refs = data["tgt_texts"]

		preds = []

		for i in range(0, len(src_texts), batch_size):
			batch_src = src_texts[i:i + batch_size]

			prompts = [
				f"Translate the following sentence from English to {LANG_NAME_MAP[lang]}:\n\n{s}\n\nTranslation:"
				for s in batch_src
			]

			tok = tokenizer(
				prompts,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=512,
			).to(device)

			gen = model.generate(
				**tok,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
			)

			gen_only = gen[:, tok["input_ids"].shape[1]:]
			decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
			preds.extend([x.strip() for x in decoded])

		score = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
		metrics[f"flores_bleu/{lang}"] = score

	if len(metrics) > 0:
		metrics["flores_bleu/avg"] = sum(metrics.values()) / len(metrics)

	return metrics

###################################
# Hidden State Cosine Similarity 
#####################################

def mean_pool_hidden(hidden_states, attention_mask):
	mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
	summed = (hidden_states * mask).sum(dim=1)
	counts = mask.sum(dim=1).clamp(min=1)
	return summed / counts


@torch.no_grad()
def compute_flores_hidden_cosine(
	model,
	tokenizer,
	flores_sets,
	device,
	batch_size=4,
	max_length=512,
):
	metrics = {}
	model.eval()

	core_model = model.module if hasattr(model, "module") else model
	n_layers = core_model.config.num_hidden_layers
	mid_layer = n_layers // 2
	last_layer = n_layers

	all_mid = []
	all_last = []

	for lang, data in flores_sets.items():
		src_texts = data["src_texts"]
		tgt_texts = data["tgt_texts"]

		for i in range(0, len(src_texts), batch_size):
			batch_src = src_texts[i:i + batch_size]
			batch_tgt = tgt_texts[i:i + batch_size]

			tok_src = tokenizer(
				batch_src,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=max_length,
			).to(device)

			tok_tgt = tokenizer(
				batch_tgt,
				return_tensors="pt",
				padding=True,
				truncation=True,
				max_length=max_length,
			).to(device)

			out_src = model(**tok_src, output_hidden_states=True, use_cache=False)
			out_tgt = model(**tok_tgt, output_hidden_states=True, use_cache=False)

			src_mid = mean_pool_hidden(out_src.hidden_states[mid_layer], tok_src["attention_mask"])
			tgt_mid = mean_pool_hidden(out_tgt.hidden_states[mid_layer], tok_tgt["attention_mask"])

			src_last = mean_pool_hidden(out_src.hidden_states[last_layer], tok_src["attention_mask"])
			tgt_last = mean_pool_hidden(out_tgt.hidden_states[last_layer], tok_tgt["attention_mask"])

			cos_mid = F.cosine_similarity(src_mid, tgt_mid, dim=-1)
			cos_last = F.cosine_similarity(src_last, tgt_last, dim=-1)

			all_mid.extend(cos_mid.detach().float().cpu().tolist())
			all_last.extend(cos_last.detach().float().cpu().tolist())
   
			del tok_src, tok_tgt, out_src, out_tgt, src_mid, tgt_mid, src_last, tgt_last, cos_mid, cos_last
			torch.cuda.empty_cache()

	metrics["hidden_cosine_mid/avg"] = float(np.mean(all_mid)) if all_mid else 0.0
	metrics["hidden_cosine_last/avg"] = float(np.mean(all_last)) if all_last else 0.0

	return metrics



def normalize_text(text: str) -> str:
	text = safe_strip(text).lower()
	text = text.replace("\u2019", "'").replace("`", "'").replace("\u201c", '"').replace("\u201d", '"')
	text = re.sub(r"^[\"'`\s:;\-\u2013\u2014\(\)\[\]\{\}]+", "", text)
	text = re.sub(r"[\"'`\s:;\-\u2013\u2014\(\)\[\]\{\}\.,!?]+$", "", text)
	text = re.sub(r"\s+", " ", text)
	return text


def extract_answer_text(text: str) -> str:
	text = safe_strip(text)
	text = re.sub(r"^\s*(answer|answer text)\s*:\s*", "", text, flags=re.I)
	text = text.split("\n")[0].strip()
	return text


def resolve_prediction_to_letter(pred_text: str, option_map: Dict[str, str]) -> Tuple[Optional[str], bool]:
	pred_raw = extract_answer_text(pred_text)
	pred_norm = normalize_text(pred_raw)
	if not pred_norm:
		return None, False

	option_norm = {letter: normalize_text(text) for letter, text in option_map.items()}

	for letter, opt_norm in option_norm.items():
		if pred_norm == opt_norm:
			return letter, True

	m = re.match(r"^([abcd])(?:[\.\)\]:\-\s]|$)", pred_norm)
	if m:
		return m.group(1).upper(), True

	candidates = []
	for letter, opt_norm in option_norm.items():
		if opt_norm and (pred_norm in opt_norm or opt_norm in pred_norm):
			candidates.append(letter)

	if len(candidates) == 1:
		return candidates[0], True

	return None, False


def build_single_language_prompt(lang: str, question: str, options: Dict[str, str]) -> str:
	return (
		f"You will be given one factual multiple-choice question in {LANG_TO_NAME[lang]}.\n"
		f"Return only the full answer text in {LANG_TO_NAME[lang]}.\n"
		f"Do not return the letter.\n"
		f"Do not explain.\n\n"
		f"Question: {question}\n"
		f"A. {options['A']}\n"
		f"B. {options['B']}\n"
		f"C. {options['C']}\n"
		f"D. {options['D']}\n\n"
		f"Answer text:"
	)


def build_grouped_fact_item(ex: Dict[str, Any]) -> Dict[str, Any]:
	langs_data = ex.get("langs", {})
	if not isinstance(langs_data, dict):
		return {"is_valid": False, "num_languages": 0}

	prompts_by_lang = {}
	meta_by_lang = {}

	for lang in LANGS:
		if lang not in langs_data:
			continue

		item = langs_data[lang]
		question = safe_strip(item.get("question", ""))
		answer_text = safe_strip(item.get("answer_text", ""))
		options = item.get("options", [])

		if not question or not isinstance(options, list) or len(options) != 4:
			continue

		options = [safe_strip(x) for x in options]
		if any(not x for x in options):
			continue

		gold_letter = answer_text_to_letter(options, answer_text)
		if gold_letter is None:
			continue

		option_map = {
			"A": options[0],
			"B": options[1],
			"C": options[2],
			"D": options[3],
		}

		prompts_by_lang[lang] = build_single_language_prompt(lang, question, option_map)
		meta_by_lang[lang] = {
			"gold_letter": gold_letter,
			"gold_text": answer_text,
			"options": option_map,
		}

	return {
		"fact_id": ex.get("fact_id", ""),
		"prompts_by_lang_json": json.dumps(prompts_by_lang, ensure_ascii=False, sort_keys=True),
		"meta_by_lang_json": json.dumps(meta_by_lang, ensure_ascii=False, sort_keys=True),
		"num_languages": len(meta_by_lang),
		"is_valid": len(meta_by_lang) > 0,
	}


def compute_group_reward(
	pred_text_by_lang: Dict[str, str],
	meta_by_lang: Dict[str, Any],
	coverage_weight: float,
	valid_option_weight: float,
	all_correct_bonus: float,
) -> Dict[str, float]:
	score = 0.0
	n_correct = 0
	n_valid = 0
	n_pred = 0

	for lang, meta in meta_by_lang.items():
		pred = pred_text_by_lang.get(lang, "")
		resolved_letter, matched_valid = resolve_prediction_to_letter(pred, meta["options"])

		if resolved_letter == meta["gold_letter"]:
			n_correct += 1
			score += 1.0
		elif safe_strip(pred) and not matched_valid:
			score -= 0.5  # hallucination penalty

		if matched_valid:
			n_valid += 1
		if safe_strip(pred):
			n_pred += 1

	if n_correct == len(meta_by_lang):
		score += 1.0

	return {
		"score": score,
		"n_correct": n_correct,
		"n_valid": n_valid,
		"n_pred": n_pred,
	}
	
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
	return {
		"fact_id": [x["fact_id"] for x in batch],
		"prompts_by_lang_json": [x["prompts_by_lang_json"] for x in batch],
		"meta_by_lang_json": [x["meta_by_lang_json"] for x in batch],
	}


@torch.no_grad()
def evaluate_wikifact_grouped(
	model,
	tokenizer,
	eval_ds: Dataset,
	max_prompt_length: int,
	max_completion_length: int,
) -> Dict[str, float]:
	was_training = model.training
	model.eval()

	total_examples = 0
	total_slots = 0
	total_correct = 0
	total_valid = 0
	total_all_correct = 0

	per_lang_correct = {lang: 0 for lang in LANGS}
	per_lang_total = {lang: 0 for lang in LANGS}

	device = next(model.parameters()).device

	for ex in eval_ds:
		prompts_by_lang = json.loads(ex["prompts_by_lang_json"])
		meta_by_lang = json.loads(ex["meta_by_lang_json"])

		langs = [lang for lang in LANGS if lang in prompts_by_lang and lang in meta_by_lang]
		prompts = [prompts_by_lang[lang] for lang in langs]

		inputs = tokenizer(
			prompts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=max_prompt_length,
		).to(device)

		input_len = inputs["input_ids"].shape[1]

		outputs = model.generate(
			**inputs,
			max_new_tokens=max_completion_length,
			do_sample=False,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

		pred_text_by_lang = {}
		for i, lang in enumerate(langs):
			gen_ids = outputs[i][input_len:]
			gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
			pred_text_by_lang[lang] = extract_answer_text(gen_text)

		total_examples += 1
		ex_all_correct = True

		for lang in langs:
			total_slots += 1
			per_lang_total[lang] += 1

			resolved_letter, matched_valid = resolve_prediction_to_letter(
				pred_text_by_lang.get(lang, ""),
				meta_by_lang[lang]["options"],
			)

			if matched_valid:
				total_valid += 1

			if resolved_letter is not None and resolved_letter == meta_by_lang[lang]["gold_letter"]:
				total_correct += 1
				per_lang_correct[lang] += 1
			else:
				ex_all_correct = False

		if ex_all_correct:
			total_all_correct += 1

	metrics = {
		"wikifact/slot_accuracy": total_correct / total_slots if total_slots else 0.0,
		"wikifact/resolution_rate": total_valid / total_slots if total_slots else 0.0,
		"wikifact/all_correct_rate": total_all_correct / total_examples if total_examples else 0.0,
		"wikifact/n_examples": float(total_examples),
	}

	for lang in LANGS:
		metrics[f"wikifact/lang_acc_{lang}"] = (
			per_lang_correct[lang] / per_lang_total[lang]
			if per_lang_total[lang] else 0.0
		)

	if was_training:
		model.train()

	return metrics



@torch.no_grad()
def run_full_eval(
	model,
	tokenizer,
	wikifact_val_ds,
	flores_eval_sets,
	mmlu_eval_sets,
	max_prompt_length,
	max_completion_length,
	device,
	global_step
):
	metrics = {}

	# FIX 3: Removed the stale `loader` that referenced undefined `ds`.
	#         The MMLU loader is built per-language below where it's actually needed.

	# WikiFact
	metrics.update(evaluate_wikifact_grouped(
		model=model,
		tokenizer=tokenizer,
		eval_ds=wikifact_val_ds,
		max_prompt_length=max_prompt_length,
		max_completion_length=max_completion_length,
	))

	if global_step % 1000 == 0:
		metrics.update(compute_flores_bleu(
			model=model,
			tokenizer=tokenizer,
			flores_sets=flores_eval_sets,
			device=device,
			max_new_tokens=128,
			batch_size=4,
		))

		metrics.update(compute_flores_hidden_cosine(
			model=model,
			tokenizer=tokenizer,
			flores_sets=flores_eval_sets,
			device=device,
			batch_size=1,
		))

	# Global MMLU per language
	model.eval()
	choice_ids = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]
	for lang, ds in mmlu_eval_sets.items():
		# FIX 4: Build a proper collate_fn for the MMLU DataLoader so that
		#         variable-length tokenized examples are padded into tensors,
		#         instead of relying on the default collator which would fail
		#         on ragged lists.
		def mmlu_collate_fn(batch):
			return {
				"input_ids": pad_sequence(
					[torch.tensor(x["input_ids"]) for x in batch],
					batch_first=True,
					padding_value=tokenizer.pad_token_id,
				),
				"attention_mask": pad_sequence(
					[torch.tensor(x["attention_mask"]) for x in batch],
					batch_first=True,
					padding_value=0,
				),
				"labels": pad_sequence(
					[torch.tensor(x["labels"]) for x in batch],
					batch_first=True,
					padding_value=-100,
				),
			}

		loader = TorchDataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			collate_fn=mmlu_collate_fn,
		)
		correct = 0
		total = 0
		for batch in loader:
			# FIX 5: batch values are already tensors from collate_fn,
			#         no need to wrap in torch.tensor() again (which would
			#         trigger a deprecation warning / copy).
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			labels = batch["labels"]  # keep on CPU for iteration
			with torch.no_grad():
				logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
			for i in range(labels.shape[0]):
				label_row = labels[i]
				label_pos = (label_row != -100).nonzero(as_tuple=True)[0]
				if len(label_pos) == 0:
					continue
				j = int(label_pos[0])
				gold_token = int(label_row[j].item())
				if gold_token not in choice_ids:
					continue
				gold_idx = choice_ids.index(gold_token)
				pred_idx = int(logits[i, j - 1, choice_ids].argmax().item())
				correct += int(pred_idx == gold_idx)
				total += 1
		metrics[f"mmlu/acc_{lang}"] = correct / total if total else 0.0

	if any(f"mmlu/acc_{lang}" in metrics for lang in LANGS):
		metrics["mmlu/acc_avg"] = float(np.mean([
			metrics[f"mmlu/acc_{lang}"]
			for lang in LANGS if f"mmlu/acc_{lang}" in metrics
		]))

	return metrics

def gather_rollout_prompts(
	batch: Dict[str, List[Any]],
	num_generations: int,
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
	"""
	Returns:
	  flat_prompts: [prompt_0, prompt_1, ...]
	  flat_index: [(fact_idx, gen_idx, lang), ...]
	"""
	flat_prompts = []
	flat_index = []

	for fact_idx, prompts_json in enumerate(batch["prompts_by_lang_json"]):
		prompts_by_lang = json.loads(prompts_json)

		langs = [lang for lang in LANGS if lang in prompts_by_lang]
		for gen_idx in range(num_generations):
			for lang in langs:
				flat_prompts.append(prompts_by_lang[lang])
				flat_index.append((fact_idx, gen_idx, lang))

	return flat_prompts, flat_index


@torch.no_grad()
def generate_grouped_rollouts(
	model,
	tokenizer,
	batch,
	num_generations,
	max_prompt_length,
	max_completion_length,
	temperature,
	top_p,
	# New params:
	use_vllm=False,
	vllm_base_url=None,
):
	device = next(model.parameters()).device
	flat_prompts, flat_index = gather_rollout_prompts(batch, num_generations)

	if use_vllm:
		# ---- vLLM path: fast generation via server ----
		generated_texts = generate_via_vllm(
			flat_prompts=flat_prompts,
			vllm_base_url=vllm_base_url,
			max_completion_length=max_completion_length,
			temperature=temperature,
			top_p=top_p,
		)

		results = {}
		seq_payloads = []

		for i, (fact_idx, gen_idx, lang) in enumerate(flat_index):
			gen_text = generated_texts[i]

			key = (fact_idx, gen_idx)
			results.setdefault(key, {})
			results[key][lang] = extract_answer_text(gen_text)

			# We still need token IDs for the logprob loss computation
			full_text = flat_prompts[i] + gen_text
			full_ids = tokenizer(
				full_text,
				add_special_tokens=False,
				truncation=True,
				max_length=max_prompt_length + max_completion_length,
			)["input_ids"]

			prompt_ids = tokenizer(
				flat_prompts[i],
				add_special_tokens=False,
				truncation=True,
				max_length=max_prompt_length,
			)["input_ids"]

			seq_payloads.append({
				"fact_idx": fact_idx,
				"gen_idx": gen_idx,
				"lang": lang,
				"input_ids": torch.tensor(full_ids, dtype=torch.long),
				"input_len": len(prompt_ids),
				"total_len": len(full_ids),
				"generated_text": gen_text,
			})

		return results, seq_payloads

	else:
		# ---- Original HF generate path ----
		was_training = model.training
		model.eval()

		inputs = tokenizer(
			flat_prompts,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=max_prompt_length,
		).to(device)

		input_len = inputs["input_ids"].shape[1]

		outputs = model.generate(
			**inputs,
			do_sample=True,
			temperature=temperature,
			top_p=top_p,
			max_new_tokens=max_completion_length,
			pad_token_id=tokenizer.pad_token_id,
			repetition_penalty=1.3,
			eos_token_id=tokenizer.eos_token_id,
		)

		if was_training:
			model.train()

		results = {}
		seq_payloads = []

		for i, (fact_idx, gen_idx, lang) in enumerate(flat_index):
			generated_ids = outputs[i][input_len:]
			generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

			key = (fact_idx, gen_idx)
			results.setdefault(key, {})
			results[key][lang] = extract_answer_text(generated_text)

			seq_payloads.append({
				"fact_idx": fact_idx,
				"gen_idx": gen_idx,
				"lang": lang,
				"input_ids": outputs[i].detach().cpu(),
				"input_len": input_len,
				"total_len": int(outputs[i].shape[0]),
				"generated_text": generated_text,
			})

		return results, seq_payloads

def compute_group_advantages(
	batch: Dict[str, List[Any]],
	grouped_preds: Dict[Tuple[int, int], Dict[str, str]],
	num_generations: int,
	coverage_weight: float,
	valid_option_weight: float,
	all_correct_bonus: float,
):
	group_rewards = {}
	group_stats = {}

	for fact_idx, meta_json in enumerate(batch["meta_by_lang_json"]):
		meta_by_lang = json.loads(meta_json)

		rewards = []
		for gen_idx in range(num_generations):
			preds = grouped_preds.get((fact_idx, gen_idx), {})
			stats = compute_group_reward(
				preds,
				meta_by_lang,
				coverage_weight=coverage_weight,
				valid_option_weight=valid_option_weight,
				all_correct_bonus=all_correct_bonus,
			)
			group_stats[(fact_idx, gen_idx)] = stats
			rewards.append(stats["score"])

		rewards_t = torch.tensor(rewards, dtype=torch.float32)
		mean = rewards_t.mean()
		std = rewards_t.std(unbiased=False)
		advantages = (rewards_t - mean) / (std + 1e-6)

		for gen_idx in range(num_generations):
			group_rewards[(fact_idx, gen_idx)] = {
				"reward": float(rewards[gen_idx]),
				"advantage": float(advantages[gen_idx].item()),
			}

	return group_rewards, group_stats

def compute_logprob_loss(
	model,
	ref_model,
	seq_payloads: List[Dict[str, Any]],
	group_rewards: Dict[Tuple[int, int], Dict[str, float]],
	kl_coef: float,
	pad_token_id: int,
	device,
	micro_batch_size: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
	if not seq_payloads:
		zero = torch.tensor(0.0, device=device)
		return zero, {"mean_reward": 0.0, "reward_std": 0.0, "mean_advantage": 0.0, "mean_kl": 0.0}

	losses = []
	rewards = []
	advantages = []
	kls = []

	for chunk_start in range(0, len(seq_payloads), micro_batch_size):
		chunk = seq_payloads[chunk_start : chunk_start + micro_batch_size]

		input_ids = torch.nn.utils.rnn.pad_sequence(
			[x["input_ids"] for x in chunk],
			batch_first=True,
			padding_value=pad_token_id,
		).to(device)

		lengths = [x["total_len"] for x in chunk]
		attention_mask = torch.zeros_like(input_ids)
		for i, l in enumerate(lengths):
			attention_mask[i, :l] = 1

		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = outputs.logits[:, :-1, :]
		target_ids = input_ids[:, 1:]
		logprobs = torch.log_softmax(logits, dim=-1)
		token_logprobs = torch.gather(logprobs, -1, target_ids.unsqueeze(-1)).squeeze(-1)

		with torch.no_grad():
			if ref_model is not None and kl_coef > 0.0:
				ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
				ref_logits = ref_outputs.logits[:, :-1, :]
				ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
				ref_token_logprobs = torch.gather(ref_logprobs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
			else:
				ref_token_logprobs = None

		for i, payload in enumerate(chunk):
			fact_idx = payload["fact_idx"]
			gen_idx = payload["gen_idx"]
			input_len = payload["input_len"]
			total_len = payload["total_len"]

			reward = group_rewards[(fact_idx, gen_idx)]["reward"]
			advantage = group_rewards[(fact_idx, gen_idx)]["advantage"]

			start = max(input_len - 1, 0)
			end = total_len - 1
			if end <= start:
				continue

			gen_logprob_mean = token_logprobs[i, start:end].mean()
			seq_loss = -advantage * gen_logprob_mean

			if ref_token_logprobs is not None:
				seq_kl = (token_logprobs[i, start:end] - ref_token_logprobs[i, start:end]).mean()
				seq_loss = seq_loss + kl_coef * seq_kl
				kls.append(float(seq_kl.detach().item()))

			losses.append(seq_loss)
			rewards.append(reward)
			advantages.append(advantage)

	if not losses:
		zero = torch.tensor(0.0, device=device)
		return zero, {"mean_reward": 0.0, "reward_std": 0.0, "mean_advantage": 0.0, "mean_kl": 0.0}

	loss = torch.stack(losses).mean()

	stats = {
		"mean_reward": float(torch.tensor(rewards).mean().item()),
		"reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if len(rewards) > 1 else 0.0,
		"mean_advantage": float(torch.tensor(advantages).mean().item()),
		"mean_kl": float(torch.tensor(kls).mean().item()) if kls else 0.0,
	}
	return loss, stats

def main():
	args = parse_args()
	if args.no_bf16:
		args.bf16 = False

	set_seed(args.seed)
	os.makedirs(args.output_dir, exist_ok=True)

	if args.report_to == "wandb":
		wandb.init(
			project=WANDB_PROJECT,
			name=args.run_name,
			config=vars(args),
		)

	print(f"Loading tokenizer for {args.model_id} ...", flush=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"

	dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
	
	global CHOICE_TOKEN_IDS
	CHOICE_TOKEN_IDS = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]
	
	# Flores loading
	flores_eval_sets = load_flores_parallel_subset(
		target_langs=["ar", "bn", "de", "es", "fr", "id", "ja", "pt", "ru", "sw", "zh"],
		split="dev",
		max_samples=args.max_eval_flores,
	)
	# MMLU loading
	mmlu_eval_sets = load_global_mmlu_dev_eval_by_lang(LANGS, tokenizer)

	print(f"Loading model for {args.model_id} ...", flush=True)
	base_model = AutoModelForCausalLM.from_pretrained(
		args.model_id,
		torch_dtype=dtype,
		device_map="auto",
	)

	if args.use_lora:
		peft_config = LoraConfig(
			r=LORA_R,
			lora_alpha=LORA_ALPHA,
			target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
			lora_dropout=0.05,
			bias="none",
			task_type=TaskType.CAUSAL_LM,
		)
		model = get_peft_model(base_model, peft_config)
		
	else:
		model = base_model
		
		
		# Right after get_peft_model, do initial sync:
	if args.use_vllm and args.use_lora:
		os.makedirs(args.lora_sync_dir, exist_ok=True)
		model.save_pretrained(args.lora_sync_dir)
		tokenizer.save_pretrained(args.lora_sync_dir)
		print(f"Initial LoRA checkpoint saved to {args.lora_sync_dir}", flush=True)

		# Load it into vLLM
		resp = requests.post(
			f"{args.vllm_base_url}/v1/load_lora_adapter",
			json={
				"lora_name": "active-lora",
				"lora_path": args.lora_sync_dir,
			},
		)
		if resp.status_code == 200:
			print("Initial LoRA adapter loaded into vLLM", flush=True)
		else:
			print(f"WARNING: Failed to load initial adapter: {resp.text}", flush=True)    

		
	model.gradient_checkpointing_enable()
	model.train()

	ref_model = None
	if args.kl_coef > 0.0:
		ref_model = AutoModelForCausalLM.from_pretrained(
			args.model_id,
			torch_dtype=dtype,
			device_map="auto",
		)
		ref_model.eval()
		for p in ref_model.parameters():
			p.requires_grad = False

	print(f"Loading dataset {args.dataset_id} ...", flush=True)
	raw_all = load_dataset(args.dataset_id)
	raw_train = raw_all["train"]
	raw_val = raw_all["validation"]
	
	print("KL coef:", args.kl_coef)
	print("Using LoRA:", args.use_lora)

	print("Building grouped train split ...", flush=True)
	train_ds = raw_train.map(build_grouped_fact_item, num_proc=32)

	print("Building grouped validation split ...", flush=True)
	val_ds = raw_val.map(build_grouped_fact_item)

	train_ds = train_ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= args.min_languages)
	val_ds = val_ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= args.min_languages)
	
	if args.max_train_samples is not None:
		train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))

	keep_cols = ["fact_id", "prompts_by_lang_json", "meta_by_lang_json"]
	train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
	val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

	if args.max_eval_wikifact is not None:
		val_ds = val_ds.select(range(min(args.max_eval_wikifact, len(val_ds))))

	print(train_ds, flush=True)
	print(val_ds, flush=True)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.per_device_train_batch_size,
		shuffle=True,
		collate_fn=collate_fn,
	)

	total_update_steps = math.ceil(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
	warmup_steps = int(total_update_steps * args.warmup_ratio)

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)
	scheduler = get_cosine_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_update_steps,
	)

	device = next(model.parameters()).device
	global_step = 0

	print("Starting grouped-rollout training ...", flush=True)
	
	reward_ema = None
	ema_alpha = 0.05

	for epoch in range(math.ceil(args.num_train_epochs)):
		for step, batch in enumerate(train_loader):
			grouped_preds, seq_payloads = generate_grouped_rollouts(
				model=model,
				tokenizer=tokenizer,
				batch=batch,
				num_generations=args.num_generations,
				max_prompt_length=args.max_prompt_length,
				max_completion_length=args.max_completion_length,
				temperature=args.temperature,
				top_p=args.top_p,
				use_vllm=args.use_vllm,
				vllm_base_url=args.vllm_base_url,
			)

			group_rewards, group_stats = compute_group_advantages(
				batch=batch,
				grouped_preds=grouped_preds,
				num_generations=args.num_generations,
				coverage_weight=args.coverage_reward_weight,
				valid_option_weight=args.valid_option_reward_weight,
				all_correct_bonus=args.all_correct_bonus,
			)

			loss, loss_stats = compute_logprob_loss(
				model=model,
				ref_model=ref_model,
				seq_payloads=seq_payloads,
				group_rewards=group_rewards,
				kl_coef=args.kl_coef,
				pad_token_id=tokenizer.pad_token_id,
				device=device,
				micro_batch_size=args.logprob_micro_batch_size,
			)

			(loss / args.gradient_accumulation_steps).backward()

			if (step + 1) % args.gradient_accumulation_steps == 0:
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)
				global_step += 1
				
								# Sync LoRA weights to vLLM
				if args.use_vllm and args.use_lora and global_step % args.vllm_sync_steps == 0:
					sync_lora_to_vllm(
						model=model,
						tokenizer=tokenizer,
						sync_dir=args.lora_sync_dir,
						vllm_base_url=args.vllm_base_url,
					)

				if global_step % args.logging_steps == 0:
					rewards = [v["reward"] for v in group_rewards.values()]
					advantages = [v["advantage"] for v in group_rewards.values()]

					log_data = {
						"train/loss": float(loss.detach().item()),
						"train/grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
						"train/learning_rate": float(scheduler.get_last_lr()[0]),
						"train/reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if len(rewards) > 1 else 0.0,
						"train/adv_mean": float(sum(advantages) / len(advantages)) if advantages else 0.0,
						"train/mean_kl": loss_stats.get("mean_kl", 0.0),
						"train/global_step": global_step,
					}

					# in the logging block, replace reward_mean logging
					reward_mean = float(sum(rewards) / len(rewards)) if rewards else 0.0
					if reward_ema is None:
						reward_ema = reward_mean
					else:
						reward_ema = ema_alpha * reward_mean + (1 - ema_alpha) * reward_ema

					log_data["train/reward_mean"] = reward_mean
					log_data["train/reward_mean_ema"] = reward_ema

					if args.report_to == "wandb":
						wandb.log(log_data, step=global_step)

					print(
						{
							"step": global_step,
							"loss": round(log_data["train/loss"], 4),
							"lr": f"{log_data['train/learning_rate']:.3e}",
							"reward_mean": round(log_data["train/reward_mean"], 4),
							"reward_std": round(log_data["train/reward_std"], 4),
							"adv_mean": round(log_data["train/adv_mean"], 4),
							"kl": round(log_data["train/mean_kl"], 6),
						},
						flush=True,
					)

					# print one sampled grouped rollout
					sample_key = sorted(grouped_preds.keys())[0]
					sample_fact_idx, sample_gen_idx = sample_key
					print(f"\n[sample rollout @ step {global_step}] fact_idx={sample_fact_idx} gen_idx={sample_gen_idx}", flush=True)
					for lang in LANGS:
						if lang in grouped_preds[sample_key]:
							print(f"{lang}: {grouped_preds[sample_key][lang]}", flush=True)
					print(
						"reward=",
						round(group_rewards[sample_key]["reward"], 4),
						"advantage=",
						round(group_rewards[sample_key]["advantage"], 4),
						flush=True,
					)
					
					if global_step % 200 == 0:
						log_sample_rollout_to_file(
							output_dir="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/grpo_samples",
							global_step=global_step,
							grouped_preds=grouped_preds,
							group_rewards=group_rewards,
							batch=batch,
						)

				if args.eval_steps > 0 and global_step % args.eval_steps == 0:
					torch.cuda.empty_cache()  # add this line
					eval_metrics = run_full_eval(
						model=model,
						tokenizer=tokenizer,
						wikifact_val_ds=val_ds,
						flores_eval_sets=flores_eval_sets,
						mmlu_eval_sets=mmlu_eval_sets,
						max_prompt_length=args.max_prompt_length,
						max_completion_length=args.max_completion_length,
						device=device,
						global_step=global_step
					)
					print(f"\n[eval @ step {global_step}] {eval_metrics}", flush=True)
					if args.report_to == "wandb":
						wandb.log(eval_metrics, step=global_step)

					# checkpointing
					checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
					model.save_pretrained(checkpoint_dir)
					tokenizer.save_pretrained(checkpoint_dir)
					checkpoints = sorted(
						[d for d in os.listdir(args.output_dir)
						if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()],
						key=lambda x: int(x.split("-")[-1])
					)
					for old in checkpoints[:-3]:
						shutil.rmtree(os.path.join(args.output_dir, old))
					
					model.train()
					torch.cuda.empty_cache()  # prevent OOM

			if global_step >= total_update_steps:
				break

		if global_step >= total_update_steps:
			break

	print(f"Saving model to {args.output_dir} ...", flush=True)
	model.save_pretrained(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)

	if args.report_to == "wandb":
		wandb.finish()


if __name__ == "__main__":
	main()