#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train OLMo-2-1124-7B on the pretokenized TED dataset using the same notebook
logic, except WITHOUT quantization.

Notebook-like choices preserved:
- LoRA r=8, alpha=16
- target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
- gradient checkpointing
- DataCollatorForLanguageModeling(mlm=False, pad_to_multiple_of=8)
- per_device_train_batch_size=8
- gradient_accumulation_steps=4
- learning_rate=2e-5
- cosine scheduler
- warmup_ratio=0.05
- weight_decay=0.01
- save_steps=200
- bf16=True
- optim="adamw_torch_fused"
 
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 training/train_ted_structured_lora.py


python training/train_ted_structured_lora.py --model_id allenai/OLMo-2-1124-7B \

Uses pretokenized dataset at:
	/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/ted_single_target_12_langs
"""

import os
import argparse
from datasets import load_dataset, concatenate_datasets
import torch
import wandb
from datasets import load_from_disk
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass
from typing import Any, Dict, List


import math
import evaluate
import torch.nn.functional as F
from transformers import TrainerCallback

MODEL_ID = "allenai/OLMo-2-1124-7B"
TOKENIZED_DATA_ROOT = "/data/jonathan/Lost-in-Mistranslation/datasets/tokenized"
MODEL_OUT_ROOT = "/data/jonathan/Lost-in-Mistranslation/models"

MODEL_ALIASES = {
	"olmo2-7b": "allenai/OLMo-2-1124-7B",
	"llama-3.1-8b": "meta-llama/Llama-3.1-8B",
	"qwen2.5-7b": "Qwen/Qwen2.5-7B",
	"gemma-2-9b": "google/gemma-2-9b",
}


def resolve_model_id(model_id_or_alias: str) -> str:
	return MODEL_ALIASES.get(model_id_or_alias, model_id_or_alias)


def default_dataset_path(model_id: str) -> str:
	model_name = model_id.split("/")[-1]
	return os.path.join(TOKENIZED_DATA_ROOT, f"{model_name}_ted_12_langs_extra_bn_sw")


def default_ckpt_dir(model_id: str, learning_rate: float) -> str:
	model_name = model_id.split("/")[-1]
	return os.path.join(MODEL_OUT_ROOT, f"{model_name}-ted-structured-lr-{learning_rate}-lora-checkpoints")


def default_output_dir(model_id: str, learning_rate: float) -> str:
	model_name = model_id.split("/")[-1]
	return os.path.join(MODEL_OUT_ROOT, f"{model_name}-ted-structured-lr-{learning_rate}-lora-final")

CKPT_DIR = ""
OUTPUT_DIR = ""
rank = int(os.environ.get("RANK", 0))
PER_DEVICE_BATCH_SIZE = 6
GRAD_ACCUM = 8

LORA_R = 32
LORA_ALPHA = 64

DATASET_PATH = ""
REQ_LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh-cn", "ja", "ar", "sw", "bn"]

NUM_EPOCHS = 1
WANDB_PROJECT = "UnLock"

FLORES_LANG_MAP = {
	"ar": "arb_Arab",
	"bn": "ben_Beng",
	"de": "deu_Latn",
	"es": "spa_Latn",
	"fr": "fra_Latn",
	"id": "ind_Latn",
	"ja": "jpn_Jpan",
	"pt": "por_Latn",
	"ru": "rus_Cyrl",
	"sw": "swh_Latn",
	"zh": "zho_Hans",
	"en": "eng_Latn",
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

OPEN_GENERATION_PROMPTS = {
	"en": "The capital of France is:",
	"es": "La capital de Francia es:",
	"fr": "La capitale de la France est :",
	"de": "Die Hauptstadt von Frankreich ist:",
	"id": "Ibu kota Prancis adalah:",
	"pt": "A capital da Franca e:",
	"ru": "Столица Франции —",
	"zh": "法国的首都是：",
	"ja": "フランスの首都は",
	"ar": "عاصمة فرنسا هي:",
	"sw": "Mji mkuu wa Ufaransa ni:",
	"bn": "ফ্রান্সের রাজধানী হলো:",
}

def print_section(title: str):
	if rank == 0:
		print("\n" + "=" * 80, flush=True)
		print(title, flush=True)
		print("=" * 80, flush=True)


def print_kv(key: str, value):
	if rank == 0:
		print(f"{key:<40} {value}", flush=True)


def format_int(x):
	return f"{x:,}"

## --------------------------------- Eval Metrics ----------------------------------------

import numpy as np

CHOICE_TOKEN_IDS = None  # set after tokenizer is loaded


def safe_perplexity(loss: float) -> float:
	return math.exp(loss) if loss < 20 else float("inf")


def get_config_model(model):
	core_model = model.module if hasattr(model, "module") else model
	if hasattr(core_model, "base_model") and hasattr(core_model.base_model, "model"):
		return core_model.base_model.model
	return core_model


def tokenize_lm_text(text: str, tokenizer, max_length: int = 1024):
	ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
	return {
		"input_ids": ids,
		"attention_mask": [1] * len(ids),
		"labels": ids,
	}


def prepare_lm_validation_dataset(ds, max_samples: int | None):
	if "validation" not in ds:
		return None

	val_ds = ds["validation"]
	if max_samples is not None:
		val_ds = val_ds.select(range(min(max_samples, len(val_ds))))

	if "labels" not in val_ds.column_names:
		val_ds = val_ds.map(lambda ex: {"labels": ex["input_ids"]})

	return val_ds


def build_flores_lm_eval_sets(flores_sets, tokenizer, max_samples: int | None, max_length: int = 1024):
	eval_sets = {}
	eos_token = tokenizer.eos_token or ""

	for lang, data in flores_sets.items():
		src_texts = data["src_texts"]
		tgt_texts = data["tgt_texts"]
		if max_samples is not None:
			src_texts = src_texts[:max_samples]
			tgt_texts = tgt_texts[:max_samples]

		examples = []
		for src, tgt in zip(src_texts, tgt_texts):
			examples.append(tokenize_lm_text(f"en: {src}\n{lang}: {tgt}" + eos_token, tokenizer, max_length=max_length))
			examples.append(tokenize_lm_text(f"{lang}: {tgt}\nen: {src}" + eos_token, tokenizer, max_length=max_length))

		eval_sets[lang] = examples

	return eval_sets


@torch.no_grad()
def compute_lm_loss(model, examples, collator, device, batch_size: int = 1):
	model.eval()
	total_loss = 0.0
	total_tokens = 0

	for i in range(0, len(examples), batch_size):
		batch = collator(examples[i:i + batch_size])
		batch = {k: v.to(device) for k, v in batch.items()}
		outputs = model(**batch, use_cache=False)
		active_tokens = int((batch["labels"] != -100).sum().item())
		total_loss += float(outputs.loss.detach().cpu()) * active_tokens
		total_tokens += active_tokens

	loss = total_loss / max(total_tokens, 1)
	return loss, safe_perplexity(loss)


class LMEvalCallback(TrainerCallback):
	def __init__(self, overall_eval_set, per_lang_eval_sets, collator, batch_size=1):
		self.overall_eval_set = overall_eval_set
		self.per_lang_eval_sets = per_lang_eval_sets
		self.collator = collator
		self.batch_size = batch_size
		self._last_logged_step = None

	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if not state.is_world_process_zero:
			return

		if self._last_logged_step == state.global_step:
			return
		self._last_logged_step = state.global_step

		device = next(model.parameters()).device
		metrics = {}

		if self.overall_eval_set is not None:
			overall_examples = [self.overall_eval_set[i] for i in range(len(self.overall_eval_set))]
			loss, ppl = compute_lm_loss(model, overall_examples, self.collator, device, batch_size=self.batch_size)
			metrics["lm_validation/loss"] = loss
			metrics["lm_validation/perplexity"] = ppl

		lang_losses = []
		for lang, examples in self.per_lang_eval_sets.items():
			loss, ppl = compute_lm_loss(model, examples, self.collator, device, batch_size=self.batch_size)
			metrics[f"lm_validation_loss/{lang}"] = loss
			metrics[f"lm_validation_ppl/{lang}"] = ppl
			lang_losses.append(loss)

		if lang_losses:
			avg_loss = float(np.mean(lang_losses))
			metrics["lm_validation_loss/avg_lang"] = avg_loss
			metrics["lm_validation_ppl/avg_lang"] = safe_perplexity(avg_loss)

		print("\nLM validation metrics:")
		for k, v in metrics.items():
			print(f"{k}: {v:.4f}")

		if wandb.run is not None:
			wandb.log(metrics, step=state.global_step)






def make_mmlu_accuracy_fn(tokenizer):
	choice_ids = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]

	def compute_mmlu_accuracy(eval_pred):
		logits, labels = eval_pred
		preds = []
		golds = []

		for i in range(labels.shape[0]):
			pos = np.where(labels[i] != -100)[0]
			if len(pos) == 0:
				continue

			j = int(pos[0])
			if j == 0:
				continue

			gold_token = int(labels[i, j])
			if gold_token not in choice_ids:
				continue

			gold_idx = choice_ids.index(gold_token)
			pred_idx = int(np.argmax(logits[i, j - 1]))
			preds.append(pred_idx)
			golds.append(gold_idx)

		acc = float(np.mean(np.array(preds) == np.array(golds))) if golds else 0.0
		return {"accuracy": acc}

	return compute_mmlu_accuracy


def add_global_mmlu_avg(metrics_dict):
	lang_accs = []
	for lang in LANGS:
		key = f"eval_{lang}_accuracy"
		if key in metrics_dict:
			lang_accs.append(metrics_dict[key])

	if lang_accs:
		metrics_dict["eval_accuracy_avg"] = float(np.mean(lang_accs))

	return metrics_dict


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
def compute_flores_bleu(model, tokenizer, flores_sets, device, max_new_tokens=128, batch_size=4):
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

			tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
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
		metrics[f"flores_bleu/en_to_{lang}"] = score

	if metrics:
		metrics["flores_bleu/en_to_x_avg"] = sum(metrics.values()) / len(metrics)

	return metrics


###################################
# Hidden State Cosine Similarity 
#####################################

@torch.no_grad()
def compute_flores_bleu_to_english(model, tokenizer, flores_sets, device, max_new_tokens=128, batch_size=4):
	bleu = evaluate.load("sacrebleu")
	metrics = {}
	model.eval()

	for lang, data in flores_sets.items():
		src_texts = data["tgt_texts"]
		refs = data["src_texts"]
		preds = []

		for i in range(0, len(src_texts), batch_size):
			batch_src = src_texts[i:i + batch_size]
			prompts = [
				f"Translate the following sentence from {LANG_NAME_MAP[lang]} to English:\n\n{s}\n\nTranslation:"
				for s in batch_src
			]

			tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
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
		metrics[f"flores_bleu/{lang}_to_en"] = score

	if metrics:
		metrics["flores_bleu/x_to_en_avg"] = sum(metrics.values()) / len(metrics)

	return metrics


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
	batch_size=8,
	max_length=512,
):
	metrics = {}
	model.eval()

	config_model = get_config_model(model)
	n_layers = config_model.config.num_hidden_layers
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

	metrics["hidden_cosine_mid/avg"] = float(np.mean(all_mid)) if all_mid else 0.0
	metrics["hidden_cosine_last/avg"] = float(np.mean(all_last)) if all_last else 0.0

	return metrics

class FloresEvalCallback(TrainerCallback):
	def __init__(self, tokenizer, flores_sets):
		self.tokenizer = tokenizer
		self.flores_sets = flores_sets
		self._last_logged_step = None

	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if not state.is_world_process_zero:
			return

		if self._last_logged_step == state.global_step:
			return
		self._last_logged_step = state.global_step

		device = next(model.parameters()).device

		extra_metrics = {}
		extra_metrics.update(
			compute_flores_bleu(
				model=model,
				tokenizer=self.tokenizer,
				flores_sets=self.flores_sets,
				device=device,
				max_new_tokens=128,
				batch_size=4,
			)
		)
		extra_metrics.update(
			compute_flores_bleu_to_english(
				model=model,
				tokenizer=self.tokenizer,
				flores_sets=self.flores_sets,
				device=device,
				max_new_tokens=128,
				batch_size=4,
			)
		)
		extra_metrics.update(
			compute_flores_hidden_cosine(
				model=model,
				tokenizer=self.tokenizer,
				flores_sets=self.flores_sets,
				device=device,
				batch_size=8,
			)
		)
		extra_metrics.update(
			compute_translation_retrieval(
				model=model,
				tokenizer=self.tokenizer,
				flores_sets=self.flores_sets,
				device=device,
				batch_size=8,
			)
		)

		print("\nExtra eval metrics:")
		for k, v in extra_metrics.items():
			print(f"{k}: {v:.4f}")

		if wandb.run is not None:
			wandb.log(extra_metrics, step=state.global_step)
########################
# Translation Retrieval
########################

@torch.no_grad()
def get_sentence_embeddings(
	model,
	tokenizer,
	texts,
	layer_idx,
	device,
	batch_size=8,
	max_length=512,
):
	embs = []

	model.eval()

	for i in range(0, len(texts), batch_size):
		batch = texts[i:i + batch_size]

		tok = tokenizer(
			batch,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=max_length,
		).to(device)

		out = model(
			**tok,
			output_hidden_states=True,
			use_cache=False,
		)

		hidden = out.hidden_states[layer_idx]   # [B, T, H]
		pooled = mean_pool_hidden(hidden, tok["attention_mask"])  # [B, H]
		pooled = F.normalize(pooled, p=2, dim=-1)

		embs.append(pooled.detach().float().cpu())

	return torch.cat(embs, dim=0)  # [N, H]

@torch.no_grad()
def compute_translation_retrieval(
	model,
	tokenizer,
	flores_sets,
	device,
	batch_size=8,
	max_length=512,
):
	metrics = {}

	config_model = get_config_model(model)
	n_layers = config_model.config.num_hidden_layers
	mid_layer = n_layers // 2
	last_layer = n_layers

	def retrieval_acc(src_embs, tgt_embs):
		sim = src_embs @ tgt_embs.T
		pred = sim.argmax(dim=1)
		gold = torch.arange(sim.size(0))
		return (pred == gold).float().mean().item()

	all_mid = []
	all_last = []

	for lang, data in flores_sets.items():
		src_texts = data["src_texts"]
		tgt_texts = data["tgt_texts"]

		src_mid = get_sentence_embeddings(
			model, tokenizer, src_texts, mid_layer, device,
			batch_size=batch_size, max_length=max_length
		)
		tgt_mid = get_sentence_embeddings(
			model, tokenizer, tgt_texts, mid_layer, device,
			batch_size=batch_size, max_length=max_length
		)

		en_to_x_mid = retrieval_acc(src_mid, tgt_mid)
		x_to_en_mid = retrieval_acc(tgt_mid, src_mid)
		all_mid.append(0.5 * (en_to_x_mid + x_to_en_mid))

		src_last = get_sentence_embeddings(
			model, tokenizer, src_texts, last_layer, device,
			batch_size=batch_size, max_length=max_length
		)
		tgt_last = get_sentence_embeddings(
			model, tokenizer, tgt_texts, last_layer, device,
			batch_size=batch_size, max_length=max_length
		)

		en_to_x_last = retrieval_acc(src_last, tgt_last)
		x_to_en_last = retrieval_acc(tgt_last, src_last)
		all_last.append(0.5 * (en_to_x_last + x_to_en_last))

	metrics["retrieval_mid/avg"] = float(np.mean(all_mid)) if all_mid else 0.0
	metrics["retrieval_last/avg"] = float(np.mean(all_last)) if all_last else 0.0

	return metrics

class OpenGenerationCallback(TrainerCallback):
	def __init__(self, tokenizer, prompts, max_new_tokens=10):
		self.tokenizer = tokenizer
		self.prompts = prompts
		self.max_new_tokens = max_new_tokens
		self._last_logged_step = None

	def on_evaluate(self, args, state, control, model=None, **kwargs):
		if not state.is_world_process_zero:
			return

		if self._last_logged_step == state.global_step:
			return
		self._last_logged_step = state.global_step

		device = next(model.parameters()).device
		model.eval()
		rows = []

		for lang, prompt in self.prompts.items():
			tok = self.tokenizer(
				prompt,
				return_tensors="pt",
				truncation=True,
				max_length=256,
			).to(device)
			with torch.no_grad():
				gen = model.generate(
					**tok,
					max_new_tokens=self.max_new_tokens,
					do_sample=False,
					pad_token_id=self.tokenizer.pad_token_id,
					eos_token_id=self.tokenizer.eos_token_id,
				)

			gen_only = gen[:, tok["input_ids"].shape[1]:]
			completion = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
			full_text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
			rows.append([lang, prompt, completion, full_text])

		print("\nOpen generation probes:")
		for lang, prompt, completion, _ in rows:
			print(f"{lang}: {prompt} {completion}")

		if wandb.run is not None:
			wandb.log({
				"open_generation": wandb.Table(
					data=rows,
					columns=["lang", "prompt", "completion", "full_text"],
				)
			}, step=state.global_step)

@dataclass
class CausalLMPaddingCollator:
	tokenizer: Any
	label_pad_token_id: int = -100
	pad_to_multiple_of: int | None = 8

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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

LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh", "ja", "ar", "sw", "bn"]
MAX_EVAL_SAMPLES_PER_LANG = 285   # e.g. 200 for a cheap proxy eval


def preprocess_logits_for_metrics(logits, labels):
	if isinstance(logits, tuple):
		logits = logits[0]
	# logits: [B, seq_len, vocab_size]
	# Return only the 4 choice columns to save memory
	return logits[:, :, CHOICE_TOKEN_IDS]  # [B, seq_len, 4]


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


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--model_id", type=str, default=MODEL_ID)
	parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
	parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR)
	parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)

	parser.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)
	parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM)
	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--warmup_ratio", type=float, default=0.05)
	parser.add_argument("--weight_decay", type=float, default=0.01)

	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=200)
	parser.add_argument("--eval_steps", type=int, default=200)
	parser.add_argument("--save_total_limit", type=int, default=3)
	parser.add_argument("--dataloader_num_workers", type=int, default=2)

	parser.add_argument("--bf16", action="store_true", default=True)
	parser.add_argument("--no_bf16", action="store_true")
	parser.add_argument("--report_to", type=str, default="wandb")
	parser.add_argument("--lm_eval_max_samples", type=int, default=512)
	parser.add_argument("--per_lang_lm_eval_max_samples", type=int, default=64)
	parser.add_argument("--open_gen_max_new_tokens", type=int, default=10)

	return parser.parse_args()


def main():
	args = parse_args()
 
	rank = int(os.environ.get("RANK", 0))

 
	if rank != 0:
		os.environ["WANDB_DISABLED"] = "true"
		os.environ["WANDB_MODE"] = "disabled"

	args.model_id = resolve_model_id(args.model_id)
	if not args.dataset_path:
		args.dataset_path = default_dataset_path(args.model_id)
	if not args.ckpt_dir:
		args.ckpt_dir = default_ckpt_dir(args.model_id, args.learning_rate)
	if not args.output_dir:
		args.output_dir = default_output_dir(args.model_id, args.learning_rate)

	if args.no_bf16:
		args.bf16 = False

	os.makedirs(args.ckpt_dir, exist_ok=True)
	os.makedirs(args.output_dir, exist_ok=True)

	# ── 1. Tokenizer ───────────────────────────────────────────────────────────
	print(f"Loading tokenizer for {args.model_id}...")
	tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	gen_tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
	if gen_tokenizer.pad_token is None:
		gen_tokenizer.pad_token = gen_tokenizer.eos_token
	gen_tokenizer.padding_side = "left"
  
	global CHOICE_TOKEN_IDS
	CHOICE_TOKEN_IDS = [
		tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
		tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
	]

	compute_metrics = make_mmlu_accuracy_fn(tokenizer)
 
	flores_eval_sets = load_flores_parallel_subset(
		target_langs=["ar", "bn", "de", "es", "fr", "id", "ja", "pt", "ru", "sw", "zh"],
		split="dev",
		max_samples=64,
	)


	# ── 2. Load pretokenized dataset ───────────────────────────────────────────
	print(f"Loading pretokenized dataset from {args.dataset_path} ...")
	ds = load_from_disk(args.dataset_path)

	train_ds = ds["train"]
	eval_ds = load_global_mmlu_dev_eval_by_lang(LANGS, tokenizer)
	collator = CausalLMPaddingCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
	overall_lm_eval_ds = prepare_lm_validation_dataset(ds, args.lm_eval_max_samples)
	per_lang_lm_eval_sets = build_flores_lm_eval_sets(
		flores_sets=flores_eval_sets,
		tokenizer=tokenizer,
		max_samples=args.per_lang_lm_eval_max_samples,
		max_length=1024,
	)
 
	if rank == 0:
		print_section("EVAL DATASET")
		print_kv("Eval languages", ", ".join(LANGS))
		total_eval_examples = sum(len(ds_lang) for ds_lang in eval_ds.values())
		print_kv("Total eval examples", format_int(total_eval_examples))
		for lang, ds_lang in eval_ds.items():
			print_kv(f"{lang} eval examples", format_int(len(ds_lang)))
			print_kv(f"{lang} sample length", len(ds_lang[0]["input_ids"]))
 
	train_ds = train_ds.map(lambda ex: {"labels": ex["input_ids"]},  num_proc=64)

	num_chunks = len(train_ds)
	world_size = int(os.environ.get("WORLD_SIZE", 1))
	effective_batch = (
		args.per_device_train_batch_size
		* args.gradient_accumulation_steps
		* world_size
	)
	max_steps = int((num_chunks * args.num_epochs) // effective_batch)

	if rank == 0:
		print(f"Train chunks:      {num_chunks}")
		print(f"Effective batch:   {effective_batch}")
		print(f"Num epochs:        {args.num_epochs}")
		print(f"Max steps:         {max_steps}")
		if eval_ds is not None:
			total_eval_examples = sum(len(ds_lang) for ds_lang in eval_ds.values())
			print(f"Validation examples: {total_eval_examples}")

	# ── 3. W&B init ────────────────────────────────────────────────────────────
	if args.report_to == "wandb" and rank == 0:
		wandb.init(
			project=WANDB_PROJECT,
			config={
				"model": args.model_id,
				"format": "grouped-paired",
				"languages": REQ_LANGS,
				"dataset_path": args.dataset_path,
				"train_chunks": num_chunks,
				"validation_examples": sum(len(ds_lang) for ds_lang in eval_ds.values()) if eval_ds is not None else 0,
				"num_epochs": args.num_epochs,
				"max_steps": max_steps,
				"lora_r": LORA_R,
				"lora_alpha": LORA_ALPHA,
				"lr": args.learning_rate,
				"batch_size": args.per_device_train_batch_size,
				"grad_accum": args.gradient_accumulation_steps,
				"quantization": "none",
				"grad_checkpointing": False,
				"bf16": args.bf16,
				"lm_eval_max_samples": args.lm_eval_max_samples,
				"per_lang_lm_eval_max_samples": args.per_lang_lm_eval_max_samples,
				"open_gen_max_new_tokens": args.open_gen_max_new_tokens,
			},
		)

	# ── 4. Load base model (NO quantization) ──────────────────────────────────
	if "model" in globals() and isinstance(globals()["model"], PeftModel):
		del globals()["model"]
		torch.cuda.empty_cache()

	if rank == 0:
		print(f"\nLoading {args.model_id} without quantization...")
  
	base_model = AutoModelForCausalLM.from_pretrained(
		args.model_id,
		dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
	)
	base_model.config.pad_token_id = tokenizer.pad_token_id
 

	#base_model.gradient_checkpointing_enable()
	#base_model.config.use_cache = False

	# ── 5. LoRA config ─────────────────────────────────────────────────────────
	peft_config = LoraConfig(
		r=LORA_R,
		lora_alpha=LORA_ALPHA,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
		lora_dropout=0.05,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
	)

	model = get_peft_model(base_model, peft_config)
	model.print_trainable_parameters()
 
	if rank == 0:
		print("Requested attention backend:", "flash_attention_2")
		print("Model config attn_implementation:", getattr(model.config, "_attn_implementation", None))

	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	total = sum(p.numel() for p in model.parameters())

	if args.report_to == "wandb" and rank == 0:
		wandb.log({
			"model/trainable_params": trainable,
			"model/trainable_pct": 100 * trainable / total,
		})

	# ── 6. Training arguments ──────────────────────────────────────────────────
	training_args = TrainingArguments(
		output_dir=args.ckpt_dir,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		max_steps=max_steps,
		logging_steps=args.logging_steps,
		bf16=args.bf16,
		optim="adamw_torch_fused",
		lr_scheduler_type="cosine",
		warmup_steps=200,
		weight_decay=args.weight_decay,
		save_strategy="steps",
		save_steps=args.save_steps,
		eval_strategy="steps",
		eval_steps=args.eval_steps,
		per_device_eval_batch_size=1,
		eval_accumulation_steps=1,
		save_total_limit=args.save_total_limit,
		load_best_model_at_end=False,
		dataloader_num_workers=args.dataloader_num_workers,
  		ddp_find_unused_parameters=False,
		report_to=["wandb"] if (args.report_to == "wandb" and rank == 0) else [],
		remove_unused_columns=False,
	)

	# ── 7. Trainer ─────────────────────────────────────────────────────────────
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
  		eval_dataset=eval_ds,
		data_collator=collator,
		compute_metrics=compute_metrics,
		preprocess_logits_for_metrics=preprocess_logits_for_metrics,
	)
 
	trainer.add_callback(
		FloresEvalCallback(
			tokenizer=gen_tokenizer,
			flores_sets=flores_eval_sets,
		)
	)
	trainer.add_callback(
		LMEvalCallback(
			overall_eval_set=overall_lm_eval_ds,
			per_lang_eval_sets=per_lang_lm_eval_sets,
			collator=collator,
			batch_size=1,
		)
	)
	trainer.add_callback(
		OpenGenerationCallback(
			tokenizer=gen_tokenizer,
			prompts=OPEN_GENERATION_PROMPTS,
			max_new_tokens=args.open_gen_max_new_tokens,
		)
	)

	if rank == 0:
		print(f"\nStarting OLMo-2 run:")
		print(f"  Format: grouped-paired | LoRA r={LORA_R}, alpha={LORA_ALPHA} | no quantization")

		print(f"  {num_chunks} chunks | max_steps={max_steps}")
		print(f"  Checkpointing every {args.save_steps} steps → {args.ckpt_dir}")

	trainer.train()
 
	if eval_ds is not None:
		eval_metrics = trainer.evaluate()
		eval_metrics = add_global_mmlu_avg(eval_metrics)

		if rank == 0:
			print(eval_metrics)
			if wandb.run is not None:
				wandb.log(eval_metrics, step=trainer.state.global_step)

	# ── 8. Save final adapter + tokenizer ─────────────────────────────────────


	if rank == 0:
	 
		lora_out = args.output_dir
		model.save_pretrained(lora_out)
		tokenizer.save_pretrained(lora_out)
		
		merged_out = os.path.join(args.output_dir, "merged")
		os.makedirs(merged_out, exist_ok=True)
		print("Reloading base model for merge...")

		base_model_for_merge = AutoModelForCausalLM.from_pretrained(
			args.model_id,
			torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
		)

		merged_model = PeftModel.from_pretrained(base_model_for_merge, lora_out)
		merged_model = merged_model.merge_and_unload()

		merged_model.save_pretrained(merged_out)
		tokenizer.save_pretrained(merged_out)

		print(f"Saved merged model to {merged_out}")

	if args.report_to == "wandb" and rank == 0:
		if len(trainer.state.log_history) > 0:
			wandb.summary["final_train_loss"] = trainer.state.log_history[-1].get("train_loss", None)
		wandb.finish()

	print(f"\nDone. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
	main()