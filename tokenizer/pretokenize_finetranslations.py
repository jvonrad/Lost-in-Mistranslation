#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pretokenize a balanced, interleaved FineTranslations corpus for multilingual CPT.

Design goals:
- balance languages by ENGLISH token budget per target language
- interleave languages so training data is not grouped by language
- use similar tokenization logic to the TED script:
	* create text chunks first
	* tokenize only after chunk creation
	* truncation=True, max_length=MAX_LENGTH
- save DatasetDict with:
	* train
	* validation

Important:
- FineTranslations does not expose a separate English subset for this use case.
  Each non-English subset already contains the English side as translated_text.
- Therefore, we build bilingual paired examples from each non-English subset:
	  en: ...
	  xx: ...
- We balance by translated_token_count (English-side tokens).

Example launch:
python training/pretokenize_finetranslations_balanced.py \
  --out_path /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/finetranslations_balanced_11langs_512 \
  --english_token_budget_per_lang 699019122
"""

import os
import random
import argparse
from typing import Dict, List, Iterator, Optional
from collections import deque

from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMo-2-1124-7B"
OUT_PATH = "/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/finetranslations_balanced_11langs_512"

# User-facing language list includes English, but English is implicit in each pair.
REQ_LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

# FineTranslations subset names from subsets.csv
LANG_TO_SUBSET = {
	"de": "deu_Latn",
	"id": "ind_Latn",
	"pt": "por_Latn",
	"ar": "arb_Arab",
	"bn": "ben_Beng",
	"sw": "swh_Latn",
	"es": "spa_Latn",
	"ru": "rus_Cyrl",
	"fr": "fra_Latn",
	"ja": "jpn_Jpan",
	"zh": "cmn_Hani",
}

# English token counts from subsets.csv
SUBSET_ENGLISH_TOKENS = {
	"deu_Latn": 39514954895,
	"ind_Latn": 44054775794,
	"por_Latn": 41489302074,
	"arb_Arab": 37510832376,
	"ben_Beng": 8620839206,
	"swh_Latn": 699019122,
	"spa_Latn": 41947981388,
	"rus_Cyrl": 39146692857,
	"fra_Latn": 38777297151,
	"jpn_Jpan": 45055472990,
	"cmn_Hani": 49772590084,
}

LANG_ORDER = ["de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

CHUNK_TOKENS = 1024
MAX_LENGTH = 1024
MAX_SIDE_TOKENS = 450


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_id", type=str, default=MODEL_ID)
	parser.add_argument("--out_path", type=str, default=OUT_PATH)
	parser.add_argument("--chunk_tokens", type=int, default=CHUNK_TOKENS)
	parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
	parser.add_argument("--val_ratio", type=float, default=0.005)
	parser.add_argument("--seed", type=int, default=42)

	# Default = fully balanced at the smallest available language (Swahili)
	parser.add_argument("--english_token_budget_per_lang", type=int, default=699_019_122) # 699_019_122

	# Optional safety cap for very large preprocessing runs
	parser.add_argument("--max_examples_per_lang", type=int, default=None)

	# Shuffle buffer for per-language streams before interleaving
	parser.add_argument("--shuffle_buffer_size", type=int, default=1000)

	return parser.parse_args()


def format_int(x):
	return f"{x:,}"


def print_kv(k, v):
	print(f"{k:<40} {v}")


def print_section(title):
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def nonempty_str(x) -> bool:
	return isinstance(x, str) and len(x.strip()) > 0


def truncate_text_to_tokens(tokenizer, text, max_tokens):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def format_pair(
    tokenizer,
    en_text: str,
    tgt_lang: str,
    tgt_text: str,
    eos_token: str,
    rng: random.Random,
    max_side_tokens: int = MAX_SIDE_TOKENS,
) -> str:
    en_text = truncate_text_to_tokens(tokenizer, en_text, max_side_tokens)
    tgt_text = truncate_text_to_tokens(tokenizer, tgt_text, max_side_tokens)

    if rng.random() < 0.5:
        return f"en: {en_text}\n{tgt_lang}: {tgt_text}{eos_token}"
    else:
        return f"{tgt_lang}: {tgt_text}\nen: {en_text}{eos_token}"


def iter_subset_examples(
	subset_name: str,
	tgt_lang: str,
	english_token_budget: int,
	max_examples: Optional[int],
	shuffle_buffer_size: int,
	seed: int,
) -> Iterator[Dict[str, str]]:
	"""
	Stream one FineTranslations subset and stop when we reach the target
	ENGLISH token budget, using translated_token_count as the budget meter.
	"""
	ds = load_dataset(
		"HuggingFaceFW/finetranslations",
		subset_name,
		split="train",
		streaming=True,
	)

	# Streaming shuffle helps avoid preserving the raw file order within each language.
	ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

	total_en_tokens = 0
	n_examples = 0

	for ex in ds:
		# Dataset fields from FineTranslations dataset card:
		# og_full_text = source-language text
		# translated_text = English translation
		# translated_token_count = English token count
		tgt_text = ex.get("og_full_text")
		en_text = ex.get("translated_text")
		en_tok = ex.get("translated_token_count")

		if not nonempty_str(tgt_text) or not nonempty_str(en_text):
			continue
		if not isinstance(en_tok, int) or en_tok <= 0:
			continue

		yield {
			"lang": tgt_lang,
			"en_text": en_text.strip(),
			"tgt_text": tgt_text.strip(),
			"en_token_count": en_tok,
		}

		total_en_tokens += en_tok
		n_examples += 1

		if max_examples is not None and n_examples >= max_examples:
			break
		if total_en_tokens >= english_token_budget:
			break


def collect_interleaved_chunks(
	tokenizer,
	english_token_budget_per_lang: int,
	chunk_tokens: int,
	max_examples_per_lang: Optional[int],
	shuffle_buffer_size: int,
	seed: int,
) -> List[Dict[str, str]]:
	"""
	Build chunks by interleaving languages round-robin.
	Each language has its own buffer so chunks are internally language-consistent,
	but the emitted chunks are interleaved across languages.
	"""
	per_lang_iter = {
		lang: iter_subset_examples(
			subset_name=LANG_TO_SUBSET[lang],
			tgt_lang=lang,
			english_token_budget=english_token_budget_per_lang,
			max_examples=max_examples_per_lang,
			shuffle_buffer_size=shuffle_buffer_size,
			seed=seed + i,
		)
		for i, lang in enumerate(LANG_ORDER)
	}

	per_lang_buffers = {lang: {"text": "", "tok": 0} for lang in LANG_ORDER}
	chunks = []
	active_langs = deque(LANG_ORDER)

	def flush_lang(lang: str):
		buf = per_lang_buffers[lang]
		if buf["text"]:
			chunks.append({"text": buf["text"], "lang": lang})
			buf["text"] = ""
			buf["tok"] = 0

	step = 0
 
	while active_langs:
		lang = active_langs.popleft()
		it = per_lang_iter[lang]
		
		step += 1
		if step % 50000 == 0:
			print(f"rounds={step:,} chunks={len(chunks):,}")

		try:
			ex = next(it)
		except StopIteration:
			flush_lang(lang)
			continue

		rng = random.Random(seed)

		seg = format_pair(
			tokenizer=tokenizer,
			en_text=ex["en_text"],
			tgt_lang=lang,
			tgt_text=ex["tgt_text"],
			eos_token=tokenizer.eos_token,
			rng=rng,
		)

		# Same rough heuristic style as your TED script: estimate before tokenization.
		seg_tok = len(seg) // 4

		if seg_tok >= chunk_tokens:
			flush_lang(lang)
			chunks.append({"text": seg, "lang": lang})
		else:
			buf = per_lang_buffers[lang]
			if buf["tok"] + seg_tok > chunk_tokens:
				flush_lang(lang)
				per_lang_buffers[lang]["text"] = seg
				per_lang_buffers[lang]["tok"] = seg_tok
			else:
				per_lang_buffers[lang]["text"] += seg
				per_lang_buffers[lang]["tok"] += seg_tok

		# Put language back into the round-robin queue
		active_langs.append(lang)

	# Final flush
	for lang in LANG_ORDER:
		flush_lang(lang)

	return chunks


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
	def tokenize(element):
		return tokenizer(
			element["text"],
			truncation=True,
			max_length=max_length,
		)

	return dataset.map(tokenize, remove_columns=["text", "lang"], num_proc=8)


def print_stats(ds: Dataset, name: str):
	lengths = [len(x) for x in ds["input_ids"]] if len(ds) > 0 else []
	total_tokens = sum(lengths)
	mean_len = total_tokens / len(lengths) if lengths else 0.0

	print(f"\n===== {name.upper()} =====")
	print(f"num_examples: {len(ds):,}")
	print(f"total_tokens: {total_tokens:,}")
	print(f"mean_len:     {mean_len:.2f}")
	if lengths:
		print(f"min_len:      {min(lengths)}")
		print(f"max_len:      {max(lengths)}")


def main():
	args = parse_args()
	os.makedirs(args.out_path, exist_ok=True)

	print(f"Loading tokenizer for {args.model_id} ...")
	tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	print_section("SUBSET PLAN")
	print_kv("Languages", ", ".join(REQ_LANGS))
	print_kv("Target langs actually sampled", ", ".join(LANG_ORDER))
	print_kv("English token budget / lang", format_int(args.english_token_budget_per_lang))
	print_kv("Total target languages", len(LANG_ORDER))
	print_kv(
		"Total nominal English tokens",
		format_int(args.english_token_budget_per_lang * len(LANG_ORDER)),
	)
	for lang in LANG_ORDER:
		subset = LANG_TO_SUBSET[lang]
		print_kv(
			f"{lang} -> {subset}",
			f"available_en_tokens={format_int(SUBSET_ENGLISH_TOKENS[subset])}",
		)

	print_section("COLLECTING INTERLEAVED CHUNKS")
	chunks = collect_interleaved_chunks(
		tokenizer=tokenizer,
		english_token_budget_per_lang=args.english_token_budget_per_lang,
		chunk_tokens=args.chunk_tokens,
		max_examples_per_lang=args.max_examples_per_lang,
		shuffle_buffer_size=args.shuffle_buffer_size,
		seed=args.seed,
	)
	print_kv("Collected chunks", format_int(len(chunks)))

	if len(chunks) < 2:
		raise ValueError("Not enough chunks collected to create train/validation split.")

	rng = random.Random(args.seed)
	rng.shuffle(chunks)

	n_val = max(1, int(len(chunks) * args.val_ratio))
	n_val = min(n_val, len(chunks) - 1)

	val_chunks = chunks[:n_val]
	train_chunks = chunks[n_val:]

	raw = DatasetDict({
		"train": Dataset.from_list(train_chunks),
		"validation": Dataset.from_list(val_chunks),
	})

	print("\nTokenizing train split...")
	train_tok = tokenize_dataset(raw["train"], tokenizer, args.max_length)

	print("Tokenizing validation split...")
	val_tok = tokenize_dataset(raw["validation"], tokenizer, args.max_length)

	final_ds = DatasetDict({
		"train": train_tok,
		"validation": val_tok,
	})

	print_stats(final_ds["train"], "train")
	print_stats(final_ds["validation"], "validation")

	print(f"\nSaving dataset to:\n{args.out_path}")
	final_ds.save_to_disk(args.out_path)

	print("\nDone.")


if __name__ == "__main__":
	main()