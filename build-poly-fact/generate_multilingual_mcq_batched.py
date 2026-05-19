#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python generate_multilingual_mcq_batched.py \
  --seed_facts_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/seed_facts_balanced.jsonl \
  --output_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text_batched.jsonl \
  --max_facts 20000 \
  --excluded_properties P35 P6 P122 P85 \
  --fact_batch_size 16
"""

import json
import time
import random
import argparse
from typing import Dict, List, Optional, Tuple, Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

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


def is_missing_label(x) -> bool:
	return x is None or not str(x).strip()


def normalize_text(s: str) -> str:
	return " ".join(str(s).strip().lower().split())


def parse_json_object(text: str) -> Optional[Dict]:
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		return None
	try:
		return json.loads(text[start:end + 1])
	except Exception:
		return None


def validate_bundle_item(obj: Dict) -> bool:
	required = ["question", "options", "answer_text"]
	if not all(k in obj for k in required):
		return False

	if not isinstance(obj["question"], str) or not obj["question"].strip():
		return False

	if not isinstance(obj["options"], list) or len(obj["options"]) != 4:
		return False
	if not all(isinstance(x, str) and x.strip() for x in obj["options"]):
		return False
	if len(set(obj["options"])) != 4:
		return False

	if not isinstance(obj["answer_text"], str) or not obj["answer_text"].strip():
		return False

	if obj["answer_text"] not in obj["options"]:
		return False

	return True


def estimate_tokens(text: str) -> int:
	return max(1, len(text) // 4)


def batched(iterable: Iterable, n: int):
	chunk = []
	for x in iterable:
		chunk.append(x)
		if len(chunk) == n:
			yield chunk
			chunk = []
	if chunk:
		yield chunk
		
		
def get_localized_subject_and_relation(
	translation_cache: Dict[Tuple[str, str], str],
	fact: Dict,
	lang: str,
) -> Tuple[str, str]:
	subject_labels = fact.get("subject_labels", {})
	property_labels = fact.get("property_labels", {})

	subject = subject_labels.get(lang)
	if is_missing_label(subject):
		fallback = subject_labels.get("en") or fact["subject"]
		subject = translation_cache.get((lang, fallback), fallback)

	relation = property_labels.get(lang)
	if is_missing_label(relation):
		fallback = property_labels.get("en") or fact["relation"]
		relation = translation_cache.get((lang, fallback), fallback)

	return subject, relation


def build_translate_prompt(text: str, lang: str, relation: Optional[str] = None) -> str:
	lang_name = LANG_TO_NAME[lang]
	relation_hint = (
		f"\nThe label refers to a Wikidata entity used as the object of the relation: {relation}."
		if relation else ""
	)
	return f"""
Translate the following Wikidata entity label into {lang_name}.{relation_hint}

Rules:
- Return only the translated entity label as plain text.
- Do not add explanation.
- Preserve the exact entity meaning.
- This is an entity name / canonical encyclopedia label, not a sentence.
- If the label is already commonly used unchanged in {lang_name}, keep it unchanged.
- If the label is ambiguous, choose the meaning consistent with the relation context.
- Do not output quotes.
- Do not output multiple alternatives.

Label:
{text}
""".strip()


def build_translation_chat_prompt(
	tokenizer: AutoTokenizer,
	text_prompt: str,
) -> str:
	messages = [
		{
			"role": "system",
			"content": "You are a translation engine. Output only the translated text.",
		},
		{
			"role": "user",
			"content": text_prompt,
		},
	]
	return tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
		#enable_thinking=False,
	)


def batch_translate_missing_labels(
	llm: LLM,
	tokenizer: AutoTokenizer,
	translation_sampling_params: SamplingParams,
	translation_cache: Dict[Tuple[str, str], str],
	requests: List[Tuple[str, str, Optional[str]]],  # (lang, text, relation)
):
	"""
	Batch-translate all uncached (lang, text) pairs.
	"""
	unique_requests = []
	seen = set()

	for lang, text, relation in requests:
		key = (lang, text)
		if key in translation_cache:
			continue
		if key in seen:
			continue
		seen.add(key)
		unique_requests.append((lang, text, relation))

	if not unique_requests:
		return

	prompts = [
		build_translation_chat_prompt(
			tokenizer,
			build_translate_prompt(text=text, lang=lang, relation=relation),
		)
		for lang, text, relation in unique_requests
	]

	t0 = time.time()
	outputs = llm.generate(prompts, translation_sampling_params)
	print(
		f"  [TRANSLATION BATCH DONE] {len(prompts)} requests in {time.time() - t0:.2f}s",
		flush=True,
	)

	for (lang, text, _relation), out in zip(unique_requests, outputs):
		translated = out.outputs[0].text.strip()
		if not translated:
			translated = text
		translation_cache[(lang, text)] = translated


def get_localized_answer_and_distractors(
	translation_cache: Dict[Tuple[str, str], str],
	fact: Dict,
	lang: str,
) -> Tuple[str, List[str]]:
	obj_labels = fact.get("object_labels", {})
	distractor_labels = fact.get("distractor_labels", {})
	distractors_fallback = fact.get("distractors", [])

	answer = obj_labels.get(lang)
	if is_missing_label(answer):
		fallback = obj_labels.get("en") or fact["object"]
		answer = translation_cache.get((lang, fallback), fallback)

	localized_distractors = []
	for i, (d_id, lbls) in enumerate(distractor_labels.items()):
		d_text = lbls.get(lang)
		if is_missing_label(d_text):
			fallback = lbls.get("en")
			if is_missing_label(fallback) and i < len(distractors_fallback):
				fallback = distractors_fallback[i]
			if is_missing_label(fallback):
				fallback = d_id
			d_text = translation_cache.get((lang, fallback), fallback)
		localized_distractors.append(d_text)

	return answer, localized_distractors


def build_mcq_prompt(
	fact: Dict,
	lang: str,
	subject: str,
	relation: str,
	answer: str,
	distractors: List[str],
) -> Tuple[str, List[str]]:
	lang_name = LANG_TO_NAME[lang]

	assert len(distractors) == 3, f"Need exactly 3 distractors for {fact['fact_id']} / {lang}"

	options = [answer] + distractors
	random.shuffle(options)

	prompt = f"""
Generate a factual multiple-choice question in {lang_name}.

Rules:
- Return ONLY valid JSON.
- Do NOT add markdown.
- Do NOT output reasoning or analysis.
- The question MUST be in {lang_name}.
- Use the provided options EXACTLY as given.
- Do NOT translate, paraphrase, or modify any option.
- Do NOT introduce new option strings.
- The answer_text MUST be exactly one of the provided options.
- Exactly one option is correct.

Return JSON with exactly these keys:
{{
  "question": "...",
  "answer_text": "..."
}}

Grounded fact:
subject: {subject}
relation: {relation}
correct_object: {answer}

Candidate options:
{options}

IMPORTANT:
- The correct answer is exactly: {answer}
- answer_text must exactly equal the correct option string.
""".strip()

	return prompt, options

def build_mcq_chat_prompt(
	tokenizer: AutoTokenizer,
	prompt: str,
) -> str:
	messages = [
		{
			"role": "system",
			"content": (
				"You are a structured data generator. "
				"Never output thinking, reasoning, analysis, or markdown. "
				"Output exactly one valid JSON object."
			),
		},
		{"role": "user", "content": prompt},
		{"role": "assistant", "content": '{\n  "question": "'},
	]

	return tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=False,
		continue_final_message=True,
		enable_thinking=False,
	)


def resolve_answer_text(answer_text: str, options: List[str], gold_answer: str) -> Optional[str]:
	if answer_text in options:
		return answer_text

	norm_to_option = {normalize_text(opt): opt for opt in options}
	norm_answer = normalize_text(answer_text)

	if norm_answer in norm_to_option:
		return norm_to_option[norm_answer]

	if normalize_text(gold_answer) in norm_to_option:
		return norm_to_option[normalize_text(gold_answer)]

	return None


def collect_missing_translation_requests(
    facts: List[Dict],
    translation_cache: Dict[Tuple[str, str], str],
) -> List[Tuple[str, str, Optional[str]]]:
    requests = []

    for fact in facts:
        subject_labels = fact.get("subject_labels", {})
        property_labels = fact.get("property_labels", {})
        obj_labels = fact.get("object_labels", {})
        distractor_labels = fact.get("distractor_labels", {})
        distractors_fallback = fact.get("distractors", [])

        for lang in LANGS:
            # subject
            subject = subject_labels.get(lang)
            if is_missing_label(subject):
                fallback = subject_labels.get("en") or fact["subject"]
                if (lang, fallback) not in translation_cache:
                    requests.append((lang, fallback, fact.get("relation")))

            # relation/property
            relation = property_labels.get(lang)
            if is_missing_label(relation):
                fallback = property_labels.get("en") or fact["relation"]
                if (lang, fallback) not in translation_cache:
                    requests.append((lang, fallback, fact.get("relation")))

            # answer/object
            answer = obj_labels.get(lang)
            if is_missing_label(answer):
                fallback = obj_labels.get("en") or fact["object"]
                if (lang, fallback) not in translation_cache:
                    requests.append((lang, fallback, fact.get("relation")))

        for _lang in LANGS:
            for i, (d_id, lbls) in enumerate(distractor_labels.items()):
                d_text = lbls.get(_lang)
                if is_missing_label(d_text):
                    fallback = lbls.get("en")
                    if is_missing_label(fallback) and i < len(distractors_fallback):
                        fallback = distractors_fallback[i]
                    if is_missing_label(fallback):
                        fallback = d_id
                    if (_lang, fallback) not in translation_cache:
                        requests.append((_lang, fallback, fact.get("relation")))

    return requests

def generate_batch_for_facts(
	llm: LLM,
	tokenizer: AutoTokenizer,
	mcq_sampling_params: SamplingParams,
	translation_sampling_params: SamplingParams,
	translation_cache: Dict[Tuple[str, str], str],
	facts: List[Dict],
	debug_prompt_sample: bool = False,
) -> Tuple[List[Dict], int]:
	"""
	Returns:
	  successful_bundles: list of completed bundle dicts
	  failed_count: number of facts dropped due to at least one language failing
	"""

	# ------------------------------------------------------------------
	# 1) Batch translation prefill for all missing labels in this fact batch
	# ------------------------------------------------------------------
	translation_requests = collect_missing_translation_requests(facts, translation_cache)
	if translation_requests:
		batch_translate_missing_labels(
			llm=llm,
			tokenizer=tokenizer,
			translation_sampling_params=translation_sampling_params,
			translation_cache=translation_cache,
			requests=translation_requests,
		)

	# ------------------------------------------------------------------
	# 2) Build all MCQ prompts for all (fact, lang) pairs in one flat batch
	# ------------------------------------------------------------------
	prompts = []
	meta = []  # (fact_idx, lang, gold_answer, options)

	for fact_idx, fact in enumerate(facts):
		for lang in LANGS:
			answer, distractors = get_localized_answer_and_distractors(
				translation_cache=translation_cache,
				fact=fact,
				lang=lang,
			)

			subject, relation = get_localized_subject_and_relation(
				translation_cache=translation_cache,
				fact=fact,
				lang=lang,
			)

			prompt, options = build_mcq_prompt(
				fact=fact,
				lang=lang,
				subject=subject,
				relation=relation,
				answer=answer,
				distractors=distractors,
			)

			chat_prompt = build_mcq_chat_prompt(tokenizer=tokenizer, prompt=prompt)

			if debug_prompt_sample and fact_idx == 0 and lang == "en":
				print("\n[CHAT PROMPT SAMPLE]")
				print(chat_prompt[:900])
				print()

			prompts.append(chat_prompt)
			meta.append((fact_idx, lang, answer, options))

	# ------------------------------------------------------------------
	# 3) Batched generation
	# ------------------------------------------------------------------
	t0 = time.time()
	outputs = llm.generate(prompts, mcq_sampling_params)
	print(
		f"  [MCQ BATCH DONE] {len(prompts)} prompts "
		f"({len(facts)} facts x {len(LANGS)} langs) in {time.time() - t0:.2f}s",
		flush=True,
	)

	# ------------------------------------------------------------------
	# 4) Parse outputs fact-by-fact; if one language fails, drop that fact only
	# ------------------------------------------------------------------
	temp_lang_results: Dict[int, Dict[str, Dict]] = {i: {} for i in range(len(facts))}
	fact_failed = set()

	for out, (fact_idx, lang, gold_answer, options) in zip(outputs, meta):
		raw = '{\n  "question": "' + out.outputs[0].text
		obj = parse_json_object(raw)

		if obj is None:
			print(
				f"  [FAIL] fact={facts[fact_idx]['fact_id']} lang={lang} parse_json failed",
				flush=True,
			)
			print(f"  raw[:400]={raw[:400]!r}", flush=True)
			fact_failed.add(fact_idx)
			continue

		resolved_answer = resolve_answer_text(obj.get("answer_text", ""), options, gold_answer)
		if resolved_answer is None:
			print(
				f"  [FAIL] fact={facts[fact_idx]['fact_id']} lang={lang} answer_text not in options",
				flush=True,
			)
			print(f"  answer_text={obj.get('answer_text')!r}", flush=True)
			print(f"  options={options!r}", flush=True)
			fact_failed.add(fact_idx)
			continue

		obj["answer_text"] = resolved_answer
		obj["options"] = options

		if not validate_bundle_item(obj):
			print(
				f"  [FAIL] fact={facts[fact_idx]['fact_id']} lang={lang} invalid bundle item",
				flush=True,
			)
			print(f"  raw[:400]={raw[:400]!r}", flush=True)
			fact_failed.add(fact_idx)
			continue

		fact = facts[fact_idx]
		obj["_expected_gold_answer"] = gold_answer
		obj["_correct_object_id"] = fact["object_id"]
		obj["_candidate_options"] = options
		obj["_accepted_answers"] = fact.get("object_labels", {})
		temp_lang_results[fact_idx][lang] = obj

	successful_bundles = []
	failed_count = 0

	for fact_idx, fact in enumerate(facts):
		if fact_idx in fact_failed:
			failed_count += 1
			continue

		# Ensure all languages succeeded
		if len(temp_lang_results[fact_idx]) != len(LANGS):
			print(
				f"  [FAIL] fact={fact['fact_id']} incomplete langs="
				f"{len(temp_lang_results[fact_idx])}/{len(LANGS)}",
				flush=True,
			)
			failed_count += 1
			continue

		bundle = {
			"fact_id": fact["fact_id"],
			"subject": fact["subject"],
			"relation": fact["relation"],
			"object": fact["object"],
			"object_id": fact["object_id"],
			"subject_id": fact["subject_id"],
			"property_id": fact["property_id"],
			"langs": temp_lang_results[fact_idx],
		}
		successful_bundles.append(bundle)

	return successful_bundles, failed_count


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--seed_facts_jsonl", required=True)
	ap.add_argument("--output_jsonl", required=True)
	ap.add_argument("--target_total_tokens", type=int, default=200_000_000)
	ap.add_argument("--max_facts", type=int, default=None)
	ap.add_argument("--dtype", type=str, default="bfloat16")
	ap.add_argument("--max_tokens", type=int, default=192)
	ap.add_argument("--tensor_parallel_size", type=int, default=1)
	ap.add_argument("--fact_batch_size", type=int, default=8)
	ap.add_argument("--debug_prompt_sample", action="store_true")
	ap.add_argument(
		"--allowed_properties",
		nargs="*",
		default=None,
		help="Only generate for these property IDs."
	)
	ap.add_argument(
		"--excluded_properties",
		nargs="*",
		default=None,
		help="Skip these property IDs."
	)
 
	args = ap.parse_args()

	random.seed(42)

	#model_id = "Qwen/Qwen3.5-35B-A3B"
	model_id = "google/gemma-3-27b-it"

	print("Loading tokenizer...", flush=True)
	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

	print("Loading vLLM model...", flush=True)
	llm = LLM(
		model=model_id,
		dtype=args.dtype,
		tensor_parallel_size=args.tensor_parallel_size,
		enforce_eager=True,
		max_model_len=4096,
	)

	mcq_sampling_params = SamplingParams(
		temperature=0.1,
		top_p=0.9,
		max_tokens=args.max_tokens,
		stop=["</think>", "```", "<|im_end|>"],
	)

	translation_sampling_params = SamplingParams(
		temperature=0.0,
		top_p=1.0,
		max_tokens=32,
		stop=["</think>", "```", "<|im_end|>", "\n"],
	)

	translation_cache: Dict[Tuple[str, str], str] = {}

	total_tokens = 0
	n_bundles = 0
	n_failed = 0
	start_time = time.time()

	# ------------------------------------------------------------
	# Load facts lazily enough, but batch them before generation
	# ------------------------------------------------------------
	allowed_properties = set(args.allowed_properties) if args.allowed_properties else None
	excluded_properties = set(args.excluded_properties or [])
	def fact_stream():
		with open(args.seed_facts_jsonl, "r", encoding="utf-8") as f_in:
			kept = 0
			for line in f_in:
				fact = json.loads(line)

				pid = fact.get("property_id")
				if allowed_properties is not None and pid not in allowed_properties:
					continue
				if pid in excluded_properties:
					continue

				if args.max_facts is not None and kept >= args.max_facts:
					break

				kept += 1
				yield fact

	with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
		for batch_idx, fact_batch in enumerate(batched(fact_stream(), args.fact_batch_size)):
			print(
				f"\n[FACT BATCH {batch_idx}] size={len(fact_batch)} "
				f"ids={[x['fact_id'] for x in fact_batch[:3]]}"
				f"{'...' if len(fact_batch) > 3 else ''}",
				flush=True,
			)

			successful_bundles, failed_in_batch = generate_batch_for_facts(
				llm=llm,
				tokenizer=tokenizer,
				mcq_sampling_params=mcq_sampling_params,
				translation_sampling_params=translation_sampling_params,
				translation_cache=translation_cache,
				facts=fact_batch,
				debug_prompt_sample=args.debug_prompt_sample and batch_idx == 0,
			)

			n_failed += failed_in_batch

			for bundle in successful_bundles:
				f_out.write(json.dumps(bundle, ensure_ascii=False) + "\n")
				bundle_token_est = estimate_tokens(json.dumps(bundle["langs"], ensure_ascii=False))
				total_tokens += bundle_token_est
				n_bundles += 1

			f_out.flush()

			elapsed = time.time() - start_time
			print(
				f"[PROGRESS] bundles={n_bundles:,} failed={n_failed:,} "
				f"approx_tokens={total_tokens:,} elapsed_min={elapsed/60:.1f} "
				f"translation_cache={len(translation_cache):,}",
				flush=True,
			)

			if total_tokens >= args.target_total_tokens:
				print("Reached target token budget.", flush=True)
				break

	print(
		f"Done. bundles={n_bundles:,} failed={n_failed:,} "
		f"approx_tokens={total_tokens:,} translation_cache={len(translation_cache):,}",
		flush=True,
	)


if __name__ == "__main__":
	main()