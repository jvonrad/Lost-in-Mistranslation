#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python generate_multilingual_mcq.py \
  --seed_facts_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/seed_facts.jsonl \
  --output_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text.jsonl \
  --max_facts 20000
"""

import json
import time
import random
import argparse
from typing import Dict, List, Optional, Tuple

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

    if "short_explanation" in obj and not isinstance(obj["short_explanation"], str):
        return False

    return True


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_translate_prompt(text: str, lang: str, relation: Optional[str] = None) -> str:
    lang_name = LANG_TO_NAME[lang]
    relation_hint = f"\nThe label refers to a Wikidata entity used as the object of the relation: {relation}." if relation else ""
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


def generate_plain_text(
    llm: LLM,
    tokenizer: AutoTokenizer,
    text_prompt: str,
    sampling_params: SamplingParams,
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
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    output = llm.generate([prompt], sampling_params)[0]
    return output.outputs[0].text.strip()


def translate_missing_label(
    llm: LLM,
    tokenizer: AutoTokenizer,
    translation_sampling_params: SamplingParams,
    text: str,
    lang: str,
    translation_cache: Dict[Tuple[str, str], str],
) -> str:
    key = (lang, text)
    if key in translation_cache:
        return translation_cache[key]

    translated = generate_plain_text(
        llm=llm,
        tokenizer=tokenizer,
        text_prompt=build_translate_prompt(text, lang),
        sampling_params=translation_sampling_params,
    )

    if not translated:
        translated = text

    translation_cache[key] = translated
    return translated


def get_localized_answer_and_distractors(
    llm: LLM,
    tokenizer: AutoTokenizer,
    translation_sampling_params: SamplingParams,
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
        answer = translate_missing_label(
            llm=llm,
            tokenizer=tokenizer,
            translation_sampling_params=translation_sampling_params,
            text=fallback,
            lang=lang,
            translation_cache=translation_cache,
        )

    localized_distractors = []
    for i, (d_id, lbls) in enumerate(distractor_labels.items()):
        d_text = lbls.get(lang)
        if is_missing_label(d_text):
            fallback = lbls.get("en")
            if is_missing_label(fallback) and i < len(distractors_fallback):
                fallback = distractors_fallback[i]
            if is_missing_label(fallback):
                fallback = d_id

            d_text = translate_missing_label(
                llm=llm,
                tokenizer=tokenizer,
                translation_sampling_params=translation_sampling_params,
                text=fallback,
                lang=lang,
                translation_cache=translation_cache,
            )
        localized_distractors.append(d_text)

    return answer, localized_distractors


def build_mcq_prompt(
    fact: Dict,
    lang: str,
    answer: str,
    distractors: List[str],
) -> Tuple[str, List[str]]:
    lang_name = LANG_TO_NAME[lang]
    subject = fact["subject"]
    relation = fact["relation"]

    assert len(distractors) == 3, f"Need exactly 3 distractors for {fact['fact_id']} / {lang}"

    options = [answer] + distractors
    random.shuffle(options)

    prompt = f"""
Generate a factual multiple-choice question in {lang_name}.

Rules:
- Return ONLY valid JSON.
- Do NOT add markdown.
- Do NOT output reasoning or analysis.
- The question and short_explanation MUST be in {lang_name}.
- Use the provided options EXACTLY as given.
- Do NOT translate, paraphrase, or modify any option.
- Do NOT introduce new option strings.
- The answer_text MUST be exactly one of the provided options.
- Exactly one option is correct.

Return JSON with exactly these keys:
{{
  "question": "...",
  "answer_text": "...",
  "short_explanation": "..."
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


def generate_batch_for_fact(
    llm: LLM,
    tokenizer: AutoTokenizer,
    mcq_sampling_params: SamplingParams,
    translation_sampling_params: SamplingParams,
    translation_cache: Dict[Tuple[str, str], str],
    fact: Dict,
    debug_prompt_sample: bool = False,
):
    prompts = []
    meta = []

    for lang in LANGS:
        answer, distractors = get_localized_answer_and_distractors(
            llm=llm,
            tokenizer=tokenizer,
            translation_sampling_params=translation_sampling_params,
            translation_cache=translation_cache,
            fact=fact,
            lang=lang,
        )

        prompt, options = build_mcq_prompt(
            fact=fact,
            lang=lang,
            answer=answer,
            distractors=distractors,
        )

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

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )

        if debug_prompt_sample and lang == "en":
            print("\n[CHAT PROMPT SAMPLE]")
            print(chat_prompt[:900])
            print()

        prompts.append(chat_prompt)
        meta.append((lang, answer, options))

    t0 = time.time()
    outputs = llm.generate(prompts, mcq_sampling_params)
    print(f"  [BATCH DONE] {len(prompts)} langs in {time.time() - t0:.2f}s", flush=True)

    results = {}
    for out, (lang, gold_answer, options) in zip(outputs, meta):
        raw = '{\n  "question": "' + out.outputs[0].text
        obj = parse_json_object(raw)

        if obj is None:
            print(f"  [FAIL] lang={lang}", flush=True)
            print(f"  raw[:400]={raw[:400]!r}", flush=True)
            return None

        resolved_answer = resolve_answer_text(obj.get("answer_text", ""), options, gold_answer)
        if resolved_answer is None:
            print(f"  [FAIL] lang={lang} answer_text not in options", flush=True)
            print(f"  answer_text={obj.get('answer_text')!r}", flush=True)
            print(f"  options={options!r}", flush=True)
            return None

        obj["answer_text"] = resolved_answer
        obj["options"] = options

        if not validate_bundle_item(obj):
            print(f"  [FAIL] lang={lang} invalid bundle item", flush=True)
            print(f"  raw[:400]={raw[:400]!r}", flush=True)
            return None

        obj["_expected_gold_answer"] = gold_answer
        obj["_correct_object_id"] = fact["object_id"]
        obj["_candidate_options"] = options
        obj["_accepted_answers"] = fact.get("object_labels", {})
        results[lang] = obj

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_facts_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--target_total_tokens", type=int, default=200_000_000)
    ap.add_argument("--max_facts", type=int, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--max_tokens", type=int, default=192)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--debug_prompt_sample", action="store_true")
    args = ap.parse_args()

    random.seed(42)

    model_id = "Qwen/Qwen3.5-35B-A3B"

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

    with open(args.seed_facts_jsonl, "r", encoding="utf-8") as f_in, \
         open(args.output_jsonl, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            if args.max_facts is not None and i >= args.max_facts:
                break

            fact = json.loads(line)
            fact_id = fact["fact_id"]
            print(f"\n[FACT {i}] id={fact_id}", flush=True)

            bundle = {
                "fact_id": fact_id,
                "subject": fact["subject"],
                "relation": fact["relation"],
                "object": fact["object"],
                "object_id": fact["object_id"],
                "subject_id": fact["subject_id"],
                "property_id": fact["property_id"],
                "langs": {},
            }

            langs_obj = generate_batch_for_fact(
                llm=llm,
                tokenizer=tokenizer,
                mcq_sampling_params=mcq_sampling_params,
                translation_sampling_params=translation_sampling_params,
                translation_cache=translation_cache,
                fact=fact,
                debug_prompt_sample=args.debug_prompt_sample and i == 0,
            )

            if langs_obj is None:
                n_failed += 1
                continue

            bundle["langs"] = langs_obj
            bundle_token_est = estimate_tokens(json.dumps(langs_obj, ensure_ascii=False))

            f_out.write(json.dumps(bundle, ensure_ascii=False) + "\n")
            f_out.flush()

            total_tokens += bundle_token_est
            n_bundles += 1

            if n_bundles % 50 == 0:
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