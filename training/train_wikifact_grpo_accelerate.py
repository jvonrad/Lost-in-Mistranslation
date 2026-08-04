#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom grouped-rollout multilingual RL trainer — Accelerate + LoRA edition.

One training item = one fact.
For each fact:
  - create 12 separate single-language prompts
  - sample num_generations grouped rollouts
  - each grouped rollout contains 12 independent generations (one per language)
  - compute one joint reward over the 12 answers
  - normalize rewards across grouped rollouts for the same fact
  - apply that advantage to all 12 outputs in the rollout

## LAUNCH (4× A100-80GB with LoRA):

accelerate launch --num_processes 2 --multi_gpu training/train_wikifact_grpo_accelerate.py \
  --model_id jvonrad/Qwen-2.5-7B-TED \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir /data/jonathan/Lost-in-Mistranslation/models/qwen-2.5-7b-ted-grpo-accelerate \
  --per_device_train_batch_size 1 \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --num_generations 8 \
  --max_completion_length 32 \
  --run_name qwen-2.5-7b-ted-grpo-accelerate \
  --eval_steps 200 \
  --max_eval_wikifact 100 \
  --bf16 \
  --use_lora \
  --kl_coef 0.0 \
  --max_train_samples 20000 \
  --gen_micro_batch_size 192 \
  --logprob_micro_batch_size 48
"""

import os
import re
import json
import math
import random
import time
import argparse
import itertools
from typing import Dict, Any, List, Optional, Tuple
import shutil
import numpy as np
import evaluate
from datasets import load_dataset, Dataset

import prompt_scaffold as pscaf

import polyfact_schema as pfs

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B"
HF_DATASET_ID = "jvonrad/PolyFact-Clean"
HF_DATASET_CONFIG = "parallel"
OUTPUT_DIR = "/data/jonathan/Lost-in-Mistranslation/models/qwen2.5-7b-grpo-accelerate"
WANDB_PROJECT = "UnLock"

LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh", "ja", "ar", "sw", "bn"]

LANG_TO_NAME = {
    "en": "English", "de": "German", "id": "Indonesian", "pt": "Portuguese",
    "ar": "Arabic", "bn": "Bengali", "sw": "Swahili", "es": "Spanish",
    "ru": "Russian", "fr": "French", "ja": "Japanese", "zh": "Chinese",
}

LANG_NAME_MAP = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "es": "Spanish",
    "fr": "French", "id": "Indonesian", "ja": "Japanese", "pt": "Portuguese",
    "ru": "Russian", "sw": "Swahili", "zh": "Chinese",
}

FLORES_LANG_MAP = {
    "ar": "arb_Arab", "bn": "ben_Beng", "de": "deu_Latn",
    "es": "spa_Latn", "fr": "fra_Latn", "id": "ind_Latn",
    "ja": "jpn_Jpan", "pt": "por_Latn", "ru": "rus_Cyrl",
    "sw": "swh_Latn", "zh": "zho_Hans", "en": "eng_Latn",
}

MAX_EVAL_SAMPLES_PER_LANG = 1000
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
VALID_LETTERS = {"A", "B", "C", "D"}

LORA_R = 64
LORA_ALPHA = 128


ACCELERATE_STATE_MARKERS = ("pytorch_model.bin", "model.safetensors", "optimizer.bin")


def save_training_stats(path: str, global_step: int, cumulative_rollout_tokens: int,
                        cumulative_wall_seconds: float, num_gpus: int) -> None:
    """Cost-reporting sidecar (rollout tokens, wall-clock, GPU-hours) written
    next to each checkpoint, so a crash+resume doesn't lose the running total —
    reviewers asked for training/token cost figures alongside accuracy."""
    stats = {
        "global_step": global_step,
        "cumulative_rollout_tokens": cumulative_rollout_tokens,
        "cumulative_wall_seconds": cumulative_wall_seconds,
        "cumulative_gpu_hours": cumulative_wall_seconds * num_gpus / 3600,
        "num_gpus": num_gpus,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def load_training_stats(path: str) -> Dict[str, Any]:
    stats_path = os.path.join(path, "training_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {}


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the highest-step checkpoint-<step> dir that has a full
    accelerator.save_state (not just an adapter/tokenizer dump from an older
    checkpoint format) — falls through to the next-highest so one stale dir
    doesn't force a full restart when a good, older checkpoint is available."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir)
             if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
    ckpts.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    for d in ckpts:
        path = os.path.join(output_dir, d)
        if any(os.path.exists(os.path.join(path, m)) for m in ACCELERATE_STATE_MARKERS):
            return path
    return None


# ─────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--dataset_id", type=str, default=HF_DATASET_ID)
    ap.add_argument("--dataset_config", type=str, default=HF_DATASET_CONFIG,
                    help="HF config name. PolyFact-Clean has no default config "
                         "(12 per-language configs + 'parallel'), so 'parallel' is "
                         "required. Pass '' for single-config datasets like WIKI-FACT.")
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--logprob_micro_batch_size", type=int, default=4)
    ap.add_argument("--use_lora", action="store_true", default=False)

    ap.add_argument("--learning_rate", type=float, default=5e-6)
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
    ap.add_argument("--gen_micro_batch_size", type=int, default=12,
                    help="Number of prompts to generate at once during rollouts (reduce if OOM)")
    ap.add_argument("--gen_cache_implementation", type=str, default="static",
                    help="KV-cache implementation for rollout/eval generation. 'static' "
                         "preallocates the cache once per generate() call instead of the "
                         "default DynamicCache's per-step reallocation — ~2x faster rollout "
                         "generation on GH200 at the batch-96/short-completion regime here, "
                         "with byte-identical greedy output (verified). Pass 'none' to fall "
                         "back to the dynamic cache.")

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument(
        "--skip_periodic_eval", action="store_true",
        help="At each eval_steps boundary, write a resumable checkpoint without "
             "running the memory-intensive benchmark evaluation first.",
    )
    ap.add_argument("--max_eval_wikifact", type=int, default=250)
    ap.add_argument("--max_eval_mmlu", type=int, default=MAX_EVAL_SAMPLES_PER_LANG,
                    help="Global-MMLU eval samples PER LANGUAGE at each eval_steps "
                         "boundary (×12 langs). Default 1000 (=12k forward passes/eval) is "
                         "the dominant periodic-eval cost and does NOT affect training — "
                         "drop to e.g. 200 to reclaim wall-clock, or raise --eval_steps.")
    ap.add_argument("--resume_from_checkpoint", type=str, default="auto",
                    help="'auto' resumes from the latest checkpoint-<step> dir under "
                         "--output_dir if one exists, 'none' always starts fresh, or "
                         "pass an explicit checkpoint dir.")

    ap.add_argument("--min_languages", type=int, default=12)

    ap.add_argument("--coverage_reward_weight", type=float, default=0.05)
    ap.add_argument("--valid_option_reward_weight", type=float, default=0.15)
    ap.add_argument("--lora_r", type=int, default=LORA_R,
                    help="LoRA rank (paper runs used 64).")
    ap.add_argument("--lora_alpha", type=int, default=None,
                    help="LoRA alpha; default 2*lora_r (the paper ratio, 64/128).")
    ap.add_argument("--ref_impl", type=str, default="separate",
                    choices=["separate", "adapter_off"],
                    help="KL reference model. 'separate' loads a second frozen copy "
                         "of model_id (paper behaviour, +15 GB). 'adapter_off' reuses "
                         "the policy with its LoRA adapter disabled -- mathematically "
                         "the same reference (LoRA init is the base model) at zero "
                         "extra memory; requires --use_lora.")
    ap.add_argument("--reward_pooling", type=str, default="group",
                    choices=["group", "per_lang"],
                    help="'group' (paper): reward pooled over all 12 languages of a "
                         "rollout group, one z-scored advantage broadcast to every "
                         "language's generation. 'per_lang' (meta-review baseline, no "
                         "cross-lingual pooling): each (fact, lang) is its own GRPO "
                         "group; own-language reward only; all_correct_bonus ignored.")
    ap.add_argument("--all_correct_bonus", type=float, default=1.0,
                    help="Reward added when ALL languages are correct (cross-lingual "
                         "consistency bonus). Ablation: 0.0 disables it, 5.0 amplifies it. "
                         "Was previously hardcoded to 1.0.")
    ap.add_argument("--bonus_shape", choices=["all_or_nothing", "power", "ladder"],
                    default="all_or_nothing",
                    help="How --all_correct_bonus is distributed over the number of "
                         "correct languages k. 'all_or_nothing' (default, byte-for-byte "
                         "the previous behaviour) pays only at k==n_langs — a SPARSE "
                         "reward that fires on ~2-10%% of rollouts. 'power' pays "
                         "bonus*(k/n)^bonus_power, convex so the marginal value of the "
                         "last language exceeds the first while still paying nothing for "
                         "being consistently WRONG. 'ladder' uses explicit rungs.")
    ap.add_argument("--bonus_power", type=float, default=4.0,
                    help="Exponent for --bonus_shape power. 1.0 is linear (no convexity, "
                         "just rescales the count); higher concentrates weight near k=n.")
    ap.add_argument("--brevity_penalty", type=float, default=0.0,
                    help="Penalty on a CORRECT answer for text beyond the gold "
                         "option, as penalty * min(1, excess_chars/len(gold)). "
                         "0.0 (default) = previous behaviour: the matcher resolves "
                         "by containment so padding is free, which is how Qwen "
                         "drifted into low-probability filler and blew up its "
                         "gradients. Keep < 1.0 so a padded CORRECT answer still "
                         "outranks a wrong one; 0.3 is a reasonable start.")
    ap.add_argument("--brevity_denom_floor", type=int, default=10,
                    help="Floor on the brevity denominator, so the same ABSOLUTE "
                         "padding costs the same in every language. Gold answers "
                         "average 11-13 chars in most languages but 7.0 (ja) and "
                         "4.8 (zh); without a floor, CJK is ~2.5x more sensitive "
                         "to identical padding.")
    ap.add_argument("--empty_penalty", type=float, default=0.0,
                    help="Penalty for an EMPTY completion. 0.0 (default) reproduces "
                         "every pre-2026-08-03 run byte for byte, but leaves the "
                         "'silence hole': empty scores 0.0, tying wrong-but-valid and "
                         "BEATING non-empty unparseable (-0.5), so the cheapest escape "
                         "from garbled output is one EOS token. All rollouts then score "
                         "alike -> std 0 -> zero gradient -> an absorbing state training "
                         "can never leave (it killed qwen-final-main at both clip 5.0 and "
                         "clip 2.0). Use 1.0 to order it correctly: "
                         "correct +1 > valid-wrong 0 > unparseable -0.5 > empty -1.")
    ap.add_argument("--dead_run_patience", type=int, default=200,
                    help="Abort after this many CONSECUTIVE optimizer steps with "
                         "reward_std == 0. Such steps contribute exactly zero gradient, "
                         "so a run that never leaves the state cannot learn and only "
                         "burns the allocation. Isolated zero-std steps are normal "
                         "(a saturated all-correct group), which is why this counts "
                         "consecutive ones; 0 disables the check.")
    ap.add_argument("--bonus_ladder", type=str, default="9:1,10:2,11:3,12:5",
                    help="Rungs for --bonus_shape ladder as 'k:absolute_reward' pairs: the "
                         "default adds +1 / +2 / +3 / +5 when 9 / 10 / 11 / 12 languages are "
                         "correct. Highest matching rung wins; rungs do NOT accumulate. "
                         "--all_correct_bonus is IGNORED in ladder mode.")
    ap.add_argument("--max_eval_flores", type=int, default=32)
    ap.add_argument("--kl_coef", type=float, default=0.0,
                    help="KL penalty toward the reference policy (k3 estimator). "
                         "Default 0.0 = OFF, matching every science run in this repo, "
                         "which all passed --kl_coef 0.0 explicitly. Was 0.05, which "
                         "silently enabled a KL term for any launcher that omitted the "
                         "flag; combined with the old k1 estimator that term rewarded "
                         "divergence rather than penalising it.")

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--length_bucketing", action="store_true",
                    help="Sort rollout prompts (generation) and sequences (loss) by "
                         "length so each micro-batch pads to its own max. Tokenizer "
                         "fertility differs ~4x across the 12 languages (bn ~307 tok "
                         "vs en ~73 for the same fact), so unsorted micro-batches pad "
                         "to the Bengali max; sorting cuts loss-pass FLOPs and the "
                         "dominant [mb,T,vocab] memory term correspondingly. Changes "
                         "no math -- the loss is a mean over sequences.")
    ap.add_argument("--fused_logprob", action="store_true",
                    help="Use F.cross_entropy for per-token logprobs instead of "
                         "materialising the full-vocab log_softmax tensor. Halves the "
                         "dominant loss-pass memory term; numerically equivalent.")
    ap.add_argument("--task_format", choices=["mcq", "freeform"], default="mcq",
                    help="Rollout prompt format. 'mcq' (default, previous behaviour) "
                         "lists the four candidates, training SELECTION AMONG SHOWN "
                         "OPTIONS. 'freeform' hides them, training closed-book recall "
                         "— the task KLAR measures. Reward matching is identical in "
                         "both; only the prompt changes.")
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Gradient-norm clip. Was hardcoded 1.0, but the measured "
                         "pre-clip norm has median 139 (Qwen) / 3.7 (OLMo), so ~99%% of "
                         "steps are renormalised and relative step magnitude is "
                         "discarded. Raise to loosen.")
    ap.add_argument("--prompt_scaffold", type=str, default="native",
                    choices=["native", "en"],
                    help="Language of the FREE-FORM prompt scaffold. 'native' "
                         "localises it per language (default); 'en' restores the "
                         "old English scaffold for an apples-to-apples ablation. "
                         "Does NOT affect the log-likelihood MCQ path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", type=str, default="wandb")
    return ap.parse_args()


# ─────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────

def safe_strip(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def answer_text_to_letter(options: List[str], answer_text: str) -> Optional[str]:
    answer_text = safe_strip(answer_text)
    options = [safe_strip(x) for x in options]
    for i, opt in enumerate(options):
        if opt == answer_text:
            return INDEX_TO_LETTER[i]
    return None


def normalize_text(text: str) -> str:
    text = safe_strip(text).lower()
    text = text.replace("\u2019", "'").replace("`", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"^[\"'`\s:;\-\u2013\u2014\(\)\[\]\{\}]+", "", text)
    text = re.sub(r"[\"'`\s:;\-\u2013\u2014\(\)\[\]\{\}\.,!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_answer_text(text: str) -> str:
    text = safe_strip(text)
    text = pscaf.strip_answer_label(text)
    text = text.split("\n")[0].strip()
    return text


# A generation that is *nothing but* an option letter: "D", "c)", "B.".
# normalize_text already strips surrounding punctuation, so this is just the
# single character — but keep the punctuation branch for safety.
# Leading option-letter prefix in any of the forms the models actually emit:
# "A)", "A.", "A -", "A:", "A]" plus optional space. Latin letters only, matching
# the scaffold, which keeps A-D Latin in every language.
_LETTER_PREFIX_RE = re.compile(r"^\s*[abcdABCD]\s*[\.\)\]:\-]\s*")

_PURE_LETTER_RE = re.compile(r"^([abcd])[\.\)\]:\-]?$", re.I)


def is_pure_letter_answer(pred_text: str) -> bool:
    """True when the model returned only the option letter and no answer text."""
    return bool(_PURE_LETTER_RE.match(normalize_text(extract_answer_text(pred_text))))


def resolve_prediction_to_letter(
    pred_text: str, option_map: Dict[str, str], allow_bare_letter: bool = False,
) -> Tuple[Optional[str], bool]:
    """Map a free-text generation onto one of the four option letters.

    The task is still multiple choice — the options ARE shown in the prompt —
    but the prompt asks for the answer *text*, so a generation consisting only
    of a letter is treated as unresolved (`allow_bare_letter=False`, default).
    Previously "D" was mapped straight to option D and scored as fully correct,
    which let GRPO collect reward by eliminating over the visible candidate
    list instead of recalling the fact — the exact shortcut the closed-book
    trainer was written to avoid, and not the skill evaluate_accuracy.py
    measures. Pass allow_bare_letter=True to restore the old behaviour.

    Text-based resolution is tried BEFORE the letter rule, so a mixed answer
    like "D, Kiran Desai" is resolved by the entity it names rather than by the
    letter it happens to start with.
    """
    pred_raw = extract_answer_text(pred_text)
    pred_norm = normalize_text(pred_raw)
    if not pred_norm:
        return None, False

    option_norm = {letter: normalize_text(text) for letter, text in option_map.items()}

    # 1. exact answer text
    for letter, opt_norm in option_norm.items():
        if pred_norm == opt_norm:
            return letter, True

    # 2. a bare letter — no answer text at all. Checked BEFORE containment:
    #    "d" is a substring of "durham", so a lone letter would otherwise be
    #    silently resolved to whichever option happens to contain that char.
    m = _PURE_LETTER_RE.match(pred_norm)
    if m:
        return (m.group(1).upper(), True) if allow_bare_letter else (None, False)

    # 3. answer text contained in / containing the generation (unambiguously).
    #    The `pred in option` direction needs >=2 chars for the same reason —
    #    2 is deliberate, not 3: CJK answers are legitimately 2 characters.
    candidates = [
        letter for letter, opt_norm in option_norm.items()
        if opt_norm and ((len(pred_norm) >= 2 and pred_norm in opt_norm)
                         or opt_norm in pred_norm)
    ]
    if len(candidates) == 1:
        return candidates[0], True

    # 4. a letter followed by text that matched nothing above
    m = re.match(r"^([abcd])(?:[\.\)\]:\-\s])", pred_norm)
    if m and allow_bare_letter:
        return m.group(1).upper(), True

    return None, False


# ─────────────────────────────────────────────
# Prompt / dataset builders
# ─────────────────────────────────────────────

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def build_prompt_eval(question: str) -> str:
    """The prompt evaluate_accuracy.py / evaluate_crosslingual_consistency.py use.

    Deliberately bare: no instruction wrapper and the options are NOT shown, so
    log-likelihood scoring measures closed-book recall. Kept byte-identical to
    the eval scripts' build_prompt so in-loop numbers are comparable.
    """
    return f"Question: {question}\nAnswer:"


def build_single_language_prompt(lang, question, options, scaffold="native",
                                 task_format="mcq"):
    """Free-form MCQ prompt, localised to `lang` (see prompt_scaffold).

    Base models continue in the language of their context, so an English
    scaffold around a non-English question biases the answer toward English
    and then fails option matching. scaffold='en' restores the old prompt.
    """
    return pscaf.build_single_language_prompt(lang, question, options, scaffold,
                                              task_format=task_format)


def build_grouped_fact_item(ex: Dict[str, Any], scaffold: str = "native",
                            task_format: str = "mcq") -> Dict[str, Any]:
    # Accepts PolyFact-Clean (`translations` + option_a..d + answer_index) and
    # legacy WIKI-FACT (`langs` + options list) alike — see polyfact_schema.
    langs_data = pfs.lang_blocks(ex)
    if not langs_data:
        return {"is_valid": False, "num_languages": 0}

    prompts_by_lang = {}
    meta_by_lang = {}

    for lang in LANGS:
        if lang not in langs_data:
            continue
        parsed = pfs.normalize_lang_item(langs_data[lang])
        if parsed is None:
            continue
        question, options, answer_text, gold_idx = parsed
        gold_letter = pfs.gold_letter(gold_idx)

        option_map = pfs.option_map(options)
        prompts_by_lang[lang] = build_single_language_prompt(
            lang, question, option_map, scaffold=scaffold, task_format=task_format)
        meta_by_lang[lang] = {
            "gold_letter": gold_letter,
            "gold_text": answer_text,
            "options": option_map,
            # Kept so periodic eval can rebuild the *evaluation* prompt
            # ("Question: {q}\nAnswer:", options hidden) rather than reusing the
            # training prompt, which lists the options and therefore measures a
            # different task. See compute_polyfact_logprob_metrics.
            "question": question,
            # Wikidata QIDs of the 4 options, in the same A-D order (PolyFact-
            # Clean only; None for legacy WIKI-FACT). Enables EXACT cross-
            # lingual option alignment for in-loop RankC — the hidden-state
            # matcher is only a fallback for data without ids.
            "option_ids": langs_data[lang].get("option_ids"),
        }

    return {
        "fact_id": ex.get("fact_id", ""),
        "prompts_by_lang_json": json.dumps(prompts_by_lang, ensure_ascii=False, sort_keys=True),
        "meta_by_lang_json": json.dumps(meta_by_lang, ensure_ascii=False, sort_keys=True),
        "num_languages": len(meta_by_lang),
        "is_valid": len(meta_by_lang) > 0,
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {
        "fact_id": [x["fact_id"] for x in batch],
        "prompts_by_lang_json": [x["prompts_by_lang_json"] for x in batch],
        "meta_by_lang_json": [x["meta_by_lang_json"] for x in batch],
    }


# ─────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────

def parse_bonus_ladder(spec: str) -> List[Tuple[int, float]]:
    """Parse "9:1,10:2,11:3,12:5" -> [(9, 1.0), (10, 2.0), (11, 3.0), (12, 5.0)].

    Keys are counts of correct languages. Values are ABSOLUTE reward added at
    that rung — NOT fractions of --all_correct_bonus, which is ignored entirely
    in ladder mode. Highest matching rung wins; rungs do not accumulate.
    """
    rungs: List[Tuple[int, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        rungs.append((int(k), float(v)))
    return sorted(rungs)


def consistency_bonus(
    n_correct: int, n_langs: int, bonus: float,
    shape: str = "all_or_nothing", power: float = 4.0,
    ladder: Optional[List[Tuple[int, float]]] = None,
) -> float:
    """Cross-lingual consistency bonus as a function of how many languages agree
    with gold.

    The base reward is the COUNT of correct languages, which is LINEAR in
    n_correct: going 11 -> 12 pays exactly as much as 5 -> 6. But Total
    Consistency is P(n_correct == n_langs) — it lives entirely in that last
    step. Making the bonus convex in n_correct is what puts weight there.

    `all_or_nothing` (default, reproduces the previous behaviour exactly) is the
    extreme case: all the weight on the final rung. That makes it a SPARSE
    reward — it fires on ~2-10% of rollouts, which is why it measured as only
    0.4-2.4% of the mean reward and contributed almost no advantage variance.
    `power` and `ladder` keep the same label-anchored quantity (a wrong answer
    is never rewarded for being consistently wrong) while shaping every rollout.
    """
    if n_langs <= 0:
        return 0.0
    if shape == "ladder":
        # Absolute rewards; --all_correct_bonus does not scale them (so the
        # bonus == 0.0 short-circuit below must NOT apply here).
        best = 0.0
        for thr, val in (ladder or []):
            if n_correct >= thr:
                best = max(best, val)
        return best
    if bonus == 0.0:
        return 0.0
    if shape == "all_or_nothing":
        return bonus if n_correct >= n_langs else 0.0
    if shape == "power":
        return bonus * ((n_correct / n_langs) ** power)
    raise ValueError(f"unknown bonus_shape {shape!r}")


def compute_group_reward(
    pred_text_by_lang: Dict[str, str],
    meta_by_lang: Dict[str, Any],
    coverage_weight: float,
    valid_option_weight: float,
    all_correct_bonus: float,
    bonus_shape: str = "all_or_nothing",
    bonus_power: float = 4.0,
    bonus_ladder: Optional[List[Tuple[int, float]]] = None,
    brevity_penalty: float = 0.0,
    brevity_denom_floor: int = 10,
    empty_penalty: float = 0.0,
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
            # Brevity term. The matcher resolves by CONTAINMENT, so "Tel Aviv"
            # and "Tel Aviv (Yosha karne)" score identically — appending text is
            # free. Qwen exploits this: completions grow to the 48-token cap
            # (34.6 -> 47.7 measured) and the padding degenerates into very
            # low-probability tokens ("(فيsm)", "(ヴァ)"). Since the REINFORCE
            # gradient is A * grad log pi, and grad log pi explodes on
            # low-probability tokens, that padding is the source of the
            # heavy gradient tail (Qwen p95 435, max 1701, vs OLMo max 5.61)
            # that destroyed two runs. OLMo never learned to pad and never
            # produced a gradient above 5.61.
            #
            # n_correct is deliberately NOT touched, so total-consistency
            # accounting and the bonus/ladder semantics are unchanged.
            #
            # MUST stay < 1.0: at 1.0 a maximally-padded CORRECT answer would
            # score 0.0, the same as a wrong-but-valid option, inverting the
            # ordering the reward depends on. 0.3 keeps the floor at 0.7.
            if brevity_penalty > 0.0:
                gold_text = (meta.get("options") or {}).get(meta["gold_letter"], "")
                p, g = safe_strip(pred), safe_strip(gold_text)
                if g:
                    # LANGUAGE ROBUSTNESS. Two corrections, both measured on
                    # PolyFact-Clean test (gold length in chars: en 11.8, de 12.8,
                    # bn 12.8 ... but ja 7.0 and zh 4.8 — CJK is far denser):
                    #
                    # 1. Strip a leading option-letter prefix. "A) " is 3 chars =
                    #    75% excess on a 4-char zh gold but 10% on a 10-char en
                    #    gold, a 7.5x difference for behaviour the model shows in
                    #    every language. Penalising that unequally would push
                    #    languages toward different output styles — the opposite
                    #    of what a consistency objective should do.
                    # 2. Floor the denominator, so the SAME ABSOLUTE padding costs
                    #    the same everywhere. Padding is padding: "(Japanese)" is
                    #    equally unwanted whether the answer is 4 or 20 chars, and
                    #    normalising by a 4-char gold made CJK ~2.5x more
                    #    sensitive. Characters (not tokens) are already the right
                    #    unit here — token counts range 2.75 (en) to 17.4 (bn) per
                    #    option, which is the tokenizer-fertility bias the eval
                    #    switched to byte normalisation to avoid.
                    p = _LETTER_PREFIX_RE.sub("", p, count=1).strip()
                    excess = max(0, len(p) - len(g))
                    denom = max(len(g), brevity_denom_floor)
                    score -= brevity_penalty * min(1.0, excess / denom)
        elif not safe_strip(pred):
            # THE SILENCE HOLE (found 2026-08-03, after it killed qwen-final-main
            # twice). Without this branch the reward table is
            #     correct +1.0 | valid-but-wrong 0.0 | unparseable -0.5 | EMPTY 0.0
            # so producing NOTHING ties the best non-correct outcome and strictly
            # beats a wrong guess. When the policy drifts into unparseable output
            # the gradient correctly pushes away from -0.5 -- and the cheapest
            # escape is one EOS token, not a valid answer. Observed end to end in
            # qwen-final-main-clip2:
            #   step 3050  rew 13.0  every language clean ("D) General Motors")
            #   step 3200  rew  4.0  std 0.00, all valid but mostly wrong
            #   step 3250  rew  3.0  garbage appearing ("what?", "19 Greentrees A")
            #   step 3300  rew  0.00 std 0.00 grad 0.00 -- all 12 languages EMPTY
            # Once every rollout is empty the rewards are identical, so std = 0,
            # advantages z-score to 0, and the gradient is exactly zero forever:
            # an ABSORBING state no amount of further training can leave. Gradient
            # clipping does not touch this (clip 5.0 died at 3050, clip 2.0 at
            # ~3270 -- tightening only slowed the walk into the hole).
            #
            # Default 0.0 keeps every pre-2026-08-03 run reproducible byte for
            # byte; pass --empty_penalty 1.0 to make silence the WORST outcome:
            #     correct +1.0 > valid-wrong 0.0 > unparseable -0.5 > empty -1.0
            score -= empty_penalty
        elif not matched_valid:
            score -= 0.5

        if matched_valid:
            n_valid += 1
        if safe_strip(pred):
            n_pred += 1

    bonus = consistency_bonus(
        n_correct, len(meta_by_lang), all_correct_bonus,
        shape=bonus_shape, power=bonus_power, ladder=bonus_ladder,
    )
    score += bonus

    return {"score": score, "n_correct": n_correct, "n_valid": n_valid,
            "n_pred": n_pred, "bonus": bonus}


def _zscore(vals: List[float]) -> List[float]:
    t = torch.tensor(vals, dtype=torch.float32)
    return ((t - t.mean()) / (t.std(unbiased=False) + 1e-6)).tolist()


def compute_group_advantages(
    batch: Dict[str, List[Any]],
    grouped_preds: Dict[Tuple[int, int], Dict[str, str]],
    num_generations: int,
    coverage_weight: float,
    valid_option_weight: float,
    all_correct_bonus: float,
    reward_pooling: str = "group",
    bonus_shape: str = "all_or_nothing",
    bonus_power: float = 4.0,
    bonus_ladder: Optional[List[Tuple[int, float]]] = None,
    brevity_penalty: float = 0.0,
    brevity_denom_floor: int = 10,
    empty_penalty: float = 0.0,
):
    """Rewards + advantages for every generated sequence.

    Returns group_rewards keyed **(fact_idx, gen_idx, lang)** in BOTH modes,
    plus group_stats keyed (fact_idx, gen_idx) for monitoring.

    reward_pooling="group" (the paper's method): the reward is pooled over all
    12 languages of a rollout group (sum of per-language correctness +
    all_correct_bonus), z-scored across the G groups of the fact, and the SAME
    advantage is broadcast to every language's generation — cross-lingual
    credit sharing: a language's output gets gradient whenever the group
    outcome varies, even if its own G rollouts were uniformly wrong.

    reward_pooling="per_lang" (meta-review baseline, no cross-lingual
    pooling): each (fact, lang) is its own GRPO group — reward per generation
    is that language's own correctness only (+1 correct / -0.5 non-empty
    unparseable / 0 else), z-scored across the G rollouts of that (fact,
    lang), applied only to that generation. all_correct_bonus is undefined
    here and ignored (warned about at startup). Note the degenerate-group
    profile differs by construction: per-language rewards are near-ternary, so
    a hard language whose G rollouts are all wrong yields std=0 -> zero
    gradient for that language — exactly the signal-starvation this baseline
    is meant to exhibit. `degenerate_frac` in group_stats tracks it.
    """
    group_rewards = {}
    group_stats = {}

    for fact_idx, meta_json in enumerate(batch["meta_by_lang_json"]):
        meta_by_lang = json.loads(meta_json)
        langs = list(meta_by_lang.keys())

        # Per-(gen, lang) correctness, computed once and reused by both modes.
        per_gl: Dict[Tuple[int, str], float] = {}
        for gen_idx in range(num_generations):
            preds = grouped_preds.get((fact_idx, gen_idx), {})
            # Pooled stats are kept for monitoring in both modes (bonus only
            # counts toward the actual reward in group mode).
            stats = compute_group_reward(
                preds, meta_by_lang,
                coverage_weight=coverage_weight,
                valid_option_weight=valid_option_weight,
                all_correct_bonus=all_correct_bonus if reward_pooling == "group" else 0.0,
                bonus_shape=bonus_shape,
                bonus_power=bonus_power,
                bonus_ladder=bonus_ladder,
                brevity_penalty=brevity_penalty,
                brevity_denom_floor=brevity_denom_floor,
                empty_penalty=empty_penalty,
            )
            group_stats[(fact_idx, gen_idx)] = stats
            for lang, meta in meta_by_lang.items():
                pred = preds.get(lang, "")
                letter, matched = resolve_prediction_to_letter(pred, meta["options"])
                if letter == meta["gold_letter"]:
                    per_gl[(gen_idx, lang)] = 1.0
                elif not safe_strip(pred):
                    # Same silence hole as the pooled path — kept in sync here so
                    # the two poolings cannot disagree about what an empty
                    # completion is worth.
                    per_gl[(gen_idx, lang)] = -empty_penalty
                elif not matched:
                    per_gl[(gen_idx, lang)] = -0.5
                else:
                    per_gl[(gen_idx, lang)] = 0.0

        if reward_pooling == "group":
            rewards = [group_stats[(fact_idx, g)]["score"] for g in range(num_generations)]
            advantages = _zscore(rewards)
            for gen_idx in range(num_generations):
                for lang in langs:
                    group_rewards[(fact_idx, gen_idx, lang)] = {
                        "reward": float(rewards[gen_idx]),
                        "advantage": float(advantages[gen_idx]),
                    }
        elif reward_pooling == "per_lang":
            n_degen = 0
            for lang in langs:
                rewards = [per_gl[(g, lang)] for g in range(num_generations)]
                advantages = _zscore(rewards)
                if max(rewards) == min(rewards):
                    n_degen += 1
                for gen_idx in range(num_generations):
                    group_rewards[(fact_idx, gen_idx, lang)] = {
                        "reward": float(rewards[gen_idx]),
                        "advantage": float(advantages[gen_idx]),
                    }
            # Fraction of this fact's language-groups that produced zero
            # gradient — the pooling baseline's expected failure mode.
            for gen_idx in range(num_generations):
                group_stats[(fact_idx, gen_idx)]["degenerate_frac"] = n_degen / max(len(langs), 1)
        else:
            raise ValueError(f"Unknown reward_pooling: {reward_pooling!r}")

    return group_rewards, group_stats


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────

def log_sample_rollout_to_file(
    output_dir: str,
    global_step: int,
    grouped_preds: Dict,
    group_rewards: Dict,
    batch: Dict,
):
    sample_dir = os.path.join(output_dir, "grpo_samples")
    os.makedirs(sample_dir, exist_ok=True)
    log_path = os.path.join(sample_dir, "rollout_samples.txt")
    sample_key = sorted(grouped_preds.keys())[0]
    sample_fact_idx, sample_gen_idx = sample_key

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
        for lang in LANGS:
            gr = group_rewards.get((sample_fact_idx, sample_gen_idx, lang))
            if gr:
                f.write(f"  [{lang}] reward={round(gr['reward'], 4)} "
                        f"advantage={round(gr['advantage'], 4)}\n")


# ─────────────────────────────────────────────
# Global MMLU evaluation
# ─────────────────────────────────────────────

def format_global_mmlu_example(ex, tokenizer):
    question = ex["question"].strip()
    a, b, c, d = ex["option_a"].strip(), ex["option_b"].strip(), ex["option_c"].strip(), ex["option_d"].strip()
    gold = ex["answer"].strip()

    prompt = f"Question: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
    target = f" {gold}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    assert len(target_ids) == 1, f"Expected 1 answer token, got {target_ids} for gold={gold}"

    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def load_global_mmlu_dev_eval_by_lang(langs, tokenizer, max_samples=MAX_EVAL_SAMPLES_PER_LANG):
    """Periodic in-training Global-MMLU monitoring — DEV split only.

    This used to load split="test" despite the function name, which meant every
    training run was watching (and implicitly selecting checkpoints on) the same
    split the paper reports Global-MMLU numbers from. Global-MMLU ships both
    'dev' and 'test' for every language config; monitoring belongs on dev, and
    the reported numbers come from evaluate_crosslingual_consistency.py on test
    (Global-MMLU-Lite) anyway. Dev is also far smaller, so this cuts the
    dominant periodic-eval cost as a side effect.

    Languages are joined on `sample_id` and returned in one shared id order, the
    way evaluate_crosslingual_consistency.py does it. This is required for the
    cross-lingual metrics: total consistency and RankC are only meaningful if
    every language is answering the SAME questions. Selecting range(max_samples)
    per language independently (the previous behaviour) does not guarantee that.
    Global-MMLU options are parallel by index across languages, so letter slots
    correspond directly and no option alignment is needed.

    Returns (eval_sets, sample_ids) where sample_ids is the shared, ordered id
    list — row i of every language's dataset is sample_ids[i].
    """
    raw = {}
    for lang in langs:
        raw[lang] = load_dataset("CohereLabs/Global-MMLU", lang, split="dev")

    if not raw:
        return {}, []

    id_col = "sample_id" if "sample_id" in next(iter(raw.values())).column_names else None
    if id_col is None:
        # Fall back to positional alignment, but say so — silently pairing
        # unrelated questions across languages would corrupt every
        # cross-lingual number downstream.
        print("[warn] Global-MMLU has no 'sample_id' column; falling back to "
              "positional alignment for cross-lingual metrics.", flush=True)
        n = min(len(ds) for ds in raw.values())
        if max_samples is not None:
            n = min(n, max_samples)
        eval_sets = {
            lang: ds.select(range(n)).map(
                lambda ex: format_global_mmlu_example(ex, tokenizer),
                remove_columns=ds.column_names,
            )
            for lang, ds in raw.items()
        }
        return eval_sets, list(range(n))

    common = None
    for ds in raw.values():
        ids = set(map(str, ds[id_col]))
        common = ids if common is None else (common & ids)
    shared = sorted(common)
    if max_samples is not None:
        shared = shared[:max_samples]
    shared_pos = {sid: i for i, sid in enumerate(shared)}

    eval_sets = {}
    for lang, ds in raw.items():
        keep = [i for i, sid in enumerate(map(str, ds[id_col])) if str(sid) in shared_pos]
        keep.sort(key=lambda i: shared_pos[str(ds[id_col][i])])
        ds = ds.select(keep)
        eval_sets[lang] = ds.map(
            lambda ex: format_global_mmlu_example(ex, tokenizer),
            remove_columns=ds.column_names,
        )
    return eval_sets, shared


# ─────────────────────────────────────────────
# FLORES BLEU evaluation
# ─────────────────────────────────────────────

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
                **tok, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            gen_only = gen[:, tok["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            preds.extend([x.strip() for x in decoded])

        score = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
        metrics[f"flores_bleu/{lang}"] = score

    if metrics:
        metrics["flores_bleu/avg"] = sum(metrics.values()) / len(metrics)
    return metrics


# ─────────────────────────────────────────────
# Hidden State Cosine Similarity
# ─────────────────────────────────────────────

def mean_pool_hidden(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def compute_flores_hidden_cosine(model, tokenizer, flores_sets, device, batch_size=4, max_length=512):
    metrics = {}
    model.eval()

    core_model = model.module if hasattr(model, "module") else model
    # Handle PEFT wrapping
    if hasattr(core_model, "base_model") and hasattr(core_model.base_model, "model"):
        config_model = core_model.base_model.model
    else:
        config_model = core_model
    n_layers = config_model.config.num_hidden_layers
    mid_layer = n_layers // 2

    all_mid, all_last = [], []

    for lang, data in flores_sets.items():
        src_texts, tgt_texts = data["src_texts"], data["tgt_texts"]

        for i in range(0, len(src_texts), batch_size):
            batch_src = src_texts[i:i + batch_size]
            batch_tgt = tgt_texts[i:i + batch_size]

            tok_src = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            tok_tgt = tokenizer(batch_tgt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

            out_src = model(**tok_src, output_hidden_states=True, use_cache=False)
            out_tgt = model(**tok_tgt, output_hidden_states=True, use_cache=False)

            src_mid = mean_pool_hidden(out_src.hidden_states[mid_layer], tok_src["attention_mask"])
            tgt_mid = mean_pool_hidden(out_tgt.hidden_states[mid_layer], tok_tgt["attention_mask"])
            src_last = mean_pool_hidden(out_src.hidden_states[n_layers], tok_src["attention_mask"])
            tgt_last = mean_pool_hidden(out_tgt.hidden_states[n_layers], tok_tgt["attention_mask"])

            all_mid.extend(F.cosine_similarity(src_mid, tgt_mid, dim=-1).float().cpu().tolist())
            all_last.extend(F.cosine_similarity(src_last, tgt_last, dim=-1).float().cpu().tolist())

            del tok_src, tok_tgt, out_src, out_tgt
            torch.cuda.empty_cache()

    metrics["hidden_cosine_mid/avg"] = float(np.mean(all_mid)) if all_mid else 0.0
    metrics["hidden_cosine_last/avg"] = float(np.mean(all_last)) if all_last else 0.0
    return metrics


# ─────────────────────────────────────────────
# WikiFact grouped eval
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_polyfact_freeform(model, tokenizer, eval_ds, max_prompt_length,
                              max_completion_length, cache_implementation="static"):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    gen_kwargs = dict(
        max_new_tokens=max_completion_length, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if cache_implementation:
        gen_kwargs["cache_implementation"] = cache_implementation

    total_examples = total_slots = total_correct = total_valid = total_all_correct = 0
    per_lang_correct = {lang: 0 for lang in LANGS}
    per_lang_total = {lang: 0 for lang in LANGS}

    for ex in eval_ds:
        prompts_by_lang = json.loads(ex["prompts_by_lang_json"])
        meta_by_lang = json.loads(ex["meta_by_lang_json"])
        langs = [lang for lang in LANGS if lang in prompts_by_lang and lang in meta_by_lang]
        prompts = [prompts_by_lang[lang] for lang in langs]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(**inputs, **gen_kwargs)

        pred_text_by_lang = {}
        for i, lang in enumerate(langs):
            gen_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            pred_text_by_lang[lang] = extract_answer_text(gen_text)

        total_examples += 1
        ex_all_correct = True
        for lang in langs:
            total_slots += 1
            per_lang_total[lang] += 1
            resolved_letter, matched_valid = resolve_prediction_to_letter(
                pred_text_by_lang.get(lang, ""), meta_by_lang[lang]["options"],
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
        "polyfact/freeform_accuracy": total_correct / total_slots if total_slots else 0.0,
        "polyfact/freeform_resolution_rate": total_valid / total_slots if total_slots else 0.0,
        "polyfact/freeform_total_consistency": total_all_correct / total_examples if total_examples else 0.0,
        "polyfact/n_examples": float(total_examples),
    }
    for lang in LANGS:
        metrics[f"polyfact/freeform_lang_acc_{lang}"] = (
            per_lang_correct[lang] / per_lang_total[lang] if per_lang_total[lang] else 0.0
        )

    if was_training:
        model.train()
    return metrics


# ─────────────────────────────────────────────
# RankC (Qi et al., EMNLP 2023) — periodic training-time monitoring
# ─────────────────────────────────────────────

RANKC_N_OPT = 4
RANKC_WEIGHTS = [math.exp(RANKC_N_OPT - j) for j in range(1, RANKC_N_OPT + 1)]
_RANKC_Z = sum(RANKC_WEIGHTS)
RANKC_WEIGHTS = [w / _RANKC_Z for w in RANKC_WEIGHTS]


def rankc_pair_polyfact(slots_a: List[int], slots_b: List[int]) -> float:
    score = 0.0
    for j in range(1, RANKC_N_OPT + 1):
        top_a, top_b = set(slots_a[:j]), set(slots_b[:j])
        score += RANKC_WEIGHTS[j - 1] * len(top_a & top_b) / j
    return score


def _greedy_align(sim: np.ndarray) -> List[int]:
    """sim[i, j] = similarity of this language's option i to English's option j.
    1:1 greedy max-similarity assignment; exact for the well-separated 4x4 case
    and avoids pulling in scipy just for a 4-item Hungarian assignment."""
    sim = sim.copy()
    n = sim.shape[0]
    align = [-1] * n
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        align[i] = int(j)
        sim[i, :] = -1e9
        sim[:, j] = -1e9
    return align


@torch.no_grad()
def compute_polyfact_logprob_metrics(model, tokenizer, eval_ds, max_prompt_length, device):
    """Log-likelihood MCQ metrics on PolyFact, mirroring the evaluate/ scripts.

    ONE forward pass over (prompt + option) for 4 options x 12 languages yields
    all three metrics, so accuracy and total consistency are free once RankC is
    being computed:

      polyfact/mcq_accuracy            argmax over the 4 option logprobs
      polyfact/mcq_total_consistency   fraction of facts correct in ALL languages
      consistency/rankc_avg[_en_x]    RankC over the full 4-option ranking

    Deliberately matches evaluate_accuracy.py rather than the training setup:
      * the EVAL prompt "Question: {q}\\nAnswer:" with options hidden, not the
        training instruction prompt that lists A-D (that measures a different
        task -- selecting among shown options vs closed-book recall);
      * BYTE normalization (logprob sum / len(option.encode("utf-8"))), which
        is evaluate_accuracy.py's current default. It used to use per-token
        mean here while the eval default moved to byte, so the two numbers were
        not comparable.

    FIXED (2026-08-02): this scored the bare letters "A".."D" instead of the
    option texts. meta["options"] is a Dict[str, str] keyed by letter, so
    `enumerate(meta["options"])` iterated KEYS -- every RankC number logged
    before this fix ranked the four letter tokens, not the answers.

    Cross-lingual option alignment: options are independently shuffled per
    language with no stored correspondence, so each language is aligned to
    English by cosine similarity of the model's OWN mean-pooled hidden states
    over the option tokens (same idea as compute_flores_hidden_cosine), no
    extra embedder needed. Caveat: that alignment is model-dependent and can
    drift as training changes representations, so this is a monitoring signal
    -- the paper number comes from evaluate_crosslingual_consistency.py, which
    aligns exactly via PolyFact-Clean's stored option_ids.
    """
    was_training = model.training
    model.eval()
    n_layers = model.config.num_hidden_layers
    mid_layer = n_layers // 2

    facts = []
    for ex in eval_ds:
        prompts_by_lang = json.loads(ex["prompts_by_lang_json"])
        meta_by_lang = json.loads(ex["meta_by_lang_json"])
        langs = [l for l in LANGS if l in prompts_by_lang and l in meta_by_lang]
        if "en" not in langs or len(langs) < 2:
            continue
        facts.append((prompts_by_lang, meta_by_lang, langs))

    if not facts:
        if was_training:
            model.train()
        return {}

    items = []
    for fi, (prompts_by_lang, meta_by_lang, langs) in enumerate(facts):
        for lang in langs:
            meta = meta_by_lang[lang]
            # Eval-side prompt (options hidden), NOT prompts_by_lang[lang].
            # Older items may predate the stored question; fall back rather than
            # crash a long training run over a stale dataset-map cache.
            question = meta.get("question")
            prompt = build_prompt_eval(question) if question else prompts_by_lang[lang]
            prompt_ids = tokenizer(
                prompt, add_special_tokens=True,
                truncation=True, max_length=max_prompt_length,
            )["input_ids"]
            # meta["options"] is a Dict[str, str] keyed "A".."D" -- iterate
            # .items() to get the option TEXT (iterating the dict gives letters).
            for letter, opt in meta["options"].items():
                oi = LETTER_TO_IDX[letter]
                opt_ids = tokenizer(" " + opt, add_special_tokens=False)["input_ids"]
                items.append({
                    "fi": fi, "lang": lang, "oi": oi,
                    "ids": prompt_ids + opt_ids, "plen": len(prompt_ids),
                    "nbytes": max(len(opt.encode("utf-8")), 1),
                })

    scores: Dict[Tuple[int, str], List[float]] = {}
    embeds: Dict[Tuple[int, str], List[Any]] = {}
    micro_bs = 64
    for start in range(0, len(items), micro_bs):
        chunk = items[start:start + micro_bs]
        ids_list = [torch.tensor(x["ids"], dtype=torch.long) for x in chunk]
        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = torch.zeros_like(input_ids)
        for i, x in enumerate(chunk):
            attention_mask[i, :len(x["ids"])] = 1

        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        target = input_ids[:, 1:]
        opt_mask = attention_mask[:, 1:].clone().float()
        for i, x in enumerate(chunk):
            keep_from = max(x["plen"] - 1, 0)
            opt_mask[i, :keep_from] = 0.0
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1) * opt_mask
        opt_len = opt_mask.sum(-1).clamp(min=1)
        sum_lp = tok_lp.sum(-1).detach().cpu()

        hidden = out.hidden_states[mid_layer][:, :-1, :].float()
        pooled = ((hidden * opt_mask.unsqueeze(-1)).sum(1) / opt_len.unsqueeze(-1)).detach().cpu()

        for i, x in enumerate(chunk):
            key = (x["fi"], x["lang"])
            # Byte normalization = evaluate_accuracy.py's `byte` mode
            # (lm-eval acc_bytes): logprob sum / UTF-8 byte length.
            byte_score = float(sum_lp[i].item()) / x["nbytes"]
            scores.setdefault(key, [0.0] * RANKC_N_OPT)[x["oi"]] = byte_score
            embeds.setdefault(key, [None] * RANKC_N_OPT)[x["oi"]] = pooled[i]

        del out, logits, target, opt_mask, logp, tok_lp, hidden, pooled
        torch.cuda.empty_cache()

    rc_sum_all = rc_n_all = 0.0
    rc_sum_enx = rc_n_enx = 0.0
    rcx_sum_all = rcx_n_all = 0.0
    rcx_sum_enx = rcx_n_enx = 0.0
    align_agree_hits = align_agree_total = 0
    mcq_correct = mcq_slots = 0
    mcq_all_correct = mcq_complete_facts = 0
    mcq_per_lang_correct = {lang: 0 for lang in LANGS}
    mcq_per_lang_total = {lang: 0 for lang in LANGS}

    for fi, (prompts_by_lang, meta_by_lang, langs) in enumerate(facts):
        # ── Log-likelihood MCQ accuracy + total consistency (free: same scores)
        fact_langs_scored = 0
        fact_all_correct = True
        for lang in langs:
            key = (fi, lang)
            if key not in scores:
                continue
            gold_idx = LETTER_TO_IDX[meta_by_lang[lang]["gold_letter"]]
            pred_idx = max(range(RANKC_N_OPT), key=lambda k: scores[key][k])
            hit = int(pred_idx == gold_idx)
            mcq_correct += hit
            mcq_slots += 1
            mcq_per_lang_correct[lang] += hit
            mcq_per_lang_total[lang] += 1
            fact_langs_scored += 1
            if not hit:
                fact_all_correct = False
        # Only facts present in ALL languages can count toward total consistency,
        # matching evaluate_crosslingual_consistency.py's n_complete denominator.
        if fact_langs_scored == len(LANGS):
            mcq_complete_facts += 1
            mcq_all_correct += int(fact_all_correct)

        en_key = (fi, "en")
        if en_key not in embeds:
            continue
        en_emb = F.normalize(torch.stack(embeds[en_key]), dim=-1)
        en_order = sorted(range(RANKC_N_OPT), key=lambda i: scores[en_key][i], reverse=True)
        en_ids = meta_by_lang.get("en", {}).get("option_ids")

        slot_rank = {"en": en_order}          # hidden-state alignment (legacy keys)
        slot_rank_exact = {"en": en_order}    # option_ids alignment (exact keys)
        for lang in langs:
            if lang == "en":
                continue
            key = (fi, lang)
            if key not in embeds:
                continue
            emb = F.normalize(torch.stack(embeds[key]), dim=-1)
            sim = (emb @ en_emb.T).numpy()
            align = _greedy_align(sim)
            order = sorted(range(RANKC_N_OPT), key=lambda i: scores[key][i], reverse=True)
            slot_rank[lang] = [align[i] for i in order]

            # EXACT alignment via Wikidata QIDs (PolyFact-Clean stores them for
            # every option incl. distractors) — model-independent, so this
            # RankC cannot drift because the hidden-state matcher drifts.
            lang_ids = meta_by_lang.get(lang, {}).get("option_ids")
            if en_ids and lang_ids and set(en_ids) == set(lang_ids):
                en_pos = {qid: k for k, qid in enumerate(en_ids)}
                align_exact = [en_pos[qid] for qid in lang_ids]
                slot_rank_exact[lang] = [align_exact[i] for i in order]
                # Diagnostic: how often does the inferred matcher agree with
                # ground truth? Declining agreement = alignment drift, the
                # confound that makes the legacy rankc_avg ambiguous.
                align_agree_hits += sum(int(a == b) for a, b in zip(align, align_exact))
                align_agree_total += RANKC_N_OPT

        for a, b in itertools.combinations(slot_rank.keys(), 2):
            rc = rankc_pair_polyfact(slot_rank[a], slot_rank[b])
            rc_sum_all += rc
            rc_n_all += 1
            if a == "en" or b == "en":
                rc_sum_enx += rc
                rc_n_enx += 1
        for a, b in itertools.combinations(slot_rank_exact.keys(), 2):
            rc = rankc_pair_polyfact(slot_rank_exact[a], slot_rank_exact[b])
            rcx_sum_all += rc
            rcx_n_all += 1
            if a == "en" or b == "en":
                rcx_sum_enx += rc
                rcx_n_enx += 1

    if was_training:
        model.train()

    out_metrics = {
        "consistency/rankc_avg": rc_sum_all / max(rc_n_all, 1),
        "consistency/rankc_avg_en_x": rc_sum_enx / max(rc_n_enx, 1),
        # Exact-QID-aligned counterparts (PolyFact-Clean only) — the
        # trustworthy in-loop consistency signal; the pair above keeps the
        # legacy hidden-state alignment for curve continuity in older runs.
        "consistency/rankc_exact_avg": rcx_sum_all / max(rcx_n_all, 1),
        "consistency/rankc_exact_avg_en_x": rcx_sum_enx / max(rcx_n_enx, 1),
        "consistency/alignment_agreement": (
            align_agree_hits / align_agree_total if align_agree_total else float("nan")),
        # Log-likelihood MCQ accuracy under the eval prompt + byte
        # normalization: the in-loop counterpart of evaluate_accuracy.py.
        "polyfact/mcq_accuracy": mcq_correct / max(mcq_slots, 1),
        "polyfact/mcq_total_consistency": mcq_all_correct / max(mcq_complete_facts, 1),
        "polyfact/mcq_n_complete_facts": float(mcq_complete_facts),
    }
    for lang in LANGS:
        out_metrics[f"polyfact/mcq_lang_acc_{lang}"] = (
            mcq_per_lang_correct[lang] / mcq_per_lang_total[lang]
            if mcq_per_lang_total[lang] else 0.0
        )
    return out_metrics


# ─────────────────────────────────────────────
# Full evaluation (runs only on main process)
# ─────────────────────────────────────────────

@torch.no_grad()
def run_full_eval(
    model, tokenizer, wikifact_val_ds, flores_eval_sets, mmlu_eval_sets,
    max_prompt_length, max_completion_length, device, global_step,
    cache_implementation="static", mmlu_sample_ids=None,
):
    metrics = {}

    # WikiFact
    metrics.update(evaluate_polyfact_freeform(
        model=model, tokenizer=tokenizer, eval_ds=wikifact_val_ds,
        max_prompt_length=max_prompt_length, max_completion_length=max_completion_length,
        cache_implementation=cache_implementation,
    ))
    metrics.update(compute_polyfact_logprob_metrics(
        model=model, tokenizer=tokenizer, eval_ds=wikifact_val_ds,
        max_prompt_length=max_prompt_length, device=device,
    ))

    # FLORES BLEU + hidden cosine (run every eval)
    metrics.update(compute_flores_bleu(
        model=model, tokenizer=tokenizer, flores_sets=flores_eval_sets,
        device=device, max_new_tokens=128, batch_size=4,
    ))

    # Hidden cosine is slower, keep it less frequent
    if global_step % 1000 == 0:
        metrics.update(compute_flores_hidden_cosine(
            model=model, tokenizer=tokenizer, flores_sets=flores_eval_sets,
            device=device, batch_size=1,
        ))

    # Global MMLU per language
    model.eval()
    mmlu_sample_ids = mmlu_sample_ids or []
    mmlu_rank: Dict[Tuple[Any, str], List[int]] = {}
    mmlu_hit: Dict[Tuple[Any, str], bool] = {}
    choice_ids = [
        tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
    ]

    for lang, ds in mmlu_eval_sets.items():
        def mmlu_collate_fn(batch):
            return {
                "input_ids": pad_sequence(
                    [torch.tensor(x["input_ids"]) for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id,
                ),
                "attention_mask": pad_sequence(
                    [torch.tensor(x["attention_mask"]) for x in batch], batch_first=True, padding_value=0,
                ),
                "labels": pad_sequence(
                    [torch.tensor(x["labels"]) for x in batch], batch_first=True, padding_value=-100,
                ),
            }

        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=mmlu_collate_fn)
        correct = total = 0
        row_base = 0                           # running row offset, batch-size agnostic
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            n_rows = labels.shape[0]
            for i in range(n_rows):
                row_idx = row_base + i         # shuffle=False -> indexes mmlu_sample_ids
                label_row = labels[i]
                label_pos = (label_row != -100).nonzero(as_tuple=True)[0]
                if len(label_pos) == 0:
                    continue
                j = int(label_pos[0])
                gold_token = int(label_row[j].item())
                if gold_token not in choice_ids:
                    continue
                gold_idx = choice_ids.index(gold_token)
                letter_logits = logits[i, j - 1, choice_ids]
                pred_idx = int(letter_logits.argmax().item())
                correct += int(pred_idx == gold_idx)
                total += 1

                # Same forward pass, nothing extra computed: keep the FULL
                # ranking over the 4 letters so total consistency and RankC come
                # for free. Global-MMLU options are parallel by index across
                # languages, so letter slots already correspond — no alignment.
                if row_idx < len(mmlu_sample_ids):
                    sid = mmlu_sample_ids[row_idx]
                    order = sorted(range(4), key=lambda k: float(letter_logits[k].item()),
                                   reverse=True)
                    mmlu_rank[(sid, lang)] = order
                    mmlu_hit[(sid, lang)] = bool(pred_idx == gold_idx)
            row_base += n_rows
        metrics[f"mmlu/acc_{lang}"] = correct / total if total else 0.0

    if any(f"mmlu/acc_{lang}" in metrics for lang in LANGS):
        metrics["mmlu/acc_avg"] = float(np.mean([
            metrics[f"mmlu/acc_{lang}"] for lang in LANGS if f"mmlu/acc_{lang}" in metrics
        ]))

    # ── Global-MMLU cross-lingual metrics (free: reuses the rankings above) ──
    scored_langs = [l for l in mmlu_eval_sets.keys()]
    if mmlu_sample_ids and len(scored_langs) >= 2:
        n_complete = n_all_correct = 0
        rc_sum = rc_n = 0.0
        rc_sum_enx = rc_n_enx = 0.0
        for sid in mmlu_sample_ids:
            present = [l for l in scored_langs if (sid, l) in mmlu_rank]
            if len(present) != len(scored_langs):
                continue                      # only fully-parallel items count
            n_complete += 1
            n_all_correct += int(all(mmlu_hit[(sid, l)] for l in present))
            for a, b in itertools.combinations(present, 2):
                rc = rankc_pair_polyfact(mmlu_rank[(sid, a)], mmlu_rank[(sid, b)])
                rc_sum += rc
                rc_n += 1
                if a == "en" or b == "en":
                    rc_sum_enx += rc
                    rc_n_enx += 1
        metrics["mmlu/total_consistency"] = n_all_correct / max(n_complete, 1)
        metrics["mmlu/rankc_avg"] = rc_sum / max(rc_n, 1)
        metrics["mmlu/rankc_avg_en_x"] = rc_sum_enx / max(rc_n_enx, 1)
        metrics["mmlu/n_complete_items"] = float(n_complete)

    return metrics


# ─────────────────────────────────────────────
# Rollout generation
# ─────────────────────────────────────────────

def gather_rollout_prompts(
    batch: Dict[str, List[Any]], num_generations: int,
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
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
    model, tokenizer, batch, num_generations,
    max_prompt_length, max_completion_length, temperature, top_p,
    gen_micro_batch_size=4, cache_implementation="static",
    length_bucketing=False,
):
    """Generate rollouts using unwrapped model, micro-batched to avoid OOM.

    gen_micro_batch_size: how many prompts to generate at once.
    Default 4 — conservative for DDP where each GPU already holds a full
    model copy + optimizer states.  Increase if you have headroom.

    cache_implementation: passed straight to model.generate(). "static"
    preallocates the KV cache once per call rather than reallocating it every
    decode step (the DynamicCache default), which roughly halves rollout
    generation time at this batch-96/short-completion regime on GH200 while
    producing byte-identical greedy output. Pass None for the dynamic cache.
    """
    device = next(model.parameters()).device
    flat_prompts, flat_index = gather_rollout_prompts(batch, num_generations)

    if length_bucketing and len(flat_prompts) > gen_micro_batch_size:
        # Sort prompts by tokenized length so each generate() chunk pads to its
        # own maximum instead of the global one. The languages differ wildly in
        # tokenizer fertility (bn ~307 tokens vs en ~73 for the same fact), so
        # unsorted chunks nearly always contain a Bengali prompt and pad
        # everything ~4x. Results are keyed by (fact_idx, gen_idx, lang), so
        # processing order is irrelevant. No-op when everything fits one chunk.
        tok_lens = [len(ids) for ids in tokenizer(flat_prompts)["input_ids"]]
        order = sorted(range(len(flat_prompts)), key=lambda i: tok_lens[i])
        flat_prompts = [flat_prompts[i] for i in order]
        flat_index = [flat_index[i] for i in order]

    was_training = model.training
    model.eval()

    gen_kwargs = dict(
        do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_completion_length, pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.3, eos_token_id=tokenizer.eos_token_id,
    )
    if cache_implementation:
        gen_kwargs["cache_implementation"] = cache_implementation

    results = {}
    seq_payloads = []

    for chunk_start in range(0, len(flat_prompts), gen_micro_batch_size):
        chunk_end = min(chunk_start + gen_micro_batch_size, len(flat_prompts))
        chunk_prompts = flat_prompts[chunk_start:chunk_end]
        chunk_index = flat_index[chunk_start:chunk_end]

        inputs = tokenizer(
            chunk_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_length,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(**inputs, **gen_kwargs)

        for i, (fact_idx, gen_idx, lang) in enumerate(chunk_index):
            generated_ids = outputs[i][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            key = (fact_idx, gen_idx)
            results.setdefault(key, {})
            results[key][lang] = extract_answer_text(generated_text)

            # Non-pad token count in the completion — HF generate right-pads every
            # sequence in a micro-batch to the longest one, so this (not
            # total_len - input_len) is the actual rollout cost for this sequence.
            num_generated_tokens = int((generated_ids != tokenizer.pad_token_id).sum().item())

            seq_payloads.append({
                "fact_idx": fact_idx,
                "gen_idx": gen_idx,
                "lang": lang,
                "input_ids": outputs[i].detach().cpu(),
                "input_len": input_len,
                "total_len": int(outputs[i].shape[0]),
                "generated_text": generated_text,
                "num_generated_tokens": num_generated_tokens,
            })

        # Free GPU memory between micro-batches
        del inputs, outputs
        torch.cuda.empty_cache()

    if was_training:
        model.train()

    return results, seq_payloads


# ─────────────────────────────────────────────
# Policy gradient loss
# ─────────────────────────────────────────────

def compute_logprob_loss(
    model, ref_model, seq_payloads, group_rewards, kl_coef, pad_token_id,
    device, micro_batch_size=4, length_bucketing=False, fused=False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not seq_payloads:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"mean_reward": 0.0, "reward_std": 0.0, "mean_advantage": 0.0, "mean_kl": 0.0}

    if length_bucketing:
        # Group similar-length sequences into the same micro-batch so each one
        # pads to its own max, not the global (Bengali-dominated) max. The loss
        # is a mean over sequences, so processing order does not change it. The
        # peak-memory term is 2*[mb, T, vocab] per RETAINED micro-batch graph,
        # linear in the padded T — unsorted, every micro-batch pads to ~445
        # tokens when the true mean is ~164.
        seq_payloads = sorted(seq_payloads, key=lambda x: x["total_len"])

    def _token_logprobs(lgts, tgt):
        """Per-token logprob of `tgt` under `lgts`.

        fused=True uses F.cross_entropy, which computes log_softmax internally
        WITHOUT materialising (or retaining for backward) the full-vocab
        [mb, T, vocab] logprob tensor that the explicit log_softmax+gather
        keeps alive per retained graph — that tensor is the single biggest
        memory term in this trainer. Same math, one fewer vocab-sized copy.
        """
        if fused:
            B_, Tm1, V_ = lgts.shape
            return -F.cross_entropy(
                lgts.reshape(-1, V_), tgt.reshape(-1), reduction="none",
            ).reshape(B_, Tm1)
        lp = torch.log_softmax(lgts, dim=-1)
        return torch.gather(lp, -1, tgt.unsqueeze(-1)).squeeze(-1)

    losses, rewards, advantages, kls = [], [], [], []

    for chunk_start in range(0, len(seq_payloads), micro_batch_size):
        chunk = seq_payloads[chunk_start:chunk_start + micro_batch_size]

        input_ids = pad_sequence(
            [x["input_ids"] for x in chunk], batch_first=True, padding_value=pad_token_id,
        ).to(device)

        lengths = [x["total_len"] for x in chunk]
        attention_mask = torch.zeros_like(input_ids)
        for i, l in enumerate(lengths):
            attention_mask[i, :l] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_logprobs = _token_logprobs(logits, target_ids)

        with torch.no_grad():
            if ref_model is not None and kl_coef > 0.0:
                if isinstance(ref_model, str) and ref_model == "adapter_off":
                    # The reference distribution is the base weights: the
                    # policy with its (zero-init at t=0) LoRA adapter turned
                    # off. Identical to a separately-loaded copy of model_id,
                    # without the +15 GB. no-grad forward, so bypassing the
                    # DDP wrapper is safe (nothing to sync).
                    unwrapped = model.module if hasattr(model, "module") else model
                    with unwrapped.disable_adapter():
                        ref_outputs = unwrapped(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_token_logprobs = _token_logprobs(ref_logits, target_ids)
            else:
                ref_token_logprobs = None

        for i, payload in enumerate(chunk):
            fact_idx = payload["fact_idx"]
            gen_idx = payload["gen_idx"]
            input_len = payload["input_len"]
            total_len = payload["total_len"]

            gr = group_rewards.get((fact_idx, gen_idx, payload["lang"]))
            if gr is None:
                continue
            reward = gr["reward"]
            advantage = gr["advantage"]

            start = max(input_len - 1, 0)
            end = total_len - 1
            if end <= start:
                continue

            gen_logprob_mean = token_logprobs[i, start:end].mean()
            seq_loss = -advantage * gen_logprob_mean

            if ref_token_logprobs is not None:
                # Schulman k3 estimator: r = log(pi_ref / pi); KL ~= exp(r) - r - 1.
                # Provably >= 0 (since e^r >= 1 + r), unbiased for KL, and far lower
                # variance than the naive k1 estimator (logpi - logpi_ref) it replaces.
                #
                # k1 was ANTI-REGULARISING here, not merely noisy. Minimising
                # kl_coef * (logpi - logpi_ref) just pushes logpi DOWN on the policy's
                # own samples; there is no attractor toward pi_ref. Since the samples
                # come from pi, each step makes the estimate more negative and the
                # gradient keeps pushing the same way. Measured on 1,500-step sweep
                # arms: KL ran to -2.44 (coef 0.02) and -8.85 (coef 0.05) within 50
                # steps, scaling with the coefficient — i.e. the "penalty" had become
                # a bonus for diverging from the reference.
                #
                # The clamp only guards exp() overflow; |r| that large means the
                # policy has already collapsed and the run is lost either way.
                log_ratio = (ref_token_logprobs[i, start:end]
                             - token_logprobs[i, start:end]).clamp(-10.0, 10.0)
                seq_kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()
                seq_loss = seq_loss + kl_coef * seq_kl
                kls.append(float(seq_kl.detach().item()))

            losses.append(seq_loss)
            rewards.append(reward)
            advantages.append(advantage)

    if not losses:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"mean_reward": 0.0, "reward_std": 0.0, "mean_advantage": 0.0, "mean_kl": 0.0}

    loss = torch.stack(losses).mean()
    stats = {
        "mean_reward": float(torch.tensor(rewards).mean().item()),
        "reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if len(rewards) > 1 else 0.0,
        "mean_advantage": float(torch.tensor(advantages).mean().item()),
        "mean_kl": float(torch.tensor(kls).mean().item()) if kls else 0.0,
    }
    return loss, stats


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    if args.no_bf16:
        args.bf16 = False

    # "none"/"" -> None (dynamic cache); anything else passed to generate() as-is.
    gen_cache_impl = args.gen_cache_implementation
    if gen_cache_impl in (None, "", "none", "None"):
        gen_cache_impl = None

    # ── Accelerator ──────────────────────────
    # Increase NCCL timeout to handle long eval periods where only rank 0 is active.
    from datetime import timedelta
    from accelerate import InitProcessGroupKwargs

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
        log_with="wandb" if args.report_to == "wandb" else None,
        kwargs_handlers=[process_group_kwargs],
    )

    accelerate_set_seed(args.seed)
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.report_to == "wandb" and is_main:
        wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args))

    # ── Tokenizer ────────────────────────────
    if is_main:
        print(f"Loading tokenizer for {args.model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32

    # ── Eval data (load on all processes — needed for tokenizer-dependent preprocessing) ──
    if is_main:
        print("Loading eval datasets ...", flush=True)

    # FLORES-200 is a script-based dataset; datasets>=4 dropped script loading, so
    # this raises on newer stacks. It only feeds the auxiliary BLEU/hidden-cosine
    # eval, so degrade gracefully to {} (those eval fns no-op on an empty set)
    # rather than kill training.
    try:
        flores_eval_sets = load_flores_parallel_subset(
            target_langs=["ar", "bn", "de", "es", "fr", "id", "ja", "pt", "ru", "sw", "zh"],
            split="dev", max_samples=args.max_eval_flores,
        )
    except Exception as e:
        if is_main:
            print(f"[warn] FLORES eval disabled (could not load: {e})", flush=True)
        flores_eval_sets = {}
    try:
        mmlu_eval_sets, mmlu_sample_ids = load_global_mmlu_dev_eval_by_lang(
            LANGS, tokenizer, max_samples=args.max_eval_mmlu)
    except Exception as e:
        if is_main:
            print(f"[warn] Global-MMLU eval disabled (could not load: {e})", flush=True)
        mmlu_eval_sets, mmlu_sample_ids = {}, []

    # ── Model ────────────────────────────────
    if is_main:
        print(f"Loading model {args.model_id} ...", flush=True)

    # Load on CPU first, then let Accelerate distribute
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype,
    )

    if args.use_lora:
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #
            lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, peft_config)
        if is_main:
            model.print_trainable_parameters()
    else:
        model = base_model

    model.gradient_checkpointing_enable()
    if args.use_lora:
        model.enable_input_require_grads()
    model.train()

    # ── Reference model for KL (only if needed) ──
    ref_model = None
    if args.kl_coef > 0.0:
        if args.ref_impl == "adapter_off":
            if not args.use_lora:
                raise ValueError("--ref_impl adapter_off requires --use_lora "
                                 "(the reference is the adapter-disabled policy).")
            if is_main:
                print("KL reference: policy with LoRA adapter disabled "
                      "(no second model).", flush=True)
            ref_model = "adapter_off"
        else:
            if is_main:
                print("Loading reference model for KL ...", flush=True)
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model_id, dtype=dtype,
            ).to(accelerator.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

    # ── Dataset ──────────────────────────────
    if is_main:
        print(f"Loading dataset {args.dataset_id} ...", flush=True)

    raw_all = pfs.load_split_dict(args.dataset_id, args.dataset_config)
    raw_train = raw_all["train"]
    raw_val = raw_all["validation"]

    if is_main:
        print("KL coef:", args.kl_coef)
        print("Using LoRA:", args.use_lora)
        print("Num processes:", accelerator.num_processes)

    # NOTE: fn_kwargs are part of the datasets .map fingerprint, so changing
    # --task_format (like --prompt_scaffold) forces one cache recompute.
    _map_kw = {"scaffold": args.prompt_scaffold, "task_format": args.task_format}
    train_ds = raw_train.map(build_grouped_fact_item, num_proc=32, fn_kwargs=_map_kw)
    val_ds = raw_val.map(build_grouped_fact_item, fn_kwargs=_map_kw)

    train_ds = train_ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= args.min_languages)
    val_ds = val_ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= args.min_languages)

    if args.max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))

    keep_cols = ["fact_id", "prompts_by_lang_json", "meta_by_lang_json"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

    if args.max_eval_wikifact is not None:
        val_ds = val_ds.select(range(min(args.max_eval_wikifact, len(val_ds))))

    if is_main:
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    # Explicit seeded generator: with shuffle=True and no generator the
    # permutation is drawn from the GLOBAL RNG, which sampling-based rollout
    # generation advances by an unpredictable amount before the first batch — so
    # the epoch order would NOT be reproducible across a relaunch and the
    # skip-forward resume below could not land on the same data. Seeding here
    # makes the order a pure function of args.seed.
    order_generator = torch.Generator()
    order_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.per_device_train_batch_size,
        shuffle=True, collate_fn=collate_fn, generator=order_generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    # ── Accelerate prepare ───────────────────
    # NOTE: We prepare model, optimizer, train_loader, scheduler together.
    # This wraps the model in DDP, shards the dataloader, etc.
    # We create a dummy scheduler first, then recreate it with the correct step count.
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader,
    )

    # Compute total steps AFTER prepare so len(train_loader) reflects sharding
    total_update_steps = math.ceil(
        len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps
    )
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    # `total_update_steps` counts OPTIMIZER steps on this rank (len(train_loader)
    # is the already-sharded length). accelerator.prepare() wraps the scheduler in
    # AcceleratedScheduler, whose .step() calls the inner scheduler num_processes
    # times — it assumes the schedule was built against the UN-sharded step count.
    # Both corrections together consume the cosine num_processes x too fast: on a
    # 4-GPU run the lr hit exactly 0.0 at step 2500 of 10000 and stayed there, so
    # 75% of the run trained with no learning at all. Scale the schedule up so the
    # double-stepping lands the cosine minimum on the real final step.
    sched_scale = max(1, accelerator.num_processes)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps * sched_scale,
        num_training_steps=total_update_steps * sched_scale,
    )
    scheduler = accelerator.prepare(scheduler)

    device = accelerator.device
    global_step = 0
    reward_ema = None
    consecutive_zero_std = 0   # dead-run guard; see --dead_run_patience
    ema_alpha = 0.05

    # Cost-reporting counters (rollout tokens, wall-clock -> GPU-hours). "prior_*"
    # carries totals from before this process started (restored from a checkpoint
    # sidecar on resume), so a crash+relaunch doesn't reset the running cost figures
    # reported for the paper's training/token-efficiency section.
    cumulative_rollout_tokens = 0
    prior_wall_seconds = 0.0

    # ── Resume from checkpoint (model, optimizer, scheduler, RNG) ──
    # Older checkpoints (pre-accelerator.save_state) only have adapter/tokenizer
    # files, not accelerate's own state artifacts (pytorch_model.bin etc.) — fall
    # back to a fresh start rather than crashing every rank if load_state can't
    # find what it expects.
    resume_dir = None
    if args.resume_from_checkpoint == "auto":
        resume_dir = find_latest_checkpoint(args.output_dir)
    elif args.resume_from_checkpoint not in (None, "none"):
        resume_dir = args.resume_from_checkpoint
    if resume_dir is not None:
        step_from_dir = int(os.path.basename(resume_dir).split("-")[-1])
        try:
            accelerator.load_state(resume_dir)
            global_step = step_from_dir
            prior_stats = load_training_stats(resume_dir)
            cumulative_rollout_tokens = prior_stats.get("cumulative_rollout_tokens", 0)
            prior_wall_seconds = prior_stats.get("cumulative_wall_seconds", 0.0)
            if is_main:
                print(f"Resumed from checkpoint {resume_dir} (global_step={global_step}, "
                      f"prior_rollout_tokens={cumulative_rollout_tokens}, "
                      f"prior_wall_hours={prior_wall_seconds / 3600:.2f}) ...", flush=True)
        except Exception as e:
            if is_main:
                print(f"[warn] could not resume from {resume_dir} ({e!r}); "
                      f"starting fresh from global_step=0.", flush=True)
            accelerator.wait_for_everyone()

    if is_main and args.reward_pooling == "per_lang" and args.all_correct_bonus != 0.0:
        print(f"[warn] --reward_pooling per_lang ignores --all_correct_bonus "
              f"({args.all_correct_bonus}); the bonus is undefined without "
              f"cross-lingual pooling.", flush=True)

    bonus_ladder = parse_bonus_ladder(args.bonus_ladder)
    if is_main and args.bonus_shape == "ladder" and args.all_correct_bonus not in (0.0, 1.0):
        print(f"[warn] --bonus_shape ladder ignores --all_correct_bonus "
              f"({args.all_correct_bonus}); ladder values are absolute rewards.", flush=True)
    if is_main and args.bonus_shape != "all_or_nothing":
        # Print the realized reward curve so the shape is auditable in the log
        # rather than inferred from the flag names.
        n_l = 12
        curve = " ".join(
            f"k={k}:{consistency_bonus(k, n_l, args.all_correct_bonus, args.bonus_shape, args.bonus_power, bonus_ladder):.2f}"
            for k in range(0, n_l + 1, 2 if args.bonus_shape == "power" else 1)
        )
        print(f"[bonus] shape={args.bonus_shape} B={args.all_correct_bonus} "
              f"power={args.bonus_power} ladder={bonus_ladder}\n[bonus] curve (n=12): {curve}",
              flush=True)

    if is_main:
        print(f"Total update steps: {total_update_steps}", flush=True)
        print("Starting grouped-rollout training ...", flush=True)

    train_start_time = time.time()

    # Per-optimizer-step wall time and allocator peak, for throughput/batch-size
    # tuning and the paper's cost table. `recent_step_times` collects every
    # optimizer step and is drained at each logging boundary, so the logged
    # value is a mean over the last --logging_steps steps rather than a single
    # noisy sample or a startup-polluted cumulative average.
    last_opt_step_time = time.time()
    recent_step_times: List[float] = []
    torch.cuda.reset_peak_memory_stats()

    # ── Resume-exact data ordering ───────────
    # `global_step` is restored from the checkpoint, but without the skip below
    # the dataloader restarts at batch 0 of a freshly reshuffled epoch, so a
    # resumed run re-walks facts it already trained on (measured over a 3-chunk
    # chain: only ~75% of the fact pool ever seen). set_epoch makes the shuffle a
    # pure function of (seed, epoch) so a relaunch reconstructs the identical
    # permutation, and skip_first_batches then fast-forwards to the exact batch
    # the run stopped on.
    steps_per_epoch = max(1, len(train_loader))
    start_epoch = global_step // steps_per_epoch
    skip_batches = global_step % steps_per_epoch
    if is_main and global_step:
        print(f"Resume-exact ordering: starting at epoch {start_epoch}, "
              f"skipping {skip_batches}/{steps_per_epoch} batches.", flush=True)

    for epoch in range(start_epoch, math.ceil(args.num_train_epochs)):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        epoch_loader = train_loader
        step_offset = 0
        if epoch == start_epoch and skip_batches:
            epoch_loader = accelerator.skip_first_batches(train_loader, skip_batches)
            step_offset = skip_batches
        for step, batch in enumerate(epoch_loader, start=step_offset):

            # ── Generation (unwrapped model, no DDP) ──
            # We unwrap so .generate() works without DDP issues.
            # Clear cache first to reclaim memory from DDP gradient buffers.
            torch.cuda.empty_cache()
            unwrapped_model = accelerator.unwrap_model(model)

            grouped_preds, seq_payloads = generate_grouped_rollouts(
                model=unwrapped_model,
                tokenizer=tokenizer,
                batch=batch,
                num_generations=args.num_generations,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                temperature=args.temperature,
                top_p=args.top_p,
                gen_micro_batch_size=args.gen_micro_batch_size,
                cache_implementation=gen_cache_impl,
                length_bucketing=args.length_bucketing,
            )

            # Rollout token cost, summed across ranks (each rank generates a
            # different shard of the batch under DDP) — accumulated every
            # micro-step, not just on optimizer-sync steps, since every rollout
            # burns compute regardless of gradient accumulation.
            step_tokens = sum(p["num_generated_tokens"] for p in seq_payloads)
            step_tokens_total = accelerator.reduce(
                torch.tensor(step_tokens, device=device, dtype=torch.long), reduction="sum",
            ).item()
            cumulative_rollout_tokens += step_tokens_total

            # ── Rewards & advantages ──
            group_rewards, group_stats = compute_group_advantages(
                batch=batch,
                grouped_preds=grouped_preds,
                num_generations=args.num_generations,
                coverage_weight=args.coverage_reward_weight,
                valid_option_weight=args.valid_option_reward_weight,
                all_correct_bonus=args.all_correct_bonus,
                reward_pooling=args.reward_pooling,
                bonus_shape=args.bonus_shape,
                bonus_power=args.bonus_power,
                bonus_ladder=bonus_ladder,
                brevity_penalty=args.brevity_penalty,
                brevity_denom_floor=args.brevity_denom_floor,
                empty_penalty=args.empty_penalty,
            )

            # ── Policy gradient loss (through the DDP-wrapped model) ──
            loss, loss_stats = compute_logprob_loss(
                model=model,  # use the wrapped model so gradients sync
                ref_model=ref_model,
                seq_payloads=seq_payloads,
                group_rewards=group_rewards,
                kl_coef=args.kl_coef,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                micro_batch_size=args.logprob_micro_batch_size,
                length_bucketing=args.length_bucketing,
                fused=args.fused_logprob,
            )

            # ── Backward (Accelerate handles scaling + DDP sync) ──
            accelerator.backward(loss / args.gradient_accumulation_steps)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                _now = time.time()
                recent_step_times.append(_now - last_opt_step_time)
                last_opt_step_time = _now

                # ── Dead-run guard ──────────────────────────────────────────
                # A step whose rollouts all score alike has std 0, so every
                # advantage z-scores to 0 and the step contributes EXACTLY zero
                # gradient. One such step is normal (a saturated all-correct
                # group). Hundreds in a row means the policy has entered an
                # absorbing state — every rollout identical, no gradient able to
                # leave it — and the remaining wall-clock is pure waste. Both
                # observed absorbing states are all-12-languages-identical:
                # empty output (reward 0.0) and unparseable output (-6.0).
                if args.dead_run_patience > 0:
                    _r = [v["reward"] for v in group_rewards.values()]
                    if _r and max(_r) == min(_r):
                        consecutive_zero_std += 1
                    else:
                        consecutive_zero_std = 0
                    if consecutive_zero_std >= args.dead_run_patience:
                        raise RuntimeError(
                            f"dead run: {consecutive_zero_std} consecutive optimizer steps "
                            f"with reward_std == 0 (reward {_r[0] if _r else float('nan'):.2f}) "
                            f"at step {global_step}. Every rollout is identical, so the "
                            f"gradient is zero and cannot recover. Restart from an earlier "
                            f"checkpoint; if the rollouts are EMPTY, pass --empty_penalty 1.0."
                        )

                # ── Logging (main process only) ──
                if is_main and global_step % args.logging_steps == 0:
                    rewards = [v["reward"] for v in group_rewards.values()]
                    advs = [v["advantage"] for v in group_rewards.values()]

                    # group_stats is keyed per (fact, gen) — one entry per rollout,
                    # not per (rollout, language) like group_rewards — so average
                    # over it directly rather than over the 12x-replicated rewards.
                    _gs = list(group_stats.values())
                    _bonus_mean = float(sum(s.get("bonus", 0.0) for s in _gs) / len(_gs)) if _gs else 0.0
                    _ncorrect_mean = float(sum(s.get("n_correct", 0) for s in _gs) / len(_gs)) if _gs else 0.0

                    reward_mean = float(sum(rewards) / len(rewards)) if rewards else 0.0
                    if reward_ema is None:
                        reward_ema = reward_mean
                    else:
                        reward_ema = ema_alpha * reward_mean + (1 - ema_alpha) * reward_ema

                    elapsed = time.time() - train_start_time
                    wall_seconds = prior_wall_seconds + elapsed

                    # Mean seconds per optimizer step since the last log, and the
                    # same figure per fact (an optimizer step covers
                    # per_device_train_batch_size facts on each of num_processes
                    # ranks). facts/step is what actually scales the loss pass.
                    step_time = (
                        sum(recent_step_times) / len(recent_step_times)
                        if recent_step_times else 0.0
                    )
                    recent_step_times.clear()
                    facts_per_step = (
                        args.per_device_train_batch_size
                        * args.gradient_accumulation_steps
                        * accelerator.num_processes
                    )
                    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    torch.cuda.reset_peak_memory_stats()
                    gpu_hours = wall_seconds * accelerator.num_processes / 3600

                    log_data = {
                        "train/loss": float(loss.detach().item()),
                        "train/grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "train/reward_mean": reward_mean,
                        "train/reward_mean_ema": reward_ema,
                        "train/reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if len(rewards) > 1 else 0.0,
                        "train/adv_mean": float(sum(advs) / len(advs)) if advs else 0.0,
                        # How much of the reward the consistency bonus actually
                        # supplies. With all_or_nothing/B=1.0 this measured at
                        # 0.4-2.4% — i.e. the objective was ~98% a correctness
                        # count. Log it so the shaping is verifiable, not assumed.
                        "train/bonus_mean": _bonus_mean,
                        "train/bonus_share": (_bonus_mean / reward_mean) if reward_mean else 0.0,
                        "train/n_correct_mean": _ncorrect_mean,
                        "train/mean_kl": loss_stats.get("mean_kl", 0.0),
                        "train/global_step": global_step,
                        "cost/cumulative_rollout_tokens": cumulative_rollout_tokens,
                        "cost/wall_clock_hours": wall_seconds / 3600,
                        "cost/gpu_hours": gpu_hours,
                        "perf/step_time_sec": step_time,
                        "perf/sec_per_fact": step_time / facts_per_step if facts_per_step else 0.0,
                        "perf/peak_mem_gb": peak_mem_gb,
                    }

                    if args.report_to == "wandb":
                        wandb.log(log_data, step=global_step)

                    steps_per_sec = global_step / elapsed
                    remaining_steps = total_update_steps - global_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_h = int(eta_seconds // 3600)
                    eta_m = int((eta_seconds % 3600) // 60)

                    print({
                        "step": f"{global_step}/{total_update_steps}",
                        "loss": round(log_data["train/loss"], 4),
                        "lr": f"{log_data['train/learning_rate']:.3e}",
                        "reward_mean": round(reward_mean, 4),
                        "reward_std": round(log_data["train/reward_std"], 4),
                        "adv_mean": round(log_data["train/adv_mean"], 4),
                        # Pre-clip norm. clip_grad_norm_ is 1.0, so any value >1
                        # means this step was renormalized — i.e. the raw gradient
                        # scale (and the loss magnitude driving it) does NOT set
                        # the effective step size. Worth seeing in the slurm log.
                        "grad_norm": round(log_data["train/grad_norm"], 3),
                        "kl": round(log_data["train/mean_kl"], 6),
                        "it/s": round(steps_per_sec, 3),
                        "eta": f"{eta_h}h{eta_m:02d}m",
                        "rollout_tok": cumulative_rollout_tokens,
                        "gpu_h": round(gpu_hours, 2),
                        "step_s": round(step_time, 3),
                        "s_per_fact": round(step_time / facts_per_step, 3) if facts_per_step else 0.0,
                        "peak_mem_gb": round(peak_mem_gb, 2),
                    }, flush=True)

                    # Sample rollout
                    sample_key = sorted(grouped_preds.keys())[0]
                    sfidx, sgidx = sample_key
                    print(f"\n[sample rollout @ step {global_step}] fact_idx={sfidx} gen_idx={sgidx}", flush=True)
                    for lang in LANGS:
                        if lang in grouped_preds[sample_key]:
                            print(f"  {lang}: {grouped_preds[sample_key][lang]}", flush=True)
                    _lang_grs = [group_rewards[k] for k in group_rewards
                                 if k[0] == sfidx and k[1] == sgidx]
                    if _lang_grs:
                        print(
                            f"  reward(mean over langs)="
                            f"{round(sum(g['reward'] for g in _lang_grs) / len(_lang_grs), 4)}  "
                            f"advantage(mean)="
                            f"{round(sum(g['advantage'] for g in _lang_grs) / len(_lang_grs), 4)}",
                            flush=True,
                        )

                    # Needs the same `> 0` guard as the eval block below:
                    # --eval_steps 0 (used to disable periodic eval entirely,
                    # e.g. in the cluster/*_pilot.sh throughput sweeps) would
                    # otherwise raise ZeroDivisionError on the first log step.
                    if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                        log_sample_rollout_to_file(
                            output_dir=args.output_dir,
                            global_step=global_step,
                            grouped_preds=grouped_preds,
                            group_rewards=group_rewards,
                            batch=batch,
                        )

                # ── Eval (rank 0 only, other ranks wait — protected by 120min NCCL timeout) ──
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if is_main:
                        torch.cuda.empty_cache()
                        eval_model = accelerator.unwrap_model(model)
                        if args.skip_periodic_eval:
                            print(
                                f"\n[checkpoint-only @ step {global_step}] "
                                "skipping periodic benchmark evaluation",
                                flush=True,
                            )
                        else:
                            eval_metrics = run_full_eval(
                                model=eval_model,
                                tokenizer=tokenizer,
                                wikifact_val_ds=val_ds,
                                flores_eval_sets=flores_eval_sets,
                                mmlu_eval_sets=mmlu_eval_sets,
                                mmlu_sample_ids=mmlu_sample_ids,
                                max_prompt_length=args.max_prompt_length,
                                max_completion_length=args.max_completion_length,
                                device=device,
                                global_step=global_step,
                                cache_implementation=gen_cache_impl,
                            )
                            print(f"\n[eval @ step {global_step}] {eval_metrics}", flush=True)
                            if args.report_to == "wandb":
                                wandb.log(eval_metrics, step=global_step)

                        torch.cuda.empty_cache()

                    # accelerator.save_state is a collective call (gathers/syncs across
                    # ranks) — must run on every process, not just rank 0, or non-main
                    # ranks hang waiting on the barrier it uses internally.
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)

                    if is_main:
                        # Plain adapter + tokenizer save for easy downstream loading/merging
                        # without needing accelerate's full-state loader.
                        eval_model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        save_training_stats(
                            checkpoint_dir, global_step, cumulative_rollout_tokens,
                            prior_wall_seconds + (time.time() - train_start_time),
                            accelerator.num_processes,
                        )
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir)
                             if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()],
                            key=lambda x: int(x.split("-")[-1]),
                        )
                        for old in checkpoints[:-3]:
                            shutil.rmtree(os.path.join(args.output_dir, old))

                    accelerator.wait_for_everyone()
                    model.train()
                    torch.cuda.empty_cache()

            if global_step >= total_update_steps:
                break
        if global_step >= total_update_steps:
            break

    # ── Save final model ─────────────────────
    accelerator.wait_for_everyone()
    if is_main:
        print(f"Saving model to {args.output_dir} ...", flush=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        final_wall_seconds = prior_wall_seconds + (time.time() - train_start_time)
        save_training_stats(
            args.output_dir, global_step, cumulative_rollout_tokens,
            final_wall_seconds, accelerator.num_processes,
        )
        print(
            f"Training cost: {cumulative_rollout_tokens:,} rollout tokens, "
            f"{final_wall_seconds / 3600:.2f}h wall-clock, "
            f"{final_wall_seconds * accelerator.num_processes / 3600:.2f} GPU-hours "
            f"({accelerator.num_processes} GPUs)", flush=True,
        )

        if args.report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    main()
