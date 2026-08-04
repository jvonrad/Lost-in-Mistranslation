#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TRL-based DAPO port of train_wikifact_grpo.py.

Conceptually identical to the bespoke grouped-rollout trainer:
  - one training "fact" = 12 single-language prompts
  - num_generations sampled rollouts per fact, each rollout = 12 completions (one
    per language) generated independently
  - one joint reward computed across the 12 completions of a rollout
  - reward normalized across rollouts of the same fact
  - that advantage applied to all 12 completions in the rollout

How it maps onto TRL's GRPOTrainer:
  - dataset is flattened to one row per (fact, lang) with columns
    {prompt, fact_id, lang, lang_index, gold_text}, laid out fact-major (rows
    [12k .. 12k+11] = the 12 languages of fact k)
  - a custom sampler shuffles at the fact level so the 12 lang rows of each
    fact stay contiguous in every generation batch
  - num_generations = N → trainer produces N completions per (fact, lang) row
  - the joint reward fn groups completions by (fact_id, gen_idx), computes one
    scalar joint reward, and replicates it to all 12 lang rows of that
    (fact, gen_idx) tuple
  - per-prompt advantage normalization across N then produces identical
    advantages for every language of the same (fact, gen_idx) — equivalent to
    the original grouped-rollout math

Loss: DAPO (token-level normalization + asymmetric clip-higher), via
  loss_type="dapo", epsilon_high=0.28, mask_truncated_completions=True.

Sizing constraint:
  per_device_train_batch_size × gradient_accumulation_steps × world_size
  must be a multiple of len(LANGS) × num_generations (= 48 for default
  num_generations=4) so every generation batch holds a whole number of
  (fact × num_generations) tuples. If --per_device_train_batch_size is left
  unset, it auto-fills to len(LANGS) × num_generations.

Run (vLLM server-mode example, single-GPU LoRA):

  CUDA_VISIBLE_DEVICES=1 FLASHINFER_DISABLE_VERSION_CHECK=1 trl vllm-serve \
    --model Qwen/Qwen2.5-7B \
    --port 8007 \
    --gpu_memory_utilization 0.75 

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 \
python training/train_polyfact_dapo.py \
    --model_id qwen2.5-7b \
    --dataset_id jvonrad/PolyFact \
    --num_generations 8 \
    --max_completion_length 32 \
    --learning_rate 5e-6 \
    --max_train_samples 20000 \
    --eval_steps 200 \
    --max_eval_flores 0 \
    --use_lora \
    --vllm_group_port 51218 \
    --use_vllm \
    --vllm_server_base_url http://localhost:8004 \
    --curriculum
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import GRPOConfig, GRPOTrainer

# Reuse helpers from the bespoke trainer so eval / formatting / correctness
# match exactly. Adding this dir to sys.path makes the script runnable as
# `python training/train_polyfact_dapo.py` without packaging.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wikifact_grpo import (  # noqa: E402
    INDEX_TO_LETTER,
    LANGS,
    LORA_ALPHA,
    LORA_R,
    MODEL_OUT_ROOT,
    OPEN_GENERATION_PROMPTS,
    WANDB_PROJECT,
    answer_text_to_letter,
    compute_flores_bleu,
    compute_flores_bleu_to_english,
    compute_flores_hidden_cosine,
    evaluate_wikifact_grouped,
    extract_answer_text,
    is_correct_text_match,
    load_flores_parallel_subset,
    load_global_mmlu_dev_eval_by_lang,
    resolve_model_id,
    resolve_prediction_to_letter,
    run_open_generation_probes,
    safe_strip,
)


# Mutable container shared between the dataset builder, reward function, and
# CurriculumCallback. Reward correctness logic and the active dataset both
# switch on this single source of truth.
CURRICULUM_PHASE: Dict[str, str] = {"phase": "closed_book"}
CURRICULUM_STATE_FILE = 'curriculum_state.json'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="allenai/OLMo-2-1124-7B")
    ap.add_argument("--dataset_id", type=str, default="jvonrad/PolyFact-Clean")
    ap.add_argument("--dataset_config", type=str, default="parallel")

    ap.add_argument("--use_lora", action="store_true", default=False)
    ap.add_argument("--use_vllm", action="store_true", default=False)
    ap.add_argument("--vllm_server_base_url", type=str, default=None,
                    help="Base URL of a running TRL vLLM server, e.g. http://localhost:8007.")
    ap.add_argument("--vllm_server_timeout", type=float, default=240.0,
                    help="Timeout in seconds for connecting to the TRL vLLM server.")
    ap.add_argument("--vllm_group_port", type=int, default=51216,
                    help="TCP port used by TRL for trainer-to-vLLM weight-sync communicator.")

    # Training hparams. The product
    #   per_device_train_batch_size × gradient_accumulation_steps × world_size
    # must be a multiple of `len(LANGS) × num_generations` so each generation
    # batch carries a whole number of (fact × num_generations) tuples and the
    # joint reward fn can group on (fact, gen_idx). If --per_device_train_batch_size
    # is left at its default (None), it auto-fills to len(LANGS) × num_generations
    # (one full fact tuple expanded per micro-batch).
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="Max number of FACTS (not rows) to train on.")

    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=48)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.5)

    # DAPO knobs.
    ap.add_argument("--loss_type", type=str, default="dapo",
                    choices=["grpo", "dapo", "dr_grpo", "bnpo", "sapo"])
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--epsilon_high", type=float, default=0.28,
                    help="DAPO clip-higher; recommended ~0.28.")
    ap.add_argument("--mask_truncated_completions", action="store_true", default=True,
                    help="DAPO overlong filtering: drop truncated completions from loss.")
    ap.add_argument("--scale_rewards", type=str, default="group",
                    choices=["group", "batch", "none"])
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0.0 disables ref model).")

    # Joint reward shaping.
    ap.add_argument("--all_correct_bonus", type=float, default=1.0)
    ap.add_argument("--min_languages", type=int, default=12,
                    help="Skip facts that don't cover this many languages.")

    # Curriculum: phase-1 MCQ → phase-2 closed-book, triggered by reward EMA.
    ap.add_argument("--curriculum", action="store_true", default=False,
                    help="Two-phase training: start with MCQ prompts (visible "
                         "options), auto-switch to closed-book once the "
                         "training reward EMA crosses --curriculum_threshold.")
    ap.add_argument("--curriculum_threshold", type=float, default=10,
                    help="Reward EMA threshold to trigger the MCQ→closed-book "
                         "switch. With 12 langs and an all-correct bonus, the "
                         "reward maxes out around 13; 6 ≈ half the langs right.")
    ap.add_argument("--curriculum_min_steps", type=int, default=50,
                    help="Minimum steps in phase 1 before the curriculum is "
                         "allowed to fire (debounces early-EMA noise).")
    ap.add_argument("--curriculum_alpha", type=float, default=0.1,
                    help="EMA smoothing factor for the curriculum trigger "
                         "(higher = more reactive, lower = more conservative).")

    # Eval cadence.
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200,
                    help="Checkpoint every N optimizer steps to "
                         "<output_dir>/checkpoint-<step>.")
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="Keep at most this many recent checkpoints "
                         "(older ones are deleted).")
    ap.add_argument("--save_only_model", action="store_true", default=False,
                    help="Skip optimizer/scheduler/RNG state in checkpoints "
                         "(smaller on disk, no resume).")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to a checkpoint dir, or 'latest' to pick the "
                         "most recent one under output_dir.")
    ap.add_argument("--max_eval_wikifact", type=int, default=100)
    ap.add_argument("--max_eval_flores", type=int, default=64)
    ap.add_argument("--open_gen_steps", type=int, default=100)
    ap.add_argument("--open_gen_max_new_tokens", type=int, default=32)

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", type=str, default="wandb")
    return ap.parse_args()


def default_run_name(model_id: str, lr: float, loss_type: str, curriculum: bool = False) -> str:
    model_name = model_id.split("/")[-1]
    suffix = "-curriculum" if curriculum else ""
    return f"{model_name}-poly-{loss_type}-lr-{lr}{suffix}"


def default_output_dir(model_id: str, lr: float, loss_type: str, curriculum: bool = False) -> str:
    return os.path.join(MODEL_OUT_ROOT, default_run_name(model_id, lr, loss_type, curriculum))


def checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    if not name.startswith('checkpoint-'):
        return None
    step = name.split('-')[-1]
    return int(step) if step.isdigit() else None


def latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if name.startswith('checkpoint-') and name.split('-')[-1].isdigit():
            candidates.append(name)
    if not candidates:
        return None
    latest = max(candidates, key=lambda name: int(name.split('-')[-1]))
    return os.path.join(output_dir, latest)


def read_curriculum_state(output_dir: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, CURRICULUM_STATE_FILE)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(state, dict):
        return {}
    return state


def write_curriculum_state(
    output_dir: str,
    phase: str,
    step: Optional[int] = None,
    reward_ema: Optional[float] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, Any] = {'phase': phase}
    if step is not None:
        payload['step'] = int(step)
    if reward_ema is not None:
        payload['reward_ema'] = float(reward_ema)
    path = os.path.join(output_dir, CURRICULUM_STATE_FILE)
    tmp_path = f'{path}.tmp.{os.getpid()}'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Dataset flattening: one row per (fact, lang), fact-major.
# ---------------------------------------------------------------------------

def build_flat_polyfact_dataset(
    raw_ds,
    langs: List[str],
    min_languages: int,
    prompt_format: str = "closed_book",
) -> Dataset:
    """One HF row per (fact, lang). Rows [k*L .. k*L + L-1] are the L languages
    of fact k. Only facts covering every lang in `langs` (with valid options
    and a resolvable gold letter) are kept — we need a complete tuple to
    compute the joint reward, and MCQ phase needs options.

    `prompt_format`:
      - "closed_book": `Question: {q}\\nAnswer:`
      - "mcq":         `Question: {q}\\nA. {a}\\nB. {b}\\nC. {c}\\nD. {d}\\nAnswer:`

    Every row stores option_a..option_d + gold_letter + gold_text regardless
    of format, so the reward function can do letter resolution in MCQ phase
    and exact text match in closed-book phase from the same dataset row.
    """
    if prompt_format not in ("closed_book", "mcq"):
        raise ValueError(f"prompt_format must be 'closed_book' or 'mcq', got {prompt_format!r}")

    out: Dict[str, List[Any]] = {
        "prompt": [],
        "fact_id": [],
        "lang": [],
        "lang_index": [],
        "gold_text": [],
        "gold_letter": [],
        "option_a": [],
        "option_b": [],
        "option_c": [],
        "option_d": [],
    }
    for ex in raw_ds:
        langs_data = ex.get("translations") or ex.get("langs") or {}
        if not isinstance(langs_data, dict):
            continue

        per_lang = {}
        for lang in langs:
            if lang not in langs_data:
                break
            item = langs_data[lang]
            question = safe_strip(item.get("question", ""))
            answer_text = safe_strip(item.get("answer_text", ""))
            opts = [safe_strip(item.get(f"option_{c}", "")) for c in ("a", "b", "c", "d")]
            if not question or not answer_text or any(not o for o in opts):
                break
            answer_idx = item.get("answer_index")
            if isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
                gold_letter = INDEX_TO_LETTER[answer_idx]
            else:
                gold_letter = answer_text_to_letter(opts, answer_text)
            if gold_letter is None:
                break
            per_lang[lang] = {
                "question": question,
                "answer_text": answer_text,
                "options": opts,
                "gold_letter": gold_letter,
            }
        if len(per_lang) < min_languages or len(per_lang) != len(langs):
            continue

        fact_id = str(ex.get("fact_id", ""))
        for li, lang in enumerate(langs):
            info = per_lang[lang]
            q = info["question"]
            opts = info["options"]
            if prompt_format == "mcq":
                prompt = (
                    f"Question: {q}\n"
                    f"A. {opts[0]}\n"
                    f"B. {opts[1]}\n"
                    f"C. {opts[2]}\n"
                    f"D. {opts[3]}\n"
                    f"Answer:"
                )
            else:
                prompt = f"Question: {q}\nAnswer:"
            out["prompt"].append(prompt)
            out["fact_id"].append(fact_id)
            out["lang"].append(lang)
            out["lang_index"].append(li)
            out["gold_text"].append(info["answer_text"])
            out["gold_letter"].append(info["gold_letter"])
            out["option_a"].append(opts[0])
            out["option_b"].append(opts[1])
            out["option_c"].append(opts[2])
            out["option_d"].append(opts[3])

    return Dataset.from_dict(out)


# ---------------------------------------------------------------------------
# Fact-grouped sampler: keeps the L lang rows of each fact contiguous in every
# generation batch so the joint reward fn always sees the full lang tuple.
# ---------------------------------------------------------------------------

class FactGroupedRepeatSampler(Sampler):
    """Replacement for TRL's RepeatSampler that shuffles at the FACT level.

    Output structure matches TRL's:
        chunk of batch_size unique indices
        → each index emitted mini_repeat_count times consecutively
        → outer block repeated repeat_count times

    Difference: the chunks are made of WHOLE facts (group_size consecutive rows
    each). This guarantees that all `group_size` lang rows of a fact land in
    the same generation batch.

    Assumes the dataset is laid out fact-major: dataset[k*group_size + l] is
    the l-th lang of the k-th fact.
    """

    def __init__(
        self,
        n_facts: int,
        group_size: int,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        seed: int = 0,
    ):
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of group_size "
                f"({group_size}); set generation_batch_size to a multiple of "
                f"{group_size} * num_generations."
            )
        self.n_facts = n_facts
        self.group_size = group_size
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        fact_perm = torch.randperm(self.n_facts, generator=gen).tolist()

        indices = []
        for f in fact_perm:
            base = f * self.group_size
            indices.extend(range(base, base + self.group_size))

        for start in range(0, len(indices), self.batch_size):
            chunk = indices[start : start + self.batch_size]
            if len(chunk) < self.batch_size:
                break
            expanded = []
            for i in chunk:
                expanded.extend([i] * self.mini_repeat_count)
            for _ in range(self.repeat_count):
                yield from expanded

    def __len__(self):
        rows = self.n_facts * self.group_size
        full_chunks = rows // self.batch_size
        return full_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count


class FactGroupedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass that swaps in FactGroupedRepeatSampler."""

    def __init__(self, *args, group_size: int = 12, **kwargs):
        self._group_size = group_size
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        n_facts = len(dataset) // self._group_size
        return FactGroupedRepeatSampler(
            n_facts=n_facts,
            group_size=self._group_size,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            seed=self.args.seed,
        )


# ---------------------------------------------------------------------------
# Joint reward function.
# ---------------------------------------------------------------------------

def make_joint_reward_func(num_generations: int, all_correct_bonus: float):
    """Closure-bound reward function that computes one scalar joint reward per
    (fact_id, gen_idx) tuple and replicates it to all `num_generations` lang
    completions sharing that tuple.

    Contract notes:
        - TRL's RepeatSampler emits each prompt mini_repeat_count=num_generations
          times consecutively, so flat index i → row_idx = i // G, gen_idx = i % G.
        - `fact_id`, `lang`, `gold_text` arrive as lists broadcast 1:1 with
          `completions` (TRL forwards dataset columns automatically).
        - The function's per-completion output is the SAME for all lang rows of
          a given (fact_id, gen_idx). Per-prompt advantage normalization then
          produces identical advantages across the 12 langs — equivalent to the
          bespoke "one advantage applied to all 12 outputs in a rollout" logic.
    """

    def joint_reward_func(
        prompts,
        completions,
        completion_ids,
        fact_id,
        lang,
        gold_text,
        gold_letter,
        option_a,
        option_b,
        option_c,
        option_d,
        log_metric=None,
        log_extra=None,
        **kwargs,
    ):
        phase = CURRICULUM_PHASE["phase"]
        n = len(completions)
        if n % num_generations != 0:
            raise RuntimeError(
                f"Reward batch size {n} is not divisible by num_generations "
                f"{num_generations}; sampler/batch alignment is off."
            )

        def is_hit(pred: str, i: int) -> bool:
            if phase == "mcq":
                # MCQ phase: accept either the letter or the option text. We
                # still credit a closed-book-style exact match against the gold
                # text too — the model is allowed to ignore the options and
                # type the answer outright.
                option_map = {
                    "A": option_a[i], "B": option_b[i],
                    "C": option_c[i], "D": option_d[i],
                }
                resolved, ok = resolve_prediction_to_letter(pred, option_map)
                if ok and resolved == gold_letter[i]:
                    return True
                return is_correct_text_match(pred, gold_text[i])
            return is_correct_text_match(pred, gold_text[i])

        # group_key → {lang: (pred, flat_idx)} so we can re-look-up options/gold for is_hit.
        groups: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        for i, comp in enumerate(completions):
            gen_idx = i % num_generations
            key = (fact_id[i], gen_idx)
            groups.setdefault(key, {})[lang[i]] = {
                "pred": extract_answer_text(comp),
                "i": i,
            }

        joint: Dict[Any, float] = {}
        n_groups_full = 0
        per_lang_correct = {lg: 0 for lg in LANGS}
        per_lang_total = {lg: 0 for lg in LANGS}
        total_correct = 0
        total_slots = 0
        for key, lang_preds in groups.items():
            score = 0.0
            n_correct = 0
            for lg, pi in lang_preds.items():
                hit = is_hit(pi["pred"], pi["i"])
                if hit:
                    n_correct += 1
                    score += 1.0
                    if lg in per_lang_correct:
                        per_lang_correct[lg] += 1
                if lg in per_lang_total:
                    per_lang_total[lg] += 1
                total_slots += 1
                total_correct += int(hit)
            if lang_preds and n_correct == len(lang_preds):
                score += all_correct_bonus
                n_groups_full += 1
            joint[key] = float(score)

        rewards = [joint[(fact_id[i], i % num_generations)] for i in range(n)]

        if log_metric is not None and groups:
            log_metric("joint/mean_score", float(np.mean(list(joint.values()))))
            log_metric("joint/frac_all_correct", n_groups_full / max(len(groups), 1))
            log_metric("joint/per_slot_accuracy", total_correct / max(total_slots, 1))
            # Phase indicator (0 = closed_book, 1 = mcq) so wandb plots show
            # exactly when the curriculum switched.
            log_metric("curriculum/phase_is_mcq", 1.0 if phase == "mcq" else 0.0)
            for lg in LANGS:
                if per_lang_total[lg] > 0:
                    log_metric(
                        f"joint/lang_acc_{lg}",
                        per_lang_correct[lg] / per_lang_total[lg],
                    )

        return rewards

    return joint_reward_func


# ---------------------------------------------------------------------------
# Curriculum: phase-1 (MCQ) → phase-2 (closed-book) trigger.
# ---------------------------------------------------------------------------

class CurriculumCallback(TrainerCallback):
    """Tracks a running EMA of `reward` from training logs. When the EMA
    crosses `threshold` (and `global_step >= min_steps`), it:
        1. forces a checkpoint save at the current step
        2. requests training to stop cleanly
        3. flips the CURRICULUM_PHASE flag to "closed_book"

    The outer `main()` then detects `fired=True` and calls trainer.train()
    again with `train_dataset=train_ds_closed` and `resume_from_checkpoint=`
    the just-saved checkpoint — HF Trainer's resume preserves LoRA weights,
    optimizer, scheduler, and global_step across the swap.
    """

    def __init__(self, threshold: float, min_steps: int, alpha: float = 0.1):
        self.threshold = threshold
        self.min_steps = min_steps
        self.alpha = alpha
        self.reward_ema: Optional[float] = None
        self.fired = False
        self.fired_step: Optional[int] = None

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if self.fired or CURRICULUM_PHASE["phase"] != "mcq":
            return control
        if not logs or "reward" not in logs:
            return control
        r = float(logs["reward"])
        self.reward_ema = (
            r if self.reward_ema is None
            else self.alpha * r + (1 - self.alpha) * self.reward_ema
        )
        # Surface the EMA so users can see the trajectory in wandb.
        logs["curriculum/reward_ema"] = self.reward_ema
        if state.global_step >= self.min_steps and self.reward_ema >= self.threshold:
            self.fired = True
            self.fired_step = state.global_step
            CURRICULUM_PHASE["phase"] = "closed_book"
            write_curriculum_state(args.output_dir, 'closed_book', step=state.global_step, reward_ema=self.reward_ema)
            print(
                f"\n[curriculum] reward_ema={self.reward_ema:.3f} >= "
                f"threshold={self.threshold} at step {state.global_step}; "
                f"saving checkpoint and switching to closed-book.",
                flush=True,
            )
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Eval callback: reuses the bespoke trainer's WikiFact / FLORES / MMLU /
# open-gen-probe routines so we don't reimplement them.
# ---------------------------------------------------------------------------

class PolyFactEvalCallback(TrainerCallback):
    def __init__(
        self,
        wikifact_val_ds,
        flores_eval_sets,
        mmlu_eval_sets,
        tokenizer,
        max_prompt_length: int,
        max_completion_length: int,
        eval_steps: int,
        open_gen_steps: int,
        open_gen_max_new_tokens: int,
        probe_log_path: str,
        report_to: str,
    ):
        self.wikifact_val_ds = wikifact_val_ds
        self.flores_eval_sets = flores_eval_sets
        self.mmlu_eval_sets = mmlu_eval_sets
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.eval_steps = eval_steps
        self.open_gen_steps = open_gen_steps
        self.open_gen_max_new_tokens = open_gen_max_new_tokens
        self.probe_log_path = probe_log_path
        self.report_to = report_to

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control
        gs = state.global_step
        if gs == 0:
            return control

        if self.open_gen_steps > 0 and gs % self.open_gen_steps == 0:
            run_open_generation_probes(
                model=model,
                tokenizer=self.tokenizer,
                prompts=OPEN_GENERATION_PROMPTS,
                global_step=gs,
                max_new_tokens=self.open_gen_max_new_tokens,
                report_to=self.report_to,
                log_path=self.probe_log_path,
            )

        if self.eval_steps > 0 and gs % self.eval_steps == 0:
            torch.cuda.empty_cache()
            device = next(model.parameters()).device

            metrics: Dict[str, float] = {}
            metrics.update(
                evaluate_wikifact_grouped(
                    model=model,
                    tokenizer=self.tokenizer,
                    eval_ds=self.wikifact_val_ds,
                    max_prompt_length=self.max_prompt_length,
                    max_completion_length=self.max_completion_length,
                )
            )

            # FLORES is expensive, so run it every other eval, matching the
            # original cadence in run_full_eval (eval_steps % (2*eval_steps)==0).
            if self.flores_eval_sets and gs % (2 * self.eval_steps) == 0:
                metrics.update(
                    compute_flores_bleu(
                        model=model,
                        tokenizer=self.tokenizer,
                        flores_sets=self.flores_eval_sets,
                        device=device,
                        max_new_tokens=64,
                        batch_size=4,
                    )
                )
                metrics.update(
                    compute_flores_bleu_to_english(
                        model=model,
                        tokenizer=self.tokenizer,
                        flores_sets=self.flores_eval_sets,
                        device=device,
                        max_new_tokens=64,
                        batch_size=4,
                    )
                )
                metrics.update(
                    compute_flores_hidden_cosine(
                        model=model,
                        tokenizer=self.tokenizer,
                        flores_sets=self.flores_eval_sets,
                        device=device,
                        batch_size=1,
                    )
                )

            # Global MMLU (cheap, run every eval).
            if self.mmlu_eval_sets:
                metrics.update(
                    _compute_global_mmlu(
                        model=model,
                        tokenizer=self.tokenizer,
                        mmlu_eval_sets=self.mmlu_eval_sets,
                        device=device,
                    )
                )

            print(f"\n[eval @ step {gs}] {metrics}", flush=True)
            if self.report_to == "wandb" and wandb.run is not None:
                wandb.log(metrics, step=gs)
            torch.cuda.empty_cache()

        return control


def _compute_global_mmlu(model, tokenizer, mmlu_eval_sets, device) -> Dict[str, float]:
    """Per-language Global MMLU accuracy via teacher-forced A/B/C/D logits.
    Lifted from train_wikifact_grpo.run_full_eval so we don't import private
    locals."""
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader as TorchDataLoader

    model.eval()
    choice_ids = [
        tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
    ]
    metrics: Dict[str, float] = {}

    def collate(batch):
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

    for lang, ds in mmlu_eval_sets.items():
        loader = TorchDataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
        correct = total = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for i in range(labels.shape[0]):
                pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    continue
                j = int(pos[0])
                gold_token = int(labels[i, j].item())
                if gold_token not in choice_ids:
                    continue
                gold_idx = choice_ids.index(gold_token)
                pred_idx = int(logits[i, j - 1, choice_ids].argmax().item())
                correct += int(pred_idx == gold_idx)
                total += 1
        metrics[f"mmlu/acc_{lang}"] = correct / total if total else 0.0

    accs = [metrics[f"mmlu/acc_{lg}"] for lg in LANGS if f"mmlu/acc_{lg}" in metrics]
    if accs:
        metrics["mmlu/acc_avg"] = float(np.mean(accs))
    return metrics


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.no_bf16:
        args.bf16 = False

    if args.use_vllm and not args.vllm_server_base_url:
        raise ValueError("--use_vllm requires --vllm_server_base_url for explicit TRL server mode. Start the vLLM server and pass its URL.")
    if args.curriculum and args.save_only_model:
        raise ValueError('--save_only_model cannot be used with --curriculum; phase 2 needs optimizer/scheduler/RNG state.')
    if args.curriculum and not (0.0 < args.curriculum_alpha <= 1.0):
        raise ValueError('--curriculum_alpha must be in the interval (0, 1].')


    args.model_id = resolve_model_id(args.model_id)
    run_name = default_run_name(args.model_id, args.learning_rate, args.loss_type, args.curriculum)
    output_dir = default_output_dir(args.model_id, args.learning_rate, args.loss_type, args.curriculum)
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Run name (wandb + on-disk): {run_name}\n"
        f"Checkpoints will be saved to: {output_dir}/checkpoint-<step>\n"
        f"  (keeping the last {args.save_total_limit}, every {args.save_steps} steps)",
        flush=True,
    )

    # Resolve --resume_from_checkpoint early because it controls curriculum phase.
    resume = args.resume_from_checkpoint
    if resume == 'latest':
        resume = latest_checkpoint_dir(output_dir)
        if resume is not None:
            print(f'Resuming from {resume}', flush=True)
        else:
            print(f'No checkpoint-* under {output_dir}; starting fresh.', flush=True)
    curriculum_state_dir = (
        os.path.dirname(os.path.normpath(resume))
        if args.curriculum and resume
        else output_dir
    )
    if not curriculum_state_dir:
        curriculum_state_dir = output_dir
    curriculum_state = read_curriculum_state(curriculum_state_dir) if args.curriculum and resume else {}
    if args.curriculum and resume and curriculum_state.get('phase') == 'closed_book':
        state_step = curriculum_state.get('step')
        try:
            state_step = int(state_step) if state_step is not None else None
        except (TypeError, ValueError):
            state_step = None
        resume_step = checkpoint_step(resume)
        if state_step is not None and resume_step is not None and resume_step < state_step:
            print('[curriculum] State says closed-book, but the checkpoint predates the switch; resuming MCQ.', flush=True)
            curriculum_state = {'phase': 'mcq'}
    if args.curriculum and resume and not curriculum_state:
        print('[curriculum] No curriculum_state.json found; assuming resume phase is MCQ.', flush=True)

    # Resolve generation/batching sizing. With both `generation_batch_size`
    # and `steps_per_generation` unset, TRL defaults to
    #   generation_batch_size = per_device * num_processes * grad_accum
    # The RepeatSampler then splits the generation batch into chunks of
    # `generation_batch_size // num_generations` unique prompts; that chunk
    # must be a whole multiple of group_size so all 12 lang rows of each
    # fact stay together for the joint reward.
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = len(LANGS) * args.num_generations
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_expanded = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * world_size
    )
    gen_batch_unit = len(LANGS) * args.num_generations
    if effective_expanded % gen_batch_unit != 0:
        raise ValueError(
            f"per_device_train_batch_size ({args.per_device_train_batch_size}) "
            f"× gradient_accumulation_steps ({args.gradient_accumulation_steps}) "
            f"× world_size ({world_size}) = {effective_expanded}, which must be a "
            f"multiple of len(LANGS) × num_generations = {gen_batch_unit} so the "
            f"joint reward fn always sees the full lang tuple of every fact."
        )

    if args.report_to == "wandb":
        wandb.init(project=WANDB_PROJECT, name=run_name, config=vars(args))

    # ---------------------- model / tokenizer ----------------------
    print(f"Loading tokenizer for {args.model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    print(f"Loading model {args.model_id} ...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    )

    if args.use_lora:
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = base_model
    model.train()

    # ---------------------- datasets ----------------------
    print(
        f"Loading dataset {args.dataset_id} (config={args.dataset_config or 'default'}) ...",
        flush=True,
    )
    if args.dataset_config:
        raw_all = load_dataset(args.dataset_id, args.dataset_config)
    else:
        raw_all = load_dataset(args.dataset_id)
    raw_train = raw_all["train"]
    raw_val = raw_all["validation"]

    # If --curriculum, we need both prompt formats; otherwise just the closed-book one.
    resume_phase = curriculum_state.get('phase') if args.curriculum else None
    if args.curriculum and resume_phase == 'closed_book':
        initial_phase = 'closed_book'
    else:
        initial_phase = 'mcq' if args.curriculum else 'closed_book'
    CURRICULUM_PHASE['phase'] = initial_phase
    if args.curriculum and resume is None:
        write_curriculum_state(output_dir, initial_phase, step=0)

    print(f"Flattening train split to (fact, lang) rows — phase 1 = {initial_phase} ...", flush=True)
    train_ds_phase1 = build_flat_polyfact_dataset(
        raw_train, LANGS, args.min_languages, prompt_format=initial_phase
    )
    train_ds_closed: Optional[Dataset] = None
    if args.curriculum:
        print("Building parallel closed-book dataset for phase 2 ...", flush=True)
        train_ds_closed = build_flat_polyfact_dataset(
            raw_train, LANGS, args.min_languages, prompt_format="closed_book"
        )

    if args.max_train_samples is not None:
        # max_train_samples counts FACTS, not rows.
        n_keep = min(args.max_train_samples, len(train_ds_phase1) // len(LANGS)) * len(LANGS)
        train_ds_phase1 = train_ds_phase1.select(range(n_keep))
        if train_ds_closed is not None:
            train_ds_closed = train_ds_closed.select(range(n_keep))

    print(
        f"Train rows: {len(train_ds_phase1)}  "
        f"({len(train_ds_phase1) // len(LANGS)} facts)",
        flush=True,
    )

    # The validation set is built in the same grouped-by-fact JSON shape the
    # bespoke `evaluate_wikifact_grouped` expects — easiest path is to reuse
    # the original `build_grouped_fact_item` here.
    from train_wikifact_grpo import build_grouped_fact_item  # local import to avoid top-level cost
    val_ds = raw_val.map(build_grouped_fact_item)
    val_ds = val_ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= args.min_languages)
    keep_cols = ["fact_id", "prompts_by_lang_json", "meta_by_lang_json"]
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    if args.max_eval_wikifact is not None:
        val_ds = val_ds.select(range(min(args.max_eval_wikifact, len(val_ds))))

    flores_eval_sets = (
        load_flores_parallel_subset(
            target_langs=[lg for lg in LANGS if lg != "en"],
            split="dev",
            max_samples=args.max_eval_flores,
        )
        if args.max_eval_flores > 0
        else {}
    )
    mmlu_eval_sets = load_global_mmlu_dev_eval_by_lang(LANGS, tokenizer)

    # ---------------------- GRPOConfig ----------------------
    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs, 
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        seed=args.seed,
        logging_steps=args.logging_steps,
        # Checkpointing — checkpoints land at <output_dir>/checkpoint-<step>,
        # where output_dir = MODEL_OUT_ROOT / <run_name>. Same `run_name` is
        # used as the wandb run name, so the on-disk dir and wandb run line up.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        save_only_model=args.save_only_model,
        report_to=args.report_to,
        # GRPO / DAPO knobs.
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        loss_type=args.loss_type,
        epsilon=args.epsilon,
        scale_rewards=args.scale_rewards,
        beta=args.beta,
        # vLLM.
        use_vllm=args.use_vllm,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_group_port=args.vllm_group_port,
    )

    # ---------------------- trainer ----------------------
    reward_fn = make_joint_reward_func(
        num_generations=args.num_generations,
        all_correct_bonus=args.all_correct_bonus,
    )

    probe_log_path = os.path.join(
        "/home/nvidia/jonathan/projects/Lost-in-Mistranslation/grpo_samples",
        run_name,
    )
    os.makedirs(os.path.dirname(probe_log_path), exist_ok=True)

    callbacks: List[TrainerCallback] = [
        PolyFactEvalCallback(
            wikifact_val_ds=val_ds,
            flores_eval_sets=flores_eval_sets,
            mmlu_eval_sets=mmlu_eval_sets,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            eval_steps=args.eval_steps,
            open_gen_steps=args.open_gen_steps,
            open_gen_max_new_tokens=args.open_gen_max_new_tokens,
            probe_log_path=probe_log_path,
            report_to=args.report_to,
        ),
    ]
    curriculum_cb: Optional[CurriculumCallback] = None
    if args.curriculum:
        curriculum_cb = CurriculumCallback(
            threshold=args.curriculum_threshold,
            min_steps=args.curriculum_min_steps,
            alpha=args.curriculum_alpha,
        )
        callbacks.append(curriculum_cb)
        print(
            f"Curriculum enabled: MCQ → closed-book at reward_ema >= "
            f"{args.curriculum_threshold} (min {args.curriculum_min_steps} steps, "
            f"alpha={args.curriculum_alpha})",
            flush=True,
        )

    trainer = FactGroupedGRPOTrainer(
        model=model,
        args=grpo_cfg,
        reward_funcs=reward_fn,
        train_dataset=train_ds_phase1,
        processing_class=tokenizer,
        group_size=len(LANGS),
        callbacks=callbacks,
    )

    # Baseline open-gen probe before training, mirroring the bespoke script.
    run_open_generation_probes(
        model=model,
        tokenizer=tokenizer,
        prompts=OPEN_GENERATION_PROMPTS,
        global_step=0,
        max_new_tokens=args.open_gen_max_new_tokens,
        report_to=args.report_to,
        log_path=probe_log_path,
    )

    print('Starting DAPO training via TRL GRPOTrainer (' + CURRICULUM_PHASE['phase'] + ') ...', flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)

    if args.curriculum and curriculum_cb is not None and curriculum_cb.fired:
        if train_ds_closed is None:
            raise RuntimeError('Curriculum fired, but the closed-book dataset was not built.')
        switch_checkpoint = latest_checkpoint_dir(output_dir)
        fired_step = curriculum_cb.fired_step or 0
        switch_step = checkpoint_step(switch_checkpoint) if switch_checkpoint is not None else None
        if switch_checkpoint is None or switch_step is None or switch_step < fired_step:
            raise RuntimeError('Curriculum fired, but no fresh checkpoint was saved for the phase switch.')

        print(f'[curriculum] Resuming closed-book phase from {switch_checkpoint}', flush=True)
        trainer.train_dataset = train_ds_closed
        if hasattr(trainer, '_buffered_inputs'):
            trainer._buffered_inputs = None
        if hasattr(trainer, 'control'):
            trainer.control = TrainerControl()
        CURRICULUM_PHASE['phase'] = 'closed_book'
        write_curriculum_state(output_dir, 'closed_book', step=fired_step, reward_ema=curriculum_cb.reward_ema)
        trainer.train(resume_from_checkpoint=switch_checkpoint)

    print(f'Training done in {time.time() - t0:.1f}s', flush=True)

    # Post-training probe.
    run_open_generation_probes(
        model=model,
        tokenizer=tokenizer,
        prompts=OPEN_GENERATION_PROMPTS,
        global_step=trainer.state.global_step,
        max_new_tokens=args.open_gen_max_new_tokens,
        report_to=args.report_to,
        log_path=probe_log_path,
    )

    final_dir = os.path.join(output_dir, "final")
    print(f"Saving final model snapshot to {final_dir} ...", flush=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    if args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
