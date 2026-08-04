#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EN-pivot DPO baseline (CM-Align style) for cross-lingual factual recall.

Reviewer-requested CLC-enhancement baseline to sit next to the GRPO method.
Faithful adaptation of CM-Align (Zhang et al., "Consistency-based Multilingual
Alignment for Large Language Models", EMNLP 2025 Findings, arXiv:2509.08541;
code: github.com/XZhang00/CM-Align) to the PolyFact / WIKI-FACT MCQ setting.

The method is *self-supervised* — it never looks at gold answers. It uses the
model's own English generations as a pivot and aligns the other languages to it:

  Phase 1 (construct):
    For each fact, sample K free-text answers per language (same single-language
    prompts the GRPO trainer uses).
    (a) Consistency-guided English reference selection: embed the K English
        candidates with a multilingual sentence encoder (LaBSE by default) and
        pick the one with the highest mean cosine similarity to the other
        English candidates  ->  y_en*   (CM-Align Cons_GIF).
    (b) Cross-lingual preference pairs: for every other language l, embed its K
        candidates and score each by cosine to y_en*  (CM-Align CL-Cons_GIF).
        chosen  = argmax cosine,   rejected = argmin cosine.
    Emit DPO rows {prompt, chosen, rejected, fact_id, lang} and cache to disk.

  Phase 2 (train):
    DPO on the constructed pairs.  LoRA policy; the reference distribution is the
    same model with the LoRA adapter disabled (base weights), so no second model
    copy is loaded.  Objective (CM-Align eq. with optional NLL term):
        L = L_DPO  +  gamma * L_NLL
        L_DPO = -log sigmoid( beta * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref)) )
        L_NLL = -lp_w / |y_w|
    Defaults follow the CM-Align GIF task: beta=0.1, gamma=0.0.

## LAUNCH (single interactive GH200 / A100 node, end-to-end):

  python training/train_wikifact_cmalign_dpo.py \
    --phase all \
    --model_id jvonrad/Qwen-2.5-7B-TED \
    --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
    --pref_data_path /projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/datasets/cmalign_pref_qwen_ted \
    --output_dir /projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/models/qwen-2.5-7b-ted-cmalign-dpo \
    --run_name qwen-2.5-7b-ted-cmalign-dpo \
    --max_facts 8000 --num_candidates 4 \
    --beta 0.1 --nll_gamma 0.0 --learning_rate 5e-6 --num_train_epochs 1 \
    --bf16

Split the phases (e.g. build the data once, then sweep DPO hyperparameters) with
--phase construct  then  --phase train.  Phase 1 is skipped automatically if the
cache at --pref_data_path already exists (override with --overwrite_pref_data).
"""

import os
import re
import json
import math
import time
import shutil
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset, Dataset, load_from_disk

import prompt_scaffold as pscaf

import polyfact_schema as pfs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

# ─────────────────────────────────────────────
# Constants (kept in sync with train_wikifact_grpo_accelerate.py)
# ─────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B"
HF_DATASET_ID = "jvonrad/PolyFact-Clean"
HF_DATASET_CONFIG = "parallel"
WANDB_PROJECT = "UnLock"

LANGS = ["en", "es", "fr", "de", "id", "pt", "ru", "zh", "ja", "ar", "sw", "bn"]

LANG_TO_NAME = {
    "en": "English", "de": "German", "id": "Indonesian", "pt": "Portuguese",
    "ar": "Arabic", "bn": "Bengali", "sw": "Swahili", "es": "Spanish",
    "ru": "Russian", "fr": "French", "ja": "Japanese", "zh": "Chinese",
}

LORA_R = 64
LORA_ALPHA = 128


# ─────────────────────────────────────────────
# Text / prompt utilities (mirrors the GRPO trainer's free-text answer format)
# ─────────────────────────────────────────────

def safe_strip(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ""


def extract_answer_text(text: str) -> str:
    text = safe_strip(text)
    text = pscaf.strip_answer_label(text)
    text = text.split("\n")[0].strip()
    return text


def build_single_language_prompt(lang, question, options, scaffold="native"):
    """Free-form MCQ prompt, localised to `lang` (see prompt_scaffold).

    Base models continue in the language of their context, so an English
    scaffold around a non-English question biases the answer toward English
    and then fails option matching. scaffold='en' restores the old prompt.
    """
    return pscaf.build_single_language_prompt(lang, question, options, scaffold)


def build_fact_prompts(ex: Dict[str, Any]) -> Dict[str, str]:
    """One free-text prompt per available language.

    Handles PolyFact-Clean (`translations` + option_a..d) and legacy WIKI-FACT
    (`langs` + options list) via the shared polyfact_schema normaliser.
    """
    langs_data = pfs.lang_blocks(ex)
    if not langs_data:
        return {}
    prompts_by_lang = {}
    for lang in LANGS:
        parsed = pfs.normalize_lang_item(langs_data.get(lang))
        if parsed is None:
            continue
        question, options, _answer_text, _gold_idx = parsed
        prompts_by_lang[lang] = build_single_language_prompt(
            lang, question, pfs.option_map(options)
        )
    return prompts_by_lang


# ─────────────────────────────────────────────
# Phase 1: preference-data construction
# ─────────────────────────────────────────────

class Embedder:
    """Multilingual sentence encoder for consistency scoring (LaBSE by default)."""

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        # Empty strings would give degenerate vectors; replace with a space so
        # cosine is defined and consistently low against real answers.
        cleaned = [t if t.strip() else " " for t in texts]
        emb = self.model.encode(
            cleaned, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=256,
        )
        return emb


def _mean_cosine_argmax(emb: np.ndarray) -> int:
    """Index of the row with the highest mean cosine similarity to the other rows."""
    if emb.shape[0] == 1:
        return 0
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / (emb.shape[0] - 1)
    return int(np.argmax(mean_sim))


def select_en_reference(en_candidates: List[str], embedder: Embedder) -> Tuple[int, str]:
    """CM-Align consistency-guided English reference selection.

    Returns the index and text of the English candidate with the highest mean
    cosine similarity to the other English candidates (Cons_GIF). With a single
    candidate, that candidate is the reference by definition.

    Single-fact convenience wrapper around `_mean_cosine_argmax`; the batched
    construction path below embeds many facts per call instead of calling this.
    """
    if len(en_candidates) == 1:
        return 0, en_candidates[0]
    emb = embedder.encode(en_candidates)              # [K, d], L2-normalized
    best = _mean_cosine_argmax(emb)
    return best, en_candidates[best]


@torch.no_grad()
def sample_candidates_for_facts(
    model, tokenizer, facts_prompts: List[Dict[str, str]], num_candidates: int,
    max_prompt_length: int, max_completion_length: int,
    temperature: float, top_p: float, gen_micro_batch_size: int, device,
) -> List[Dict[str, List[str]]]:
    """Sample `num_candidates` completions per language for a whole batch of facts.

    facts_prompts[i] = {lang: prompt} for fact i. All (fact, lang, candidate)
    tuples across the whole super-batch are flattened into one generation
    stream and chunked by gen_micro_batch_size, so one `generate()` call
    covers many facts instead of one — this is the throughput fix over the
    old per-fact `sample_candidates` (kept only implicitly here; superseded).

    Returns candidates[i] = {lang: [num_candidates texts]}, aligned to
    facts_prompts.
    """
    flat_prompts, flat_owner = [], []
    for fi, prompts_by_lang in enumerate(facts_prompts):
        for lang, prompt in prompts_by_lang.items():
            for _ in range(num_candidates):
                flat_prompts.append(prompt)
                flat_owner.append((fi, lang))

    out: List[Dict[str, List[str]]] = [{} for _ in facts_prompts]
    model.eval()
    for start in range(0, len(flat_prompts), gen_micro_batch_size):
        chunk = flat_prompts[start:start + gen_micro_batch_size]
        owners = flat_owner[start:start + gen_micro_batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_length,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]
        gen = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_completion_length, repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        texts = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)
        for (fi, lang), text in zip(owners, texts):
            out[fi].setdefault(lang, []).append(extract_answer_text(text))
        del inputs, gen
    torch.cuda.empty_cache()
    return out


def save_checkpoint(rows: List[Dict[str, Any]], meta: Dict[str, Any], path: str) -> None:
    """Atomically replace the preference-dataset checkpoint at `path`.

    Writes to a sibling tmp dir first and renames over `path`, so a process
    killed mid-write leaves the previous good checkpoint intact instead of a
    corrupt/partial arrow file.
    """
    tmp_path = path.rstrip("/") + "_tmp"
    os.makedirs(os.path.dirname(tmp_path.rstrip("/")) or ".", exist_ok=True)
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    Dataset.from_list(rows).save_to_disk(tmp_path)
    with open(os.path.join(tmp_path, "_checkpoint_meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp_path, path)


def construct_preference_data(args, device) -> Dataset:
    # ── Resume from a checkpoint before loading anything expensive ──
    meta_path = os.path.join(args.pref_data_path, "_checkpoint_meta.json")
    rows: List[Dict[str, Any]] = []
    n_facts_used = 0
    resume_batch_start = 0
    if os.path.exists(meta_path) and not args.overwrite_pref_data:
        with open(meta_path) as f:
            ckpt_meta = json.load(f)
        if ckpt_meta.get("done"):
            print(f"[construct] checkpoint at {args.pref_data_path} already complete "
                  f"({ckpt_meta.get('n_pairs', '?')} pairs); skipping construction.", flush=True)
            return load_from_disk(args.pref_data_path)
        resume_batch_start = ckpt_meta.get("next_batch_start", 0)
        n_facts_used = ckpt_meta.get("n_facts_used", 0)
        rows = load_from_disk(args.pref_data_path).to_list()
        print(f"[construct] resuming from checkpoint: batch_start={resume_batch_start}, "
              f"{len(rows)} pairs, {n_facts_used} facts used so far", flush=True)

    print(f"[construct] loading tokenizer/model {args.model_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    ).to(device)

    print(f"[construct] loading embedder {args.embedding_model} ...", flush=True)
    embedder = Embedder(args.embedding_model, device=device)

    print(f"[construct] loading dataset {args.dataset_id} ...", flush=True)
    raw = (load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)
           if args.dataset_config else
           load_dataset(args.dataset_id, split=args.dataset_split))
    if args.max_facts is not None:
        raw = raw.shuffle(seed=args.seed).select(range(min(args.max_facts, len(raw))))
    # Disjoint contiguous shard of the (globally shuffled) fact set, so N parallel
    # GPU workers cover all facts exactly once. Merge the per-shard datasets before
    # the train phase (see cluster/run_cmalign_sharded.sh).
    if args.num_shards > 1:
        n = len(raw)
        per = math.ceil(n / args.num_shards)
        lo = min(args.shard_index * per, n)
        hi = min(lo + per, n)
        raw = raw.select(range(lo, hi))
        print(f"[construct] shard {args.shard_index}/{args.num_shards}: "
              f"facts [{lo}, {hi}) of {n}", flush=True)

    n_facts_total = len(raw)
    t0 = time.time()

    for batch_start in range(resume_batch_start, n_facts_total, args.facts_per_gen_batch):
        batch_end = min(batch_start + args.facts_per_gen_batch, n_facts_total)
        exs = raw.select(range(batch_start, batch_end))

        facts_prompts: List[Dict[str, str]] = []
        fact_ids: List[str] = []
        for local_i, ex in enumerate(exs):
            prompts_by_lang = build_fact_prompts(ex)
            if "en" not in prompts_by_lang or len(prompts_by_lang) < args.min_languages:
                continue
            facts_prompts.append(prompts_by_lang)
            fact_ids.append(ex.get("fact_id", f"idx{batch_start + local_i}"))

        if not facts_prompts:
            continue

        cand_results = sample_candidates_for_facts(
            model, tokenizer, facts_prompts, args.num_candidates,
            args.max_prompt_length, args.max_completion_length,
            args.temperature, args.top_p, args.gen_micro_batch_size, device,
        )

        # One embedding call for every candidate text in this whole super-batch
        # (was up to 12 tiny embedder.encode() calls PER FACT before).
        flat_texts: List[str] = []
        flat_loc: List[Tuple[int, str, int]] = []   # (fact_local_idx, lang, cand_idx)
        for fi, cand_by_lang in enumerate(cand_results):
            for lang, cands in cand_by_lang.items():
                for ci, c in enumerate(cands):
                    flat_texts.append(c)
                    flat_loc.append((fi, lang, ci))
        flat_emb = embedder.encode(flat_texts) if flat_texts else np.zeros((0, 1), dtype=np.float32)

        # Regroup embeddings back to emb_by_fact_lang[fi][lang] = [K, d] array,
        # aligned index-for-index with cand_results[fi][lang].
        emb_by_fact_lang: Dict[int, Dict[str, List[np.ndarray]]] = {}
        for (fi, lang, ci), emb in zip(flat_loc, flat_emb):
            emb_by_fact_lang.setdefault(fi, {}).setdefault(lang, []).append(emb)

        for fi, cand_by_lang in enumerate(cand_results):
            langs = list(cand_by_lang.keys())
            if "en" not in cand_by_lang:
                continue
            en_cands = cand_by_lang["en"]
            en_emb = np.stack(emb_by_fact_lang[fi]["en"])           # [K, d]
            valid_idx = [i for i, c in enumerate(en_cands) if c.strip()]
            if not valid_idx:
                continue
            best_local = _mean_cosine_argmax(en_emb[valid_idx])
            en_ref_idx = valid_idx[best_local]
            y_en_ref = en_cands[en_ref_idx]
            ref_emb = en_emb[en_ref_idx]                            # [d]

            fact_id = fact_ids[fi]
            prompts_by_lang = facts_prompts[fi]
            target_langs = langs if args.include_en_pairs else [l for l in langs if l != "en"]

            for l in target_langs:
                cands = cand_by_lang[l]
                if len(cands) < 2:
                    continue
                emb = np.stack(emb_by_fact_lang[fi][l])             # [K, d]
                cos = emb @ ref_emb                                 # [K]
                w_idx = int(np.argmax(cos))
                l_idx = int(np.argmin(cos))
                if w_idx == l_idx:
                    continue
                chosen, rejected = cands[w_idx], cands[l_idx]
                if chosen.strip() == rejected.strip():
                    continue
                if float(cos[w_idx] - cos[l_idx]) < args.min_margin:
                    continue
                rows.append({
                    "fact_id": fact_id,
                    "lang": l,
                    "prompt": prompts_by_lang[l],
                    "chosen": chosen,
                    "rejected": rejected,
                    "en_reference": y_en_ref,
                    "cos_chosen": float(cos[w_idx]),
                    "cos_rejected": float(cos[l_idx]),
                })
            n_facts_used += 1

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0.0
        eta_s = (n_facts_total - batch_end) / rate if rate > 0 else 0
        print(
            f"[construct] fact {batch_end}/{n_facts_total} | used {n_facts_used} | "
            f"pairs {len(rows)} | {rate:.2f} facts/s | eta {int(eta_s // 3600)}h{int((eta_s % 3600) // 60):02d}m",
            flush=True,
        )

        # Periodic checkpoint: survives a killed job (e.g. login-node session
        # teardown killing the srun client) instead of losing hours of generation.
        if batch_end % args.checkpoint_every < args.facts_per_gen_batch:
            save_checkpoint(
                rows,
                {"next_batch_start": batch_end, "n_facts_used": n_facts_used,
                 "n_pairs": len(rows), "done": False},
                args.pref_data_path,
            )
            print(f"[construct] checkpoint saved @ fact {batch_end} ({len(rows)} pairs)", flush=True)

    print(f"[construct] done: {len(rows)} preference pairs from {n_facts_used} facts", flush=True)
    save_checkpoint(
        rows,
        {"next_batch_start": n_facts_total, "n_facts_used": n_facts_used,
         "n_pairs": len(rows), "done": True},
        args.pref_data_path,
    )
    print(f"[construct] saved preference dataset to {args.pref_data_path}", flush=True)

    # free the generation model before the training phase reloads it
    del model, embedder
    torch.cuda.empty_cache()
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────
# Phase 2: DPO training
# ─────────────────────────────────────────────

def tokenize_pair(tokenizer, prompt: str, completion: str,
                  max_prompt_length: int, max_completion_length: int):
    """Return (input_ids, labels) for prompt + ' ' + completion + eos.

    Prompt tokens are masked (-100) in labels; completion tokens (incl. eos) are
    kept. Prompt keeps special tokens (BOS where the model uses one); the
    completion is appended without special tokens plus an explicit eos.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True,
                           truncation=True, max_length=max_prompt_length)["input_ids"]
    comp_text = " " + completion.strip()
    comp_ids = tokenizer(comp_text, add_special_tokens=False,
                         truncation=True, max_length=max_completion_length)["input_ids"]
    comp_ids = comp_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids
    return input_ids, labels


class DPOCollator:
    def __init__(self, tokenizer, max_prompt_length, max_completion_length):
        self.tok = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids, labels, is_chosen = [], [], []
        # Stack chosen first then rejected so the two halves align by index.
        for key, flag in (("chosen", 1), ("rejected", 0)):
            for f in features:
                ids, labs = tokenize_pair(
                    self.tok, f["prompt"], f[key],
                    self.max_prompt_length, self.max_completion_length,
                )
                input_ids.append(torch.tensor(ids, dtype=torch.long))
                labels.append(torch.tensor(labs, dtype=torch.long))
                is_chosen.append(flag)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tok.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tok.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_chosen": torch.tensor(is_chosen, dtype=torch.bool),
        }


ACCELERATE_STATE_MARKERS = ("pytorch_model.bin", "model.safetensors", "optimizer.bin")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Highest-step checkpoint-<step> dir with a full accelerator.save_state
    (skips adapter-only dumps from an incompatible/older checkpoint format)."""
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


def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Sum of log-probs of the completion tokens for each sequence in the batch."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target != -100
    safe_target = target.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, -1, safe_target.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask
    return tok_logp.sum(dim=-1), mask.sum(dim=-1)


def dpo_step(model, batch, beta, nll_gamma):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    is_chosen = batch["is_chosen"]

    # Policy logps (adapter enabled).
    pol_logp, pol_len = sequence_logps(model, input_ids, attention_mask, labels)
    # Reference logps: same model with the LoRA adapter disabled (base weights).
    with torch.no_grad(), model.disable_adapter():
        ref_logp, _ = sequence_logps(model, input_ids, attention_mask, labels)

    chosen = is_chosen
    rejected = ~is_chosen
    pol_w, pol_l = pol_logp[chosen], pol_logp[rejected]
    ref_w, ref_l = ref_logp[chosen], ref_logp[rejected]

    logits = (pol_w - ref_w) - (pol_l - ref_l)
    dpo_loss = -F.logsigmoid(beta * logits).mean()

    loss = dpo_loss
    nll_loss = torch.tensor(0.0, device=loss.device)
    if nll_gamma > 0.0:
        nll_loss = (-pol_w / pol_len[chosen].clamp(min=1)).mean()
        loss = loss + nll_gamma * nll_loss

    with torch.no_grad():
        acc = (logits > 0).float().mean()
        margin = logits.mean()
    stats = {
        "dpo_loss": float(dpo_loss.item()),
        "nll_loss": float(nll_loss.item()),
        "reward_acc": float(acc.item()),
        "reward_margin": float(margin.item()),
    }
    return loss, stats


def train_dpo(args, pref_ds: Optional[Dataset], device):
    from accelerate import Accelerator
    from accelerate.utils import set_seed as accelerate_set_seed

    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerate_set_seed(args.seed)
    is_main = accelerator.is_main_process

    if pref_ds is None:
        pref_ds = load_from_disk(args.pref_data_path)
    if is_main:
        print(f"[train] {len(pref_ds)} preference pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype, trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if is_main:
        model.print_trainable_parameters()
    model.train()

    collator = DPOCollator(tokenizer, args.max_prompt_length, args.max_completion_length)
    loader = DataLoader(
        pref_ds, batch_size=args.per_device_train_batch_size,
        shuffle=True, collate_fn=collator, num_workers=4, drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    total_steps = math.ceil(len(loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scheduler = accelerator.prepare(scheduler)

    if args.report_to == "wandb" and is_main and wandb is not None:
        wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args))

    global_step = 0

    # ── Resume from checkpoint (model, optimizer, scheduler, RNG) ──
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
            if is_main:
                print(f"[train] resumed from checkpoint {resume_dir} (global_step={global_step})", flush=True)
        except Exception as e:
            if is_main:
                print(f"[train] [warn] could not resume from {resume_dir} ({e!r}); "
                      f"starting fresh from global_step=0.", flush=True)
            accelerator.wait_for_everyone()

    if is_main:
        print(f"[train] total update steps: {total_steps}", flush=True)

    t0 = time.time()
    for epoch in range(math.ceil(args.num_train_epochs)):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(model):
                loss, stats = dpo_step(
                    accelerator.unwrap_model(model), batch, args.beta, args.nll_gamma,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if is_main and global_step % args.logging_steps == 0:
                    elapsed = time.time() - t0
                    log = {
                        "train/loss": float(loss.item()),
                        "train/dpo_loss": stats["dpo_loss"],
                        "train/nll_loss": stats["nll_loss"],
                        "train/reward_acc": stats["reward_acc"],
                        "train/reward_margin": stats["reward_margin"],
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/global_step": global_step,
                    }
                    if args.report_to == "wandb" and wandb is not None:
                        wandb.log(log, step=global_step)
                    eta = (total_steps - global_step) / (global_step / elapsed) if global_step else 0
                    print({
                        "step": f"{global_step}/{total_steps}",
                        "loss": round(log["train/loss"], 4),
                        "dpo": round(stats["dpo_loss"], 4),
                        "acc": round(stats["reward_acc"], 3),
                        "margin": round(stats["reward_margin"], 3),
                        "lr": f"{log['train/lr']:.2e}",
                        "eta": f"{int(eta // 3600)}h{int((eta % 3600) // 60):02d}m",
                    }, flush=True)

                # Periodic checkpoint: full training state (model+optimizer+scheduler+
                # RNG) so a killed job resumes instead of losing everything back to
                # the single end-of-run save this loop used to have.
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state is a collective call — every rank must
                    # reach it, not just is_main, or non-main ranks hang on its barrier.
                    accelerator.save_state(checkpoint_dir)
                    if is_main:
                        accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir)
                             if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()],
                            key=lambda x: int(x.split("-")[-1]),
                        )
                        for old in checkpoints[:-3]:
                            shutil.rmtree(os.path.join(args.output_dir, old))
                        print(f"[train] checkpoint saved @ step {global_step}", flush=True)
                    accelerator.wait_for_everyone()

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    if is_main:
        print(f"[train] saving to {args.output_dir} ...", flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        if args.save_merged:
            merged_out = os.path.join(args.output_dir, "merged")
            os.makedirs(merged_out, exist_ok=True)
            print("[train] merging LoRA into base weights ...", flush=True)
            del unwrapped, model
            torch.cuda.empty_cache()
            base_for_merge = AutoModelForCausalLM.from_pretrained(
                args.model_id, dtype=dtype, trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base_for_merge, args.output_dir)
            merged = merged.merge_and_unload()
            merged.save_pretrained(merged_out)
            tokenizer.save_pretrained(merged_out)
            print(f"[train] saved merged model to {merged_out}", flush=True)

        if args.report_to == "wandb" and wandb is not None:
            wandb.finish()


# ─────────────────────────────────────────────
# Args / main
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["construct", "train", "all"], default="all")

    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--dataset_id", type=str, default=HF_DATASET_ID)
    ap.add_argument("--dataset_split", type=str, default="train")
    ap.add_argument("--dataset_config", type=str, default=HF_DATASET_CONFIG,
                    help="HF config name; PolyFact-Clean requires 'parallel'. "
                         "Pass '' for single-config datasets like WIKI-FACT.")
    ap.add_argument("--pref_data_path", type=str, required=True,
                    help="Where the constructed preference dataset is saved/loaded (save_to_disk).")
    ap.add_argument("--overwrite_pref_data", action="store_true", default=False)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    # construction
    ap.add_argument("--embedding_model", type=str, default="sentence-transformers/LaBSE",
                    help="Multilingual encoder for consistency scoring. CM-Align used "
                         "Alibaba-NLP/gte-multilingual-base; LaBSE is this project's "
                         "cross-lingual alignment encoder and the default here.")
    ap.add_argument("--max_facts", type=int, default=8000)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Split construction across N disjoint fact shards (one GPU each).")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_candidates", type=int, default=4)
    ap.add_argument("--min_languages", type=int, default=12)
    ap.add_argument("--include_en_pairs", action="store_true", default=False,
                    help="Also build a preference pair for English (chosen=en reference).")
    ap.add_argument("--min_margin", type=float, default=0.0,
                    help="Drop a pair if cos(chosen)-cos(rejected) < this (0 keeps all).")
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--facts_per_gen_batch", type=int, default=16,
                    help="Facts grouped into one generation super-batch before the "
                         "single embedder.encode() call for the whole group. Was 1 "
                         "(one generate() + up to 12 tiny encode() calls per fact) — "
                         "the throughput fix; raise if GPU memory allows.")
    ap.add_argument("--gen_micro_batch_size", type=int, default=768,
                    help="Flat (fact,lang,candidate) sequences per generate() call. "
                         "16 facts * ~12 langs * 4 candidates = ~768 fits one call "
                         "on a GH200 96GB for Qwen/OLMo-7B; lower if you OOM.")
    ap.add_argument("--checkpoint_every", type=int, default=500,
                    help="Save the partial preference dataset to --pref_data_path every "
                         "N facts, so a killed job resumes instead of losing all progress "
                         "(the loop only used to save once, at the very end).")

    # DPO
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll_gamma", type=float, default=0.0,
                    help="Weight of the auxiliary NLL term on the chosen response "
                         "(CM-Align: 0.0 for GIF, 1.0 for Math).")
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200,
                    help="Full training-state checkpoint (model+optimizer+scheduler+RNG) "
                         "every N update steps, 0 to disable. Previously this loop only "
                         "saved once at the very end.")
    ap.add_argument("--resume_from_checkpoint", type=str, default="auto",
                    help="'auto' resumes from the latest checkpoint-<step> dir under "
                         "--output_dir if one exists, 'none' always starts fresh, or "
                         "pass an explicit checkpoint dir.")
    ap.add_argument("--save_merged", action="store_true", default=True)
    ap.add_argument("--no_save_merged", dest="save_merged", action="store_false")

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--prompt_scaffold", type=str, default="native",
                    choices=["native", "en"],
                    help="Language of the FREE-FORM prompt scaffold. 'native' "
                         "localises it per language (default); 'en' restores the "
                         "old English scaffold for an apples-to-apples ablation. "
                         "Does NOT affect the log-likelihood MCQ path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", type=str, default="wandb")
    args = ap.parse_args()
    if args.no_bf16:
        args.bf16 = False
    if args.phase in ("train", "all") and not args.output_dir:
        ap.error("--output_dir is required for the train phase")
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pref_ds = None
    if args.phase in ("construct", "all"):
        # construct_preference_data checks --pref_data_path's own checkpoint metadata:
        # skips work if already marked done, resumes mid-shard from a partial checkpoint
        # otherwise, or starts fresh if --overwrite_pref_data is passed.
        pref_ds = construct_preference_data(args, device)

    if args.phase in ("train", "all"):
        train_dpo(args, pref_ds if args.phase == "all" else None, device)


if __name__ == "__main__":
    main()
