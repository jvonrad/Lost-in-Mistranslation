#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCO baseline on PolyFact-Clean — reviewer-requested CLC baseline.

Port of "Post-Training Language Models for Crosslingual Consistency"
(Liu, Qi, et al., ICML 2026; github.com/Betswish/ConsistencyRL) to this
project's data and training stack. Their released stack pins torch 2.7.1 +
transformers 4.55 + a bundled trl fork — torch 2.7.x has no aarch64/cu126
wheel on Isambard, so instead of running their trainer we port the method:

  * DATA (faithful to their data/sampling.py, BMLAMA-style): one instance =
    the same fact in two languages, with a chosen and a rejected candidate
    picked BY SHARED INDEX from the parallel option lists. The chosen
    candidate is RANDOM, not gold — the objective is label-free; it trains
    cross-lingual *agreement*, not correctness. PolyFact-Clean's `option_ids`
    (Wikidata QIDs, per language) give the exact index alignment their
    method requires; options are independently shuffled per language, so
    lang2's options are re-ordered into lang1's index order via QID here.

  * LOSS (ported 1:1 from their trl/trainer/dco_trainer.py::dco_loss and
    unit-tested numerically against the original source):
        reward_i = (pi_chosen_i - ref_chosen_i) - (pi_rejected_i - ref_rejected_i)
        offset_i = ref_chosen_i - ref_rejected_i
        loss = |reward_1 - (1/beta)*offset_2| + |reward_2 - beta*offset_1|
    Sequence logprobs are SUMS over completion tokens (their get_batch_logps
    convention, standard DPO).

  * REFERENCE MODEL: the policy with its LoRA adapter disabled (exact — the
    adapter is zero-init; proven equal to a separately loaded base model),
    instead of their second full-model copy. Deviation from their full-FT
    setup: we train LoRA r128/alpha 256 to stay in this project's
    post-2026-08-02 run family; state this in the paper.

Construction needs no generation (candidates come from the dataset), so this
is a single-phase trainer and, with no rollouts, costs ~an order of magnitude
less than a GRPO arm.

Launch: sbatch cluster/dco_baseline.sbatch  (see that file for configs).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polyfact_schema as pfs  # noqa: E402

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
WANDB_PROJECT = "UnLock"


# ─────────────────────────────────────────────
# DCO loss — ported 1:1 from ConsistencyRL trl/trainer/dco_trainer.py
# ─────────────────────────────────────────────

def dco_loss(
    chosen_1_logps: torch.Tensor,
    rejected_1_logps: torch.Tensor,
    chosen_2_logps: torch.Tensor,
    rejected_2_logps: torch.Tensor,
    ref_chosen_1_logps: torch.Tensor,
    ref_rejected_1_logps: torch.Tensor,
    ref_chosen_2_logps: torch.Tensor,
    ref_rejected_2_logps: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Their dco_loss verbatim (self.* -> args); tested against the original."""
    offset_1 = ref_chosen_1_logps - ref_rejected_1_logps
    offset_2 = ref_chosen_2_logps - ref_rejected_2_logps
    reward_chosen_1 = chosen_1_logps - ref_chosen_1_logps - rejected_1_logps + ref_rejected_1_logps
    reward_chosen_2 = chosen_2_logps - ref_chosen_2_logps - rejected_2_logps + ref_rejected_2_logps

    losses = torch.abs(reward_chosen_1 - (1.0 / beta) * offset_2) + \
        torch.abs(reward_chosen_2 - beta * offset_1)
    return losses


# ─────────────────────────────────────────────
# Instance construction (their data/sampling.py shape, PolyFact-Clean source)
# ─────────────────────────────────────────────

def build_prompt(question: str) -> str:
    # The eval-side prompt (evaluate_accuracy.py), so DCO's checkpoint is
    # trained on exactly the distribution the paper's metrics probe.
    return f"Question: {question}\nAnswer:"


def build_dco_instances(
    rows, langs: List[str], num_instances: int, seed: int,
) -> List[Dict[str, Any]]:
    """One instance per sampled fact: same fact, two languages, chosen/rejected
    candidates index-aligned across the pair via option_ids."""
    rng = random.Random(seed)
    instances = []
    order = list(range(len(rows)))
    rng.shuffle(order)

    for ridx in order:
        if len(instances) >= num_instances:
            break
        ex = rows[ridx]
        blocks = pfs.lang_blocks(ex)
        avail = [l for l in langs if l in blocks]
        if len(avail) < 2:
            continue
        lang1, lang2 = rng.sample(avail, 2)

        p1 = pfs.normalize_lang_item(blocks[lang1])
        p2 = pfs.normalize_lang_item(blocks[lang2])
        if p1 is None or p2 is None:
            continue
        q1, opts1, _, _ = p1
        q2, opts2, _, _ = p2

        ids1 = blocks[lang1].get("option_ids")
        ids2 = blocks[lang2].get("option_ids")
        if not ids1 or not ids2 or set(ids1) != set(ids2) or len(set(ids1)) != 4:
            continue  # exact QID alignment required; drop rather than guess
        # Reorder lang2's options into lang1's index order.
        pos2 = {qid: k for k, qid in enumerate(ids2)}
        opts2_aligned = [opts2[pos2[qid]] for qid in ids1]

        chosen_id, rejected_id = rng.sample(range(4), 2)
        instances.append({
            "prompt_1": build_prompt(q1),
            "chosen_1": " " + opts1[chosen_id],
            "rejected_1": " " + opts1[rejected_id],
            "prompt_2": build_prompt(q2),
            "chosen_2": " " + opts2_aligned[chosen_id],
            "rejected_2": " " + opts2_aligned[rejected_id],
            "fact_id": ex.get("fact_id", ""),
            "lang_1": lang1,
            "lang_2": lang2,
        })
    return instances


# ─────────────────────────────────────────────
# Tokenization + sequence logprobs (sum over completion tokens)
# ─────────────────────────────────────────────

def encode_pair(tokenizer, prompt: str, completion: str, max_length: int):
    p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = (p + c)[:max_length]
    n_completion = max(0, min(len(c), max_length - len(p)))
    return ids, n_completion


def batch_seq_logps(model, input_ids, attention_mask, completion_mask):
    """Sum of label logprobs over completion tokens (DPO/DCO convention)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    lps = torch.gather(torch.log_softmax(logits.float(), dim=-1), -1,
                       targets.unsqueeze(-1)).squeeze(-1)
    return (lps * completion_mask[:, 1:]).sum(-1)


def collate(batch, tokenizer, max_length, device):
    """Flatten to 4*bs sequences ordered [c1 | r1 | c2 | r2] blocks."""
    seqs, comp_ns = [], []
    for key_p, key_c in (("prompt_1", "chosen_1"), ("prompt_1", "rejected_1"),
                         ("prompt_2", "chosen_2"), ("prompt_2", "rejected_2")):
        for ex in batch:
            ids, n_c = encode_pair(tokenizer, ex[key_p], ex[key_c], max_length)
            seqs.append(torch.tensor(ids, dtype=torch.long))
            comp_ns.append(n_c)
    input_ids = pad_sequence(seqs, batch_first=True,
                             padding_value=tokenizer.pad_token_id).to(device)
    attn = torch.zeros_like(input_ids)
    comp = torch.zeros_like(input_ids, dtype=torch.float)
    for i, (s, n_c) in enumerate(zip(seqs, comp_ns)):
        attn[i, :len(s)] = 1
        if n_c > 0:
            comp[i, len(s) - n_c:len(s)] = 1.0
    return input_ids, attn, comp


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dataset_id", default="jvonrad/PolyFact-Clean")
    ap.add_argument("--dataset_config", default="parallel")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--num_instances", type=int, default=40000,
                    help="Training instances (1 per sampled fact). Their paper "
                         "used 5000; 40000 matches this project's GRPO data budget.")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--beta", type=float, default=1.0, help="Their default.")
    ap.add_argument("--learning_rate", type=float, default=1e-5, help="Their default.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=4, help="Their default.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--report_to", default="wandb")
    return ap.parse_args()


def main():
    args = parse_args()
    langs = [l for l in args.langs.split(",") if l]
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset_id}/{args.dataset_config} ...", flush=True)
    raw = pfs.load_split_dict(args.dataset_id, args.dataset_config)["train"]
    instances = build_dco_instances(raw, langs, args.num_instances, args.seed)
    print(f"Built {len(instances):,} DCO instances "
          f"(requested {args.num_instances:,})", flush=True)
    with open(os.path.join(args.output_dir, "dco_instances_meta.json"), "w") as f:
        json.dump({"n": len(instances), "seed": args.seed,
                   "langs": langs, "example": instances[0]}, f, ensure_ascii=False, indent=1)

    print(f"Loading model {args.model_id} ...", flush=True)
    dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    bs = args.per_device_train_batch_size
    steps_per_epoch = math.ceil(len(instances) / bs)
    total_updates = math.ceil(steps_per_epoch * args.num_train_epochs
                              / args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates)

    use_wandb = args.report_to == "wandb" and wandb is not None
    if use_wandb:
        wandb.init(project=WANDB_PROJECT, name=args.run_name, config=vars(args))

    print(f"Training: {len(instances):,} instances, bs {bs}, "
          f"{total_updates:,} optimizer updates", flush=True)
    model.train()
    global_step = 0
    t0 = time.time()
    running = []

    for epoch in range(math.ceil(args.num_train_epochs)):
        for i in range(0, len(instances), bs):
            batch = instances[i:i + bs]
            n = len(batch)
            input_ids, attn, comp = collate(batch, tokenizer, args.max_length, device)

            # Reference logps: the SAME weights with the adapter disabled —
            # exact zero-init reference, no second model (proven elsewhere in
            # this repo; see --ref_impl adapter_off in the GRPO trainer).
            with torch.no_grad(), model.disable_adapter():
                ref = batch_seq_logps(model, input_ids, attn, comp)
            pol = batch_seq_logps(model, input_ids, attn, comp)

            c1, r1, c2, r2 = pol.split(n)
            rc1, rr1, rc2, rr2 = ref.split(n)
            losses = dco_loss(c1, r1, c2, r2, rc1, rr1, rc2, rr2, beta=args.beta)
            loss = losses.mean() / args.gradient_accumulation_steps
            loss.backward()

            if ((i // bs) + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                running.append(float(loss.item()) * args.gradient_accumulation_steps)
                if global_step % args.logging_steps == 0:
                    # agreement: do the two languages prefer the same candidate?
                    with torch.no_grad():
                        agree = (((c1 - r1) > 0) == ((c2 - r2) > 0)).float().mean().item()
                    stats = {
                        "train/loss": sum(running) / len(running),
                        "train/pref_agreement": agree,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "perf/sec_per_step": (time.time() - t0) / max(global_step, 1),
                    }
                    running.clear()
                    print({k: round(v, 5) if isinstance(v, float) else v
                           for k, v in stats.items()},
                          f"({global_step}/{total_updates})", flush=True)
                    if use_wandb:
                        wandb.log(stats, step=global_step)

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

            if global_step >= total_updates:
                break
        if global_step >= total_updates:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    merged = model.merge_and_unload()
    merged_dir = os.path.join(args.output_dir, "merged")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    hours = (time.time() - t0) / 3600
    print(f"Done: {global_step} updates in {hours:.2f} h; merged model at {merged_dir}",
          flush=True)
    if use_wandb:
        wandb.log({"cost/wall_clock_hours": hours}, step=global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
