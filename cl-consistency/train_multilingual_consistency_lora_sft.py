#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import wandb

'''
python train_multilingual_consistency_lora_sft.py \
  --hf_dataset_id jonny-vr/WIKI-FACT \
  --model_id allenai/OLMo-2-1124-7B \
  --facts_per_device_batch 1 \
  --gradient_accumulation_steps 8 \
  --num_epochs 1 \
  --ckpt_dir /data/jonathan/Lost-in-Mistranslation/models/wikifact_sft_lora_ckpts \
  --output_dir /data/jonathan/Lost-in-Mistranslation/models/wikifact_sft_lora \
  --run_name wiki-fact-sft-lora-75
'''


MODEL_ID = "allenai/OLMo-2-1124-7B"
HF_DATASET_ID = "jonny-vr/WIKI-FACT"

CKPT_DIR = "/tmp/olmo2_wikifact_ckpts"
OUTPUT_DIR = "/tmp/olmo2_wikifact_final"

LORA_R = 32
LORA_ALPHA = 64

FACTS_PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
NUM_EPOCHS = 1

WANDB_PROJECT = "UnLock"
CONSISTENCY_WEIGHT = 0.5

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
MAX_EVAL_SAMPLES_PER_LANG = 200

CHOICE_TOKEN_IDS = None
rank = int(os.environ.get("RANK", 0))


def print0(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


@dataclass
class CausalLMPaddingCollator:
    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        fact_ids = [f["fact_id"] for f in features]
        langs = [f["lang"] for f in features]
        gold_letter_idx = [f["gold_letter_idx"] for f in features]

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
            "fact_id": fact_ids,
            "lang": langs,
            "gold_letter_idx": torch.tensor(gold_letter_idx, dtype=torch.long),
        }


def build_mcq_prompt(question: str, options: List[str]) -> str:
    a, b, c, d = options
    return (
        f"Question: {question}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n"
        f"Answer:"
    )


def mcq_row_to_features(question: str, options: List[str], gold_idx: int, tokenizer):
    prompt = build_mcq_prompt(question, options)
    target = f" {'ABCD'[gold_idx]}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    if len(target_ids) != 1:
        raise ValueError(f"Expected 1-token answer target, got {target_ids}")

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "gold_letter_idx": gold_idx,
    }


def flatten_wikifact_split(raw_split: Dataset, tokenizer, langs: List[str]) -> Dataset:
    rows = []

    for ex in raw_split:
        fact_id = ex["fact_id"]
        langs_dict = ex["langs"]

        for lang in langs:
            if lang not in langs_dict:
                continue

            item = langs_dict[lang]
            question = item["question"]
            options = item["options"]
            answer_text = item["answer_text"]

            if not isinstance(options, list) or len(options) != 4:
                continue
            if answer_text not in options:
                continue

            gold_idx = options.index(answer_text)
            feats = mcq_row_to_features(question, options, gold_idx, tokenizer)

            rows.append({
                "fact_id": fact_id,
                "lang": lang,
                "gold_letter_idx": gold_idx,
                **feats,
            })

    ds = Dataset.from_list(rows)
    ds = ds.sort("fact_id")
    return ds


def filter_lang(ds: Dataset, lang: str) -> Dataset:
    return ds.filter(lambda x: x["lang"] == lang)


def format_global_mmlu_example(ex, tokenizer):
    question = ex["question"].strip()
    a = ex["option_a"].strip()
    b = ex["option_b"].strip()
    c = ex["option_c"].strip()
    d = ex["option_d"].strip()
    gold = ex["answer"].strip()

    gold_idx = "ABCD".index(gold)

    feats = mcq_row_to_features(question, [a, b, c, d], gold_idx, tokenizer)
    return {
        "fact_id": f"mmlu::{question[:64]}",
        "lang": "unknown",
        "gold_letter_idx": gold_idx,
        **feats,
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


def make_mcq_accuracy_fn(tokenizer):
    choice_ids = [
        tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
    ]

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        preds, golds = [], []

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

    return compute_accuracy


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[:, :, CHOICE_TOKEN_IDS]


def extract_answer_logits(outputs_logits: torch.Tensor, labels: torch.Tensor, choice_ids: List[int]) -> torch.Tensor:
    out = []
    for i in range(labels.size(0)):
        pos = torch.where(labels[i] != -100)[0]
        if len(pos) == 0:
            raise ValueError("No answer position found")
        j = int(pos[0].item())
        if j == 0:
            raise ValueError("Answer position at 0 invalid")
        out.append(outputs_logits[i, j - 1, choice_ids])
    return torch.stack(out, dim=0)


class FactGroupedBatchSampler(Sampler[List[int]]):
    """
    Assumes dataset is sorted by fact_id and all rows for one fact are contiguous.
    Produces batches containing N facts * all their language rows.
    """
    def __init__(self, dataset: Dataset, facts_per_batch: int = 1, drop_last: bool = False):
        self.dataset = dataset
        self.facts_per_batch = facts_per_batch
        self.drop_last = drop_last

        self.fact_groups = []
        current = []
        last_fact = None

        for idx, fid in enumerate(dataset["fact_id"]):
            if last_fact is None or fid == last_fact:
                current.append(idx)
            else:
                self.fact_groups.append(current)
                current = [idx]
            last_fact = fid
        if current:
            self.fact_groups.append(current)

    def __iter__(self):
        for i in range(0, len(self.fact_groups), self.facts_per_batch):
            batch_groups = self.fact_groups[i:i + self.facts_per_batch]
            if self.drop_last and len(batch_groups) < self.facts_per_batch:
                continue
            batch = []
            for g in batch_groups:
                batch.extend(g)
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.fact_groups) // self.facts_per_batch
        return math.ceil(len(self.fact_groups) / self.facts_per_batch)


class ConsistencyTrainer(Trainer):
    def __init__(self, *args, consistency_weight: float = 0.5, choice_token_ids: List[int] = None,
                 facts_per_device_batch: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_weight = consistency_weight
        self.choice_token_ids = choice_token_ids
        self.facts_per_device_batch = facts_per_device_batch
        self.ce = torch.nn.CrossEntropyLoss()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        batch_sampler = FactGroupedBatchSampler(
            self.train_dataset,
            facts_per_batch=self.facts_per_device_batch,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        fact_ids = inputs.pop("fact_id")
        _langs = inputs.pop("lang")
        gold_letter_idx = inputs.pop("gold_letter_idx")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        answer_logits = extract_answer_logits(
            outputs_logits=outputs.logits,
            labels=inputs["labels"],
            choice_ids=self.choice_token_ids,
        )  # [B, 4]

        ce_loss = self.ce(answer_logits, gold_letter_idx)

        groups = {}
        for i, fid in enumerate(fact_ids):
            groups.setdefault(fid, []).append(i)

        probs = F.softmax(answer_logits, dim=-1)
        log_probs = F.log_softmax(answer_logits, dim=-1)
        kl_terms = []

        for _, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idxs_t = torch.tensor(idxs, device=answer_logits.device)
            mean_probs = probs[idxs_t].mean(dim=0, keepdim=True).detach()
            for idx in idxs:
                kl = F.kl_div(
                    log_probs[idx:idx+1],
                    mean_probs,
                    reduction="batchmean",
                )
                kl_terms.append(kl)

        consistency_loss = (
            torch.stack(kl_terms).mean()
            if kl_terms else torch.tensor(0.0, device=answer_logits.device)
        )

        loss = ce_loss + self.consistency_weight * consistency_loss

        if self.state.global_step % 10 == 0 and rank == 0:
            self.log({
                "train/ce_loss": ce_loss.detach().item(),
                "train/consistency_loss": consistency_loss.detach().item(),
            })

        return (loss, outputs) if return_outputs else loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--hf_dataset_id", type=str, default=HF_DATASET_ID)
    ap.add_argument("--ckpt_dir", type=str, default=CKPT_DIR)
    ap.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--run_name", type=str)
    ap.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)
    ap.add_argument("--facts_per_device_batch", type=int, default=FACTS_PER_DEVICE_BATCH)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--dataloader_num_workers", type=int, default=32)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--report_to", type=str, default="wandb")
    ap.add_argument("--consistency_weight", type=float, default=CONSISTENCY_WEIGHT)
    ap.add_argument("--max_train_facts", type=int, default=None)
    ap.add_argument("--max_val_facts", type=int, default=None)
    return ap.parse_args()


def maybe_limit_raw_split(ds: Dataset, max_facts: Optional[int]) -> Dataset:
    if max_facts is None:
        return ds
    return ds.select(range(min(max_facts, len(ds))))


def count_unique_facts(flat_ds: Dataset) -> int:
    return len(set(flat_ds["fact_id"]))


def eval_named_dataset(trainer: Trainer, ds: Dataset, name: str):
    metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=name)
    print0(metrics)
    return metrics


def main():
    args = parse_args()

    if rank != 0:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

    if args.no_bf16:
        args.bf16 = False

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print0(f"Loading tokenizer for {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    global CHOICE_TOKEN_IDS
    CHOICE_TOKEN_IDS = [
        tokenizer(" A", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" B", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" C", add_special_tokens=False)["input_ids"][-1],
        tokenizer(" D", add_special_tokens=False)["input_ids"][-1],
    ]

    compute_metrics = make_mcq_accuracy_fn(tokenizer)

    print0(f"Loading HF dataset: {args.hf_dataset_id}")
    raw = load_dataset(args.hf_dataset_id)

    raw_train = maybe_limit_raw_split(raw["train"], args.max_train_facts)
    raw_val = maybe_limit_raw_split(raw["validation"], args.max_val_facts)

    print0("Flattening WIKI-FACT train ...")
    train_ds = flatten_wikifact_split(raw_train, tokenizer, LANGS)

    print0("Flattening WIKI-FACT validation ...")
    val_ds = flatten_wikifact_split(raw_val, tokenizer, LANGS)

    val_by_lang = {lang: filter_lang(val_ds, lang) for lang in LANGS}

    print0(f"Train fact groups: {count_unique_facts(train_ds):,}")
    print0(f"Train rows: {len(train_ds):,}")
    print0(f"Validation fact groups: {count_unique_facts(val_ds):,}")
    print0(f"Validation rows: {len(val_ds):,}")

    print0("Loading Global-MMLU dev eval sets ...")
    global_mmlu_eval = load_global_mmlu_dev_eval_by_lang(LANGS, tokenizer)

    collator = CausalLMPaddingCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    num_train_facts = count_unique_facts(train_ds)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_facts = args.facts_per_device_batch * args.gradient_accumulation_steps * world_size
    max_steps = int((num_train_facts * args.num_epochs) // effective_batch_facts)

    print0(f"Facts per device batch: {args.facts_per_device_batch}")
    print0(f"Effective batch (facts): {effective_batch_facts}")
    print0(f"Max steps: {max_steps}")

    if args.report_to == "wandb" and rank == 0:
        wandb.init(
            project=WANDB_PROJECT,
            name=args.run_name,
            config={
                "model": args.model_id,
                "hf_dataset_id": args.hf_dataset_id,
                "train_facts": num_train_facts,
                "train_rows": len(train_ds),
                "val_rows": len(val_ds),
                "max_steps": max_steps,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "consistency_weight": args.consistency_weight,
                "facts_per_device_batch": args.facts_per_device_batch,
            },
        )

    print0("Loading base model ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=1,  # ignored by custom fact-grouped batch sampler
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        report_to=["wandb"] if (args.report_to == "wandb" and rank == 0) else [],
        remove_unused_columns=False,
    )

    trainer = ConsistencyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # WIKI-FACT validation during training
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        consistency_weight=args.consistency_weight,
        choice_token_ids=CHOICE_TOKEN_IDS,
        facts_per_device_batch=args.facts_per_device_batch,
    )

    print0("Starting training ...")
    trainer.train()

    print0("Final evaluation on WIKI-FACT validation (overall) ...")
    eval_named_dataset(trainer, val_ds, "wikifact_val")

    print0("Final evaluation on WIKI-FACT validation (per language) ...")
    for lang in LANGS:
        eval_named_dataset(trainer, val_by_lang[lang], f"wikifact_val_{lang}")

    print0("Final evaluation on Global-MMLU dev (per language) ...")
    for lang in LANGS:
        eval_named_dataset(trainer, global_mmlu_eval[lang], f"global_mmlu_{lang}")

    if rank == 0:
        lora_out = args.output_dir
        model.save_pretrained(lora_out)
        tokenizer.save_pretrained(lora_out)

        merged_out = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_out, exist_ok=True)

        print0("Reloading base model for merge ...")
        base_model_for_merge = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
        merged_model = PeftModel.from_pretrained(base_model_for_merge, lora_out)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(merged_out)
        tokenizer.save_pretrained(merged_out)

        print0(f"Saved merged model to {merged_out}")

        if args.report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    main()