#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_tokalign_cpt.py \
  --model_path /data/jonathan/Lost-in-Mistranslation/tokalign/olmo2_to_custom151k/TokAlign-Init-7B \
  --dataset_path /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/olmo_new_tokenizer_packed_1024 \
  --output_dir /data/jonathan/Lost-in-Mistranslation/models/olmo2_to_custom151k/cpt_run_full \
  --wikifact_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/multilingual_mcq_text_filtered_zh_simplified_val.jsonl \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --logging_steps 10 \
  --eval_steps 500 \
  --save_steps 500 \
  --num_train_epochs 1 \
  --wikifact_max_examples_per_lang 250 \
  --run_name olmo2_custom151k_tokalign_cpt_full 
'''

import os
import math
import json
import argparse

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from collections import defaultdict

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
import torch.distributed as dist
import wandb
import os

MAX_SEQ_LEN = 1024
LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0

class PretokenizedCausalCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int = 1024):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, examples):
        trunc_lens = [min(len(x["input_ids"]), self.max_seq_len) for x in examples]
        max_len = max(trunc_lens)

        input_ids = []
        attention_mask = []
        labels = []

        for ex in examples:
            ids = ex["input_ids"][:self.max_seq_len]
            mask = ex.get("attention_mask", [1] * len(ex["input_ids"]))[:self.max_seq_len]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(ids + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
class TokAlignTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        logs = dict(logs)

        if "loss" in logs:
            logs["train_perplexity"] = math.exp(logs["loss"])

        super().log(logs, start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # keep lang out of model forward
        inputs_for_model = dict(inputs)

        outputs = model(**inputs_for_model)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss



def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] skipping bad json line {i}: {e}")
    return rows


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def score_candidates_batch(
    model,
    tokenizer,
    examples,
    device,
    score_mode="avg",
    max_length=2048,
):
    flat_texts = []
    meta = []

    for ex_idx, ex in enumerate(examples):
        prompt = ex["prompt"]
        for opt_idx, opt in enumerate(ex["options"]):
            full_text = prompt + " " + opt
            flat_texts.append(full_text)
            meta.append((ex_idx, opt_idx, prompt, opt))

    enc = tokenizer(
        flat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        token_logprobs = torch.gather(
            logprobs, 2, target_ids.unsqueeze(-1)
        ).squeeze(-1)

    prompt_token_lists = tokenizer(
        [m[2] for m in meta],
        padding=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_lens = [len(x) for x in prompt_token_lists]

    scores = [[None] * len(ex["options"]) for ex in examples]

    for row_idx, (ex_idx, opt_idx, _prompt, _opt) in enumerate(meta):
        plen = prompt_lens[row_idx]
        seq_len = int(attention_mask[row_idx].sum().item())

        start = max(plen - 1, 0)
        end = seq_len - 1

        opt_lp = token_logprobs[row_idx, start:end]
        opt_mask = target_mask[row_idx, start:end]
        opt_lp = opt_lp[opt_mask.bool()]

        if opt_lp.numel() == 0:
            score = -1e9
        else:
            if score_mode == "sum":
                score = float(opt_lp.sum().item())
            elif score_mode == "avg":
                score = float(opt_lp.mean().item())
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

        scores[ex_idx][opt_idx] = score

    return scores


def evaluate_wikifact(
    model,
    tokenizer,
    input_jsonl,
    batch_size=8,
    max_examples_per_lang=0,
    score_mode="avg",
    max_length=2048,
):
    rows = load_rows(input_jsonl)
    device = next(model.parameters()).device

    per_lang_correct = defaultdict(int)
    per_lang_total = defaultdict(int)

    overall_correct = 0
    overall_total = 0

    model_was_training = model.training
    model.eval()

    for lang in LANGS:
        lang_examples = []
        

        for row in rows:
            if "langs" not in row or lang not in row["langs"]:
                continue
            item = row["langs"][lang]

            question = item.get("question", "").strip()
            options = item.get("options", [])
            gold = item.get("answer_text", "").strip()

            if not question or not isinstance(options, list) or len(options) != 4 or gold not in options:
                continue

            lang_examples.append({
                "fact_id": row.get("fact_id"),
                "lang": lang,
                "prompt": build_prompt(question),
                "options": options,
                "gold": gold,
            })
        
        print(f"[WIKI-FACT] Starting lang={lang} with {len(lang_examples)} examples")

        if max_examples_per_lang > 0:
            lang_examples = lang_examples[:max_examples_per_lang]

        for i in range(0, len(lang_examples), batch_size):
            batch = lang_examples[i:i + batch_size]
            score_lists = score_candidates_batch(
                model=model,
                tokenizer=tokenizer,
                examples=batch,
                device=device,
                score_mode=score_mode,
                max_length=max_length,
            )

            for ex, scores in zip(batch, score_lists):
                pred_idx = max(range(len(scores)), key=lambda k: scores[k])
                pred = ex["options"][pred_idx]
                correct = int(pred == ex["gold"])

                per_lang_correct[lang] += correct
                per_lang_total[lang] += 1
                overall_correct += correct
                overall_total += 1

    if model_was_training:
        model.train()

    metrics = {}
    for lang in LANGS:
        total = per_lang_total[lang]
        acc = per_lang_correct[lang] / max(total, 1)
        metrics[f"wikifact_acc/{lang}"] = acc
        metrics[f"wikifact_n/{lang}"] = total

    metrics["wikifact_acc/overall"] = overall_correct / max(overall_total, 1)
    metrics["wikifact_n/overall"] = overall_total
    return metrics


class WandbExtraMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        wikifact_jsonl=None,
        wikifact_batch_size=8,
        wikifact_max_examples_per_lang=250,
        wikifact_score_mode="avg",
        wikifact_max_length=2048,
        probe_prompts=None,
    ):
        self.tokenizer = tokenizer
        self.wikifact_jsonl = wikifact_jsonl
        self.wikifact_batch_size = wikifact_batch_size
        self.wikifact_max_examples_per_lang = wikifact_max_examples_per_lang
        self.wikifact_score_mode = wikifact_score_mode
        self.wikifact_max_length = wikifact_max_length
        self.probe_prompts = probe_prompts or []

    def _log_generation_probes(self, model, step):
        if not is_main_process() or wandb is None or len(self.probe_prompts) == 0:
            return

        model_was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        rows = []
        for prompt in self.probe_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            rows.append([prompt, text])

        if model_was_training:
            model.train()
            
            
        if is_main_process():
            wandb.log(
                {"generation_probes": wandb.Table(data=rows, columns=["prompt", "output"])},
                step=step,
            )

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return

        # compute perplexity from eval_loss if available
        eval_logs = {}
        if metrics is not None and "eval_loss" in metrics:
            try:
                eval_logs["perplexity/validation"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                eval_logs["perplexity/validation"] = float("inf")

        # run WIKI-FACT only on main process
        if is_main_process() and self.wikifact_jsonl is not None:
            wikifact_metrics = evaluate_wikifact(
                model=model.module if hasattr(model, "module") else model,
                tokenizer=self.tokenizer,
                input_jsonl=self.wikifact_jsonl,
                batch_size=self.wikifact_batch_size,
                max_examples_per_lang=self.wikifact_max_examples_per_lang,
                score_mode=self.wikifact_score_mode,
                max_length=self.wikifact_max_length,
            )
            eval_logs.update(wikifact_metrics)

        if is_main_process() and wandb is not None and len(eval_logs) > 0:
            wandb.log(eval_logs, step=state.global_step)

        if model is not None:
            self._log_generation_probes(
                model=model.module if hasattr(model, "module") else model,
                step=state.global_step,
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--wikifact_jsonl", default=None)
    ap.add_argument("--wikifact_batch_size", type=int, default=8)
    ap.add_argument("--wikifact_max_examples_per_lang", type=int, default=250)
    ap.add_argument("--wikifact_score_mode", choices=["sum", "avg"], default="avg")
    ap.add_argument("--wikifact_max_length", type=int, default=2048)

    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)

    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.01)

    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=3)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", type=str, default="tokalign-cpt")
    ap.add_argument("--project", type=str, default="tokalign-cpt")

    args = ap.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.dataset_path)
    train_ds = dataset["train"]
    eval_ds = dataset["validation"].select(range(min(5000, len(dataset["validation"]))))


    if wandb is not None and is_main_process():
        wandb.init(
            project="UnLock",
            name=args.run_name,
            config=vars(args),
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        #attn_implementation="flash_attention_2"
    )
    model.config.use_cache = False

    collator = PretokenizedCausalCollator(tokenizer.pad_token_id)
    
    warmup_steps = int(args.max_steps * args.warmup_ratio) if args.max_steps > 0 else 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        bf16=True,
        tf32=False,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        report_to="wandb" if wandb is not None else "none",
        run_name=args.run_name,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    probe_prompts = [
        "The capital of France is",
        "Was ist die Hauptstadt von Frankreich?",
        "Jelaskan apa itu black hole.",
        "ترجم إلى العربية: The weather is nice today.",
    ]

    trainer = TokAlignTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[
            WandbExtraMetricsCallback(
                tokenizer=tokenizer,
                wikifact_jsonl=args.wikifact_jsonl,
                wikifact_batch_size=args.wikifact_batch_size,
                wikifact_max_examples_per_lang=args.wikifact_max_examples_per_lang,
                wikifact_score_mode=args.wikifact_score_mode,
                wikifact_max_length=args.wikifact_max_length,
                probe_prompts=probe_prompts,
            )
        ],
    )

    trainer.tokenizer = tokenizer
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_metrics = trainer.evaluate()
    if is_main_process():
        print("Final eval metrics:", final_metrics)

    if wandb is not None and is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()