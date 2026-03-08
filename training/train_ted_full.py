#!/usr/bin/env python3
# Full-parameter continued-pretraining from a pretokenized HF dataset
#
# Launch:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# torchrun --standalone --nproc_per_node=8 finetune_full_pretok.py
#
# pip install -U "transformers>=4.41" "datasets>=2.19" accelerate wandb

import os
import wandb
import torch

from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


# -----------------
# Config
# -----------------
BASE_MODEL_LLAMA = "meta-llama/Llama-2-7b-hf"
BASE_MODEL_OLMO = "allenai/OLMo-7B-hf"
OLMO_3_7B = "allenai/Olmo-3-7B-Instruct"
OLMO_3_7B_BASE = "allenai/Olmo-3-1025-7B"
OLMO_2_7B_BASE = "allenai/OLMo-2-1124-7B"

BASE_MODEL = OLMO_2_7B_BASE

DATASET_PATH = (
    "/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/OLMo-2-1124-7B-ted2025-pretokenized-num-langs-12-chunk-1024-notags"
)

MAX_STEPS = 300
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM = 8

OUTDIR = (
    f"/data/jonathan/Lost-in-Mistranslation/models/"
    f"{BASE_MODEL.split('/')[-1]}-ted2025-cpt-fullft-{MAX_STEPS}steps-multilingual-notags"
)


# -----------------
# Helpers
# -----------------
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

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch

def get_dist_info():
    try:
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world = dist.get_world_size() if is_dist else 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return is_dist, rank, world, local_rank
    except Exception:
        return False, 0, 1, 0


is_dist, rank, world, local_rank = get_dist_info()


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


print(f"[rank {rank}] script started", flush=True)


# -----------------
# Tokenizer
# -----------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"[rank {rank}] tokenizer loaded", flush=True)


# -----------------
# Dataset
# -----------------
train_ds = load_from_disk(DATASET_PATH)
print(f"[rank {rank}] dataset loaded", flush=True)

# Keep only the columns needed for causal LM training
keep_cols = {"input_ids", "attention_mask", "labels"}
remove_cols = [c for c in train_ds.column_names if c not in keep_cols]
if len(remove_cols) > 0:
    train_ds = train_ds.remove_columns(remove_cols)

print(f"[rank {rank}] dataset columns pruned", flush=True)

if rank == 0:
    print_section("DATASET")
    print_kv("Dataset path", DATASET_PATH)
    print_kv("Num examples", format_int(len(train_ds)))
    print_kv("Columns", ", ".join(train_ds.column_names))

    lengths = [len(train_ds[i]["input_ids"]) for i in range(min(5, len(train_ds)))]
    print_kv("First few sequence lengths", lengths)


# -----------------
# Model
# -----------------
print(f"[rank {rank}] loading model", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.pad_token_id = tok.pad_token_id

print(f"[rank {rank}] model loaded", flush=True)

if rank == 0:
    print_section("MODEL")
    print_kv("Base model", BASE_MODEL)
    print_kv("Max position embeddings", getattr(model.config, "max_position_embeddings", None))
    print_kv("Tokenizer pad token", repr(tok.pad_token))
    print_kv("Tokenizer eos token", repr(tok.eos_token))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_kv("Total parameters", format_int(total_params))
    print_kv("Trainable parameters", format_int(trainable_params))


# -----------------
# Training args
# -----------------
training_args = TrainingArguments(
    output_dir=OUTDIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    logging_steps=5,
    bf16=True,
    optim="adamw_torch",
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, int(0.03 * MAX_STEPS)),
    save_strategy="no",
    report_to=["wandb"] if rank == 0 else [],
    run_name=f"{BASE_MODEL.split('/')[-1]}-ted2025-fullft-{MAX_STEPS}steps",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

if rank == 0:
    print_section("TRAINING")
    print_kv("Per-device batch size", PER_DEVICE_BATCH_SIZE)
    print_kv("Gradient accumulation", GRAD_ACCUM)
    print_kv("World size", world)
    print_kv(
        "Effective global batch size",
        PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * world,
    )
    print_kv("Learning rate", LEARNING_RATE)
    print_kv("Weight decay", WEIGHT_DECAY)
    print_kv("Max steps", MAX_STEPS)
    print_kv("Output dir", OUTDIR)


# -----------------
# Trainer
# -----------------
collator = CausalLMPaddingCollator(tokenizer=tok, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tok,
    data_collator=collator,
)

if rank == 0:
    print_section("SAMPLE")
    ex = train_ds[0]
    print_kv("Sample input length", len(ex["input_ids"]))
    print_kv("First 20 input_ids", ex["input_ids"][:20])

print(f"[rank {rank}] trainer built", flush=True)


# -----------------
# W&B
# -----------------
if rank == 0:
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "lost-in-mistranslation"),
        name=os.getenv("WANDB_NAME", None),
        config={
            "base_model": BASE_MODEL,
            "training_type": "full_ft",
            "dataset_path": DATASET_PATH,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "world_size": world,
        },
    )

print(f"[rank {rank}] starting train()", flush=True)
trainer.train()


# -----------------
# Save
# -----------------
final_out = f"{OUTDIR}/final"
trainer.save_model(final_out)
tok.save_pretrained(final_out)

if rank == 0:
    print_section("SAVE")
    print_kv("Saved full model to", final_out)