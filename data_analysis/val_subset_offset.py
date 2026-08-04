#!/usr/bin/env python
"""Measure the in-loop eval-set offset between --max_eval_wikifact 150 and 100.

WHY THIS EXISTS
---------------
The 1,500-step sweep arms were launched with `--max_eval_wikifact 150`; the
10,000-step final arms with `100`. The trainer builds the eval set as

    val_ds = val_ds.select(range(min(args.max_eval_wikifact, len(val_ds))))

so the 100-fact set is the *first 100 rows* of the 150-fact set. The two runs
are therefore scored on different item pools, and every sweep-vs-final metric
comparison carries a fixed composition offset on top of any real difference.

This script measures that offset on the UNTRAINED base model, where there is no
training effect to confound it: whatever gap appears here is present, unchanged,
in every step-matched comparison between the two families of runs.

It reuses the trainer's own metric functions, so the numbers are the same
quantities the training logs print -- not a re-implementation.

Usage:
  python data_analysis/val_subset_offset.py --model Qwen/Qwen2.5-7B
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

import polyfact_schema as pfs  # noqa: E402
from train_wikifact_grpo_accelerate import (  # noqa: E402
    build_grouped_fact_item,
    compute_polyfact_logprob_metrics,
    evaluate_polyfact_freeform,
)

KEYS = [
    ("polyfact/freeform_accuracy", "ffAcc"),
    ("polyfact/freeform_resolution_rate", "ffRes"),
    ("polyfact/freeform_total_consistency", "ffTotC"),
    ("polyfact/mcq_accuracy", "mcqAcc"),
    ("polyfact/mcq_total_consistency", "mcqTotC"),
    ("consistency/rankc_exact_avg", "rankcX"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dataset_id", default="jvonrad/PolyFact-Clean")
    ap.add_argument("--dataset_config", default="parallel")
    ap.add_argument("--prompt_scaffold", default="native")
    ap.add_argument("--task_format", default="mcq")
    ap.add_argument("--min_languages", type=int, default=12)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=48)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    raw = pfs.load_split_dict(a.dataset_id, a.dataset_config)["validation"]
    kw = {"scaffold": a.prompt_scaffold, "task_format": a.task_format}
    ds = raw.map(build_grouped_fact_item, fn_kwargs=kw)
    ds = ds.filter(lambda x: x["is_valid"] and x["num_languages"] >= a.min_languages)
    keep = ["fact_id", "prompts_by_lang_json", "meta_by_lang_json"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    print(f"validation pool after filtering: {len(ds)} facts", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    dev = next(model.parameters()).device

    subsets = {
        "val[0:100]  (final runs)": ds.select(range(100)),
        "val[0:150]  (sweep runs)": ds.select(range(150)),
        "val[100:150] (the tail only)": ds.select(range(100, 150)),
    }

    rows = {}
    for name, sub in subsets.items():
        with torch.no_grad():
            m = evaluate_polyfact_freeform(
                model, tok, sub, a.max_prompt_length, a.max_completion_length
            )
            m.update(
                compute_polyfact_logprob_metrics(
                    model, tok, sub, a.max_prompt_length, dev
                )
            )
        rows[name] = m
        print(f"  done {name}", flush=True)

    hdr = f"{'subset':<30}" + "".join(f"{s:>9}" for _, s in KEYS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, m in rows.items():
        print(f"{name:<30}" + "".join(f"{m.get(k, float('nan')):>9.3f}" for k, _ in KEYS))

    a100 = rows["val[0:100]  (final runs)"]
    a150 = rows["val[0:150]  (sweep runs)"]
    print("\nOFFSET the sweep runs get for free (150-set minus 100-set), in pp:")
    print(f"{'':<30}" + "".join(f"{s:>9}" for _, s in KEYS))
    print(f"{'delta':<30}" + "".join(
        f"{100 * (a150.get(k, 0) - a100.get(k, 0)):>+9.2f}" for k, _ in KEYS))


if __name__ == "__main__":
    main()
