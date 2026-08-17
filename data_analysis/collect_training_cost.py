#!/usr/bin/env python
"""Collect per-run training cost from the training_stats.json sidecars.

The GRPO trainer writes a sidecar next to every checkpoint AND at the end of
training, each carrying CUMULATIVE totals restored across resumes. Summing the
files therefore multiplies every run by its checkpoint count -- the totals must
be deduplicated to one record per run, keeping the largest (latest) one.

The sidecars live only on Lustre, so this script exists to distil them into a
committed artefact (results/training_cost.json + a LaTeX table) before that
scratch space goes away.

Usage:
  python data_analysis/collect_training_cost.py
  python data_analysis/collect_training_cost.py --latex
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re

PROJ = "/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation"
OUT = "results/training_cost.json"

# run dir -> (family, method) for the runs the paper reports. Anything not
# listed is still collected, just not shown in the paper table.
PAPER = {
    "olmo-sft-n10000": ("OLMo", "SFT"),
    "olmo-cpt-sft-n10000": ("OLMo", "CPT+SFT"),
    "olmo-dco-n10000-r128": ("OLMo", "DCO"),
    "olmo-cpt-dco-n10000-r128": ("OLMo", "CPT+DCO"),
    "olmo-cmalign-n10000": ("OLMo", "CM-Align"),
    "final/olmo-final-ladder-clip5": ("OLMo", "GRPO"),
    "qwen-sft-n10000": ("Qwen", "SFT"),
    "qwen-cpt-sft-n10000": ("Qwen", "CPT+SFT"),
    "qwen-dco-n10000-r128": ("Qwen", "DCO"),
    "qwen-cpt-dco-n10000-r128": ("Qwen", "CPT+DCO"),
    "qwen-cmalign-n10000": ("Qwen", "CM-Align"),
    "sweep/qwen-sweep-clip5": ("Qwen", "GRPO"),
}


def run_root(path):
    """Strip the trailing checkpoint-N component so a run's sidecars group."""
    rel = os.path.relpath(os.path.dirname(path), f"{PROJ}/models")
    return re.sub(r"/checkpoint-\d+$", "", rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    best = {}
    for root, _, files in os.walk(f"{PROJ}/models"):
        if "training_stats.json" not in files:
            continue
        p = os.path.join(root, "training_stats.json")
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        gh = d.get("cumulative_gpu_hours") or d.get("gpu_hours") or 0.0
        rec = {"gpu_hours": gh,
               "wall_seconds": d.get("cumulative_wall_seconds")
               or d.get("wall_seconds") or 0.0,
               "rollout_tokens": d.get("cumulative_rollout_tokens") or 0,
               "steps": d.get("global_step") or d.get("step") or 0}
        k = run_root(p)
        # Cumulative fields only grow, so the largest record is the latest.
        if k not in best or rec["gpu_hours"] > best[k]["gpu_hours"]:
            best[k] = rec

    os.makedirs("results", exist_ok=True)
    io.open(OUT, "w", encoding="utf-8").write(
        json.dumps(best, indent=1, sort_keys=True))
    tot = sum(r["gpu_hours"] for r in best.values())
    print(f"{len(best)} distinct runs, {tot:.1f} GPU-h total -> {OUT}")

    print(f"\n{'run':<42}{'GPU-h':>9}{'steps':>9}{'Mtok':>9}")
    for k, r in sorted(best.items(), key=lambda x: -x[1]["gpu_hours"])[:15]:
        print(f"  {k:<40}{r['gpu_hours']:>9.1f}{r['steps']:>9}"
              f"{r['rollout_tokens'] / 1e6:>9.1f}")

    if not a.latex:
        return
    print("\n% --- paper cost table ---")
    print(r"\begin{tabular}{llrr}")
    print(r"\toprule")
    print(r"Model & Method & GPU-h & Steps \\")
    print(r"\midrule")
    for fam in ("OLMo", "Qwen"):
        for k, (f, meth) in PAPER.items():
            if f != fam or k not in best:
                continue
            r = best[k]
            print(f"{fam} & {meth} & {r['gpu_hours']:.1f} & {r['steps']:,} \\\\")
        if fam == "OLMo":
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
