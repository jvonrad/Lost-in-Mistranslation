#!/usr/bin/env python3
"""Aggregate KLAR per-sample predictions (results/klar/*_klar.json) into
overall / contaminated / clean-subset accuracies per model.

Subsets (per (relation, index), from klar_polyfact_contamination.json):
  contaminated : exact (rel, subj, obj) triple present in PolyFact-train
  clean-shared : same relation universe as PolyFact but triple NOT in train
  non-shared   : relations PolyFact does not cover at all
The reviewer question is whether GRPO's KLAR gains survive on the clean
subsets (i.e., are not memorization of PolyFact-train facts).
"""
import json, os, sys
from collections import defaultdict

R = "results/klar"
ORDER = [
    ("OLMo base", "OLMo-2-1124-7B"),
    ("OLMo CPT", "OLMo-2-1124-7B-TED"),
    ("OLMo SFT", "olmo-2-7b-wikifact-sft"),
    ("OLMo CPT+SFT", "olmo-2-7b-aligned-wikifact-sft"),
    ("OLMo GRPO", "olmo-2-7b-grpo-att-mlp-full"),
    ("OLMo CPT+GRPO", "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"),
    ("Qwen base", "Qwen-2.5-7B"),
    ("Qwen CPT", "Qwen-2.5-7B-TED"),
    ("Qwen SFT", "Qwen-2.5-7B-SFT-CE-random"),
    ("Qwen CPT+SFT", "Qwen-2.5-7B-TED-SFT"),
    ("Qwen GRPO", "Qwen-2.5-7B-grpo-consistent"),
    ("Qwen CPT+GRPO", "Qwen-2.5-7B-TED-grpo"),
]

with open("evaluate/alignments/klar_polyfact_contamination.json") as f:
    lab = json.load(f)
shared = {(r, i) for r, i in lab["shared_relation_keys"]}
contam = {(r, i) for r, i in lab["contaminated_keys"]}

def acc(rows):
    return 100 * sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")

print(f"{'Model':15}{'overall':>9}{'contam':>9}{'clean-sh':>10}{'non-sh':>9}"
      f"{'  n':>7}{'n_c':>6}{'n_cs':>6}{'n_ns':>7}")
for label, base in ORDER:
    path = os.path.join(R, f"{base}_klar.json")
    if not os.path.exists(path):
        print(f"{label:15}  MISSING {path}")
        continue
    with open(path) as f:
        recs = json.load(f)
    if isinstance(recs, dict):
        recs = recs.get("records", recs.get("samples", []))
    c  = [r for r in recs if (r["relation"], r["index"]) in contam]
    cs = [r for r in recs if (r["relation"], r["index"]) in shared
          and (r["relation"], r["index"]) not in contam]
    ns = [r for r in recs if (r["relation"], r["index"]) not in shared]
    print(f"{label:15}{acc(recs):9.2f}{acc(c):9.2f}{acc(cs):10.2f}{acc(ns):9.2f}"
          f"{len(recs):7d}{len(c):6d}{len(cs):6d}{len(ns):7d}")
