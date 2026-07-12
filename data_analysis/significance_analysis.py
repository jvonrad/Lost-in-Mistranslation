#!/usr/bin/env python3
"""Paired bootstrap CIs + significance tests for consistency-eval results.

Reviewers (ACL ARR 8689) asked for confidence intervals, especially for the
small Global-MMLU deltas. All metrics are recomputed per fact from the saved
per-option scores in results/*_consistency.json, validated against the
reported aggregates, then bootstrapped at the fact level (resampling facts
keeps all languages of a fact together, which preserves the cross-language
correlation structure). Deltas between models are paired: the same resampled
fact indices are applied to both models.

Usage:
    python data_analysis/significance_analysis.py \
        [--n_boot 10000] [--seed 0] [--out results/significance]
"""
import argparse
import itertools
import json
import math
import os

import numpy as np

LANGS_PF = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
LANGS_GM = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "fr", "ja", "zh"]  # Lite: no ru
N_OPT = 4
RANKC_W = np.exp(N_OPT - np.arange(1, N_OPT + 1))
RANKC_W = RANKC_W / RANKC_W.sum()

FAMILIES = {
    "OLMo": {
        "Base": "OLMo-2-1124-7B",
        "CPT": "OLMo-2-1124-7B-TED",
        "SFT": "olmo-2-7b-wikifact-sft",
        "CPT+SFT": "olmo-2-7b-aligned-wikifact-sft",
        "GRPO": "OLMo-2-7B-grpo",
        "CPT+GRPO": "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint",
    },
    "Qwen": {
        "Base": "Qwen-2.5-7B",
        "CPT": "Qwen-2.5-7B-TED",
        "SFT": "Qwen-2.5-7B-SFT-CE-random",
        "CPT+SFT": "Qwen-2.5-7B-TED-SFT",
        "GRPO": "Qwen-2.5-7B-grpo-consistent",
        "CPT+GRPO": "Qwen-2.5-7B-TED-grpo",
    },
}
COMPARISONS = [
    ("SFT", "Base"), ("CPT", "Base"), ("GRPO", "Base"),
    ("CPT+SFT", "Base"), ("CPT+GRPO", "Base"), ("GRPO", "SFT"),
]
METRICS = ["acc", "totcons", "rankc", "agree"]
METRIC_LABEL = {"acc": "Avg accuracy", "totcons": "Total consistency",
                "rankc": "RankC", "agree": "Answer agreement"}


def rankc_pair_fact(slots_a, slots_b):
    s = 0.0
    for j in range(1, N_OPT + 1):
        s += RANKC_W[j - 1] * len(set(slots_a[:j]) & set(slots_b[:j])) / j
    return s


def load_run(path, langs, alignment=None):
    """Per-fact metric vectors for one model run.

    Returns fact_ids (sorted, only facts with all langs) and dict of
    np.array vectors aligned to fact_ids:
      acc      mean correctness over languages
      totcons  1.0 if correct in every language
      rankc    mean over language pairs of RankC contribution
      agree    mean over language pairs of aligned-answer agreement
    """
    with open(path) as f:
        data = json.load(f)
    preds = data["predictions"]
    fact_ids = sorted(fid for fid, per in preds.items()
                      if all(l in per for l in langs))
    pairs = list(itertools.combinations(langs, 2))
    n = len(fact_ids)
    acc = np.zeros(n)
    totcons = np.zeros(n)
    rankc = np.zeros(n)
    agree = np.zeros(n)
    for i, fid in enumerate(fact_ids):
        per = preds[fid]
        correct = np.array([per[l]["correct"] for l in langs], dtype=float)
        acc[i] = correct.mean()
        totcons[i] = float(correct.all())
        # per-lang: pred slot + score-ranked slots in the aligned (en) space
        slot_rank, pred_slot = {}, {}
        for l in langs:
            scores = np.asarray(per[l]["scores"])
            order = np.argsort(-scores, kind="stable")
            amap = alignment[fid][l] if alignment is not None else list(range(N_OPT))
            slot_rank[l] = [amap[int(o)] for o in order]
            pred_slot[l] = amap[int(per[l]["pred_idx"])]
        rc = ag = 0.0
        for la, lb in pairs:
            rc += rankc_pair_fact(slot_rank[la], slot_rank[lb])
            ag += float(pred_slot[la] == pred_slot[lb])
        rankc[i] = rc / len(pairs)
        agree[i] = ag / len(pairs)
    reported = {
        "acc": np.mean([data["per_language_accuracy"][l]["accuracy"] for l in langs]),
        "totcons": data["total_consistency"]["all_langs_correct_fraction"],
        "rankc": data["rankc"]["average"],
        "agree": data["answer_agreement"]["average"],
    }
    return fact_ids, {"acc": acc, "totcons": totcons, "rankc": rankc, "agree": agree}, reported


def fmt_ci(lo, hi):
    return f"[{100*lo:+.2f}, {100*hi:+.2f}]"


def run_benchmark(name, suffix, langs, alignment, results_dir, n_boot, rng, report):
    print(f"\n=== {name} ===")
    runs, fact_ref = {}, None
    for fam, variants in FAMILIES.items():
        for var, base in variants.items():
            path = os.path.join(results_dir, f"{base}_{suffix}.json")
            fids, vecs, reported = load_run(path, langs, alignment)
            if fact_ref is None:
                fact_ref = fids
            assert fids == fact_ref, f"fact-id mismatch for {base}"
            # validation: recomputation must match the reported aggregates
            for m in METRICS:
                diff = abs(vecs[m].mean() - reported[m])
                assert diff < 5e-3, (f"{base} {m}: recomputed {vecs[m].mean():.4f} "
                                     f"vs reported {reported[m]:.4f}")
            runs[(fam, var)] = vecs
            print(f"  validated {fam:5s} {var:9s} "
                  + " ".join(f"{m}={vecs[m].mean():.4f}" for m in METRICS))

    n = len(fact_ref)
    idx = rng.integers(0, n, size=(n_boot, n))

    report.append(f"\n## {name} ({len(langs)} languages, {n} parallel facts)\n")
    report.append("Point estimates with 95% percentile bootstrap CIs "
                  f"({n_boot:,} resamples of facts; facts resampled as units so "
                  "all languages of a fact stay together). All values in %.\n")
    # per-model table
    report.append("| Family | Model | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |")
    report.append("|---|---|" + "---|" * len(METRICS))
    boot_cache = {}
    for (fam, var), vecs in runs.items():
        cells = []
        for m in METRICS:
            boots = vecs[m][idx].mean(axis=1)
            boot_cache[(fam, var, m)] = boots
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cells.append(f"{100*vecs[m].mean():.2f} [{100*lo:.2f}, {100*hi:.2f}]")
        report.append(f"| {fam} | {var} | " + " | ".join(cells) + " |")

    # paired deltas
    report.append("\n**Paired deltas** (same bootstrap fact indices for both models; "
                  "two-sided bootstrap p-value):\n")
    report.append("| Family | Comparison | Metric | Δ (pp) | 95% CI | p |")
    report.append("|---|---|---|---|---|---|")
    for fam in FAMILIES:
        for a, b in COMPARISONS:
            for m in METRICS:
                va, vb = runs[(fam, a)][m], runs[(fam, b)][m]
                point = va.mean() - vb.mean()
                boots = boot_cache[(fam, a, m)] - boot_cache[(fam, b, m)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
                p = min(1.0, max(p, 1.0 / n_boot))
                star = " *" if (lo > 0 or hi < 0) else ""
                report.append(f"| {fam} | {a} − {b} | {METRIC_LABEL[m]} | "
                              f"{100*point:+.2f}{star} | {fmt_ci(lo, hi)} | {p:.4f} |")
    report.append("\n`*` = 95% CI excludes zero.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--alignment_cache",
                    default="evaluate/alignments/polyfact_test_alignment.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(args.alignment_cache) as f:
        alignment = json.load(f)

    report = ["# Bootstrap CIs and paired significance tests",
              "",
              "Recomputed per fact from the raw per-option scores saved in "
              "`results/*_consistency.json`; every recomputed aggregate was "
              "validated against the reported value (tolerance 5e-3) before "
              "bootstrapping."]
    run_benchmark("Global-MMLU-Lite", "gmmlu_lite_consistency", LANGS_GM,
                  None, args.results_dir, args.n_boot, rng, report)
    run_benchmark("PolyFact (test)", "polyfact_consistency", LANGS_PF,
                  alignment, args.results_dir, args.n_boot, rng, report)

    out_path = os.path.join(args.out, "significance_report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
