#!/usr/bin/env python
"""Recompute BMLAMA-53 metrics on the DECONTAMINATED subset, post hoc.

No GPU and no re-running of any model: the result JSONs store each option's
score per language per fact, so accuracy, total consistency and RankC can all be
recomputed on any subset of facts.

Contamination is defined as in data_analysis/bmlama_contamination.py -- an exact
(subject, object) match against PolyFact-Clean train, both entities >= 4 chars.
That is 69 of 3,070 items (2.25%). The droplist is keyed by BMLAMA's RAW row
index, which is exactly what the evaluator uses for its fact ids
("bmlama53_<i>"), so the join is exact rather than positional-and-hopeful.

A looser "subject seen in training" criterion is also available (848 items,
27.6%) via --mode subject; the strict pair overlap is the default because
sharing a subject is not itself leakage of the answer.

Usage:
  python data_analysis/bmlama_decontaminated.py                    # all results
  python data_analysis/bmlama_decontaminated.py --mode subject
  python data_analysis/bmlama_decontaminated.py --models qwen-base qwen-sw-clip5
"""
import argparse, glob, io, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluate"))
from evaluate_crosslingual_consistency import rankc_pair  # noqa: E402

DROPLIST = "evaluate/alignments/bmlama53_contamination_droplist.json"


def ranks_from_scores(scores):
    """Option indices ordered best-first -- the input rankc_pair expects."""
    return [i for i, _ in sorted(enumerate(scores), key=lambda t: -t[1])]


def recompute(path, drop):
    d = json.load(io.open(path, encoding="utf-8"))
    P = d["predictions"]
    items = list(P.items()) if isinstance(P, dict) else list(enumerate(P))
    langs = d["config"]["langs"]

    kept = [(k, v) for k, v in items if k not in drop]
    per_lang_hits = {l: [0, 0] for l in langs}
    all_correct = 0
    rankc_sum, rankc_n = 0.0, 0

    for _, rec in kept:
        if not isinstance(rec, dict):
            continue
        ok_all = True
        for l in langs:
            v = rec.get(l)
            if not isinstance(v, dict):
                ok_all = False
                continue
            per_lang_hits[l][1] += 1
            if v.get("correct"):
                per_lang_hits[l][0] += 1
            else:
                ok_all = False
        if ok_all:
            all_correct += 1
        # RankC over every language pair, from the stored option scores
        rs = {l: ranks_from_scores(rec[l]["scores"])
              for l in langs if isinstance(rec.get(l), dict) and "scores" in rec[l]}
        ls = sorted(rs)
        for a in range(len(ls)):
            for b in range(a + 1, len(ls)):
                rankc_sum += rankc_pair(rs[ls[a]], rs[ls[b]])
                rankc_n += 1

    acc = {l: 100 * h / t for l, (h, t) in per_lang_hits.items() if t}
    return {
        "n_kept": len(kept),
        "n_dropped": len(items) - len(kept),
        "acc_mean": sum(acc.values()) / len(acc) if acc else float("nan"),
        "total_consistency": 100 * all_correct / len(kept) if kept else float("nan"),
        "rankc": 100 * rankc_sum / rankc_n if rankc_n else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pair", "subject"], default="pair",
                    help="pair = exact (subject,object) overlap (69 items, default); "
                         "subject = any shared subject entity (848 items, looser)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()

    dl = json.load(io.open(DROPLIST, encoding="utf-8"))
    drop = set(dl["exact_pair_drop"] if a.mode == "pair" else dl["subject_overlap_drop"])
    print(f"mode={a.mode}: dropping {len(drop)} of {dl['n_raw_rows']} BMLAMA items "
          f"({100*len(drop)/dl['n_raw_rows']:.2f}%)\n")

    files = sorted(glob.glob(os.path.join(a.results_dir, "*_bmlama53_consistency.json")))
    if a.models:
        files = [f for f in files if any(m in os.path.basename(f) for m in a.models)]

    print(f"{'model':<26} {'kept':>6} {'Acc':>7} {'TotCons':>8} {'RankC':>7}   "
          f"{'dAcc':>6} {'dTotC':>6} {'dRankC':>7}")
    base = {}
    rows = []
    for f in files:
        tag = os.path.basename(f).replace("_bmlama53_consistency.json", "")
        try:
            r = recompute(f, drop)
        except Exception as e:
            print(f"{tag:<26}  ERROR {e!r}")
            continue
        rows.append((tag, r))
        if tag.endswith("-base"):
            base[tag.split("-")[0]] = r
    for tag, r in rows:
        fam = tag.split("-")[0]
        b = base.get(fam)
        d = (f"{r['acc_mean']-b['acc_mean']:>+6.2f} {r['total_consistency']-b['total_consistency']:>+6.2f} "
             f"{r['rankc']-b['rankc']:>+7.2f}") if b and not tag.endswith("-base") else " " * 21
        print(f"{tag:<26} {r['n_kept']:>6} {r['acc_mean']:>7.2f} {r['total_consistency']:>8.2f} "
              f"{r['rankc']:>7.2f}   {d}")


if __name__ == "__main__":
    main()
