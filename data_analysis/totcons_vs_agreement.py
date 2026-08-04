#!/usr/bin/env python
"""Is Total Consistency the same thing as cross-lingual CONSISTENCY?

TotCons counts facts answered CORRECTLY in all 12 languages. The tempting
reading is that it also measures agreement, on the reasoning that a model is
unlikely to pick the same WRONG answer in all 12 languages. This script tests
that reasoning instead of assuming it, by partitioning every fact into:

  all_correct        every language right                      (= TotCons)
  agree_wrong        every language names the SAME entity, and it is wrong
  disagree           the languages do not all name the same entity

`agree` is exact, not string-matched: options are independently shuffled per
language, so each prediction index is mapped through the dataset's `option_ids`
(the Wikidata QID of every option, per language) before comparison. If
agree_wrong is small relative to all_correct, TotCons is a fair proxy for
agreement; if it is comparable or larger, the two are different measurements and
saying "consistent" when the table reports "correct in all 12" is wrong.

Usage:
  python data_analysis/totcons_vs_agreement.py
  python data_analysis/totcons_vs_agreement.py --models qwen-base qwen-sw-clip5
"""
import argparse
import glob
import io
import json
import os

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]


def load_option_ids(dataset_id, config, split):
    """fact_id -> lang -> [QID per option index], from the dataset itself."""
    from datasets import load_dataset
    ds = load_dataset(dataset_id, config, split=split)
    out = {}
    for row in ds:
        fid = row.get("fact_id")
        tr = row.get("translations") or {}
        per_lang = {}
        for lang in LANGS:
            item = tr.get(lang)
            if not isinstance(item, dict):
                continue
            oid = item.get("option_ids")
            if isinstance(oid, list) and len(oid) == 4 and all(oid) and len(set(oid)) == 4:
                per_lang[lang] = list(oid)
        if fid and len(per_lang) == len(LANGS):
            out[fid] = per_lang
    return out


def classify(path, oids):
    d = json.load(io.open(path, encoding="utf-8"))
    P = d["predictions"]
    n = all_correct = agree_wrong = disagree = 0
    skipped = 0
    for fid, per_lang in P.items():
        ids = oids.get(fid)
        if ids is None or any(l not in per_lang for l in LANGS):
            skipped += 1
            continue
        n += 1
        if all(per_lang[l]["correct"] for l in LANGS):
            all_correct += 1
            continue
        # map each language's chosen option index to the entity it names
        picked = {ids[l][per_lang[l]["pred_idx"]] for l in LANGS}
        if len(picked) == 1:
            agree_wrong += 1
        else:
            disagree += 1
    return {"n": n, "skipped": skipped, "all_correct": all_correct,
            "agree_wrong": agree_wrong, "disagree": disagree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="jvonrad/PolyFact-Clean")
    ap.add_argument("--config", default="parallel")
    ap.add_argument("--split", default="test")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()

    oids = load_option_ids(a.dataset_id, a.config, a.split)
    print(f"loaded option_ids for {len(oids)} facts\n")

    files = sorted(glob.glob(os.path.join(a.results_dir, "*_polyfact_clean_consistency.json")))
    if a.models:
        files = [f for f in files if any(m + "_" in os.path.basename(f) for m in a.models)]

    hdr = (f"{'model':<24} {'n':>5} | {'allCorr':>8} {'agrWrong':>9} {'disagree':>9} | "
           f"{'ANY agree':>10} {'TotCons':>8} {'gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        tag = os.path.basename(f).replace("_polyfact_clean_consistency.json", "")
        try:
            r = classify(f, oids)
        except Exception as e:
            print(f"{tag:<24}  ERROR {e!r}")
            continue
        n = r["n"] or 1
        tc = 100 * r["all_correct"] / n
        aw = 100 * r["agree_wrong"] / n
        dg = 100 * r["disagree"] / n
        print(f"{tag:<24} {r['n']:>5} | {tc:>7.2f}% {aw:>8.2f}% {dg:>8.2f}% | "
              f"{tc + aw:>9.2f}% {tc:>7.2f}% {aw:>+6.2f}")
    print("\nANY agree = all 12 languages name the same entity (right OR wrong).")
    print("gap       = how much TotCons UNDERSTATES full agreement, in pp.")


if __name__ == "__main__":
    main()
