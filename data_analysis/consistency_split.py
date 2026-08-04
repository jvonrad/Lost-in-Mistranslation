#!/usr/bin/env python
"""Split full cross-lingual agreement into its correct and incorrect halves.

Total Consistency as reported counts facts answered CORRECTLY in every language.
That silently drops the other way a model can be consistent: naming the SAME
entity everywhere and being wrong. Measured on PolyFact, that population is not
negligible -- for olmo-base it is 1.47% against 1.72% all-correct, i.e. nearly
half of all fully-consistent answers -- so "TotC" understates agreement by a
model-dependent amount, and by MORE for the weaker model. Reporting x / y makes
both halves visible:

    x = % of facts correct in every language        (the old TotC)
    y = % of facts where every language names the
        SAME entity and that entity is wrong

Agreement is exact, never string-matched:
  PolyFact   options are independently shuffled per language, so each
             prediction index is mapped through the dataset's `option_ids`
             (the Wikidata QID of each option, per language) before comparison.
  Global-MMLU-Lite
             options are parallel by index by construction, so `pred_idx` is
             directly comparable across languages.

A fact only counts if every evaluated language has a prediction for it, matching
the denominator `total_consistency.all_langs_correct_fraction` uses.

Usage:
  python data_analysis/consistency_split.py --tags qwen-base qwen-dco-10k
  python data_analysis/consistency_split.py --all
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
from typing import Dict, Optional

LANGS_PF = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

_OPTION_IDS: Optional[Dict[str, Dict[str, list]]] = None


def polyfact_option_ids(dataset_id="jvonrad/PolyFact-Clean", config="parallel",
                        split="test"):
    """fact_id -> lang -> [QID per option index]. Cached across calls."""
    global _OPTION_IDS
    if _OPTION_IDS is not None:
        return _OPTION_IDS
    from datasets import load_dataset
    ds = load_dataset(dataset_id, config, split=split)
    out = {}
    for row in ds:
        fid, tr = row.get("fact_id"), (row.get("translations") or {})
        per = {}
        for lang in LANGS_PF:
            item = tr.get(lang)
            if not isinstance(item, dict):
                continue
            oid = item.get("option_ids")
            if isinstance(oid, list) and len(oid) == 4 and all(oid) and len(set(oid)) == 4:
                per[lang] = list(oid)
        if fid and len(per) == len(LANGS_PF):
            out[fid] = per
    _OPTION_IDS = out
    return out


def split_consistency(path: str, benchmark: str):
    """Return (all_correct_pct, agree_wrong_pct, n_facts) for one result JSON."""
    d = json.load(io.open(path, encoding="utf-8"))
    P = d["predictions"]
    langs = d["config"]["langs"]
    oids = polyfact_option_ids() if benchmark == "polyfact" else None

    n = correct = agree_wrong = 0
    for fid, per_lang in P.items():
        if any(l not in per_lang or not isinstance(per_lang[l], dict) for l in langs):
            continue
        if benchmark == "polyfact":
            ids = oids.get(fid)
            if ids is None:
                continue
            picked = {ids[l][per_lang[l]["pred_idx"]] for l in langs}
        else:
            # options parallel by index -> the index IS the entity key
            picked = {per_lang[l]["pred_idx"] for l in langs}
        n += 1
        if all(per_lang[l]["correct"] for l in langs):
            correct += 1
        elif len(picked) == 1:
            agree_wrong += 1
    if not n:
        return None, None, 0
    return 100 * correct / n, 100 * agree_wrong / n, n


SUFFIX = {"polyfact": "_polyfact_clean_consistency.json",
          "gmmlu_lite": "_gmmlu_lite_consistency.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()

    tags = a.tags or []
    if a.all or not tags:
        tags = sorted({os.path.basename(f).replace(SUFFIX["polyfact"], "")
                       for f in glob.glob(os.path.join(a.results_dir, "*" + SUFFIX["polyfact"]))})

    print(f"{'tag':<26} {'PF corr':>8} {'PF wrong':>9} {'PF n':>6}   "
          f"{'GM corr':>8} {'GM wrong':>9} {'GM n':>6}")
    for t in tags:
        cells = []
        for b in ("polyfact", "gmmlu_lite"):
            p = os.path.join(a.results_dir, t + SUFFIX[b])
            if not os.path.exists(p):
                cells += [None, None, 0]
                continue
            cells += list(split_consistency(p, b))
        fmt = lambda v, w, p=1: (f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'--':>{w}}")
        print(f"{t:<26} {fmt(cells[0],8)} {fmt(cells[1],9)} {cells[2]:>6}   "
              f"{fmt(cells[3],8)} {fmt(cells[4],9)} {cells[5]:>6}")


if __name__ == "__main__":
    main()
