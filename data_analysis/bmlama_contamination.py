#!/usr/bin/env python3
"""Contamination analysis: jvonrad/PolyFact-Clean  vs  BMLAMA (Qi et al., EMNLP 2023).

BMLAMA (JRQi/BMLAMA17, JRQi/BMLAMA53) is the benchmark behind RankC, the
cross-lingual consistency metric reviewers asked us to adopt. Before using it as
an *evaluation* set for models post-trained on PolyFact-Clean train, we need to
know how much of it we have already trained on.

Matching is necessarily surface-form based: BMLAMA ships only
`Prompt` / `Ans` / `Candidate Ans` / `Subject` -- **no Wikidata IDs and no
relation field** -- whereas PolyFact-Clean carries (subject_id, property_id,
object_id). So, unlike the KLAR analysis (which could match exact QID triples),
here we match normalized English strings and report:

  1. exact (subject, object) fact-pair overlap  -- the contamination number
  2. subject-entity overlap                     -- weaker, entity familiarity
  3. object/answer-space overlap
  4. relation-template alignment                -- which BMLAMA relations are
     even in PolyFact's relation universe (the analogue of KLAR's
     shared / non-shared relation split)

Reported against PolyFact-Clean *train* (what models saw) and separately
against its test split.

Usage:  python data_analysis/bmlama_contamination.py [--out results/contamination_bmlama.json]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter

from datasets import load_dataset

POLYFACT = "jvonrad/PolyFact-Clean"
BMLAMA_SETS = ["JRQi/BMLAMA17", "JRQi/BMLAMA53"]

# Same floor as data_analysis/contamination_analysis.py: ignore very short
# normalized strings so that stopword-ish entities don't create trivial hits.
MIN_ENTITY_LEN = 4

# BMLAMA has no relation column, so we map its prompt templates onto PolyFact
# property ids by hand. Only templates whose relation PolyFact actually covers
# are listed; everything else counts as "non-shared" relation.
#
# This matters: matching on (subject, object) ALONE overcounts. e.g. BMLAMA's
# "[X] is the capital of <mask>" (P1376) pairs Warsaw->Poland, and PolyFact has
# the same pair under P17 "country" -- same entities, different relation, so it
# is NOT the same fact. Strict matching requires the relation to line up too.
TEMPLATE_TO_PID = {
    "the official language of [x] is": "P37",
    "[x] is located in": "P17",
    "[x] died in": "P20",
    "[x] is developed by": "P178",
    "[x] is produced by": "P176",
    "[x] was written in": "P407",
    "the original language of [x] is": "P407",
}


def template_key(t: str) -> str:
    """Normalize a template for lookup (BMLAMA53 appends ' <mask>.')."""
    k = t.casefold().replace("<mask>", " ").strip()
    k = re.sub(r"\s+", " ", k).strip(" .")
    return k


def norm(s) -> str:
    """NFKC + casefold + strip surrounding punctuation/space + collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = s.strip()
    # BMLAMA answers look like " Madrid." -- drop the trailing sentence period
    s = s.strip(" \t\n\r.,;:!?。．")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def template_of(prompt: str, subject: str) -> str:
    """Turn 'Charles II of Spain was born in' -> '[X] was born in' so BMLAMA
    items can be grouped by relation (BMLAMA has no relation column)."""
    p = str(prompt)
    if subject and subject in p:
        p = p.replace(subject, "[X]")
    else:  # fall back to a case-insensitive replace
        p = re.sub(re.escape(str(subject)), "[X]", p, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", p).strip()


def load_polyfact():
    ds = load_dataset(POLYFACT, "parallel")
    out = {}
    for split in ds:
        rows = []
        for r in ds[split]:
            rows.append(
                {
                    "subj": norm(r["subject"]),
                    "obj": norm(r["object"]),
                    "rel": r["relation"],
                    "pid": r["property_id"],
                }
            )
        out[split] = rows
    return out


def load_bmlama(name):
    """English config only -- contamination is a fact-level property, and the
    other languages are translations of the same underlying triples."""
    d = load_dataset(name, "en")
    split = "test" if "test" in d else list(d.keys())[0]
    rows = []
    for r in d[split]:
        subj = r.get("Subject", "")
        rows.append(
            {
                "subj": norm(subj),
                "obj": norm(r.get("Ans", "")),
                "tmpl": template_of(r.get("Prompt", ""), subj),
                "cands": [norm(c) for c in _parse_cands(r.get("Candidate Ans", ""))],
            }
        )
    return rows


def _parse_cands(raw):
    if isinstance(raw, list):
        return raw
    try:
        import ast

        v = ast.literal_eval(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def big(s: str) -> bool:
    return len(s) >= MIN_ENTITY_LEN


def analyse(pf, bm, bm_name):
    pf_train, pf_test = pf["train"], pf.get("test", [])

    def pairs(rows):
        return {(r["subj"], r["obj"]) for r in rows if big(r["subj"]) and big(r["obj"])}

    def subjects(rows):
        return {r["subj"] for r in rows if big(r["subj"])}

    def objects(rows):
        return {r["obj"] for r in rows if big(r["obj"])}

    tr_pairs, te_pairs = pairs(pf_train), pairs(pf_test)
    tr_subj, tr_obj = subjects(pf_train), objects(pf_train)

    # Relation-aware: (subject, object, property_id)
    tr_triples = {
        (r["subj"], r["obj"], r["pid"])
        for r in pf_train
        if big(r["subj"]) and big(r["obj"])
    }

    n = len(bm)
    bm_pairs_hit_train = [r for r in bm if (r["subj"], r["obj"]) in tr_pairs]
    bm_pairs_hit_test = [r for r in bm if (r["subj"], r["obj"]) in te_pairs]

    # Strict: same subject, same object, AND the template's relation maps to the
    # same PolyFact property. Also gives us the shared/non-shared relation split
    # (the analogue of the KLAR clean-shared / non-shared subsets).
    strict_hits, shared_rel, non_shared_rel, clean_shared = [], [], [], []
    for r in bm:
        pid = TEMPLATE_TO_PID.get(template_key(r["tmpl"]))
        if pid is None:
            non_shared_rel.append(r)
            continue
        shared_rel.append(r)
        if (r["subj"], r["obj"], pid) in tr_triples:
            strict_hits.append(r)
        else:
            clean_shared.append(r)
    bm_subj_hit = [r for r in bm if r["subj"] in tr_subj]
    bm_obj_hit = [r for r in bm if r["obj"] in tr_obj]

    # Relation-template view: which templates contribute the contamination
    tmpl_counts = Counter(r["tmpl"] for r in bm)
    tmpl_contam = Counter(r["tmpl"] for r in bm_pairs_hit_train)
    tmpl_table = []
    for t, c in tmpl_counts.most_common():
        hit = tmpl_contam.get(t, 0)
        tmpl_table.append(
            {
                "template": t,
                "n": c,
                "exact_fact_in_polyfact_train": hit,
                "pct": round(100 * hit / c, 2) if c else 0.0,
            }
        )

    res = {
        "benchmark": bm_name,
        "n_bmlama_facts_en": n,
        "polyfact_clean_train_facts": len(pf_train),
        "polyfact_clean_test_facts": len(pf_test),
        "exact_fact_pair_in_polyfact_train": len(bm_pairs_hit_train),
        "exact_fact_pair_in_polyfact_train_pct": round(100 * len(bm_pairs_hit_train) / n, 3),
        "exact_fact_pair_in_polyfact_test": len(bm_pairs_hit_test),
        "exact_fact_pair_in_polyfact_test_pct": round(100 * len(bm_pairs_hit_test) / n, 3),
        # relation-aware (subject, object, property_id) -- the defensible number
        "strict_triple_in_polyfact_train": len(strict_hits),
        "strict_triple_in_polyfact_train_pct": round(100 * len(strict_hits) / n, 3),
        "shared_relation_items": len(shared_rel),
        "shared_relation_pct": round(100 * len(shared_rel) / n, 3),
        "clean_shared_items": len(clean_shared),
        "non_shared_relation_items": len(non_shared_rel),
        "non_shared_relation_pct": round(100 * len(non_shared_rel) / n, 3),
        "contamination_within_shared_relations_pct": (
            round(100 * len(strict_hits) / len(shared_rel), 3) if shared_rel else 0.0
        ),
        "subject_entity_in_polyfact_train": len(bm_subj_hit),
        "subject_entity_in_polyfact_train_pct": round(100 * len(bm_subj_hit) / n, 3),
        "answer_string_in_polyfact_train_objects": len(bm_obj_hit),
        "answer_string_in_polyfact_train_objects_pct": round(100 * len(bm_obj_hit) / n, 3),
        "n_templates": len(tmpl_counts),
        "templates": tmpl_table,
        "examples_contaminated": [
            {"subject": r["subj"], "object": r["obj"], "template": r["tmpl"]}
            for r in bm_pairs_hit_train[:15]
        ],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/contamination_bmlama.json")
    args = ap.parse_args()

    print(f"loading {POLYFACT} (parallel) ...", flush=True)
    pf = load_polyfact()
    print({k: len(v) for k, v in pf.items()}, flush=True)

    report = {"polyfact": POLYFACT, "min_entity_len": MIN_ENTITY_LEN, "benchmarks": []}
    for name in BMLAMA_SETS:
        print(f"loading {name} (en) ...", flush=True)
        bm = load_bmlama(name)
        r = analyse(pf, bm, name)
        report["benchmarks"].append(r)
        print(
            f"\n=== {name} (en, n={r['n_bmlama_facts_en']}) ===\n"
            f"  exact (subject,object) fact in PolyFact-Clean TRAIN : "
            f"{r['exact_fact_pair_in_polyfact_train']} ({r['exact_fact_pair_in_polyfact_train_pct']}%)\n"
            f"  exact (subject,object) fact in PolyFact-Clean TEST  : "
            f"{r['exact_fact_pair_in_polyfact_test']} ({r['exact_fact_pair_in_polyfact_test_pct']}%)\n"
            f"  subject entity seen in train                        : "
            f"{r['subject_entity_in_polyfact_train']} ({r['subject_entity_in_polyfact_train_pct']}%)\n"
            f"  answer string in train object vocabulary            : "
            f"{r['answer_string_in_polyfact_train_objects']} ({r['answer_string_in_polyfact_train_objects_pct']}%)\n"
            f"  STRICT (subject,object,relation) in TRAIN           : "
            f"{r['strict_triple_in_polyfact_train']} ({r['strict_triple_in_polyfact_train_pct']}%)\n"
            f"  shared-relation items / clean-shared / non-shared   : "
            f"{r['shared_relation_items']} / {r['clean_shared_items']} / {r['non_shared_relation_items']}\n"
            f"  contamination WITHIN shared relations               : "
            f"{r['contamination_within_shared_relations_pct']}%\n"
            f"  distinct relation templates                         : {r['n_templates']}",
            flush=True,
        )
        print("  top templates by contamination:", flush=True)
        for t in sorted(r["templates"], key=lambda x: -x["exact_fact_in_polyfact_train"])[:8]:
            print(
                f"    {t['exact_fact_in_polyfact_train']:5d}/{t['n']:5d} ({t['pct']:5.1f}%)  {t['template'][:70]}",
                flush=True,
            )

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
