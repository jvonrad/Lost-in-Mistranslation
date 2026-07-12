#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contamination analysis for PolyFact (reviewer request, ACL ARR 2026 May #8689).

Reviewers kSu5 / Yeui / aDXs asked us to check whether PolyFact's Wikidata-derived
facts overlap (a) internally between train and the test/val splits and (b) with the
out-of-domain evaluation sets KLAR-CLC and Global-MMLU at the entity/fact level.

This script produces three analyses:

  PART 1  PolyFact split integrity      train vs. validation vs. test
          - exact Wikidata triple overlap (subject_id, property_id, object_id)
          - (subject_id, property_id) overlap  (same query, answer given away)
          - subject / object / property entity overlap
          The generalization claim requires test facts NOT to be in train.

  PART 2  PolyFact  x  KLAR-CLC         (OOD factual-recall benchmark)
          Both are Wikidata-grounded. We match on shared relations (property
          ids) using normalized English subject/object strings:
          - shared relations (PIDs)
          - exact fact triples (relation, subject, object)
          - (relation, subject) pairs   -> same question is trainable
          - subject-entity / object-entity overlap
          Reported from KLAR's side too: what % of KLAR eval items correspond to
          a (relation, subject) the model saw in PolyFact-train.

  PART 3  PolyFact  x  Global-MMLU      (broad-knowledge MCQ benchmark)
          No shared entity ids, so we use surface overlap:
          - co-mention: a Global-MMLU item that mentions BOTH the subject and
            the object of some PolyFact fact is testing that same fact
          - verbatim question n-gram (8-gram) overlap (detects item reuse)

Outputs a JSON report and a Markdown summary under data_analysis/contamination/.

Usage
-----
    python data_analysis/contamination_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict, Counter
from datetime import datetime, timezone

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLYFACT_REPO = "jvonrad/PolyFact"
POLYFACT_CONFIG = "parallel"          # keeps subject_id / property_id / object_id
KLAR_REPO = "mingyang26/KLAR-CLC"
GMMLU_REPO = "CohereLabs/Global-MMLU"

# KLAR relations we know exist (file names); relation_id (PID) is read per file.
KLAR_RELATIONS = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work",
    "headquarters_location", "instrument", "language_of_work_or_name",
    "languages_spoken", "location_of_formation", "manufacturer",
    "native_language", "occupation", "official_language", "owned_by",
    "place_of_birth", "place_of_death", "religion",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contamination")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\W_]+", flags=re.UNICODE)


def norm(s) -> str:
    """NFKC + casefold + strip punctuation/underscores + collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_polyfact():
    """Returns {split: [fact_dict, ...]} for the parallel config."""
    out = {}
    for split in ["train", "validation", "test"]:
        ds = load_dataset(POLYFACT_REPO, POLYFACT_CONFIG, split=split)
        rows = []
        for r in ds:
            rows.append({
                "fact_id": r["fact_id"],
                "subject_id": r["subject_id"],
                "property_id": r["property_id"],
                "object_id": r["object_id"],
                "subject": r["subject"],
                "object": r["object"],
                "relation": r["relation"],
                "n_subj": norm(r["subject"]),
                "n_obj": norm(r["object"]),
            })
        out[split] = rows
        print(f"  PolyFact/{split}: {len(rows)} facts")
    return out


def load_klar():
    """Returns list of KLAR facts (en) with the file-level relation_id (PID)."""
    from huggingface_hub import hf_hub_download

    facts = []
    for rel in KLAR_RELATIONS:
        try:
            path = hf_hub_download(
                KLAR_REPO, f"en/{rel}.json", repo_type="dataset"
            )
        except Exception as e:
            print(f"  [warn] KLAR en/{rel}.json unavailable: {e}")
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        pid = d.get("relation_id")
        for s in d.get("samples", []):
            facts.append({
                "relation_name": rel,
                "property_id": pid,
                "index": s.get("index"),
                "subject": s.get("subject", ""),
                "object": s.get("object", ""),
                "n_subj": norm(s.get("subject", "")),
                "n_obj": norm(s.get("object", "")),
            })
    print(f"  KLAR-CLC (en): {len(facts)} facts across "
          f"{len({f['property_id'] for f in facts})} relations")
    return facts


def load_gmmlu():
    """Returns list of Global-MMLU (en) items with normalized full text."""
    ds = load_dataset(GMMLU_REPO, "en", split="test")
    items = []
    for r in ds:
        parts = [r.get("question", "")] + [
            r.get(f"option_{x}", "") for x in ("a", "b", "c", "d")
        ]
        text = " ".join(p for p in parts if p)
        items.append({
            "sample_id": r["sample_id"],
            "subject_category": r.get("subject_category", ""),
            "text": text,
            "n_text": norm(text),
        })
    print(f"  Global-MMLU (en): {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# PART 1 -- PolyFact split integrity
# ---------------------------------------------------------------------------

def part1_split_integrity(pf):
    def triples(rows):
        return {(r["subject_id"], r["property_id"], r["object_id"]) for r in rows}

    def subj_rel(rows):
        return {(r["subject_id"], r["property_id"]) for r in rows}

    train = pf["train"]
    res = {}
    for split in ["validation", "test"]:
        s = pf[split]
        t_tr, t_s = triples(train), triples(s)
        sr_tr, sr_s = subj_rel(train), subj_rel(s)
        subj_tr = {r["subject_id"] for r in train}
        obj_tr = {r["object_id"] for r in train}
        prop_tr = {r["property_id"] for r in train}
        subj_s = {r["subject_id"] for r in s}
        obj_s = {r["object_id"] for r in s}
        prop_s = {r["property_id"] for r in s}
        n = len(s)
        res[split] = {
            "n_facts": n,
            "exact_triple_in_train": len(t_s & t_tr),
            "exact_triple_in_train_pct": round(100 * len(t_s & t_tr) / n, 3),
            "subject_relation_in_train": len(sr_s & sr_tr),
            "subject_relation_in_train_pct": round(100 * len(sr_s & sr_tr) / n, 3),
            "subject_entity_in_train_pct": round(100 * len(subj_s & subj_tr) / len(subj_s), 2),
            "object_entity_in_train_pct": round(100 * len(obj_s & obj_tr) / len(obj_s), 2),
            "property_in_train_pct": round(100 * len(prop_s & prop_tr) / len(prop_s), 2),
            "n_unique_properties": len(prop_s),
        }
    return res


# ---------------------------------------------------------------------------
# PART 2 -- PolyFact x KLAR-CLC
# ---------------------------------------------------------------------------

def part2_klar(pf, klar):
    pf_props = {r["property_id"] for split in pf.values() for r in split}
    klar_props = {f["property_id"] for f in klar}
    shared_props = pf_props & klar_props

    # Restrict both sides to shared relations for fact-level matching.
    klar_s = [f for f in klar if f["property_id"] in shared_props]

    def index(rows):
        triple = set()          # (pid, n_subj, n_obj)
        subj_rel = set()        # (pid, n_subj)
        subjects = set()        # n_subj
        objects = set()         # n_obj
        for r in rows:
            if r["property_id"] not in shared_props:
                continue
            triple.add((r["property_id"], r["n_subj"], r["n_obj"]))
            subj_rel.add((r["property_id"], r["n_subj"]))
            subjects.add(r["n_subj"])
            objects.add(r["n_obj"])
        return triple, subj_rel, subjects, objects

    klar_triple, klar_sr, klar_subj, klar_obj = index(klar_s)

    res = {
        "polyfact_relations": len(pf_props),
        "klar_relations": len(klar_props),
        "shared_relations": sorted(shared_props),
        "n_shared_relations": len(shared_props),
        "klar_facts_in_shared_relations": len(klar_s),
        "by_polyfact_split": {},
        "klar_side": {},
    }

    for split in ["train", "test"]:
        pf_triple, pf_sr, pf_subj, pf_obj = index(pf[split])
        res["by_polyfact_split"][split] = {
            "polyfact_facts_in_shared_relations": sum(
                1 for r in pf[split] if r["property_id"] in shared_props),
            "exact_fact_triples_shared_with_klar": len(pf_triple & klar_triple),
            "subject_relation_pairs_shared_with_klar": len(pf_sr & klar_sr),
            "subject_entities_shared_with_klar": len(pf_subj & klar_subj),
            "object_entities_shared_with_klar": len(pf_obj & klar_obj),
        }

    # KLAR-side view: how much of the KLAR eval set is "trainable" from PolyFact-train.
    pf_train_triple, pf_train_sr, _, _ = index(pf["train"])
    n_klar = len(klar_s)
    klar_triple_hit = sum(
        1 for f in klar_s
        if (f["property_id"], f["n_subj"], f["n_obj"]) in pf_train_triple)
    klar_sr_hit = sum(
        1 for f in klar_s
        if (f["property_id"], f["n_subj"]) in pf_train_sr)
    res["klar_side"] = {
        "n_klar_eval_facts_shared_relations": n_klar,
        "klar_facts_exact_in_polyfact_train": klar_triple_hit,
        "klar_facts_exact_in_polyfact_train_pct": round(100 * klar_triple_hit / n_klar, 3),
        "klar_facts_subject_relation_in_polyfact_train": klar_sr_hit,
        "klar_facts_subject_relation_in_polyfact_train_pct": round(100 * klar_sr_hit / n_klar, 3),
    }
    return res


def write_klar_contamination_labels(pf, klar, path):
    """
    Emit a label file the KLAR evaluator consumes to split its eval set into
    'contaminated' (fact is memorizable from PolyFact-train) vs 'clean'.

    A KLAR fact is CONTAMINATED iff its relation is one PolyFact also covers AND
    its exact English triple (relation, subject, object) appears in PolyFact-train.
    Keys are (relation_name, index): `index` is stable across all 17 KLAR
    languages, so labeling once in English propagates to every language.
    """
    pf_props = {r["property_id"] for split in pf.values() for r in split}
    klar_props = {f["property_id"] for f in klar}
    shared_props = pf_props & klar_props

    pf_train_triple = {
        (r["property_id"], r["n_subj"], r["n_obj"])
        for r in pf["train"] if r["property_id"] in shared_props
    }

    contaminated = []
    shared_relation_keys = []
    seen = set()
    for f in klar:
        if f["property_id"] not in shared_props or f["index"] is None:
            continue
        key = [f["relation_name"], f["index"]]
        tkey = (f["relation_name"], f["index"])
        if tkey in seen:               # dedupe (same fact appears once per lang file, but en only here)
            continue
        seen.add(tkey)
        shared_relation_keys.append(key)
        if (f["property_id"], f["n_subj"], f["n_obj"]) in pf_train_triple:
            contaminated.append(key)

    out = {
        "definition": ("A KLAR fact is 'contaminated' iff its relation is shared "
                       "with PolyFact and its exact English (relation, subject, "
                       "object) triple is present in PolyFact-train."),
        "shared_relations": sorted(shared_props),
        "n_shared_relation_facts": len(shared_relation_keys),
        "n_contaminated": len(contaminated),
        "n_clean_shared": len(shared_relation_keys) - len(contaminated),
        "contaminated_keys": contaminated,
        "shared_relation_keys": shared_relation_keys,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  wrote KLAR contamination labels -> {path} "
          f"(contaminated={len(contaminated)}, clean_shared={out['n_clean_shared']})")
    return out


# ---------------------------------------------------------------------------
# PART 3 -- PolyFact x Global-MMLU
# ---------------------------------------------------------------------------

MIN_ENTITY_LEN = 4          # normalized-char length floor to avoid trivial hits
STOP_ENTITIES = {           # generic strings that would create noise
    "true", "false", "none", "yes", "no", "water", "air", "gold", "iron",
}


def _entity_word_lengths(entities):
    return sorted({len(e.split()) for e in entities})


def part3_gmmlu(pf, gmmlu):
    # Build entity -> fact indices from ALL PolyFact facts (train+val+test).
    subj_to_facts = defaultdict(set)
    obj_to_facts = defaultdict(set)
    all_entities = set()
    for split, rows in pf.items():
        for r in rows:
            fid = r["fact_id"]
            s, o = r["n_subj"], r["n_obj"]
            if len(s) >= MIN_ENTITY_LEN and s not in STOP_ENTITIES:
                subj_to_facts[s].add(fid)
                all_entities.add(s)
            if len(o) >= MIN_ENTITY_LEN and o not in STOP_ENTITIES:
                obj_to_facts[o].add(fid)
                all_entities.add(o)

    word_lens = _entity_word_lengths(all_entities)
    max_ng = min(max(word_lens) if word_lens else 1, 6)

    def ngrams_present(tokens):
        """All 1..max_ng word n-grams of a token list, as phrases."""
        present = set()
        L = len(tokens)
        for n in range(1, max_ng + 1):
            for i in range(L - n + 1):
                present.add(" ".join(tokens[i:i + n]))
        return present

    co_mention_hits = []          # MMLU items testing a full PolyFact fact
    subj_only = 0                 # items mentioning some PolyFact subject
    by_category = Counter()
    for item in gmmlu:
        toks = item["n_text"].split()
        phrases = ngrams_present(toks)
        subj_hit_facts = set()
        obj_hit_facts = set()
        for ph in phrases:
            if ph in subj_to_facts:
                subj_hit_facts |= subj_to_facts[ph]
            if ph in obj_to_facts:
                obj_hit_facts |= obj_to_facts[ph]
        if subj_hit_facts:
            subj_only += 1
        both = subj_hit_facts & obj_hit_facts     # same fact: subj AND obj present
        if both:
            co_mention_hits.append({
                "sample_id": item["sample_id"],
                "subject_category": item["subject_category"],
                "n_facts": len(both),
                "example_fact_id": sorted(both)[0],
            })
            by_category[item["subject_category"]] += 1

    # Verbatim question reuse: 8-gram overlap between PolyFact EN questions and MMLU.
    pf_en_q = _polyfact_en_questions()
    NG = 8
    pf_ngrams = set()
    for q in pf_en_q:
        toks = norm(q).split()
        for i in range(len(toks) - NG + 1):
            pf_ngrams.add(" ".join(toks[i:i + NG]))
    reuse = 0
    for item in gmmlu:
        toks = item["n_text"].split()
        hit = any(" ".join(toks[i:i + NG]) in pf_ngrams
                  for i in range(len(toks) - NG + 1))
        if hit:
            reuse += 1

    n = len(gmmlu)
    return {
        "n_gmmlu_items": n,
        "polyfact_unique_entities_indexed": len(all_entities),
        "entity_max_ngram": max_ng,
        "items_mentioning_a_polyfact_subject": subj_only,
        "items_mentioning_a_polyfact_subject_pct": round(100 * subj_only / n, 3),
        "items_co_mentioning_full_fact": len(co_mention_hits),
        "items_co_mentioning_full_fact_pct": round(100 * len(co_mention_hits) / n, 3),
        "co_mention_by_category": dict(by_category.most_common()),
        "co_mention_examples": co_mention_hits[:15],
        "verbatim_question_8gram_overlap_items": reuse,
        "verbatim_question_8gram_overlap_pct": round(100 * reuse / n, 3),
    }


_PF_EN_Q_CACHE = None


def _polyfact_en_questions():
    global _PF_EN_Q_CACHE
    if _PF_EN_Q_CACHE is not None:
        return _PF_EN_Q_CACHE
    qs = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset(POLYFACT_REPO, "en", split=split)
        qs.extend(ds["question"])
    _PF_EN_Q_CACHE = qs
    return qs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown(report, path):
    L = []
    L.append("# PolyFact contamination analysis\n")
    L.append(f"_Generated {report['generated_utc']}_\n")
    L.append("PolyFact facts are Wikidata triples `(subject_id, property_id, "
             "object_id)`, which lets us measure overlap at the entity, "
             "relation, and full-triple level rather than by fuzzy string "
             "matching alone.\n")

    p1 = report["part1_split_integrity"]
    L.append("## 1. PolyFact split integrity (train vs. test/validation)\n")
    L.append("| target split | facts | exact triple in train | (subject,relation) in train | subj-entity in train | obj-entity in train |")
    L.append("|---|---|---|---|---|---|")
    for split, d in p1.items():
        L.append(f"| {split} | {d['n_facts']} | "
                 f"{d['exact_triple_in_train']} ({d['exact_triple_in_train_pct']}%) | "
                 f"{d['subject_relation_in_train']} ({d['subject_relation_in_train_pct']}%) | "
                 f"{d['subject_entity_in_train_pct']}% | {d['object_entity_in_train_pct']}% |")
    L.append("\nEntity-level overlap is expected and by design (the splits draw "
             "on the same entity universe); the relevant leakage numbers are the "
             "exact-triple and (subject, relation) columns.\n")

    p2 = report["part2_klar"]
    L.append("## 2. PolyFact x KLAR-CLC (OOD factual-recall benchmark)\n")
    L.append(f"- PolyFact relations: {p2['polyfact_relations']}, "
             f"KLAR relations: {p2['klar_relations']}, "
             f"**shared: {p2['n_shared_relations']}** "
             f"({', '.join(p2['shared_relations'])}).")
    L.append(f"- KLAR eval facts in shared relations: {p2['klar_facts_in_shared_relations']}.\n")
    ks = p2["klar_side"]
    L.append("**KLAR-side (does the model train on KLAR's eval facts?):**")
    L.append(f"- KLAR eval facts whose exact `(relation, subject, object)` is in "
             f"PolyFact-train: **{ks['klar_facts_exact_in_polyfact_train']} / "
             f"{ks['n_klar_eval_facts_shared_relations']} "
             f"({ks['klar_facts_exact_in_polyfact_train_pct']}%)**.")
    L.append(f"- KLAR eval facts whose `(relation, subject)` query is in "
             f"PolyFact-train: **{ks['klar_facts_subject_relation_in_polyfact_train']} "
             f"({ks['klar_facts_subject_relation_in_polyfact_train_pct']}%)**.\n")
    L.append("| PolyFact split | facts (shared rel.) | exact triples w/ KLAR | (subj,rel) w/ KLAR | subj entities | obj entities |")
    L.append("|---|---|---|---|---|---|")
    for split, d in p2["by_polyfact_split"].items():
        L.append(f"| {split} | {d['polyfact_facts_in_shared_relations']} | "
                 f"{d['exact_fact_triples_shared_with_klar']} | "
                 f"{d['subject_relation_pairs_shared_with_klar']} | "
                 f"{d['subject_entities_shared_with_klar']} | "
                 f"{d['object_entities_shared_with_klar']} |")
    L.append("")

    p3 = report["part3_gmmlu"]
    L.append("## 3. PolyFact x Global-MMLU (broad-knowledge MCQ)\n")
    L.append("No shared Wikidata ids, so overlap is measured on surface strings.")
    L.append(f"- Global-MMLU items: {p3['n_gmmlu_items']}; PolyFact entities indexed: {p3['polyfact_unique_entities_indexed']}.")
    L.append(f"- Items co-mentioning a full PolyFact fact (both subject AND object): "
             f"**{p3['items_co_mentioning_full_fact']} "
             f"({p3['items_co_mentioning_full_fact_pct']}%)**.")
    L.append(f"- Items mentioning any PolyFact subject entity: "
             f"{p3['items_mentioning_a_polyfact_subject']} "
             f"({p3['items_mentioning_a_polyfact_subject_pct']}%).")
    L.append(f"- Verbatim question 8-gram overlap: "
             f"**{p3['verbatim_question_8gram_overlap_items']} "
             f"({p3['verbatim_question_8gram_overlap_pct']}%)**.")
    if p3["co_mention_by_category"]:
        L.append(f"- Co-mention by MMLU category: {p3['co_mention_by_category']}.")
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading datasets ...")
    pf = load_polyfact()
    klar = load_klar()
    gmmlu = load_gmmlu()

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "polyfact": f"{POLYFACT_REPO}:{POLYFACT_CONFIG}",
        "klar": KLAR_REPO,
        "global_mmlu": GMMLU_REPO,
    }

    print("PART 1: PolyFact split integrity ...")
    report["part1_split_integrity"] = part1_split_integrity(pf)

    print("PART 2: PolyFact x KLAR-CLC ...")
    report["part2_klar"] = part2_klar(pf, klar)
    labels_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "evaluate", "alignments", "klar_polyfact_contamination.json")
    labels_path = os.path.normpath(labels_path)
    write_klar_contamination_labels(pf, klar, labels_path)

    print("PART 3: PolyFact x Global-MMLU ...")
    report["part3_gmmlu"] = part3_gmmlu(pf, gmmlu)

    json_path = os.path.join(args.out_dir, "contamination_report.json")
    md_path = os.path.join(args.out_dir, "contamination_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_markdown(report, md_path)

    print(f"\nWrote:\n  {json_path}\n  {md_path}\n")
    print("=" * 70)
    with open(md_path, encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
