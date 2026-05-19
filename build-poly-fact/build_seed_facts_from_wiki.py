#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build generator-ready seed facts from Wikidata truthy triples, with balanced
sampling across properties.

Input assumptions:
1. A filtered truthy-triples file, either JSONL or .nt:
   JSONL line:
     {"subject_id":"Q23","property_id":"P39","object_id":"Q11696"}

   NT line:
     <http://www.wikidata.org/entity/Q23> <http://www.wikidata.org/prop/direct/P39> <http://www.wikidata.org/entity/Q11696> .

2. A labels JSONL file:
   {"id":"Q23","labels":{"en":"George Washington","de":"George Washington", ...}}
   {"id":"P39","labels":{"en":"position held","de":"Amt", ...}}
   {"id":"Q11696","labels":{"en":"President of the United States", ...}}

3. A types JSONL file:
   {"id":"Q11696","types":["Q6256","Q7275", ...]}

Output:
- seed_facts.jsonl compatible with your MCQ generation pipeline

Example:
python build_seed_facts_from_wiki.py \
  --triples_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/balanced_sample.nt \
  --labels_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/labels.jsonl \
  --types_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/types_map.jsonl \
  --output_jsonl /data/jonathan/Lost-in-Mistranslation/datasets/Wiki-triplets/seed_facts.jsonl \
  --max_facts 200000
"""

import json
import random
import argparse
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

# Good factual properties for MCQ generation
DEFAULT_ALLOWED_PROPERTIES = {
    # geography
    "P36",   # capital
    "P17",   # country
    "P30",   # continent
    "P37",   # official language
    "P38",   # currency
    "P47",   # shares border with

    # biography
    "P27",   # country of citizenship
    "P19",   # place of birth
    "P20",   # place of death
    "P69",   # educated at
    "P108",  # employer

    # creative works / media
    "P50",   # author
    "P57",   # director
    "P170",  # creator
    "P178",  # developer
    "P136",  # genre
    "P495",  # country of origin
    "P407",  # language of work or name
    "P400",  # platform

    # org / relations
    "P176",  # manufacturer

    # government / country
    "P35",   # head of state
    "P6",    # head of government
    "P85",   # anthem
    "P122",  # basic form of government

    # culture / science
    "P84",   # architect
    "P61",   # discoverer or inventor
}

NT_TRIPLE_RE = re.compile(
    r'^<http://www\.wikidata\.org/entity/(Q\d+)> '
    r'<http://www\.wikidata\.org/prop/direct/(P\d+)> '
    r'<http://www\.wikidata\.org/entity/(Q\d+)> \.?$'
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_jsonl", required=True)
    ap.add_argument("--labels_jsonl", required=True)
    ap.add_argument("--types_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)

    ap.add_argument("--max_facts", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--allowed_properties",
        nargs="*",
        default=None,
        help="Optional explicit property IDs. If omitted, uses DEFAULT_ALLOWED_PROPERTIES.",
    )
    ap.add_argument(
        "--min_label_langs",
        type=int,
        default=5,
        help="Minimum number of target languages required for subject/object labels.",
    )
    ap.add_argument(
        "--min_distractor_label_langs",
        type=int,
        default=3,
        help="Minimum number of languages for distractor label eligibility.",
    )
    ap.add_argument(
        "--require_unique_subject_property",
        action="store_true",
        help="Keep only facts where (subject_id, property_id) maps to exactly one object_id.",
    )
    ap.add_argument(
        "--per_property_cap",
        type=int,
        default=None,
        help="Optional hard cap on final written facts per property.",
    )
    ap.add_argument(
        "--shuffle_properties_each_round",
        action="store_true",
        help="Shuffle property order on each round-robin pass.",
    )
    return ap.parse_args()


def load_labels(path: str) -> Dict[str, Dict[str, str]]:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels[row["id"]] = row.get("labels", {})
    return labels


def load_types_map(path: str) -> Dict[str, Set[str]]:
    types_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            types_map[row["id"]] = set(row.get("types", []))
    return types_map


def best_label(label_map: Optional[Dict[str, str]], lang: str = "en") -> Optional[str]:
    if not label_map:
        return None
    if label_map.get(lang):
        return label_map[lang]
    for fallback in ["en", "de", "fr", "es", "ru", "ar", "ja", "zh", "pt", "id", "bn", "sw"]:
        if label_map.get(fallback):
            return label_map[fallback]
    for _, v in label_map.items():
        if v:
            return v
    return None


def has_reasonable_multilingual_coverage(
    label_map: Optional[Dict[str, str]],
    langs: List[str],
    min_count: int = 5,
) -> bool:
    if not label_map:
        return False
    count = sum(1 for l in langs if label_map.get(l))
    return count >= min_count


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def iter_triples(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.endswith(".nt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = NT_TRIPLE_RE.match(line.strip())
                if not m:
                    continue
                s, p, o = m.groups()
                yield {"subject_id": s, "property_id": p, "object_id": o}
    else:
        raise ValueError(f"Unsupported triple file format: {path}")


from collections import defaultdict

def choose_distractor_ids_fast(
    rng,
    gold_object_id,
    gold_label_en,
    gold_types,
    all_candidate_ids,
    type_to_object_ids,
    object_best_en,
    max_len_diff=15,
):
    """
    Fast distractor chooser:
    1. Prefer same-type candidates if available
    2. Fall back to all eligible objects for the property
    3. Filter by approximate label length
    """
    # First try type-matched pool
    typed_pool = set()
    for t in gold_types:
        typed_pool.update(type_to_object_ids.get(t, []))
    typed_pool.discard(gold_object_id)

    typed_pool = [
        oid for oid in typed_pool
        if abs(len(object_best_en[oid]) - len(gold_label_en)) <= max_len_diff
    ]

    if len(typed_pool) >= 3:
        return rng.sample(typed_pool, 3)

    # Fallback to property-wide pool
    fallback_pool = [
        oid for oid in all_candidate_ids
        if oid != gold_object_id
    ]

    if len(fallback_pool) < 3:
        return None

    # Prefer length-similar fallback if possible
    similar_fallback = [
        oid for oid in fallback_pool
        if abs(len(object_best_en[oid]) - len(gold_label_en)) <= max_len_diff
    ]

    pool = similar_fallback if len(similar_fallback) >= 3 else fallback_pool
    if len(pool) < 3:
        return None

    return rng.sample(pool, 3)


def build_candidate_facts(
    triples_by_property,
    labels,
    types,
    rng,
    min_distractor_label_langs,
):
    """
    Faster candidate fact builder.

    Main speedup:
    - preprocess object metadata once per property
    - avoid rescanning all unique objects for every triple
    """
    candidate_facts_by_property = {}

    for p, triples in triples_by_property.items():
        print(f"[candidate_build] property={p} n_triples={len(triples):,}", flush=True)

        unique_objects = sorted({t["object_id"] for t in triples})
        if len(unique_objects) < 4:
            print(f"[candidate_build_done] property={p} facts=0 (not enough unique objects)", flush=True)
            continue

        # ------------------------------------------------------------
        # Precompute object metadata once for this property
        # ------------------------------------------------------------
        object_label_by_id = {}
        object_best_en = {}
        object_types = {}
        eligible_object_ids = []

        for oid in unique_objects:
            oid_labels = labels.get(oid)
            if not oid_labels:
                continue

            oid_best = best_label(oid_labels, "en")
            if not oid_best:
                continue

            if not has_reasonable_multilingual_coverage(
                oid_labels, LANGS, min_count=min_distractor_label_langs
            ):
                continue

            object_label_by_id[oid] = oid_labels
            object_best_en[oid] = oid_best
            object_types[oid] = types.get(oid, set())
            eligible_object_ids.append(oid)

        if len(eligible_object_ids) < 4:
            print(f"[candidate_build_done] property={p} facts=0 (not enough eligible objects)", flush=True)
            continue

        # Map type -> eligible object IDs
        type_to_object_ids = defaultdict(list)
        for oid in eligible_object_ids:
            for tpe in object_types[oid]:
                type_to_object_ids[tpe].append(oid)

        facts_for_property = []

        # ------------------------------------------------------------
        # Build facts
        # ------------------------------------------------------------
        for idx, t in enumerate(triples, start=1):
            if idx % 10000 == 0:
                print(f"  {p}: processed {idx:,}/{len(triples):,}", flush=True)

            gold_object_id = t["object_id"]
            if gold_object_id not in object_best_en:
                continue

            gold_label_en = object_best_en[gold_object_id]
            gold_types = object_types.get(gold_object_id, set())

            distractor_ids = choose_distractor_ids_fast(
                rng=rng,
                gold_object_id=gold_object_id,
                gold_label_en=gold_label_en,
                gold_types=gold_types,
                all_candidate_ids=eligible_object_ids,
                type_to_object_ids=type_to_object_ids,
                object_best_en=object_best_en,
            )
            if distractor_ids is None:
                continue

            distractor_labels = {d: object_label_by_id[d] for d in distractor_ids}
            distractor_texts = [object_best_en[d] for d in distractor_ids]

            subject_text = best_label(t["subject_labels"], "en")
            relation_text = best_label(t["property_labels"], "en")
            object_text = gold_label_en

            if not subject_text or not relation_text or not object_text:
                continue

            # Distinct English surface forms
            normalized = [normalize_text(x) for x in distractor_texts + [object_text]]
            if len(set(normalized)) != 4:
                continue

            fact = {
                "fact_id": f"{t['subject_id']}|{t['property_id']}|{t['object_id']}",
                "subject_id": t["subject_id"],
                "property_id": t["property_id"],
                "object_id": t["object_id"],
                "subject": subject_text,
                "relation": relation_text,
                "object": object_text,
                "distractors": distractor_texts,
                "subject_labels": t["subject_labels"],
                "property_labels": t["property_labels"],
                "object_labels": t["object_labels"],
                "distractor_labels": distractor_labels,
            }

            facts_for_property.append(fact)

        print(f"[candidate_build_done] property={p} facts={len(facts_for_property):,}", flush=True)

        if facts_for_property:
            candidate_facts_by_property[p] = facts_for_property

    return candidate_facts_by_property

def round_robin_balanced_sample(
    candidate_facts_by_property: Dict[str, List[Dict]],
    max_facts: int,
    rng: random.Random,
    per_property_cap: Optional[int] = None,
    shuffle_properties_each_round: bool = False,
) -> Tuple[List[Dict], Counter]:
    """
    Near-equal balancing across properties via round-robin.

    This avoids the huge skew you saw earlier where a few properties dominate.
    """
    pools = {}
    for p, facts in candidate_facts_by_property.items():
        facts_copy = facts[:]
        rng.shuffle(facts_copy)
        pools[p] = facts_copy

    properties = sorted(pools.keys())
    ptr = {p: 0 for p in properties}
    written_per_property = Counter()
    selected = []

    while len(selected) < max_facts:
        made_progress = False
        prop_order = properties[:]
        if shuffle_properties_each_round:
            rng.shuffle(prop_order)

        for p in prop_order:
            if len(selected) >= max_facts:
                break

            if per_property_cap is not None and written_per_property[p] >= per_property_cap:
                continue

            i = ptr[p]
            if i >= len(pools[p]):
                continue

            selected.append(pools[p][i])
            ptr[p] += 1
            written_per_property[p] += 1
            made_progress = True

        if not made_progress:
            break

    return selected, written_per_property


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    allowed_properties = (
        set(args.allowed_properties) if args.allowed_properties else set(DEFAULT_ALLOWED_PROPERTIES)
    )

    print("Loading labels...")
    labels = load_labels(args.labels_jsonl)

    print("Loading types...")
    types = load_types_map(args.types_jsonl)

    print("Reading and filtering triples...")
    triples_by_property = defaultdict(list)
    raw_kept = 0

    for row in iter_triples(args.triples_jsonl):
        p = row["property_id"]
        if p not in allowed_properties:
            continue

        s = row["subject_id"]
        o = row["object_id"]

        if s not in labels or p not in labels or o not in labels:
            continue

        s_labels = labels[s]
        p_labels = labels[p]
        o_labels = labels[o]

        if not has_reasonable_multilingual_coverage(
            s_labels, LANGS, min_count=args.min_label_langs
        ):
            continue
        if not has_reasonable_multilingual_coverage(
            o_labels, LANGS, min_count=args.min_label_langs
        ):
            continue

        triples_by_property[p].append(
            {
                "subject_id": s,
                "property_id": p,
                "object_id": o,
                "subject_labels": s_labels,
                "property_labels": p_labels,
                "object_labels": o_labels,
            }
        )
        raw_kept += 1

    print(f"Triples kept after basic filtering: {raw_kept:,}")
    print("Initial triples per property:")
    for p, triples in sorted(triples_by_property.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"  {p:<6} {len(triples):>10,}  {best_label(labels.get(p, {}), 'en') or 'UNKNOWN'}")

    if args.require_unique_subject_property:
        print("\nApplying unique (subject_id, property_id) constraint...")
        filtered = {}
        for p, triples in triples_by_property.items():
            sp_to_objects = defaultdict(set)
            for t in triples:
                sp_to_objects[(t["subject_id"], t["property_id"])].add(t["object_id"])

            kept = [
                t for t in triples
                if len(sp_to_objects[(t["subject_id"], t["property_id"])]) == 1
            ]
            filtered[p] = kept
            print(f"  {p:<6} kept {len(kept):>10,} / {len(triples):>10,}")
        triples_by_property = filtered

    print("\nBuilding candidate seed facts...")
    candidate_facts_by_property = build_candidate_facts(
        triples_by_property=triples_by_property,
        labels=labels,
        types=types,
        rng=rng,
        min_distractor_label_langs=args.min_distractor_label_langs,
    )

    print("Eligible facts per property:")
    for p, facts in sorted(candidate_facts_by_property.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"  {p:<6} {len(facts):>10,}  {best_label(labels.get(p, {}), 'en') or 'UNKNOWN'}")

    if not candidate_facts_by_property:
        print("No candidate facts survived. Exiting.")
        return

    print("\nSelecting balanced sample...")
    selected_facts, written_per_property = round_robin_balanced_sample(
        candidate_facts_by_property=candidate_facts_by_property,
        max_facts=args.max_facts,
        rng=rng,
        per_property_cap=args.per_property_cap,
        shuffle_properties_each_round=args.shuffle_properties_each_round,
    )

    print(f"Selected facts: {len(selected_facts):,}")
    print("Final selected distribution:")
    for p, count in sorted(written_per_property.items(), key=lambda x: (-x[1], x[0])):
        label = best_label(labels.get(p, {}), "en") or "UNKNOWN"
        pct = 100.0 * count / max(1, len(selected_facts))
        print(f"  {p:<6} {count:>10,}  {pct:>6.2f}%  {label}")

    print(f"\nWriting to {args.output_jsonl} ...")
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for i, fact in enumerate(selected_facts, start=1):
            out.write(json.dumps(fact, ensure_ascii=False) + "\n")
            if i % 10000 == 0:
                print(f"  written={i:,}")

    print("\nDone.")
    print(f"Final written={len(selected_facts):,}")


if __name__ == "__main__":
    main()