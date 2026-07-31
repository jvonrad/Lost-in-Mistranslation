#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build PolyFact-Clean: PolyFact minus the three highest-ambiguity relations.

Per the paper (Appendix D.2), the recommended PolyFact-Clean filter excludes
the three relations the LLM-judge verification flagged as most ambiguous:

    country of origin (P495), genre (P136), place of birth (P19)

This removes ~17% of items. The filter is defined on the relation's Wikidata
property id, so it applies identically to the `parallel` config and to each of
the 12 per-language configs (fact_id encodes the property as
subject_id|property_id|object_id, but property_id is matched explicitly from
the parallel config rather than parsed out of the string).

CPU-only and single-threaded by design — safe to run on a login node.

Usage:
  python build-poly-fact/build_polyfact_clean.py \
    --raw_dir  $SCRATCH/polyfact_clean/raw \
    --out_dir  $SCRATCH/polyfact_clean/clean

Then upload with build-poly-fact/upload_polyfact_clean.py.
"""

import argparse
import json
import os
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
CONFIGS = ["parallel"] + LANGS

# Appendix D.2: the three highest-ambiguity relations, excluded from PolyFact-Clean.
EXCLUDED_PROPERTY_IDS = {
	"P495": "country of origin",
	"P136": "genre",
	"P19": "place of birth",
}

# v2 addition. Not one of the top-3 by judge ambiguity (Table 7 puts it at 11%),
# excluded on separate, explicitly stated grounds: `employer` is semantically
# many-to-one (a person may hold several jobs, so the enforced single gold is
# arbitrary), and it contains every known split-integrity defect in the release
# (all 73 duplicate train rows and all 3 train/test leaked facts).
EMPLOYER_PROPERTY_ID = {"P108": "employer"}


def excluded_fact_ids(raw_dir, excluded_props):
	"""Collect fact_ids of the excluded relations from the parallel config."""
	excluded = set()
	kept_per_split = {}
	relation_counts = Counter()
	for split in SPLITS:
		path = os.path.join(raw_dir, "data", "parallel", f"{split}.parquet")
		table = pq.read_table(path, columns=["fact_id", "property_id", "relation"])
		fact_ids = table.column("fact_id").to_pylist()
		pids = table.column("property_id").to_pylist()
		rels = table.column("relation").to_pylist()
		n_excl = 0
		for fid, pid, rel in zip(fact_ids, pids, rels):
			if pid in excluded_props:
				excluded.add(fid)
				relation_counts[f"{rel} ({pid})"] += 1
				n_excl += 1
		kept_per_split[split] = (len(fact_ids), len(fact_ids) - n_excl)
	return excluded, kept_per_split, relation_counts


def integrity_drops(raw_dir, excluded):
	"""fact_ids to drop from test (train/test leakage). Duplicates are handled per row."""
	ids = {}
	for split in SPLITS:
		t = pq.read_table(os.path.join(raw_dir, "data", "parallel", f"{split}.parquet"),
		                  columns=["fact_id"])
		ids[split] = [f for f in t.column("fact_id").to_pylist() if f not in excluded]
	leaked = set(ids["train"]) & set(ids["test"])
	return leaked


def filter_config(raw_dir, out_dir, config, excluded, leaked_from_test, dedup, stats):
	src_dir = os.path.join(raw_dir, "data", config)
	dst_dir = os.path.join(out_dir, "data", config)
	os.makedirs(dst_dir, exist_ok=True)
	for split in SPLITS:
		table = pq.read_table(os.path.join(src_dir, f"{split}.parquet"))
		fids = table.column("fact_id").to_pylist()
		keep_flags = []
		seen = set()
		for fid in fids:
			ok = fid not in excluded
			if ok and split == "test" and fid in leaked_from_test:
				ok = False                      # drop the leaked copy from test
			if ok and dedup:
				if fid in seen:
					ok = False                  # keep only the first occurrence
				else:
					seen.add(fid)
			keep_flags.append(ok)
		filtered = table.filter(pa.array(keep_flags))
		# preserve the HF feature metadata so the config's schema is unchanged
		filtered = filtered.replace_schema_metadata(table.schema.metadata)
		pq.write_table(filtered, os.path.join(dst_dir, f"{split}.parquet"), compression="snappy")
		stats[f"{config}/{split}"] = (table.num_rows, filtered.num_rows)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--raw_dir", required=True, help="local copy of the jvonrad/PolyFact files")
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--exclude_employer", action="store_true",
	                help="v2: also drop the `employer` (P108) relation")
	ap.add_argument("--fix_integrity", action="store_true",
	                help="v2: drop train/test-leaked facts from test and de-duplicate rows")
	args = ap.parse_args()

	excluded_props = dict(EXCLUDED_PROPERTY_IDS)
	if args.exclude_employer:
		excluded_props.update(EMPLOYER_PROPERTY_ID)

	excluded, kept_per_split, relation_counts = excluded_fact_ids(args.raw_dir, excluded_props)
	total = sum(n for n, _ in kept_per_split.values())
	print(f"Excluded relations: {', '.join(f'{v} ({k})' for k, v in excluded_props.items())}")
	for name, n in relation_counts.most_common():
		print(f"  {name:<32} {n:>7,}")
	print(f"\nexcluded facts: {len(excluded):,} / {total:,} ({100 * len(excluded) / total:.2f}%)")

	leaked = integrity_drops(args.raw_dir, excluded) if args.fix_integrity else set()
	if args.fix_integrity:
		print(f"train/test leaked facts dropped from test: {len(leaked)}"
		      f"{' ' + str(sorted(leaked)) if leaked else ''}")

	stats = {}
	for config in CONFIGS:
		filter_config(args.raw_dir, args.out_dir, config, excluded, leaked,
		              args.fix_integrity, stats)
		kept = sum(k for c, (_, k) in stats.items() if c.startswith(f"{config}/"))
		print(f"  {config:<10} -> {kept:,} rows")

	# every config must end up with exactly the same fact set
	sizes = {c.split("/")[1]: k for c, (_, k) in stats.items() if c.startswith("parallel/")}
	for config in CONFIGS:
		for split in SPLITS:
			_, kept = stats[f"{config}/{split}"]
			assert kept == sizes[split], f"{config}/{split}: {kept} != {sizes[split]}"
	print("\nall configs agree on per-split sizes:", sizes)

	meta = {
		"source_dataset": "jvonrad/PolyFact",
		"excluded_property_ids": excluded_props,
		"n_excluded_facts": len(excluded),
		"n_source_facts": total,
		"pct_removed": round(100 * len(excluded) / total, 2),
		"integrity_fixed": args.fix_integrity,
		"leaked_facts_dropped_from_test": sorted(leaked),
		"per_split_kept": sizes,
		"per_config_rows": {k: v[1] for k, v in stats.items()},
	}
	with open(os.path.join(args.out_dir, "filter_stats.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"\nwrote {args.out_dir}/filter_stats.json")


if __name__ == "__main__":
	main()
