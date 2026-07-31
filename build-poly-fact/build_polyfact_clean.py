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


def excluded_fact_ids(raw_dir):
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
			if pid in EXCLUDED_PROPERTY_IDS:
				excluded.add(fid)
				relation_counts[f"{rel} ({pid})"] += 1
				n_excl += 1
		kept_per_split[split] = (len(fact_ids), len(fact_ids) - n_excl)
	return excluded, kept_per_split, relation_counts


def filter_config(raw_dir, out_dir, config, excluded, stats):
	src_dir = os.path.join(raw_dir, "data", config)
	dst_dir = os.path.join(out_dir, "data", config)
	os.makedirs(dst_dir, exist_ok=True)
	for split in SPLITS:
		table = pq.read_table(os.path.join(src_dir, f"{split}.parquet"))
		keep = pa.array([fid not in excluded for fid in table.column("fact_id").to_pylist()])
		filtered = table.filter(keep)
		# preserve the HF feature metadata so the config's schema is unchanged
		filtered = filtered.replace_schema_metadata(table.schema.metadata)
		pq.write_table(filtered, os.path.join(dst_dir, f"{split}.parquet"), compression="snappy")
		stats[f"{config}/{split}"] = (table.num_rows, filtered.num_rows)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--raw_dir", required=True, help="local copy of the jvonrad/PolyFact files")
	ap.add_argument("--out_dir", required=True)
	args = ap.parse_args()

	excluded, kept_per_split, relation_counts = excluded_fact_ids(args.raw_dir)
	total = sum(n for n, _ in kept_per_split.values())
	print(f"Excluded relations: {', '.join(f'{v} ({k})' for k, v in EXCLUDED_PROPERTY_IDS.items())}")
	for name, n in relation_counts.most_common():
		print(f"  {name:<32} {n:>7,}")
	print(f"\nexcluded facts: {len(excluded):,} / {total:,} ({100 * len(excluded) / total:.2f}%)")

	stats = {}
	for config in CONFIGS:
		filter_config(args.raw_dir, args.out_dir, config, excluded, stats)
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
		"excluded_property_ids": EXCLUDED_PROPERTY_IDS,
		"n_excluded_facts": len(excluded),
		"n_source_facts": total,
		"pct_removed": round(100 * len(excluded) / total, 2),
		"per_split_kept": sizes,
		"per_config_rows": {k: v[1] for k, v in stats.items()},
	}
	with open(os.path.join(args.out_dir, "filter_stats.json"), "w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)
	print(f"\nwrote {args.out_dir}/filter_stats.json")


if __name__ == "__main__":
	main()
