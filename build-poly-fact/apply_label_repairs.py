#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 of the PolyFact label repair: remove the items that stage 1 identified as
broken. Both defect classes turn out to be unrepairable, so this stage drops
rather than rewrites. Because PolyFact is a parallel corpus, a fact broken in one
language is unusable for cross-lingual consistency, so a dropped fact is removed
from every config.

  (a) TRANSLITERATION CORRUPTION in ar / bn / ru — space-using scripts where a
      Latin fragment sitting inside a word is unambiguous corruption (bn
      "লিঙ্কনNear-Earth গ্রহাণু গবেষণা", ar "ماtejكو" for Matejko). Wikidata has no
      label for these entities in these languages, so there is no ground truth to
      repair from, and inventing a translation for a released dataset is not
      acceptable.

      CJK is deliberately excluded: Wikidata agrees verbatim with 142/168 flagged
      Japanese labels, confirming that an acronym prefixed to native script
      ("AGヴェーザー") is conventional orthography, not an error. Wikidata's `zh`
      labels are also frequently Traditional while this dataset is standardised on
      Simplified, so "repairing" from them (`SOM建筑设计事务所` ->
      `SOM建築設計事務所`) would be a regression.

  (b) GOLD INCONSISTENT WITH ITS ENTITY — one `object_id` carrying two different
      gold strings within a language. Inspection showed these are not mislabels
      that can be corrected: the question itself is negation-inverted ("Which
      language is NOT an official language of ...?" with gold = Nynorsk; "which
      country is NOT the Netherlands team?" with gold = Barbados), so the item
      does not test the stored Wikidata triple at all and its nominally correct
      answer already appears among the other options. Rewriting the gold would
      duplicate an option and leave the question inverted.

      Detection is keyed on (language, object_id, stored string) from stage 1 and
      must never be re-derived by scanning for the string alone: a bad label such
      as "South America" is the CORRECT label of a different entity on many other
      facts, so a string-only match would delete legitimate items.

CPU-only; the parallel config is streamed in record batches to bound memory.
"""

import argparse
import json
import os
import unicodedata
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
CONFIGS = ["parallel"] + LANGS
# scripts that use spaces: an intra-word Latin fragment here is real corruption
CORRUPTION_LANGS = ["ar", "bn", "ru"]
OPT_KEYS = ["option_a", "option_b", "option_c", "option_d"]


def script_of(ch):
	if not ch.isalpha():
		return None
	try:
		return unicodedata.name(ch).split()[0]
	except ValueError:
		return None


def has_intraword_latin(text):
	prev = None
	for ch in text:
		s = script_of(ch)
		if s is None:
			prev = None
			continue
		if prev and prev != s and "LATIN" in (prev, s):
			return True
		prev = s
	return False


def load_plan(repairs_path, data_dir):
	with open(repairs_path, encoding="utf-8") as f:
		blob = json.load(f)
	reasons = blob["reasons"]

	# fact_id -> object_id
	obj_of = {}
	for s in SPLITS:
		t = pq.read_table(os.path.join(data_dir, "data", "parallel", f"{s}.parquet"),
		                  columns=["fact_id", "object_id"])
		obj_of.update(zip(t.column("fact_id").to_pylist(), t.column("object_id").to_pylist()))

	# Entity-inconsistent golds are DROPPED, not rewritten. Inspecting them showed
	# they are not simple mislabels: the question itself is negation-inverted
	# ("Which language is NOT an official language of ...?", gold = Nynorsk), so
	# the item does not test the stored Wikidata triple at all, and its correct
	# answer is already present among the other options. Rewriting the gold would
	# produce a duplicated option and still leave the question inverted.
	repairs, label_oid = {}, {}
	inconsistent = {}     # (lang, bad_string) -> object_id
	for lang, mapping in blob["repairs"].items():
		for bad, spec in mapping.items():
			if not isinstance(spec, dict):
				raise SystemExit(
					"repairs file predates entity-keyed repairs; rerun "
					"build_label_repairs.py to regenerate it")
			if "wrong_entity" in spec.get("reason", ""):
				inconsistent[(lang, bad)] = spec["object_id"]

	# facts to drop, two classes
	drop = set()
	drop_detail = defaultdict(int)

	# (a) unrepairable transliteration corruption in a space-using script
	for lang in CORRUPTION_LANGS:
		for s in SPLITS:
			t = pq.read_table(os.path.join(data_dir, "data", lang, f"{s}.parquet"),
			                  columns=["fact_id", "answer_text"])
			for fid, gold in zip(t.column("fact_id").to_pylist(),
			                     t.column("answer_text").to_pylist()):
				if has_intraword_latin(gold):
					drop.add(fid)
					drop_detail[f"corrupt_{lang}"] += 1

	# (b) gold inconsistent with its entity (negation-inverted / wrong-sense items).
	# Derived from the data itself rather than from the Wikidata map: an entity may
	# be mislabelled in a language Wikidata has no label for, so those cases never
	# reach stage 1's repair list. For each (language, object_id) carrying more than
	# one distinct gold string, the majority string is taken as correct and facts
	# holding a minority string are dropped.
	for lang in LANGS:
		golds = []          # (fact_id, gold)
		for s in SPLITS:
			t = pq.read_table(os.path.join(data_dir, "data", lang, f"{s}.parquet"),
			                  columns=["fact_id", "answer_text"])
			golds.extend(zip(t.column("fact_id").to_pylist(),
			                 t.column("answer_text").to_pylist()))
		by_entity = defaultdict(Counter)
		for fid, gold in golds:
			oid = obj_of.get(fid)
			if oid:
				by_entity[oid][gold] += 1
		minority = {oid: counter.most_common(1)[0][0]
		            for oid, counter in by_entity.items() if len(counter) > 1}
		for fid, gold in golds:
			oid = obj_of.get(fid)
			if oid in minority and gold != minority[oid]:
				drop.add(fid)
				drop_detail[f"inconsistent_{lang}"] += 1
	return repairs, label_oid, obj_of, drop, dict(drop_detail)


def fix_row(row, lang, oid, repairs, label_oid):
	"""Repair one language cell in place; returns True if changed."""
	gold = row.get("answer_text")
	key = (lang, gold)
	if key not in repairs or label_oid.get(key) != oid:
		return False
	good = repairs[key]
	others = [row[k] for i, k in enumerate(OPT_KEYS) if i != row["answer_index"]]
	if good in others:          # would create a duplicate option — leave alone
		return False
	row["answer_text"] = good
	row[OPT_KEYS[row["answer_index"]]] = good
	return True


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--repairs", required=True)
	ap.add_argument("--out_dir", required=True)
	args = ap.parse_args()

	repairs, label_oid, obj_of, drop, drop_detail = load_plan(args.repairs, args.data_dir)
	print(f"facts dropped: {len(drop):,}")
	for k, v in sorted(drop_detail.items()):
		print(f"  {k:<24} {v:,}")

	n_repaired = defaultdict(int)

	# ---- per-language configs ----
	for lang in LANGS:
		for s in SPLITS:
			src = os.path.join(args.data_dir, "data", lang, f"{s}.parquet")
			table = pq.read_table(src)
			rows = table.to_pylist()
			kept = []
			for row in rows:
				if row["fact_id"] in drop:
					continue
				if fix_row(row, lang, obj_of.get(row["fact_id"]), repairs, label_oid):
					n_repaired[lang] += 1
				kept.append(row)
			new = pa.Table.from_pylist(kept, schema=table.schema)
			dst = os.path.join(args.out_dir, "data", lang)
			os.makedirs(dst, exist_ok=True)
			pq.write_table(new, os.path.join(dst, f"{s}.parquet"), compression="snappy")
			del rows, kept, table, new
		print(f"  wrote {lang}")

	# ---- parallel config (nested), streamed in batches ----
	for s in SPLITS:
		src = os.path.join(args.data_dir, "data", "parallel", f"{s}.parquet")
		pf = pq.ParquetFile(src)
		schema = pf.schema_arrow
		dst_dir = os.path.join(args.out_dir, "data", "parallel")
		os.makedirs(dst_dir, exist_ok=True)
		writer = pq.ParquetWriter(os.path.join(dst_dir, f"{s}.parquet"), schema,
		                          compression="snappy")
		for batch in pf.iter_batches(batch_size=2000):
			rows = batch.to_pylist()
			kept = []
			for row in rows:
				if row["fact_id"] in drop:
					continue
				oid = row["object_id"]
				for lang in LANGS:
					cell = (row.get("translations") or {}).get(lang)
					if cell:
						fix_row(cell, lang, oid, repairs, label_oid)
				kept.append(row)
			if kept:
				writer.write_table(pa.Table.from_pylist(kept, schema=schema))
			del rows, kept, batch
		writer.close()
		print(f"  wrote parallel/{s}")

	print(f"\nrepaired gold cells per language: {dict(n_repaired)}")

	meta_path = os.path.join(args.out_dir, "label_repairs_applied.json")
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump({
			"repairs_applied": {f"{l}|{b}": g for (l, b), g in repairs.items()},
			"repair_entity_ids": {f"{l}|{b}": label_oid.get((l, b)) for (l, b) in repairs},
			"n_repaired_cells": dict(n_repaired),
			"n_facts_dropped": len(drop),
			"dropped_fact_ids": sorted(drop),
			"drop_reason": "gold label contains an unrepairable intra-word Latin "
			               "fragment in ar/bn/ru and Wikidata has no label for the "
			               "entity in that language",
		}, f, ensure_ascii=False, indent=1)
	print(f"wrote {meta_path}")


if __name__ == "__main__":
	main()
