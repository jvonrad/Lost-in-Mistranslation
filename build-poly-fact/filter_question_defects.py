#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3 of the PolyFact-Clean build: remove items whose QUESTION is defective,
independent of whether the labels are correct.

  A. ANSWER LEAKAGE — the gold answer string occurs inside the question text, so
     the item is solvable by copying with no factual knowledge ("Who manufactured
     the Nokia 5000?" -> Nokia). 97% arise because the subject's name contains the
     object's name, which makes `manufacturer` (21% of items) and `developer` (8%)
     by far the worst affected relations.

     This matters more for this paper than the raw rate suggests: a copied answer
     is identical in every language, so leaked items are scored as perfectly
     cross-lingually consistent regardless of what the model knows, inflating
     exactly the consistency metrics the work reports.

  B. AMBIGUOUS QUESTIONS — one question string carrying more than one gold answer,
     because the subject label does not identify the entity ("Who is the creator
     of the self-portrait?" has 25 distinct golds; "On which continent is Danişment
     located?" answers both Asia and Europe). Such items are unanswerable as posed.

A fact is dropped if it is defective in ANY language: PolyFact is parallel and the
consistency metrics read all 12 languages, so a fact that leaks in French but not
Chinese still corrupts the cross-lingual measurement.

Leakage matching is word-boundary aware for space-using scripts to avoid false
positives (e.g. gold "Bono" inside an unrelated longer word); CJK has no spaces so
a plain substring test is used there.

CPU-only, streamed one language at a time; safe on a login node.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
CONFIGS = ["parallel"] + LANGS
NO_WORD_BOUNDARY = {"ja", "zh"}     # scripts written without spaces
MIN_GOLD_LEN = 3


def leaks(gold, question, lang):
	if not gold or len(gold) < MIN_GOLD_LEN:
		return False
	g, q = gold.casefold(), question.casefold()
	if lang in NO_WORD_BOUNDARY:
		return g in q
	# require the gold to appear as a whole token, not inside a longer word
	return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", q) is not None


def find_defects(data_dir):
	leak_ids, ambig_ids = set(), set()
	per_lang = defaultdict(lambda: [0, 0])
	for lang in LANGS:
		rows = []
		for s in SPLITS:
			t = pq.read_table(os.path.join(data_dir, "data", lang, f"{s}.parquet"),
			                  columns=["fact_id", "question", "answer_text"])
			rows.extend(zip(t.column("fact_id").to_pylist(),
			                t.column("question").to_pylist(),
			                t.column("answer_text").to_pylist()))
		q_to_golds = defaultdict(set)
		for fid, q, gold in rows:
			if leaks(gold, q, lang):
				leak_ids.add(fid)
				per_lang[lang][0] += 1
			q_to_golds[q].add(gold)
		ambiguous_q = {q for q, g in q_to_golds.items() if len(g) > 1}
		for fid, q, _ in rows:
			if q in ambiguous_q:
				ambig_ids.add(fid)
				per_lang[lang][1] += 1
		print(f"  {lang}: leak {per_lang[lang][0]:,}, ambiguous {per_lang[lang][1]:,}")
		del rows
	return leak_ids, ambig_ids, {k: tuple(v) for k, v in per_lang.items()}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	args = ap.parse_args()

	leak_ids, ambig_ids, per_lang = find_defects(args.data_dir)
	drop = leak_ids | ambig_ids
	print(f"\nunion over languages: leakage {len(leak_ids):,}, ambiguous {len(ambig_ids):,}, "
	      f"total dropped {len(drop):,}")

	sizes = {}
	for config in CONFIGS:
		dst = os.path.join(args.out_dir, "data", config)
		os.makedirs(dst, exist_ok=True)
		if config == "parallel":
			for s in SPLITS:
				pf = pq.ParquetFile(os.path.join(args.data_dir, "data", config, f"{s}.parquet"))
				schema = pf.schema_arrow
				writer = pq.ParquetWriter(os.path.join(dst, f"{s}.parquet"), schema,
				                          compression="snappy")
				kept_n = 0
				for batch in pf.iter_batches(batch_size=2000):
					rows = [r for r in batch.to_pylist() if r["fact_id"] not in drop]
					if rows:
						writer.write_table(pa.Table.from_pylist(rows, schema=schema))
						kept_n += len(rows)
					del rows, batch
				writer.close()
				sizes[s] = kept_n
		else:
			for s in SPLITS:
				table = pq.read_table(os.path.join(args.data_dir, "data", config, f"{s}.parquet"))
				keep = pa.array([f not in drop for f in table.column("fact_id").to_pylist()])
				filtered = table.filter(keep).replace_schema_metadata(table.schema.metadata)
				pq.write_table(filtered, os.path.join(dst, f"{s}.parquet"), compression="snappy")
				del table, filtered
		print(f"  wrote {config}")

	print(f"\nremaining: {sum(sizes.values()):,} facts {sizes}")
	with open(os.path.join(args.out_dir, "question_defects_removed.json"), "w",
	          encoding="utf-8") as f:
		json.dump({
			"n_answer_leakage": len(leak_ids),
			"n_ambiguous_question": len(ambig_ids),
			"n_dropped_total": len(drop),
			"per_language_hits": {k: {"leak": v[0], "ambiguous": v[1]}
			                      for k, v in per_lang.items()},
			"per_split_kept": sizes,
			"leaked_fact_ids": sorted(leak_ids),
			"ambiguous_fact_ids": sorted(ambig_ids),
		}, f, ensure_ascii=False, indent=1)
	print("wrote question_defects_removed.json")


if __name__ == "__main__":
	main()
