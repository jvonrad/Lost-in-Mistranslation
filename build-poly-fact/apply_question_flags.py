#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attach the test-split manual question review as a column, without touching data.

`data_analysis/aggregate_review.py` produced per-(fact, language) verdicts for
every test-split question: `ok`, or a defect class (`subject` = asks about a
different entity, `type` = right subject wrong kind of thing, `relation` =
wrong property, `unsure`). This was reviewed by an LLM against the English
question and the subject's Wikidata name, not by a native speaker of each
language — treat it as a first-pass audit, not ground truth.

The reviewer found real errors, but many are legitimate localisation choices
that a heuristic comparison cannot always distinguish from drift (translated
vs. transliterated titles, alternate release titles, style differences). So
this column is additive metadata, not a filter: nothing is dropped based on
it. Users decide their own threshold, same as `n_langs_verified`.

Adds `question_verified` (nullable bool) to every row of every config:
  train / validation        -> null   (not reviewed; the review covered test only)
  test, verdict "ok"        -> true
  test, verdict subject/type/relation -> false
  test, verdict "unsure"    -> null    (reviewer could not tell either way —
                                        indistinguishable from "not reviewed",
                                        which is fine: both mean "no claim")

CPU-only, streamed; safe on a login node.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPT = ["option_a", "option_b", "option_c", "option_d"]
Q_RE = re.compile(r"verdicts_questions_([a-z]+)_(\d+)\.json$")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--review_dir", required=True,
	                help="directory holding the raw verdicts_questions_*.json "
	                     "files (must include 'ok' rows, unlike the aggregated "
	                     "drops file which keeps only defects)")
	args = ap.parse_args()

	# fact_id -> lang -> True (ok) / False (subject/type/relation) / None (unsure)
	verified = defaultdict(dict)
	n_rows = 0
	for fn in sorted(os.listdir(args.review_dir)):
		m = Q_RE.search(fn)
		if not m:
			continue
		lang = m.group(1)
		with open(os.path.join(args.review_dir, fn), encoding="utf-8") as f:
			rows = json.load(f)
		for r in rows:
			n_rows += 1
			v = r.get("verdict", "unsure")
			verified[r["fact_id"]][lang] = None if v == "unsure" else (v == "ok")
	print(f"{n_rows:,} raw question verdicts loaded, "
	      f"{len(verified):,} distinct test facts covered")

	def src(split, config):
		return os.path.join(args.data_dir, "data", config, f"{split}.parquet")

	for l in LANGS + ["parallel"]:
		os.makedirs(os.path.join(args.out_dir, "data", l), exist_ok=True)

	for s in SPLITS:
		base = pq.read_table(src(s, "en")).schema
		lang_schema = pa.schema(list(base) + [pa.field("question_verified", pa.bool_())])
		for l in LANGS:
			t = pq.read_table(src(s, l))
			if s == "test" and l == "en":
				# English was the review's reference question, never itself
				# reviewed against anything — it is true by construction.
				col = [True] * t.num_rows
			elif s == "test":
				col = [verified.get(fid, {}).get(l) for fid in
				       t.column("fact_id").to_pylist()]
			else:
				col = [None] * t.num_rows
			t2 = t.append_column("question_verified", pa.array(col, type=pa.bool_()))
			pq.write_table(t2, os.path.join(args.out_dir, "data", l, f"{s}.parquet"),
			               compression="snappy")

		par = pq.read_table(src(s, "parallel"))
		par_base = par.schema
		# translations.<lang> is the per-language struct we need to extend;
		# translations itself is the outer struct-of-languages and has no
		# option_ids etc. of its own.
		outer_tr = par_base.field("translations").type
		per_lang_tr = outer_tr.field(LANGS[0]).type
		new_tr_struct = pa.struct(list(per_lang_tr) +
		                          [pa.field("question_verified", pa.bool_())])
		rows = par.to_pylist()
		for r in rows:
			for l in LANGS:
				if s == "test" and l == "en":
					r["translations"][l]["question_verified"] = True
				elif s == "test":
					r["translations"][l]["question_verified"] = (
						verified.get(r["fact_id"], {}).get(l))
				else:
					r["translations"][l]["question_verified"] = None
		par_schema = pa.schema([
			f if f.name != "translations" else
			pa.field("translations", pa.struct([pa.field(l, new_tr_struct)
			                                    for l in LANGS]))
			for f in par_base])
		pq.write_table(pa.Table.from_pylist(rows, schema=par_schema),
		               os.path.join(args.out_dir, "data", "parallel", f"{s}.parquet"),
		               compression="snappy")
		print(f"  wrote {s}")

	n_true = n_false = n_null = 0
	t = pq.read_table(os.path.join(args.out_dir, "data", "en", "test.parquet"))
	for v in t.column("question_verified").to_pylist():
		if v is True:
			n_true += 1
		elif v is False:
			n_false += 1
		else:
			n_null += 1
	print(f"\ntest en: verified=True {n_true:,}, verified=False {n_false:,}, "
	      f"unreviewed {n_null:,}")


if __name__ == "__main__":
	main()
