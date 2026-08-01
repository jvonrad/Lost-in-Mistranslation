#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair the Arabic `country of citizenship` questions that ask for a MUNICIPALITY.

784 Arabic questions on the `country of citizenship` relation render the property
as بلدية ("municipality") instead of بلد ("country"):

    ما هي بلدية كينوارد إلمسلي؟        "What is Kenward Elmslie's MUNICIPALITY?"
    ما هي بلدية مواطنة آلان لانديرز؟    "What is Alan Landers's municipality of citizenship?"

The gold answer is a country in every case, so the item is unanswerable as posed
in Arabic while being correct in the other 11 languages and in the stored triple.

This is a CONSISTENCY REPAIR, not a translation. The same relation is rendered
correctly in 4,824 other Arabic questions, and this rewrites the broken 784 to
match that existing majority template rather than to any wording invented here:

    ما هي بلد مواطنة X؟     4,496 occurrences  <- the target form
    ما هي بلد المواطنة X؟      37 occurrences  <- target for the المواطنة variant

(The majority form pairs the feminine interrogative ما هي with the masculine بلد,
which is loose Modern Standard Arabic. That inconsistency is pre-existing and is
deliberately preserved: matching the corpus's own convention keeps these 784 items
indistinguishable from the other 4,824, which matters more for a benchmark than
grammatical tidiness would. Changing ما هي -> ما هو here would make the repaired
items stand out as a distinguishable subset.)

Three substitutions, verified exhaustive over the affected set (0 anomalies, no
question contains بلدية twice, all 784 begin with ما هي):

    بلدية مواطنة   -> بلد مواطنة      241   noun swap
    بلدية المواطنة -> بلد المواطنة      37   noun swap
    بلدية          -> بلد مواطنة      506   noun swap + restores the dropped
                                            "citizenship" word, matching the
                                            dominant template

Scope is strictly `relation == "country of citizenship"`. بلدية appears in 129
questions on other relations (`country`, `official language`, `architect`,
`continent`) where the subject genuinely IS a municipality; those are untouched.

Only Arabic question text changes: no entity, option, distractor, answer index or
any other language is modified, so no resampling is required.

CPU-only, streamed; safe on a login node.
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
RELATION = "country of citizenship"
BUG = "بلدية"

# order matters: the two-word forms must be tried before the bare noun
SUBS = [("بلدية مواطنة", "بلد مواطنة"),
        ("بلدية المواطنة", "بلد المواطنة"),
        ("بلدية", "بلد مواطنة")]


def repair(question):
	"""Returns the repaired question, or None if this question needs no change."""
	if BUG not in question:
		return None
	for src, dst in SUBS:
		if src in question:
			return question.replace(src, dst, 1)
	return None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--log_out", default="results/arabic_citizenship_repairs.json")
	args = ap.parse_args()

	def src_path(split, config):
		return os.path.join(args.data_dir, "data", config, f"{split}.parquet")

	for c in LANGS + ["parallel"]:
		os.makedirs(os.path.join(args.out_dir, "data", c), exist_ok=True)

	# which fact_ids need repair, decided once from `parallel` so the per-language
	# ar config and the nested parallel copy can never diverge
	fixed = {}
	for s in SPLITS:
		for b in pq.ParquetFile(src_path(s, "parallel")).iter_batches(
				batch_size=2000, columns=["fact_id", "relation", "translations"]):
			for r in b.to_pylist():
				if r["relation"] != RELATION:
					continue
				new = repair(r["translations"]["ar"]["question"])
				if new is not None:
					fixed[r["fact_id"]] = (r["translations"]["ar"]["question"], new)
			del b
	print(f"{len(fixed):,} Arabic questions to repair")

	log = []
	for s in SPLITS:
		# --- per-language configs: only `ar` changes, the rest are copied ---
		for l in LANGS:
			t = pq.read_table(src_path(s, l))
			if l == "ar":
				qs = t.column("question").to_pylist()
				fids = t.column("fact_id").to_pylist()
				n = 0
				for i, fid in enumerate(fids):
					if fid in fixed:
						assert qs[i] == fixed[fid][0], f"ar config diverged at {fid}"
						qs[i] = fixed[fid][1]
						n += 1
				idx = t.schema.get_field_index("question")
				t = t.set_column(idx, "question", pa.array(qs, type=pa.string()))
				print(f"  {s}/ar: {n:,} repaired")
			pq.write_table(t, os.path.join(args.out_dir, "data", l, f"{s}.parquet"),
			               compression="snappy")

		# --- parallel config ---
		pf = pq.ParquetFile(src_path(s, "parallel"))
		schema = pf.schema_arrow
		writer = pq.ParquetWriter(
			os.path.join(args.out_dir, "data", "parallel", f"{s}.parquet"),
			schema, compression="snappy")
		for b in pf.iter_batches(batch_size=2000):
			rows = b.to_pylist()
			for r in rows:
				if r["fact_id"] in fixed:
					old, new = fixed[r["fact_id"]]
					r["translations"]["ar"]["question"] = new
					if len(log) < 800:
						log.append({"fact_id": r["fact_id"], "split": s,
						            "was": old, "now": new})
			writer.write_table(pa.Table.from_pylist(rows, schema=schema))
			del b, rows
		writer.close()
		print(f"  wrote {s}")

	os.makedirs(os.path.dirname(args.log_out), exist_ok=True)
	with open(args.log_out, "w", encoding="utf-8") as f:
		json.dump({"n_repaired": len(fixed),
		           "relation": RELATION,
		           "substitutions": [{"from": a, "to": b} for a, b in SUBS],
		           "repairs": log}, f, ensure_ascii=False, indent=1)
	print(f"\nwrote {args.log_out}")


if __name__ == "__main__":
	main()
