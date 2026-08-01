#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: drop the remaining defective items and RESAMPLE all distractors so that
the option set stops leaking the answer.

Why. An audit of v4 showed a model-free baseline — "pick the option that occurs
most often as a gold answer in the training split" — scoring 69.97% on test
against 25% chance. The cause is distractor sampling: 42.2% of distractors never
serve as a gold anywhere, and in 12.8% of items the gold is the only option that
ever does, so golds and distractors come from different distributions and
"answer-likeness" nearly solves the task. This matters especially here because
SFT and GRPO train on this data and can learn that prior directly while the base
models cannot, and because the prior is entity-based and therefore
language-independent, so exploiting it produces high cross-lingual consistency —
inflating the very metrics the work reports.

Fix. Distractors are drawn from the entities that DO serve as gold answers for the
same relation, sampled in proportion to how often each is a gold — that is, from
the relation's own answer distribution. Golds follow that distribution by
definition, so the four options become exchangeable draws from a single
distribution and no statistic monotone in frequency can pick out the gold.
Drawing from the relation's own pool also gives type matching for free.
Measured on the test split, "pick the most frequent option" falls from 69.97% to
29.15%, and "pick the least frequent option" sits at the same 29.15% — the
symmetry showing the remainder is not usable signal but the structural
consequence of distractors having to differ from the gold.

Sampling is done over ENTITIES, then rendered into each language with that
entity's canonical label, so all 12 languages offer exactly the same four
candidates — the parallel premise now holds by construction rather than by
accident. The chosen entity ids are stored in a new `option_ids` column aligned
with `option_a..option_d`, which lets RankC align options across languages by
exact identity instead of a LaBSE fallback.

Also dropped here:
  * asteroid-designation subjects ("(85254) 1993 TG12"), whose answers are
    extremely concentrated (`Spacewatch` alone is the gold for 517 of them) and
    which are catalogue lookups rather than world knowledge;
  * items carrying an option in a script foreign to the language config (CJK in
    German, Cyrillic in Swahili), which are eliminable without knowing the fact;
  * items whose question never names its subject (English detector).

NOTE: resampling changes the task. Scores on this release are NOT comparable with
scores on the original PolyFact distractors.

CPU-only; safe on a login node.
"""

import argparse
import json
import bisect
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPT = ["option_a", "option_b", "option_c", "option_d"]
LATIN_LANGS = {"en", "de", "id", "pt", "sw", "es", "fr"}
FOREIGN_SCRIPTS = {"CJK", "HIRAGANA", "KATAKANA", "CYRILLIC", "ARABIC", "BENGALI",
                   "HANGUL", "DEVANAGARI", "HEBREW", "GREEK", "THAI", "TAMIL",
                   "TELUGU", "HAN"}
ASTEROID = re.compile(r"^\(\d+\)\s|^\d{4}\s[A-Z]{2}\d*$")


def foreign_script(text):
	for ch in text:
		if not ch.isalpha():
			continue
		try:
			if unicodedata.name(ch).split()[0] in FOREIGN_SCRIPTS:
				return True
		except ValueError:
			pass
	return False


def mentions_subject(question, subject):
	if not subject:
		return True
	q = question.casefold()
	toks = [t for t in re.split(r"[\s,;:'\"()\[\]/–—-]+", subject.casefold()) if len(t) >= 4]
	return any(t in q for t in toks) if toks else subject.casefold() in q


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--drop_relations", default="",
	                help="comma-separated relation names to remove entirely")
	ap.add_argument("--drop_entities", default=None,
	                help="JSON file with a 'union' list of object_ids to remove "
	                     "(defective labels found by the provenance audit)")
	ap.add_argument("--cap_answer_share", type=float, default=None,
	                help="max share of a relation's items any single answer entity may "
	                     "hold, e.g. 0.25. Excess items are dropped at random, keeping "
	                     "split proportions. Needed because P(gold | entity is an option) "
	                     "cannot be brought to chance for an entity that is the correct "
	                     "answer more often than 1/4 of the time — no distractor scheme "
	                     "can make it look wrong often enough.")
	args = ap.parse_args()
	rng = random.Random(args.seed)

	# ---------- load ----------
	meta = {}
	for s in SPLITS:
		for r in pq.read_table(os.path.join(args.data_dir, "data", "parallel", f"{s}.parquet"),
		                       columns=["fact_id", "subject", "subject_id", "relation",
		                                "property_id", "object", "object_id"]).to_pylist():
			r["split"] = s
			meta[r["fact_id"]] = r
	lang_rows = {}
	for l in LANGS:
		d = {}
		for s in SPLITS:
			for r in pq.read_table(os.path.join(args.data_dir, "data", l, f"{s}.parquet")).to_pylist():
				d[r["fact_id"]] = r
		lang_rows[l] = d
	print(f"loaded {len(meta):,} facts")

	# ---------- drops ----------
	drop, why = set(), Counter()
	for fid, m in meta.items():
		if ASTEROID.search(m["subject"] or ""):
			drop.add(fid)
			why["asteroid_subject"] += 1
	for l in LATIN_LANGS:
		for fid, r in lang_rows[l].items():
			if fid in drop:
				continue
			if any(foreign_script(r[c]) for c in OPT):
				drop.add(fid)
				why["foreign_script_option"] += 1
	for fid, r in lang_rows["en"].items():
		if fid in drop:
			continue
		if not mentions_subject(r["question"], meta[fid]["subject"]):
			drop.add(fid)
			why["subject_not_in_question"] += 1
	# whole relations judged untrustworthy (e.g. `currency`, where only 10 of 346
	# answer entities have a Wikidata label in all 12 languages, so the labels
	# cannot be verified and manual inspection found a high rate of substituted
	# currencies — Georgian maneti rendered as lari, Korean yen as won)
	drop_rels = {r.strip() for r in args.drop_relations.split(",") if r.strip()}
	for fid, m in meta.items():
		if m["relation"] in drop_rels and fid not in drop:
			drop.add(fid)
			why[f"relation_{m['relation']}"] += 1

	# entities with a demonstrably defective label in some language
	if args.drop_entities:
		with open(args.drop_entities, encoding="utf-8") as f:
			bad = set(json.load(f)["union"])
		for fid, m in meta.items():
			if m["object_id"] in bad and fid not in drop:
				drop.add(fid)
				why["defective_label_entity"] += 1

	# cap any single answer entity's share of its relation
	if args.cap_answer_share:
		cap = args.cap_answer_share
		by_rel = defaultdict(list)
		for fid in (f for f in meta if f not in drop):
			by_rel[meta[fid]["relation"]].append(fid)
		for rel, fids in by_rel.items():
			counts = Counter(meta[f]["object_id"] for f in fids)
			# largest total T satisfying sum_e min(n_e, cap*T) == T
			lo, hi = 0.0, float(len(fids))
			for _ in range(200):
				T = (lo + hi) / 2
				if sum(min(n, cap * T) for n in counts.values()) >= T:
					lo = T
				else:
					hi = T
			allowed = max(1, int(cap * lo))
			for ent, n in counts.items():
				if n <= allowed:
					continue
				ent_fids = [f for f in fids if meta[f]["object_id"] == ent]
				# drop proportionally within each split so the split ratios survive
				per_split = defaultdict(list)
				for f in ent_fids:
					per_split[meta[f]["split"]].append(f)
				to_drop = len(ent_fids) - allowed
				for s in SPLITS:
					share = len(per_split[s]) / len(ent_fids)
					k = min(len(per_split[s]), int(round(to_drop * share)))
					rng.shuffle(per_split[s])
					for f in per_split[s][:k]:
						drop.add(f)
						why[f"answer_share_cap_{rel}"] += 1
	print(f"dropping {len(drop):,} facts: {dict(why)}")
	keep = [f for f in meta if f not in drop]
	print(f"remaining {len(keep):,}")

	# ---------- entity -> canonical label per language (from surviving golds) ----------
	ent_label = {l: {} for l in LANGS}
	for l in LANGS:
		for fid in keep:
			ent_label[l][meta[fid]["object_id"]] = lang_rows[l][fid]["answer_text"]

	# ---------- per-relation gold pools ----------
	pool = defaultdict(Counter)          # relation -> entity -> gold count
	for fid in keep:
		pool[meta[fid]["relation"]][meta[fid]["object_id"]] += 1
	# Distractors are drawn from the relation's own gold distribution: entity e is
	# sampled with probability proportional to how often e is a gold answer for
	# that relation. Golds are by definition distributed that way too, so the four
	# options are exchangeable draws from a single distribution and no statistic
	# monotone in frequency can identify the gold.
	#
	# Two alternatives were implemented and measured, both worse (test-split
	# "pick the most frequent option", chance 25%, original data 69.97%):
	#   * hard same-log2-frequency bucket, widening when a bucket is too small:
	#     43.2% — the buckets holding the few very frequent entities are tiny, so
	#     the search widened and handed exactly those items rarer distractors.
	#   * exact frequency-tier match, nearest frequencies otherwise: 35.9% —
	#     uniform sampling among neighbours under-weights frequent entities
	#     relative to the distribution golds actually follow.
	# Distribution matching gives 29.3%. The residual above chance is structural:
	# the three distractors must differ from the gold, so a relation's single most
	# frequent answer can never be handed a more frequent distractor, and 49% of
	# items have a gold that occurs 20+ times.
	order = {}      # relation -> (entities sorted by freq asc, freqs, cumulative weights)
	for rel, c in pool.items():
		ents = sorted(c, key=lambda e: c[e])
		freqs = [c[e] for e in ents]
		cumw, tot = [], 0
		for f in freqs:
			tot += f
			cumw.append(tot)
		order[rel] = (ents, freqs, cumw)
		print(f"  {rel:<26} {len(ents):>6} entities, "
		      f"max gold freq {max(freqs):>4}, total {sum(freqs):,}")

	def sample_range(rel, lo, hi, k, exclude):
		"""k distinct entities from ents[lo:hi], weighted by gold frequency."""
		ents, _, cumw = order[rel]
		if hi - lo <= 0 or k <= 0:
			return [] if k <= 0 else None
		base = cumw[lo - 1] if lo > 0 else 0
		span = cumw[hi - 1] - base
		if span <= 0:
			return None
		out, seen = [], set(exclude)
		for _ in range(60 * k):
			if len(out) == k:
				return out
			i = bisect.bisect_left(cumw, base + rng.random() * span, lo, hi)
			e = ents[min(i, hi - 1)]
			if e not in seen:
				seen.add(e)
				out.append(e)
		return out if len(out) == k else None

	def draw(rel, gold, n=3):
		"""n distinct entities != gold, drawn from the relation's gold distribution."""
		return sample_range(rel, 0, len(order[rel][0]), n, {gold})

	# ---------- resample ----------
	def acceptable(fid, ents):
		"""All four entities must render to distinct strings in EVERY language, and
		no distractor may appear in the question text (which would make it
		spuriously attractive to a model that copies from the prompt) or be the
		subject of the question itself."""
		subject = (meta[fid]["subject"] or "").casefold()
		for l in LANGS:
			question = lang_rows[l][fid]["question"].casefold()
			seen = set()
			for i, e in enumerate(ents):
				lab = ent_label[l].get(e)
				if lab is None or lab in seen:
					return False
				seen.add(lab)
				if i == 0:
					continue                      # index 0 is the gold
				low = lab.casefold()
				if len(low) > 2 and (low in question or low == subject):
					return False
		return True

	chosen = {}
	failed = 0
	for fid in keep:
		rel, gold = meta[fid]["relation"], meta[fid]["object_id"]
		picked = None
		for _ in range(40):
			trial = draw(rel, gold)
			if trial is None:
				continue
			if acceptable(fid, [gold] + trial):
				picked = trial
				break
		if picked is None:
			failed += 1
			continue
		chosen[fid] = picked
	print(f"\nresampled {len(chosen):,} facts; {failed:,} could not be satisfied and are dropped")
	keep = [f for f in keep if f in chosen]

	# ---------- write per-language configs ----------
	per_split = Counter()
	os.makedirs(args.out_dir, exist_ok=True)
	lang_out = {l: defaultdict(list) for l in LANGS}
	for fid in keep:
		m = meta[fid]
		ents = [m["object_id"]] + chosen[fid]
		for l in LANGS:
			order = ents[:]
			rng.shuffle(order)                     # independent shuffle per language
			labs = [ent_label[l][e] for e in order]
			gold_lab = ent_label[l][m["object_id"]]
			row = dict(lang_rows[l][fid])
			for i, c in enumerate(OPT):
				row[c] = labs[i]
			row["option_ids"] = order
			row["answer_text"] = gold_lab
			row["answer_index"] = order.index(m["object_id"])
			lang_out[l][m["split"]].append(row)
		per_split[m["split"]] += 1

	base_schema = pq.read_table(
		os.path.join(args.data_dir, "data", LANGS[0], "test.parquet")).schema
	# an input that already carries option_ids (any re-run of this stage) must not
	# get a second field of the same name — parquet accepts it, but the result
	# cannot be read back: "Can't unify schema with duplicate field names".
	fields = list(base_schema)
	if "option_ids" not in base_schema.names:
		fields.append(pa.field("option_ids", pa.list_(pa.string())))
	lang_schema = pa.schema(fields)
	extra = [f for f in base_schema.names
	         if f not in ("fact_id", "language", "subject", "relation", "object",
	                      "question", "answer_text", "answer_index", "option_ids")
	         and f not in OPT]
	for l in LANGS:
		d = os.path.join(args.out_dir, "data", l)
		os.makedirs(d, exist_ok=True)
		for s in SPLITS:
			pq.write_table(pa.Table.from_pylist(lang_out[l][s], schema=lang_schema),
			               os.path.join(d, f"{s}.parquet"), compression="snappy")
		print(f"  wrote {l}")
	del lang_out, lang_rows                        # both are ~12x the release

	# ---------- rebuild the parallel config ----------
	# Streamed in lockstep across the 12 files just written: their row order is
	# identical because every one was built from the same `keep` iteration.
	# Materialising them again to index by fact_id is what OOM-kills this stage.
	tr_fields = [pa.field("question", pa.string()), pa.field("answer_text", pa.string()),
	             pa.field("answer_index", pa.int64())] + \
	            [pa.field(c, pa.string()) for c in OPT] + \
	            [pa.field("option_ids", pa.list_(pa.string()))]
	tr_struct = pa.struct(tr_fields)
	par_schema = pa.schema([
		pa.field("fact_id", pa.string()), pa.field("subject", pa.string()),
		pa.field("subject_id", pa.string()), pa.field("relation", pa.string()),
		pa.field("property_id", pa.string()), pa.field("object", pa.string()),
		pa.field("object_id", pa.string()),
		pa.field("translations", pa.struct([pa.field(l, tr_struct) for l in LANGS])),
	] + [base_schema.field(f) for f in extra])
	d = os.path.join(args.out_dir, "data", "parallel")
	os.makedirs(d, exist_ok=True)
	for s in SPLITS:
		its = {l: pq.ParquetFile(os.path.join(args.out_dir, "data", l, f"{s}.parquet")
		                         ).iter_batches(batch_size=1000) for l in LANGS}
		writer = pq.ParquetWriter(os.path.join(d, f"{s}.parquet"), par_schema,
		                          compression="snappy")
		while True:
			try:
				chunk = {l: next(its[l]).to_pylist() for l in LANGS}
			except StopIteration:
				break
			rows = []
			for i in range(len(chunk["en"])):
				m = meta[chunk["en"][i]["fact_id"]]
				tr = {}
				for l in LANGS:
					r = chunk[l][i]
					tr[l] = {"question": r["question"], "answer_text": r["answer_text"],
					         "answer_index": r["answer_index"],
					         **{c: r[c] for c in OPT},
					         "option_ids": r["option_ids"]}
				rows.append({
					"fact_id": m["fact_id"], "subject": m["subject"],
					"subject_id": m["subject_id"], "relation": m["relation"],
					"property_id": m["property_id"], "object": m["object"],
					"object_id": m["object_id"], "translations": tr,
					**{f: chunk["en"][i][f] for f in extra}})
			writer.write_table(pa.Table.from_pylist(rows, schema=par_schema))
			del chunk, rows
		writer.close()
	print("  wrote parallel")

	with open(os.path.join(args.out_dir, "resample_report.json"), "w", encoding="utf-8") as f:
		json.dump({"dropped": dict(why), "n_dropped": len(drop),
		           "unsatisfiable_resample": failed,
		           "n_facts": len(keep), "per_split": dict(per_split),
		           "seed": args.seed,
		           "method": "distractors drawn from the same relation's gold-entity pool, "
		                     "matched on log2 gold-frequency bucket; entity-level sampling "
		                     "rendered into all 12 languages via canonical labels"}, f,
		          ensure_ascii=False, indent=1)
	print(f"\nfinal: {len(keep):,} facts {dict(per_split)}")


if __name__ == "__main__":
	main()
