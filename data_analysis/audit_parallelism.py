#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep audit of PolyFact-Clean's *parallel* premise and question quality.

The dataset's central claim — and the precondition for RankC and any
cross-lingual consistency metric — is that a fact offers the SAME four candidate
entities in every language, differing only in surface form and shuffle order.
Gold answers were verified elsewhere; the distractors never were, because
distractor entity ids are not stored in the release.

They can be recovered now: after the v3 label repairs each entity has exactly one
canonical gold string per language, so a label -> entity table built from gold
answers resolves most distractor strings too.

Checks:
  A. Cross-lingual option alignment. Map each language's 4 options to entity ids
     and test whether every language yields the same 4-entity set for a fact.
  B. Untranslated questions: a non-English question byte-identical to English.
  C. Answer leakage: the gold string appearing inside the question text.
  D. Contradictory duplicates: one question string with two different gold answers.
  E. Degenerate fields: empty/whitespace questions or options; an option equal to
     the subject.

CPU-only, one language held in memory at a time; safe on a login node.
"""

import argparse
import os
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPT_COLS = ["option_a", "option_b", "option_c", "option_d"]


def fetch(repo, config, split, local_dir=None):
	if local_dir:
		return pq.read_table(os.path.join(local_dir, "data", config, f"{split}.parquet"))
	return pq.read_table(hf_hub_download(
		repo, f"data/{config}/{split}.parquet", repo_type="dataset"))


def read_lang(repo, lang, local_dir):
	rows = []
	for s in SPLITS:
		t = fetch(repo, lang, s, local_dir)
		rows.extend(t.to_pylist())
	return rows


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--local_dir", default=None, help="read a local build instead of the Hub")
	ap.add_argument("--out", default="results/polyfact_parallelism_audit.md")
	args = ap.parse_args()

	meta = {}
	for s in SPLITS:
		t = fetch(args.repo, "parallel", s, args.local_dir)
		for r in t.select(["fact_id", "object_id", "subject", "relation"]).to_pylist():
			meta[r["fact_id"]] = r
	print(f"{len(meta):,} facts")

	lines = []

	def out(s=""):
		print(s)
		lines.append(s)

	out(f"# Parallelism & question-quality audit — `{args.repo}`\n")

	en_questions = {}
	en_entity_sets = {}
	align_mismatch = Counter()
	align_unresolved = Counter()
	untranslated_q = Counter()
	leak = Counter()
	contradictions = Counter()
	degenerate = Counter()
	subject_as_option = Counter()
	examples = defaultdict(list)

	for lang in LANGS:
		rows = read_lang(args.repo, lang, args.local_dir)

		# canonical label -> entity, from gold answers
		label2ent = {}
		for r in rows:
			label2ent[r["answer_text"]] = meta[r["fact_id"]]["object_id"]

		q_to_golds = defaultdict(set)
		for r in rows:
			fid = r["fact_id"]
			m = meta[fid]
			opts = [r[c] for c in OPT_COLS]
			q = r["question"]

			# ---- A: map options to entities
			ents, unres = set(), 0
			for o in opts:
				e = label2ent.get(o)
				if e is None:
					unres += 1
				else:
					ents.add(e)
			if unres:
				align_unresolved[lang] += 1
			if lang == "en":
				en_entity_sets[fid] = (frozenset(ents), unres)
			else:
				ref_set, ref_unres = en_entity_sets.get(fid, (None, 0))
				if ref_set is not None and unres == 0 and ref_unres == 0:
					if frozenset(ents) != ref_set:
						align_mismatch[lang] += 1
						if len(examples[f"align_{lang}"]) < 3:
							examples[f"align_{lang}"].append(
								(m["subject"], m["relation"], sorted(ents), sorted(ref_set)))

			# ---- B: untranslated question
			if lang == "en":
				en_questions[fid] = q
			elif en_questions.get(fid) == q:
				untranslated_q[lang] += 1
				if len(examples[f"untrans_{lang}"]) < 3:
					examples[f"untrans_{lang}"].append(q)

			# ---- C: gold string inside the question
			gold = r["answer_text"]
			if gold and len(gold) > 2 and gold.casefold() in q.casefold():
				leak[lang] += 1
				if len(examples[f"leak_{lang}"]) < 3:
					examples[f"leak_{lang}"].append((q, gold))

			# ---- D: same question, different gold
			q_to_golds[q].add(gold)

			# ---- E: degenerate fields
			if not q.strip() or any(not o.strip() for o in opts):
				degenerate[lang] += 1
			subj = m["subject"]
			if subj and subj in opts:
				subject_as_option[lang] += 1
				if len(examples[f"subjopt_{lang}"]) < 3:
					examples[f"subjopt_{lang}"].append((subj, m["relation"], q[:70]))

		contradictions[lang] = sum(1 for g in q_to_golds.values() if len(g) > 1)
		if lang == "en":
			for q, golds in q_to_golds.items():
				if len(golds) > 1 and len(examples["contra_en"]) < 4:
					examples["contra_en"].append((q, sorted(golds)))
		print(f"  scanned {lang}")
		del rows

	n = len(meta)
	out("## A. Cross-lingual option alignment (vs English)\n")
	out("Each language's 4 options are mapped to Wikidata entity ids via the canonical "
	    "gold-label table; a mismatch means a language offers a different candidate set "
	    "for the same fact.\n")
	out("| lang | facts with a different entity set | facts with an unmappable option |")
	out("|---|---|---|")
	for lang in LANGS:
		if lang == "en":
			out(f"| en | (reference) | {align_unresolved['en']:,} ({100*align_unresolved['en']/n:.1f}%) |")
			continue
		out(f"| {lang} | **{align_mismatch[lang]:,}** | {align_unresolved[lang]:,} "
		    f"({100*align_unresolved[lang]/n:.1f}%) |")
	out("")
	for lang in LANGS:
		for subj, rel, got, ref in examples.get(f"align_{lang}", []):
			out(f"- [{lang}] {subj} — *{rel}*: {got} vs en {ref}")
	out("")

	out("## B. Untranslated questions (byte-identical to English)\n")
	out("| lang | count | share |")
	out("|---|---|---|")
	for lang in LANGS[1:]:
		out(f"| {lang} | {untranslated_q[lang]:,} | {100*untranslated_q[lang]/n:.2f}% |")
	out("")
	for lang in LANGS[1:]:
		for q in examples.get(f"untrans_{lang}", [])[:2]:
			out(f"- [{lang}] {q[:110]}")
	out("")

	out("## C. Gold answer leaked into the question text\n")
	out("| lang | count | share |")
	out("|---|---|---|")
	for lang in LANGS:
		out(f"| {lang} | {leak[lang]:,} | {100*leak[lang]/n:.2f}% |")
	out("")
	for lang in ["en", "de", "fr"]:
		for q, g in examples.get(f"leak_{lang}", [])[:3]:
			out(f"- [{lang}] gold `{g}` in: {q[:100]}")
	out("")

	out("## D. Identical question with different gold answers\n")
	out("| lang | question strings with >1 gold |")
	out("|---|---|")
	for lang in LANGS:
		out(f"| {lang} | {contradictions[lang]:,} |")
	out("")
	for q, golds in examples.get("contra_en", []):
		out(f"- {q[:95]} → {golds}")
	out("")

	out("## E. Degenerate fields\n")
	out(f"- empty question or option: **{sum(degenerate.values()):,}** across all languages")
	out(f"- an option equal to the subject entity: "
	    f"**{sum(subject_as_option.values()):,}**")
	for lang in LANGS:
		for subj, rel, q in examples.get(f"subjopt_{lang}", [])[:2]:
			out(f"  - [{lang}] subject `{subj}` is an option — *{rel}* — {q}")
	out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
