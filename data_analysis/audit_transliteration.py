#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transliteration / label-normalization audit, focused on the non-Latin-script
languages (ar, bn, ru, ja, zh).

The premise of PolyFact is that the SAME Wikidata entity is asked about in every
language, so a given object_id should have ONE canonical label per language.
Three failure modes are checked:

  1. Label inconsistency: the same object_id rendered with two or more different
     strings within one language (a normalization failure — it means a model can
     be "right" in one fact and "wrong" in another for the identical entity, and
     it breaks cross-lingual option alignment).
  2. Mixed-script labels: one string containing both Latin and native script
     (e.g. "bullfrog প্রোডাকশনস"), i.e. a half-transliterated label.
  3. Case/spacing-only variants of the same label.

Gold answers only: distractor entity ids are not stored in the release, so
distractors cannot be checked this way.

CPU-only, streamed; safe on a login node.
"""

import argparse
import os
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
NON_LATIN = ["ar", "bn", "ru", "ja", "zh"]
SPLITS = ["train", "validation", "test"]


def fetch(repo, config, split):
	return pq.read_table(hf_hub_download(
		repo, f"data/{config}/{split}.parquet", repo_type="dataset"))


def scripts_in(text):
	out = set()
	for ch in text:
		if not ch.isalpha():
			continue
		try:
			name = unicodedata.name(ch).split()[0]
		except ValueError:
			continue
		out.add(name)
	return out


def norm_loose(s):
	"""Case- and whitespace-insensitive form, for detecting trivial variants."""
	return " ".join(s.casefold().split())


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--out", default="results/polyfact_transliteration_audit.md")
	args = ap.parse_args()

	# fact_id -> (object_id, relation, subject)
	meta = {}
	for s in SPLITS:
		t = fetch(args.repo, "parallel", s)
		for fid, oid, rel, subj in zip(
			t.column("fact_id").to_pylist(), t.column("object_id").to_pylist(),
			t.column("relation").to_pylist(), t.column("subject").to_pylist()):
			meta[fid] = (oid, rel, subj)
	print(f"loaded {len(meta):,} facts")

	lines = []

	def out(s=""):
		print(s)
		lines.append(s)

	out(f"# Transliteration / label-normalization audit — `{args.repo}`\n")

	incons_summary = {}
	mixed_summary = {}
	examples = defaultdict(list)
	mixed_examples = defaultdict(list)

	for lang in LANGS:
		by_entity = defaultdict(Counter)
		mixed = 0
		total = 0
		for s in SPLITS:
			t = fetch(args.repo, lang, s)
			for fid, gold in zip(t.column("fact_id").to_pylist(),
			                     t.column("answer_text").to_pylist()):
				oid = meta.get(fid, (None,))[0]
				if oid is None:
					continue
				by_entity[oid][gold] += 1
				total += 1
				sc = scripts_in(gold)
				if lang in NON_LATIN and "LATIN" in sc and len(sc) > 1:
					mixed += 1
					if len(mixed_examples[lang]) < 6:
						mixed_examples[lang].append((meta[fid][2], meta[fid][1], gold))
		# entities rendered inconsistently
		incons = {o: c for o, c in by_entity.items() if len(c) > 1}
		# of those, how many differ only by case/spacing
		trivial = sum(1 for c in incons.values() if len({norm_loose(x) for x in c}) == 1)
		incons_summary[lang] = (len(incons), len(by_entity), trivial)
		mixed_summary[lang] = (mixed, total)
		for o, c in list(incons.items())[:4]:
			if len(examples[lang]) < 5:
				variants = " | ".join(f"`{k}` ×{v}" for k, v in c.most_common())
				examples[lang].append((o, variants))
		print(f"  scanned {lang}: {len(incons):,} inconsistent entities, {mixed:,} mixed-script")

	out("## 1. Same entity, different label within one language\n")
	out("An `object_id` appearing with more than one distinct gold string. "
	    "`case/space only` counts those whose variants differ only by casing or "
	    "whitespace (trivially normalizable); the rest are genuine variant labels.\n")
	out("| lang | entities with >1 label | distinct entities | share | of which case/space only |")
	out("|---|---|---|---|---|")
	for lang in LANGS:
		n, tot, triv = incons_summary[lang]
		out(f"| {lang} | {n:,} | {tot:,} | {100*n/max(tot,1):.2f}% | {triv:,} |")
	out("")
	for lang in NON_LATIN:
		if examples[lang]:
			out(f"**{lang} examples:**\n")
			for oid, variants in examples[lang]:
				out(f"- `{oid}`: {variants}")
			out("")

	out("## 2. Mixed-script gold labels (non-Latin languages)\n")
	out("A single label containing Latin characters alongside the native script — "
	    "a partially transliterated entity name.\n")
	out("| lang | mixed-script golds | total | share |")
	out("|---|---|---|---|")
	for lang in NON_LATIN:
		m, tot = mixed_summary[lang]
		out(f"| {lang} | {m:,} | {tot:,} | {100*m/max(tot,1):.2f}% |")
	out("")
	for lang in NON_LATIN:
		if mixed_examples[lang]:
			out(f"**{lang} examples** (subject — relation → gold):\n")
			for subj, rel, gold in mixed_examples[lang]:
				out(f"- {subj} — *{rel}* → `{gold}`")
			out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
