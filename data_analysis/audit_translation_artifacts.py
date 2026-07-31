#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Follow-up to audit_polyfact_clean.py: separate legitimate cross-lingual identity
(proper nouns such as "Michelangelo" or "Berlin", which are the same string in
many languages) from genuine untranslated / English-canonical fallback labels.

This is the analysis Reviewer kSu5 asked for after the rebuttal ("translation
artifacts, ambiguous entity labels, English-canonical answers, label
normalization").

Two diagnostics:
  1. Gold-answer identity with English, broken down BY RELATION. Proper-noun
     relations (place of birth, employer, architect) are expected to be high;
     common-noun relations (official language, currency, genre, continent) are
     expected to be low, so a high rate there indicates an untranslated label.
  2. Facts whose gold is identical to English in EVERY one of the 11 non-English
     languages ("never translated anywhere"), which is the strongest signal of a
     fallback label rather than a genuine cross-lingual proper noun.

Plus: gold answers written in Latin script in the non-Latin-script languages,
with examples.

CPU-only, streamed, single-threaded; safe on a login node.
"""

import argparse
import json
import os
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
NON_EN = [l for l in LANGS if l != "en"]
SPLITS = ["train", "validation", "test"]
NON_LATIN = {"ar", "bn", "ru", "ja", "zh"}


def fetch(repo, config, split):
	return pq.read_table(hf_hub_download(
		repo, f"data/{config}/{split}.parquet", repo_type="dataset"))


def is_latin(text):
	for ch in text:
		if not ch.isalpha():
			continue
		try:
			if unicodedata.name(ch).split()[0] == "LATIN":
				return True
			return False
		except ValueError:
			continue
	return False


def gold_by_fact(repo, lang):
	out = {}
	for s in SPLITS:
		t = fetch(repo, lang, s)
		fids = t.column("fact_id").to_pylist()
		golds = t.column("answer_text").to_pylist()
		out.update(zip(fids, golds))
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--out", default="results/polyfact_translation_artifacts.md")
	args = ap.parse_args()

	# fact_id -> relation, from the parallel config
	rel_of, subj_of = {}, {}
	for s in SPLITS:
		t = fetch(args.repo, "parallel", s)
		for fid, rel, subj in zip(t.column("fact_id").to_pylist(),
		                          t.column("relation").to_pylist(),
		                          t.column("subject").to_pylist()):
			rel_of[fid] = rel
			subj_of[fid] = subj
	print(f"loaded {len(rel_of):,} facts")

	en = gold_by_fact(args.repo, "en")
	identical_count = Counter()          # fact_id -> how many languages match English
	per_rel = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # rel -> lang -> [same, total]
	latin_examples = defaultdict(list)
	latin_counts = {}

	for l in NON_EN:
		golds = gold_by_fact(args.repo, l)
		n_latin = 0
		for fid, g in golds.items():
			e = en.get(fid)
			if e is None:
				continue
			rel = rel_of.get(fid, "?")
			same = int(g == e)
			per_rel[rel][l][0] += same
			per_rel[rel][l][1] += 1
			if same:
				identical_count[fid] += 1
			if l in NON_LATIN and is_latin(g):
				n_latin += 1
				if len(latin_examples[l]) < 6:
					latin_examples[l].append((subj_of.get(fid, "?"), rel, g, e))
		latin_counts[l] = n_latin / max(len(golds), 1)
		print(f"  scanned {l}")

	lines = []

	def out(s=""):
		print(s)
		lines.append(s)

	out(f"# Translation-artifact audit — `{args.repo}`\n")

	out("## 1. Gold answer identical to English, by relation\n")
	out("High values are expected where the object is a proper noun; they are a "
	    "red flag where the object is a common noun (marked ⚠).\n")
	COMMON_NOUN = {"official language", "language of work or name", "currency",
	               "continent", "country", "country of citizenship"}
	header = "| relation | " + " | ".join(NON_EN) + " | mean |"
	out(header)
	out("|---" * (len(NON_EN) + 2) + "|")
	rows = []
	for rel, by_lang in per_rel.items():
		vals = [by_lang[l][0] / max(by_lang[l][1], 1) for l in NON_EN]
		rows.append((sum(vals) / len(vals), rel, vals))
	for mean, rel, vals in sorted(rows, reverse=True):
		mark = " ⚠" if rel in COMMON_NOUN and mean > 0.15 else ""
		out(f"| {rel}{mark} | " + " | ".join(f"{100*v:.0f}%" for v in vals) +
		    f" | **{100*mean:.0f}%** |")
	out("")

	out("## 2. Facts never translated in ANY language\n")
	n_all = sum(1 for c in identical_count.values() if c == len(NON_EN))
	n_facts = len(en)
	out(f"- gold identical to English in **all {len(NON_EN)}** non-English languages: "
	    f"**{n_all:,} / {n_facts:,} ({100*n_all/n_facts:.1f}%)**")
	dist = Counter(identical_count.get(f, 0) for f in en)
	out(f"- distribution of how many languages share the English gold:")
	out("\n| #languages identical to en | facts |")
	out("|---|---|")
	for k in sorted(dist):
		out(f"| {k} | {dist[k]:,} |")
	by_rel_all = Counter(rel_of[f] for f, c in identical_count.items() if c == len(NON_EN))
	if by_rel_all:
		out("\nRelations most affected (identical in all 11 languages):\n")
		out("| relation | facts |")
		out("|---|---|")
		for rel, n in by_rel_all.most_common(8):
			out(f"| {rel} | {n:,} |")
	out("")

	out("## 3. Latin-script gold answers in non-Latin-script languages\n")
	out("| lang | share of gold answers in Latin script |")
	out("|---|---|")
	for l in NON_LATIN:
		out(f"| {l} | {100*latin_counts[l]:.1f}% |")
	out("")
	for l in NON_LATIN:
		if latin_examples[l]:
			out(f"**{l} examples** (subject — relation → gold / English gold):\n")
			for subj, rel, g, e in latin_examples[l]:
				out(f"- {subj} — *{rel}* → `{g}` / en `{e}`")
			out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
