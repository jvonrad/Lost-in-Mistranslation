#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit the released PolyFact / PolyFact-Clean against the claims made in the
ACL ARR 2026 May rebuttal (submission 8689).

Claims checked:
  C1  "0/100,037 (subject, relation) pairs map to >1 object"      (to aDXs, Q1i)
  C2  "we preselect 22 relations"                                  (to aDXs, Q1i)
  C3  dataset size 100K / 95,000 train / 2,500 val / 2,500 test    (paper S3, B6)
  C4  PolyFact-Clean excludes the 3 highest-ambiguity relations    (App. D.2 / Table 7)
      vs. the rebuttal to kSu5 which says "genre, employer"
  C5  split integrity (no train/test leakage, no duplicate facts)
  C6  translation quality: untranslated / English-canonical options
      (kSu5's remaining objection: "translation artifacts, ambiguous entity
       labels, English-canonical answers, label normalization")
  C7  option position balance (options "independently shuffled per language")

CPU-only, single-threaded; safe on a login node.

Usage:
  python data_analysis/audit_polyfact_clean.py --out results/polyfact_clean_audit.md
"""

import argparse
import json
import os
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
# languages whose native script differs from Latin: an option byte-identical to
# the English one is almost certainly an untranslated fallback label
NON_LATIN = {"ar", "bn", "ru", "ja", "zh"}


def fetch(repo, config, split):
	return pq.read_table(hf_hub_download(
		repo, f"data/{config}/{split}.parquet", repo_type="dataset"))


def script_of(text):
	"""Dominant Unicode script name of a string, ignoring digits/punctuation."""
	counts = Counter()
	for ch in text:
		if not ch.isalpha():
			continue
		try:
			name = unicodedata.name(ch).split()[0]
		except ValueError:
			continue
		counts[name] += 1
	return counts.most_common(1)[0][0] if counts else None


def audit(repo, lines):
	def out(s=""):
		print(s)
		lines.append(s)

	out(f"# Audit of `{repo}`\n")

	# ---------- parallel config: facts, relations, uniqueness ----------
	par = {s: fetch(repo, "parallel", s) for s in SPLITS}
	n_by_split = {s: t.num_rows for s, t in par.items()}
	total = sum(n_by_split.values())

	rows = []
	for s in SPLITS:
		t = par[s]
		cols = {c: t.column(c).to_pylist() for c in
		        ["fact_id", "subject", "subject_id", "relation", "property_id",
		         "object", "object_id"]}
		for i in range(t.num_rows):
			r = {c: cols[c][i] for c in cols}
			r["split"] = s
			rows.append(r)

	rels = Counter(r["relation"] for r in rows)

	out("## C2/C3 — size and relation count\n")
	out(f"- facts: **{total:,}** ({', '.join(f'{s} {n:,}' for s, n in n_by_split.items())})")
	out(f"- distinct relations: **{len(rels)}**")
	out("")

	# C1: (subject, relation) -> multiple objects
	out("## C1 — \"0 (subject, relation) pairs map to >1 object\"\n")
	sr = defaultdict(set)
	for r in rows:
		sr[(r["subject_id"], r["property_id"])].add(r["object_id"])
	multi = {k: v for k, v in sr.items() if len(v) > 1}
	out(f"- distinct (subject_id, property_id) pairs: {len(sr):,}")
	out(f"- pairs mapping to >1 distinct object_id: **{len(multi):,}**")
	if multi:
		affected = sum(1 for r in rows if (r["subject_id"], r["property_id"]) in multi)
		out(f"- facts involved: **{affected:,}** ({100 * affected / total:.2f}% of the dataset)")
		by_rel = Counter()
		for r in rows:
			if (r["subject_id"], r["property_id"]) in multi:
				by_rel[r["relation"]] += 1
		out("\n  | relation | facts in 1-to-many pairs |")
		out("  |---|---|")
		for rel, n in by_rel.most_common():
			out(f"  | {rel} | {n:,} |")
		out("\n  examples:")
		shown = 0
		subj_name = {r["subject_id"]: r["subject"] for r in rows}
		obj_name = {r["object_id"]: r["object"] for r in rows}
		for (sid, pid), objs in list(multi.items()):
			if shown >= 5:
				break
			rel = next(r["relation"] for r in rows if r["subject_id"] == sid and r["property_id"] == pid)
			names = ", ".join(f"{obj_name.get(o, o)} ({o})" for o in sorted(objs))
			out(f"  - {subj_name.get(sid, sid)} — *{rel}* → {names}")
			shown += 1
	out("")

	# C5: leakage / duplicates
	out("## C5 — split integrity\n")
	ids_by_split = {s: [r["fact_id"] for r in rows if r["split"] == s] for s in SPLITS}
	for s in SPLITS:
		dup = len(ids_by_split[s]) - len(set(ids_by_split[s]))
		out(f"- {s}: {len(ids_by_split[s]):,} rows, {len(set(ids_by_split[s])):,} unique "
		    f"→ **{dup} duplicate rows**")
	tr, va, te = (set(ids_by_split[s]) for s in SPLITS)
	out(f"- train∩test: **{len(tr & te)}**, train∩val: **{len(tr & va)}**, val∩test: **{len(va & te)}**")
	if tr & te:
		out(f"  - leaked fact_ids: {sorted(tr & te)}")
	out("")

	# ---------- per-language configs (streamed: only English is held in memory) ----------
	OPT_COLS = ["fact_id", "option_a", "option_b", "option_c", "option_d",
	            "answer_text", "answer_index"]

	def iter_lang(lang):
		for s in SPLITS:
			t = fetch(repo, lang, s)
			cols = {c: t.column(c).to_pylist() for c in OPT_COLS}
			for i in range(t.num_rows):
				yield {c: cols[c][i] for c in OPT_COLS}
			del cols, t

	# compact English reference: fact_id -> (frozenset(options), gold)
	en_ref = {}
	for r in iter_lang("en"):
		en_ref[r["fact_id"]] = (
			frozenset(r[f"option_{k}"] for k in "abcd"), r["answer_text"])
	print(f"  en reference loaded ({len(en_ref):,} facts)")

	position = {}
	artifact_rows = {}
	bad_dup = bad_gold = 0
	for l in LANGS:
		same_opt = tot_opt = same_gold = tot_gold = wrong_script = 0
		pos = Counter()
		for r in iter_lang(l):
			opts = [r[f"option_{k}"] for k in "abcd"]
			pos[r["answer_index"]] += 1
			if len(set(opts)) != 4:
				bad_dup += 1
			if r["answer_text"] != opts[r["answer_index"]]:
				bad_gold += 1
			ref = en_ref.get(r["fact_id"])
			if ref and l != "en":
				# options are shuffled per language, so compare as sets
				same_opt += len(set(opts) & ref[0])
				tot_opt += 4
				same_gold += int(r["answer_text"] == ref[1])
				tot_gold += 1
				if l in NON_LATIN and script_of(r["answer_text"]) == "LATIN":
					wrong_script += 1
		position[l] = pos
		if l != "en":
			artifact_rows[l] = (same_opt / max(tot_opt, 1), same_gold / max(tot_gold, 1),
			                    wrong_script / max(tot_gold, 1) if l in NON_LATIN else None)
		print(f"  scanned {l}")

	out("## C6 — translation artifacts / English-canonical options\n")
	out("Share of a language's 4 options that are byte-identical to an English option "
	    "for the same fact (an untranslated label), and share of *gold answers* "
	    "identical to English:\n")
	out("| lang | options identical to en | gold identical to en | gold in Latin script |")
	out("|---|---|---|---|")
	for l in LANGS:
		if l == "en":
			continue
		o, g, w = artifact_rows[l]
		ws = f"{100 * w:.1f}%" if w is not None else "—"
		out(f"| {l} | {100 * o:.1f}% | {100 * g:.1f}% | {ws} |")
	out("")

	out("## C7 — option position balance\n")
	out("| lang | A | B | C | D | max−min |")
	out("|---|---|---|---|---|---|")
	for l in LANGS:
		c = position[l]
		n = sum(c.values())
		pcts = [100 * c[i] / n for i in range(4)]
		out(f"| {l} | " + " | ".join(f"{p:.1f}%" for p in pcts) +
		    f" | {max(pcts) - min(pcts):.1f}pp |")
	out("")

	out("## Option integrity\n")
	out(f"- (fact, language) items with duplicate options: **{bad_dup:,}**")
	out(f"- items where `answer_text` != `option_{{answer_index}}`: **{bad_gold:,}**")
	out("")
	return rels, rows


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--also_full", action="store_true",
	                help="also audit jvonrad/PolyFact for comparison")
	ap.add_argument("--out", default="results/polyfact_clean_audit.md")
	args = ap.parse_args()

	lines = []
	rels_clean, rows_clean = audit(args.repo, lines)

	if args.also_full:
		lines.append("\n---\n")
		rels_full, _ = audit("jvonrad/PolyFact", lines)
		lines.append("\n## C4 — relations removed by the Clean filter\n")
		removed = sorted(set(rels_full) - set(rels_clean))
		lines.append(f"- removed: **{', '.join(removed)}**")
		kept_but_named = [r for r in ["employer", "genre", "country of origin", "place of birth"]
		                  if r in rels_clean]
		lines.append(f"- named in the kSu5 rebuttal as excluded but still present: "
		             f"**{', '.join(kept_but_named) or 'none'}**")
		print("\n".join(lines[-4:]))

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
