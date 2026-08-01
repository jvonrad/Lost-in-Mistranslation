#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the manual-review worksheets for PolyFact-Clean.

Two error classes need human (or model) judgement, and they have different units:

  LABELS are per ENTITY. 59,980 facts carry only 15,977 distinct answer entities,
  and because distractors are drawn from the gold-entity pool, judging an entity's
  label fixes it everywhere it appears — as a gold and as an option, in all three
  splits. Only the labels that are BOTH unattested by Wikidata AND semantically
  distant from the English name can hide a wrong referent (the phonetic ones
  scored 0 errors in 180 reviewed), which is 7,708 judgements covering the whole
  release.

  QUESTIONS are per FACT and do not dedupe (1,515 distinct subjects across 1,517
  test facts), so only the test split is reviewable at all. This is the class an
  external review surfaced: `Q1361995` stores the commune Jessains but its Arabic
  question asks about the Falkland Islands.

Worksheets are grouped so a reviewer sees the same entity's other 11 languages
side by side — cross-language triangulation is the strongest signal available
where Wikidata is silent. Output is one file per language (chunked), plus a
matching empty verdict file the reviewer fills in.

CPU-only; safe on a login node.
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from rapidfuzz import fuzz
from unidecode import unidecode

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
VARIANTS = {"en": ["en-gb", "en-ca"], "de": ["de-at", "de-ch"], "pt": ["pt-br"],
            "es": ["es-419"],
            "zh": ["zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk", "zh-sg", "zh-mo"]}
WIKI = {l: f"{l}wiki" for l in LANGS}
PAREN = re.compile(r"\s*[\(（][^()（）]*[\)）]\s*$")
NONALNUM = re.compile(r"[^a-z0-9]+")
VOWELS = re.compile(r"[aeiou]")
SEM_CUTOFF = 60

try:
	from hanziconv import HanziConv
	def fold(s):
		return HanziConv.toSimplified(s)
except ImportError:
	def fold(s):
		return s


def norm(s, lang=None):
	s = unicodedata.normalize("NFKC", s).strip().casefold()
	return fold(s) if lang == "zh" else s


def rom(s):
	return NONALNUM.sub(" ", unidecode(unicodedata.normalize("NFKC", s)).lower()).strip()


def sim(a, b):
	x, y = rom(a), rom(b)
	if not x or not y:
		return 0.0
	sc = [fuzz.ratio(x, y), fuzz.token_set_ratio(x, y), fuzz.partial_ratio(x, y)]
	sx, sy = VOWELS.sub("", x), VOWELS.sub("", y)
	if sx and sy:
		sc.append(fuzz.ratio(sx, sy))
	return max(sc)


def attested(label, lang, w):
	n = norm(label, lang)
	for c in [lang] + VARIANTS.get(lang, []):
		if w["labels"].get(c) and norm(w["labels"][c], lang) == n:
			return True
		if any(norm(a, lang) == n for a in w["aliases"].get(c, [])):
			return True
	t = w["sitelinks"].get(WIKI[lang])
	return bool(t) and n in (norm(t, lang), norm(PAREN.sub("", t), lang))


def chunks(seq, n):
	for i in range(0, len(seq), n):
		yield i // n, seq[i:i + n]


LABEL_HEAD = """# Label review — {lang} (part {part})

{n} answer labels that Wikidata cannot confirm and that are NOT phonetic
renderings of the English name, so each is a translation that could denote the
wrong thing. Judge whether the **{lang}** label names the same entity as the
English one.

Verdict for each row:
  `ok`     names the right entity (a correct translation OR a fair transliteration)
  `wrong`  names something else, or is not a word in this language
  `unsure` cannot tell without a native speaker

`ok` covers awkward-but-identifiable. Reserve `wrong` for a reader being pointed at
a different thing: `Pembe` (an animal's horn) for the town Horn, `鳄鱼` (crocodile)
for the game Croc, `Hifadhi` (a nature reserve) for a music conservatory.

`n_facts` = how many facts this entity is the answer to; `others` = the same
entity in other languages, for triangulation.

| # | qid | relation | english | {lang} label | romanised | n_facts | others |
|---|---|---|---|---|---|---|---|
"""

Q_HEAD = """# Test-split question review — {lang} (part {part})

{n} test questions. Judge whether the **{lang}** question asks about the same
subject, and the same kind of thing, as the English question.

Verdict for each row:
  `ok`       same subject, same question
  `subject`  asks about a DIFFERENT entity (the severe class)
  `type`     right subject, wrong entity type ("the series Little Children" for a
             film; "the month Septemvri" for a town)
  `relation` asks for the wrong property ("municipality of citizenship" instead of
             country)
  `unsure`   cannot tell

`wikidata_subject` is what Wikidata calls this subject in {lang}; if the question
uses a different name for the subject, that is the `subject` class.

| # | fact_id | relation | subject | wikidata_subject | english question | {lang} question | {lang} gold |
|---|---|---|---|---|---|---|---|
"""


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--obj_cache", default=None)
	ap.add_argument("--subj_cache", default=None)
	ap.add_argument("--label_chunk", type=int, default=350)
	ap.add_argument("--question_chunk", type=int, default=400)
	args = ap.parse_args()
	sc = os.environ.get("SCRATCH", ".")
	obj = json.load(open(args.obj_cache or f"{sc}/wikidata_deep_cache.json",
	                     encoding="utf-8"))
	subj_path = args.subj_cache or f"{sc}/wikidata_subject_cache.json"
	subj = json.load(open(subj_path, encoding="utf-8")) if os.path.exists(subj_path) else {}

	def src(s):
		return os.path.join(args.data_dir, "data", "parallel", f"{s}.parquet")

	ent = defaultdict(dict)
	nf = Counter()
	rel_of = {}
	test_rows = []
	for s in SPLITS:
		for b in pq.ParquetFile(src(s)).iter_batches(batch_size=2000):
			for r in b.to_pylist():
				nf[r["object_id"]] += 1
				rel_of[r["object_id"]] = r["relation"]
				for l in LANGS:
					ent[r["object_id"]][l] = r["translations"][l]["answer_text"]
				if s == "test":
					test_rows.append(r)
			del b
	print(f"{len(ent):,} entities, {len(test_rows):,} test facts")

	os.makedirs(args.out_dir, exist_ok=True)
	index = {"labels": [], "questions": []}

	# ---------------- label worksheets ----------------
	per_lang = defaultdict(list)
	for qid, d in ent.items():
		w = obj.get(qid) or {"labels": {}, "aliases": {}, "sitelinks": {}}
		for l in LANGS:
			if attested(d[l], l, w) or sim(d[l], d["en"]) >= SEM_CUTOFF:
				continue
			per_lang[l].append(qid)
	for l, qids in per_lang.items():
		qids.sort(key=lambda q: -nf[q])                # costliest entities first
		for part, ch in chunks(qids, args.label_chunk):
			path = os.path.join(args.out_dir, f"labels_{l}_{part:02d}.md")
			rows = []
			for i, qid in enumerate(ch, 1):
				d = ent[qid]
				others = "; ".join(f"{o}={d[o]}" for o in LANGS
				                   if o not in (l, "en"))[:150]
				rows.append(f"| {i} | {qid} | {rel_of[qid]} | {d['en']} | "
				            f"**{d[l]}** | {rom(d[l])} | {nf[qid]} | {others} |")
			with open(path, "w", encoding="utf-8") as f:
				f.write(LABEL_HEAD.format(lang=l, part=part, n=len(ch))
				        + "\n".join(rows) + "\n")
			index["labels"].append({"lang": l, "part": part, "n": len(ch),
			                        "file": path,
			                        "qids": ch})
	print(f"label worksheets: {len(index['labels'])} files, "
	      f"{sum(x['n'] for x in index['labels']):,} judgements")

	# ---------------- test question worksheets ----------------
	for l in LANGS:
		if l == "en":
			continue
		items = []
		for r in test_rows:
			w = subj.get(r["subject_id"]) or {"labels": {}, "aliases": {},
			                                  "sitelinks": {}}
			names = [w["labels"].get(c) for c in [l] + VARIANTS.get(l, [])
			         if w["labels"].get(c)]
			t = w["sitelinks"].get(WIKI[l])
			if t:
				names.append(t)
			items.append((r, names[0] if names else ""))
		for part, ch in chunks(items, args.question_chunk):
			path = os.path.join(args.out_dir, f"questions_{l}_{part:02d}.md")
			rows = []
			for i, (r, wname) in enumerate(ch, 1):
				tr = r["translations"][l]
				rows.append(f"| {i} | {r['fact_id']} | {r['relation']} | "
				            f"{r['subject']} | {wname or '—'} | "
				            f"{r['translations']['en']['question']} | "
				            f"**{tr['question']}** | {tr['answer_text']} |")
			with open(path, "w", encoding="utf-8") as f:
				f.write(Q_HEAD.format(lang=l, part=part, n=len(ch))
				        + "\n".join(rows) + "\n")
			index["questions"].append({"lang": l, "part": part, "n": len(ch),
			                           "file": path,
			                           "fact_ids": [r["fact_id"] for r, _ in ch]})
	print(f"question worksheets: {len(index['questions'])} files, "
	      f"{sum(x['n'] for x in index['questions']):,} judgements")

	with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
		json.dump(index, f, ensure_ascii=False)
	print(f"wrote {args.out_dir}/index.json")


if __name__ == "__main__":
	main()
