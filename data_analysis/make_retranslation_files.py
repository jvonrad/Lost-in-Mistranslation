#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build worksheets for re-translating the test questions the manual review rejected.

The review found 477 (fact, language) test items whose generated question is
defective: it asks about a different entity (`subject`), about the wrong kind of
thing (`type`), or for the wrong property (`relation`). The stored triple and the
English question are correct in every case — only the target-language question is
broken — so these are repairable by translating the English question properly.

Template-filling from Wikidata was tried first and rejected: 60% of the affected
items have no Wikidata label for their subject in the broken language, so filling
a template would mean inventing the subject's name in that language, which is the
same operation that produced these hallucinations.

Each worksheet carries, per item:
  * the English question (the reference — it is correct)
  * the CURRENT broken question and the reviewer's note on what is wrong with it
  * the gold answer AS ALREADY WRITTEN in the target language (must not be
    changed, and must not appear in the question — see leakage below)
  * the subject and relation

and, per relation, STYLE ANCHORS: real correct questions from this same corpus in
this same language and relation. These matter. Questions in PolyFact were written
by Gemma-3-27B with natural phrasing variation; a translation that reads as
freshly-written by a different model would make the repaired items a
distinguishable subset of a benchmark whose whole purpose is cross-lingual
comparison. The anchors exist so the output matches the corpus register rather
than the translator's own voice.

ANSWER LEAKAGE is the one hard constraint. The build pipeline has a whole stage
(`filter_question_defects.py`) removing questions that contain their own answer,
because a copied answer is identical in every language and therefore scores as
perfect cross-lingual consistency regardless of what the model knows. A
re-translation that mentions the gold answer would silently reintroduce exactly
that defect, so the worksheet states the constraint and the apply stage rechecks
it mechanically.

CPU-only; safe on a login node.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

LANGS = ["de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
LEGACY_Q_RE = re.compile(r"verdicts_questions_([a-z]+)_(\d+)\.json$")
CENSUS_Q_RE = re.compile(r"census_([a-z]{2})_(\d+)\.json$")
CENSUS_VAL_Q_RE = re.compile(r"census_validation_([a-z]{2})_(\d+)\.json$")

HEAD = """# Re-translate defective test questions — {lang} ({n} items)

Every item below has a **correct English question** and a **broken {lang}
question**. Write a correct {lang} question that asks exactly what the English
one asks, about the same subject, for the same property.

## Hard rules

1. **Never include the gold answer in the question.** The answer is shown so you
   know what the question must NOT give away. A question containing its own
   answer is a worse defect than the one you are fixing. This is checked
   mechanically and violations are rejected.
2. **Keep the subject.** The question must name the same entity the English
   question names. Most of these are broken precisely because the subject drifted.
3. **Do not translate a proper name by its meaning.** A studio called "Leaf" is
   not the word for a leaf; the town "Horn" is not an animal's horn. Transliterate
   or keep the original when there is no established {lang} name.
4. **Match the corpus style.** Study the STYLE ANCHORS for the relation — they are
   real, correct questions from this dataset in {lang}. Match their register,
   length and phrasing conventions. Do not write in a noticeably different or more
   polished voice than the anchors.
5. Output the question only — no quotes, no commentary, no romanisation.

## Style anchors by relation

{anchors}

## Items

{items}

## Output

Write JSON to `{out_path}` — one object per item, ALL {n} items, in order:

```json
[{{"fact_id": "Q1|P2|Q3", "question": "<your corrected {lang} question>"}}]
```

If an item is genuinely impossible (e.g. you cannot determine how to name the
subject in {lang} without inventing it), set `"question": null` and add
`"why": "<short reason>"` instead of guessing. A null is far better than a
fabricated name — the item stays flagged and is not silently corrupted.
"""


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--review_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--anchors", type=int, default=6)
	ap.add_argument("--seed", type=int, default=20260801)
	ap.add_argument("--review_format",
	                choices=("legacy", "full_census", "full_census_validation"),
	                default="legacy",
	                help="legacy reads verdicts_questions_LANG_PART.json; "
	                     "full_census reads census_LANG_PART.json (test split); "
	                     "full_census_validation reads census_validation_LANG_PART.json; "
	                     "both full_census* variants ignore the en_gold files")
	ap.add_argument("--split", default="test", choices=("test", "validation"),
	                help="which split's parquet + question_verified anchors to read")
	ap.add_argument("--exclude_retrans_dir",
	                help="optional prior retranslation directory; every "
	                     "(fact_id, lang) present in retranslated_LANG.json is "
	                     "excluded, whether its old proposal passed or failed")
	args = ap.parse_args()
	rng = random.Random(args.seed)
	os.makedirs(args.out_dir, exist_ok=True)

	excluded = set()
	if args.exclude_retrans_dir:
		for fn in sorted(os.listdir(args.exclude_retrans_dir)):
			m = re.match(r"retranslated_([a-z]{2})\.json$", fn)
			if not m:
				continue
			with open(os.path.join(args.exclude_retrans_dir, fn), encoding="utf-8") as f:
				for r in json.load(f):
					excluded.add((r["fact_id"], m.group(1)))
		print(f"{len(excluded)} prior (fact,language) items excluded")

	defects = defaultdict(dict)          # lang -> fact_id -> (verdict, note)
	q_re = {"legacy": LEGACY_Q_RE, "full_census": CENSUS_Q_RE,
	        "full_census_validation": CENSUS_VAL_Q_RE}[args.review_format]
	for fn in sorted(os.listdir(args.review_dir)):
		m = q_re.search(fn)
		if not m:
			continue
		lang = m.group(1)
		with open(os.path.join(args.review_dir, fn), encoding="utf-8") as f:
			for r in json.load(f):
				if (r.get("verdict") in ("subject", "type", "relation") and
						(r["fact_id"], lang) not in excluded):
					defects[lang][r["fact_id"]] = (r["verdict"], r.get("note", ""))
	n_tot = sum(len(v) for v in defects.values())
	print(f"{n_tot} defective (fact,language) items across {len(defects)} languages")

	test = pq.read_table(os.path.join(args.data_dir, "data", "parallel",
	                                  f"{args.split}.parquet")).to_pylist()
	by_fid = {r["fact_id"]: r for r in test}

	# style anchors: correct (question_verified is not False) questions per relation
	anchors = defaultdict(lambda: defaultdict(list))
	for r in test:
		for l in LANGS:
			tr = r["translations"][l]
			if tr.get("question_verified") is False:
				continue
			if r["fact_id"] in defects.get(l, {}):
				continue
			anchors[l][r["relation"]].append(tr["question"])

	index = []
	for lang, items in sorted(defects.items()):
		rels = sorted({by_fid[f]["relation"] for f in items if f in by_fid})
		abloc = []
		for rel in rels:
			pool = anchors[lang].get(rel, [])
			picks = rng.sample(pool, min(args.anchors, len(pool)))
			abloc.append(f"**{rel}**")
			abloc += [f"  - {p}" for p in picks] or ["  - (none available)"]
			abloc.append("")
		rows = []
		for i, (fid, (verdict, note)) in enumerate(sorted(items.items()), 1):
			r = by_fid.get(fid)
			if not r:
				continue
			tr = r["translations"][lang]
			rows.append(
				f"### {i}. `{fid}`  ({verdict})\n"
				f"- relation: **{r['relation']}**\n"
				f"- subject: **{r['subject']}**\n"
				f"- English question (CORRECT, this is the reference): "
				f"{r['translations']['en']['question']}\n"
				f"- current {lang} question (BROKEN): {tr['question']}\n"
				f"- what the reviewer found wrong: {note or '(no note)'}\n"
				f"- gold answer in {lang} (MUST NOT appear in your question): "
				f"**{tr['answer_text']}**\n")
		path = os.path.join(args.out_dir, f"retranslate_{lang}.md")
		out_path = os.path.join(args.out_dir, f"retranslated_{lang}.json")
		with open(path, "w", encoding="utf-8") as f:
			f.write(HEAD.format(lang=lang, n=len(rows), anchors="\n".join(abloc),
			                    items="\n".join(rows), out_path=out_path))
		index.append({"lang": lang, "n": len(rows), "file": path,
		              "out": out_path,
		              "fact_ids": sorted(items)})
		print(f"  {lang}: {len(rows)} items -> {path}")

	with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
		json.dump(index, f, ensure_ascii=False)
	print(f"\nwrote {args.out_dir}/index.json")


if __name__ == "__main__":
	main()
