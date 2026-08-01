#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-coverage census with the STRICT prompt, over a full split (test or validation).

The original test review (16,687 items) used a prompt that turned out to
undercount: the same reviewers, given a prompt that names known bug classes and
warns about morphology false positives, found ~1.4x more defects on IDENTICAL
test data in a controlled comparison. This regenerates worksheets for every fact
in the given split (not a sample) with that stricter prompt, so the release can
honestly claim "every evaluation item was verified" rather than "a sample was".

Also used for validation: the earlier validation review (`make_validation_review.py`,
300/600-fact sample) was drawn from the pre-rebalance 1,464-fact validation pool.
After `rebalance_splits.py` moved 1,020 of those facts into test, the 444 facts left
in validation have ZERO overlap with that sample (confirmed by diffing fact_ids) —
every one of them is currently unreviewed. `--split validation` closes that gap with
the same strict methodology already used for test, at a fraction of the size.

Two tasks, same split as `make_validation_review.py`:
  TRANSLATION (11 non-English languages) — does the question ask about the same
  subject and property as English?
  ENGLISH-vs-GOLD (English) — does the English question, as written, have the
  gold as its answer? This is the class every prior pass was structurally blind
  to (it used English as the reference); a controlled 600-fact sample found it at
  3.0% [1.9%, 4.7%].

Chunked at ~400 items per worksheet so this is a full census, not a sample, while
keeping each agent's task tractable.

CPU-only; safe on a login node.
"""

import argparse
import json
import os

import pyarrow.parquet as pq

LANGS = ["de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

TRANS_HEAD = """# Full {split} census — translation review — {lang} (part {part}, {n} items)

For EVERY row, judge whether the **{lang}** question asks about the same subject
and the same property as the English question.

- `ok` — same subject, same property. Translation-style differences, localized or
  transliterated titles, and awkward phrasing are all `ok`. Prefer `ok` when a
  reader would still land on the right entity.
- `subject` — asks about a DIFFERENT entity (severe). Real examples found in this
  corpus: a village replaced by a similar-sounding country (Özbek → Uzbekistan);
  a film's question replaced by a different film's; a proper name meaning-
  translated into an unrelated common noun (the software "DIET" -> "a dietary
  regimen"); a building substituted for a different, more famous one (a
  Darmstadt church -> St Isaac's Cathedral).
- `type` — right subject, wrong kind of thing: a TV episode called a film, a song
  called a book, a settlement turned into a company or a person ("born" instead
  of "located"), a PLACE treated as if it were a LANGUAGE (a village rendered as
  "the X language", then asked what country it is official in), or a proper
  name read as a common noun and a month (the town "Septemvri" read as the month
  September).
- `relation` — wrong property, including reversed relations ("who discovered X"
  becoming "what did X discover", "educated at" becoming "taught at", "located
  in" becoming "founded in" or "born in").
- `unsure` — cannot tell.

Note: morphological variation is NOT a defect. Russian declines proper nouns,
German compounds and case-inflects them, Bengali fuses Latin-script names to
native suffixes. Only flag `subject` when a DIFFERENT real-world entity is named.

| # | fact_id | relation | subject | english question | {lang} question | {lang} gold |
|---|---|---|---|---|---|---|
{rows}

Write JSON to `{out}` — one object per row, ALL {n} rows:
`[{{"fact_id":"Q1|P2|Q3","verdict":"ok"}}, {{"fact_id":"...","verdict":"subject","note":"..."}}]`
Include "note" only for non-`ok` rows.
"""

EN_HEAD = """# Full {split} census — ENGLISH-vs-GOLD review (part {part}, {n} items)

This is NOT a translation check and NOT a grammar check. For every row, judge one
thing: **would this English question, exactly as written, have the stored gold as
its correct answer?**

The stored triple (subject / relation / object) is generally reliable. The
question was written by a language model and can add specifics the triple never
claimed — a year, a place, a medium, a qualifier. When such an invented specific
points at a DIFFERENT real-world entity than the gold, the item is broken even
though the triple is fine.

Real examples found in this corpus:

- *"Who directed the **1979** film Darr?"* — the film is from **1993**.
- *"Who directed the **1962** film The Devil's Hand?"* — gold Christian E.
  Christiansen directed the **2014** film.
- *"Who created the **statue** of Toussaint Louverture in Nantes, France?"* —
  gold Philippe Niang directed the **2012 TV film**.

Verdicts:
- `ok` — the question as written has the gold as its answer.
- `conflict` — a detail in the question (year, medium, location, qualifier)
  contradicts the gold, so the question's true answer is something else.
- `vague` — the question is too underspecified to have a unique answer, but
  nothing in it actively contradicts the gold.
- `unsure` — you cannot determine this without research you cannot do.

Use your own world knowledge; use WebSearch sparingly, only to settle a specific
suspected conflict, not for all rows. If you are confident a stated detail is
wrong for the gold entity, say `conflict` and name the discrepancy. If you simply
do not know the entity, `unsure` — do not guess.

| # | fact_id | relation | subject | english question | gold answer |
|---|---|---|---|---|---|
{rows}

Write JSON to `{out}` — one object per row, ALL {n} rows:
`[{{"fact_id":"Q1|P2|Q3","verdict":"ok"}}, {{"fact_id":"...","verdict":"conflict","note":"question says 1979, film is from 1993"}}]`
Include "note" only for non-`ok` rows.
"""


def chunks(seq, n):
	for i in range(0, len(seq), n):
		yield i // n, seq[i:i + n]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--chunk", type=int, default=400)
	ap.add_argument("--split", default="test", choices=("test", "validation"))
	args = ap.parse_args()
	os.makedirs(args.out_dir, exist_ok=True)

	test = pq.read_table(os.path.join(args.data_dir, "data", "parallel",
	                                  f"{args.split}.parquet")).to_pylist()
	print(f"{len(test):,} {args.split} facts")

	prefix = "census" if args.split == "test" else f"census_{args.split}"
	index = []
	for l in LANGS:
		for part, ch in chunks(test, args.chunk):
			rows = []
			for i, r in enumerate(ch, 1):
				tr = r["translations"][l]
				rows.append(f"| {i} | {r['fact_id']} | {r['relation']} | "
				            f"{r['subject']} | {r['translations']['en']['question']} "
				            f"| **{tr['question']}** | {tr['answer_text']} |")
			out = os.path.join(args.out_dir, f"{prefix}_{l}_{part:02d}.json")
			path = os.path.join(args.out_dir, f"{prefix}_{l}_{part:02d}.md")
			with open(path, "w", encoding="utf-8") as f:
				f.write(TRANS_HEAD.format(split=args.split, lang=l, part=part, n=len(rows),
				                          rows="\n".join(rows), out=out))
			index.append({"task": "translation", "lang": l, "part": part,
			              "n": len(rows), "file": path, "out": out})
	for part, ch in chunks(test, args.chunk):
		rows = []
		for i, r in enumerate(ch, 1):
			rows.append(f"| {i} | {r['fact_id']} | {r['relation']} | {r['subject']} "
			            f"| {r['translations']['en']['question']} "
			            f"| **{r['translations']['en']['answer_text']}** |")
		out = os.path.join(args.out_dir, f"{prefix}_en_gold_{part:02d}.json")
		path = os.path.join(args.out_dir, f"{prefix}_en_gold_{part:02d}.md")
		with open(path, "w", encoding="utf-8") as f:
			f.write(EN_HEAD.format(split=args.split, part=part, n=len(rows), rows="\n".join(rows),
			                       out=out))
		index.append({"task": "english_gold", "lang": "en", "part": part,
		              "n": len(rows), "file": path, "out": out})

	with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
		json.dump(index, f, ensure_ascii=False)
	n_trans = sum(x["n"] for x in index if x["task"] == "translation")
	n_en = sum(x["n"] for x in index if x["task"] == "english_gold")
	print(f"{len([x for x in index if x['task']=='translation'])} translation "
	      f"worksheets ({n_trans:,} items)")
	print(f"{len([x for x in index if x['task']=='english_gold'])} english-gold "
	      f"worksheets ({n_en:,} items)")
	print(f"wrote {args.out_dir}/index.json")


if __name__ == "__main__":
	main()
