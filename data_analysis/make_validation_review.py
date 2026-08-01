#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample the validation split for review, with two distinct tasks.

Validation (1,464 facts) has never been reviewed — the manual pass covered test
only, because test is what the paper's numbers come from. This samples it to
estimate whether validation carries the same defect rate, which matters if
validation drove any model selection.

Two tasks, because the test review had a blind spot:

  TRANSLATION (11 non-English languages) — the same check the test review ran:
  does the target-language question ask about the same subject and property as
  the English one?

  ENGLISH-vs-GOLD (English) — a check nothing has ever run. The test review used
  English as its reference, so it could not detect a wrong English question, and
  English was marked verified "by construction". Re-translators then found
  English questions contradicting their own gold: one asks about "the 1962 film
  The Devil's Hand" while the gold directed the 2014 film; another asks who
  created "the statue of Toussaint Louverture in Nantes" while the gold directed
  the TV film. Neither the triple nor the translations are at fault there — the
  English question invented a false specifier. This task asks, for the first
  time, how common that is.

The English task is deliberately NOT "is this good English" — it is "would this
question, as written, have the stored gold as its answer". That requires checking
the question's factual claims against the gold, which is why it is a separate task
with its own instructions rather than a twelfth language in the translation pass.

Sampling is stratified by relation so no relation is missed, seeded for
reproducibility.

CPU-only; safe on a login node.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import pyarrow.parquet as pq

LANGS = ["de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]

TRANS_HEAD = """# Validation-split translation review — {lang} ({n} items)

For EVERY row, judge whether the **{lang}** question asks about the same subject
and the same property as the English question.

- `ok` — same subject, same property. Translation-style differences, localized or
  transliterated titles, and awkward phrasing are all `ok`. Prefer `ok` when a
  reader would still land on the right entity.
- `subject` — asks about a DIFFERENT entity (severe). Real examples found in the
  test split: a village replaced by a similar-sounding country (Özbek →
  Uzbekistan); a film's question replaced by a different film's.
- `type` — right subject, wrong kind of thing: a TV episode called a film, a song
  called a book, a proper name read as a common noun (the studio "Leaf" as the
  plant part), or a PLACE treated as if it were a LANGUAGE (a known bug — a
  village rendered as "the X language", then asked what country it is official in).
- `relation` — wrong property, including reversed relations ("who discovered X"
  becoming "what did X discover", "educated at" becoming "taught at").
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

EN_HEAD = """# Validation-split ENGLISH-vs-GOLD review ({n} items)

This is NOT a translation check and NOT a grammar check. For every row, judge one
thing: **would this English question, exactly as written, have the stored gold as
its correct answer?**

The stored triple (subject / relation / object) is generally reliable. The
question was written by a language model and can add specifics the triple never
claimed — a year, a place, a medium, a qualifier. When such an invented specific
points at a DIFFERENT real-world entity than the gold, the item is broken even
though the triple is fine.

Two real examples found in the test split:

- *"Who directed the **1962** film The Devil's Hand?"* with gold
  `Christian E. Christiansen` — he directed the **2014** film. The year is
  invented and points elsewhere.
- *"Who created the **statue** of Toussaint Louverture in Nantes, France?"* with
  gold `Philippe Niang` — he directed the **2012 TV film**. The medium is invented.

Verdicts:
- `ok` — the question as written has the gold as its answer.
- `conflict` — a detail in the question (year, medium, location, qualifier)
  contradicts the gold, so the question's true answer is something else.
- `vague` — the question is too underspecified to have a unique answer (e.g. the
  subject is a bare ambiguous name), but nothing in it actively contradicts the
  gold.
- `unsure` — you cannot determine this without research you cannot do.

Use your own world knowledge. If you are confident a stated detail is wrong for
the gold entity, say `conflict` and name the discrepancy. If you simply do not
know the entity, `unsure` — do not guess.

| # | fact_id | relation | subject | english question | gold answer |
|---|---|---|---|---|---|
{rows}

Write JSON to `{out}` — one object per row, ALL {n} rows:
`[{{"fact_id":"Q1|P2|Q3","verdict":"ok"}}, {{"fact_id":"...","verdict":"conflict","note":"question says 1962, gold is the 2014 film"}}]`
Include "note" only for non-`ok` rows.
"""


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--n", type=int, default=300, help="facts to sample")
	ap.add_argument("--n_english", type=int, default=600,
	                help="facts for the English-vs-gold task (cheaper, so larger)")
	ap.add_argument("--seed", type=int, default=20260801)
	ap.add_argument("--split", default="validation",
	                help="which split to sample. Pass `test` to run the SAME "
	                     "prompt over already-reviewed test facts: that is the "
	                     "control that separates a real split difference from a "
	                     "stricter reviewer, since the two passes otherwise differ "
	                     "in both split and prompt at once.")
	args = ap.parse_args()
	rng = random.Random(args.seed)
	os.makedirs(args.out_dir, exist_ok=True)

	val = pq.read_table(os.path.join(args.data_dir, "data", "parallel",
	                                 f"{args.split}.parquet")).to_pylist()
	by_rel = defaultdict(list)
	for r in val:
		by_rel[r["relation"]].append(r)
	for v in by_rel.values():
		rng.shuffle(v)

	def stratified(k):
		"""proportional-by-relation, at least 1 per relation, deterministic"""
		out, i = [], 0
		pools = {rel: list(v) for rel, v in by_rel.items()}
		while len(out) < k and any(pools.values()):
			for rel in sorted(pools):
				if pools[rel] and len(out) < k:
					out.append(pools[rel].pop())
			i += 1
		return out

	sample = stratified(min(args.n, len(val)))
	en_sample = stratified(min(args.n_english, len(val)))
	print(f"translation sample: {len(sample)} facts x {len(LANGS)} langs = "
	      f"{len(sample)*len(LANGS):,} judgements")
	print(f"english-vs-gold sample: {len(en_sample)} facts")

	index = []
	for l in LANGS:
		rows = []
		for i, r in enumerate(sample, 1):
			tr = r["translations"][l]
			rows.append(f"| {i} | {r['fact_id']} | {r['relation']} | {r['subject']} "
			            f"| {r['translations']['en']['question']} | **{tr['question']}** "
			            f"| {tr['answer_text']} |")
		out = os.path.join(args.out_dir, f"valverdicts_{l}.json")
		path = os.path.join(args.out_dir, f"validate_{l}.md")
		with open(path, "w", encoding="utf-8") as f:
			f.write(TRANS_HEAD.format(lang=l, n=len(rows), rows="\n".join(rows),
			                          out=out))
		index.append({"task": "translation", "lang": l, "n": len(rows),
		              "file": path, "out": out})
		print(f"  {l}: {len(rows)} -> {path}")

	rows = []
	for i, r in enumerate(en_sample, 1):
		rows.append(f"| {i} | {r['fact_id']} | {r['relation']} | {r['subject']} "
		            f"| {r['translations']['en']['question']} "
		            f"| **{r['translations']['en']['answer_text']}** |")
	out = os.path.join(args.out_dir, "valverdicts_en_gold.json")
	path = os.path.join(args.out_dir, "validate_en_gold.md")
	with open(path, "w", encoding="utf-8") as f:
		f.write(EN_HEAD.format(n=len(rows), rows="\n".join(rows), out=out))
	index.append({"task": "english_gold", "lang": "en", "n": len(rows),
	              "file": path, "out": out})
	print(f"  en-vs-gold: {len(rows)} -> {path}")

	with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
		json.dump(index, f, ensure_ascii=False)


if __name__ == "__main__":
	main()
