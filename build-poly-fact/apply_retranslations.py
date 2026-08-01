#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply the re-translated test questions, gating every one on mechanical checks.

The translators were given the English question, the broken target-language
question and the reviewer's diagnosis, and asked to write a correct question.
Nothing they produce is trusted on the strength of that instruction alone: a
model asked to repair hallucinations can introduce new ones, and the failure
modes here are known and checkable, so they are checked.

A proposed question is REJECTED (the item keeps its old text and stays flagged
`question_verified = False`) if any of these fire:

  leakage       the gold answer occurs in the question. This is the defect the
                pipeline's own `filter_question_defects.py` stage exists to
                remove: a copied answer is byte-identical in every language, so a
                leaked item scores as perfectly cross-lingually consistent no
                matter what the model knows — inflating exactly the metric this
                dataset is built to measure. Word-boundary aware for space-using
                scripts, plain substring for ja/zh.
  distractor    an option other than the gold occurs in the question, which makes
                that distractor spuriously attractive to a prompt-copying model.
                Same rule the resample stage already enforces.
  script        the question is not written in the target language's script (a
                translator that echoed the English, or answered in the wrong
                language).
  length        implausibly short, or wildly longer than this language's real
                questions — a proxy for commentary, refusals, or truncation.
  unchanged     byte-identical to the broken question it was meant to replace.
  null          the translator declined (it was told to prefer null over
                inventing a name it could not verify).

Accepted questions get `question_verified = True` and `question_regenerated =
True`, so a reader can always identify — and exclude — every item whose text did
not come from the original generator. That column is the reason this is a
defensible repair rather than a silent rewrite of a benchmark.

CPU-only, streamed; safe on a login node.
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPT = ["option_a", "option_b", "option_c", "option_d"]
NO_WORD_BOUNDARY = {"ja", "zh"}
# Minimum gold length to bother leak-checking. 3 is right for alphabetic scripts
# (shorter strings false-positive inside longer words), but CJK writes whole words
# in 2 characters — 日本 is "Japan", 苏联 is "Soviet Union" — so a 3-char floor
# silently exempted every short CJK gold from the check. That gap left 63 genuinely
# answer-leaking items in the corpus.
MIN_GOLD_LEN = 3
MIN_GOLD_LEN_CJK = 2

# Facts where the ENGLISH question contradicts the stored gold, found incidentally
# by translators who checked the gold against what the English asked. The triple is
# fine and the English question invented a false specifier, so no translation can
# be correct: a faithful one inherits the error, an unfaithful one drifts. Left
# untranslated and still flagged, for a human to resolve at the source.
#   Q3522061 en says "the 1962 film The Devil's Hand"; gold Christian E.
#            Christiansen directed the 2014 film.
#   Q3535650 en says "the statue of Toussaint Louverture in Nantes"; gold Philippe
#            Niang directed the 2012 TV film.
ENGLISH_GOLD_CONFLICT = {"Q3522061|P57|Q5109556", "Q3535650|P170|Q3380445"}
# the script a correct question in this language must actually contain
SCRIPT = {"ar": "ARABIC", "bn": "BENGALI", "ru": "CYRILLIC", "zh": "CJK",
          "ja": ("CJK", "HIRAGANA", "KATAKANA")}


def leaks(needle, question, lang):
	floor = MIN_GOLD_LEN_CJK if lang in NO_WORD_BOUNDARY else MIN_GOLD_LEN
	if not needle or len(needle) < floor:
		return False
	n, q = needle.casefold(), question.casefold()
	if lang in NO_WORD_BOUNDARY:
		return n in q
	return re.search(rf"(?<!\w){re.escape(n)}(?!\w)", q) is not None


def has_script(text, lang):
	want = SCRIPT.get(lang)
	if not want:
		return True                      # Latin-script languages: nothing to assert
	want = (want,) if isinstance(want, str) else want
	for ch in text:
		if not ch.isalpha():
			continue
		try:
			name = unicodedata.name(ch).split()[0]
		except ValueError:
			continue
		if name in want:
			return True
	return False


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--retrans_dir", required=True)
	ap.add_argument("--review_dir", required=True)
	ap.add_argument("--report", default="results/retranslation_report.md")
	ap.add_argument("--classes", default="subject,relation,type",
	                help="which defect classes to actually rewrite. `subject` and "
	                     "`relation` are unambiguous; `type` carries most of the "
	                     "reviewer's judgement calls, so pass "
	                     "'subject,relation' to rebuild conservatively and leave "
	                     "type-flagged items with their original text.")
	args = ap.parse_args()
	apply_classes = {c.strip() for c in args.classes.split(",") if c.strip()}

	def src(split, config):
		return os.path.join(args.data_dir, "data", config, f"{split}.parquet")

	# ---- current verdicts, so rejected items keep question_verified=False ----
	Q_RE = re.compile(r"verdicts_questions_([a-z]+)_(\d+)\.json$")
	verdict = {}
	for fn in sorted(os.listdir(args.review_dir)):
		m = Q_RE.search(fn)
		if not m:
			continue
		with open(os.path.join(args.review_dir, fn), encoding="utf-8") as f:
			for r in json.load(f):
				verdict[(r["fact_id"], m.group(1))] = r.get("verdict", "unsure")

	# ---- proposed replacements ----
	proposed = {}
	for fn in sorted(os.listdir(args.retrans_dir)):
		m = re.match(r"retranslated_([a-z]+)\.json$", fn)
		if not m:
			continue
		lang = m.group(1)
		with open(os.path.join(args.retrans_dir, fn), encoding="utf-8") as f:
			for r in json.load(f):
				if r.get("question"):
					proposed[(r["fact_id"], lang)] = r["question"].strip()
	print(f"{len(proposed):,} proposed re-translations loaded")

	# ---- length reference per language, from the existing corpus ----
	test = pq.read_table(src("test", "parallel")).to_pylist()
	lens = defaultdict(list)
	for r in test:
		for l in LANGS:
			lens[l].append(len(r["translations"][l]["question"]))
	bounds = {l: (max(8, int(0.35 * (sum(v) / len(v)))), int(3.0 * (sum(v) / len(v))))
	          for l, v in lens.items()}

	# ---- gate every proposal ----
	accepted, rejected = {}, []
	for r in test:
		for l in LANGS:
			key = (r["fact_id"], l)
			new = proposed.get(key)
			if new is None:
				continue
			tr = r["translations"][l]
			gold = tr["answer_text"]
			others = [tr[c] for c in OPT if tr[c] != gold]
			why = None
			if r["fact_id"] in ENGLISH_GOLD_CONFLICT:
				why = "English question contradicts its own gold; unfixable here"
			elif verdict.get(key) not in apply_classes:
				why = f"class not selected ({verdict.get(key)})"
			elif new == tr["question"]:
				why = "unchanged"
			elif leaks(gold, new, l):
				why = "leakage: contains the gold answer"
			elif any(leaks(o, new, l) for o in others):
				why = "contains a distractor"
			elif not has_script(new, l):
				why = "wrong script"
			elif not (bounds[l][0] <= len(new) <= bounds[l][1]):
				why = f"implausible length {len(new)} (expected {bounds[l]})"
			if why:
				rejected.append({"fact_id": r["fact_id"], "lang": l, "why": why,
				                 "proposed": new[:160]})
			else:
				accepted[key] = new
	print(f"accepted {len(accepted):,}, rejected {len(rejected):,}")

	# ---- write ----
	for c in LANGS + ["parallel"]:
		os.makedirs(os.path.join(args.out_dir, "data", c), exist_ok=True)

	def flags(fid, l):
		"""(question_verified, question_regenerated) after this stage."""
		if (fid, l) in accepted:
			return True, True
		v = verdict.get((fid, l))
		if v is None:
			# English is null, not True. An earlier build marked it True "by
			# construction" because it was the review's reference — but the
			# review never checked English against the gold, and translators
			# then found English questions that contradict their own gold
			# (ENGLISH_GOLD_CONFLICT). `True` was an overclaim; null is the
			# honest value for "no claim was made".
			return None, False
		if v == "unsure":
			return None, False
		return v == "ok", False

	for s in SPLITS:
		for l in LANGS:
			t = pq.read_table(src(s, l))
			fids = t.column("fact_id").to_pylist()
			if s == "test":
				qs = t.column("question").to_pylist()
				qv, qr = [], []
				for i, fid in enumerate(fids):
					if (fid, l) in accepted:
						qs[i] = accepted[(fid, l)]
					a, b = flags(fid, l)
					qv.append(a)
					qr.append(b)
				t = t.set_column(t.schema.get_field_index("question"), "question",
				                 pa.array(qs, type=pa.string()))
				t = t.set_column(t.schema.get_field_index("question_verified"),
				                 "question_verified", pa.array(qv, type=pa.bool_()))
			else:
				qr = [False] * t.num_rows
			t = t.append_column("question_regenerated",
			                    pa.array(qr, type=pa.bool_()))
			pq.write_table(t, os.path.join(args.out_dir, "data", l, f"{s}.parquet"),
			               compression="snappy")

		par = pq.read_table(src(s, "parallel"))
		outer = par.schema.field("translations").type
		per_lang = pa.struct(list(outer.field(LANGS[0]).type) +
		                     [pa.field("question_regenerated", pa.bool_())])
		par_schema = pa.schema([
			f if f.name != "translations" else
			pa.field("translations", pa.struct([pa.field(l, per_lang)
			                                    for l in LANGS]))
			for f in par.schema])
		rows = par.to_pylist()
		for r in rows:
			for l in LANGS:
				tr = r["translations"][l]
				if s == "test":
					if (r["fact_id"], l) in accepted:
						tr["question"] = accepted[(r["fact_id"], l)]
					a, b = flags(r["fact_id"], l)
					tr["question_verified"] = a
					tr["question_regenerated"] = b
				else:
					tr["question_regenerated"] = False
		pq.write_table(pa.Table.from_pylist(rows, schema=par_schema),
		               os.path.join(args.out_dir, "data", "parallel", f"{s}.parquet"),
		               compression="snappy")
		print(f"  wrote {s}")

	by_lang = Counter(l for _, l in accepted)
	rej_why = Counter(x["why"].split(":")[0] for x in rejected)
	# Only items the review flagged were ever offered for re-translation, but the
	# three verdict classes are not equally certain: `subject` (asks about a
	# different entity) is unambiguous, while `type` (right subject, wrong kind of
	# thing) carries most of the judgement calls. Reporting the split makes it
	# visible how much of the rewriting rests on the softer class.
	by_class = Counter(verdict.get(k, "?") for k in accepted)
	flagged_class = Counter(v for v in verdict.values()
	                        if v in ("subject", "type", "relation"))
	lines = ["# Re-translation of defective test questions\n",
	         f"- proposed: **{len(proposed):,}**",
	         f"- accepted: **{len(accepted):,}**",
	         f"- rejected by automatic gates: **{len(rejected):,}**\n",
	         "Only questions the manual review flagged as defective were offered "
	         "for re-translation; the 16,207 reviewed questions marked `ok` or "
	         "`unsure` were never shown to a translator, and no answer label was "
	         "touched at this stage.\n",
	         "## Accepted by original defect class\n",
	         "| verdict class | flagged | rewritten | certainty |",
	         "|---|---|---|---|"]
	CERT = {"subject": "unambiguous — asks about a different entity",
	        "relation": "mostly unambiguous — wrong property",
	        "type": "**judgement-heavy** — right subject, wrong kind of thing"}
	for c in ("subject", "relation", "type"):
		lines.append(f"| {c} | {flagged_class[c]} | {by_class[c]} | {CERT[c]} |")
	lines += ["", "## Accepted per language\n",
	          "| lang | accepted |", "|---|---|"]
	lines += [f"| {l} | {n} |" for l, n in sorted(by_lang.items())]
	lines += ["", "## Rejection reasons\n", "| reason | n |", "|---|---|"]
	lines += [f"| {w} | {n} |" for w, n in rej_why.most_common()]
	if rejected:
		lines += ["", "## Rejected items\n"]
		lines += [f"- `{x['fact_id']}` [{x['lang']}] — {x['why']}\n  - proposed: {x['proposed']}"
		          for x in rejected[:60]]
	os.makedirs(os.path.dirname(args.report), exist_ok=True)
	with open(args.report, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(os.path.join(args.out_dir, "retranslations_applied.json"), "w",
	          encoding="utf-8") as f:
		json.dump({"accepted": [{"fact_id": k[0], "lang": k[1], "question": v}
		                        for k, v in accepted.items()],
		           "rejected": rejected}, f, ensure_ascii=False, indent=1)
	print(f"\nwrote {args.report}")


if __name__ == "__main__":
	main()
