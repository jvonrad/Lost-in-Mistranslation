#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate the manual-review verdicts into per-language rates and a drop list.

Reads every `verdicts_labels_*.json` and `verdicts_questions_*.json` written by
the reviewers, checks each file against the worksheet it was supposed to cover
(a reviewer that silently skipped rows would otherwise inflate the pass rate),
and reports:

  * label error rate per language, over the reviewed semantic stratum;
  * the same rate reweighted to the whole release, since the reviewed stratum is
    only the 7,708 labels that Wikidata could not attest AND that are not
    phonetic renderings — the other 185,480 labels are either Wikidata-attested
    or transliterations, which measured 0 errors in 180 sampled;
  * question defect rates on the test split, split by class (`subject` = asks
    about a different entity, `type` = wrong kind of thing, `relation` = wrong
    property);
  * the entity and fact drop lists.

CPU-only.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

LABEL_RE = re.compile(r"verdicts_labels_([a-z]+)_(\d+)\.json$")
Q_RE = re.compile(r"verdicts_questions_([a-z]+)_(\d+)\.json$")
# every label outside the reviewed stratum: Wikidata-attested, or a phonetic
# rendering (0 errors in 180 reviewed, 95% CI upper bound 2.1%)
TOTAL_LABELS = 193188


def load(path):
	try:
		with open(path, encoding="utf-8") as f:
			return json.load(f)
	except Exception as e:
		print(f"  !! {os.path.basename(path)}: {e}")
		return None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--review_dir", required=True)
	ap.add_argument("--out", default="results/manual_review_report.md")
	ap.add_argument("--drops_out", default="results/manual_review_drops.json")
	args = ap.parse_args()

	with open(os.path.join(args.review_dir, "index.json"), encoding="utf-8") as f:
		index = json.load(f)
	expect_lab = {(x["lang"], x["part"]): x for x in index["labels"]}
	expect_q = {(x["lang"], x["part"]): x for x in index["questions"]}

	lab = defaultdict(Counter)
	lab_bad = {}
	q = defaultdict(Counter)
	q_bad = []
	missing = []
	short = []

	for fn in sorted(os.listdir(args.review_dir)):
		p = os.path.join(args.review_dir, fn)
		m = LABEL_RE.search(fn)
		if m:
			lang, part = m.group(1), int(m.group(2))
			rows = load(p)
			if rows is None:
				continue
			exp = expect_lab.get((lang, part))
			if exp and len(rows) < exp["n"]:
				short.append(f"labels {lang}_{part:02d}: {len(rows)}/{exp['n']}")
			for r in rows:
				v = r.get("verdict", "unsure")
				lab[lang][v] += 1
				if v in ("wrong", "unsure"):
					lab_bad[r["qid"]] = {"lang": lang, "verdict": v,
					                     "meaning": r.get("meaning", ""),
					                     "suggest": r.get("suggest", "")}
			continue
		m = Q_RE.search(fn)
		if m:
			lang, part = m.group(1), int(m.group(2))
			rows = load(p)
			if rows is None:
				continue
			exp = expect_q.get((lang, part))
			if exp and len(rows) < exp["n"]:
				short.append(f"questions {lang}_{part:02d}: {len(rows)}/{exp['n']}")
			for r in rows:
				v = r.get("verdict", "unsure")
				q[lang][v] += 1
				if v != "ok":
					q_bad.append({"fact_id": r["fact_id"], "lang": lang,
					              "verdict": v, "note": r.get("note", "")})

	for k, x in expect_lab.items():
		if not os.path.exists(os.path.join(
				args.review_dir, f"verdicts_labels_{k[0]}_{k[1]:02d}.json")):
			missing.append(f"labels {k[0]}_{k[1]:02d} ({x['n']} rows)")
	for k, x in expect_q.items():
		if not os.path.exists(os.path.join(
				args.review_dir, f"verdicts_questions_{k[0]}_{k[1]:02d}.json")):
			missing.append(f"questions {k[0]}_{k[1]:02d} ({x['n']} rows)")

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out("# Manual review of PolyFact-Clean\n")
	if missing:
		out(f"**INCOMPLETE — {len(missing)} worksheets not yet reviewed:** "
		    f"{', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}\n")
	if short:
		out(f"**Short files (reviewer skipped rows):** {', '.join(short)}\n")

	# ---------------- labels ----------------
	out("## Answer labels\n")
	out("Reviewed stratum = labels Wikidata cannot attest that are also NOT "
	    "phonetic renderings of the English name. Labels outside it are either "
	    "Wikidata-attested or transliterations (0 errors in 180 sampled).\n")
	out("| lang | reviewed | ok | wrong | unsure | error rate |")
	out("|---|---|---|---|---|---|")
	tot = Counter()
	for l in sorted(lab, key=lambda x: -sum(lab[x].values())):
		c = lab[l]
		n = sum(c.values())
		tot.update(c)
		out(f"| {l} | {n:,} | {c['ok']:,} | **{c['wrong']:,}** | {c['unsure']:,} | "
		    f"**{100*c['wrong']/n:.1f}%** |")
	N = sum(tot.values())
	if N:
		out(f"| **all** | **{N:,}** | {tot['ok']:,} | **{tot['wrong']:,}** | "
		    f"{tot['unsure']:,} | **{100*tot['wrong']/N:.1f}%** |")
		out("")
		out(f"- distinct entities with a confirmed-wrong label: "
		    f"**{len({k for k, v in lab_bad.items() if v['verdict'] == 'wrong'}):,}**")
		out(f"- as a share of all {TOTAL_LABELS:,} answer labels in the release: "
		    f"**{100*tot['wrong']/TOTAL_LABELS:.2f}%**")
	out("")

	# ---------------- questions ----------------
	if q:
		out("## Test-split questions\n")
		out("`subject` = asks about a different entity (severe). `type` = right "
		    "subject, wrong kind of thing. `relation` = wrong property.\n")
		out("| lang | reviewed | ok | subject | type | relation | unsure | defect rate |")
		out("|---|---|---|---|---|---|---|---|")
		qt = Counter()
		for l in sorted(q, key=lambda x: -sum(q[x].values())):
			c = q[l]
			n = sum(c.values())
			qt.update(c)
			bad = n - c["ok"]
			out(f"| {l} | {n:,} | {c['ok']:,} | **{c['subject']:,}** | {c['type']:,} | "
			    f"{c['relation']:,} | {c['unsure']:,} | **{100*bad/n:.1f}%** |")
		QN = sum(qt.values())
		out(f"| **all** | **{QN:,}** | {qt['ok']:,} | **{qt['subject']:,}** | "
		    f"{qt['type']:,} | {qt['relation']:,} | {qt['unsure']:,} | "
		    f"**{100*(QN-qt['ok'])/QN:.1f}%** |")
		out("")
		facts = {x["fact_id"] for x in q_bad if x["verdict"] in ("subject", "type",
		                                                        "relation")}
		out(f"- distinct test facts with a defect in at least one language: "
		    f"**{len(facts):,}**")
		out("")

	with open(args.drops_out, "w", encoding="utf-8") as f:
		json.dump({
			"union": sorted(k for k, v in lab_bad.items() if v["verdict"] == "wrong"),
			"label_verdicts": lab_bad,
			"question_defects": q_bad,
			"complete": not missing,
		}, f, ensure_ascii=False)
	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out} and {args.drops_out}")


if __name__ == "__main__":
	main()
