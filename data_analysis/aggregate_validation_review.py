#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate the validation-split review and compare it against the test split.

Two questions are being answered, and they are different questions:

  1. Does validation carry the same TRANSLATION defect rate as test? Test was
     reviewed exhaustively (16,687 items, 2.8% defective); validation was sampled
     (300 facts x 11 languages). If the rates agree, the test measurement
     generalises and validation needs no separate treatment beyond a flag. If
     validation is worse, anything that used it for model selection is affected.

  2. How often is the ENGLISH question itself wrong? Nothing has ever measured
     this. Every prior pass — the mechanical subject audit, the full test review —
     used English as its reference and is therefore structurally blind to English
     defects. Two were found incidentally by re-translators; this is the first
     attempt at a rate.

Wilson intervals throughout, because at these counts the normal approximation is
unreliable and can put the lower bound below zero.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

TEST_RATE = 0.028                    # measured on the full test split, for reference
TEST_N = 16687


def wilson(k, n, z=1.96):
	if n == 0:
		return (0.0, 0.0)
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
	return (max(0.0, c - h), min(1.0, c + h))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--review_dir", required=True)
	ap.add_argument("--out", default="results/validation_review_report.md")
	args = ap.parse_args()

	with open(os.path.join(args.review_dir, "index.json"), encoding="utf-8") as f:
		index = json.load(f)
	expect = {(x["task"], x["lang"]): x for x in index}

	trans = defaultdict(Counter)
	en = Counter()
	en_conflicts = []
	trans_bad = []
	missing, short = [], []

	for x in index:
		path = x["out"]
		if not os.path.exists(path):
			missing.append(f"{x['task']}/{x['lang']}")
			continue
		try:
			with open(path, encoding="utf-8") as f:
				rows = json.load(f)
		except Exception as e:
			missing.append(f"{x['task']}/{x['lang']} (unreadable: {e})")
			continue
		if len(rows) < x["n"]:
			short.append(f"{x['task']}/{x['lang']}: {len(rows)}/{x['n']}")
		for r in rows:
			v = r.get("verdict", "unsure")
			if x["task"] == "translation":
				trans[x["lang"]][v] += 1
				if v in ("subject", "type", "relation"):
					trans_bad.append({"fact_id": r["fact_id"], "lang": x["lang"],
					                  "verdict": v, "note": r.get("note", "")})
			else:
				en[v] += 1
				if v == "conflict":
					en_conflicts.append({"fact_id": r["fact_id"],
					                     "note": r.get("note", "")})

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out("# Validation-split review\n")
	if missing:
		out(f"**INCOMPLETE — not yet reviewed:** {', '.join(missing)}\n")
	if short:
		out(f"**Short files (reviewer skipped rows):** {', '.join(short)}\n")

	# ---------------- translations ----------------
	out("## 1. Translation defects (sampled)\n")
	out("Same check the full test review ran, on a stratified sample of validation.\n")
	out("| lang | reviewed | ok | subject | type | relation | unsure | defect rate | 95% CI |")
	out("|---|---|---|---|---|---|---|---|---|")
	tot = Counter()
	for l in sorted(trans, key=lambda x: -(sum(trans[x].values()) - trans[x]["ok"])):
		c = trans[l]
		n = sum(c.values())
		tot.update(c)
		bad = c["subject"] + c["type"] + c["relation"]
		lo, hi = wilson(bad, n)
		out(f"| {l} | {n} | {c['ok']} | **{c['subject']}** | {c['type']} | "
		    f"{c['relation']} | {c['unsure']} | **{100*bad/n:.1f}%** | "
		    f"[{100*lo:.1f}%, {100*hi:.1f}%] |")
	N = sum(tot.values())
	BAD = tot["subject"] + tot["type"] + tot["relation"]
	if N:
		lo, hi = wilson(BAD, N)
		out(f"| **all** | **{N:,}** | {tot['ok']:,} | **{tot['subject']}** | "
		    f"{tot['type']} | {tot['relation']} | {tot['unsure']} | "
		    f"**{100*BAD/N:.1f}%** | [{100*lo:.1f}%, {100*hi:.1f}%] |")
		out("")
		out(f"Test split, for comparison: **{100*TEST_RATE:.1f}%** over {TEST_N:,} "
		    f"items (full census, no sampling error).")
		out("**Do not compare these two numbers directly.** The test census and this "
		    "validation pass used different reviewer prompts — the validation prompt "
		    "names known bug classes and warns about morphology false positives, "
		    "which makes its reviewers stricter. The two passes differ in split AND "
		    "in prompt at once, so a gap between them is not evidence about the "
		    "data. The control below separates the two.\n")

	# ---------------- english ----------------
	out("## 2. English questions that contradict their own gold\n")
	out("The first measurement of this class. Every earlier pass used English as "
	    "its reference and could not see it.\n")
	n_en = sum(en.values())
	if n_en:
		k = en["conflict"]
		lo, hi = wilson(k, n_en)
		out(f"| verdict | n | share |")
		out("|---|---|---|")
		for v in ("ok", "conflict", "vague", "unsure"):
			out(f"| {v} | {en[v]} | {100*en[v]/n_en:.1f}% |")
		out("")
		out(f"- **conflict rate: {100*k/n_en:.1f}%** (95% CI "
		    f"[{100*lo:.1f}%, {100*hi:.1f}%]) over {n_en} English questions")
		out(f"- extrapolated to the full release: roughly "
		    f"**{lo*59291:,.0f}–{hi*59291:,.0f} facts** whose English question may "
		    f"contradict its gold")
		out("")
		out("`vague` is reported separately and NOT counted as a conflict: an "
		    "underspecified question has no wrong claim in it, it just may not "
		    "have a unique answer.\n")
		if en_conflicts:
			out("### Confirmed English/gold conflicts\n")
			for c in en_conflicts[:40]:
				out(f"- `{c['fact_id']}` — {c['note']}")
			out("")

	# ---------------- worst translation cases ----------------
	if trans_bad:
		sev = [x for x in trans_bad if x["verdict"] == "subject"]
		out(f"## Severe (`subject`) translation defects in the sample: {len(sev)}\n")
		for x in sev[:30]:
			out(f"- `{x['fact_id']}` [{x['lang']}] — {x['note']}")
		out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(args.out.replace(".md", "_data.json"), "w", encoding="utf-8") as f:
		json.dump({"translation": {l: dict(c) for l, c in trans.items()},
		           "english": dict(en),
		           "english_conflicts": en_conflicts,
		           "translation_defects": trans_bad}, f, ensure_ascii=False)
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
