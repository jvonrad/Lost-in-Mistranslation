#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turn the unverifiable-label pool from `verify_labels_deep.py` into a manual
review sample with a defensible weighting.

Labels that Wikidata cannot confirm are not uniformly risky. A label that is a
phonetic rendering of the entity's name ("ডগলাস অ্যাডামস" for Douglas Adams)
cannot be a semantic hallucination — at worst the transliteration is awkward,
and the item still identifies the right entity. The dangerous case is a label
that shares no sound with the original, because it is either a genuine
translated name ("Estados Unidos") or an invention that names a DIFFERENT thing
("Ular derik" = rattlesnake, for the Brazilian city Cascavel).

So each unverifiable label is romanised (`unidecode`) and scored against the
English label. Abjads lose their vowels in romanisation, so a consonant-skeleton
comparison runs alongside the plain ratio. The pool splits into:

  translit   high similarity — a phonetic rendering, low risk
  semantic   low similarity — a translated or invented name, the at-risk stratum

Both strata are sampled (the low-risk one to check that the split is honest),
stratum sizes are recorded, and the review file groups rows by entity so all 12
languages can be compared side by side — cross-language triangulation is the
strongest signal available when Wikidata is silent.

The resulting per-stratum error rates recombine into a pool-wide estimate via
`--score` once the reviewed file is filled in.
"""

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

from rapidfuzz import fuzz
from unidecode import unidecode

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
VOWELS = re.compile(r"[aeiou]")
NONALNUM = re.compile(r"[^a-z0-9]+")
TRANSLIT_CUTOFF = 60


def rom(s):
	return NONALNUM.sub(" ", unidecode(unicodedata.normalize("NFKC", s)).lower()).strip()


def skeleton(s):
	return VOWELS.sub("", s)


def similarity(label, en):
	"""Best evidence that `label` is a phonetic rendering of `en`."""
	a, b = rom(label), rom(en)
	if not a or not b:
		return 0.0
	scores = [fuzz.ratio(a, b), fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b)]
	sa, sb = skeleton(a), skeleton(b)
	if sa and sb:                                   # abjads drop vowels when romanised
		scores.append(fuzz.ratio(sa, sb))
	return max(scores)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pool", default="results/polyfact_unverifiable_pool.json")
	ap.add_argument("--out", default="results/unverifiable_sample.md")
	ap.add_argument("--stats_out", default="results/unverifiable_strata.json")
	ap.add_argument("--n_semantic", type=int, default=420)
	ap.add_argument("--n_translit", type=int, default=180)
	ap.add_argument("--seed", type=int, default=20260731)
	ap.add_argument("--cutoff", type=float, default=TRANSLIT_CUTOFF)
	args = ap.parse_args()

	with open(args.pool, encoding="utf-8") as f:
		pool = json.load(f)
	print(f"{len(pool):,} unverifiable labels")

	by_entity = defaultdict(dict)
	for p in pool:
		by_entity[p["qid"]][p["lang"]] = p

	for p in pool:
		p["sim"] = similarity(p["label"], p["en"])
		p["stratum"] = "translit" if p["sim"] >= args.cutoff else "semantic"

	strata = Counter(p["stratum"] for p in pool)
	print(f"strata: {dict(strata)}")

	# per-language and per-relation shape of the at-risk stratum
	sem = [p for p in pool if p["stratum"] == "semantic"]
	print("semantic by lang:", Counter(p["lang"] for p in sem).most_common())

	rng = random.Random(args.seed)
	sample = {}
	for st, n in (("semantic", args.n_semantic), ("translit", args.n_translit)):
		rows = [p for p in pool if p["stratum"] == st]
		sample[st] = rng.sample(rows, min(n, len(rows)))

	lines = ["# Unverifiable-label review sample\n",
	         f"Pool: {len(pool):,} labels with no attested Wikidata name "
	         f"({strata['semantic']:,} semantic / {strata['translit']:,} translit).",
	         f"Sampled {len(sample['semantic'])} semantic + {len(sample['translit'])} "
	         f"translit, seed {args.seed}.\n",
	         "`sim` = romanised similarity to the English label. Rows are grouped by "
	         "entity so the other languages' labels are visible for triangulation.\n"]

	for st in ("semantic", "translit"):
		lines.append(f"\n## Stratum: {st} ({len(sample[st])} rows)\n")
		lines.append("| # | qid | relation | english | lang | dataset label | romanised | sim | other langs |")
		lines.append("|---|---|---|---|---|---|---|---|---|")
		for i, p in enumerate(sample[st], 1):
			sibs = "; ".join(f"{l}={by_entity[p['qid']][l]['label']}"
			                 for l in LANGS if l in by_entity[p["qid"]] and l != p["lang"])
			lines.append(f"| {i} | {p['qid']} | {p['relation']} | {p['en']} | {p['lang']} | "
			             f"**{p['label']}** | {rom(p['label'])} | {p['sim']:.0f} | {sibs[:110]} |")
		lines.append("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(args.stats_out, "w", encoding="utf-8") as f:
		json.dump({
			"pool_size": len(pool),
			"strata_sizes": dict(strata),
			"cutoff": args.cutoff,
			"seed": args.seed,
			"sample": {st: [{k: p[k] for k in
			                 ("qid", "lang", "label", "en", "relation", "sim", "n_facts")}
			                for p in sample[st]] for st in sample},
			"by_lang": {st: dict(Counter(p["lang"] for p in pool if p["stratum"] == st))
			            for st in strata},
			"by_relation": {st: dict(Counter(p["relation"] for p in pool
			                                 if p["stratum"] == st))
			                for st in strata},
		}, f, ensure_ascii=False, indent=1)
	print(f"wrote {args.out} and {args.stats_out}")


if __name__ == "__main__":
	main()
