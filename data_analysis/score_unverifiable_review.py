#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score the manual review of `results/unverifiable_sample.md` and extrapolate to
the whole unverifiable pool.

Every row of the sample was judged by hand against the English label and the
same entity's labels in the other 11 languages. Judgements are recorded here as
row indices so the estimate is reproducible and auditable — each index is listed
with the entity and what is wrong with it.

  tier A  the label denotes something OTHER than the entity, or is not a word.
          A model answering the question correctly would be marked wrong, and a
          model reading the option list is shown a false fact.
  tier B  the label identifies the right entity but imprecisely (an institution's
          type is wrong, a modifier is dropped, a transliteration is off by a
          letter). Recoverable; not a factual error.

The two strata are extrapolated separately at their own rates and recombined by
size, and every count is re-weighted by `n_facts` — the number of facts each
entity actually serves as the gold answer for — because the errors cluster in
obscure entities that appear in very few items. An unweighted label rate would
overstate the damage to the benchmark.
"""

import argparse
import json
from collections import Counter, defaultdict

# ---- manual judgements: sample row index -> (entity, what is wrong) ----------
# Stratum "semantic", 420 rows reviewed.
SEMANTIC_A = {
	20:  ("Q245456 5th Cell", "ar 'الخامسة خلية' — studio name translated literally, ungrammatical"),
	29:  ("Q722976 Antonov plant", "sw 'Kiwi ya…' — 'Kiwanda' (factory) garbled to 'Kiwi'"),
	64:  ("Q2065466 Terminal Reality", "zh '终端现实' — studio name translated literally"),
	89:  ("Q601299 Video System", "sw 'Mfumo wa Video' — studio name translated literally"),
	96:  ("Q1616636 Hewett Watson", "zh '休·沃森' = Hugh Watson — wrong given name"),
	98:  ("Q2001088 Northern State University", "ar = 'North Dakota State University' — different institution"),
	111: ("Q659918 Horn (place)", "sw 'Pembe' = an animal's horn"),
	# NB row 113 (Q739552 Square) is NOT counted: the sampled label was bn
	# 'স্কোয়ার', a correct transliteration. Its sw sibling 'Mraba' (the shape)
	# is wrong, but scoring an unsampled sibling would bias the rate upward.
	123: ("Q11325179 Nude Maker", "sw 'Mtengenezaji wa Uchi' = maker of nakedness"),
	138: ("Q3003137 Croc", "zh '鳄鱼' = crocodile"),
	154: ("Q252733 Object Management Group", "sw 'Vituko' = antics, not objects"),
	178: ("Q3026228 Propaganda Games", "zh '宣传游戏' — studio name translated literally"),
	184: ("Q1778277 Odense Steel Shipyard", "sw 'Umelodi wa Chuma' — not a Swahili word for shipyard"),
	211: ("Q372608 University of Basel", "bn 'বার্ল' = Barl — misspelt city"),
	212: ("Q3569449 Wisdom Tree", "ru 'Древо мудрости' — studio name translated literally"),
	236: ("Q3064033 Łucznik Arms Factory", "sw 'Kiwiya' — 'Kiwanda' garbled again"),
	264: ("Q3851105 Silver (person)", "bn 'রূপা' = the metal silver"),
	284: ("Q778568 Asam brothers", "ja '浅井兄弟' = Asai brothers, a Japanese surname"),
	293: ("Q245456 5th Cell", "zh '第五单元游戏公司' — name translated + 'game company' appended"),
	309: ("Q697289 National University of Tainan", "ar = 'National University Taiwan' — different institution"),
	321: ("Q464476 Black Flag (band)", "sw 'Bendera Nyeusi' = a black flag"),
	362: ("Q3340630 Nicolas Nicole", "zh '尼古拉斯·尼古拉斯' = Nicolas Nicolas — surname replaced"),
	364: ("Q2994578 Conservatory of Nice", "sw 'Hifadhi' = a nature reserve"),
	367: ("Q2745586 Genius Sonority", "zh '天才音速' = genius sound-speed"),
	401: ("Q4348557 Microcabin", "zh '微仓' = micro warehouse"),
	416: ("Q13117583 Star Theory Games", "zh '星理论游戏' — studio name translated literally"),
}
SEMANTIC_B = {
	40:  ("Q1165635 Lowell Observatory", "sw 'Chuo cha Utafiti' = research college"),
	45:  ("Q1329478 Czech Technical University", "bn drops 'Czech' -> 'Prague Technical University'"),
	76:  ("Q65560029 Jacques Delrue", "ru 'Делюр' — drops the r"),
	109: ("Q421739 Nat. Univ. of Distance Education", "id 'Terbuka' = open, not distance"),
	143: ("Q511291 Royal Academy of Art", "sw 'Chuo Kikuu' = university, not academy"),
	162: ("Q2822452 Royal Academy of Arts of Liège", "sw 'Chuo Kikuu' = university"),
	272: ("Q4666924 Aberdeen Grammar School", "ar/id 'grammatical school'"),
	330: ("Q265058 Hungarian Academy of Sciences", "sw 'Chuo Kikuu' = university"),
	336: ("Q9391434 United Kingdom of Poland", "zh drops 'United'"),
	369: ("Q7571404 Southwestern College", "sw name translated + college->university"),
	384: ("Q216941 Xi River", "bn 'সি' = Si"),
}
# Stratum "translit", 180 rows reviewed. No tier-A errors were found.
TRANSLIT_A = {}
TRANSLIT_B = {
	167: ("Q2660091 Griptonite Games", "bn 'Gripstone Games'"),
	179: ("Q61793216 Triband", "es 'Tribanda'"),
}
JUDGED = {"semantic": (SEMANTIC_A, SEMANTIC_B), "translit": (TRANSLIT_A, TRANSLIT_B)}


def wilson(k, n, z=1.96):
	"""Wilson score interval — behaves sanely at k=0, unlike the normal approx."""
	if n == 0:
		return (0.0, 0.0)
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
	return (max(0.0, c - h), min(1.0, c + h))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--strata", default="results/unverifiable_strata.json")
	ap.add_argument("--pool", default="results/polyfact_unverifiable_pool.json")
	ap.add_argument("--out", default="results/unverifiable_review_report.md")
	ap.add_argument("--n_facts_total", type=int, default=60169)
	ap.add_argument("--n_labels_total", type=int, default=193188)
	args = ap.parse_args()

	st = json.load(open(args.strata, encoding="utf-8"))
	pool = json.load(open(args.pool, encoding="utf-8"))
	sizes = st["strata_sizes"]

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out("# Hallucination rate in PolyFact-Clean's unverifiable labels\n")
	out(f"Wikidata attests {args.n_labels_total - sizes['semantic'] - sizes['translit']:,} "
	    f"of the {args.n_labels_total:,} answer labels via its label, a language variant, "
	    f"a Wikipedia sitelink or an alias. The remaining "
	    f"**{sizes['semantic'] + sizes['translit']:,}** are unverifiable model translations "
	    f"and are the only place a hallucination can hide. 600 of them were reviewed by "
	    f"hand.\n")

	# ---- per-stratum rates -------------------------------------------------
	out("## Measured error rates\n")
	out("| stratum | pool | reviewed | tier A (wrong referent) | rate | 95% CI | tier B (imprecise) |")
	out("|---|---|---|---|---|---|---|")
	rates = {}
	for s in ("semantic", "translit"):
		A, B = JUDGED[s]
		n = len(st["sample"][s])
		lo, hi = wilson(len(A), n)
		rates[s] = (len(A) / n, lo, hi)
		out(f"| {s} | {sizes[s]:,} | {n} | {len(A)} | **{100*len(A)/n:.2f}%** | "
		    f"[{100*lo:.2f}%, {100*hi:.2f}%] | {len(B)} ({100*len(B)/n:.1f}%) |")
	out("")
	out("Tier A = the label denotes something other than the entity, or is not a word. "
	    "Tier B = right entity, imprecise wording. Only tier A can make a correct model "
	    "answer score as wrong.\n")

	# ---- label-level extrapolation ----------------------------------------
	N = sizes["semantic"] + sizes["translit"]
	pt = sum(sizes[s] * rates[s][0] for s in rates)
	lo = sum(sizes[s] * rates[s][1] for s in rates)
	hi = sum(sizes[s] * rates[s][2] for s in rates)
	out("## Extrapolated to the whole dataset\n")
	out(f"- bad labels in the unverifiable pool: **{pt:,.0f}** of {N:,} "
	    f"(**{100*pt/N:.2f}%**), 95% CI [{lo:,.0f}, {hi:,.0f}]")
	out(f"- as a share of ALL {args.n_labels_total:,} answer labels: "
	    f"**{100*pt/args.n_labels_total:.2f}%**, 95% CI "
	    f"[{100*lo/args.n_labels_total:.2f}%, {100*hi/args.n_labels_total:.2f}%]")
	out("")

	# ---- fact-weighted exposure -------------------------------------------
	# errors cluster in obscure entities, so weight by how many facts each serves
	sample_nf = {s: [r["n_facts"] for r in st["sample"][s]] for s in ("semantic", "translit")}
	err_nf = {}
	for s in ("semantic", "translit"):
		A, _ = JUDGED[s]
		err_nf[s] = [st["sample"][s][i - 1]["n_facts"] for i in A]
	mean_all = {s: sum(sample_nf[s]) / len(sample_nf[s]) for s in sample_nf}
	mean_err = {s: (sum(err_nf[s]) / len(err_nf[s]) if err_nf[s] else 0.0)
	            for s in err_nf}

	out("## Fact-level exposure\n")
	out("| stratum | mean facts/entity, all sampled | mean facts/entity, tier-A errors |")
	out("|---|---|---|")
	for s in ("semantic", "translit"):
		e = f"{mean_err[s]:.2f}" if err_nf[s] else "—"
		out(f"| {s} | {mean_all[s]:.2f} | {e} |")
	out("")

	# expected (fact, language) items carrying a wrong gold label
	items = sum(sizes[s] * rates[s][0] * mean_err[s] for s in rates)
	items_hi = sum(sizes[s] * rates[s][2] * (mean_err[s] or mean_all[s]) for s in rates)
	total_items = args.n_facts_total * 12
	out(f"- expected (fact, language) items whose gold label is wrong: "
	    f"**{items:,.0f}** of {total_items:,} (**{100*items/total_items:.3f}%**), "
	    f"95% upper bound {items_hi:,.0f} ({100*items_hi/total_items:.3f}%)")
	out(f"- i.e. roughly **{items/12:,.0f} facts' worth** of damage spread across "
	    f"12 languages; no fact is wrong in all 12, since the English label is "
	    f"attested for 99.6% of entities")
	out("")

	# ---- where the errors are ---------------------------------------------
	out("## Where the errors are\n")
	langs = Counter()
	rels = Counter()
	for s in ("semantic", "translit"):
		A, _ = JUDGED[s]
		for i in A:
			r = st["sample"][s][i - 1]
			langs[r["lang"]] += 1
			rels[r["relation"]] += 1
	out("| language | tier-A errors found | | relation | tier-A errors found |")
	out("|---|---|---|---|---|")
	lr = list(langs.most_common())
	rr = list(rels.most_common())
	for i in range(max(len(lr), len(rr))):
		a = f"{lr[i][0]} | {lr[i][1]}" if i < len(lr) else " | "
		b = f"{rr[i][0]} | {rr[i][1]}" if i < len(rr) else " | "
		out(f"| {a} | | {b} |")
	out("")

	out("## Every tier-A error found\n")
	for s in ("semantic", "translit"):
		A, _ = JUDGED[s]
		for i, (ent, why) in sorted(A.items()):
			r = st["sample"][s][i - 1]
			out(f"- **{ent}** [{r['lang']}, {r['relation']}, {r['n_facts']} facts] — {why}")
	out("")
	out("## Tier-B (imprecise but identifiable)\n")
	for s in ("semantic", "translit"):
		_, B = JUDGED[s]
		for i, (ent, why) in sorted(B.items()):
			r = st["sample"][s][i - 1]
			out(f"- {ent} [{r['lang']}] — {why}")
	out("")

	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
