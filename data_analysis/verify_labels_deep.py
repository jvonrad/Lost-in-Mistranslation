#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep label verification for PolyFact-Clean.

`audit_label_provenance.py` compared each answer label against the entity's
Wikidata label in that exact language code, and called anything else
`no_ground_truth` — an unverifiable model translation. That overstates the
problem badly, because Wikidata holds attested names for an entity in three
more places the audit never looked at:

  1. LANGUAGE VARIANTS. Chinese lives under `zh-hans` / `zh-hant` / `zh-cn` /
     `zh-tw` as well as `zh`, Portuguese under `pt-br`, German under `de-at` /
     `de-ch`. Asking only for `zh` makes a correct Simplified label look like it
     contradicts a Traditional one.
  2. WIKIPEDIA SITELINKS. An entity with no `bn` label field very often still
     has a bn.wikipedia.org article, and that article's title IS the attested
     Bengali name. This matters most for exactly the low-coverage languages
     (bn 78% / sw 82% "no ground truth") where the risk was thought worst.
  3. ALIASES in any of the above variants.

Each (entity, language) is scored against the union of those sources:

  verified            exact match to the language's own Wikidata label
  variant             exact match to a language-variant label (zh-hans, pt-br, …)
  sitelink            exact match to the language's Wikipedia article title
  alias               exact match to an alias in the language or a variant
  CONTRADICTS         ground truth exists in some form and the label matches none
  no_ground_truth     no attested name anywhere — a genuinely unverifiable
                      model translation, the only pool that can hide a
                      hallucination
  unknown_entity      the entity returned nothing from the API

The residual `no_ground_truth` set is written out as a sampling frame
(`--pool_out`) for manual review, since no automatic check can settle it.

CPU + Wikidata API; safe on a login node. Responses cached to disk.
"""

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
API = "https://www.wikidata.org/w/api.php"
UA = "PolyFact-label-verify/0.2 (research; jonathan.vonrad@gmail.com)"

# Wikidata stores regional/script variants under their own codes; a label in any
# of them is an attested name for the same language.
VARIANTS = {
	"en": ["en-gb", "en-ca"],
	"de": ["de-at", "de-ch"],
	"pt": ["pt-br"],
	"es": ["es-419"],
	"zh": ["zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk", "zh-sg", "zh-mo"],
}
WIKI = {l: f"{l}wiki" for l in LANGS}
ALL_CODES = LANGS + [v for vs in VARIANTS.values() for v in vs]

# "Douglas Adams (writer)" -> also accept "Douglas Adams"
PAREN = re.compile(r"\s*[\(（][^()（）]*[\)）]\s*$")

try:
	from hanziconv import HanziConv
	def s2t(s):
		# Traditional -> Simplified is many-to-one and therefore well defined;
		# the reverse is not (历 -> 歷 or 曆) and hanziconv guesses it wrong.
		return HanziConv.toSimplified(s)
except ImportError:                                   # optional
	def s2t(s):
		return s


def fetch(repo, config, split, local_dir):
	if local_dir:
		return pq.read_table(os.path.join(local_dir, "data", config, f"{split}.parquet"))
	return pq.read_table(hf_hub_download(repo, f"data/{config}/{split}.parquet",
	                                     repo_type="dataset"))


def norm(s, lang=None):
	s = unicodedata.normalize("NFKC", s).strip().casefold()
	if lang == "zh":                                  # Simplified/Traditional are the same name
		s = s2t(s)
	return s


def load_wikidata(qids, cache_path):
	cache = {}
	if os.path.exists(cache_path):
		with open(cache_path, encoding="utf-8") as f:
			cache = json.load(f)
	todo = [q for q in qids if q not in cache]
	print(f"{len(cache):,} entities cached, fetching {len(todo):,}", flush=True)
	sess = requests.Session()
	sess.headers["User-Agent"] = UA
	for i in range(0, len(todo), 50):
		chunk = todo[i:i + 50]
		for attempt in range(5):
			try:
				r = sess.get(API, params={
					"action": "wbgetentities", "ids": "|".join(chunk),
					"props": "labels|aliases|sitelinks",
					"languages": "|".join(ALL_CODES),
					"sitefilter": "|".join(WIKI.values()),
					"format": "json"}, timeout=45)
				if r.status_code == 429:
					time.sleep(float(r.headers.get("Retry-After", 5)))
					continue
				r.raise_for_status()
				for qid, ent in (r.json().get("entities") or {}).items():
					cache[qid] = {
						"labels": {l: v["value"]
						           for l, v in (ent.get("labels") or {}).items()},
						"aliases": {l: [a["value"] for a in v]
						            for l, v in (ent.get("aliases") or {}).items()},
						"sitelinks": {s: v["title"]
						              for s, v in (ent.get("sitelinks") or {}).items()},
					}
				break
			except Exception as e:
				print(f"  retry {attempt}: {e}", flush=True)
				time.sleep(2 ** attempt)
		for q in chunk:
			cache.setdefault(q, {"labels": {}, "aliases": {}, "sitelinks": {}})
		if (i // 50) % 20 == 0:
			print(f"  {min(i + 50, len(todo)):,}/{len(todo):,}", flush=True)
			with open(cache_path, "w", encoding="utf-8") as f:
				json.dump(cache, f, ensure_ascii=False)
	with open(cache_path, "w", encoding="utf-8") as f:
		json.dump(cache, f, ensure_ascii=False)
	return cache


def classify(label, lang, w):
	"""Strongest evidence supporting `label` as a name for this entity in `lang`."""
	n = norm(label, lang)
	own = w["labels"].get(lang)
	if own and norm(own, lang) == n:
		return "verified", None
	for v in VARIANTS.get(lang, []):
		vl = w["labels"].get(v)
		if vl and norm(vl, lang) == n:
			return "variant", None
	title = w["sitelinks"].get(WIKI[lang])
	if title and n in (norm(title, lang), norm(PAREN.sub("", title), lang)):
		return "sitelink", None
	for code in [lang] + VARIANTS.get(lang, []):
		if any(norm(a, lang) == n for a in w["aliases"].get(code, [])):
			return "alias", None
	truth = own or next((w["labels"].get(v) for v in VARIANTS.get(lang, [])
	                     if w["labels"].get(v)), None) or title
	if truth:
		return "CONTRADICTS", truth
	return "no_ground_truth", None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--local_dir", default=None)
	ap.add_argument("--cache", default=None)
	ap.add_argument("--out", default="results/polyfact_label_verification.md")
	ap.add_argument("--pool_out", default="results/polyfact_unverifiable_pool.json")
	args = ap.parse_args()
	cache_path = args.cache or os.path.join(
		os.environ.get("SCRATCH", "."), "wikidata_deep_cache.json")

	# the `parallel` config nests all 12 languages, so one pass covers everything
	meta = {}
	ent_label = defaultdict(dict)
	for s in SPLITS:
		for r in fetch(args.repo, "parallel", s, args.local_dir).select(
				["fact_id", "object_id", "subject", "relation", "translations"]).to_pylist():
			meta[r["fact_id"]] = {k: r[k] for k in
			                      ("fact_id", "object_id", "subject", "relation")}
			for l in LANGS:
				ent_label[r["object_id"]][l] = r["translations"][l]["answer_text"]
	print(f"{len(meta):,} facts, {len(ent_label):,} distinct answer entities", flush=True)

	wd = load_wikidata(sorted(ent_label), cache_path)

	status = defaultdict(Counter)
	entity_status = defaultdict(dict)
	contra = defaultdict(list)
	pool = []
	rel_of_entity = defaultdict(Counter)
	for m in meta.values():
		rel_of_entity[m["object_id"]][m["relation"]] += 1

	for qid, per_lang in ent_label.items():
		w = wd.get(qid) or {"labels": {}, "aliases": {}, "sitelinks": {}}
		if not any((w["labels"], w["aliases"], w["sitelinks"])):
			for l in LANGS:
				status[l]["unknown_entity"] += 1
				entity_status[qid][l] = "unknown_entity"
			continue
		for l, lab in per_lang.items():
			st, truth = classify(lab, l, w)
			status[l][st] += 1
			entity_status[qid][l] = st
			if st == "CONTRADICTS" and len(contra[l]) < 8:
				contra[l].append((qid, lab, truth))
			if st == "no_ground_truth":
				pool.append({
					"qid": qid, "lang": l, "label": lab,
					"en": per_lang.get("en", ""),
					"relation": rel_of_entity[qid].most_common(1)[0][0],
					"n_facts": sum(rel_of_entity[qid].values()),
				})

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	order = ["verified", "variant", "sitelink", "alias",
	         "CONTRADICTS", "no_ground_truth", "unknown_entity"]
	out(f"# Deep label verification — `{args.repo}`\n")
	out(f"{len(ent_label):,} distinct answer entities × 12 languages "
	    f"({12*len(ent_label):,} labels). Distractors are drawn from the gold-entity "
	    f"pool, so this covers every option string in the dataset.\n")
	out("Ground truth is the union of the Wikidata label, its language variants "
	    "(`zh-hans`, `pt-br`, …), the language's Wikipedia article title, and aliases.\n")
	out("| lang | verified | variant | sitelink | alias | **attested total** | "
	    "**CONTRADICTS** | no ground truth |")
	out("|---|---|---|---|---|---|---|---|")
	tot_att = tot_ngt = tot_con = 0
	for l in LANGS:
		c = status[l]
		n = sum(c.values())
		att = c["verified"] + c["variant"] + c["sitelink"] + c["alias"]
		tot_att += att
		tot_ngt += c["no_ground_truth"]
		tot_con += c["CONTRADICTS"]
		out(f"| {l} | {100*c['verified']/n:.1f}% | {100*c['variant']/n:.1f}% | "
		    f"{100*c['sitelink']/n:.1f}% | {100*c['alias']/n:.1f}% | "
		    f"**{100*att/n:.1f}%** | **{c['CONTRADICTS']:,} ({100*c['CONTRADICTS']/n:.1f}%)** | "
		    f"{100*c['no_ground_truth']/n:.1f}% |")
	N = 12 * len(ent_label)
	out(f"| **all** | | | | | **{100*tot_att/N:.1f}%** | **{tot_con:,} "
	    f"({100*tot_con/N:.1f}%)** | **{100*tot_ngt/N:.1f}%** |")
	out("")
	out(f"Only **{tot_ngt:,} / {N:,} ({100*tot_ngt/N:.1f}%)** labels have no attested "
	    f"name behind them; these are the only ones that could be a silent "
	    f"hallucination, and they are the sampling frame in `{args.pool_out}`.\n")

	for l in ["zh", "ar", "bn", "sw", "ja", "ru"]:
		if contra[l]:
			out(f"**{l} — remaining contradictions:**\n")
			for qid, lab, truth in contra[l][:5]:
				out(f"- `{qid}`: dataset `{lab}` / Wikidata `{truth}`")
			out("")

	out("## Unverifiable labels by relation\n")
	by_rel = Counter(p["relation"] for p in pool)
	out("| relation | unverifiable labels | distinct entities |")
	out("|---|---|---|")
	ents_by_rel = defaultdict(set)
	for p in pool:
		ents_by_rel[p["relation"]].add(p["qid"])
	for rel, n in by_rel.most_common(15):
		out(f"| {rel} | {n:,} | {len(ents_by_rel[rel]):,} |")
	out("")

	# how many facts have a fully attested gold in all 12 languages
	full = sum(1 for m in meta.values()
	           if all(entity_status.get(m["object_id"], {}).get(l) in
	                  ("verified", "variant", "sitelink", "alias") for l in LANGS))
	out("## Fact-level coverage\n")
	out(f"- facts whose gold label is attested in **all 12** languages: "
	    f"**{full:,} / {len(meta):,} ({100*full/len(meta):.1f}%)**")
	nver = Counter(sum(1 for l in LANGS
	                   if entity_status.get(m["object_id"], {}).get(l) in
	                   ("verified", "variant", "sitelink", "alias"))
	               for m in meta.values())
	out("\n| languages attested | facts |")
	out("|---|---|")
	for k in sorted(nver, reverse=True):
		out(f"| {k} | {nver[k]:,} |")
	out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(args.out.replace(".md", "_entities.json"), "w", encoding="utf-8") as f:
		json.dump(entity_status, f, ensure_ascii=False)
	with open(args.pool_out, "w", encoding="utf-8") as f:
		json.dump(pool, f, ensure_ascii=False)
	print(f"\nwrote {args.out} and {args.pool_out} ({len(pool):,} unverifiable labels)")


if __name__ == "__main__":
	main()
