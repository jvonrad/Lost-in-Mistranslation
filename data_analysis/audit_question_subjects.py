#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Does each question actually ask about the fact's stored subject?

Every audit in this repo so far checked ANSWER LABELS. But PolyFact's questions
were generated per language independently by Gemma-3-27B, and an external review
found cases where the generated question asks about a different entity entirely:
`Q1361995` stores the French commune *Jessains*, and its Arabic question reads
"إلى أي قارة تنتمي جزر فوكلاند؟" — "which continent do the FALKLAND ISLANDS belong
to?" — with gold "Europe". The triple is right, the English question is right, and
the Arabic item is factually false. Nothing in the label-side pipeline can see
this.

Matching the question against a romanisation of the ENGLISH subject does not work:
questions legitimately translate common-noun subjects (`instant film` -> 即时成像胶片,
`Palazzo Vecchio` -> قصر فيكيو), so that test flags 53% of Chinese as suspect while
almost all of it is correct. The only reliable reference is what Wikidata itself
calls the subject in that language.

So: fetch the subject's Wikidata label / aliases / Wikipedia title per language,
and for the subset where Wikidata HAS a name, check whether the question contains
it. That subset is large and unbiased, so its miss rate estimates the rate for the
whole release. Facts where the subject has no attested name in a language are
reported separately as uncheckable rather than counted either way.

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

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
API = "https://www.wikidata.org/w/api.php"
UA = "PolyFact-question-subject-audit/0.1 (research; jonathan.vonrad@gmail.com)"
VARIANTS = {
	"en": ["en-gb", "en-ca"], "de": ["de-at", "de-ch"], "pt": ["pt-br"],
	"es": ["es-419"],
	"zh": ["zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk", "zh-sg", "zh-mo"],
}
WIKI = {l: f"{l}wiki" for l in LANGS}
ALL_CODES = LANGS + [v for vs in VARIANTS.values() for v in vs]
PAREN = re.compile(r"\s*[\(（][^()（）]*[\)）]\s*$")

try:
	from hanziconv import HanziConv
	def fold_han(s):
		return HanziConv.toSimplified(s)
except ImportError:
	def fold_han(s):
		return s


def norm(s, lang=None):
	s = unicodedata.normalize("NFKC", s).strip().casefold()
	return fold_han(s) if lang == "zh" else s


def load_wikidata(qids, cache_path):
	cache = {}
	if os.path.exists(cache_path):
		with open(cache_path, encoding="utf-8") as f:
			cache = json.load(f)
	todo = [q for q in qids if q not in cache]
	print(f"{len(cache):,} subjects cached, fetching {len(todo):,}", flush=True)
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
					"sitefilter": "|".join(WIKI.values()), "format": "json"}, timeout=45)
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
		if (i // 50) % 10 == 0:   # checkpoint often: a kill mid-fetch loses only this much
			print(f"  {min(i + 50, len(todo)):,}/{len(todo):,}", flush=True)
			with open(cache_path, "w", encoding="utf-8") as f:
				json.dump(cache, f, ensure_ascii=False)
	with open(cache_path, "w", encoding="utf-8") as f:
		json.dump(cache, f, ensure_ascii=False)
	return cache


def names_for(w, lang):
	"""Every string Wikidata attests as this entity's name in `lang`."""
	out = []
	for code in [lang] + VARIANTS.get(lang, []):
		if w["labels"].get(code):
			out.append(w["labels"][code])
		out.extend(w["aliases"].get(code, []))
	t = w["sitelinks"].get(WIKI[lang])
	if t:
		out += [t, PAREN.sub("", t)]
	return [x for x in out if x and len(x) >= 3]


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--cache", default=None)
	ap.add_argument("--out", default="results/question_subject_audit.md")
	ap.add_argument("--flags_out", default="results/question_subject_flags.json")
	args = ap.parse_args()
	cache_path = args.cache or os.path.join(
		os.environ.get("SCRATCH", "."), "wikidata_subject_cache.json")

	def src(s):
		return os.path.join(args.data_dir, "data", "parallel", f"{s}.parquet")

	subs = set()
	for s in SPLITS:
		for b in pq.ParquetFile(src(s)).iter_batches(batch_size=4000,
		                                             columns=["subject_id"]):
			subs.update(b.column("subject_id").to_pylist())
	print(f"{len(subs):,} distinct subjects", flush=True)
	wd = load_wikidata(sorted(subs), cache_path)

	checked = Counter()
	missed = Counter()
	uncheckable = Counter()
	flags = []
	for s in SPLITS:
		for b in pq.ParquetFile(src(s)).iter_batches(batch_size=2000):
			for r in b.to_pylist():
				w = wd.get(r["subject_id"]) or {"labels": {}, "aliases": {},
				                                "sitelinks": {}}
				for l in LANGS:
					names = names_for(w, l)
					if not names:
						uncheckable[l] += 1
						continue
					checked[l] += 1
					q = norm(r["translations"][l]["question"], l)
					if not any(norm(n, l) in q for n in names):
						missed[l] += 1
						if len(flags) < 4000:
							flags.append({
								"fact_id": r["fact_id"], "lang": l,
								"subject": r["subject"],
								"wikidata_names": names[:3],
								"question": r["translations"][l]["question"],
								"relation": r["relation"]})
			del b

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out(f"# Does each question name its stored subject? — `{args.data_dir}`\n")
	out("The subject's Wikidata name in the target language is the reference; a "
	    "question that contains none of its attested names (label, aliases, "
	    "Wikipedia title, language variants) is not asking about the stored "
	    "entity. Facts whose subject has no attested name in a language cannot be "
	    "checked and are excluded from the rate rather than counted as either.\n")
	out("| lang | checkable | question omits the subject | rate | uncheckable |")
	out("|---|---|---|---|---|")
	for l in LANGS:
		c = checked[l]
		out(f"| {l} | {c:,} | {missed[l]:,} | **{100*missed[l]/c:.1f}%** | "
		    f"{uncheckable[l]:,} |" if c else f"| {l} | 0 | — | — | {uncheckable[l]:,} |")
	out("")
	tot_c, tot_m = sum(checked.values()), sum(missed.values())
	out(f"Overall **{tot_m:,} / {tot_c:,} ({100*tot_m/tot_c:.1f}%)** checkable "
	    f"(fact, language) items have a question that never names the stored "
	    f"subject.\n")
	out("A miss is not automatically an error: the generator may have used a name "
	    "Wikidata does not list. The flagged items are written to the JSON for "
	    "manual review, which is what settles the actual rate.\n")

	by_rel = Counter(f["relation"] for f in flags)
	out("## Flagged items by relation (sample)\n")
	out("| relation | flagged |")
	out("|---|---|")
	for rel, n in by_rel.most_common(10):
		out(f"| {rel} | {n:,} |")
	out("")
	out("## Examples\n")
	for f in flags[:25]:
		out(f"- [{f['lang']}] subject `{f['subject']}` "
		    f"(wikidata: {', '.join(f['wikidata_names'][:2])})\n"
		    f"  - Q: {f['question'][:110]}")
	out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(args.flags_out, "w", encoding="utf-8") as f:
		json.dump(flags, f, ensure_ascii=False)
	print(f"\nwrote {args.out} and {args.flags_out}")


if __name__ == "__main__":
	main()
