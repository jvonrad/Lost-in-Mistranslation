#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check every answer label in PolyFact-Clean against Wikidata, to separate labels
that came from Wikidata from labels the generation model invented.

PolyFact's construction used Gemma-3-27B to write the questions and, where the
Wikidata triple store had no label for an entity in a target language, to
TRANSLATE the label. Those translations have no ground truth behind them and are
the most likely place for silent errors — especially in the non-Latin-script
languages, where a wrong transliteration is invisible to a reader of the English
data.

Every label is sorted into one of:
  verified        exact match to the entity's Wikidata label in that language
  alias           exact match to one of its Wikidata aliases
  CONTRADICTS     Wikidata HAS a label in that language and it is different —
                  the generator overrode real data, the highest-severity class
  no_ground_truth Wikidata has no label in that language, so the string is an
                  unverifiable model translation
  unknown_entity  the entity itself returned nothing from the API

Because v5 samples distractors from the pool of gold entities, checking gold
labels covers every string that appears as an option anywhere.

CPU + Wikidata API; safe on a login node. Responses are cached to disk.
"""

import argparse
import json
import os
import time
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
NON_LATIN = ["ar", "bn", "ru", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
API = "https://www.wikidata.org/w/api.php"
UA = "PolyFact-label-provenance/0.1 (research; jonathan.vonrad@gmail.com)"


def fetch(repo, config, split, local_dir):
	if local_dir:
		return pq.read_table(os.path.join(local_dir, "data", config, f"{split}.parquet"))
	return pq.read_table(hf_hub_download(repo, f"data/{config}/{split}.parquet",
	                                     repo_type="dataset"))


def norm(s):
	return unicodedata.normalize("NFKC", s).strip().casefold()


def load_wikidata(qids, cache_path):
	cache = {}
	if os.path.exists(cache_path):
		with open(cache_path, encoding="utf-8") as f:
			cache = json.load(f)
	todo = [q for q in qids if q not in cache]
	print(f"{len(cache):,} entities cached, fetching {len(todo):,}")
	sess = requests.Session()
	sess.headers["User-Agent"] = UA
	for i in range(0, len(todo), 50):
		chunk = todo[i:i + 50]
		for attempt in range(5):
			try:
				r = sess.get(API, params={
					"action": "wbgetentities", "ids": "|".join(chunk),
					"props": "labels|aliases", "languages": "|".join(LANGS),
					"format": "json"}, timeout=30)
				if r.status_code == 429:
					time.sleep(float(r.headers.get("Retry-After", 5)))
					continue
				r.raise_for_status()
				data = r.json()
				for qid, ent in (data.get("entities") or {}).items():
					cache[qid] = {
						"labels": {l: v["value"] for l, v in (ent.get("labels") or {}).items()},
						"aliases": {l: [a["value"] for a in v]
						            for l, v in (ent.get("aliases") or {}).items()},
					}
				break
			except Exception as e:
				print(f"  retry {attempt}: {e}")
				time.sleep(2 ** attempt)
		for q in chunk:
			cache.setdefault(q, {"labels": {}, "aliases": {}})
		if (i // 50) % 20 == 0:
			print(f"  {min(i + 50, len(todo)):,}/{len(todo):,}", flush=True)
			with open(cache_path, "w", encoding="utf-8") as f:
				json.dump(cache, f, ensure_ascii=False)
	with open(cache_path, "w", encoding="utf-8") as f:
		json.dump(cache, f, ensure_ascii=False)
	return cache


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--local_dir", default=None)
	ap.add_argument("--cache", default=None)
	ap.add_argument("--out", default="results/polyfact_label_provenance.md")
	args = ap.parse_args()
	cache_path = args.cache or os.path.join(
		os.environ.get("SCRATCH", "."), "wikidata_label_cache.json")

	meta = {}
	for s in SPLITS:
		for r in fetch(args.repo, "parallel", s, args.local_dir).select(
				["fact_id", "object_id", "subject", "relation"]).to_pylist():
			meta[r["fact_id"]] = r
	# entity -> canonical label per language (one per entity, verified elsewhere)
	ent_label = defaultdict(dict)
	for l in LANGS:
		for s in SPLITS:
			t = fetch(args.repo, l, s, args.local_dir).select(["fact_id", "answer_text"])
			for f, g in zip(t.column("fact_id").to_pylist(), t.column("answer_text").to_pylist()):
				ent_label[meta[f]["object_id"]][l] = g
	print(f"{len(meta):,} facts, {len(ent_label):,} distinct answer entities")

	wd = load_wikidata(sorted(ent_label), cache_path)

	status = defaultdict(Counter)
	examples = defaultdict(list)
	entity_status = defaultdict(dict)
	for qid, per_lang in ent_label.items():
		w = wd.get(qid) or {"labels": {}, "aliases": {}}
		if not w["labels"] and not w["aliases"]:
			for l in LANGS:
				status[l]["unknown_entity"] += 1
				entity_status[qid][l] = "unknown_entity"
			continue
		for l, lab in per_lang.items():
			wlab = w["labels"].get(l)
			al = w["aliases"].get(l, [])
			if wlab and norm(lab) == norm(wlab):
				st = "verified"
			elif any(norm(lab) == norm(a) for a in al):
				st = "alias"
			elif wlab:
				st = "CONTRADICTS"
				if len(examples[l]) < 6:
					examples[l].append((qid, lab, wlab))
			else:
				st = "no_ground_truth"
			status[l][st] += 1
			entity_status[qid][l] = st

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out(f"# Label provenance vs Wikidata — `{args.repo}`\n")
	out(f"{len(ent_label):,} distinct answer entities × 12 languages. Because v5 draws "
	    f"distractors from the gold-entity pool, this covers every option string.\n")
	out("| lang | verified | alias | **CONTRADICTS** | no ground truth | unknown entity |")
	out("|---|---|---|---|---|---|")
	for l in LANGS:
		c = status[l]
		n = sum(c.values())
		out(f"| {l} | {100*c['verified']/n:.1f}% | {100*c['alias']/n:.1f}% | "
		    f"**{c['CONTRADICTS']:,} ({100*c['CONTRADICTS']/n:.1f}%)** | "
		    f"{100*c['no_ground_truth']/n:.1f}% | {100*c['unknown_entity']/n:.1f}% |")
	out("")
	out("`CONTRADICTS` = Wikidata has a label in that language and the dataset uses a "
	    "different string, i.e. the generator overrode real data. `no ground truth` = "
	    "Wikidata has no label in that language, so the string is an unverifiable "
	    "model translation.\n")

	for l in NON_LATIN:
		if examples[l]:
			out(f"**{l} — dataset label vs Wikidata label:**\n")
			for qid, lab, wlab in examples[l][:5]:
				out(f"- `{qid}`: dataset `{lab}` / Wikidata `{wlab}`")
			out("")

	# facts touched by a contradiction or an unverifiable label, per language
	out("## Fact-level exposure\n")
	out("| lang | facts whose gold CONTRADICTS Wikidata | facts whose gold has no ground truth |")
	out("|---|---|---|")
	for l in LANGS:
		con = sum(1 for f, m in meta.items()
		          if entity_status.get(m["object_id"], {}).get(l) == "CONTRADICTS")
		ngt = sum(1 for f, m in meta.items()
		          if entity_status.get(m["object_id"], {}).get(l) == "no_ground_truth")
		out(f"| {l} | {con:,} ({100*con/len(meta):.1f}%) | {ngt:,} ({100*ngt/len(meta):.1f}%) |")
	out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	with open(args.out.replace(".md", "_entities.json"), "w", encoding="utf-8") as f:
		json.dump(entity_status, f, ensure_ascii=False)
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
