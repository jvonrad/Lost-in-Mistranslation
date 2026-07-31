#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 of the PolyFact label repair: find broken gold labels and resolve the
correct string for each from Wikidata.

Two defect classes are targeted (both found by data_analysis/audit_transliteration.py):

  A. Wrong entity. The same `object_id` appears under two different gold strings
     within one language, and the minority string names a different entity
     entirely (e.g. ru `Q1321` "Spanish" rendered as "нюнорск"/Nynorsk).
  B. Transliteration corruption. A Latin-script fragment sits directly against
     native script with no separator (e.g. bn "লিঙ্কনNear-Earth গ্রহাণু গবেষণা",
     ar "ماtejكو" for Matejko). CJK is checked too but its counts are noisy,
     because an acronym prefixed to native script is conventional orthography.

For every affected (language, object_id) the canonical Wikidata label is fetched
and used as the replacement, which is the ground truth the dataset claims to be
built on. A replacement is only emitted when Wikidata actually has a label in that
language and it differs from the stored string.

Output: a JSON map {lang: {bad_string: good_string}} plus provenance, consumed by
apply_label_repairs.py.

CPU-only, single-threaded, ~20 Wikidata calls; safe on a login node.
"""

import argparse
import json
import os
import time
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import requests

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "PolyFact-label-repair/0.1 (research; jonathan.vonrad@gmail.com)"


def script_of(ch):
	if not ch.isalpha():
		return None
	try:
		return unicodedata.name(ch).split()[0]
	except ValueError:
		return None


def has_intraword_latin(text):
	"""Latin directly touching another script with no separating character."""
	prev = None
	for ch in text:
		s = script_of(ch)
		if s is None:
			prev = None
			continue
		if prev and prev != s and "LATIN" in (prev, s):
			return True
		prev = s
	return False


def fetch_labels(session, qids, langs):
	"""Batch wbgetentities label lookup; returns {qid: {lang: label}}."""
	out = {}
	qids = list(qids)
	for i in range(0, len(qids), 50):
		chunk = qids[i:i + 50]
		for attempt in range(5):
			try:
				r = session.get(WIKIDATA_API, params={
					"action": "wbgetentities", "ids": "|".join(chunk),
					"props": "labels", "languages": "|".join(langs), "format": "json",
				}, timeout=30)
				if r.status_code == 429:
					time.sleep(float(r.headers.get("Retry-After", 5)))
					continue
				r.raise_for_status()
				data = r.json()
				for qid, ent in (data.get("entities") or {}).items():
					out[qid] = {l: v["value"] for l, v in (ent.get("labels") or {}).items()}
				break
			except Exception as e:
				print(f"  [wikidata] {e}; retry {attempt}")
				time.sleep(2 ** attempt)
		time.sleep(0.2)
		print(f"  fetched {min(i + 50, len(qids))}/{len(qids)} entities", flush=True)
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True, help="built PolyFact-Clean tree")
	ap.add_argument("--out", required=True)
	args = ap.parse_args()

	# fact_id -> object_id
	obj_of = {}
	for s in SPLITS:
		t = pq.read_table(os.path.join(args.data_dir, "data", "parallel", f"{s}.parquet"),
		                  columns=["fact_id", "object_id"])
		obj_of.update(zip(t.column("fact_id").to_pylist(), t.column("object_id").to_pylist()))
	print(f"{len(obj_of):,} facts")

	# per language: entity -> Counter(gold strings); and the set of corrupt strings
	suspects = defaultdict(dict)     # lang -> bad_string -> object_id (if resolvable)
	reasons = defaultdict(dict)      # lang -> bad_string -> reason
	for lang in LANGS:
		by_entity = defaultdict(Counter)
		for s in SPLITS:
			t = pq.read_table(os.path.join(args.data_dir, "data", lang, f"{s}.parquet"),
			                  columns=["fact_id", "answer_text"])
			for fid, gold in zip(t.column("fact_id").to_pylist(),
			                     t.column("answer_text").to_pylist()):
				oid = obj_of.get(fid)
				if oid:
					by_entity[oid][gold] += 1
		for oid, counter in by_entity.items():
			modal = counter.most_common(1)[0][0]
			for label, n in counter.items():
				bad_translit = has_intraword_latin(label)
				wrong_entity = len(counter) > 1 and label != modal
				if bad_translit or wrong_entity:
					suspects[lang][label] = oid
					reasons[lang][label] = ("wrong_entity" if wrong_entity else "") + \
					                       ("+" if wrong_entity and bad_translit else "") + \
					                       ("transliteration" if bad_translit else "")
		print(f"  {lang}: {len(suspects[lang]):,} suspect labels")

	need = {oid for m in suspects.values() for oid in m.values()}
	print(f"\nfetching Wikidata labels for {len(need):,} entities")
	session = requests.Session()
	session.headers["User-Agent"] = USER_AGENT
	wd = fetch_labels(session, need, LANGS)

	repairs = defaultdict(dict)
	unresolved = defaultdict(list)
	for lang, mapping in suspects.items():
		for bad, oid in mapping.items():
			good = (wd.get(oid) or {}).get(lang)
			if good and good != bad:
				# object_id is essential: a bad string like "South America" is the
				# LEGITIMATE label of another entity, so the repair must be applied
				# only to facts whose object_id is the one that was mislabelled.
				repairs[lang][bad] = {"good": good, "object_id": oid,
				                      "reason": reasons[lang][bad]}
			else:
				unresolved[lang].append({"label": bad, "object_id": oid,
				                         "reason": reasons[lang][bad],
				                         "wikidata_label": good})

	total_fix = sum(len(v) for v in repairs.values())
	total_un = sum(len(v) for v in unresolved.values())
	print(f"\nrepairable: {total_fix:,} distinct labels; unresolved: {total_un:,}")
	for lang in LANGS:
		if repairs.get(lang) or unresolved.get(lang):
			print(f"  {lang}: fix {len(repairs.get(lang, {})):,}, "
			      f"unresolved {len(unresolved.get(lang, [])):,}")

	with open(args.out, "w", encoding="utf-8") as f:
		json.dump({"repairs": repairs, "unresolved": unresolved,
		           "reasons": {l: reasons[l] for l in reasons}}, f,
		          ensure_ascii=False, indent=1)
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
