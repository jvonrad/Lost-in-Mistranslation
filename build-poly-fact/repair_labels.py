#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5: repair answer labels instead of deleting the facts that carry them.

`data_analysis/verify_labels_deep.py` checked every answer label against the
union of Wikidata's label, its language variants, the language's Wikipedia
article title and its aliases. 70.6% are attested outright; a hand review of 600
of the remaining 55,903 (`score_unverifiable_review.py`) put the rate of labels
that actually denote the WRONG THING at 0.12% of items. Discarding the 30,577
facts that are not attested in all 12 languages would therefore cost half the
dataset to remove an error already an order of magnitude below the noise floor —
and would strip the long tail preferentially, since obscure entities are exactly
the ones Wikidata has no label for.

So this stage changes only what can be justified item by item:

  1. Swahili `Kiwanda` (factory) garbled to `Kiwi`/`Kiwiya`, a systematic
     generation defect surviving in 28 labels ("Kiwi ya Traktori ya
     Chelyabinsk"). Mechanical and unambiguous, so it is applied by rule.

  2. Labels where transliteration stopped mid-word and left a Latin remnant fused
     to the native script (`টাডেউশ তোলভিński`, `গ্দাńsk`, `লিège`). These have no
     safe automatic repair, so their entities are dropped. Detection requires a
     LATIN character adjacent to a non-Latin one with the Latin run in lowercase:
     uppercase runs are genuine acronyms and must survive (`AGヴェーザー`,
     `Access游戏`, `SOM建筑设计事务所`), the Japanese prolongation mark `ー` is not
     Latin despite what an "is it ASCII" test concludes, and Indic vowel signs are
     combining marks rather than letters, so a run-builder that splits on
     non-alphabetic characters silently misses every Bengali case.

  3. The entities whose labels the manual review of `unverifiable_sample.md`
     confirmed denote the WRONG THING (`zh 鳄鱼` = crocodile for the game Croc,
     `sw Hifadhi ya Nice` = a nature reserve for the Conservatory of Nice). These
     are dropped rather than rewritten: a correct replacement would mean inventing
     an Arabic or Japanese transliteration, which is how new errors get made.

  4. `n_langs_verified` (0-12) is attached to every row, so a reader can restrict
     to the fully-attested subset without the release making that choice for them.

NOT done, deliberately: overwriting the 608 labels that differ from Wikidata's.
Inspection showed that set is dominated by valid alternative names rather than
errors — `pt` has NASA's expansion where Wikidata has the acronym, `sw` has CERN
where Wikidata has the Swahili expansion, `en` has Alludo where Wikidata still
says Corel, and most of the rest differ only by a hyphen or an accent. Applying
Wikidata wholesale would have replaced "Skidmore, Owings & Merrill" with "SOM".
A label differing from Wikidata's is not evidence that it is wrong.

Reads only the `parallel` config and regenerates the 12 per-language configs from
it, so the two can never disagree. Output feeds `resample_distractors.py`.

CPU-only, no network (uses the cache written by verify_labels_deep.py); safe on
a login node.
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
NON_LATIN_LANGS = {"ar", "bn", "ru", "ja", "zh"}
ATTESTED = ("verified", "variant", "sitelink", "alias")

VARIANTS = {
	"en": ["en-gb", "en-ca"], "de": ["de-at", "de-ch"], "pt": ["pt-br"],
	"es": ["es-419"],
	"zh": ["zh-hans", "zh-hant", "zh-cn", "zh-tw", "zh-hk", "zh-sg", "zh-mo"],
}
WIKI = {l: f"{l}wiki" for l in LANGS}
PAREN = re.compile(r"\s*[\(（][^()（）]*[\)）]\s*$")
# `kiwanda` is noun class 7, so its concord is `cha`; the garbled forms carry the
# wrong concord too ("Kiwi ya Traktori"), and fixing only the noun leaves it wrong.
KIWANDA = [(re.compile(r"\bKiwi(?:ya)?\s+ya\b"), "Kiwanda cha"),
           (re.compile(r"\bKiwi(?:ya)?\b"), "Kiwanda")]

# Entities whose label the manual review confirmed denotes something other than
# the entity. Dropped, not rewritten — see the module docstring. Each is listed
# with the language it failed in and what the label actually means.
CONFIRMED_WRONG = {
	"Q2065466":  "zh 终端现实 — Terminal Reality, studio name translated",
	"Q601299":   "sw Mfumo wa Video — Video System, studio name translated",
	"Q1616636":  "zh 休·沃森 = Hugh Watson, not Hewett Watson",
	"Q2001088":  "ar = North Dakota State University, not Northern State",
	"Q659918":   "sw Pembe = an animal's horn; Horn is a town",
	"Q11325179": "sw Mtengenezaji wa Uchi = maker of nakedness (Nude Maker)",
	"Q3003137":  "zh 鳄鱼 = crocodile; Croc is a game",
	"Q252733":   "sw Vituko = antics (Object Management Group)",
	"Q3026228":  "zh 宣传游戏 — Propaganda Games, studio name translated",
	"Q1778277":  "sw Umelodi wa Chuma — not a Swahili word for shipyard",
	"Q372608":   "bn বার্ল = Barl, not Basel",
	"Q3569449":  "ru Древо мудрости — Wisdom Tree, studio name translated",
	"Q3851105":  "bn রূপা = the metal silver; Silver is a person",
	"Q778568":   "ja 浅井兄弟 = Asai brothers, a Japanese surname (Asam)",
	"Q697289":   "ar = National University Taiwan, not of Tainan",
	"Q464476":   "sw Bendera Nyeusi = a black flag; Black Flag is a band",
	"Q3340630":  "zh 尼古拉斯·尼古拉斯 = Nicolas Nicolas (Nicolas Nicole)",
	"Q2994578":  "sw Hifadhi ya Nice = a nature reserve, not a conservatory",
	"Q2745586":  "zh 天才音速 = genius sound-speed (Genius Sonority)",
	"Q4348557":  "zh 微仓 = micro warehouse (Microcabin)",
	"Q13117583": "zh 星理论游戏 — Star Theory Games, studio name translated",
	"Q245456":   "ar/zh — 5th Cell, studio name translated in two languages",
	"Q941731":   "zh 小泳者 = little swimmer, for Evergreen State College",
	"Q2670031":  "zh 顶点集团 = Apex Group, for Climax Studios",
}

try:
	from hanziconv import HanziConv
	def s2t(s):
		# Traditional -> Simplified is many-to-one and therefore well defined;
		# the reverse is not (历 -> 歷 or 曆) and hanziconv guesses it wrong.
		return HanziConv.toSimplified(s)
except ImportError:
	def s2t(s):
		return s


def norm(s, lang=None):
	s = unicodedata.normalize("NFKC", s).strip().casefold()
	return s2t(s) if lang == "zh" else s


def _script(ch):
	try:
		return unicodedata.name(ch).split()[0]
	except ValueError:
		return ""


def severed(s):
	"""True if transliteration stopped mid-word, leaving a lowercase Latin remnant
	fused to native script. Uppercase runs are acronyms and are not a defect."""
	for tok in s.split():
		runs = []                                  # (is_latin, text) for each run
		for ch in tok:
			# Indic vowel signs and other combining marks are not `isalpha`, but
			# they belong to the script of the character they attach to; treating
			# them as separators hides exactly the splices we are looking for.
			if unicodedata.category(ch).startswith("M"):
				if runs:
					runs[-1] = (runs[-1][0], runs[-1][1] + ch)
				continue
			if not ch.isalpha():
				runs.append((None, ch))
				continue
			lat = _script(ch) == "LATIN"
			if runs and runs[-1][0] == lat:
				runs[-1] = (lat, runs[-1][1] + ch)
			else:
				runs.append((lat, ch))
		for i, (lat, text) in enumerate(runs):
			if lat is not True or not text.islower():
				continue
			nbr = [runs[j][0] for j in (i - 1, i + 1) if 0 <= j < len(runs)]
			if False in nbr:                       # touches a non-Latin script run
				return True
	return False


def truth_for(lang, w):
	"""Wikidata's own name for the entity in `lang`, by evidence strength."""
	if w["labels"].get(lang):
		return w["labels"][lang]
	for v in VARIANTS.get(lang, []):
		if w["labels"].get(v):
			return w["labels"][v]
	t = w["sitelinks"].get(WIKI[lang])
	return PAREN.sub("", t) if t else None


def attested(label, lang, w):
	n = norm(label, lang)
	for code in [lang] + VARIANTS.get(lang, []):
		if w["labels"].get(code) and norm(w["labels"][code], lang) == n:
			return True
		if any(norm(a, lang) == n for a in w["aliases"].get(code, [])):
			return True
	t = w["sitelinks"].get(WIKI[lang])
	return bool(t) and n in (norm(t, lang), norm(PAREN.sub("", t), lang))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", required=True)
	ap.add_argument("--out_dir", required=True)
	ap.add_argument("--cache", default=None)
	ap.add_argument("--drop_out", default=None,
	                help="where to write the entity drop list for resample_distractors")
	args = ap.parse_args()
	cache_path = args.cache or os.path.join(
		os.environ.get("SCRATCH", "."), "wikidata_deep_cache.json")
	drop_out = args.drop_out or os.path.join(args.out_dir, "severed_entities.json")

	with open(cache_path, encoding="utf-8") as f:
		wd = json.load(f)

	# Only the entity->label map is held in memory; the fact rows are streamed
	# twice instead, because materialising 60k nested rows plus 13 output copies
	# of them is enough to get the process OOM-killed on a login node.
	def src(split):
		return os.path.join(args.data_dir, "data", "parallel", f"{split}.parquet")

	ent_label = defaultdict(dict)
	n_facts = 0
	for s in SPLITS:
		pf = pq.ParquetFile(src(s))
		for batch in pf.iter_batches(batch_size=2000,
		                             columns=["object_id", "translations"]):
			for r in batch.to_pylist():
				n_facts += 1
				for l in LANGS:
					ent_label[r["object_id"]][l] = r["translations"][l]["answer_text"]
			del batch
	print(f"{n_facts:,} facts, {len(ent_label):,} answer entities")

	# ---------- 1: the one repair that is mechanical and unambiguous ----------
	repairs = {}                                   # qid -> lang -> new label
	log = {"kiwanda": []}
	for qid, per_lang in ent_label.items():
		sw = per_lang["sw"]
		fixed = sw
		for pat, sub in KIWANDA:
			fixed = pat.sub(sub, fixed)
		if fixed != sw:
			repairs[qid] = {"sw": fixed}
			log["kiwanda"].append({"qid": qid, "was": sw, "now": fixed})
	n_rep = sum(len(v) for v in repairs.values())
	print(f"repairs: {n_rep:,} Swahili Kiwanda labels")

	def label(qid, l):
		return repairs.get(qid, {}).get(l, ent_label[qid][l])

	# ---------- 2 + 3: entities that cannot be repaired, only dropped ----------
	sev = {qid for qid in ent_label for l in NON_LATIN_LANGS
	       if severed(label(qid, l))}
	confirmed = {q for q in CONFIRMED_WRONG if q in ent_label}
	drop_set = sev | confirmed
	drop_facts = Counter()
	for s in SPLITS:
		for batch in pq.ParquetFile(src(s)).iter_batches(batch_size=4000,
		                                                 columns=["object_id"]):
			for q in batch.column("object_id").to_pylist():
				if q in drop_set:
					drop_facts["severed" if q in sev else "confirmed_wrong"] += 1
	n_drop = sum(drop_facts.values())
	print(f"dropping {len(drop_set):,} entities / {n_drop:,} facts "
	      f"({100*n_drop/n_facts:.2f}%): {dict(drop_facts)}")
	if len(confirmed) < len(CONFIRMED_WRONG):
		print(f"  note: {len(CONFIRMED_WRONG) - len(confirmed)} confirmed-wrong "
		      f"entities are already absent from this build")

	# ---------- 4: verification count after repair ----------
	nver = {}
	for qid in ent_label:
		w = wd.get(qid) or {"labels": {}, "aliases": {}, "sitelinks": {}}
		nver[qid] = sum(1 for l in LANGS if attested(label(qid, l), l, w))
	print("n_langs_verified histogram:", dict(sorted(Counter(nver.values()).items())))

	tr_fields = [pa.field("question", pa.string()), pa.field("answer_text", pa.string()),
	             pa.field("answer_index", pa.int64())] + \
	            [pa.field(c, pa.string()) for c in OPT] + \
	            [pa.field("option_ids", pa.list_(pa.string()))]
	tr_struct = pa.struct(tr_fields)
	par_schema = pa.schema([
		pa.field("fact_id", pa.string()), pa.field("subject", pa.string()),
		pa.field("subject_id", pa.string()), pa.field("relation", pa.string()),
		pa.field("property_id", pa.string()), pa.field("object", pa.string()),
		pa.field("object_id", pa.string()),
		pa.field("translations", pa.struct([pa.field(l, tr_struct) for l in LANGS])),
		pa.field("n_langs_verified", pa.int64()),
	])
	lang_schema = pa.schema(
		[pa.field("fact_id", pa.string()), pa.field("language", pa.string()),
		 pa.field("subject", pa.string()), pa.field("relation", pa.string()),
		 pa.field("object", pa.string()), pa.field("question", pa.string())] +
		[pa.field(c, pa.string()) for c in OPT] +
		[pa.field("answer_text", pa.string()), pa.field("answer_index", pa.int64()),
		 pa.field("option_ids", pa.list_(pa.string())),
		 pa.field("n_langs_verified", pa.int64())])

	# ---------- write repaired configs (options re-rendered via option_ids) -------
	# One batch is resident at a time: 13 writers are opened per split and fed
	# incrementally, so peak memory is a batch rather than the whole release.
	for l in LANGS + ["parallel"]:
		os.makedirs(os.path.join(args.out_dir, "data", l), exist_ok=True)
	for s in SPLITS:
		w = {l: pq.ParquetWriter(os.path.join(args.out_dir, "data", l, f"{s}.parquet"),
		                         lang_schema, compression="snappy") for l in LANGS}
		w["parallel"] = pq.ParquetWriter(
			os.path.join(args.out_dir, "data", "parallel", f"{s}.parquet"),
			par_schema, compression="snappy")
		for batch in pq.ParquetFile(src(s)).iter_batches(batch_size=2000):
			buf = {l: [] for l in LANGS}
			par = []
			for r in batch.to_pylist():
				qid, tr = r["object_id"], {}
				for l in LANGS:
					t = dict(r["translations"][l])
					ids = list(t["option_ids"])
					for i, c in enumerate(OPT):
						t[c] = label(ids[i], l)
					t["answer_text"] = label(qid, l)
					t["answer_index"] = ids.index(qid)
					tr[l] = t
					buf[l].append({
						"fact_id": r["fact_id"], "language": l,
						"subject": r["subject"], "relation": r["relation"],
						"object": r["object"], "question": t["question"],
						**{c: t[c] for c in OPT},
						"answer_text": t["answer_text"],
						"answer_index": t["answer_index"],
						"option_ids": ids, "n_langs_verified": nver[qid]})
				par.append({**{k: r[k] for k in
				               ("fact_id", "subject", "subject_id", "relation",
				                "property_id", "object", "object_id")},
				            "translations": tr, "n_langs_verified": nver[qid]})
			for l in LANGS:
				w[l].write_table(pa.Table.from_pylist(buf[l], schema=lang_schema))
			w["parallel"].write_table(pa.Table.from_pylist(par, schema=par_schema))
			del batch, buf, par
		for x in w.values():
			x.close()
		print(f"  wrote {s}")
	print(f"wrote 13 configs to {args.out_dir}")

	with open(drop_out, "w", encoding="utf-8") as f:
		json.dump({"union": sorted(drop_set),
		           "severed": sorted(sev),
		           "confirmed_wrong": {q: CONFIRMED_WRONG[q] for q in sorted(confirmed)},
		           "reason": "transliteration severed mid-word, or manual review "
		                     "confirmed the label denotes a different entity"}, f,
		          ensure_ascii=False, indent=1)
	with open(os.path.join(args.out_dir, "label_repairs.json"), "w",
	          encoding="utf-8") as f:
		json.dump({"n_repaired_labels": n_rep,
		           "n_dropped_entities": len(drop_set),
		           "n_dropped_facts": n_drop,
		           "dropped_by_reason": dict(drop_facts),
		           "n_langs_verified_hist": dict(sorted(Counter(nver.values()).items())),
		           "examples": log}, f, ensure_ascii=False, indent=1)
	print(f"wrote {drop_out} and label_repairs.json")


if __name__ == "__main__":
	main()
