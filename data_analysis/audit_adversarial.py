#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adversarial audit of PolyFact-Clean: the checks a hostile reviewer would run.

Earlier audits established that the dataset is internally consistent. This one
asks whether it actually MEASURES factual recall, i.e. whether the task can be
beaten without knowing the fact, and whether every item is answerable at all.

  1. HEURISTIC BASELINES. Accuracy of model-free strategies over the 4 options:
     most globally frequent gold string, longest / shortest option, alphabetically
     first, and the per-relation majority object. Anything well above 25% is
     exploitable structure that inflates every reported score.
  2. DISTRACTOR TYPE MATCHING. Share of distractors that never occur as a gold
     answer for the same relation — i.e. drawn from the wrong semantic pool, so
     they can be eliminated on type alone.
  3. DISTRACTOR CONCENTRATION. Whether a handful of entities supply most
     distractors for a relation, which lets a model learn "never pick X".
  4. UNANSWERABLE ITEMS. Questions that do not mention their subject at all
     ("Who was the creator?"), checked directly in English and via a
     question-length ratio against English for the other languages.
  5. LENGTH BIAS. Gold-vs-distractor character length; a systematic gap is a
     free signal.
  6. TRAIN/TEST SUBJECT AND OBJECT OVERLAP, and relation balance across splits.

CPU-only, streamed; safe on a login node.
"""

import argparse
import os
import re
import statistics
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LANGS = ["en", "de", "id", "pt", "ar", "bn", "sw", "es", "ru", "fr", "ja", "zh"]
SPLITS = ["train", "validation", "test"]
OPT = ["option_a", "option_b", "option_c", "option_d"]


def fetch(repo, config, split, local_dir):
	if local_dir:
		return pq.read_table(os.path.join(local_dir, "data", config, f"{split}.parquet"))
	return pq.read_table(hf_hub_download(repo, f"data/{config}/{split}.parquet",
	                                     repo_type="dataset"))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--repo", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--local_dir", default=None)
	ap.add_argument("--out", default="results/polyfact_adversarial_audit.md")
	args = ap.parse_args()

	meta = {}
	for s in SPLITS:
		for r in fetch(args.repo, "parallel", s, args.local_dir).select(
				["fact_id", "subject", "relation", "object", "object_id"]).to_pylist():
			r["split"] = s
			meta[r["fact_id"]] = r
	print(f"{len(meta):,} facts")

	en = {}
	for s in SPLITS:
		for r in fetch(args.repo, "en", s, args.local_dir).to_pylist():
			en[r["fact_id"]] = r

	lines = []

	def out(t=""):
		print(t)
		lines.append(t)

	out(f"# Adversarial audit — `{args.repo}`\n")

	# ---------- 1. heuristic baselines (English) ----------
	gold_freq = Counter(r["answer_text"] for r in en.values())
	rel_major = defaultdict(Counter)
	for fid, r in en.items():
		rel_major[meta[fid]["relation"]][r["answer_text"]] += 1
	rel_top = {rel: c.most_common(1)[0][0] for rel, c in rel_major.items()}

	strategies = {
		"random (expected)": None,
		"most frequent gold string": lambda o, rel: max(o, key=lambda x: gold_freq[x]),
		"longest option": lambda o, rel: max(o, key=len),
		"shortest option": lambda o, rel: min(o, key=len),
		"alphabetically first": lambda o, rel: sorted(o)[0],
		"per-relation majority object": lambda o, rel: (
			rel_top[rel] if rel_top[rel] in o else o[0]),
	}
	res = {k: Counter() for k in strategies}
	totals = Counter()
	for fid, r in en.items():
		split = meta[fid]["split"]
		rel = meta[fid]["relation"]
		opts = [r[c] for c in OPT]
		gold = r["answer_text"]
		totals[split] += 1
		for name, fn in strategies.items():
			if fn is None:
				continue
			if fn(opts, rel) == gold:
				res[name][split] += 1

	out("## 1. Model-free heuristic baselines (English)\n")
	out("Accuracy of strategies that use no factual knowledge. Chance is 25%.\n")
	out("| strategy | train | test |")
	out("|---|---|---|")
	out(f"| random (expected) | 25.00% | 25.00% |")
	for name in strategies:
		if strategies[name] is None:
			continue
		out(f"| {name} | {100*res[name]['train']/totals['train']:.2f}% | "
		    f"**{100*res[name]['test']/totals['test']:.2f}%** |")
	out("")

	# ---------- 2. distractor type matching ----------
	gold_by_rel = defaultdict(set)
	for fid, r in en.items():
		gold_by_rel[meta[fid]["relation"]].add(r["answer_text"])
	all_golds = set(gold_freq)
	off_pool = Counter()
	unseen = Counter()
	tot_d = Counter()
	off_examples = defaultdict(list)
	for fid, r in en.items():
		rel = meta[fid]["relation"]
		for o in [r[c] for c in OPT]:
			if o == r["answer_text"]:
				continue
			tot_d[rel] += 1
			if o not in gold_by_rel[rel]:
				off_pool[rel] += 1
				if o not in all_golds:
					unseen[rel] += 1
				elif len(off_examples[rel]) < 3:
					other = [rr for rr, gs in gold_by_rel.items() if o in gs][:2]
					off_examples[rel].append((r["question"][:55], o, other))
	out("## 2. Distractor type matching (English)\n")
	out("A distractor that never appears as a gold answer for the same relation is "
	    "drawn from the wrong semantic pool and can be eliminated on type alone. "
	    "`unseen` are strings that are not a gold answer anywhere, so their pool is "
	    "unknown.\n")
	out("| relation | distractors | off-pool | of which unseen | off-pool rate |")
	out("|---|---|---|---|---|")
	for rel in sorted(tot_d, key=lambda r: -off_pool[r] / max(tot_d[r], 1)):
		out(f"| {rel} | {tot_d[rel]:,} | {off_pool[rel]:,} | {unseen[rel]:,} | "
		    f"**{100*off_pool[rel]/tot_d[rel]:.1f}%** |")
	out("")
	for rel, exs in list(off_examples.items())[:4]:
		for q, o, other in exs[:1]:
			out(f"- [{rel}] `{o}` offered for *{q}…* — elsewhere a gold for {other}")
	out("")

	# ---------- 3. distractor concentration ----------
	out("## 3. Distractor concentration (English)\n")
	out("| relation | distinct distractors | top-1 share | top-10 share |")
	out("|---|---|---|---|")
	d_by_rel = defaultdict(Counter)
	for fid, r in en.items():
		for o in [r[c] for c in OPT]:
			if o != r["answer_text"]:
				d_by_rel[meta[fid]["relation"]][o] += 1
	for rel in sorted(d_by_rel):
		c = d_by_rel[rel]
		tot = sum(c.values())
		top1 = c.most_common(1)[0][1] / tot
		top10 = sum(n for _, n in c.most_common(10)) / tot
		out(f"| {rel} | {len(c):,} | {100*top1:.1f}% | {100*top10:.1f}% |")
	out("")

	# ---------- 4. unanswerable: subject not in question ----------
	def mentions_subject(question, subject):
		if not subject:
			return True
		q = question.casefold()
		# any subject token of 4+ chars appearing in the question counts as a mention
		toks = [t for t in re.split(r"[\s,;:'\"()\[\]/–—-]+", subject.casefold()) if len(t) >= 4]
		if not toks:
			return subject.casefold() in q
		return any(t in q for t in toks)

	missing = [fid for fid, r in en.items()
	           if not mentions_subject(r["question"], meta[fid]["subject"])]
	out("## 4. Items whose question never names its subject (English)\n")
	out(f"- **{len(missing):,} / {len(en):,} ({100*len(missing)/len(en):.2f}%)**")
	by_rel = Counter(meta[f]["relation"] for f in missing)
	by_split = Counter(meta[f]["split"] for f in missing)
	out(f"- by split: {dict(by_split)}")
	out("\n| relation | items |")
	out("|---|---|")
	for rel, n in by_rel.most_common(6):
		out(f"| {rel} | {n:,} |")
	out("")
	for f in missing[:5]:
		out(f"- subject `{meta[f]['subject']}` — *{meta[f]['relation']}* — "
		    f"Q: {en[f]['question'][:80]}")
	out("")

	# per-language question length ratio vs English (proxy for a dropped subject)
	out("### Question length relative to English\n")
	out("| lang | median chars | ratio vs en | questions <60% of en length |")
	out("|---|---|---|---|")
	en_len = {f: len(r["question"]) for f, r in en.items()}
	med_en = statistics.median(en_len.values())
	for lang in LANGS:
		lens = {}
		for s in SPLITS:
			t = fetch(args.repo, lang, s, args.local_dir).select(["fact_id", "question"])
			for f, q in zip(t.column("fact_id").to_pylist(), t.column("question").to_pylist()):
				lens[f] = len(q)
		med = statistics.median(lens.values())
		short = sum(1 for f, L in lens.items() if en_len.get(f, 0) and L < 0.6 * en_len[f])
		out(f"| {lang} | {med:.0f} | {med/med_en:.2f} | {short:,} ({100*short/len(lens):.1f}%) |")
	out("")

	# ---------- 5. length bias ----------
	gl, dl, longest_is_gold = [], [], 0
	for fid, r in en.items():
		gold = r["answer_text"]
		opts = [r[c] for c in OPT]
		gl.append(len(gold))
		dl.extend(len(o) for o in opts if o != gold)
		if max(opts, key=len) == gold:
			longest_is_gold += 1
	out("## 5. Length bias (English)\n")
	out(f"- mean gold length **{statistics.mean(gl):.1f}** chars vs "
	    f"mean distractor length **{statistics.mean(dl):.1f}**")
	out(f"- gold is the longest option in **{100*longest_is_gold/len(en):.2f}%** "
	    f"of items (chance 25%)")
	out("")

	# ---------- 6. split overlap ----------
	subj = {s: {meta[f]["subject"] for f in meta if meta[f]["split"] == s} for s in SPLITS}
	objs = {s: {meta[f]["object_id"] for f in meta if meta[f]["split"] == s} for s in SPLITS}
	out("## 6. Train/test overlap and balance\n")
	out(f"- subjects shared between train and test: **{len(subj['train'] & subj['test']):,}** "
	    f"({100*len(subj['train'] & subj['test'])/len(subj['test']):.1f}% of test subjects)")
	out(f"- answer entities shared between train and test: "
	    f"**{len(objs['train'] & objs['test']):,}** "
	    f"({100*len(objs['train'] & objs['test'])/len(objs['test']):.1f}% of test answers)")
	rel_tr = Counter(meta[f]["relation"] for f in meta if meta[f]["split"] == "train")
	rel_te = Counter(meta[f]["relation"] for f in meta if meta[f]["split"] == "test")
	ntr, nte = sum(rel_tr.values()), sum(rel_te.values())
	worst = max(rel_te, key=lambda r: abs(rel_te[r]/nte - rel_tr[r]/ntr))
	out(f"- largest train/test relation-share gap: *{worst}* "
	    f"{100*rel_tr[worst]/ntr:.1f}% vs {100*rel_te[worst]/nte:.1f}%")
	out("")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nwrote {args.out}")


if __name__ == "__main__":
	main()
