#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upload the PolyFact-Clean subset built by build_polyfact_clean.py to the Hub.

Writes the dataset card (config declarations mirror jvonrad/PolyFact so the same
`load_dataset(repo, lang)` calls work) and pushes the parquet tree.

Usage:
  python build-poly-fact/upload_polyfact_clean.py \
    --clean_dir $SCRATCH/polyfact_clean/clean \
    --repo_id   jvonrad/PolyFact-Clean \
    [--private] [--dry_run]
"""

import argparse
import json
import os

from huggingface_hub import HfApi

LANGS = ["ar", "bn", "de", "en", "es", "fr", "id", "ja", "pt", "ru", "sw", "zh"]
CONFIGS = LANGS + ["parallel"]

CARD = """---
license: cc-by-sa-4.0
task_categories:
  - multiple-choice
  - question-answering
language:
{lang_yaml}
configs:
{config_yaml}
---

# PolyFact-Clean

The recommended clean subset of [`jvonrad/PolyFact`](https://huggingface.co/datasets/jvonrad/PolyFact):
parallel multilingual factual multiple-choice QA grounded in Wikidata, with the
highest-ambiguity relations removed and split integrity repaired.

**{n_facts:,} facts × 12 languages** ({train:,} train / {validation:,} validation / {test:,} test),
fully aligned by `fact_id` across all per-language configs.

## What is filtered

Quality verification of PolyFact (LLM-as-judge plus human review; see Appendix D.2 of
the paper) found ambiguity concentrated in a small number of relations, where a
subject can plausibly take several correct objects. PolyFact-Clean excludes four
relations:

| Relation | Wikidata property | Facts removed | Reason |
|---|---|---|---|
| country of origin | `P495` | 5,704 | highest judged ambiguity (35%) |
| place of birth | `P19` | 5,710 | highest judged ambiguity (25%) |
| genre | `P136` | 5,693 | highest judged ambiguity (25%) |
| employer | `P108` | 5,700 | semantically many-to-one; see below |
| *(label defects)* | — | 3,018 | corrupt or inverted items; see below |
| **Total** | | **25,831 ({pct}% of 100,113)** | |

The first three are the top-3 relations by judged ambiguity (Appendix D.2, Table 7).
`employer` is excluded on separate, explicit grounds rather than by that ranking: a
person may hold several jobs over time, so retaining a single gold object is
arbitrary, and in the source release this relation contained **every** known
split-integrity defect (all 73 duplicated training rows and all 3 facts that
appeared in both `train` and `test`). Removing it eliminates those defects entirely.

Surviving rows are byte-identical to their PolyFact counterparts and the schema is
unchanged. All 13 configs (12 languages + `parallel`) contain exactly the same
{n_facts:,} facts.

Verified in this release: **0** duplicate rows in any split, **0** `fact_id`
overlap between `train`, `validation` and `test`, **0** duplicate options, **0**
`answer_text`/`answer_index` mismatches, **0** entities carrying more than one gold
label in any language, **0** remaining intra-word transliteration corruption in
ar/bn/ru, and gold-answer position within 0.6pp of uniform in every language.

Reproduce with, in order:

```bash
python build-poly-fact/build_polyfact_clean.py --exclude_employer --fix_integrity \
  --raw_dir <PolyFact> --out_dir <tmp>
python build-poly-fact/build_label_repairs.py --data_dir <tmp> --out repairs.json
python build-poly-fact/apply_label_repairs.py --data_dir <tmp> --repairs repairs.json \
  --out_dir <final>
```

## Label-quality repairs

A further **3,018 facts** were removed for label defects found by auditing gold
answers against their Wikidata entity ids. Both classes proved unrepairable, so
the items were dropped rather than rewritten; the affected `fact_id`s and the
rationale are recorded in `label_repairs_applied.json`.

**Transliteration corruption (3,015 facts).** Gold labels in which a Latin-script
fragment sits inside a native-script word — Bengali
`লিঙ্কনNear-Earth গ্রহাণু গবেষণা` ("Lincoln Near-Earth Asteroid Research"), Arabic
`ماtejكو` (Matejko), Russian `cтатер` (Latin `c` for Cyrillic `с`). Wikidata holds
no label for these entities in these languages, so there is no ground truth to
repair from and inventing a translation for a released dataset is not acceptable.
Counts before removal: bn 2,937, ar 65, ru 13.

Chinese and Japanese were deliberately left untouched. Wikidata agrees verbatim
with 142 of 168 flagged Japanese labels, confirming that an acronym prefixed to
native script (`AGヴェーザー`, `SOM建筑设计事务所`) is conventional orthography in a
script that does not use spaces, not an error. Wikidata's `zh` labels are also
frequently Traditional while this dataset is standardised on Simplified, so
"repairing" from them would have been a regression.

**Negation-inverted questions (13 facts).** Detected as one `object_id` carrying
two different gold strings within a language. Inspection showed the question, not
the label, is wrong: e.g. ru *"Какой язык **не является** официальным языком
Соединённых Штатов Колумбии?"* ("which language is **not** an official language…")
with gold `нюнорск` (Nynorsk), and ja *"オランダ代表**ではない**国はどれですか？"*
with gold `バルバドス` (Barbados). These items do not test the stored Wikidata
triple at all, and their nominally correct answer already appears among the other
options, so they cannot be fixed by relabelling. Note that questions are generated
per language independently, so an inversion in one language breaks parallelism for
that fact.

This detector only surfaces inversions that happen to collide with another gold
label for the same entity, so it is a lower bound; a small number of
negation-inverted questions may remain. Most apparent negations in the corpus are
false positives from work titles (e.g. *"Who is the author of 'Do not go gentle
into that good night'?"*), which are not defects.

## Usage

```python
from datasets import load_dataset

# One language at a time (SFT / eval)
ds = load_dataset("{repo_id}", "en")
print(ds["train"][0])

# All languages aligned per fact (cross-lingual training)
par = load_dataset("{repo_id}", "parallel")
print(par["train"][0]["translations"]["en"])
```

## Schema

Identical to `jvonrad/PolyFact` — see that dataset card for the full column
description. Per-language configs are flat (one row per fact/language, with
`question`, `option_a`..`option_d`, `answer_text`, `answer_index`); the `parallel`
config has one row per fact with a `translations` dict keyed by language code, plus
the Wikidata `subject_id` / `property_id` / `object_id` grounding.

## Relation coverage

The release contains 19 relations (15 after this filter). Three relations named in
the paper's construction description — `capital` (P36), `shares border with` (P47)
and `platform` (P400) — are absent from PolyFact itself. They are eliminated by the
`--require_unique_subject_property` construction constraint, which keeps only
(subject, relation) pairs with exactly one object: those properties are
intrinsically multi-valued in Wikidata (12 of 20 major countries carry more than
one `P36` value once historical capitals are included, e.g. Japan has 9).

## Citation

```bibtex
@article{{polyfact2026,
  title  = {{Improving Cross-Lingual Factual Recall via Consistency-Driven Reinforcement Learning}},
  author = {{von Rad, Jonathan and Arts, Louis and Burgess, George and ODonnell, Harry and
            Oikonomidis Doumpas, Ektor and Kolokytha, Eleftheria and S\\'anchez, Eduardo and
            Lu, Yao and Stenetorp, Pontus}},
  journal = {{arXiv preprint arXiv:2606.06586}},
  year   = {{2026}}
}}
```
"""


def build_card(stats, repo_id):
	lang_yaml = "\n".join(f"  - {l}" for l in LANGS)
	config_yaml = "\n".join(
		f"  - config_name: {c}\n    data_files:\n"
		+ "\n".join(f"      - split: {s}\n        path: data/{c}/{s}.parquet"
		            for s in ["train", "validation", "test"])
		for c in CONFIGS
	)
	per_split = stats["per_split_kept"]
	return CARD.format(
		lang_yaml=lang_yaml, config_yaml=config_yaml, repo_id=repo_id,
		n_facts=sum(per_split.values()), pct=stats["pct_removed"], **per_split,
	)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--clean_dir", required=True)
	ap.add_argument("--repo_id", default="jvonrad/PolyFact-Clean")
	ap.add_argument("--private", action="store_true")
	ap.add_argument("--dry_run", action="store_true")
	args = ap.parse_args()

	with open(os.path.join(args.clean_dir, "filter_stats.json"), encoding="utf-8") as f:
		stats = json.load(f)
	card = build_card(stats, args.repo_id)
	card_path = os.path.join(args.clean_dir, "README.md")
	with open(card_path, "w", encoding="utf-8") as f:
		f.write(card)
	print(f"wrote {card_path} ({len(card)} chars)")

	if args.dry_run:
		print("\n--- dry run: not uploading ---")
		print(card[:1500])
		return

	api = HfApi()
	api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
	print(f"uploading {args.clean_dir} -> {args.repo_id} (private={args.private})")
	api.upload_folder(
		folder_path=args.clean_dir,
		repo_id=args.repo_id,
		repo_type="dataset",
		commit_message="Add PolyFact-Clean: PolyFact minus the 3 highest-ambiguity relations",
		ignore_patterns=["*.tmp"],
	)
	print(f"done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
	main()
