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
three highest-ambiguity relations removed.

**{n_facts:,} facts × 12 languages** ({train:,} train / {validation:,} validation / {test:,} test),
fully aligned by `fact_id` across all per-language configs.

## What is filtered

Quality verification of PolyFact (LLM-as-judge plus human review; see Appendix D.2 of
the paper) found ambiguity concentrated in a small number of relations, where a
subject can plausibly take several correct objects. PolyFact-Clean excludes the three
relations reported there as most ambiguous:

| Relation | Wikidata property | Facts removed |
|---|---|---|
| country of origin | `P495` | 5,704 |
| genre | `P136` | 5,693 |
| place of birth | `P19` | 5,710 |
| **Total** | | **17,107 ({pct}% of 100,113)** |

Nothing else is changed: rows that survive the filter are byte-identical to their
PolyFact counterparts, the schema is unchanged, and the 16 remaining relations keep
their full fact sets. The filter is applied on `property_id`, so all 13 configs
(12 languages + `parallel`) contain exactly the same {n_facts:,} facts.

Reproduce it with `build-poly-fact/build_polyfact_clean.py` in the code release.

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

## Known issues inherited from PolyFact

These are present in the source dataset and are **not** introduced by the filter.
They are documented here rather than silently repaired, so that PolyFact-Clean stays
exactly the relation-level filter described in the paper. The affected ids are listed
in `known_issues.json`.

- **3 facts appear in both `train` and `test`** (all `employer`/`P108`):
  `Q30279183|P108|Q36188`, `Q1890643|P108|Q214341`, `Q3158256|P108|Q156598`.
  Drop them from `test` for a strictly leakage-free evaluation (0.14% of the test split).
- **61 `fact_id`s are duplicated within `train`** (73 excess rows), also all
  `employer`/`P108`.

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
