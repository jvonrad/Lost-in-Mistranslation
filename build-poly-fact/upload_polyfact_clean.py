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
| *(question defects)* | — | 6,667 | leaked or ambiguous items; see below |
| currency | `P38` | 1,056 | labels unverifiable; see below |
| *(item defects)* | — | 2,928 | asteroid / foreign-script / subject-less; see below |
| *(defective labels)* | — | 721 | cross-script or substituted; see below |
| *(answer-share cap)* | — | 2,766 | over-represented answers; see below |
| *(wrong or severed labels)* | — | 799 | confirmed by manual review; see below |
| *(CJK answer leakage)* | — | 61 | missed by a threshold bug; see below |
| **Total** | | **{n_removed:,} ({pct}% of 100,113)** | |

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
`answer_text`/`answer_index`/`option_ids` mismatches, **0** entities carrying more
than one gold label in any language, **0** intra-word transliteration corruption in
ar/bn/ru, **0** questions containing their own answer, **0** questions containing a
distractor (see the CJK caveat below), **0** question strings with conflicting
gold answers among rows marked `question_verified=true`, **0**
foreign-script options, **0** cross-script contaminated labels, **0** labels that
are another entity's canonical Wikidata label, **0** cross-lingual option-set
mismatches, and gold-answer position within 0.4pp of uniform in every language.

Reproduce with, in order:

```bash
python build-poly-fact/build_polyfact_clean.py --exclude_employer --fix_integrity \
  --raw_dir <PolyFact> --out_dir <s1>
python build-poly-fact/build_label_repairs.py --data_dir <s1> --out repairs.json
python build-poly-fact/apply_label_repairs.py --data_dir <s1> --repairs repairs.json \
  --out_dir <s2>
python build-poly-fact/filter_question_defects.py --data_dir <s2> --out_dir <s3>
python build-poly-fact/resample_distractors.py --data_dir <s3> --out_dir <s4> \
  --cap_answer_share 0.40 --drop_relations currency --drop_entities bad_entities.json
python data_analysis/verify_labels_deep.py --local_dir <s4>
python build-poly-fact/repair_labels.py --data_dir <s4> --out_dir <s5>
python build-poly-fact/resample_distractors.py --data_dir <s5> --out_dir <s6> --cap_answer_share 0.40 --drop_entities <s5>/severed_entities.json
python data_analysis/make_review_files.py --data_dir <s6> --out_dir <review>
# manually (or LLM-assisted) fill in verdicts_labels_*.json / verdicts_questions_*.json in <review>/
python data_analysis/aggregate_review.py --review_dir <review>
python build-poly-fact/resample_distractors.py --data_dir <s6> --out_dir <s7> --cap_answer_share 0.40 --drop_entities results/manual_review_entities.json
python build-poly-fact/apply_question_flags.py --data_dir <s7> --out_dir <s8> --review_dir <review>
python build-poly-fact/repair_arabic_citizenship.py --data_dir <s8> --out_dir <s9>
python build-poly-fact/rebalance_splits.py --data_dir <s9> --out_dir <v13>
python data_analysis/make_full_test_census.py --data_dir <v13> --out_dir <census>
# Fill every census_*.json worksheet, then validate and aggregate:
python data_analysis/aggregate_full_test_census.py --review-dir <census>
python data_analysis/make_retranslation_files.py --data_dir <v13> \
  --review_dir <census> --out_dir <retranslations> --review_format full_census \
  --exclude_retrans_dir <prior_retranslations>
# Fill every retranslated_*.json worksheet, then apply the gated repairs:
python build-poly-fact/apply_full_census.py --data_dir <v13> --out_dir <s10> \
  --review_dir <census> --retrans_dir <retranslations> \
  --english_adjudication results/english_conflict_adjudication.json \
  --translation_adjudication results/census_translation_adjudication.json
python build-poly-fact/repair_distractor_leaks.py --data_dir <s10> --out_dir <final>
python build-poly-fact/verify_v13.py --data_dir <final> --reference_dir <v13>
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

## Question-quality filtering

A further **6,667 facts** were removed for defects in the question itself, listed
in `question_defects_removed.json`. A fact is dropped if it is defective in ANY
language: the corpus is parallel and the consistency metrics read all 12
languages, so a fact that leaks in French but not Chinese still corrupts the
cross-lingual measurement.

**Answer leakage (4,550 facts).** The gold answer occurs verbatim inside the
question, so the item is solvable by copying with no factual knowledge — "Who
manufactured the Nokia 5000?" → `Nokia`, "Who developed the Sega TeraDrive?" →
`Sega`. In 97% of cases this is because the subject's name contains the object's
name, which concentrated the problem in `manufacturer` (21.1% of its items) and
`developer` (7.7%).

This mattered more than the raw rate suggests. A copied answer is byte-identical
in every language, so a leaked item scores as perfectly cross-lingually consistent
no matter what the model knows — inflating precisely the consistency metrics this
dataset exists to measure, and potentially differently for models that differ in
how readily they copy from the prompt.

Matching is word-boundary aware for space-using scripts (so a short gold inside a
longer unrelated word is not a false positive) and plain-substring for Japanese
and Chinese, which are written without spaces.

**Ambiguous questions (2,341 facts).** One question string carrying more than one
gold answer, because the subject label does not identify the entity: *"Who is the
creator of the self-portrait?"* has **25** distinct golds across 25 different
paintings, *"Who is the creator of the painting Madonna and Child?"* has 4, and
*"On which continent is Danişment located?"* is answered both Asia and Europe.
Such items are unanswerable as posed.

## Resampled distractors — read this before comparing scores

> **The distractors in this release are not the ones in `jvonrad/PolyFact`.**
> Accuracy here is not comparable with accuracy on the original distractors.

An audit of the previous release found that a model-free baseline — *pick the
option that occurs most often as a gold answer in the training split* — scored
**69.97% on the test set** against 25% chance. The cause was distractor sampling:
42.2% of distractors never served as a gold answer anywhere, and in 12.8% of items
the gold was the only option that ever did, so "answer-likeness" nearly solved the
task. That is a validity problem for any benchmark, and a sharper one for
post-training research: a model fine-tuned on this data can learn the
answer-entity prior directly while an untuned baseline cannot, and because the
prior is entity-based it is language-independent, so exploiting it yields high
*cross-lingual consistency* without any cross-lingual knowledge.

All distractors were therefore redrawn. For each relation, distractors are sampled
from the entities that do serve as gold answers **for that same relation**, with
probability proportional to how often each is a gold. Golds follow that same
distribution by definition, so the four options are exchangeable draws from one
distribution and no statistic monotone in frequency can identify the gold.
Sampling from the relation's own pool also gives type matching for free. A
distractor is rejected if it appears in the question text or is the question's
subject.

Model-free baselines on the test split, before and after:

| heuristic | original distractors | this release |
|---|---|---|
| pick most frequent gold string | **69.97%** | **27.28%** |
| pick least frequent gold string | — | 30.81% |
| longest option | 15.77% | 23.15% |
| chance | 25.00% | 25.00% |

**Please report this floor alongside model scores.** Roughly 30% is reachable with
no factual knowledge at all, so a model scoring 60% is ~30 points above the
attainable floor, not 35 above chance.

The residual above 25% is structural and was not removable. Distractors must
differ from the gold, so a relation's most frequent answer can never be handed a
more frequent distractor. Four schemes were implemented and measured: hard
log2-frequency buckets (43.2%), exact frequency-tier matching (35.9%),
frequency-rank balancing (33.2%), and the distribution matching used here.
Capping answer shares more aggressively does not help either — it only moves the
signal from the "most frequent" heuristic into the "least frequent" one, leaving
the best available heuristic flat at ~31% while costing more data.

### New column: `option_ids`

Distractors are now sampled as **entities** and rendered into each language via
that entity's canonical label, so all 12 languages offer exactly the same four
candidates. The Wikidata ids are stored in `option_ids`, aligned positionally with
`option_a..option_d`, in both the per-language and `parallel` configs.

This makes the parallel premise hold by construction: cross-lingual option-set
mismatch is **0** facts, against 0.4–1.4% previously (and previously only ~25% of
facts were even checkable, because distractor ids were not stored). Metrics such as
RankC can now align options across languages by exact entity identity rather than
by string or embedding similarity.

## Label provenance: how wrong are the labels, measured

PolyFact's construction used Gemma-3-27B to write the questions and, where
Wikidata had no label for an entity in a target language, to **translate** the
label. Those translations have no ground truth behind them, so this release
quantifies the resulting error rate rather than guessing at it.

Every answer label was checked against the union of the entity's Wikidata label,
its **language variants** (`zh-hans`, `zh-hant`, `pt-br`, …), the language's
**Wikipedia article title**, and its **aliases**. Checking only the bare language
code, as an earlier audit did, is badly misleading: it reported 25% of Chinese
labels as contradicting Wikidata, when almost all were Simplified text compared
against a Traditional label.

| | attested | contradicts Wikidata | no attestation to check against |
|---|---|---|---|
| all 12 languages | **70.6%** | **0.3%** (608 of 193,188) | 28.9% |
| worst (sw / bn) | 18.7% / 21.8% | 0.1% / 0.2% | 81.1% / 77.9% |
| best (en / fr) | 99.6% / 98.2% | 0.2% / 0.2% | 0.1% / 1.5% |

A hallucination can only hide in the unattested 28.9% (55,903 labels). Those split
sharply by kind: a label that is a **phonetic rendering** of the name
(`ডগলাস অ্যাডামস` for Douglas Adams) cannot denote something else, whereas a label
that shares no sound with the original is either a real translation or an
invention. Romanising each label and scoring it against the English name separates
the two, and **600 labels were then reviewed by hand**:

| stratum | size | reviewed | denote the wrong thing | 95% CI |
|---|---|---|---|---|
| phonetic transliteration | 48,201 | 180 | **0** | [0%, 2.1%] |
| semantic translation | 7,702 | 420 | **25 (6.0%)** | [4.1%, 8.6%] |

Weighted by stratum size and by how many facts each entity actually serves, this
puts the label error rate at **0.12% of (fact, language) items** — 95% upper bound
0.43%. The failure mode is narrow and almost entirely one thing: **a proper name
that is also a common noun gets translated instead of kept.** `Propaganda Games` →
宣传游戏, `Wisdom Tree` → Древо мудрости, `Black Flag` → Bendera Nyeusi, `Croc` →
鳄鱼 (crocodile), `Horn` (a town) → Pembe (an animal's horn), `Conservatory of Nice`
→ Hifadhi (a nature reserve). 18 of the 25 were Swahili or Chinese, and 17 fell in
`developer` / `manufacturer` / `creator`.

Every entity confirmed wrong by that review was dropped, along with 98 whose
transliteration stopped mid-word and left a Latin remnant fused to the native
script (`টাডেউশ তোলভিński`, `গ্দাńsk`) — 172 facts in total.

**Not done, deliberately.** The 608 labels that differ from Wikidata's were left
alone rather than overwritten. Inspection showed that set is dominated by valid
alternative names, not errors: `pt` carries NASA's expansion where Wikidata has the
acronym, `sw` carries `CERN` where Wikidata has the Swahili expansion, `en` carries
`Alludo` where Wikidata still says `Corel`, and most of the rest differ by a hyphen
or an accent. Applying Wikidata wholesale would have replaced
"Skidmore, Owings & Merrill" with "SOM". A label differing from Wikidata's is not
evidence that it is wrong.

### New column: `n_langs_verified`

Rather than shipping only the labels that could be verified — which would cost half
the dataset to remove a 0.12% error rate, and would strip the long tail
preferentially, since obscure entities are exactly the ones Wikidata lacks labels
for — every row carries a count of **how many of the 12 languages have this fact's
gold label attested in Wikidata** (0–12).

```python
strict = ds.filter(lambda r: r["n_langs_verified"] == 12)   # 29,542 facts (49.8%)
```

Report headline numbers on the full release and confirm them on the strict subset.

`currency` was dropped entirely rather than filtered: only **10 of its 346 answer
entities** have a Wikidata label in all 12 languages, so its labels cannot be
verified at all, and inspection found a high rate of substituted currencies
(Georgian *maneti* rendered as *lari*, Korean *yen* as *won*, Portuguese *real* as
*Brazilian real*).

## Manual review: the full unattested-label stratum and the full test split

The 0.12%-error estimate above came from a 600-label sample. Rather than stop
there, every remaining unverifiable label that could plausibly hide a wrong
referent was reviewed — **7,708 labels** (the full "semantic" stratum: unattested
by Wikidata AND not a phonetic rendering of the English name). Questions received
two exhaustive passes. The first reviewed the old 1,503-fact test split (16,687
non-English questions). After the test split was rebalanced to 2,523 facts, a new
strict-prompt census reviewed **all 27,753 non-English questions** plus all **2,523
English questions against their stored golds**. The strict prompt names known bug
classes and warns against morphology false positives; a controlled comparison
showed that it finds about 1.4x as many defects as the original prompt.

**This review was LLM-assisted, not native-speaker-verified.** Treat the
findings as a thorough first pass, not a ground-truth annotation; a systematic
recheck by fluent speakers would be needed to publish the exact rates as final.

| | reviewed | confirmed wrong | rate |
|---|---|---|---|
| labels (all splits) | 7,708 | 275 (207 distinct entities) | 3.6% |
| old test questions, original prompt | 16,687 | 173 subject + 236 type + 71 relation | 2.8% excl. unsure |
| current test translations, strict prompt | 27,753 | 331 subject + 483 type + 254 relation | **3.85%** (95% Wilson CI 3.63–4.08%) |
| current English vs. gold | 2,523 | 103 conflicts | **4.08%** (95% Wilson CI 3.38–4.93%) |

**Labels**: the 207 confirmed-wrong entities (620 facts) were dropped from this
release — see the counts table above. Worst by language: sw 6.9%, id 7.5%
(small n=173), zh 3.7%, ar 2.1%, bn 1.2%. The failure mode is the same one found
in the original 600-label sample — a proper name that is also a common noun gets
translated instead of kept — but at this scale it also caught institution-name
errors specific to Swahili (`Royal Danish Navy` → "Royal Sea of Denmark",
`Electorate of Hesse` → "Election of Hesse", `chemistry faculty` → "pottery
faculty") that a smaller sample had not surfaced.

**Test questions were NOT dropped.** One target-language verdict was overturned
at the repair gate because its note demonstrably belonged to a different row;
the census rates above retain the raw review result for auditability. A
`subject`-class defect (question asks
about a different entity than the fact's subject — e.g. `Q1361995` stores the
French commune *Jessains* but its Arabic question asked about the Falkland
Islands) is a real correctness bug. But most flagged items are `type` (right
subject, wrong kind of thing — a TV episode called a film) or `relation` (wrong
property, including reversed relations), and both classes mix genuine drift with
the reviewer's own uncertainty about localized titles. Auto-dropping whole facts
on that basis would have been a bigger, less reversible decision than shipping
the finding. Instead:

### New column: `question_verified`

Every test-split and validation-split (fact, language) pair carries
`question_verified`: `true` if the question was checked and asks about the right
subject and property, `false` if a defect was confirmed, and `null` if the
reviewer marked it `unsure` or `vague`. English is checked against the gold in
both splits. Train remains `null` — no full census has been run on it. If
English contradicts the gold, target-language questions for that fact are
conservatively `false` unless they were regenerated from a corrected source;
fidelity to a false English reference is not factual verification.

```python
clean_test = ds.filter(lambda r: all(r["translations"][l]["question_verified"]
                                     is not False for l in LANGS))
```

Two systematic bugs surfaced independently in multiple languages, worth knowing
about even where `question_verified` is `true` for a given row, since they are
subject-driven rather than translation-noise:

- **Place read as a language.** Several obscure villages (`Ditak`, `Aralez`,
  `Nor Artik`, `Hase`, `Kemperi`, `Maicas`, `Ogassa`, `Morés`, `Mirnoe`, `Saini`)
  have no non-English Wikidata label, and independently across Arabic, Swahili,
  Chinese, Japanese and Russian, the generator rendered "the official language of
  X" as if X itself were a language ("the Ditak LANGUAGE"), then asked what
  country that invented language is official in. Confirmed via the English
  question, which is unaffected — the bug is generation-side, not a translation
  artifact of a correct English source.
- **"Municipality of citizenship" (Arabic only) — FIXED in this release.**
  784 `country of citizenship` questions rendered the property as بلدية
  ("municipality") instead of بلد ("country"), making the item unanswerable in
  Arabic while correct in the other 11 languages and in the stored triple. This
  was repaired rather than dropped, because the same relation is already rendered
  correctly in 4,824 other Arabic questions: the broken 784 were rewritten to the
  corpus's own majority template (`ما هي بلد مواطنة X؟`, 4,496 pre-existing
  occurrences) rather than to any wording invented for this release. Three
  substitutions covered all 784 with no anomalies — `بلدية مواطنة` → `بلد مواطنة`
  (241), `بلدية المواطنة` → `بلد المواطنة` (37), and bare `بلدية` → `بلد مواطنة`
  (506, which also restores the dropped "citizenship" word). Scope was strictly
  this relation; `بلدية` is left untouched in the 129 questions on other relations
  (`country`, `official language`, `architect`, `continent`) where the subject
  genuinely is a municipality. Only Arabic question text changed — verified by
  diffing every field of all 59,352 facts × 12 languages, which found exactly 784
  differences, all in `ar.question`. Per-fact before/after pairs are in
  `arabic_citizenship_repairs.json`.

  Note the target template pairs the feminine interrogative `ما هي` with the
  masculine `بلد`, which is loose Modern Standard Arabic. That inconsistency is
  pre-existing in the 4,496 majority items and was deliberately preserved, so the
  repaired questions are indistinguishable from the rest of the relation; a
  grammatically tidier `ما هو` would have made them a detectable subset.

## Re-translated questions, and a leakage bug this surfaced

Confirmed target-language defects were re-translated from English by per-language
model translators. Each translator received the broken question, the reviewer's
diagnosis, the gold answer it must not reveal, and **style anchors**: real correct
questions from this corpus in the same language and relation. Every proposal was
then gated for gold leakage, distractor leakage, script, plausible length, and an
unchanged response.

The earlier pass accepted 455 target-language repairs. The strict census offered
1,050 new target-language defects not already handled in that pass; **803 were
accepted**. Another 246 fluent proposals were deliberately rejected because the
English source itself conflicts with the gold, so translating it cannot establish
factual correctness. One apparent defect was overturned when its review note was
shown to belong to an unrelated row. Separately, English editorial adjudication
accepted **99 neutral English rewrites**, left three suspect stored triples flagged,
and overturned one corporate-name false positive.

Every accepted item is marked **`question_regenerated = True`**. The final test
split contains **1,357 regenerated questions** in total (455 earlier target-language
repairs + 803 strict-census target repairs + 99 English repairs). This makes the
repair auditable rather than a silent rewrite: analyses worried about mixing
generators can exclude them with one filter. No answer label was changed by the
question-repair stage.

### CJK answer leakage (61 facts removed)

Gating the re-translations for answer leakage exposed a bug in the pipeline's own
leakage filter. `filter_question_defects.py` skipped any gold shorter than 3
characters, which is correct for alphabetic scripts but wrong for CJK, where a
whole word fits in two characters — 日本 is "Japan", 苏联 is "Soviet Union". Every
short CJK gold was therefore exempt from the check that removed 4,550 leaked items
elsewhere, and **61 genuinely answer-leaking facts survived into every release up
to this one**:

    苏联第301步兵师属于哪个国家？        gold 苏联   ("The Soviet 301st Rifle Division
                                                   belongs to which country?")
    雀巢咖啡是由谁制造的？               gold 雀巢   ("Nescafé is made by whom?")
    英国广播公司国际频道属于哪个国家？     gold 英国   ("BBC World Service belongs to
                                                   which country?")

These are the same subject-contains-object pattern the filter was built to catch.
They are now removed (55 zh, 8 ja; a fact is dropped if it leaks in any language,
per the existing rule) and the threshold is script-aware. This matters more than 61
items suggests: a copied answer is identical in every language, so a leaked item
scores as perfectly cross-lingually consistent regardless of what the model knows —
inflating precisely the metric this dataset exists to measure.

The final all-split verifier also found two Chinese **distractor** leaks in train:
the subject of the 258th Infantry Division question contained the Germany
distractor, and `US Montauban 82` contained the United States distractor. Those
two distractor entities were replaced from the same `country` gold pool across
all 12 languages; the facts, questions, golds, and answer positions were retained.

### English questions that contradict their own gold

Translators first exposed questions where the **English itself** is wrong while
the stored triple is generally fine:

- `Q3522061` asks about "the **1962** film The Devil's Hand"; the gold, Christian
  E. Christiansen, directed the **2014** film.
- `Q3535650` asks who created "the **statue** of Toussaint Louverture in Nantes";
  the gold, Philippe Niang, directed the **2012 TV film**.

That discovery led to the full English-vs-gold census reported above: **103/2,523
(4.08%, 95% Wilson CI 3.38–4.93%)** were initially marked conflict. Editorial
adjudication found 99 question-generation errors, repaired them with neutral
relation-faithful questions, identified three suspect stored triples that remain
`question_verified=false`, and overturned one false positive. The 21 English
questions marked `vague` remain `null` and are reported separately rather than
counted as conflicts.

A faithful translation of a false English question inherits the error. Target
questions for the 102 confirmed English/gold conflicts are therefore conservatively
flagged unless regenerated from a corrected source; no fact was silently dropped or
given a new gold. Full per-item decisions are recorded in
`english_conflict_adjudication.json` in the build artifacts.

## Validation-split census

The test census's methodology was repeated on the full validation split (444
facts, all 12 languages) after a fact-id diff showed a rebalance-related gap: an
earlier 300/600-fact validation sample had been reviewed before validation was
rebalanced (1,464 → 444 facts), and none of the 444 facts that ended up staying
in validation were among the sampled ones. Coverage was therefore 0% until this
pass — same strict prompt as the test census, all 444 facts, not a sample.

| | reviewed | confirmed wrong | rate |
|---|---|---|---|
| validation translations | 4,884 | 80 subject + 77 type + 16 relation | **3.54%** (95% Wilson CI 3.06–4.10%) |
| validation English vs. gold | 444 | 9 conflicts | **2.03%** (95% Wilson CI 1.07–3.81%) |

Both rates are statistically indistinguishable from test's (3.85% and 4.08%
respectively) — this is the first same-prompt, same-methodology comparison across
the two splits, and it shows no evidence that either split is dirtier.

A hypothesis worth naming and ruling out: are the flagged facts simply the harder
ones (obscure entities a translator would struggle with regardless)? No.
Checked two ways — subject Wikipedia-coverage across the 12 languages (defect
facts averaged 4.60/12, clean facts 4.46/12 — no gap, if anything the reverse)
and base-model recall accuracy on the analogous test-split facts across 4
model variants (gaps of ±0.005–0.019, no consistent direction. What *does* predict
a defect is **relation**: `country`, `author`, `director`, `official language`
and `language of work or name` run 38–43% defective, while `country of
citizenship`, `place of death` and `educated at` run 2.5–4.7% — a systematic,
template-level effect, not a difficulty effect. Dropping defective facts instead
of fixing them would have skewed the relation distribution, not the difficulty
distribution.

All 106 confirmed-defective validation facts were repaired rather than dropped,
via the same gated re-translation pipeline as the test census (per-language
translators given style anchors, then gated on leakage/distractor/script/length).
170 of 184 proposed fixes were accepted (161 translation repairs + 9 English
repairs); 14 were rejected because the translation faithfully carried over a
now-corrected English detail and needs a second pass from the fixed source —
the same residual class the test census left, not a new defect. This pass also
fixed a previously-undocumented bug found incidentally during review: Korean
script (Hangul) contaminating a small number of non-Korean questions (a
Korean-origin proper name/title that failed to transliterate) — 1 validation
fact and 2 test facts, all now corrected; a full-dataset mechanical scan (Hangul
has no legitimate occurrence in any of the 12 languages) found 49 total
occurrences, the rest in train, left as a documented residual defect class
rather than fixed before this release.

## Answer-share cap

No single entity may be the correct answer for more than 40% of its relation's
items. `Gunter Demnig` was the answer to 62% of all `creator` questions (3,173
near-identical Stolperstein items) and `Asia` to 44% of `continent` questions, which
made the benchmark repeatedly test a handful of facts and inflated any model that
knew them. Capping brought the worst P(gold | entity is an option) from 0.638 to
0.463. A stricter 25% cap was measured and rejected: it reaches 0.367 but costs
10.6% of the data and makes the least-frequent-option heuristic stronger.

## Other item-level removals

- **2,202 asteroid-designation subjects** (`(85254) 1993 TG12`), whose answers are
  extremely concentrated — `Spacewatch` alone was the gold for 517 of them — and
  which are catalogue lookups rather than world knowledge.
- **480 items with an option in a script foreign to the language config**: CJK
  names inside German questions, Cyrillic inside Swahili. Such an option is
  eliminable without knowing the fact, and in 87 cases the foreign-script string
  was the gold itself.
- **234 items whose question never names its subject** — e.g. subject
  `103733 Bernardharris` with the question "Who is credited with the discovery of
  the dwarf planet Makemake?". Unanswerable as posed. (Detected in English.)
- 12 further facts for which no valid distractor set could be drawn.

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
		n_facts=sum(per_split.values()), pct=stats["pct_removed"],
		n_removed=100113 - sum(per_split.values()), **per_split,
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
