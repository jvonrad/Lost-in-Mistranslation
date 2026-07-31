# Adversarial audit — `jvonrad/PolyFact-Clean`

## 1. Model-free heuristic baselines (English)

Accuracy of strategies that use no factual knowledge. Chance is 25%.

| strategy | train | test |
|---|---|---|
| random (expected) | 25.00% | 25.00% |
| most frequent gold string | 78.38% | **78.99%** |
| longest option | 17.04% | **15.77%** |
| shortest option | 30.95% | **31.32%** |
| alphabetically first | 24.56% | **26.89%** |
| per-relation majority object | 37.85% | **36.44%** |

## 2. Distractor type matching (English)

A distractor that never appears as a gold answer for the same relation is drawn from the wrong semantic pool and can be eliminated on type alone. `unseen` are strings that are not a gold answer anywhere, so their pool is unknown.

| relation | distractors | off-pool | of which unseen | off-pool rate |
|---|---|---|---|---|
| place of death | 16,887 | 13,392 | 13,232 | **79.3%** |
| discoverer or inventor | 9,582 | 6,758 | 6,508 | **70.5%** |
| director | 15,441 | 9,967 | 9,808 | **64.5%** |
| manufacturer | 10,194 | 5,743 | 5,389 | **56.3%** |
| educated at | 16,818 | 8,884 | 8,809 | **52.8%** |
| creator | 15,372 | 7,720 | 6,942 | **50.2%** |
| country of citizenship | 16,953 | 7,851 | 6,185 | **46.3%** |
| developer | 14,211 | 6,332 | 5,556 | **44.6%** |
| country | 15,012 | 6,048 | 4,412 | **40.3%** |
| language of work or name | 16,251 | 6,132 | 4,308 | **37.7%** |
| continent | 14,982 | 5,054 | 5,054 | **33.7%** |
| architect | 15,420 | 4,667 | 4,630 | **30.3%** |
| official language | 16,089 | 4,232 | 3,640 | **26.3%** |
| author | 6,459 | 880 | 857 | **13.6%** |
| currency | 3,174 | 213 | 213 | **6.7%** |

- [country] `Joseon` offered for *During the British occupation of Manila, which country …* — elsewhere a gold for ['country of citizenship']
- [country of citizenship] `Guinea-Bissau` offered for *What country is Eric Degenhardt a citizen of?…* — elsewhere a gold for ['country']
- [official language] `Hindustani` offered for *What is the official language of Alberuela de Tubo?…* — elsewhere a gold for ['language of work or name']
- [language of work or name] `Valencian` offered for *What language is associated with Furrer?…* — elsewhere a gold for ['official language']

## 3. Distractor concentration (English)

| relation | distinct distractors | top-1 share | top-10 share |
|---|---|---|---|
| architect | 3,700 | 0.2% | 1.4% |
| author | 1,534 | 0.2% | 2.0% |
| continent | 24 | 15.5% | 75.8% |
| country | 498 | 0.8% | 6.5% |
| country of citizenship | 751 | 0.6% | 5.6% |
| creator | 2,563 | 0.1% | 1.0% |
| currency | 364 | 0.9% | 6.9% |
| developer | 2,937 | 0.3% | 1.8% |
| director | 7,372 | 0.1% | 0.6% |
| discoverer or inventor | 2,721 | 3.2% | 23.3% |
| educated at | 4,185 | 0.2% | 1.1% |
| language of work or name | 237 | 1.2% | 11.3% |
| manufacturer | 2,437 | 0.2% | 2.0% |
| official language | 274 | 0.8% | 7.4% |
| place of death | 9,162 | 0.1% | 0.9% |

## 4. Items whose question never names its subject (English)

- **267 / 67,615 (0.39%)**
- by split: {'train': 253, 'validation': 6, 'test': 8}

| relation | items |
|---|---|
| language of work or name | 69 |
| author | 62 |
| discoverer or inventor | 35 |
| director | 35 |
| developer | 17 |
| creator | 14 |

- subject `Youth` — *language of work or name* — Q: What language is commonly associated with manga and anime?
- subject `(523) Ada` — *discoverer or inventor* — Q: Who is credited with discovering Ada Lovelace's portrait?
- subject `Biology` — *language of work or name* — Q: In what language was the original scientific description of the human species, *
- subject `103733 Bernardharris` — *discoverer or inventor* — Q: Who is credited with the discovery of the dwarf planet Makemake?
- subject `Frequency` — *author* — Q: Who is the author of the webcomic xkcd?

### Question length relative to English

| lang | median chars | ratio vs en | questions <60% of en length |
|---|---|---|---|
| en | 46 | 1.00 | 0 (0.0%) |
| de | 49 | 1.07 | 1,362 (2.0%) |
| id | 43 | 0.93 | 2,721 (4.0%) |
| pt | 46 | 1.00 | 1,280 (1.9%) |
| ar | 32 | 0.70 | 16,497 (24.4%) |
| bn | 38 | 0.83 | 7,522 (11.1%) |
| sw | 41 | 0.89 | 6,501 (9.6%) |
| es | 49 | 1.07 | 1,582 (2.3%) |
| ru | 47 | 1.02 | 1,366 (2.0%) |
| fr | 49 | 1.07 | 585 (0.9%) |
| ja | 22 | 0.48 | 53,343 (78.9%) |
| zh | 15 | 0.33 | 66,669 (98.6%) |

## 5. Length bias (English)

- mean gold length **12.7** chars vs mean distractor length **13.9**
- gold is the longest option in **17.03%** of items (chance 25%)

## 6. Train/test overlap and balance

- subjects shared between train and test: **115** (6.7% of test subjects)
- answer entities shared between train and test: **678** (70.5% of test answers)
- largest train/test relation-share gap: *architect* 7.6% vs 7.3%



---

## 1 (corrected) — the frequency prior, computed honestly

The "most frequent gold string" row above is inflated: it counts each item's own
gold, so a gold appearing once still beats distractors appearing zero times.
Recomputed the way an actual model could exploit it — frequencies taken from the
TRAIN split only, applied to TEST, ties broken at random:

| baseline | train (leave-one-out) | test |
|---|---|---|
| random | 25.00% | 25.00% |
| **train-gold-frequency prior** | **69.40%** | **69.97%** |

Mechanism: **42.2%** of distractors never appear as a gold answer anywhere in the
dataset, and in **12.8%** of items the gold is the ONLY option that ever serves as
a gold. Golds and distractors are therefore drawn from different distributions,
and "which entities are answers here" is worth ~45 points over chance without any
factual knowledge.

This is a validity threat specific to this paper's design: SFT and GRPO are
trained ON this data, so they can learn the answer-entity prior directly, whereas
the base models cannot. The prior is also entity-identity based and therefore
language-independent, so exploiting it yields HIGH cross-lingual consistency —
inflating exactly the metrics the paper reports, and inflating them more for the
fine-tuned systems than for the baselines.

Suggested mitigation, in order of strength: (a) resample distractors for each
relation from the pool of entities that DO serve as gold answers for that
relation, matched on frequency; (b) failing that, report the frequency-prior
baseline alongside model scores so readers can see the floor; (c) at minimum,
evaluate on the subset where all four options are attested golds.

## Partially transliterated gold answers (residual)

The strict intra-word detector used to build v4 deliberately ignores a Latin run
separated by a space, to avoid flagging conventional acronyms (`استوديو MDHR`).
That leaves a milder class: a native-script label containing a lowercase Latin
WORD, which indicates a name that was only half transliterated.

| lang | count | share |
|---|---|---|
| bn | 801 | 1.18% |
| zh | 452 | 0.67% |
| ja | 81 | 0.12% |
| ru | 45 | 0.07% |
| ar | 17 | 0.03% |

Examples: bn `হেনরি ই. Holt`, `bullfrog প্রোডাকশনস`; zh `Access游戏`, `Vic东海`;
ja `ラトledge`, `スコット・Wheeler`; ru `Фонд Mozilla`.

Note some are legitimate (`Фонд Mozilla`, `Apacheソフトウェア財団` — brand names kept
in Latin by convention), so this count is an upper bound, not a defect count.

## Asteroid-designation items

2,202 facts (3.3%; 51 in test) have an asteroid designation as their subject —
`(17507) 1992 HH5`, `(85254) 1993 TG12` — all in `discoverer or inventor`. Their
answers are extremely concentrated: `Spacewatch` alone is the gold for 517 of
them, the top 5 answers cover ~1,150.
