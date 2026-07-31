# Audit of `jvonrad/PolyFact-Clean`

## C2/C3 — size and relation count

- facts: **83,006** (train 78,847, validation 2,067, test 2,092)
- distinct relations: **16**

## C1 — "0 (subject, relation) pairs map to >1 object"

- distinct (subject_id, property_id) pairs: 82,930
- pairs mapping to >1 distinct object_id: **0**

## C5 — split integrity

- train: 78,847 rows, 78,774 unique → **73 duplicate rows**
- validation: 2,067 rows, 2,067 unique → **0 duplicate rows**
- test: 2,092 rows, 2,092 unique → **0 duplicate rows**
- train∩test: **3**, train∩val: **0**, val∩test: **0**
  - leaked fact_ids: ['Q1890643|P108|Q214341', 'Q30279183|P108|Q36188', 'Q3158256|P108|Q156598']

## C6 — translation artifacts / English-canonical options

Share of a language's 4 options that are byte-identical to an English option for the same fact (an untranslated label), and share of *gold answers* identical to English:

| lang | options identical to en | gold identical to en | gold in Latin script |
|---|---|---|---|
| de | 52.8% | 46.1% | — |
| id | 53.6% | 51.5% | — |
| pt | 48.3% | 45.0% | — |
| ar | 0.8% | 0.5% | 0.5% |
| bn | 0.9% | 0.6% | 1.5% |
| sw | 52.4% | 49.3% | — |
| es | 49.1% | 47.9% | — |
| ru | 10.4% | 10.8% | 13.6% |
| fr | 48.3% | 49.2% | — |
| ja | 1.9% | 1.3% | 2.1% |
| zh | 2.1% | 1.8% | 3.2% |

## C7 — option position balance

| lang | A | B | C | D | max−min |
|---|---|---|---|---|---|
| en | 25.1% | 25.0% | 25.0% | 24.9% | 0.2pp |
| de | 24.8% | 25.1% | 25.1% | 25.1% | 0.3pp |
| id | 24.8% | 25.1% | 25.1% | 25.1% | 0.3pp |
| pt | 24.9% | 24.8% | 25.4% | 24.9% | 0.5pp |
| ar | 24.9% | 25.1% | 24.9% | 25.1% | 0.2pp |
| bn | 24.8% | 24.8% | 25.1% | 25.2% | 0.4pp |
| sw | 24.9% | 25.0% | 25.0% | 25.1% | 0.2pp |
| es | 24.7% | 24.9% | 25.1% | 25.3% | 0.6pp |
| ru | 25.1% | 24.8% | 25.1% | 25.0% | 0.4pp |
| fr | 24.9% | 25.0% | 25.1% | 25.0% | 0.2pp |
| ja | 25.1% | 25.1% | 24.9% | 24.9% | 0.2pp |
| zh | 24.9% | 25.2% | 24.8% | 25.1% | 0.4pp |

## Option integrity

- (fact, language) items with duplicate options: **0**
- items where `answer_text` != `option_{answer_index}`: **0**



---

## C1 on the FULL dataset (`jvonrad/PolyFact`)

- facts: **100,113**
- distinct (subject_id, property_id) pairs: **100,037**
- pairs mapping to >1 distinct object_id: **0**

The rebuttal's "0/100,037 (subject, relation) pairs map to >1 object" is exactly
correct: 100,037 is the count of distinct (subject, relation) PAIRS, not of facts.
Note this uniqueness holds *by construction* (one object retained per
subject-relation), so it answers aDXs's Q1(i) but is not independent evidence that
the questions are semantically unambiguous - that is what the LLM-judge ambiguity
rate measures separately.

## C4 - which relations the Clean filter actually removes

- full PolyFact: 19 relations; PolyFact-Clean: 16 relations
- removed: **country of origin (P495), genre (P136), place of birth (P19)**
  - matches Appendix D.2 / Table 7 and the rebuttal to aDXs
- **`employer` is RETAINED in PolyFact-Clean.** The rebuttal to Reviewer kSu5 says
  the clean set "excluded the highest ambiguity properties (such as genre,
  employer)". `genre` is excluded; `employer` is not (Table 7 lists it at 11%
  ambiguity, well below the excluded three). Now that the dataset is public this
  is directly checkable by the reviewer.
- `employer` is also the relation containing every split-integrity defect: all 73
  duplicate train rows and all 3 train/test leaked facts.

## C3 - reported vs. actual split sizes

| | paper / checklist B6 | actual (full PolyFact) | actual (Clean) |
|---|---|---|---|
| train | 95,000 | 95,097 | 78,847 |
| validation | 2,500 | 2,493 | 2,067 |
| test | 2,500 | 2,523 | 2,092 |
| total | 100,000 | 100,113 | 83,006 |

## C2 - relation count

The paper (App. D.1) says 22 relations are retained and lists 22 names, but the
released data contains **19**. Absent from the release: `capital`,
`shares border with`, `platform`.
