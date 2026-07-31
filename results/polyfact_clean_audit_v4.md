# Audit of `jvonrad/PolyFact-Clean`

## C2/C3 — size and relation count

- facts: **67,615** (train 64,233, validation 1,664, test 1,718)
- distinct relations: **15**

## C1 — "0 (subject, relation) pairs map to >1 object"

- distinct (subject_id, property_id) pairs: 67,615
- pairs mapping to >1 distinct object_id: **0**

## C5 — split integrity

- train: 64,233 rows, 64,233 unique → **0 duplicate rows**
- validation: 1,664 rows, 1,664 unique → **0 duplicate rows**
- test: 1,718 rows, 1,718 unique → **0 duplicate rows**
- train∩test: **0**, train∩val: **0**, val∩test: **0**

## C6 — translation artifacts / English-canonical options

Share of a language's 4 options that are byte-identical to an English option for the same fact (an untranslated label), and share of *gold answers* identical to English:

| lang | options identical to en | gold identical to en | gold in Latin script |
|---|---|---|---|
| de | 52.7% | 46.9% | — |
| id | 55.2% | 51.1% | — |
| pt | 47.9% | 43.8% | — |
| ar | 0.8% | 0.5% | 0.5% |
| bn | 0.7% | 0.5% | 1.1% |
| sw | 54.9% | 51.6% | — |
| es | 49.3% | 47.1% | — |
| ru | 8.3% | 9.6% | 11.9% |
| fr | 48.4% | 49.0% | — |
| ja | 1.6% | 1.2% | 2.0% |
| zh | 2.1% | 1.9% | 3.4% |

## C7 — option position balance

| lang | A | B | C | D | max−min |
|---|---|---|---|---|---|
| en | 25.1% | 25.0% | 25.0% | 24.8% | 0.3pp |
| de | 24.7% | 25.1% | 25.2% | 25.1% | 0.5pp |
| id | 24.8% | 25.0% | 25.0% | 25.2% | 0.4pp |
| pt | 24.9% | 24.9% | 25.4% | 24.8% | 0.6pp |
| ar | 24.8% | 25.1% | 25.0% | 25.1% | 0.3pp |
| bn | 24.8% | 24.9% | 25.2% | 25.2% | 0.4pp |
| sw | 25.0% | 25.1% | 24.9% | 25.0% | 0.2pp |
| es | 24.7% | 24.9% | 25.1% | 25.3% | 0.6pp |
| ru | 25.2% | 24.6% | 25.1% | 25.0% | 0.5pp |
| fr | 24.9% | 24.8% | 25.1% | 25.1% | 0.3pp |
| ja | 25.0% | 25.0% | 24.9% | 25.1% | 0.2pp |
| zh | 24.8% | 25.4% | 24.8% | 25.0% | 0.6pp |

## Option integrity

- (fact, language) items with duplicate options: **0**
- items where `answer_text` != `option_{answer_index}`: **0**

