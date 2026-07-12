# PolyFact contamination analysis

_Generated 2026-07-11T13:58:08+00:00_

PolyFact facts are Wikidata triples `(subject_id, property_id, object_id)`, which lets us measure overlap at the entity, relation, and full-triple level rather than by fuzzy string matching alone.

## 1. PolyFact split integrity (train vs. test/validation)

| target split | facts | exact triple in train | (subject,relation) in train | subj-entity in train | obj-entity in train |
|---|---|---|---|---|---|
| validation | 2493 | 0 (0.0%) | 0 (0.0%) | 10.52% | 72.2% |
| test | 2523 | 3 (0.119%) | 3 (0.119%) | 11.67% | 73.44% |

Entity-level overlap is expected and by design (the splits draw on the same entity universe); the relevant leakage numbers are the exact-triple and (subject, relation) columns.

## 2. PolyFact x KLAR-CLC (OOD factual-recall benchmark)

- PolyFact relations: 19, KLAR relations: 20, **shared: 8** (P176, P178, P19, P20, P27, P30, P37, P407).
- KLAR eval facts in shared relations: 1207.

**KLAR-side (does the model train on KLAR's eval facts?):**
- KLAR eval facts whose exact `(relation, subject, object)` is in PolyFact-train: **157 / 1207 (13.007%)**.
- KLAR eval facts whose `(relation, subject)` query is in PolyFact-train: **180 (14.913%)**.

| PolyFact split | facts (shared rel.) | exact triples w/ KLAR | (subj,rel) w/ KLAR | subj entities | obj entities |
|---|---|---|---|---|---|
| train | 42927 | 157 | 168 | 223 | 154 |
| test | 1139 | 8 | 8 | 10 | 86 |

## 3. PolyFact x Global-MMLU (broad-knowledge MCQ)

No shared Wikidata ids, so overlap is measured on surface strings.
- Global-MMLU items: 14042; PolyFact entities indexed: 113386.
- Items co-mentioning a full PolyFact fact (both subject AND object): **328 (2.336%)**.
- Items mentioning any PolyFact subject entity: 11687 (83.229%).
- Verbatim question 8-gram overlap: **7 (0.05%)**.
- Co-mention by MMLU category: {'Humanities': 268, 'Social Sciences': 38, 'Other': 14, 'Medical': 4, 'STEM': 3, 'Business': 1}.
