# Full strict-prompt validation census

Ran on 2026-08-01 in response to a direct question: is the current validation split
(444 facts, post-rebalance) manually reviewed at all? It was not — the earlier
validation review (300/600-fact sample) was drawn from the pre-rebalance 1,464-fact
pool, and `rebalance_splits.py` moved 1,020 of those facts into test. Diffing
fact_ids confirmed **zero overlap** between that sample and the current 444-fact
validation set. This census closes that gap: all 444 validation facts, all 11
non-English languages plus English-vs-gold, same strict prompt as the full test
census (`data_analysis/make_full_test_census.py --split validation`).

12 worksheets, 12 subagents, one per worksheet. All 12 verdict files passed
structural validation (row count, fact_id/order match against the worksheet,
no duplicates) before aggregation.

## Translation defects (11 languages, exhaustive — not a sample)

| lang | n | ok | subject | type | relation | unsure | defect rate |
|---|---|---|---|---|---|---|---|
| ar | 444 | 416 | 10 | 16 | 1 | 1 | 6.1% |
| sw | 444 | 417 | 4 | 20 | 2 | 1 | 5.9% |
| ru | 444 | 419 | 13 | 6 | 3 | 3 | 5.0% |
| ja | 444 | 426 | 5 | 11 | 0 | 2 | 3.6% |
| zh | 444 | 423 | 8 | 7 | 1 | 5 | 3.6% |
| de | 444 | 430 | 12 | 2 | 0 | 0 | 3.2% |
| bn | 444 | 430 | 5 | 3 | 4 | 2 | 2.7% |
| es | 444 | 425 | 9 | 2 | 1 | 7 | 2.7% |
| pt | 444 | 434 | 3 | 5 | 2 | 0 | 2.3% |
| fr | 444 | 427 | 5 | 4 | 1 | 7 | 2.3% |
| id | 444 | 435 | 6 | 1 | 1 | 1 | 1.8% |
| **all** | **4,884** | **4,513** | **80** | **77** | **16** | **29** | **3.54%** (95% Wilson CI 3.06–4.10%) |

## English-vs-gold

431 ok / 9 conflict / 3 vague / 1 unsure — **conflict rate 2.03%** (95% Wilson CI
1.07–3.81%).

## Comparison to the test census

This is the first *controlled* split comparison this project has run — same strict
prompt on both splits, unlike the earlier validation-sample-vs-original-test-review
comparison, which confounded prompt strictness with split.

| | test (2,523 facts) | validation (444 facts) |
|---|---|---|
| translation defect rate | 3.85% [3.63, 4.08] | 3.54% [3.06, 4.10] |
| English-conflict rate | 4.08% [3.38, 4.93] | 2.03% [1.07, 3.81] |

Both CIs overlap. Validation is not measurably dirtier or cleaner than test under
the same rubric — the two splits appear to carry the same underlying defect rate.

## Known bug recurrence

- **Arabic "municipality of citizenship" (بلدية) — confirmed NOT recurring.** The
  reviewing agent explicitly checked all 43 `country of citizenship` rows in this
  batch; all correctly use بلد. The prior fix holds under fresh, independent review.
- **"Place/work read as a language" — recurs, and is broader than previously
  documented.** Confirmed again in sw/ja/zh (the known pattern: a name with no
  target-language Wikidata label gets rendered as "the X language," then asked
  its own official language). New in this batch: Swahili shows the same reversed
  construction on **P407 (language of work/name)**, not just P37 (official
  language) — 14 of Swahili's 20 `type` flags are this pattern across both
  relations. Previously only documented as a P37 bug.
- **Truncated Spanish questions — recurs.** 2 more instances in this batch
  (rows 50, 245 of `census_validation_es_00.md`), consistent with the class found
  in the test census.

## New finding: Korean-script (Hangul) contamination

Not caught by any prior review pass because it's orthogonal to the subject/type/
relation rubric — the subject and property are usually still correct, only a
Korean-origin proper name or work title is left in raw Hangul instead of being
transliterated into the target script (e.g. `Q495870`'s `ja`/`zh` questions ask
about "오윤교" instead of a transliterated form). Two reviewing agents independently
flagged the same fact_id as `unsure` for exactly this reason, which prompted a
full-dataset mechanical scan (Hangul is not a legitimate character range in any of
this dataset's 12 languages, so any hit is unambiguous — zero false-positive risk).

**49 (fact, language) hits across the whole 59,291-fact release**: 47 in train
(~40 distinct facts, several hit in 2+ languages), 2 in validation (1 fact,
`Q495870`, both `ja` and `zh`), 2 in test (`Q7640060` in `fr`+`zh`, `Q2743574` in
`bn`). All traced to Korean-origin subjects/titles (e.g. `오윤교`, `권정혁`,
`일성록`/Ilseongnok, `트롤 슬레이어`/Troll Slayer) whose name/title failed to
transliterate.

Scope is small (49/711,492 language-rows ≈ 0.007%) and the fix pattern already
exists (`make_retranslation_files.py` → per-language subagent → gated
`apply_full_census.py`-style acceptance). Given the 3-day submission deadline,
recommend: fix the 2 test-split and 1 validation-split facts now (cheap, and
these are the reviewer-visible splits), document train's ~40 facts as a known
residual defect class in the dataset card rather than fixing before the deadline.

## Recommendation

- Validation census is now complete and at parity with test's rigor.
- Fix-vs-flag decision for the 173 translation defects + 9 English conflicts
  follows the same policy already established for test (retranslate confirmed
  question defects via the gated pipeline; flag English conflicts for editorial
  review; no automatic drops). Not yet applied — pending a go-ahead, given
  competing deadline priorities.
- The Hangul-contamination fix (2 test + 1 validation fact) is small enough to
  bundle into that same fix pass.
