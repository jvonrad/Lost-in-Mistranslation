# Handoff: full strict-prompt test census (completed)

> **Completed 2026-08-01.** All 84 worksheets passed structural validation; the
> strict census report is in `results/full_test_census_report.md`. English
> conflicts were adjudicated in `results/english_conflict_adjudication.json`,
> gated repairs are summarized in `results/full_census_apply_report.json`, and
> the final 59,291-fact build passed `build-poly-fact/verify_v13.py`. PolyFact-Clean
> v13 was uploaded to `jvonrad/PolyFact-Clean`; remote README and parallel-test
> parquet checksums matched the verified local build exactly.

Written because the current session is running out of tokens mid-task. Everything
needed to resume is below and on disk — nothing is in this session's memory only.

## What this is

PolyFact-Clean's test split was rebalanced (`build-poly-fact/rebalance_splits.py`,
already run) from 1,503 to **2,523 facts** (moved 1,020 facts out of validation,
which shrank from 1,464 to 444), because a bootstrap power analysis on the paper's
real GRPO-vs-SFT predictions showed the accuracy delta (+0.78pp) was one cleaning
pass away from losing significance at the old size. That rebalance is DONE and
verified (zero integrity errors, fact set unchanged, zero cross-split overlap) —
see `results/split_rebalance.json` for the exact fact_ids moved.

The rebalanced test set at `$SCRATCH/v13` (data-only, no metadata patches applied
yet) needs a **full strict-prompt census** — not a sample — because an earlier
controlled comparison found the ORIGINAL test review prompt undercounted defects
by ~1.4x. This full census is currently ~40% complete via parallel subagents.

## Where everything is

- **Data to review**: `$SCRATCH/v13` (rebalanced, not yet fixed) — parallel +
  12 per-language parquet configs, same schema as the published release.
- **Worksheets**: `$SCRATCH/fullcensus/census_*.md` — 84 files total:
  - `census_{lang}_{00..06}.md` for lang in [de, id, pt, ar, bn, sw, es, ru, fr, ja, zh]
    (11 langs × ~7 parts = 77 files, ~400 items each, ~27,753 items total)
  - `census_en_gold_{00..06}.md` (7 files, ~2,523 items total)
  - `$SCRATCH/fullcensus/index.json` lists every worksheet with its `out` path
- **Generator** (already run, re-run only if you need to regenerate):
  `data_analysis/make_full_test_census.py --data_dir $SCRATCH/v13 --out_dir $SCRATCH/fullcensus`

## Two review tasks — read the .md file's own header for full instructions

1. **Translation review** (11 languages): does the target-language question ask
   about the same subject and property as the English one? Verdicts: `ok`,
   `subject` (severe — different entity), `type` (wrong kind of thing), `relation`
   (wrong property), `unsure`. Each worksheet's own header has language-specific
   calibration notes (Russian case-inflection, Bengali script-fusion, Swahili's
   `Ki-` fake-language-prefix bug, etc.) — READ THE FILE, don't guess the rubric.

2. **English-vs-gold review** (`en_gold` files): would this English question, AS
   WRITTEN, have the stored gold as its answer? This is a fact-checking task, not
   a translation task — the model sometimes invents a year/medium/location that
   contradicts its own gold (e.g. "the 1979 film Darr" when Darr is from 1993).
   Verdicts: `ok`, `conflict`, `vague`, `unsure`. Measured previously on a 600-fact
   sample at 3.0% conflict rate [95% CI 1.9-4.7%] — expect a similar rate here.

## How to check what's done vs still needed

```bash
comm -23 <(ls $SCRATCH/fullcensus/census_*.md | sed 's#.*/##;s/\.md$//' | sort) \
         <(ls $SCRATCH/fullcensus/census_*.json 2>/dev/null | sed 's#.*/##;s/\.json$//' | sort)
```
This lists worksheet names with NO corresponding verdict JSON yet — i.e. what
still needs a reviewer. As of this handoff, **33 of 84 are done**; run the command
above for the live count, don't trust this number as time passes.

Each worksheet's own "Write JSON to" line names its exact output path
(`$SCRATCH/fullcensus/census_{X}.json`) — always write there, one JSON array of
`{"fact_id": ..., "verdict": ..., "note": ...}` per row, ALL rows in the file, in
order. `note` only required for non-`ok` rows.

## After every worksheet has a verdict JSON

1. **Aggregate.** Adapt `data_analysis/aggregate_validation_review.py` (or write
   a new small script) to read `$SCRATCH/fullcensus/index.json` +
   `$SCRATCH/fullcensus/census_*.json` the same way it currently reads
   `valreview/index.json` + `valverdicts_*.json` — the JSON shapes are identical,
   only the file-naming pattern differs (`census_{lang}_{part}.json` vs
   `valverdicts_{lang}.json`). Report defect rate per language + English-conflict
   rate, both with Wilson CIs (helper already in that script).

2. **Decide what to fix vs flag.** Established policy this whole build (see the
   dataset card sections already written — read
   `build-poly-fact/upload_polyfact_clean.py`'s `CARD` string for the full
   rationale and precedent):
   - Confirmed WRONG LABELS → drop the entity globally (pattern:
     `results/manual_review_drops.json` → `build-poly-fact/resample_distractors.py
     --drop_entities`). Already done for earlier rounds; only new confirmed-wrong
     labels found in `type`/`subject` notes above would need this, and there
     shouldn't be many since labels were already reviewed exhaustively in an
     earlier pass (7,708 labels, v10).
   - Confirmed WRONG QUESTIONS (this census's `subject`/`type`/`relation`) →
     **do NOT auto-drop.** Precedent: `build-poly-fact/apply_retranslations.py`
     re-translates flagged (fact,lang) pairs using per-language subagents with
     corpus style anchors, then gates every proposal on leakage / distractor /
     script / length checks before accepting. Reuse that script's pattern
     (`make_retranslation_files.py` → per-language agent → `apply_retranslations.py`)
     for whatever this new census confirms, restricted to the fact_ids not
     already covered by the previous round's `results/retranslation_report.md`.
   - ENGLISH-vs-GOLD conflicts (a NEW class, never fixed anywhere yet) → these
     need editorial judgment (fix the English question, or fix/flag the Wikidata
     triple if the gold itself is wrong — one case, `Q6195056`, was found to have
     a wrong gold in the validation sample). No automated fix exists; this is
     the one part of the pipeline requiring a human/agent to actually decide per
     item, not just apply a mechanical gate.
   - Set `question_verified` (bool, nullable) and `question_regenerated` (bool)
     columns on the final build exactly as `apply_retranslations.py` already does
     — see that script for the exact schema-patching logic (it's not trivial:
     `translations.<lang>` is a NESTED struct-of-structs in the `parallel` config;
     see the comment in that file about the bug where grabbing the OUTER struct
     type instead of the per-language inner struct silently produced a malformed
     schema).

3. **Build v13 final + upload.** Once fixes are applied to `$SCRATCH/v13`:
   ```bash
   # verify integrity (adapt the check block from any of the v10/v11/v12 build
   # steps in this session's history — option_ids / answer_index consistency,
   # zero leakage via the `leaks()` function in apply_retranslations.py, fact-set
   # unchanged vs v13 pre-fix)
   python build-poly-fact/upload_polyfact_clean.py --clean_dir <final> \
     --repo_id jvonrad/PolyFact-Clean --dry_run   # inspect README.md first
   python build-poly-fact/upload_polyfact_clean.py --clean_dir <final> \
     --repo_id jvonrad/PolyFact-Clean              # then actually upload
   ```
   Update the dataset card (`CARD` string in `upload_polyfact_clean.py`) with:
   the new split sizes (train 56,324 / validation 444 / test 2,523), the full
   census's measured defect rate (replacing/supplementing the old 2.8%/sampled
   numbers), and the English-vs-gold conflict rate as its own documented section
   (there is precedent text for this already in the CARD from the validation-
   sample finding — extend it, don't replace it, since that section explains
   the discovery process).

## Known systematic bugs already confirmed by this session (context, not new work)

These recur across languages and MOST worksheets in this batch have already
re-confirmed them — don't be surprised to see them again, they're not each a new
discovery:
- **Place read as a language**: obscure villages/places with no non-English
  Wikidata label get a fabricated "X language" reading (`Ki-` prefix in Swahili,
  spurious `语`/`語` suffix in Chinese/Japanese, `-язык` pattern in Russian).
- **Proper name meaning-translated**: brand/studio/place names translated by
  dictionary meaning instead of transliterated/kept (`DIET`→"diet", `Woosten`→
  "born in", `Septemvri`→"the month September").
- **"Located" → "born"/"founded"/"lived"**: relation drift where a place's
  location question gets reworded as if the place were a person or company.
- **Work-type collapse**: albums/episodes/series/talks/poems relabeled as
  "film" or "book" by default.
- **Arabic municipality-of-citizenship bug**: already FIXED in a prior round
  (`build-poly-fact/repair_arabic_citizenship.py`, 784 items) — should NOT
  recur, but if any worksheet flags `بلدية` on a `country of citizenship`
  question, that's a regression worth flagging loudly.
- **Truncated Spanish questions** (new this round): a few Spanish questions cut
  off mid-sentence with no actual interrogative (e.g. "...es un nombre de
  origen…"). Treat as `unsure`/defective, not `ok`.

## Cost/scale reference

77 translation worksheets × ~400 items + 7 english-gold worksheets × ~360 items
≈ 30,276 total judgements. At ~300-400 items/agent this is what's already been
scoped into 84 worksheets — no further splitting needed, just get every worksheet
reviewed. Session's concurrency cap observed at 20 simultaneous subagents.
