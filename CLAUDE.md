# CLAUDE.md

Code for the paper "Improving Cross-Lingual Factual Recall via Consistency-Driven
Reinforcement Learning" (ACL ARR 2026 May, submission 8689, arXiv 2606.06586).
Compares CPT / SFT / GRPO for cross-lingual factual recall on OLMo-2-1124-7B and
Qwen-2.5-7B, using the PolyFact dataset (100K Wikidata facts × 12 languages).

## Environment

- **AWS Trainium instances (trn1/trn2)**: run `bash setup_trainium.sh` once on a
  fresh instance (idempotent; handles Ubuntu 26.04 / kernel 7.0 quirks including
  a Neuron driver source patch). Then `source ~/neuron_venv/bin/activate`
  (sets `PJRT_DEVICE=NEURON`) and pass `--device xla` to eval scripts.
- Neuron caches **failed** compilations in `/var/tmp/neuron-compile-cache` —
  after fixing a compiler error, delete that directory or the error replays.
- GPU machines: the same scripts run with `--device cuda` (auto-detected).

## Datasets (HF hub, account `jvonrad`)

- `jvonrad/WIKI-FACT` — nested schema: one row per fact, `langs` dict with
  per-language `question` / `options` / `answer_text`. Used by the eval scripts.
- `jvonrad/PolyFact` — same facts, per-language configs (`option_a..d`,
  `answer_index`) plus a `parallel` config. This is the paper-facing release.
- Languages: en, de, id, pt, ar, bn, sw, es, ru, fr, ja, zh. Test split: 2,523
  usable facts. Options are 4 MCQ candidates (1 gold + 3 distractors), same
  entities across languages but **independently shuffled per language, with no
  stored distractor entity ids**.

## Evaluation

- `evaluate/evaluate_accuracy.py` — per-language accuracy only (logprob scoring
  of the 4 options, `--score_mode avg` = length-normalized; prompt is
  `Question: {q}\nAnswer:`). Renamed from `evaluate_consistency.py` because it
  never measured consistency.
- `evaluate/evaluate_crosslingual_consistency.py` — accuracy **plus** the
  consistency metrics reviewers asked for: Total Consistency (correct in ALL 12
  languages + histogram), RankC (Qi et al. EMNLP 2023; softmax-weighted top-j
  overlap per language pair), pairwise answer agreement. Supports
  `--benchmark polyfact` and `--benchmark global_mmlu` (Global-MMLU joins
  languages on `sample_id`, options parallel by index).
- **Always pass `--alignment_cache evaluate/alignments/polyfact_test_alignment.json`
  for PolyFact test.** RankC needs distractors aligned across languages; this
  committed cache (exact match + LaBSE assignment, 0 unaligned) makes numbers
  comparable across model variants. Recomputing it may produce slightly
  different alignments.
- Results JSONs include raw per-option scores, so new metrics can be computed
  from saved output without re-running the model.

## Running evals on Trainium (batch-size ceiling + two-model parallelism)

**Batch-size ceiling on XLA/Neuron — this is a correctness trap, not just perf.**
`score_candidates_batch` upcasts the full-vocab logits to fp32 for a logsumexp.
At `--batch_size 16` the resulting `[batch*4, seq, ~100k]` fp32 tensor exceeds a
Neuron memory limit and the runtime returns **silently corrupted logits** (no
error): every option scores ~equally and accuracy collapses to chance (~0.25 on
4-way MCQ). At `--batch_size 16 --max_length 256` it crashes outright. **Use
`--batch_size 8` or less on `--device xla`** — 8 is verified numerically
identical to batch 4 (en=0.828 on 128 facts) and ~2× faster. `--max_length 192`
is safe: the longest tokenized `Question:/Answer:` + option over the entire test
split is 168 tokens, so nothing truncates, and it trims padding vs the 512
default. If a run reports near-chance accuracy on a model you expect to work,
suspect batch size before the model or the alignment cache.

**trn2.3xlarge has 2 usable logical NeuronCores** (`logical-neuroncore-config 2`
pairs physical cores 0-1 and 2-3, ~48 GB each). A default `--device xla` process
grabs the **whole** device, so a second process fails `nrt_init()` with
`NRT_FAILURE`. To run two 7B models at once, pin each to a logical core with
`NEURON_RT_VISIBLE_CORES`:

```bash
source ~/neuron_venv/bin/activate          # sets PJRT_DEVICE, adds neuron-ls/neuron-top to PATH
COMMON="--benchmark polyfact --hf_dataset jvonrad/WIKI-FACT --split test \
  --batch_size 8 --device xla --max_length 192 \
  --alignment_cache evaluate/alignments/polyfact_test_alignment.json"

NEURON_RT_VISIBLE_CORES=0-1 setsid nohup python -u evaluate/evaluate_crosslingual_consistency.py \
  $COMMON --model MODEL_A --output_json results/MODEL_A_polyfact_consistency.json \
  > results/MODEL_A_polyfact_consistency.log 2>&1 < /dev/null &

NEURON_RT_VISIBLE_CORES=2-3 setsid nohup python -u evaluate/evaluate_crosslingual_consistency.py \
  $COMMON --model MODEL_B --output_json results/MODEL_B_polyfact_consistency.json \
  > results/MODEL_B_polyfact_consistency.log 2>&1 < /dev/null &
```

Notes for a clean parallel run:
- Every forward pass is padded to one fixed `[batch*4, max_length]` shape, so all
  OLMo-2-7B variants share a single compiled graph (cache key ignores weights).
  Warm it once (any short run at the same batch/max_length) and both parallel
  processes hit the cache with **zero** compilation and no write race.
- **Never `kill` a process mid-compilation.** A killed compile can leave a
  truncated entry in `/var/tmp/neuron-compile-cache` that later runs load as
  garbage. If that's suspected, `rm -rf /var/tmp/neuron-compile-cache` and let it
  recompile. (Neuron also caches *failed* compiles there — same fix.)
- Sanity-check the first `acc so far` line per job (expect >0.6 for `en` on these
  models) before trusting a long run; chance-level early means stop and check
  batch size.
- `neuron-top` (live per-core util/mem) and `neuron-ls` (which PID owns which
  core) are the `nvidia-smi` equivalents; both are on PATH after activating the
  venv.

## Session findings (2026-07-11/12, trn2.48xlarge)

- **HBM compile limit is 24 GB per compiled graph on trn2**, not the ~48 GB a
  paired logical core nominally has — `neuronx-cc` rejects graphs above 24 GB
  peak (`NCC_EOOM002`). Qwen (152k vocab) OOMs at batch 2 / max_length 1024;
  OLMo (100k vocab) fits. Verified-fitting Global-MMLU-Lite configs:
  **OLMo `--batch_size 2 --max_length 1024`, Qwen `--batch_size 2 --max_length 768`**.
- `--benchmark global_mmlu_lite` (CohereLabs/Global-MMLU-Lite) added to
  `evaluate_crosslingual_consistency.py`. Lite has **no Russian** config — pass
  `--langs en,de,id,pt,ar,bn,sw,es,fr,ja,zh` (11 langs). 400 fully parallel
  facts/lang, split `test`. Long Bengali tail (max ~2.8k tokens): ~1.5 % of
  prompts right-truncate at these max_lengths (model-agnostic, affects all
  models equally).
- On trn2.48xlarge (`logical-neuroncore-config 2`): 16 devices × 4 physical
  cores = 32 logical cores; pin jobs with `NEURON_RT_VISIBLE_CORES=0-1`, `2-3`,
  … `62-63`. Root volume is only ~7 GB — mount an instance-store NVMe and point
  `HF_HOME`, `UV_CACHE_DIR`, `TMPDIR`, and the Neuron cache
  (`NEURON_CC_FLAGS=--cache_dir=...`) at it (done in `~/neuron_venv/bin/activate`).
- **GRPO training is not runnable on Trainium**: on-policy generation is
  structurally blocked (per-token recompilation in manual fixed-shape decode;
  HF `generate` emits ops neuronx-cc lacks — `sort` from `torch.isin`, int64
  `dot` — and with `--disable-hlo-operand-type-check=evrf_035` one generate
  graph compiled >30 min without finishing). transformers-neuronx supports no
  Qwen/OLMo arch; NxD-Inference supports qwen2 but would need a two-model
  loop with per-step weight sync. Run the bonus ablation on GPU.
- `--all_correct_bonus` in `train_wikifact_grpo_accelerate.py` was **dead code**
  (bonus hardcoded `+1.0`); now wired, default 1.0 preserves paper behavior.
  Ablation: `--all_correct_bonus 0.0` vs `5.0` (launch commands in chat/rebuttal
  notes; needs CUDA GPUs, e.g. 4×A100 with `--use_lora`).
- **Checkpoint discrepancy (unresolved)**: all committed "OLMo GRPO" consistency
  results evaluate `jvonrad/OLMo-2-7B-grpo`, but the README model list says the
  paper's GRPO is `jvonrad/olmo-2-7b-grpo-att-mlp-full` (both repos exist). This
  may explain OLMo SFT > GRPO on PolyFact in the new tables (paper claims the
  reverse). Re-evaluate att-mlp-full before quoting OLMo GRPO numbers.
- `data_analysis/significance_analysis.py` recomputes per-fact metrics from the
  saved score JSONs (validated against reported aggregates) and produces paired
  fact-level bootstrap CIs / p-values → `results/significance/significance_report.md`.
  Headline: OLMo GRPO on Global-MMLU-Lite improves RankC +3.3pp and agreement
  +4.2pp (p≈1e-4) with flat accuracy; Qwen GRPO on PolyFact +2.4–4.0pp on all
  metrics (p≈1e-4).

## Known inconsistencies to resolve

- The paper appendix says the PolyFact evaluator wraps questions in a
  language-specific instruction prompt; the actual scripts use the plain
  `Question:/Answer:` wrapper. Reconcile before the rebuttal.
- README references `requirements.txt`, which does not exist in the repo.

## Rebuttal work outstanding (from ACL ARR reviews, July 2026)

Done: direct consistency metrics (RankC + total consistency + agreement) for
all 12 models on PolyFact AND Global-MMLU-Lite (results/*_consistency.json);
bootstrap CIs + paired significance tests (results/significance/). Still requested by reviewers: CLC-enhancement baselines (DCO,
EN-pivot DPO / CM-Align, representation intervention), a GRPO ablation without
the all-language consistency bonus, expanded related work (Qi et al. 2023,
Fierro & Søgaard 2022, X-FACTR, Paths Not Taken), training/token cost
reporting, and confidence intervals for the small Global-MMLU deltas.
