# CLAUDE.md

Code for the paper "Improving Cross-Lingual Factual Recall via Consistency-Driven
Reinforcement Learning" (ACL ARR 2026 May, submission 8689, arXiv 2606.06586).
Compares CPT / SFT / GRPO for cross-lingual factual recall on OLMo-2-1124-7B and
Qwen-2.5-7B, using the PolyFact dataset (100K Wikidata facts × 12 languages).

## Environment

### Isambard-AI phase 2, project `u6sg` (current platform, from 2026-08-01)

Successor to the old `u6jh` allocation — same cluster family, **new account and
new paths**, and a fresh home with nothing pre-installed. Budget: ~500 GH200
hours.

- Account `brics.u6sg`, partition `workq`, 1320 nodes × 4× **GH200 120 GB**
  (97.8 GB usable, sm_90, aarch64, 288 cores, 460 GB host RAM). Driver
  565.57.01 = CUDA 12.7, so **cu126 wheels** are the safe build.
- `cluster/env.sh` is the single source of truth for account, paths, caches and
  conda activation. Source it at the top of every job. Two directories, never
  interchangeable: `REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation` (git
  checkout = code) and `PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation`
  (Lustre data/outputs: `datasets/`, `models/`, `logs/`). Sending checkpoints to
  `$REPO/models` will blow the /home quota.
- **`/home` has a hard 101 GB quota**; Lustre `/projects` and `/scratch` do not.
  `HF_HOME=/projects/u6sg/jvonrad.u6sg/hf_cache`, `TMPDIR`/`PIP_CACHE_DIR` on
  `$SCRATCH` — all set by `cluster/env.sh` and by a conda `activate.d` hook.
- **Login nodes cap each user at a 4 GiB memory cgroup**
  (`/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max`). Anything that
  loads a 7B model, or even a parallel `snapshot_download`, dies with a bare
  `Killed` and no traceback. Do model downloads and every python job on a
  compute node — see `cluster/prefetch_models.sbatch`.
- **Compute nodes have outbound internet** (hf.co and pypi.org both reachable
  from `nid*`), unlike the u6jh setup. Do *not* blanket-set `HF_HUB_OFFLINE=1`;
  models missing from the cache just download on the node.
- **A CPU-only job (`--gpus=0`) sits in PENDING on workq** — it is a GPU
  partition. Use `--gpus=1` even for pure-download or orchestrator jobs.
- **In an sbatch payload `$0` is the spool copy** under
  `/var/spool/slurmd/job<N>/`, so `source "$(dirname "$0")/env.sh"` fails with
  "No such file". Source `cluster/env.sh` by absolute path. (Several ported
  u6jh scripts had this latent bug.)
- QOS wall limits: `workq_qos` 24 h, `interactive_qos` 8 h, `restricted48`
  48 h. Long GRPO runs still need `--resume_from_checkpoint auto` + resubmit.
- Env: `~/miniforge3`, conda env **`grpo`** — python 3.11, torch 2.12.1+cu126,
  transformers 5.13.1, accelerate 1.14.0, peft 0.20.0, datasets 5.0.1,
  sentence-transformers 5.6.1. `trl` is *not* installed (so
  `training/train_polyfact_dapo.py` is not runnable; the CM-Align DPO trainer
  is hand-rolled and does not need it).
- Job scripts: `cluster/prefetch_models.sbatch` (cache all 12 checkpoints +
  datasets), `cluster/eval_sweep.sbatch` (array job, one model per GPU,
  `BENCHMARK=polyfact|gmmlu_lite|klar`), `cluster/models.txt` (the canonical
  12-model list consumed by the sweep).

### Legacy platforms

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
-`jvonrad/PolyFact-Clean` — curated dataset excluding noisy and hallucinated labels 
and properties. **Should be used for eval**
- Languages: en, de, id, pt, ar, bn, sw, es, ru, fr, ja, zh. Test split: 2,523
  usable facts. Options are 4 MCQ candidates (1 gold + 3 distractors), same
  entities across languages but **independently shuffled per language, with no
  stored distractor entity ids**.

## Training data: PolyFact-Clean (switched 2026-08-02)

All training scripts now default to **`jvonrad/PolyFact-Clean`, config
`parallel`** (was WIKI-FACT / PolyFact).

- **A config name is mandatory.** PolyFact-Clean exposes 13 configs
  (`ar`…`zh` + `parallel`) with **no default**, so a bare
  `load_dataset("jvonrad/PolyFact-Clean")` raises. Every trainer has a
  `--dataset_config` (SFT: `--hf_dataset_config`) defaulting to `parallel`;
  pass `''` for single-config datasets like WIKI-FACT.
- Splits: **train 56,324 / validation 444 / test 2,039**. Measured: 100 % of
  rows parse and 100 % carry all 12 languages, so `--min_languages 12` discards
  nothing and the usable training set is the full 56,324. (Test is 2,039 — the
  "2,523 usable facts" figure elsewhere in this file is WIKI-FACT's.)
- **Two schemas, one normaliser.** `training/polyfact_schema.py` is the single
  place that knows about either shape; all trainers call it. Do not re-implement
  this parsing per script — that duplication is how the RankC letter bug below
  survived.
  - PolyFact-Clean/PolyFact: `row["translations"][lang]` with
    `option_a..option_d`, `answer_index`, `option_ids`. `answer_index` is an
    **int** in both splits (verified 2026-08-02; an earlier note here claimed
    string — that was a display artifact of a str()-ed debug print). The
    schema normaliser's int() coercion is harmless and stays as belt-and-braces.
  - WIKI-FACT (legacy, still supported): `row["langs"][lang]` with an
    `options` list and no answer index.
  - Gold resolution prefers `answer_index`, falls back to matching
    `answer_text`, and trusts `answer_text` if the two ever disagree.
- Every `cluster/` launcher pinned `--dataset_id jvonrad/WIKI-FACT` explicitly,
  which would have silently overridden the new default — all were repointed. If
  you add a launcher, either omit `--dataset_id` or pass the config too.

## Training-time metrics (reworked 2026-08-02)

Periodic eval (`run_full_eval`, every `--eval_steps`) now reports metrics whose
definitions match `evaluate/`, under a `polyfact/` namespace (was `wikifact/`):

| metric | meaning |
|---|---|
| `polyfact/freeform_accuracy` | greedy generation + string-match to an option (renamed from `slot_accuracy`) |
| `polyfact/freeform_total_consistency` | generation-based, correct in ALL 12 langs (was `all_correct_rate`) |
| `polyfact/mcq_accuracy` | **log-likelihood MCQ accuracy** — the `evaluate_accuracy.py` metric |
| `polyfact/mcq_total_consistency` | LL-based, correct in ALL 12 langs |
| `consistency/rankc_avg[_en_x]` | RankC over the 4-option ranking |
| `mmlu/total_consistency`, `mmlu/rankc_avg[_en_x]` | Global-MMLU cross-lingual metrics |

- The `mcq_*` metrics and Global-MMLU consistency are **free**: they reuse
  forward passes that were already being run and discarded (RankC computed all
  4 option scores then kept only the ranking; the MMLU loop kept only `argmax`).
- `mcq_*` deliberately uses the **eval** prompt `Question: {q}\nAnswer:` with
  options hidden — NOT the training prompt, which lists A–D and therefore
  measures a different task (choosing among shown options vs closed-book
  recall). This is why `build_grouped_fact_item` now stores `question` in the
  per-language meta; that changed the dataset-map fingerprint, so the map
  recomputes once.
- Scoring is **byte-normalized** (lm-eval `acc_bytes`), matching
  `evaluate_accuracy.py`'s current default. RankC used per-token `avg` while
  the eval default moved to byte, so the two were not comparable.

### Three bugs fixed here — read before trusting old runs

1. **RankC ranked the LETTERS, not the answers.** `meta["options"]` is a
   `Dict[str, str]` keyed `"A".."D"`, so `enumerate(meta["options"])` iterated
   *keys*: the function scored `" A"/" B"/" C"/" D"` and pooled hidden states
   over those tokens, making the cross-lingual alignment meaningless too. Every
   `consistency/rankc_avg` logged before 2026-08-02 is invalid.
2. **Global-MMLU monitoring loaded `split="test"`** despite the function being
   named `load_global_mmlu_dev_eval_by_lang` — training watched, and implicitly
   selected checkpoints on, the reported split. Now `dev` (285 rows/lang vs
   14,042), which also removes most of the periodic-eval cost.
3. **Languages were not joined.** The loader took `range(max_samples)` per
   language independently; cross-lingual metrics require the *same* items, so
   it now joins on `sample_id` (warns + falls back to positional if absent).

**In-loop RankC exact alignment (added 2026-08-02, live from chunk c2 of the
production chains):** `build_grouped_fact_item` now stores `option_ids` in
the per-language meta, and `compute_polyfact_logprob_metrics` logs
`consistency/rankc_exact_avg[_en_x]` (QID-aligned, model-independent — the
trustworthy in-loop signal) ALONGSIDE the legacy hidden-state-aligned
`consistency/rankc_avg[_en_x]` (kept for curve continuity), plus
`consistency/alignment_agreement` = fraction of option pairings where the
hidden-state matcher agrees with ground truth. Declining agreement with a
flat exact-RankC = alignment drift, not consistency loss; this decides the
ambiguity in e.g. the Qwen rankc dip observed early in the 40k run. The
meta change bumps the dataset-map fingerprint (one recompute at c2 start).

`training/train_wikifact_grpo.py` (the older non-accelerate trainer) still has
the old `wikifact/` keys and the same dict-iteration pattern — it was left
alone; sync it before using it for anything.

## GH200 GRPO throughput/memory pilot (2026-08-02, Qwen-2.5-7B + LoRA, 2×GPU DDP)

Measured with the new `perf/step_time_sec` / `perf/peak_mem_gb` instrumentation
(mean over post-warmup optimizer steps; `cluster/bs_pilot.sh`,
`cluster/gen_pilot.sh`, English scaffold, G=8, `logprob_micro_batch_size 48`):

- **Batch size is NOT a throughput lever.** bs=1: 5.68 s/step (2.84 s/fact,
  51 GB peak); bs=2: 10.69 s/step (2.67 s/fact, 67 GB); **bs=3 OOMs** ("95.00
  GiB total capacity" — the GH200 "120GB" name counts Grace LPDDR; usable HBM
  is 95 GiB). Only ~6 % per-fact gain from doubling, because generation and the
  loss pass are both compute-saturated and linear in sequence count; only fixed
  per-step overhead amortizes.
- **`--num_generations` cost is linear** (bs=1): G=4 2.79 s / 35.8 GB, G=8
  4.81 s / 44.0 GB, G=16 9.13 s / 60.4 GB. No amortization — each extra rollout
  costs its full share. Completions average ~33 of the 48-token cap regardless
  of G (the base model rambles; only the first line reaches the reward).
- **Degenerate-group rate (all G rollouts same reward ⇒ zero gradient) was NOT
  properly measured** — 0 occurrences in 3 post-warmup steps per config is not
  a rate. reward_std per step ranged 0.75–2.22. A real measurement needs ~50
  steps (~5 min on 2 GPUs).
- **Where the memory goes:** `compute_logprob_loss` materialises logits
  [mb, T, 152k] AND a same-size log_softmax copy per micro-batch, and retains
  ALL micro-batch graphs until the single backward — so the term scales with
  batch size and `logprob_micro_batch_size` does not cap it. T is the padded
  length: tokenizer fertility spans **bn ≈ 307 tokens vs en ≈ 73** for the same
  fact, so unsorted micro-batches pad to ~445 when the mean is ~164 (≈2.7×
  wasted FLOPs+memory). A padded-to-max model reproduces the measured peaks
  almost exactly; padded-to-mean would put bs=3 at ~42 GB.
- Mitigations implemented as trainer flags and measured head-to-head
  (`cluster/throughput_pilot.sh`, 1 GPU each, bs=1, G=8, NATIVE scaffold):

  | variant | s/step | peak GB |
  |---|---|---|
  | control | 11.14 | 70.4 |
  | **`--length_bucketing`** (+ gen chunks of 32) | **7.51** | **59.8** |
  | `--fused_logprob` | 11.33 | 70.3 |
  | both + `logprob_micro_batch_size 96` | 9.16 | 91.3 |

  Verdict: **adopt `--length_bucketing`** (−33 % step time, −10.6 GB; changes
  no math — sorts gen prompts and loss sequences by length so each micro-batch
  pads to its own max). `--fused_logprob` is a NO-OP for memory: log_softmax
  retains its *output* for backward while cross_entropy retains the *logits* —
  one same-sized [mb,T,vocab] tensor either way (the flag stays, harmless and
  exact in fp32, but don't expect anything from it). mbs 96 is actively bad
  WITH bucketing — a single 96-seq chunk pads to the global max, defeating the
  sort AND nearly OOMing; keep `logprob_micro_batch_size` ≤ 48. Lowering mbs
  alone (without bucketing) does NOT reduce peak memory either — all
  micro-batch graphs are retained until the one backward.
- **The native prompt scaffold (`--prompt_scaffold native`, default since
  2026-08-02) costs ~2× step time vs the English scaffold** (control 11.1
  s/step vs 5.68 measured pre-switch at otherwise-equal config) — localised
  scaffold text tokenizes far longer in bn/ar/etc. Bucketing recovers most of
  it (7.5 s). Scaffold choice is a throughput knob, not just a science knob.
- **Degenerate-group rate at G=8 is 0/50** (52-step run, native scaffold,
  strict matcher, base Qwen, bs=1/rank): no step had reward_std==0 (min 0.63,
  mean 1.44) — i.e. no zero-gradient steps at training start, so there is
  currently NO gradient-quality argument for bigger batches or G>8, and bs is
  no throughput lever either. Caveat: measured at init; the rate will rise
  late in training as all-correct groups appear. Reward mean was −1.49 —
  base Qwen fails most languages under the native scaffold + no-letter-credit
  matcher, leaving GRPO plenty of signal.
- **With bucketing, the bs ceiling moves 2 → 3** (native scaffold, 2-GPU DDP,
  mbs 48): bs=1 7.84 s/step (3.92 s/fact, ~60 GB), bs=2 14.21 (3.55, 70.5),
  bs=3 20.62 (3.44, 85.8), bs=4 OOM. Throughput still ~flat per fact.
- **Recommended production config**: `--per_device_train_batch_size 1
  --num_generations 8 --length_bucketing --gen_micro_batch_size 32
  --logprob_micro_batch_size 48` (+ defaults). At ~3.9 s/fact, one epoch over
  the 56,324-fact PolyFact-Clean train split on a 2-GPU DDP pair ≈ 61 h wall
  ≈ **122 GPU-h** (needs 3 resubmits under the 24 h QOS cap; budget was ~500).
- Cluster ops: hold a placeholder allocation (`cluster/interactive_job.sbatch`,
  `srun --jobid=<id> --overlap ...`) instead of queueing per experiment —
  exclusive-node jobs can pend ~1 h while `--gpus=1` jobs start in seconds.
  Chain follow-up allocations with `sbatch --dependency=afterany:<jobid>`.
  Warm the datasets `.map` cache once before launching parallel runs (the map
  fingerprint includes the `scaffold` fn_kwarg — changing `--prompt_scaffold`
  recomputes it).

## Per-language GRPO baseline (meta-review request, implemented 2026-08-02)

`--reward_pooling {group,per_lang}` in `train_wikifact_grpo_accelerate.py`
(default `group` = paper behaviour, byte-for-byte: pooled rewards replicated
per language leave every logged statistic unchanged).

- The cross-lingual coupling being ablated is TWO mechanisms, and `per_lang`
  removes both: (1) the reward pooled over all 12 languages (+bonus), and
  (2) the z-scored advantage broadcast to every language's generation — which
  gives e.g. Swahili gradient signal even when all G Swahili rollouts were
  wrong, as long as the group outcome varied.
- `per_lang`: each (fact, lang) is its own GRPO group; reward = own-language
  correctness only (+1 / −0.5 unparseable / 0); advantage z-scored within that
  (fact, lang) over G, applied only to its own generation. `all_correct_bonus`
  is undefined here and ignored (startup warning). `group_rewards` is now keyed
  `(fact_idx, gen_idx, lang)` in BOTH modes.
- Expected mechanism difference (log it, don't be surprised by it): per-language
  rewards are near-ternary, so hard languages produce all-identical reward
  groups → std=0 → zero gradient. `degenerate_frac` in group_stats measures it.
  (The pooled method measured 0/50 degenerate steps at init.)
- Launch: `sbatch cluster/perlang_baseline.sbatch` (Qwen, 40k facts, 2 GPUs,
  ≈44 GPU-h; config matches the nobonus/bonus5 ablation family). Three-rung
  dissection: full (pooling+bonus) / nobonus (pooling only) / per_lang
  (neither). CAVEAT: post-2026-08-02 runs use PolyFact-Clean + native scaffold
  + no-letter-credit matcher — compare against a matched re-run of the pooled
  method, not the old WIKI-FACT-era checkpoints.

## Hyperparameter sweep: LoRA rank × learning rate (`cluster/hp_sweep.sh`, 2026-08-02)

Successive halving instead of a full-length grid (which would cost 9×44 ≈ 396
GPU-h ≈ most of the budget): rung 1 = all 9 configs (r ∈ {64,128,256} × lr ∈
{3e-6,1e-5,3e-5}, α=2r, kl_coef FIXED 0.0 to match the ablation family) at
1,500 facts, submitted as independent 1-GPU jobs, --time 4:30 (~28 GPU-h);
rung 2 = top-3 fresh at 5,000 facts (~31 GPU-h; use a non-exclusive
--gpus=3 allocation — on an exclusive node the 4th GPU is billed idle);
rung 3 = winner as a clean full 40k 2-GPU run (44 GPU-h). ≈103 GPU-h total. Design points:
- All configs share one seed → identical fact order → paired comparison.
- Rungs re-run fresh rather than resuming rung-1 checkpoints (cross-rung resume
  semantics with a changed max_train_samples are fragile; re-simulating 1,500
  facts wastes only ~12 GPU-h total).
- Ranking = last periodic-eval `polyfact/mcq_accuracy` + `mcq_total_consistency`
  (`hp_sweep.sh report` prints the table). Noise floor at 150-fact eval ≈ ±1 pp:
  treat closer configs as ties, prefer smaller r / smaller lr.
- Short-horizon caveat: the LARGEST lr tends to look best early and degrade or
  destabilise late — that's what rung 2's longer horizon and the
  `freeform_resolution_rate` collapse signal are for. With kl=0 there is no
  KL anchor, so watch resolution rate as the drift alarm.
- **RESULT (rung 1, 2026-08-02): winner r=128, lr=1e-5.** Exact tie with
  r256/lr1e-5 on mcq_acc (0.6017 both, base ≈ 0.55-0.58); r256's +1.3pp
  TotCons is inside the ±2pp noise floor and r128 leads RankC, so the
  pre-registered tie rule (smaller rank) applies. lr=1e-5 dominated every
  rank; lr=3e-5 degraded with rank and at r256 COLLAPSED to chance with
  resolution_rate 0.0 (killed mid-run). Rung 2 was skipped (user call, time)
  — the 1e-5-vs-3e-6 ordering is the horizon-stable part, so the risk is
  contained. Full 40k production run launched directly with the winner
  (job 5870736, 4-GPU DDP ≈ 22 h wall, one QOS window; effective optimizer
  batch is 4 facts/step — keep all post-2026-08 family runs at 4 GPUs).
- New trainer flags this needed: `--lora_r` / `--lora_alpha` (were hardcoded
  64/128) and **`--ref_impl adapter_off`** — the KL reference is the policy
  with its LoRA adapter disabled instead of a second loaded copy of model_id.
  Proven exact (max |Δlogits| = 0 vs a separately loaded base). Saves ~15 GB,
  which is what lets 4 KL configs share a node. `separate` stays the default.

## DCO baseline (`training/train_polyfact_dco.py`, ported 2026-08-02)

Reviewer-requested CLC baseline: "Post-Training Language Models for
Crosslingual Consistency" (Liu, Qi, et al., ICML 2026;
github.com/Betswish/ConsistencyRL). Their stack pins torch 2.7.1 (no
aarch64/cu126 wheel exists) + a bundled trl fork, so the METHOD was ported
into this repo instead of running their trainer:

- **Objective is label-free consistency, not correctness**: an instance is the
  same fact in two languages with chosen/rejected candidates picked BY SHARED
  INDEX — the chosen candidate is RANDOM, not gold. Loss (their dco_loss,
  copied and verified exact to 0.0 against their source via ast-extraction):
  `|reward_1 − (1/β)·offset_2| + |reward_2 − β·offset_1|`, sequence logps =
  SUM over completion tokens. This is the pure-consistency contrast to GRPO's
  joint consistency+correctness reward.
- **PolyFact-Clean is a perfect data fit**: `option_ids` give the exact
  cross-language candidate index alignment the method requires (lang2 options
  reordered into lang1's index order by QID; verified 0 mismatches on real
  data). Instances built in-process, seeded — no generation phase, no
  rollouts, so ~1-1.5 s/step at bs 4 → 40k instances ≈ 3-5 h on ONE GPU
  (~an order of magnitude cheaper than a GRPO arm).
- Deviations from their setup, to state in the paper: LoRA r128/α256 (they
  full-FT; keeps the post-2026-08 run family comparable), adapter-off
  reference (exact, no second model), 12-language random pairs (they mostly
  train en–X), eval-style prompt `Question: {q}\nAnswer:`.
- Faithful bits: β=1.0, lr 1e-5, bs 4, 1 epoch, random-candidate chosen
  (their sampling.py logic), sum-logps convention.
- Launched 2026-08-02: `qwen-dco-n40000-r128`, `qwen-dco-n5000-r128` (their
  paper's 5k scale), `olmo-dco-n40000-r128` — 1 GPU each.
- **RESULTS (PolyFact-Clean test, 2026-08-02)** — DCO is a strong baseline:
  qwen-base 0.513 acc / 0.054 TotCons / 0.624 RankC → dco-40k **0.569 /
  0.143 / 0.698** (every language up; dco-5k already 0.563/0.110/0.674 for
  10 min of training). OLMo: acc ~flat overall (0.444→0.451) but **en
  −5.7pp** while low-res gains — consistency via homogenization, direction
  depends on base asymmetry; TotCons 0.017→0.073, RankC 0.573→0.649.
  The GRPO bar on Qwen is therefore 0.569/0.143 at a 70× cost handicap
  (DCO-40k = 1.2 GPU-h). Caveats for the writeup: DCO trains on the exact
  eval prompt + option-text completions (part of the MCQ gain may be
  distribution sharpening, not knowledge — weigh KLAR heavily), and it is
  label-free (can homogenize toward wrong answers; the OLMo en drop is the
  evidence). Result JSONs: results/{qwen,olmo}-{base,dco-*}_polyfact_clean_
  consistency.json.
- The per-language GRPO baseline runs were CANCELLED in favour of this
  (user call): `--reward_pooling per_lang` stays implemented+tested in the
  trainer but has no completed runs.
- Login-node gotcha rediscovered the hard way: torch defaults to 288 threads
  on the Grace CPU and `backward()` thrashes inside the 4 GiB cgroup —
  export OMP_NUM_THREADS=8 for any CPU-side test.

## CLC baseline results, new regime (2026-08-02) — the bar GRPO must clear

All same-device (GH200), PolyFact-Clean test / KLAR / Global-MMLU-Lite.
Cross-device reproducibility confirmed: fresh GH200 KLAR base numbers match the
committed Trainium ones to 3-4 decimals (qwen 0.4282 vs 0.4285).

**DCO** (label-free consistency, 1.2 GPU-h): huge in-domain MCQ gains, but they
do NOT transfer to generation or out of domain.
| metric | qwen base→dco | olmo base→dco |
|---|---|---|
| PolyFact acc / TotCons / RankC | .513→.569 / .054→.143 / .624→.698 | .444→.451 / .017→.073 / .573→.649 |
| KLAR clean-all | .420→**.409 (−1.1pp)** | .185→.202 (+1.7pp) |
| G-MMLU-Lite acc / RankC | .636→.635 / .698→.704 | .445→.453 / .550→.555 |
DCO's MCQ gain is largely option-ranking sharpening under the eval format
(it trains on exactly that format). It HURTS generation on the strong
multilingual model (Qwen) and helps only the weak generator (OLMo .18 base).
Consistency gains are domain-bound: +7.4pp RankC in-domain → +0.5pp OOD.

**CM-Align** (EN-pivot DPO): the strongest generation baseline, and it
replicates across BOTH families with near-full transfer to unseen languages
(17-lang KLAR, 7 seen / 10 OOD):
| model | full | seen-7 | OOD-10 |
|---|---|---|---|
| qwen base→cmalign | .405→.494 (**+8.9**) | .477→.568 (+9.1) | .358→.445 (**+8.7**) |
| olmo base→cmalign | .179→.252 (**+7.3**) | .246→.333 (+8.7) | .133→.198 (**+6.5**) |
Nearly every language improves (qwen vi +31, ko +17, ar +17). No seen/OOD gap
for Qwen — a genuinely language-agnostic improvement, not leaked content.

Implication for the paper: DCO owns cheap in-domain MCQ consistency;
CM-Align owns generation transfer. GRPO's claim must be BOTH — in-domain
consistency competitive with DCO *and* KLAR gains in CM-Align's league — or
the 70x compute premium (87 vs 1.2 GPU-h) is hard to defend.

## SAME-PIPELINE METHOD COMPARISON (2026-08-02) — the decisive table

All methods re-evaluated on ONE pipeline (PolyFact-Clean/byte/exact-QID,
KLAR 6-lang, G-MMLU-Lite, BMLAMA-53). Deltas vs each model's OWN base.
GRPO here = the PAPER's existing checkpoints (old training regime); the
new-regime runs are separate.

**Qwen — GRPO's thesis is confirmed:**
| method | PF acc | PF TotCons | PF RankC | KLAR clean | BMLAMA acc |
|---|---|---|---|---|---|
| GRPO (Qwen-2.5-7B-grpo-consistent) | +4.7 | +2.3 | +1.8 | **+2.1** | +0.4 |
| DCO | +5.6 | **+9.0** | **+7.4** | **−1.1** | +3.5 |
GRPO is the ONLY method improving consistency AND generation. DCO wins
consistency ~4x over but LOSES generation. Reproduces the earlier Trainium
finding (+2.3pp clean-shared) on different hardware and pipeline.

**OLMo — GRPO improves consistency but DEGRADES generation:**
| method | PF acc | PF TotCons | PF RankC | KLAR clean |
|---|---|---|---|---|
| GRPO (olmo-2-7b-grpo-att-mlp-full) | +3.5 | +2.8 | +1.7 | **−4.3** |
| DCO | +0.7 | **+5.6** | **+7.6** | +1.7 |

### ⚠ ACTION ITEM: which checkpoint produced the paper's OLMo KLAR row?

CORRECTED 2026-08-02 (an earlier note here wrongly blamed the protocol):
**the KLAR protocol reproduces EXACTLY.** The paper's KLAR aggregation is
`evaluate_klar.py` over ALL 17 KLAR languages, split 7 "Seen"
(en,es,fr,ru,zh,ja,ar) / 10 "OOD" (ca,el,fa,he,hu,ko,nl,tr,uk,vi):

| | Seen | OOD |
|---|---|---|
| paper baseline | 24.6 | 13.3 |
| our olmo-base, 17-lang split | **24.56** | **13.30** |

Beware `evaluate_klar.py`'s DEFAULT `--langs` is only 6 languages
(es,fr,ru,zh,ja,ar — **no English**), which gives 18.28 for the same base
model. Comparing that against the paper's 7-language "Seen" is apples/oranges
and produced a false "protocol mismatch" alarm. **Always pass the explicit
17-language list when comparing to the paper's KLAR table.**

With base + protocol + aggregation all confirmed identical, the discrepancy is
isolated to the CHECKPOINT: paper GRPO = 29.0/16.7, but
`jvonrad/olmo-2-7b-grpo-att-mlp-full` measures **20.49/10.65** (a LOSS vs
base). The 2026-07-12 identification of that repo as "the paper's GRPO" was
made by matching PolyFact deltas only — KLAR was never used to confirm it.
**HUNT RESULT (17-lang KLAR, base reproduces to 0.04pp so this is signal):**
| checkpoint | Seen | OOD | Δ base |
|---|---|---|---|
| paper's GRPO row | 29.0 | 16.7 | +4.4 / +3.4 |
| `olmo-2-7b-grpo-att-mlp-full` (README GRPO) | 20.49 | 10.65 | **−4.07 / −2.65** |
| `OLMo-2-7B-grpo` ("the weaker run") | 26.60 | 15.79 | **+2.03 / +2.49** |

**HUNT COMPLETE — the two checkpoints have EXACTLY COMPLEMENTARY profiles;
neither is positive throughout:**
| OLMo ckpt (Δpp vs base) | PFacc | PFtotc | PFrkc | KLseen | KLood | GMacc | GMrkc |
|---|---|---|---|---|---|---|---|
| `olmo-2-7b-grpo-att-mlp-full` | **+3.45** | **+2.75** | **+1.73** | −4.07 | −2.65 | −0.52 | +1.40 |
| `OLMo-2-7B-grpo` | **−0.75** | **−0.29** | **−0.28** | **+2.03** | **+2.49** | −0.48 | **+3.48** |

att-mlp wins PolyFact & loses KLAR; the alt wins KLAR & loses PolyFact. Each
matches exactly ONE row of the paper's table — the pattern you would expect if
the paper's OLMo row was assembled from TWO different runs. Only OLMo is
affected; the Qwen story reproduces cleanly and is internally consistent.

**⚠ COMPARABILITY CAVEAT (do not repeat this mistake):** only KLAR can be
compared to the paper's numbers at all, because only there does our base
reproduce the published baseline (24.56/13.30 vs 24.6/13.3). The paper's
PolyFact row is **WIKI-FACT with a High/Low resource split**, and its
Global-MMLU row is **full Global-MMLU**, whereas we measure **PolyFact-Clean
12-lang mean** and **Global-MMLU-Lite**. Different datasets AND different
aggregations — e.g. our olmo-base G-MMLU-Lite is 44.45% vs the paper's
38.72/31.79, which is a dataset difference, NOT a discrepancy.

**The two checkpoints disagree in SIGN, and the paper's row matches the one the
2026-07-12 note said NOT to quote.** That identification used PolyFact deltas
only. Neither reproduces 29.0/16.7 exactly (26.60 is 2.4pp short — too large
for noise at this protocol's precision). So either the paper's table mixes
checkpoints across benchmarks, or a third (unuploaded) checkpoint exists, or
the KLAR row used different decoding. NEEDS AUTHOR RECORDS (W&B/run logs) —
further compute cannot resolve it. Do not quote an OLMo GRPO KLAR number until
it is settled.

**Recommended framing**: lead with Qwen ("GRPO is the only method improving
cross-lingual consistency AND factual generation; consistency-only baselines
each move one modality only"), and report OLMo's generation regression as an
honest model-dependence limitation rather than burying it.

## BMLAMA-53 evaluation (added 2026-08-02)

`--benchmark bmlama53` in `evaluate_crosslingual_consistency.py`. RankC's native
benchmark (Qi et al. EMNLP 2023), so RankC gets its real dynamic range here:
`rankc_pair` now derives n from the data (identical at n=4; at n=10 the
most-disagreeing pair scores 0.003 vs the RankC@4 floor of 0.0902).
- **11 of our 12 languages** — BMLAMA-53 has no `sw` config (warns + drops).
  3,036 of 3,070 facts kept; pools are index-parallel across languages
  (verified 0 gold-idx/pool-size mismatches), so alignment is identity.
- **Scoring adaptation, documented**: 27% of templates have a NON-final
  `<mask>` ("X is <mask> citizen."), so we split at the mask and score
  `" candidate" + suffix`. Suffix is constant within an item → byte
  normalisation stays fair; preserves grammatical-agreement signal. The
  released DCO probe assumes BMLAMA-17's pre-stripped prompts + a vLLM
  perplexity path, so it is not directly reusable.
- **Contamination vs PolyFact-Clean train** (`data_analysis/bmlama_contamination.py`,
  `results/contamination_bmlama.json`): only **2.2%** exact (subject,object)
  overlap, 1.7% strict triple, and 86% of items use relations absent from
  PolyFact — close to a clean OOD generalisation test.

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
- **Cross-lingual option alignment — read this before choosing flags.** RankC
  and answer agreement need each language's 4 options matched to English.
  - **PolyFact-Clean: pass NOTHING.** Its `parallel` config stores `option_ids`
    (the Wikidata QID of every option, per language), so the evaluator derives
    the alignment *exactly* via `align_by_option_ids`. Verified over all 2,523
    test facts: 4 distinct QIDs, identical set across all 12 languages,
    `option_ids[answer_index] == object_id` everywhere.
  - **Do NOT pass `--alignment_cache .../polyfact_test_alignment.json` with
    PolyFact-Clean.** That cache was built for WIKI-FACT. Curation resampled
    distractors and reshuffled options, so it is now wrong: only 1,503/2,523
    fact_ids appear in it at all, and of the mappings that do, just
    2,142/18,036 (11.9%) match the true QID alignment. It fails silently and
    produces garbage RankC/AnsAgr.
  - WIKI-FACT has no option ids, so it still needs the cached
    string-match + LaBSE alignment. `--no_option_id_alignment` forces that old
    path for parity with pre-2026-08 runs.
- **`--score_mode` now defaults to `byte`** (was `avg`) in both
  `evaluate_accuracy.py` and `evaluate_crosslingual_consistency.py`:
  `sum` (lm-eval `acc`) / `avg` (per-token mean) / `char` (per-character =
  lm-eval `acc_norm`) / `byte` (per-UTF-8 byte = lm-eval `acc_bytes`).
  Results JSONs save the raw option logprob **sum** plus the option token
  count, so **all four modes are recomputable post-hoc without re-running a
  model** (`derive_scores`); `data_analysis/rescore_results.py --mode X` rewrites
  saved JSONs in place, which is how a sweep whose runs used different modes is
  made uniform.
  - Byte is the default because it is tokenizer-independent, which matters when
    per-language numbers are compared side by side. Token-mean divides by a
    count ranging from 2.75 (en) to 17.44 (bn) tokens per option, and the
    spread of that divisor across the four competing options is 2-4.6x larger
    in bn/ja/zh than in English — an uneven length penalty landing hardest on
    exactly the low-resource languages.
  - Note for framing: lm-eval's own defaults are **not** tokenizer-dependent
    (`acc_norm` divides by characters; `acc` is a string logprob), so published
    low-resource gaps measured with lm-eval are not a normalization artifact.
    The artifact is specific to token-mean. Tokenizer fertility still hurts
    low-resource languages, but through model quality, not the eval metric.
- **AnsAgr is computed but no longer reported in the tables.** RankC's w_1 is
  0.644, so RankC is ~2/3 answer agreement — they are near-duplicate metrics,
  not independent evidence. It stays in the result JSONs. Dropping it saves no
  compute (same pairwise loop as RankC).
- **RankC here is RankC@4, not Qi et al.'s number.** With only the 4 MCQ
  candidates, top-4 sets always overlap fully and top-3 sets always overlap
  >= 2/3, so RankC cannot go below **0.0902**; two independent random rankings
  score **0.3768** (both brute-force verified over all 4!x4! ranking pairs).
  Reported values sit on that pedestal and deltas are compressed by x0.91 —
  do not compare them to Qi et al., who rank a large candidate pool. The
  output JSON carries `rankc.floor`, `rankc.chance` and
  `rankc.average_rescaled`. Note also w_1 = 0.644, so RankC is ~2/3 just answer
  agreement: the two metrics are **not independent evidence**.

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
- **Checkpoint discrepancy (RESOLVED 2026-07-12)**: the paper's OLMo GRPO is
  `jvonrad/olmo-2-7b-grpo-att-mlp-full` (as the README says) — its PolyFact
  delta-vs-base (+3.0 avg12, +3.5 TotCons, p=1e-4) matches the paper's WIKI-FACT
  GRPO uplift, and it beats SFT on all metrics. `jvonrad/OLMo-2-7B-grpo` is a
  different, weaker run (flat/below base) — its results
  (`results/OLMo-2-7B-grpo_*`) should NOT be quoted as the paper's GRPO.
  `significance_analysis.py` and the report now use att-mlp-full.
- `data_analysis/significance_analysis.py` recomputes per-fact metrics from the
  saved score JSONs (validated against reported aggregates) and produces paired
  fact-level bootstrap CIs / p-values → `results/significance/significance_report.md`.
  Headline (with the correct att-mlp-full GRPO checkpoint): on PolyFact, GRPO is
  the only method improving ALL consistency metrics, every GRPO−Base and
  GRPO−SFT delta significant (p≤0.002; TotCons +4.16pp over SFT); SFT raises
  accuracy (+2.25pp) but NOT total consistency (−0.71pp n.s.) — the
  "accuracy-without-consistency" contrast used in the rebuttal. Qwen replicates
  (+2.4–4.0pp, p≤1e-4). On Global-MMLU-Lite, GRPO accuracy is unchanged
  (−0.9pp, CI [−3.1,+1.3]) and consistency is directionally positive but n.s.
  (RankC +1.2pp) — do NOT claim significant OOD consistency gains.
- **KLAR generation eval runs on Trainium** (`evaluate/evaluate_klar.py`,
  `--device xla`): its `_greedy_xla` uses a fixed prompt window (`--max-length
  448`) and one compiled graph per decode step (`--max-new-tokens 10` → 10
  graphs, compile-once then cached for every model of the same family). The
  int64-dot compiler error (`NCC_EVRF035`) it hits is fixed by appending
  `--disable-hlo-operand-type-check=evrf_035` to `NEURON_CC_FLAGS` (safe here:
  the int64 dot is over position indices ≪ 2^24, exactly representable in fp).
  This flag does NOT rescue HF `generate()` (its graph compiles >30 min,
  unusable) — only this manual fixed-shape loop.
- `evaluate_klar.py --tokenizer` used to default to the OLMo tokenizer instead
  of `--model` (silent garbage for Qwen); fixed to default to the model. Pass
  `--contamination-labels evaluate/alignments/klar_polyfact_contamination.json`
  and `--output-json` for per-sample records; aggregate clean-vs-contaminated
  splits across models with `data_analysis/klar_contamination_split.py`
  (overall / contaminated / clean-shared / non-shared subsets, from
  `results/klar/*_klar.json`).
- KLAR clean-subset motivation: 157/1,207 KLAR facts in PolyFact-shared
  relations have their exact triple in PolyFact-train (13.0% of shared, 6.0% of
  all 2,619) — reviewers asked whether GRPO's KLAR gains survive on the
  non-overlapping subset.
- **KLAR 12-model results (2026-07-12, done)**: `results/klar/<model>_klar.{json,log}`
  + `results/klar/contamination_split_report.txt` (6 langs es,fr,ru,zh,ja,ar;
  3-shot template 1; greedy 10 tok; bs16). Headlines: contaminated subset is
  clearly inflated (Qwen base 55.1 vs 42.5 clean-shared). **Qwen GRPO gains
  survive decontamination** (vs base: clean-shared +2.3pp, non-shared +1.5pp;
  vs SFT: +2.7pp on both clean subsets). **OLMo CPT+GRPO survives strongly**
  (vs base: clean-shared +8.3pp, non-shared +2.7pp). CAVEAT: plain OLMo GRPO
  (att-mlp-full) scores BELOW base on KLAR here (14.1 vs 18.2 overall) —
  opposite of the paper's KLAR table; protocol differs from the paper's KLAR
  eval (template/n-shot/matching), so reconcile before quoting OLMo plain-GRPO
  KLAR numbers. OLMo SFT's KLAR drop (14.2 vs 18.2) replicates the paper's
  direction; Qwen CPT+SFT collapses to 12.5 (worth a look).
- `evaluate_klar.py` `_greedy_xla` was rewritten (2026-07-12) to compile ONE
  graph per model family: position_ids precomputed on CPU (no on-device cumsum
  → no int64 dot, no compiler flag needed) and per-step logit-read/token-write
  done via gather/scatter with runtime index tensors (no Python-int positions
  baked into the graph). Compile is ~4 min/family at bs16 (bs32 exceeds the
  24 GB HBM ceiling). The earlier per-step-graph variants compiled ~50 min PER
  STEP — never revert to Python-int indexing in this loop.

## GRPO rollout generation speed (2026-07-15, GH200)

- **Root cause of slow GRPO: rollouts generated with NO KV cache.** The model
  has `gradient_checkpointing_enable()` set; even though `generate_grouped_rollouts`
  calls `model.eval()`, transformers still refuses the default cache during
  rollouts, so decoding is quadratic. Fix: pass `cache_implementation="static"`
  to `model.generate()`, which forces a real (preallocated) KV cache back on.
- **New flag `--gen_cache_implementation` (default `static`)** in
  `train_wikifact_grpo_accelerate.py`, threaded into both the rollout path
  (`generate_grouped_rollouts`) and periodic eval (`evaluate_wikifact_grouped`).
  Pass `none` for the old dynamic/no-cache behavior. Existing launch commands
  need no change — they pick up the speedup automatically.
- **Measured (Qwen-2.5-7B + LoRA, batch 96 = 1 fact × 8 gens, real trainer code
  path, isolated GH200):** generation `none` 3.80s → `static` 1.98s (~1.9×).
  Standalone micro-bench at the same batch: 4.17s → 2.06s. Greedy output is
  **byte-identical** static-vs-dynamic on both Qwen and OLMo-2-1124-7B (0/24
  mismatches each) — safe, no quality change. The `` `use_cache` incompatible
  with gradient checkpointing`` warning that still prints comes from the
  training forward pass, NOT generation — ignore it.
- Throughput is **batch-starved at 96 sequences** (GH200 idle): with static
  cache, 96→2.06s, 192→3.66s, 384→7.03s (seq/s 46→52→55). Raising
  `per_device_train_batch_size` to 2 (→192 prompts/gen-call) is a further ~1.4×
  on generation, but it also doubles the optimizer batch (changes training
  semantics / the paper's "1 fact/step") — only do it deliberately.
- **vLLM: not adopted.** vLLM 0.25.1 (latest aarch64 wheel) hard-pins
  `torch==2.11.0`, but the `grpo` env runs torch 2.12.1 — co-installing would
  downgrade torch and break transformers 5.13 / the aarch64 build. That forces a
  separate-env vLLM *server* + per-step policy-weight sync (the heavyweight
  verl/TRL pattern), whose sync overhead for these tiny 96×≤48-token rollouts is
  exactly why an earlier in-place vLLM attempt "wasn't quicker". Static cache
  gives the 2× with zero new deps, in-process. Revisit vLLM only if moving to
  much larger rollout batches/longer completions where its continuous batching
  would dominate the sync cost.
- **The per-step compute is otherwise near-optimal.** Profiled (Qwen-7B, 96
  rollouts, static-cache gen): gen 2.1s / logprob-fwd 1.3s / backward 2.8s /
  ~6.0s total, 36 GB peak. Backward dominates via gradient-checkpointing
  recompute, but every attempt to cut it lost: grad-ckpt-off OOMs (the loss fn
  retains all 96 sequences' autograd graphs before one backward); restructuring
  to backward-per-micro-batch + no-ckpt fits only at mbs≤16 and is *slower*
  (6.98–7.79s — small batches underutilize the GPU worse than recompute costs);
  bigger logprob micro-batch (48→96) and length-sorting change nothing (the loss
  pass is compute-bound). Grad checkpointing at mbs 48–96 is the right tradeoff;
  leave it on. The loss pass is inherent GRPO cost (full 7B fwd+bwd over all
  rollouts) — no free lunch there.
- **Periodic-eval cost knob `--max_eval_mmlu` (default 1000).** The MMLU eval
  scores `max_eval_mmlu × 12 langs` examples every `--eval_steps` (200) — 12k
  forward passes/eval at the default, the dominant periodic cost, and it does
  NOT affect training. Drop to ~200 (or raise `--eval_steps`) to reclaim
  wall-clock with zero risk.
- **Bigger batch (`--per_device_train_batch_size 2`) is a *modest* lever, not the
  1.4× it looks like from gen-throughput alone.** Measured (Qwen, static cache):
  batch-2 = 11.59s/2 facts = **5.80s/fact vs 6.04s/fact at batch-1, ~4% faster**
  — only generation amortizes (2.07→1.84s/fact), the loss pass is compute-bound
  and linear (unchanged). Verified it runs clean (peak 48 GB). It doubles the
  optimizer batch vs the paper's 1-fact/step, so it's opt-in — keep default 1.
- **/home is chronically near its 101 GB quota** (~85 GB is the HF model cache).
  A `pip install` defaults its wheel cache to `~/.cache/pip` on /home and can
  push it over quota (breaks file writes with `EDQUOT`). Set
  `PIP_CACHE_DIR`/`TMPDIR` to `$SCRATCH` for any big install, and
  `rm -rf ~/.cache/pip` to reclaim.

## PolyFact-Clean evaluation sweep (2026-08-01, trn2.3xlarge)

`bash evaluate/run_polyfact_clean_sweep.sh` evaluates all 14 checkpoints (the
12 README models + `OLMo-2-7B-CM-Align` / `Qwen-2.5-CM-Align`) on
PolyFact-Clean test, writing `results/polyfact_clean/<tag>_polyfact_clean.json`.
It runs two lanes — OLMo on `NEURON_RT_VISIBLE_CORES=0-1`, Qwen on `2-3` — so
each family compiles its own graph once in its own lane with no compile-cache
write race. Re-running skips any model whose JSON already exists.

`python data_analysis/aggregate_polyfact_clean.py` turns those JSONs into
`results/polyfact_clean/report.md`: the Acc/TotCons/RankC/AnsAgr table plus
per-language accuracy, each under all four scoring rules, and repeated on the
`n_langs_verified == 12` subset — all recomputed from saved scores, no GPU.

Facts about the split, verified 2026-08-01: **2,523 facts, every one present in
all 12 languages**, no malformed entries, and `max_length 192` still truncates
nothing (longest tokenized prompt+option is 180 for OLMo, 162 for Qwen).
Only **1,269/2,523** facts have `n_langs_verified == 12`, which is why the
aggregator reports that subset separately.

Fresh-instance setup notes (bare Ubuntu 26.04 trn2, not a Neuron DLAMI):
- `setup_trainium.sh` aborted under `set -u` because `LD_LIBRARY_PATH` is unset
  on a bare AMI; the appended activate lines now use `${LD_LIBRARY_PATH:-}`.
- Root volume is ~7 GB. Mount the instance-store NVMe (`/dev/nvme1n1`, ~430 GB,
  ships unformatted) at `/mnt/nvme`, put the venv there, and export `HF_HOME`,
  `UV_CACHE_DIR`, `TMPDIR` and `NEURON_CC_FLAGS=--cache_dir=...` from the venv's
  `activate`. Without `HF_HOME` the model cache silently targets `~/.cache` and
  fills the root disk after two 7B checkpoints.
- `uv venv` refuses a path that already exists, so a pre-created symlink at
  `~/neuron_venv` breaks the script — create the venv at its real NVMe path
  first, then symlink.
- `--batch_size > 8` on `--device xla` now hard-errors instead of silently
  returning corrupted logits.

## Question-quality auditing: what works and what does NOT

A manual spot-check (26 facts x 12 langs, `results/polyfact_clean_spotcheck_report.md`)
found a defect class no existing audit catches: **the question names the right
subject but means the wrong thing**. Examples: bn asks about the PlayStation
"Qore" *service* rather than the Qore programming language; the novel "Voices"
becomes the common noun in id/ar/sw/zh ("what language does the voice use?");
a long parody title is regenerated per language into three different films.
Root cause is that questions were **generated per language**, not translated.

**Embedding-based detection does not work for this — two attempts, both
measured against 8 hand-confirmed bad instances, both failed. Do not rebuild
them** (`data_analysis/detect_question_outliers.py` keeps both for the record):
- **Mean LaBSE cosine to the 11 parallel questions: 1/8.** The question
  template is identical across languages by construction, so cosine tracks the
  relation, not the entity — a right-relation/wrong-entity question still
  scores ~0.85. The Voices ar/zh cases sat at the 24th-28th percentile.
- **Bitext margin + retrieval rank (Artetxe & Schwenk style): 2/8**, at a 36%
  fact flag rate concentrated in zh/ja/ar (the rank criterion is not
  per-language normalized, so it re-introduced the low-resource bias the margin
  z-score was designed to avoid). Two separate failure modes: *vague but still
  nearest* — against only 2,523 candidates even a subject-less question is
  uniquely closest to its own fact (retrieval_rank 0 for Voices id/zh, Qore
  bn); and *sharp but wrong* — Q7711911 ru scored margin_z **+2.07**, better
  than average, because it is precisely about the wrong film. A parody and its
  target are near-identical in embedding space by construction.

**Use the LLM judge instead** (`data_analysis/judge_question_equivalence.py`):
one Batch API request per fact showing the triple plus all 12 questions,
structured-output JSON verdict per language. Run `--self_test` first — it
judges only the hand-reviewed facts and scores recall against the same 8
instances, so the judge is validated before spending on the full split.
Needs `pip install anthropic` and `ANTHROPIC_API_KEY` (or `ant auth login`).
Batch API is 50% off: ~$11 for 2,523 facts on `claude-opus-5`, ~$4.50 on
`claude-sonnet-5`. Capability matters more than price here — world knowledge is
exactly what the embedding detectors lacked.

**Judge results, full test split (2026-08-01, `results/item_quality_judge.json`)**
— 2,523 facts, `claude-opus-5`, effort=medium, Batch API. 4,354 issues over
1,802 facts; **544 facts (21.6%) carry a high-confidence defect**. High-conf
counts: question 502, validity 357, label 187 (of which **76 gold-label**).
Top problems: wrong_entity 247, type_leakage 166, answer_leakage 165,
fabricated_detail 134, garbled-label 85. High-conf issues are **3x more common
in sw (147) and zh (143) than de (49)** — data quality is not uniform across
languages.

**KEY ROBUSTNESS RESULT — the High/Low-resource gap is NOT a data artifact.**
Recomputing per-language accuracy with every high-confidence defective item
dropped moves the gap by **-0.03pp (OLMo base)** and **-0.24pp (Qwen base)**;
no per-language accuracy shifts by more than 0.7pp. The language-correlated
defect *rate* is real, but the absolute rate (5.7% of items in sw vs 1.9% in
de) is far too low to move a ~14pp gap. This is the answer to the obvious
reviewer question and it is free to recompute for any model, since result JSONs
store per-fact scores.

**COST LESSON — estimate from a probe, not from arithmetic.** The run cost
**$35.02**, not the ~$19 estimated: input was **3,331 tokens/fact**, not 1,450.
Two causes, both avoidable: (a) the system prompt (~800 tok) and JSON schema
(~500 tok) are re-sent on *every* request — ~40% of the input bill for bytes
that never change; (b) content was under-counted because 48 option strings
across 12 languages tokenize far worse than English (bn is ~17 tok/option, a
number measured earlier in the same session and not applied). Before any future
batch: run ~50 requests, read actual `usage`, extrapolate — and put
`cache_control` on the system block (fixed prefix x 2,523 requests is the
textbook prompt-caching case, worth ~$5-8 here).

**CURATED TEST SET (2026-08-01):**
`evaluate/alignments/polyfact_clean_test_droplist.json` — **484 facts dropped,
2,039 kept**. Criteria: any high-confidence judge issue, MINUS the 117/302
`wrong_entity` flags rescued because the "wrong" name is an official Wikidata
label/alias for that language (the judge's one systematic FP class — Alexander
Neufeld IS the German name of Sándor Nemes). Precision evidence: 28-flag
stratified human recheck (26-27 correct; contested cases adjudicated against
Wikidata). `aggregate_polyfact_clean.py` now emits CURATED tables automatically
(`--drop_list ''` to disable). Composition caveat: drops concentrate in
title-bearing relations (director 36%, language-of-work 31%, discoverer 50%)
vs citizenship 6.7% — the curated set tilts toward proper-noun relations.
Effect on finished models: acc ±0.7pp, TotCons −0.2-0.4pp, High/Low gap
+0.5-0.9pp (dropped leakage items were easy everywhere, so removing them
slightly WIDENS the gap). Paper framing: report full 2,523 AND curated 2,039;
the near-identical numbers are the robustness argument.

**PUSHED TO HUB 2026-08-01 (user-approved, mid-sweep):** commit `88a05d08`,
built by `build-poly-fact/curate_test_release.py` (--stage validates the whole
tree — counts, cross-config fact_id identity, flag removal, option_ids — then
--push uploads as ONE commit). Test is now 2,039 everywhere; the
`question_verified`/`question_regenerated` columns were REMOVED from all
configs (nested-struct removal needs `map(..., features=rebuilt)` — map alone
null-fills, staging validation caught this); droplist ships in the dataset at
`curation/test_droplist.json`. Pre-curation revision pinned at
`evaluate/alignments/polyfact_clean_precuration_revision.txt` (`c89817f8`);
`aggregate_polyfact_clean.py` resolves gold/alignment at that pinned revision
and guards mixed coverage.

**SWEEP RESTARTED on the curated set (user request, 2026-08-01 23:13):** all
14 models re-run uniformly on the 2,039-fact v2 test. The 4 completed full-set
(2,523) results were archived to `results/polyfact_clean_precuration/`
(olmo_base, olmo_cpt, qwen_base, qwen_cpt) — aggregate them with
`--results_dir results/polyfact_clean_precuration` for the full-vs-curated
robustness comparison on those models. No recompilation on restart: the
compiled graph shape `[batch*4, max_length]` is dataset-size-independent.
Curated numbers from a native 2,039 run are IDENTICAL to filtering a 2,523 run
by the droplist (per-fact scores are independent) — the restart buys uniform
provenance, not different numbers.

Also unresolved: **`question_verified` and `n_langs_verified` contradict each
other** in both directions (one fact has 12 true but `n_langs_verified=9`;
another has 11 false but `n_langs_verified=11`). 1,194/30,276 (3.9%) question
instances are `question_verified=false`, and that set is enriched for genuinely
broken questions — but at least one badly broken fact carries all-true flags.
Pin down the semantics before either field is used as a filter or cited.

## TRAINER BUGS FOUND 2026-08-02 — read before trusting any run

Three defects in `train_wikifact_grpo_accelerate.py`, all now fixed. Any run
started before 2026-08-02 evening is affected by #1.

**1. The cosine LR schedule was consumed `num_processes` times too fast.**
`total_update_steps` is computed AFTER `accelerator.prepare()`, so it is already
the per-rank (sharded) count — but `prepare(scheduler)` wraps it in
`AcceleratedScheduler`, whose `.step()` calls the inner scheduler
`num_processes` times because it assumes the schedule was built against the
UN-sharded count. Both corrections applied to the same quantity. On a 4-GPU run
the lr hit **exactly 0.0 at step 2500 of 10000** and stayed there — 75% of the
run trained with no learning at all. Verified by reconstruction: predicted lr
matched the logged value to 4 significant figures at every step.
Fix: scale `num_warmup_steps`/`num_training_steps` by `accelerator.num_processes`.

**2. The KL penalty was ANTI-regularising.** The old k1 estimator
`(logpi - logpi_ref).mean()` has no attractor toward the reference — minimising
it just suppresses `logpi` on the policy's own samples, and since the samples
come from the policy the estimate runs away negative. Measured: KL reached
**-2.44 at kl_coef 0.02 and -8.85 at 0.05 within 50 steps**, scaling with the
coefficient, i.e. the "penalty" had become a bonus for divergence. Replaced with
Schulman k3 (`r = logpi_ref - logpi; exp(r) - r - 1`), which is provably >= 0.
Verified post-fix: 0/43 negative values across three arms, and the dose-response
inverted correctly (looser clip drifts more; stronger anchor suppresses it).
**`--kl_coef` default changed 0.05 -> 0.0.** Every science launcher in
`cluster/` passes `--kl_coef 0.0` explicitly (audited), so no published result
is affected; only `pilot_node.sbatch` omitted it, and that measured throughput.

**3. Resume did not skip forward.** `global_step` was restored but the
dataloader restarted at batch 0 of a reshuffled epoch, so a resumed run
re-walked facts it had already trained on (~75% distinct-fact coverage over a
3-chunk chain). Fixed with a seeded generator + `set_epoch` +
`accelerator.skip_first_batches`. The seeded generator is load-bearing: with
`shuffle=True` and no explicit generator the permutation comes from the GLOBAL
RNG, which sampling-based rollout generation advances unpredictably, so the
order was not reproducible and no skip could have landed on the same data.

Also: `--max_grad_norm` was hardcoded to 1.0 (now a flag), and `grad_norm` is
printed to stdout (it was wandb-only).

## Optimizer steps are wall-clock-bound, NOT GPU-bound (2026-08-02)

Measured step wall-time: Qwen 4-GPU 8.17s, Qwen 1-GPU 7.57s, OLMo 4-GPU 9.88s.
DDP parallelises WITHIN a step (each rank takes its own fact), so **step time is
~independent of GPU count**. Consequences:

- N optimizer steps costs the same wall-clock at 1, 2 or 4 GPUs. GPUs buy DATA
  PER STEP, not steps.
- At fixed wall-clock, MORE GPUs strictly dominates: same updates, more facts.
- To get more steps you must spend more wall-clock. 20,000 steps = ~42h (Qwen)
  / ~56h (OLMo) regardless of allocation.

**THE PAPER'S GRPO RUNS WERE 20,000 UPDATES AT 1 FACT/STEP.** From W&B
(`jonathan-von-rad/UnLock`), four finished runs all share `steps=20000, bs=1,
n=20000` — `steps == n_facts` proves single-GPU. The 2026-08-02 4-GPU runs got
**2,540 (Qwen) / 2,100 (OLMo)** updates: an 8x shortfall, compounding bug #1
with the 4x step reduction from 4 facts/step. Those checkpoints evaluate
flat-to-negative on ALL FOUR benchmarks (PolyFact-Clean, G-MMLU-Lite,
BMLAMA-53, KLAR) and are unusable.

**RETRACTION — the bonus ablation is step-count confounded, not a null result.**
W&B shows `qwen-grpo-nobonus-ablation` crashed at **7,790** steps and
`qwen-grpo-bonus50-ablation` (which is bonus **5.0**, the dir name is a
leftover) at **12,850**, versus the paper's bonus-1.0 run at 20,000. The B=5 arm
trained 65% longer than B=0 and 36% shorter than B=1. So neither "bonus 5 beats
bonus 0 by +6.4pp" nor "bonus 5 is worse than bonus 1" isolates the bonus.
The bonus question is UNRESOLVED.

## THE SILENCE HOLE — an absorbing state in the reward (found 2026-08-03)

The reward table was, per language:

| output | score |
|---|---|
| correct option | **+1.0** |
| valid but wrong option | 0.0 |
| non-empty, unparseable | **−0.5** |
| **EMPTY** | **0.0** |

Emitting NOTHING ties the best non-correct outcome and strictly beats a wrong
guess. So when the policy drifts into garbled output the gradient correctly
pushes away from −0.5 — and the cheapest escape is one EOS token, not a valid
answer. Once all G rollouts are empty their rewards are identical, std = 0,
advantages z-score to 0, and the gradient is **exactly zero forever**: an
absorbing state training can never leave. Full trace from
`qwen-final-main-clip2` (logs/main-clip2-5888747.out):

```
3050  rew 13.00  every language clean ("D) General Motors")
3200  rew  4.00  std 0.00 — all valid, mostly wrong
3250  rew  3.00  garbage appearing ("what？", "トマto先生；", "19 Greentrees A")
      eval: resolution_rate 0.500, ffAcc 0.299   <- half of outputs unparseable
3300+ rew  0.00  std 0.00  grad 0.00 — all 12 languages EMPTY
```

**There are TWO distinct failure modes, with different levers — do not conflate
them.** Both were observed from the SAME checkpoint-3000:

| | mode A: one-shot destruction | mode B: the silence hole |
|---|---|---|
| run | clip 5.0 (5880532_0) | clip 2.0 (5888747) |
| died at | step 3,050 | ~step 3,270 |
| trace | 3000 grad **6.52** → 3050 rew **−6.00** | 250 steps HEALTHY (12.06/8.19/4.38), then res 0.500 → rew **0.00** |
| output | pure token salad (`strugg NotSupportedException (longleftrightarrow`) | empty in all 12 languages |
| cause | a single above-clip update destroys the policy | garbage(−0.5) → silence(0.0) is an uphill move |
| lever | **`--max_grad_norm`** | **`--empty_penalty`** |

clip 2.0 therefore **worked** — it prevented mode A, and the run then died of an
unrelated cause. An earlier note here claimed "clipping is not the mechanism";
that is true only of mode B. Both levers are needed together.

**Any state where all G rollouts score alike is absorbing** (std 0 → advantages 0
→ zero gradient), so `--empty_penalty` does not eliminate the all-unparseable
−6.00 state — it only makes silence unattractive. `--dead_run_patience` is the
backstop for whatever remains.

Two new flags, both defaulting to previous behaviour:
- **`--empty_penalty`** (default 0.0 = reproduces every earlier run byte for
  byte). Pass **1.0** to order the reward correctly:
  `correct +1 > valid-wrong 0 > unparseable −0.5 > empty −1`. Verified across
  en/de/zh/ja/bn/ar: at 0.0 an all-empty group scores 0.00, at 1.0 it scores
  −6.00 — strictly the worst outcome.
- **`--dead_run_patience`** (default 200). Aborts after that many CONSECUTIVE
  optimizer steps with `reward_std == 0`, which provably contribute zero
  gradient. Isolated zero-std steps are normal (a saturated all-correct group),
  hence *consecutive*. Costs nothing when healthy; saves the allocation when not.

**`--brevity_penalty` alone would NOT have prevented this and may worsen it** —
it only fires on CORRECT answers, so it leaves empty at 0.0 while adding
pressure toward shorter output. Ship it only together with `--empty_penalty`.

**checkpoint-3000 IS a sound restart point** (an earlier note here said
otherwise — that read the 2,000/2,500 dip as evidence the whole regime was
degraded, which the step-3000 eval contradicts): res 0.998, ffAcc 0.716,
ffTotCons 0.180, the run's best, with empty output essentially absent. The two
prior resumes failed because each had only one of the two levers, not because
the checkpoint was poisoned. `cluster/main_ep1_resume.sbatch` (job 5889956)
resumes it with clip 2.0 **and** `--empty_penalty 1.0` **and**
`--dead_run_patience 200`. Its 2,000/2,500 checkpoints (res 0.954, ffTotCons
0.020/0.010) are the genuinely weak ones.

## Reward composition: the objective is ~98% accuracy, ~2% consistency

`reward = (count of correct languages) + all_correct_bonus * 1[k == 12]`.
Measured share of mean reward contributed by the bonus: **Qwen 2.4%, OLMo 0.4%**
(it only fires on the 2-10% of rollouts that get all 12 right). Pooling rewards
a COUNT, not AGREEMENT — two rollouts correct in disjoint sets of 6 languages
score identically to one correct in the same 6 everywhere. And the count is
LINEAR in k, so 11->12 pays exactly as much as 5->6, while Total Consistency
lives entirely in that last step.

`--bonus_shape {all_or_nothing,power,ladder}` addresses this by making the bonus
convex in k. `ladder` takes ABSOLUTE rung values (`--bonus_ladder "9:1,10:2,
11:3,12:5"`, ignoring `--all_correct_bonus`). Default `all_or_nothing`
reproduces previous behaviour byte-for-byte.

## OLMo-2 has NO grouped-query attention — 8.7x the KV cache

`num_key_value_heads`: Qwen-2.5-7B **4** (GQA), OLMo-2-1124-7B **32** (full MHA).
Per-sequence KV cache 0.06 vs 0.52 MB per 1k tokens. Any generation batch size
tuned on Qwen is ~9x too large for OLMo. This is why CM-Align's construct phase
OOM'd on OLMo four times at batch sizes Qwen ran fine, and why a 20k-fact OLMo
CM-Align run needs `gen_micro_batch_size ~96` and 20+ hours — it was dropped.
KL arms need `--logprob_micro_batch_size 16 --fused_logprob`: the adapter-off
reference forward is under `no_grad`, so its `[mb,T,vocab]` tensors are
TRANSIENT and DO scale with mbs (unlike the retained policy graphs), and
`fused_logprob` — a no-op on the policy pass — genuinely removes a vocab-sized
copy there.

## Sweep results (2026-08-03) — `max_grad_norm` is the dominant knob

Qwen, 1,500 steps, 1 fact/step, lr 1e-5, r128, seed 42. Ranked on the FULL
2,039-fact PolyFact-Clean test and 17-lang KLAR (not the 150-fact in-loop eval).

| arm | PF Acc | PF TotCons | PF RankC | **KLAR** |
|---|---|---|---|---|
| control (clip 1.0) | +3.99 | +2.11 | +1.50 | **-1.20** |
| **clip 5.0** | +4.72 | +2.35 | +1.63 | **+8.67** |
| clip 20.0 | +3.36 | +1.81 | +1.31 | +7.36 |
| ladder (clip 1.0) | +4.48 | +2.16 | +1.69 | **+8.17** |
| paper GRPO (20,000 steps) | +4.72 | +2.31 | +1.79 | +2.24 |
| CM-Align (reference) | +3.77 | +0.98 | +1.06 | +8.85 |

**Raising `max_grad_norm` 1.0 -> 5.0 moves KLAR by ~10pp** and brings GRPO level
with CM-Align on generation while keeping a >2x consistency advantage. clip 20
is worse than clip 5 on every metric, so ~5 is near the optimum, not "looser is
better". The ladder reaches +8.17 independently at clip 1.0, via the reward
landscape — its median gradient is 0.87 vs control's 20.0. Every arm that
escaped the high-gradient regime got +8; the one pinned in it regressed.

**MCQ EVALUATION IS BLIND TO THIS.** On PolyFact all four arms look
interchangeable (spread near the paired noise floor). A 10pp generation gap was
invisible. Do not rank configs on MCQ metrics alone.

**1,500 steps reproduces the paper's 20,000-step checkpoint** on MCQ
(+4.72/+2.35/+1.63 vs +4.72/+2.31/+1.79) and beats it 4x on KLAR. The killed
4-GPU runs had 2,540 steps — MORE — and scored +0.76/-0.49, so the failure was
the LR bug plus 4 facts/step, not step count alone.

**KL does not help** (k3 estimator, 0/60 negative in all arms): at clip 5 it
HURTS (ffTotCons 0.193 -> 0.153 at kl 0.02); at clip 20 it is a wash; kl 0.05 is
worst on both metrics. This justifies `kl_coef=0`.

**Ladder rungs must be reachable.** `--bonus_ladder "9:1,10:2,11:3,12:5"` bands
as 9-10:+1, ... 12:+5 (highest matching rung wins). Qwen ends a 1,500-step run
at k~8.5 with 22-32% of steps at k>=9. **OLMo plateaus at k~5.7 and never
reaches 12** (best step-mean 11.44), so those rungs fire on 5-10% of steps — and
because advantages are z-scored within the group of 8, a bonus no rollout
reaches contributes EXACTLY ZERO. Use `6:1,8:2,10:3,12:5` for OLMo, or
`--bonus_shape power` which is non-zero at every k and needs no tuning.

## Paper table audit (2026-08-02) — what reproduces and what does not

**The High/Low split is: LOW = {id, bn, sw}, HIGH = the other 9.** Recovered by
search and confirmed by the user. With it, on PolyFact-Clean + Global-MMLU-Lite:

- **Every Global-MMLU cell reproduces EXACTLY** (all 14 models, to 0.1pp).
- **Every PolyFact cell reproduces to rounding.**
- **OLMo GRPO KLAR 29.0/16.7 is NOT reproducible.** Every other cell in that row
  matches `olmo-2-7b-grpo-att-mlp-full`, but that checkpoint scores
  **20.5/10.5** — a regression vs the 24.6/13.2 baseline, reported as a gain.
  `OLMo-2-7B-grpo` gives 26.6/15.7 but its PolyFact cells (46.0/36.7) contradict
  the rest of the row. No single checkpoint produces this row.
- **CM-Align KLAR is dashed but measured**: OLMo 33.3/19.6, Qwen 56.8/44.3 —
  the best KLAR numbers in the table, beating GRPO in both families.
- Caption says "nine unseen languages"; the OOD split is **ten**.
- CPT/SFT PolyFact and KLAR cells are unverifiable (no results for those
  checkpoints on PolyFact-Clean or 17-lang KLAR).

## Known inconsistencies to resolve

- The paper appendix says the PolyFact evaluator wraps questions in a
  language-specific instruction prompt; the actual scripts use the plain
  `Question:/Answer:` wrapper. Reconcile before the rebuttal.
- README references `requirements.txt`, which does not exist in the repo.

## Rebuttal work outstanding (from ACL ARR reviews, July 2026)

Done: direct consistency metrics (RankC + total consistency + agreement) for
all 12 models on PolyFact AND Global-MMLU-Lite (results/*_consistency.json);
bootstrap CIs + paired significance tests (results/significance/). Still requested by reviewers: CLC-enhancement baselines (DCO,
EN-pivot DPO / CM-Align, representation intervention), expanded related work
(Qi et al. 2023, Fierro & Søgaard 2022, X-FACTR, Paths Not Taken), and
confidence intervals for the small Global-MMLU deltas.

Training/token cost reporting: `train_wikifact_grpo_accelerate.py` now tracks
and persists this itself (2026-07-13) — cumulative rollout tokens (non-pad
completion tokens, summed across DDP ranks every micro-step), wall-clock
seconds, and GPU-hours (wall-clock × `accelerator.num_processes`). Logged to
wandb (`cost/*`) and printed every `--logging_steps`; a `training_stats.json`
sidecar is written next to every checkpoint AND the final `output_dir`, and
its `cumulative_*` fields are restored on `--resume_from_checkpoint auto` so
the totals stay correct across crashes/restarts (see the checkpoint/resume
work below). Still needs: pulling these numbers out of a finished run's
`training_stats.json` into the paper's actual cost table.

## EN-pivot DPO / CM-Align baseline (`training/train_wikifact_cmalign_dpo.py`)

Self-contained implementation of the reviewer-requested EN-pivot DPO baseline,
adapting CM-Align (Zhang et al., EMNLP 2025 Findings, arXiv:2509.08541;
github.com/XZhang00/CM-Align) to the WIKI-FACT MCQ setting. Self-supervised —
never uses gold labels.
- **Phase `construct`**: sample K free-text answers/lang (same single-language
  prompts as the GRPO trainer), pick the most self-consistent English candidate
  as the pivot (max mean cosine among English candidates, LaBSE embeddings),
  then per other language chosen=argmax / rejected=argmin cosine-to-pivot.
  Preference pairs cached via `save_to_disk` at `--pref_data_path`.
- **Phase `train`**: hand-rolled DPO (no `trl` — it isn't installed; the older
  `train_polyfact_dapo.py` depends on it and is NOT runnable here). LoRA policy
  r=64/α=128; the reference distribution is the same model with the adapter
  disabled (`model.disable_adapter()`), so no second model copy. Objective
  `L_DPO + gamma*L_NLL`, defaults β=0.1, γ=0.0 (CM-Align GIF task).
- **Embedder**: defaults to `sentence-transformers/LaBSE` (this project's
  cross-lingual encoder, cached; CM-Align's original was gte-multilingual-base,
  swap via `--embedding_model`). `sentence-transformers` was pip-installed into
  the `grpo` env for this.
- Launch: `sbatch cluster/cmalign_dpo.sbatch` (Qwen base, 1 GPU, offline; runs
  both phases). OLMo via `--export=ALL,MODEL_ID=allenai/OLMo-2-1124-7B,TAG=olmo`.
  Logic unit-tested (scratchpad): dpo_loss = -log(0.5) at LoRA zero-init confirms
  the disable_adapter reference path. Then eval the `merged/` output with
  `evaluate/evaluate_crosslingual_consistency.py`.
