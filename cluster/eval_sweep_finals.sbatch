#!/bin/bash
# Rank the sweep arms on the FULL 2,039-fact PolyFact-Clean test rather than the
# 150-fact in-loop eval.
#
# WHY: the in-loop eval has ~2.1pp standard error on TotCons and the spread
# between arms is 2-3pp — it cannot separate them. The full test set gives
# ~0.6pp, a 3.7x tighter metric, for ~4 min of GPU per arm. This is the number
# the full-run config should be chosen from.
#
# Each task merges its LoRA adapter into the base model, evaluates, then deletes
# the merged copy (15 GB each; 9 arms would otherwise leave 135 GB on Lustre).
#
# Submit only once the sweep arms have written their final adapters:
#   sbatch cluster/eval_sweep_finals.sbatch
#   sbatch --array=0,6 cluster/eval_sweep_finals.sbatch     # a subset
#SBATCH --job-name=eval-sweep-finals
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-sweepfinal-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-sweepfinal-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# adapter_dir | base_model | result_tag
ENTRIES=(
  "qwen-sweep-control|Qwen/Qwen2.5-7B|qwen-sw-control"
  "qwen-sweep-clip5|Qwen/Qwen2.5-7B|qwen-sw-clip5"
  "qwen-sweep-clip20|Qwen/Qwen2.5-7B|qwen-sw-clip20"
  "qwen-sweep-clip20-kl005|Qwen/Qwen2.5-7B|qwen-sw-clip20-kl005"
  "qwen-sweep-clip5-kl002|Qwen/Qwen2.5-7B|qwen-sw-clip5-kl002"
  "qwen-sweep-clip20-kl002|Qwen/Qwen2.5-7B|qwen-sw-clip20-kl002"
  "qwen-sweep-ladder|Qwen/Qwen2.5-7B|qwen-sw-ladder"
  "olmo-clip1.0|allenai/OLMo-2-1124-7B|olmo-sw-clip1"
  "olmo-clip5.0|allenai/OLMo-2-1124-7B|olmo-sw-clip5"
  "qwen-pilot-ladder-clip5|Qwen/Qwen2.5-7B|qwen-sw-ladder-clip5"
  "olmo-pilot-ladder-clip5|allenai/OLMo-2-1124-7B|olmo-sw-ladder-clip5"
  "../qwen-dco-n10000-r128/merged|MERGED|qwen-dco-10k"
  "../olmo-dco-n10000-r128/merged|MERGED|olmo-dco-10k"
  "../qwen-cmalign-n10000/merged|MERGED|qwen-cmalign-10k"
  "../olmo-cmalign-n10000/merged|MERGED|olmo-cmalign-10k"
  "../qwen-cmalign-n20000/merged|MERGED|qwen-cmalign-20k"
  "../final/qwen-final-ladder-clip5/checkpoint-2000|Qwen/Qwen2.5-7B|qwen-ladder-clip5-s2000"
  # 17-18: DCO applied ON TOP of the CPT checkpoints, matched to the 10k facts
  # every other post-2026-08 arm uses. Qwen CPT = jvonrad/Qwen-2.5-7B-TED;
  # OLMo CPT = jvonrad/olmo-2-7b-finetranslations (the April lineage that
  # CPT+SFT and CPT+GRPO were built from, NOT the May OLMo-2-1124-7B-TED).
  "../qwen-cpt-dco-n10000-r128/merged|MERGED|qwen-cpt-dco-10k"
  "../olmo-cpt-dco-n10000-r128/merged|MERGED|olmo-cpt-dco-10k"
  # 19-23: EARLY ESTIMATE from the five in-flight 10k GRPO arms. These are the
  # LATEST SURVIVING checkpoint of each run -- the trainer hardcodes a keep-last-3
  # policy (train_wikifact_grpo_accelerate.py:2348), so earlier states are already
  # deleted and "best checkpoint" can only be chosen from what remains. The runs
  # are still training, so these numbers are provisional and will move.
  # The ADAPTERS were copied to final/_preserved/ first: the deleter would
  # otherwise remove them mid-eval (main-ep1 saves every 250 steps, so its
  # checkpoint-4750 had <1.5h to live). Only adapter+tokenizer are copied,
  # 1.2-1.3 GB each rather than the 20 GB full save_state.
  "../final/_preserved/qwen-nobonus-s6000|Qwen/Qwen2.5-7B|qwen-nobonus-s6000"
  "../final/_preserved/olmo-main10k-s5000|allenai/OLMo-2-1124-7B|olmo-main10k-s5000"
  "../final/_preserved/olmo-ladder10k-s5000|allenai/OLMo-2-1124-7B|olmo-ladder10k-s5000"
  "../final/_preserved/qwen-ladder-c1-s5000|Qwen/Qwen2.5-7B|qwen-ladder-c1-s5000"
  "../final/_preserved/qwen-main-ep1-s4750|Qwen/Qwen2.5-7B|qwen-main-ep1-s4750"
  # 24-27: SFT retrained on the SAME 10k facts as GRPO/DCO/CM-Align, with
  # --consistency_weight 0.0 (PURE cross-entropy). The released SFT checkpoints
  # used 40k facts, so the old SFT rows compared a 4x-data method against
  # 10k-data ones; and the auxiliary consistency term had to go, since leaving
  # it on puts a cross-lingual consistency objective inside the very baseline
  # GRPO's consistency reward is measured against.
  "../qwen-sft-n10000/merged|MERGED|qwen-sft-10k"
  "../olmo-sft-n10000/merged|MERGED|olmo-sft-10k"
  "../qwen-cpt-sft-n10000/merged|MERGED|qwen-cpt-sft-10k"
  "../olmo-cpt-sft-n10000/merged|MERGED|olmo-cpt-sft-10k"
  # 28-32: SECOND snapshot of the five in-flight 10k arms, ~1,500-2,000 steps
  # later than entries 19-23. Two snapshots of the same runs give a within-run
  # trend, which is the only way to tell "still improving" from "already peaked"
  # before the annealed endpoints land. Adapters copied to _preserved/ first;
  # the keep-last-3 deleter would otherwise remove them mid-eval.
  "../final/_preserved/qwen-nobonus-s8000|Qwen/Qwen2.5-7B|qwen-nobonus-s8000"
  "../final/_preserved/olmo-main10k-s6500|allenai/OLMo-2-1124-7B|olmo-main10k-s6500"
  "../final/_preserved/olmo-ladder10k-s6500|allenai/OLMo-2-1124-7B|olmo-ladder10k-s6500"
  "../final/_preserved/qwen-ladder-c1-s7000|Qwen/Qwen2.5-7B|qwen-ladder-c1-s7000"
  "../final/_preserved/qwen-main-ep1-s6750|Qwen/Qwen2.5-7B|qwen-main-ep1-s6750"
  # 33-36: the 1,500-fact points of the data-scaling curve. Same recipes as the
  # 10k runs, only --max_train_facts differs, so the curve isolates data.
  "../qwen-dco-n1500-r128/merged|MERGED|qwen-dco-1500"
  "../olmo-dco-n1500-r128/merged|MERGED|olmo-dco-1500"
  "../qwen-sft-n1500/merged|MERGED|qwen-sft-1500"
  "../olmo-sft-n1500/merged|MERGED|olmo-sft-1500"
  # 37: OLMo GRPO at 1,500 steps with the recipe that won the Qwen sweep. The
  # existing olmo-sw-clip5 used --max_train_samples 1000, so OLMo has never been
  # run at the step count where the Qwen gains appear. This is the matched
  # comparison, and the test of whether OLMo's flat TotCons is a family
  # limitation or 500 missing steps.
  "olmo-sweep-clip5-n1500|allenai/OLMo-2-1124-7B|olmo-grpo-1500"
  # 38-39: CM-Align at 1,500 facts, completing the data-scaling curve.
  "../qwen-cmalign-n1500/merged|MERGED|qwen-cmalign-1500"
  "../olmo-cmalign-n1500/merged|MERGED|olmo-cmalign-1500"
  # 40-44: FINAL checkpoints of the five 10k arms. The three Qwen runs reached
  # step 10,000 with the cosine schedule fully annealed (lr -> 0); the two OLMo
  # arms hit the 23h QOS wall at 9,000 and 8,500 (OLMo is ~9.9 s/step vs Qwen's
  # 7.5), so those are the last checkpoints rather than annealed endpoints --
  # state that when comparing OLMo against Qwen here.
  "../final/_preserved/qwen-nobonus-final|Qwen/Qwen2.5-7B|qwen-nobonus-final"
  "../final/_preserved/qwen-ladder-c1-final|Qwen/Qwen2.5-7B|qwen-ladder-c1-final"
  "../final/_preserved/qwen-main-ep1-final|Qwen/Qwen2.5-7B|qwen-main-ep1-final"
  "../final/_preserved/olmo-ladder10k-final|allenai/OLMo-2-1124-7B|olmo-ladder10k-final"
  "../final/_preserved/olmo-main10k-final|allenai/OLMo-2-1124-7B|olmo-main10k-final"
)
IFS='|' read -r ADIR BASE TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
ADAPTER="$PROJ/models/sweep/$ADIR"
MERGED="$PROJ/models/sweep/_merged_$ADIR"

echo "[eval-final] task=$SLURM_ARRAY_TASK_ID tag=$TAG adapter=$ADAPTER @ $(date -Is)"

# Entries with BASE=MERGED are already full merged models (e.g. the DCO trainer
# writes one directly), so there is nothing to merge and nothing to clean up.
if [[ "$BASE" == "MERGED" ]]; then MERGED="$ADAPTER"; SKIP_MERGE=1; else SKIP_MERGE=0; fi
if [[ "$SKIP_MERGE" == "0" && ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "no final adapter at $ADAPTER — did the arm finish?"; exit 3
fi

if [[ "$SKIP_MERGE" == "0" ]]; then
python -u - "$ADAPTER" "$BASE" "$MERGED" <<'PY'
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
adapter, base_id, out = sys.argv[1:4]
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter, dtype=torch.bfloat16).merge_and_unload()
# Guard against silently evaluating the base model if the adapter is a no-op.
ref = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
rp = dict(ref.named_parameters())
d = max(((p - rp[n]).abs().max().item() for n, p in list(model.named_parameters())[:400] if n in rp),
        default=0.0)
print(f"merge check: max |delta| = {d:.6f}", flush=True)
assert d > 0, "adapter merge was a no-op — refusing to evaluate a base-model copy"
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
try: AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
except Exception: AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
PY
[[ $? -ne 0 ]] && { echo "merge failed"; exit 4; }
fi

BENCHMARK="${BENCHMARK:-polyfact}"
if [[ "$BENCHMARK" == "bmlama53" ]]; then
  # 11 of our 12 languages (BMLAMA-53 has no `sw` config). 93% of items have a
  # 10-candidate pool, so RankC has real dynamic range here (floor ~0.003) unlike
  # PolyFact's 4-option format (floor 0.0902) -- this is RankC's native benchmark.
  python -u evaluate/evaluate_crosslingual_consistency.py \
    --benchmark bmlama53 --split test \
    --model "$MERGED" \
    --device cuda --batch_size 16 --max_length 192 \
    --score_mode byte \
    --langs en,de,id,pt,ar,bn,es,ru,fr,ja,zh \
    --output_json "$REPO/results/${TAG}_bmlama53_consistency.json"
  STATUS=$?
  [[ "$SKIP_MERGE" == "0" ]] && rm -rf "$MERGED"
  echo "[eval-final] $TAG / bmlama53 finished status $STATUS @ $(date -Is)"
  exit $STATUS
fi

if [[ "$BENCHMARK" == "polyfact_freeform" ]]; then
  FFSUF=""; [[ "${N_SHOT:-0}" != "0" ]] && FFSUF="_${N_SHOT}shot"
  # Same PolyFact-Clean test facts as the MCQ run, but GENERATED and scored with
  # KLAR's matcher -- so in-domain recognition vs in-domain generation differ
  # only in protocol.
  python -u evaluate/evaluate_polyfact_freeform.py \
    --model "$MERGED" \
    --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel --split test \
    --batch_size 32 --max_new_tokens 16 --n_shot "${N_SHOT:-0}" \
    --output_json "$REPO/results/${TAG}_polyfact_freeform${FFSUF:-}.json"
  STATUS=$?
  [[ "$SKIP_MERGE" == "0" ]] && rm -rf "$MERGED"
  echo "[eval-final] $TAG / polyfact_freeform finished status $STATUS @ $(date -Is)"
  exit $STATUS
fi

if [[ "$BENCHMARK" == "gmmlu_lite" ]]; then
  # Global-MMLU-Lite has NO Russian config, so this is 11 languages, not 12 --
  # matching every previously-committed *_gmmlu_lite_consistency.json so the new
  # rows are comparable to the paper-checkpoint rows.
  python -u evaluate/evaluate_crosslingual_consistency.py \
    --benchmark global_mmlu_lite --split test \
    --model "$MERGED" \
    --device cuda --batch_size 16 --max_length 768 \
    --score_mode byte \
    --langs en,de,id,pt,ar,bn,sw,es,fr,ja,zh \
    --output_json "$REPO/results/${TAG}_gmmlu_lite_consistency.json"
  STATUS=$?
  [[ "$SKIP_MERGE" == "0" ]] && rm -rf "$MERGED"
  echo "[eval-final] $TAG / gmmlu_lite finished status $STATUS @ $(date -Is)"
  exit $STATUS
fi

if [[ "$BENCHMARK" == "klar" ]]; then
  # ANSWER_EXTRACT=first_nonempty_line re-scores completions that open with a
  # newline. Results go to a SEPARATE *_klar_alllangs_nl.json so the historical
  # first_line numbers are never overwritten and the two can be diffed.
  AX="${ANSWER_EXTRACT:-first_line}"
  KSUF=""; [[ "$AX" != "first_line" ]] && KSUF="_nl"
  # 17-language KLAR, the protocol the CM-Align/DCO comparison used. --langs is
  # nargs="+" so it MUST be space-separated; comma-separated silently scores 0
  # samples and still exits 0.
  python -u evaluate/evaluate_klar.py \
    --model "$MERGED" --tokenizer "$MERGED" \
    --device cuda --klar-root datasets/KLAR-CLC \
    --langs ar ca el en es fa fr he hu ja ko nl ru tr uk vi zh \
    --batch-size 32 --max-length 448 \
    --contamination-labels evaluate/alignments/klar_polyfact_contamination.json \
    --answer-extract "$AX" \
    --output-json "$REPO/results/klar/${TAG}_klar_alllangs${KSUF}.json"
  STATUS=$?
  [[ "$SKIP_MERGE" == "0" ]] && rm -rf "$MERGED"
  echo "[eval-final] $TAG / klar finished status $STATUS @ $(date -Is)"
  exit $STATUS
fi

python -u evaluate/evaluate_crosslingual_consistency.py \
  --benchmark polyfact \
  --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel --split test \
  --model "$MERGED" \
  --device cuda --batch_size 32 --max_length 192 \
  --score_mode byte \
  --output_json "$REPO/results/${TAG}_polyfact_clean_consistency.json"
STATUS=$?

[[ "$SKIP_MERGE" == "0" ]] && rm -rf "$MERGED"
echo "[eval-final] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
