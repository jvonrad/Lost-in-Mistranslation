#!/bin/bash
# Successive-halving hyperparameter sweep: LoRA rank × learning rate.
#
# Why not a full-length grid: 9 configs × 44 GPU-h = 396 GPU-h = most of the
# remaining budget. Instead:
#
#   rung 1  all 9 configs, 1,500 facts each, 1 GPU per config, 4 per node
#           (~3.1 h/config ≈ 28 GPU-h total; waves 0,1 = 4 configs, wave 2 = 1
#           — schedule wave 2 as a non-exclusive --gpus=1 job, or fold it in
#           beside rung-2 work, so 3 GPUs don't idle on an exclusive node)
#   rung 2  top-3 by val metric, fresh runs at 5,000 facts
#           (~10.4 h/config ≈ 31 GPU-h, one wave)
#           NOTE: on an EXCLUSIVE node the 4th GPU idles but is billed
#           anyway — request a non-exclusive --gpus=3 allocation for rung 2
#           so the top-3 cut saves real hours (or run a 4th runner-up for
#           free if you're on a whole node regardless).
#   rung 3  winner only: fresh full 40k production run on 2 GPUs (44 GPU-h)
#
# ≈103 GPU-h vs 396 — and the reported checkpoint is a clean full run, not a
# promoted sweep artifact. Rungs use FRESH runs rather than checkpoint
# promotion: re-simulating rung-1's 1,500 facts inside rung 2 wastes ~12 GPU-h
# total, far cheaper than validating cross-rung resume semantics (dataloader
# order under a changed max_train_samples, RNG restore, etc.).
#
# Noise control — all configs share ONE seed, so every run sees the same facts
# in the same order and differences are paired, not drowned in sampling noise.
# The val metric (polyfact/mcq_* on the 444-fact validation split) has a noise
# floor of roughly ±0.6 pp accuracy / ±2 pp TotCons: treat configs within that
# as TIES and prefer the smaller rank / smaller lr (cheaper, safer).
#
# KL configs use --ref_impl adapter_off (policy with LoRA disabled == base
# model, proven exact), so no +15 GB second model and four runs fit one node.
#
# Kill-early signals while a wave runs (check the logs, don't wait it out):
#   * reward climbs but polyfact/freeform_resolution_rate collapses -> reward
#     hacking, kill the config
#   * kl term exploding at kl_coef=0 configs late in training is EXPECTED
#     rung-2 behaviour to watch, not an error
#
# Usage (inside a full-node allocation):
#   bash cluster/hp_sweep.sh rung1 <wave 0|1|2>
#   bash cluster/hp_sweep.sh rung2 "r64_kl0.05 r64_kl0.02 r16_kl0.05"
#   bash cluster/hp_sweep.sh report
set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

SWEEP_DIR="$PROJ/models/hp_sweep"
LOG_DIR="$PROJ/logs/hp_sweep"
mkdir -p "$SWEEP_DIR" "$LOG_DIR"

# The grid: rank {64,128,256} x lr {3e-6, 1e-5, 3e-5}. alpha = 2r throughout,
# so lr is the one global scale knob and rank is pure capacity. kl_coef is
# FIXED at 0.0 to match the ablation family (nobonus/bonus5/per_lang all use
# kl 0); the adapter_off ref machinery stays available for a later KL pass.
# lr grid is log-spaced around the known-good 1e-5.
RANKS=(64 128 256)
LRS=(3e-6 1e-5 3e-5)

MODEL="${MODEL_ID:-Qwen/Qwen2.5-7B}"
SEED=42          # SHARED across all configs — paired comparison
RUNG1_FACTS=1500
RUNG2_FACTS=5000

run_config () {  # tag gpu port n_facts lora_r lr
  local TAG="$1" GPU="$2" PORT="$3" N="$4" R="$5" LR="$6"
  echo "--- $TAG on GPU $GPU (r=$R alpha=$((2*R)) lr=$LR facts=$N)"
  CUDA_VISIBLE_DEVICES="$GPU" \
  accelerate launch --num_processes 1 --main_process_port "$PORT" \
    training/train_wikifact_grpo_accelerate.py \
    --model_id "$MODEL" \
    --output_dir "$SWEEP_DIR/$TAG" \
    --run_name "hp-$TAG" \
    --use_lora --bf16 \
    --lora_r "$R" \
    --kl_coef 0.0 \
    --num_generations 8 \
    --learning_rate "$LR" \
    --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
    --per_device_train_batch_size 1 \
    --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
    --max_train_samples "$N" --num_train_epochs 1 \
    --seed "$SEED" \
    --logging_steps 25 \
    --eval_steps 250 --max_eval_wikifact 150 --max_eval_mmlu 100 \
    --resume_from_checkpoint auto \
    --report_to wandb \
    > "$LOG_DIR/$TAG.log" 2>&1
  echo "    $TAG done rc=$?"
}

case "${1:-}" in
  rung1)
    # 9 configs / 4 GPUs: waves 0,1 have 4 configs, wave 2 has 1.
    WAVE="${2:?usage: hp_sweep.sh rung1 <0|1|2>}"
    ALL=()
    for R in "${RANKS[@]}"; do for LR in "${LRS[@]}"; do ALL+=("$R:$LR"); done; done
    for i in 0 1 2 3; do
      IDX=$(( WAVE * 4 + i ))
      (( IDX >= ${#ALL[@]} )) && continue
      R="${ALL[$IDX]%%:*}"; LR="${ALL[$IDX]##*:}"
      run_config "r${R}_lr${LR}" "$i" "$(( 29900 + i ))" "$RUNG1_FACTS" "$R" "$LR" &
    done
    wait
    ;;
  rung2)
    # Space-separated tags of the promoted configs, e.g. "r64_kl0.05 r16_kl0.02 ..."
    read -ra TAGS <<< "${2:?usage: hp_sweep.sh rung2 \"tagA tagB tagC\"}"
    for i in "${!TAGS[@]}"; do
      TAG="${TAGS[$i]}"
      R="${TAG#r}"; R="${R%%_*}"; LR="${TAG##*_lr}"
      run_config "${TAG}_r2" "$i" "$(( 29900 + i ))" "$RUNG2_FACTS" "$R" "$LR" &
    done
    wait
    ;;
  report)
    # Rank configs by their LAST periodic-eval val metrics.
    python - "$LOG_DIR" <<'PY'
import glob, re, sys, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.log"))):
    txt = open(f, errors="ignore").read()
    evals = re.findall(r"\[eval @ step (\d+)\] (\{.*\})", txt)
    if not evals:
        rows.append((os.path.basename(f)[:-4], None)); continue
    import ast
    last = ast.literal_eval(evals[-1][1])
    rows.append((os.path.basename(f)[:-4], last))
def key(r):
    m = r[1] or {}
    return -(m.get("polyfact/mcq_accuracy", 0) + m.get("polyfact/mcq_total_consistency", 0))
rows.sort(key=key)
print(f"{'config':22s} {'mcq_acc':>8s} {'mcq_totcons':>11s} {'rankc':>7s} {'freeform':>9s} {'resol':>6s}")
for tag, m in rows:
    if m is None:
        print(f"{tag:22s}  (no eval yet)"); continue
    print(f"{tag:22s} {m.get('polyfact/mcq_accuracy', float('nan')):8.4f} "
          f"{m.get('polyfact/mcq_total_consistency', float('nan')):11.4f} "
          f"{m.get('consistency/rankc_avg', float('nan')):7.4f} "
          f"{m.get('polyfact/freeform_accuracy', float('nan')):9.4f} "
          f"{m.get('polyfact/freeform_resolution_rate', float('nan')):6.4f}")
print("\nnoise floor on 150-fact eval: ~±1 pp accuracy — treat closer configs as ties;")
print("prefer smaller rank / smaller lr among ties. Promote top-3 to rung2.")
PY
    ;;
  submit_rung1)
    # PREFERRED: each config as its own 1-GPU sbatch job (proportional billing;
    # failures stop billing at exit; no idle-sibling waste; schedules fast).
    for R in "${RANKS[@]}"; do for LR in "${LRS[@]}"; do
      TAG="r${R}_lr${LR}"
      sbatch --job-name="hp-$TAG" \
        --export=ALL,R="$R",LR="$LR",FACTS="$RUNG1_FACTS",TAG="$TAG" \
        cluster/hp_sweep_job.sbatch
    done; done
    ;;
  submit_rung2)
    read -ra TAGS <<< "${2:?usage: hp_sweep.sh submit_rung2 \"tagA tagB tagC\"}"
    for TAG in "${TAGS[@]}"; do
      R="${TAG#r}"; R="${R%%_*}"; LR="${TAG##*_lr}"
      sbatch --job-name="hp-${TAG}_r2" --time=13:00:00 \
        --export=ALL,R="$R",LR="$LR",FACTS="$RUNG2_FACTS",TAG="${TAG}_r2" \
        cluster/hp_sweep_job.sbatch
    done
    ;;
  *)
    echo "usage: hp_sweep.sh submit_rung1 | submit_rung2 \"tags\" | rung1 <wave> | rung2 \"tags\" | report"; exit 2 ;;
esac
