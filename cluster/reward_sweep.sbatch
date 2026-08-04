#!/bin/bash
# Diagnostic sweep over the four knobs implicated by the post-mortem of the
# truncated production runs (see CLAUDE.md):
#
#   max_grad_norm  — measured pre-clip norm median 139 (Qwen) vs a hardcoded
#                    clip of 1.0, so ~99% of steps were renormalised and
#                    relative step magnitude was discarded.
#   (task_format   — DROPPED: pure freeform already tested by hand and the
#                    model found it too hard. A 50/50 "mixed" variant is the
#                    remaining option but is not implemented.)
#   bonus_shape    — the all-or-nothing bonus fires on ~2-10% of rollouts and
#                    measured 0.4-2.4% of mean reward. Scaling it 1->5 was a
#                    NULL result, so the graded ladder is a mechanism bet, not
#                    a safe one.
#   kl_coef        — 0.0 leaves no anchor; grad norms tripled over training.
#
# ONE GPU PER CONFIG, as an array: --gpus=1 jobs start in seconds on this
# cluster while 4-GPU jobs pend, so 7 independent 1-GPU jobs schedule far
# faster than 2 node reservations. 1,500 facts on 1 GPU = 1,500 optimizer
# steps at 1 fact/step -- which also matches the PAPER's 1-fact/step semantics,
# unlike the 4-GPU production runs (4 facts/step).
#
# ~3.1 h (control) to ~3.9 h (kl arms, which add a reference forward pass per
# micro-batch) + evals => ~4.2 h worst case, ~26 GPU-h total.
#
# Design: a PARTIAL grid over (max_grad_norm x kl_coef) plus one ladder arm.
# The (clip 1.0, kl 0.02) cell was dropped deliberately — at the tight clip the
# policy barely moves, so there is little drift for a KL anchor to prevent and
# that cell carries the least information. Its slot went to (clip 20, kl 0.05),
# giving a KL DOSE-RESPONSE (0.0 / 0.02 / 0.05) at the loosest clip, which is
# exactly where drift should appear. Cost: the pure KL main effect at clip 1.0
# is no longer estimable; gain: three points on the axis that matters.
# Clip and KL are crossed rather than tested alone because they are mechanically
# coupled: loosening the clip lets the policy move further per step, so drift
# risk — the thing kl_coef exists to control — rises with it. Testing either
# alone would confound "bigger steps help" with "bigger steps drift". With only
# two factors the interaction IS estimable, unlike a 2^4 design whose
# interactions would sit under the 150-fact eval's ~2-3pp TotCons noise floor.
#
# Submit: sbatch cluster/reward_sweep.sbatch
#SBATCH --job-name=rwd-sweep
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-6
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/rwd-sweep-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/rwd-sweep-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MTAG="${MTAG:-qwen}"

# name | extra flags (relative to the control)
CONFIGS=(
  "control|"
  "clip5|--max_grad_norm 5.0"
  "clip20|--max_grad_norm 20.0"
  "clip20-kl005|--max_grad_norm 20.0 --kl_coef 0.05 --ref_impl adapter_off --logprob_micro_batch_size 16 --fused_logprob"
  "clip5-kl002|--max_grad_norm 5.0 --kl_coef 0.02 --ref_impl adapter_off --logprob_micro_batch_size 16 --fused_logprob"
  "clip20-kl002|--max_grad_norm 20.0 --kl_coef 0.02 --ref_impl adapter_off --logprob_micro_batch_size 16 --fused_logprob"
  "ladder|--bonus_shape ladder"
)
IFS='|' read -r NAME EXTRA <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TAG="${MTAG}-sweep-${NAME}"

echo "[sweep] task=$SLURM_ARRAY_TASK_ID name=$NAME extra='$EXTRA' on $(hostname) @ $(date -Is)"

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/sweep/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --reward_pooling group \
  --all_correct_bonus 1.0 \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 1500 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 500 --max_eval_wikifact 150 --max_eval_mmlu 100 \
  --logging_steps 25 \
  --report_to wandb \
  $EXTRA

STATUS=$?
echo "[sweep] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
