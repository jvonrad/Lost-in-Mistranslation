#!/bin/bash
# Resume the ladder run from its last HEALTHY checkpoint, with the clip tightened
# back to 1.0.
#
# WHY: arm 3 (ladder + clip 5.0) collapsed catastrophically at step ~2,200.
# Mechanism, from the logged trace:
#   step 2000  reward 13.69  grad  1.12   healthy
#   step 2050  reward 14.00  grad  0.00   saturated - all 8 rollouts scored alike,
#                                         so std=0, advantages z-score to 0, no gradient
#   step 2150  reward 13.06  grad 10.96   rare group WITH variance -> large update
#   step 2200  reward -6.00  grad  0.00   policy destroyed, and no gradient to recover
# -6.00 is exactly 12 x -0.5: non-empty but unparseable output in every language.
#
# The ladder drives reward to saturation, which makes gradients sparse and bursty;
# clip 5.0 then lets the rare burst land at ~half strength. clip 1.0 is precisely
# the guard against that, and the 1,500-step sweep measured ladder-at-clip1 at
# +8.67/+8.17 KLAR with no instability.
#
# CAVEAT FOR THE WRITE-UP: this is a HYBRID, not a clean ladder-at-clip1 run --
# steps 0-2,000 were trained at clip 5.0 and 2,000-10,000 at clip 1.0. The
# optimizer moments and LR schedule carry over from the checkpoint. Describe it
# that way; a clean arm would need a fresh 22.2h run, which does not fit the
# remaining window.
#
# 8,000 remaining steps x ~8.0 s = ~17.8h.
#
# Submit: sbatch cluster/ladder_clip1_resume.sbatch
#SBATCH --job-name=ladder-clip1
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=19:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ladder-clip1-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ladder-clip1-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

CKPT="$PROJ/models/final/qwen-final-ladder-clip5/checkpoint-2000"
TAG="qwen-final-ladder-clip1-from2k"

echo "[ladder-clip1] resuming from $CKPT @ $(date -Is)"
[[ -f "$CKPT/optimizer.bin" ]] || { echo "checkpoint incomplete at $CKPT"; exit 3; }

# NOTE: a distinct --output_dir so the collapsed run's artifacts are left intact;
# --resume_from_checkpoint takes the explicit path, so it does not need to live
# in the new output_dir.
python -u training/train_wikifact_grpo_accelerate.py \
  --model_id Qwen/Qwen2.5-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/final/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm 1.0 \
  --bonus_shape ladder --bonus_ladder 9:1,10:2,11:3,12:5 \
  --reward_pooling group \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 10000 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 500 --max_eval_wikifact 100 --max_eval_mmlu 100 \
  --logging_steps 50 \
  --resume_from_checkpoint "$CKPT" \
  --report_to wandb

STATUS=$?
echo "[ladder-clip1] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
