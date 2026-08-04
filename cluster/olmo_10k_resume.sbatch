#!/bin/bash
# Finish the two OLMo 10k arms, which hit the 23h QOS wall short of step 10,000.
#
#   olmo-final-main-clip5     stopped at 8,500  -> 1,500 steps left
#   olmo-final-ladder-clip5   stopped at 9,000  -> 1,000 steps left
#
# OLMo runs ~9.9 s/step against Qwen's ~7.5 (no GQA: 32 KV heads vs 4, so an
# 8.7x larger KV cache during rollout generation), which is why 10,000 steps
# fitted in 23h for the three Qwen arms and not for these two.
#
# WHY FINISHING MATTERS: the cosine schedule anneals to lr=0 at step 10,000, so
# a checkpoint at 8,500 is mid-flight at ~15% of peak LR, not a settled model.
# Every earlier comparison in this project that put a mid-cosine checkpoint next
# to an annealed one was misleading for exactly this reason.
#
# Flags are copied verbatim from cluster/final_runs.sbatch -- in particular
# --max_train_samples 10000 must stay, because total_update_steps defines the
# schedule and lowering it would re-anneal over the remaining steps instead of
# continuing the original curve. --resume_from_checkpoint restores global_step,
# optimizer moments and the cumulative cost counters, and skip_first_batches
# replays the same fact order.
#
# Submit: sbatch cluster/olmo_10k_resume.sbatch
#SBATCH --job-name=olmo-10k-fin
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-1
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo10kfin-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo10kfin-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# tag | last checkpoint | arm-specific reward flags (as in final_runs.sbatch)
ENTRIES=(
  "olmo-final-main-clip5|8500|--all_correct_bonus 1.0"
  "olmo-final-ladder-clip5|9000|--bonus_shape ladder --bonus_ladder 6:1,8:2,10:3,12:5"
)
IFS='|' read -r TAG STEP EXTRA <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
CKPT="$PROJ/models/final/$TAG/checkpoint-$STEP"

echo "[olmo-10k-fin] $TAG resuming from step $STEP @ $(date -Is)"
[[ -f "$CKPT/optimizer.bin" ]] || { echo "incomplete checkpoint at $CKPT"; exit 3; }

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id allenai/OLMo-2-1124-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/final/$TAG" \
  --run_name "${TAG}-finish" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm 5.0 \
  --reward_pooling group \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 10000 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 500 --max_eval_wikifact 100 --max_eval_mmlu 100 \
  --logging_steps 50 \
  --dead_run_patience 200 \
  --resume_from_checkpoint "$CKPT" \
  --report_to wandb \
  $EXTRA

STATUS=$?
echo "[olmo-10k-fin] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
