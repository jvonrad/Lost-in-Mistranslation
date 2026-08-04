#!/bin/bash
# Pilot: ladder bonus COMBINED with max_grad_norm 5.0, one arm per model.
#
# The existing sweeps test these separately, never together:
#   Qwen sweep 5876625: arm 6 = ladder @ clip 1.0, arm 1 = clip5 @ all_or_nothing
#   OLMo probe 5877536: clip 1.0 vs clip 5.0, no ladder arm at all
# So this fills the missing cell before committing to a 4-GPU full run.
#
# STEP COUNTS ARE DELIBERATELY UNEQUAL, to match each model's own comparison set:
#   Qwen 1,500 (matches the sweep arms)   OLMo 1,000 (matches the clip probe)
# The cosine LR spans the whole run, so a 1,000-step Qwen pilot would finish at
# lr~0 while the 1,500-step arms are still at ~2.5e-6 — the curves would not be
# comparable at any shared step. Everything else (r128, lr 1e-5, kl 0, G=8,
# 1 fact/step, seed 42) is identical to the runs each is compared against.
#
# ~3.2 h (Qwen) / ~2.7 h (OLMo), 1 GPU each, ~6 GPU-h total.
#
# Submit: sbatch cluster/ladder_clip5_pilot.sbatch
#SBATCH --job-name=ladder-pilot
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ladder-pilot-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ladder-pilot-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# model_id | mtag | n_facts | eval_steps
ENTRIES=(
  "Qwen/Qwen2.5-7B|qwen|1500|500"
  "allenai/OLMo-2-1124-7B|olmo|1000|250"
)
IFS='|' read -r MODEL_ID MTAG NFACTS EVSTEPS <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
TAG="${MTAG}-pilot-ladder-clip5"

echo "[pilot] task=$SLURM_ARRAY_TASK_ID model=$MODEL_ID facts=$NFACTS on $(hostname) @ $(date -Is)"

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/sweep/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm 5.0 \
  --bonus_shape ladder \
  --reward_pooling group \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples "$NFACTS" --num_train_epochs 1 \
  --seed 42 \
  --eval_steps "$EVSTEPS" --max_eval_wikifact 150 --max_eval_mmlu 100 \
  --logging_steps 25 \
  --report_to wandb

STATUS=$?
echo "[pilot] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
