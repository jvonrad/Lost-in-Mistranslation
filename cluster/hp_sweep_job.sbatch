#!/bin/bash
# One hyperparameter-sweep config as its own 1-GPU job.
#
# Per-config jobs instead of node waves because Isambard bills proportionally
# (--gpus=1 = 0.25 NHR/h, verified in sacct): a crashed config stops billing
# when it exits, no GPU idle-bills waiting for slower siblings, and 1-GPU jobs
# schedule in seconds while exclusive nodes can pend an hour.
# --cpus-per-task stays <= 72 so the GPU fraction (1/4) dominates the
# max-of-fractions billing rule.
#
# Submit via: bash cluster/hp_sweep.sh submit_rung1   (or submit_rung2 "tags")
#SBATCH --job-name=hp-sweep
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/hp_sweep/slurm-%x-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/hp_sweep/slurm-%x-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

: "${R:?need R}" "${LR:?need LR}" "${FACTS:?need FACTS}" "${TAG:?need TAG}"
mkdir -p "$PROJ/logs/hp_sweep" "$PROJ/models/hp_sweep"

echo "hp-sweep $TAG: job $SLURM_JOB_ID on $(hostname) @ $(date -Is) (r=$R alpha=$((2*R)) lr=$LR facts=$FACTS)"

# Unique rendezvous port: two sweep jobs can share a node (non-exclusive).
accelerate launch --num_processes 1 --main_process_port $(( 20000 + SLURM_JOB_ID % 9999 )) \
  training/train_wikifact_grpo_accelerate.py \
  --model_id Qwen/Qwen2.5-7B \
  --output_dir "$PROJ/models/hp_sweep/$TAG" \
  --run_name "hp-$TAG" \
  --use_lora --bf16 \
  --lora_r "$R" \
  --kl_coef 0.0 \
  --num_generations 8 \
  --learning_rate "$LR" \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples "$FACTS" --num_train_epochs 1 \
  --seed 42 \
  --logging_steps 25 \
  --eval_steps 500 --max_eval_wikifact 150 --max_eval_mmlu 100 \
  --resume_from_checkpoint auto \
  --report_to wandb \
  > "$PROJ/logs/hp_sweep/$TAG.log" 2>&1
RC=$?
echo "hp-sweep $TAG finished rc=$RC @ $(date -Is)"
exit $RC
