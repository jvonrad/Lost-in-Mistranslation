#!/bin/bash
# Full production GRPO run with the rung-1 sweep winner's hyperparameters.
# (Rung 2 deliberately skipped — time constraint; the 3e-5 short-horizon trap
# self-eliminated in rung 1, so the 1e-5 vs 3e-6 ranking is the stable part.)
#
# 4-GPU DDP: 40k facts / 4 ranks = 10k optimizer steps ≈ 22 h wall ≈ 87 GPU-h,
# fits ONE 24 h QOS window (2-GPU would need a resubmit at ~44 h wall for the
# same GPU-h). Effective optimizer batch = 4 facts/step; keep every run in the
# post-2026-08-02 family (per_lang baseline etc.) at 4 GPUs for comparability.
#
# Paper method: group reward pooling + all_correct_bonus 1.0, PolyFact-Clean,
# native scaffold, no-letter-credit matcher, length bucketing.
#
# Submit: sbatch --export=ALL,R=<rank>,LR=<lr>[,MODEL_ID=...,MTAG=...] cluster/full_grpo.sbatch
#   MODEL_ID defaults to Qwen/Qwen2.5-7B (MTAG qwen); for OLMo pass
#   MODEL_ID=allenai/OLMo-2-1124-7B MTAG=olmo. Same LoRA targets apply —
#   OLMo-2 uses LLaMA-style module names (q/k/v/o/gate/up/down_proj).
#SBATCH --job-name=grpo-full
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-full-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-full-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

: "${R:?need R (winner lora_r)}" "${LR:?need LR (winner learning rate)}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MTAG="${MTAG:-qwen}"
TAG="${MTAG}-grpo-full-r${R}-lr${LR}"

echo "grpo-full: job $SLURM_JOB_ID on $(hostname) @ $(date -Is) (model=$MODEL_ID r=$R alpha=$((2*R)) lr=$LR)"

accelerate launch --num_processes 4 --multi_gpu \
  training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --output_dir "$PROJ/models/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r "$R" \
  --learning_rate "$LR" \
  --kl_coef 0.0 \
  --reward_pooling group \
  --all_correct_bonus 1.0 \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 40000 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 200 --max_eval_wikifact 150 --max_eval_mmlu 200 \
  --resume_from_checkpoint auto \
  --report_to wandb
