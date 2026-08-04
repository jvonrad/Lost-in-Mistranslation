#!/bin/bash
# Self-contained GRPO all_correct_bonus=5.0 ablation job (the high-consistency-
# reward counterpart to the running bonus=0.0 ablation). Requests its own
# fresh 2-GPU allocation rather than reusing another job, so it can't collide
# with anything else running on the shared account.
#
# The training command IS the batch payload (not a placeholder interactive
# shell + a separately-launched srun step) — this runs under Slurm's own job
# cgroup for its whole lifetime, immune to the login-node session-kill issue
# (systemd KillUserProcesses=yes) that killed earlier runs launched via
# setsid nohup from a login node.
#
# QOS caps a single job at 24h wall time, but this ablation needs ~48h at 2
# GPUs. --resume_from_checkpoint auto (default) means re-submitting this same
# script after it hits the time limit continues from the latest checkpoint
# instead of restarting — check cluster/status.sh / the run's own log to see
# how far it got, then just `sbatch cluster/bonus5_ablation.sbatch` again.
#
# Submit: sbatch cluster/bonus5_ablation.sbatch
#SBATCH --job-name=grpo-bonus5-ablation
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-bonus5-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-bonus5-%j.err

set -uo pipefail
# REPO is the actual git checkout (code) -- lives under /home, NOT /projects.
# /projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation is a separate, pre-existing
# *data/output-only* directory (datasets/, models/, logs/) that has never
# contained training/, evaluate/, cluster/, etc. Every earlier "Lustre mount
# race" failure blamed on this script (nid011229, nid010530, and again on
# nid010320 via the interactive-job variant of this same command) was actually
# this wrong-path bug, not a timing issue -- waiting longer never helped
# because the file was never going to appear under /projects.
REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
cd "$REPO"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate grpo
export TOKENIZERS_PARALLELISM=false

echo "grpo-bonus5-ablation: job $SLURM_JOB_ID on $(hostname) @ $(date -Is)"

accelerate launch --num_processes 2 --multi_gpu \
  training/train_wikifact_grpo_accelerate.py \
  --model_id Qwen/Qwen2.5-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/qwen-grpo-bonus50-ablation" \
  --run_name qwen-grpo-bonus50-ablation \
  --use_lora --bf16 \
  --kl_coef 0.0 \
  --all_correct_bonus 5.0 \
  --num_generations 8 \
  --learning_rate 1e-5 \
  --max_completion_length 48 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_train_samples 40000 \
  --num_train_epochs 1 \
  --eval_steps 200 \
  --max_eval_wikifact 100 \
  --gen_micro_batch_size 192 \
  --logprob_micro_batch_size 48 \
  --resume_from_checkpoint auto \
  --report_to wandb
