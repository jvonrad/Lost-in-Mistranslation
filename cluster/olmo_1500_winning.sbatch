#!/bin/bash
# OLMo at 1,500 facts with the WINNING recipe, matched to qwen-sweep-clip5.
#
# WHY: the OLMo clip sweep ran --max_train_samples 1000 (cluster/olmo_clip_sweep.sbatch),
# so `olmo-sw-clip5` is a 1,000-step run while the Qwen winner `qwen-sw-clip5` is
# 1,500. The two families have never been compared at matched steps under this
# recipe, and 1,500 is where the Qwen gains land (CLAUDE.md: 1,500 correct-LR
# steps reproduce the paper's 20,000-step checkpoint on MCQ and beat it 4x on
# KLAR). A 1,000-step OLMo may simply be undertrained rather than
# recipe-incompatible -- olmo-sw-clip5 gained KLAR (+1.9/+1.0) but nothing on
# TotCons (+0.05), which is the metric the paper's claim rests on.
#
# Every flag below is copied from the arm that won the sweep: clip 5.0,
# all_correct_bonus 1.0 (all_or_nothing), lr 1e-5, kl 0.0, LoRA r128, G=8,
# native scaffold, length bucketing. The ONLY differences from qwen-sweep-clip5
# are the model and --max_train_samples 1000 -> 1500.
#
# Deliberately NOT included: --empty_penalty and --brevity_penalty. Both are
# post-hoc fixes; adding them here would make this arm incomparable to the Qwen
# winner, which is the entire point of the run. OLMo also never showed the
# silence/padding drift (max gradient 5.61 vs Qwen's 1701), so the risk is low.
#
# ~1,500 steps x ~9.9 s = ~4.1 h on one GPU.
#
# Submit: sbatch cluster/olmo_1500_winning.sbatch
#SBATCH --job-name=olmo-1500
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo1500-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo1500-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

TAG="olmo-sweep-clip5-n1500"

echo "[olmo-1500] $TAG @ $(date -Is)"

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id allenai/OLMo-2-1124-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/sweep/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm 5.0 \
  --all_correct_bonus 1.0 \
  --reward_pooling group \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 1500 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 500 --max_eval_wikifact 150 --max_eval_mmlu 100 \
  --logging_steps 50 \
  --dead_run_patience 200 \
  --report_to wandb

STATUS=$?
echo "[olmo-1500] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
