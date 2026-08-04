#!/bin/bash
# Final production runs, ONE GPU PER RUN.
#
# WHY 1 GPU RATHER THAN 4 — this is not just a scheduling workaround:
#   * Step wall-time is independent of GPU count (DDP parallelises WITHIN a
#     step), so 23h buys ~10,000 steps either way. GPUs buy FACTS PER STEP.
#   * 1 fact/step is the setup every good result tonight was produced under:
#     the 1,500-step single-GPU clip5 arm hit +8.67 KLAR, while a 2,540-step
#     4-GPU run managed -0.21.
#   * A 23h x 4-GPU job needs a whole node held for a day; 239 nodes on workq
#     were already "planned" for other jobs. 1-GPU jobs start in seconds here.
#   * 5 runs x 1 GPU = ~115 GPU-h, versus 184 for two 4-GPU runs.
#
# Arms 0-4 use --max_grad_norm 5.0, the one intervention measured on BOTH
# families (Qwen KLAR -1.20 -> +8.67; OLMo -4.58 -> +1.34).
#
# Arm 5 (ladder-clip1) OVERRIDES that back to 1.0. Arm 3 (ladder + clip5)
# CATASTROPHICALLY COLLAPSED at step ~2,200: the ladder drove reward to
# saturation (14 of a possible 17), so most groups had zero reward variance and
# hence zero gradient; the rare group that did have variance produced a
# grad_norm of 10.96, which clip5 passed through at ~half strength and which
# destroyed the policy (reward pinned at -6.00 = 12 x -0.5, resolution 0.000,
# gradient 0.00, unrecoverable). The tight clip is exactly the guard against
# that rare large step, and the sweep measured ladder-at-clip1 at +8.17 KLAR
# on its own -- so this arm tests the ladder WITHOUT the interaction that
# killed arm 3. The last flag wins in argparse, so the trailing
# --max_grad_norm 1.0 overrides the 5.0 set below.
#
# 10,000 facts for BOTH models, so the comparison is fact-aligned across methods
# (GRPO / DCO / CM-Align) AND across models. At 1 fact/step that is 10,000
# optimizer steps. Measured 8.0 s/step (Qwen) and 10.0 s/step (OLMo), both
# including periodic evals, so:
#   Qwen  ~22.2h -> completes inside the 23h limit
#   OLMo  ~27.8h -> WILL BE KILLED at ~8,280 steps (cosine 83% done, lr ~6.5e-7).
#     Deliberate: the last periodic checkpoint is used, and round aligned fact
#     counts were preferred over a completed anneal.
#
# Submit: sbatch cluster/final_runs.sbatch
#         sbatch --array=0,1 cluster/final_runs.sbatch    # main runs only
#SBATCH --job-name=final-grpo
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/final-grpo-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/final-grpo-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# name | model_id | mtag | n_facts | extra flags
ENTRIES=(
  "main|Qwen/Qwen2.5-7B|qwen|10000|--all_correct_bonus 1.0"
  "main|allenai/OLMo-2-1124-7B|olmo|10000|--all_correct_bonus 1.0"
  "nobonus|Qwen/Qwen2.5-7B|qwen|10000|--all_correct_bonus 0.0"
  "ladder|Qwen/Qwen2.5-7B|qwen|10000|--bonus_shape ladder --bonus_ladder 9:1,10:2,11:3,12:5"
  "ladder|allenai/OLMo-2-1124-7B|olmo|10000|--bonus_shape ladder --bonus_ladder 6:1,8:2,10:3,12:5"
)
IFS='|' read -r NAME MODEL_ID MTAG NFACTS EXTRA <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
TAG="${MTAG}-final-${NAME}-clip5"

# OLMo plateaus near k~5.7 and never reaches 12, so the Qwen rungs (9/10/11/12)
# would fire on 5-10% of steps -- and a bonus no rollout in the group of 8
# reaches contributes EXACTLY ZERO after z-scoring. Hence the rescaled rungs.

echo "[final] task=$SLURM_ARRAY_TASK_ID tag=$TAG facts=$NFACTS extra='$EXTRA' on $(hostname) @ $(date -Is)"

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/final/$TAG" \
  --run_name "$TAG" \
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
  --max_train_samples "$NFACTS" --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 500 --max_eval_wikifact 100 --max_eval_mmlu 100 \
  --logging_steps 50 \
  --report_to wandb \
  $EXTRA

STATUS=$?
echo "[final] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
