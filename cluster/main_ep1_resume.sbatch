#!/bin/bash
# Resume qwen-final-main from checkpoint-3000 with BOTH failure modes covered.
#
# This checkpoint has now been resumed twice and died twice -- but of two
# DIFFERENT causes, and only one lever was in place each time:
#
#   clip 5.0 (job 5880532_0), died step 3,050 -- MODE A, one-shot destruction:
#     3000  rew  7.56  grad 6.52   single above-clip update
#     3050  rew -6.00  grad 0.00   pure token salad in all 12 languages
#                                  ("strugg NotSupportedException (longleftrightarrow")
#     -6.00 = 12 x -0.5, the worst score reachable. Gradient clipping is the
#     lever here, and clip 5.0 was too loose to stop it.
#
#   clip 2.0 (job 5888747), died ~step 3,270 -- MODE B, the silence hole:
#     3050..3250  rew 12.06 / 8.19 / 4.38   250 steps of HEALTHY training,
#                                           no catastrophic update -> clip 2.0
#                                           did its job on mode A
#     3250  eval resolution_rate 0.500      drifting into unparseable output
#     3300+ rew 0.00  std 0.00  grad 0.00   all 12 languages EMPTY
#     Empty scored 0.0 while unparseable scored -0.5, so the cheapest escape
#     from garbled output was one EOS token. --empty_penalty is the lever.
#
# So: keep clip 2.0 (covers A, demonstrated) and add --empty_penalty 1.0
# (covers B), which reorders the reward to
#     correct +1 > valid-wrong 0 > unparseable -0.5 > empty -1
# and makes the garbage->silence drift uphill instead of downhill.
#
# HONEST RESIDUAL RISK: any state where all G rollouts score alike is absorbing,
# because std 0 -> advantages 0 -> zero gradient. --empty_penalty does not remove
# the all-unparseable (-6.00) state, it only makes silence unattractive.
# --dead_run_patience 200 is the backstop: it aborts ~25 min after a run enters
# ANY such state instead of burning the rest of the allocation.
#
# CAVEAT FOR THE WRITE-UP: hybrid run -- steps 0-3,000 at clip 5.0 / empty
# penalty 0.0, then clip 2.0 / empty penalty 1.0. Not a clean single-config arm.
#
# 7,000 remaining steps x ~9 s = ~17.5 h.
#
# Submit: sbatch cluster/main_ep1_resume.sbatch
#SBATCH --job-name=main-ep1
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=19:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/main-ep1-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/main-ep1-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

CKPT="$PROJ/models/final/qwen-final-main-clip5/checkpoint-3000"
TAG="qwen-final-main-ep1-from3k"

echo "[main-ep1] resuming from $CKPT @ $(date -Is)"
[[ -f "$CKPT/optimizer.bin" ]] || { echo "checkpoint incomplete at $CKPT"; exit 3; }

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id Qwen/Qwen2.5-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/final/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm 2.0 \
  --empty_penalty 1.0 \
  --dead_run_patience 200 \
  --all_correct_bonus 1.0 \
  --reward_pooling group \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 10000 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 250 --max_eval_wikifact 100 --max_eval_mmlu 100 \
  --logging_steps 50 \
  --resume_from_checkpoint "$CKPT" \
  --report_to wandb

STATUS=$?
echo "[main-ep1] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
