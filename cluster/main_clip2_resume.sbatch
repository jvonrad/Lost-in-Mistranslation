#!/bin/bash
# Resume qwen-final-main from checkpoint-3000 with the clip tightened to 2.0.
#
# WHY: the clip5 main arm collapsed at step 3,050 -- the SAME failure that killed
# the ladder arm at 2,200, so this is a property of clip 5.0 at scale, not of the
# ladder:
#   step 3000  rew  7.56  grad 6.52   large gradient, clipped 6.52 -> 5.0
#   step 3050  rew -6.00  grad 0.00   policy destroyed, zero gradient, unrecoverable
# The eval at 3,000 was the run's BEST (res 0.998, acc 0.716, ffTotCons 0.180),
# so checkpoint-3000 is a genuinely good state to restart from.
#
# clip 2.0 rather than 1.0: clip 1.0 was measured to SUPPRESS generation transfer
# (Qwen KLAR -1.20 at clip1 vs +8.67 at clip5), so reverting all the way would
# throw away the main finding. 2.0 keeps steps meaningfully larger than the
# suppressive regime while capping the 6.5-and-up updates that did the damage.
# UNTESTED value -- 1.0, 5.0 and 20.0 are the measured points.
#
# CAVEAT FOR THE WRITE-UP: hybrid schedule -- steps 0-3,000 at clip 5.0 and
# 3,000+ at clip 2.0. Not a clean single-clip run.
#
# Submit: sbatch cluster/main_clip2_resume.sbatch
#SBATCH --job-name=main-clip2
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=17:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/main-clip2-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/main-clip2-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

CKPT="$PROJ/models/final/qwen-final-main-clip5/checkpoint-3000"
TAG="qwen-final-main-clip2-from3k"

echo "[main-clip2] resuming from $CKPT @ $(date -Is)"
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
echo "[main-clip2] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
