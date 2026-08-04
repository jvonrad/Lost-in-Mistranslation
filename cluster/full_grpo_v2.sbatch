#!/bin/bash
# Corrected full production GRPO run. Submitted EARLY (with a dependency on the
# sweep) so it accrues queue priority while the sweep runs; the winning config
# is injected later via an env file, not baked in at submit time.
#
# Fixes since the killed runs (all in the trainer, see CLAUDE.md):
#   * cosine LR now spans the real step count (was consumed num_processes x too
#     fast -> lr hit 0.0 at step 2500 of 10000)
#   * resume-exact data ordering (seeded generator + set_epoch +
#     skip_first_batches) so a resubmit continues instead of re-walking facts
#   * grad_norm printed to stdout; --max_grad_norm and --task_format exposed
#
# CONFIG INJECTION: writes nothing itself. If cluster/full_run.env exists it is
# sourced and may set EXTRA (extra trainer flags), MODEL_ID, MTAG. If it does
# NOT exist the run proceeds with the corrected-control config — which is
# exactly the fallback we would choose anyway, so an unattended start is safe.
#
# Submit:
#   sbatch --dependency=afterany:<sweep_jobid> cluster/full_grpo_v2.sbatch
#   sbatch --dependency=afterany:<sweep_jobid> \
#          --export=ALL,MODEL_ID=allenai/OLMo-2-1124-7B,MTAG=olmo \
#          cluster/full_grpo_v2.sbatch
#SBATCH --job-name=grpo-full-v2
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/grpo-full-v2-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/grpo-full-v2-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MTAG="${MTAG:-qwen}"
EXTRA="${EXTRA:-}"

# Late-bound config from the sweep analysis. Sourced AFTER the defaults above so
# it can override MODEL_ID/MTAG/EXTRA, but it is entirely optional.
if [[ -f "$REPO/cluster/full_run.env" ]]; then
  echo "[full-v2] sourcing cluster/full_run.env:"; cat "$REPO/cluster/full_run.env"
  source "$REPO/cluster/full_run.env"
else
  echo "[full-v2] no cluster/full_run.env — running corrected-control config"
fi

TAG="${MTAG}-grpo-v2-r128-lr1e-5${TAG_SUFFIX:-}"
echo "[full-v2] job $SLURM_JOB_ID on $(hostname) @ $(date -Is)"
echo "[full-v2] model=$MODEL_ID tag=$TAG extra='$EXTRA'"

accelerate launch --num_processes 4 --multi_gpu \
  training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
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
  --report_to wandb \
  $EXTRA

echo "[full-v2] $TAG finished status $? @ $(date -Is)"
