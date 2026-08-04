#!/bin/bash
# Launch (or resume) a GRPO all_correct_bonus ablation inside an existing
# Slurm allocation, pinned to a GPU subset via CUDA_VISIBLE_DEVICES (and
# optionally a specific node, for multi-node allocations).
#
# Runs on the LOGIN node (issues one srun --overlap step on the compute node).
# Meant to be launched with setsid nohup, protected by the trainer's own
# checkpoint/resume support (accelerator.save_state + training_stats.json) —
# a killed login-node session costs at most one --eval_steps interval, not
# the whole run.
#
# Env knobs (all optional):
#   JOBID       target allocation id            (default: read cluster/state/current_job)
#   NODE        pin to one node of a multi-node allocation (default: unset, any node)
#   MODEL_ID    base model                      (default Qwen/Qwen2.5-7B)
#   GPUS        comma list of physical GPUs     (default 2,3)
#   MAX_TRAIN_SAMPLES  total facts              (default 40000)
#   ALL_CORRECT_BONUS  cross-lingual consistency bonus ablation value (default 0.0)
#   RUN_NAME    wandb run name / output subdir  (default qwen-grpo-bonus<X>-ablation)
#   OUT         output dir                      (default models/<RUN_NAME>)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation

JOBID="${JOBID:-$(cut -d' ' -f1 "$REPO/cluster/state/current_job" 2>/dev/null)}"
NODE="${NODE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
GPUS="${GPUS:-2,3}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-40000}"
ALL_CORRECT_BONUS="${ALL_CORRECT_BONUS:-0.0}"
BONUS_TAG="$(echo "$ALL_CORRECT_BONUS" | tr -d '.')"
RUN_NAME="${RUN_NAME:-qwen-grpo-bonus${BONUS_TAG}-ablation}"
OUT="${OUT:-$PROJ/models/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$REPO/logs/${RUN_NAME}.log}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_PROCESSES="${#GPU_ARR[@]}"

log() { echo "[$(date -Is)] [ablation] $*"; }
CONDA='source ~/miniforge3/etc/profile.d/conda.sh; conda activate grpo'

NODE_FLAGS=()
if [[ -n "$NODE" ]]; then
    NODE_FLAGS=(-w "$NODE" --nodes=1 --ntasks=1 --gres=gpu:"$NUM_PROCESSES")
fi

log "job=$JOBID node=${NODE:-any} model=$MODEL_ID gpus=$GPUS num_processes=$NUM_PROCESSES " \
    "all_correct_bonus=$ALL_CORRECT_BONUS run_name=$RUN_NAME out=$OUT"

srun --jobid="$JOBID" --overlap "${NODE_FLAGS[@]}" bash -c "
  $CONDA
  cd $REPO
  export TOKENIZERS_PARALLELISM=false
  export CUDA_VISIBLE_DEVICES=$GPUS
  accelerate launch --num_processes $NUM_PROCESSES --multi_gpu \
    training/train_wikifact_grpo_accelerate.py \
    --model_id '$MODEL_ID' \
    --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
    --output_dir '$OUT' \
    --run_name '$RUN_NAME' \
    --use_lora --bf16 \
    --kl_coef 0.0 \
    --all_correct_bonus $ALL_CORRECT_BONUS \
    --num_generations 8 \
    --learning_rate 1e-5 \
    --max_completion_length 48 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --num_train_epochs 1 \
    --eval_steps 200 \
    --max_eval_wikifact 100 \
    --gen_micro_batch_size 192 \
    --logprob_micro_batch_size 48 \
    --resume_from_checkpoint auto \
    --report_to wandb
" > "$LOG_FILE" 2>&1
log "ablation srun step finished (exit $?)"
