#!/bin/bash
# Run GRPO on the three GPUs assigned to nid010658. One srun task retains
# peer-GPU visibility for NCCL and starts three local workers with ranks 0-2.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation

JOBID="${JOBID:?set JOBID to an active Slurm allocation}"
MODEL_ID="${MODEL_ID:-allenai/OLMo-2-1124-7B}"
RUN_NAME="${RUN_NAME:-olmo-grpo-bonus0-ablation}"
OUT="${OUT:-$PROJ/models/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$REPO/logs/${RUN_NAME}.log}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29632}"
NUM_NODES=1
NUM_PROCESSES=3
EVAL_STEPS="${EVAL_STEPS:-50}"

run_worker() {
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate grpo
  cd "$REPO"
  export TOKENIZERS_PARALLELISM=false
  export WORLD_SIZE="$NUM_PROCESSES"
  python training/train_wikifact_grpo_accelerate.py \
    --model_id "$MODEL_ID" \
    --dataset_id jvonrad/WIKI-FACT \
    --output_dir "$OUT" \
    --run_name "$RUN_NAME" \
    --use_lora --bf16 \
    --kl_coef 0.0 \
    --all_correct_bonus 0.0 \
    --num_generations 8 \
    --learning_rate 1e-5 \
    --max_completion_length 48 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_train_samples 40000 \
    --num_train_epochs 1 \
    --eval_steps "$EVAL_STEPS" \
    --skip_periodic_eval \
    --max_eval_wikifact 100 \
    --gen_micro_batch_size 192 \
    --logprob_micro_batch_size 48 \
    --resume_from_checkpoint auto \
    --report_to wandb
}

if [[ "${1:-}" == worker ]]; then
  run_worker
  exit $?
fi

echo "[$(date -Is)] [ablation] job=$JOBID model=$MODEL_ID processes=$NUM_PROCESSES nodes=$NUM_NODES bonus=0.0 out=$OUT" | tee -a "$LOG_FILE"

export JOBID REPO MODEL_ID RUN_NAME OUT EVAL_STEPS NUM_PROCESSES MASTER_ADDR MASTER_PORT

srun --jobid="$JOBID" --overlap --exact -w nid010658 \
  --nodes=1 --ntasks=1 --cpus-per-task=216 --mem-per-gpu=100G --gpus-per-node=3 \
  bash -c "
    RANK=0 LOCAL_RANK=0 bash '$REPO/cluster/run_grpo_ablation_uneven.sh' worker & p0=\$!
    RANK=1 LOCAL_RANK=1 bash '$REPO/cluster/run_grpo_ablation_uneven.sh' worker & p1=\$!
    RANK=2 LOCAL_RANK=2 bash '$REPO/cluster/run_grpo_ablation_uneven.sh' worker & p2=\$!
    rc=0
    wait \$p0 || rc=\$?
    wait \$p1 || rc=\$?
    wait \$p2 || rc=\$?
    exit \$rc
  " >> "$LOG_FILE" 2>&1 &
node1_pid=$!

status=0
wait "$node1_pid" || status=$?

echo "[$(date -Is)] [ablation] srun finished with exit $status" | tee -a "$LOG_FILE"
exit "$status"
