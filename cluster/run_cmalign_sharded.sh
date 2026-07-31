#!/bin/bash
# Orchestrate a sharded CM-Align / EN-pivot DPO run inside an existing Slurm
# allocation: run the (slow) construct phase in parallel across N GPUs, merge
# the per-shard preference datasets, then run the DPO train phase across those
# same N GPUs (accelerate --multi_gpu). Both phases checkpoint/resume, so a
# killed/time-limited run continues from the latest checkpoint on retry.
#
# Call this either as the payload of its own sbatch job (JOBID=$SLURM_JOB_ID,
# immune to login-node session kill -- see cluster/olmo_cmalign.sbatch) or as
# a login-node background process against an existing interactive allocation:
#   setsid nohup bash cluster/run_cmalign_sharded.sh > logs/cmalign_<tag>.log 2>&1 < /dev/null &
#
# Env knobs (all optional):
#   JOBID      interactive allocation id      (default: read cluster/state/current_job)
#   NODE       pin to one node of a multi-node allocation (default: unset, any node)
#   MODEL_ID   base model                     (default Qwen/Qwen2.5-7B)
#   TAG        short name for paths/run       (default qwen)
#   GPUS       comma list, one per shard      (default 0,1)
#   MAX_FACTS  total facts before sharding    (default 40000)
#   GEN_MICRO_BATCH_SIZE  construct-phase generation batch (default 768; lower
#                          if a model OOMs -- OLMo needed 384 on a 96GB GH200)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROJ=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation

JOBID="${JOBID:-$(cut -d' ' -f1 "$REPO/cluster/state/current_job" 2>/dev/null)}"
NODE="${NODE:-}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
TAG="${TAG:-qwen}"
GPUS="${GPUS:-0,1}"
MAX_FACTS="${MAX_FACTS:-40000}"
# 768 was tuned for Qwen; OLMo OOM'd at 768 on a 96GB GH200 (93GB+ in use at
# crash time) despite a similar/smaller vocab, so this is now a knob rather
# than hardcoded -- pass a lower value (e.g. 384) for models with a bigger
# per-sequence memory footprint.
GEN_MICRO_BATCH_SIZE="${GEN_MICRO_BATCH_SIZE:-768}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"

# On a multi-node allocation, srun needs an explicit node + gres or it can
# fail with "Insufficient GRES available" (seen for real targeting job
# 5638362, which spans 2 nodes) -- pin to one node when NODE is set.
ONE_GPU_FLAGS=()
ALL_GPU_FLAGS=()
NO_GPU_FLAGS=()
if [[ -n "$NODE" ]]; then
    ONE_GPU_FLAGS=(-w "$NODE" --nodes=1 --ntasks=1 --gres=gpu:1)
    ALL_GPU_FLAGS=(-w "$NODE" --nodes=1 --ntasks=1 --gres=gpu:"$NUM_SHARDS")
    NO_GPU_FLAGS=(-w "$NODE" --nodes=1 --ntasks=1)
fi
MERGED="$PROJ/datasets/cmalign_pref_${TAG}"
# "2.5-7b" was Qwen-specific and wrong for any other TAG (e.g. olmo); kept as
# the exact default for TAG=qwen only, so the already-running Qwen DPO train
# job's checkpoint dir still resolves the same way if this script is invoked
# again later. Any other TAG gets a clean "<tag>-cmalign-dpo" name, or pass
# OUT= explicitly to override.
_DEFAULT_OUT_NAME="${TAG}-cmalign-dpo"
[[ "$TAG" == "qwen" ]] && _DEFAULT_OUT_NAME="qwen-2.5-7b-cmalign-dpo"
OUT="${OUT:-$PROJ/models/$_DEFAULT_OUT_NAME}"

log() { echo "[$(date -Is)] [orch] $*"; }
CONDA='source ~/miniforge3/etc/profile.d/conda.sh; conda activate grpo'
COMMON_ENV='export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false'

log "job=$JOBID model=$MODEL_ID tag=$TAG shards=$NUM_SHARDS gpus=$GPUS max_facts=$MAX_FACTS"

# ── Phase 1: parallel construct, one shard per GPU ─────────────────────────
declare -a SHARD_DIRS
pids=()
for i in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$i]}"
    # With NODE set, --gres=gpu:1 cgroup-restricts the step to exactly one
    # physical GPU, always remapped to local index 0 -- setting
    # CUDA_VISIBLE_DEVICES to the job-level index (e.g. 1) then points at a
    # device that doesn't exist in that step's own view, so torch silently
    # falls back to CPU (confirmed live: torch.cuda.is_available()=False,
    # 272 CPU threads spinning, zero GPU memory used). Without NODE, no gres
    # restriction is applied and the step sees the job's full GPU set, so the
    # job-level index is correct there instead.
    cvd="$gpu"
    [[ -n "$NODE" ]] && cvd=0
    shard_dir="$PROJ/datasets/cmalign_pref_${TAG}_shard${i}"
    SHARD_DIRS+=("$shard_dir")
    log "launch construct shard $i on GPU $gpu (CUDA_VISIBLE_DEVICES=$cvd) -> $shard_dir"
    srun --jobid="$JOBID" --overlap "${ONE_GPU_FLAGS[@]}" bash -c "
      $CONDA
      cd $REPO
      $COMMON_ENV
      export CUDA_VISIBLE_DEVICES=$cvd
      python -u training/train_wikifact_cmalign_dpo.py \
        --phase construct --model_id '$MODEL_ID' --dataset_id jvonrad/WIKI-FACT \
        --pref_data_path '$shard_dir' --output_dir /tmp/unused_${TAG}_${i} \
        --max_facts $MAX_FACTS --num_shards $NUM_SHARDS --shard_index $i \
        --num_candidates 4 --facts_per_gen_batch 16 --gen_micro_batch_size $GEN_MICRO_BATCH_SIZE \
        --report_to none --bf16
    " > "$REPO/logs/cmalign_${TAG}_construct_shard${i}.log" 2>&1 &
    pids+=($!)
done

log "waiting for ${#pids[@]} construct shards ..."
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
if [[ "$fail" -ne 0 ]]; then
    log "ERROR: a construct shard failed; see logs/cmalign_${TAG}_construct_shard*.log"
    exit 1
fi
log "all construct shards done"

# ── Merge shard datasets ───────────────────────────────────────────────────
log "merging shards -> $MERGED"
DIRS_PY="[$(printf "'%s'," "${SHARD_DIRS[@]}")]"
srun --jobid="$JOBID" --overlap "${NO_GPU_FLAGS[@]}" bash -c "
  $CONDA
  python - <<PY
from datasets import load_from_disk, concatenate_datasets
dirs = $DIRS_PY
parts = [load_from_disk(d) for d in dirs]
merged = concatenate_datasets(parts)
merged.save_to_disk('$MERGED')
print('[merge] total pairs:', len(merged))
PY
" > "$REPO/logs/cmalign_${TAG}_merge.log" 2>&1
if [[ $? -ne 0 ]]; then log "ERROR: merge failed"; cat "$REPO/logs/cmalign_${TAG}_merge.log"; exit 1; fi
cat "$REPO/logs/cmalign_${TAG}_merge.log"

# ── Phase 2: DPO train on all shard GPUs (freed by construct finishing) ────
log "launch DPO train on GPUs $GPUS ($NUM_SHARDS procs) -> $OUT"
srun --jobid="$JOBID" --overlap "${ALL_GPU_FLAGS[@]}" bash -c "
  $CONDA
  cd $REPO
  $COMMON_ENV
  export CUDA_VISIBLE_DEVICES=$GPUS
  accelerate launch --num_processes $NUM_SHARDS --multi_gpu \
    training/train_wikifact_cmalign_dpo.py \
    --phase train --model_id '$MODEL_ID' \
    --pref_data_path '$MERGED' --output_dir '$OUT' \
    --run_name '${TAG}-cmalign-dpo' \
    --beta 0.1 --nll_gamma 0.0 --learning_rate 5e-6 --num_train_epochs 1 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --resume_from_checkpoint auto \
    --bf16 --report_to wandb
" > "$REPO/logs/cmalign_${TAG}_train.log" 2>&1
log "DPO train finished (exit $?); model at $OUT/merged"
