#!/bin/bash
# EN-pivot DPO (CM-Align) baseline on Isambard-AI.
# Runs both phases end-to-end (construct preference data -> DPO train) on one GPU.
#
# Submit for a given base model, e.g.:
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B,TAG=qwen cluster/cmalign_dpo.sbatch
#   sbatch --export=ALL,MODEL_ID=allenai/OLMo-2-1124-7B,TAG=olmo cluster/cmalign_dpo.sbatch
# Or run the same python line by hand inside an interactive allocation.
#SBATCH --job-name=cmalign-dpo
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=06:00:00   # 40k measured ~8h (both phases); 20k ~4h + headroom
#SBATCH --gpus=1
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/cmalign-dpo-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/cmalign-dpo-%j.err

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
TAG="${TAG:-qwen}"
MAX_FACTS="${MAX_FACTS:-40000}"
NUM_CANDIDATES="${NUM_CANDIDATES:-4}"
# Extra trainer flags, e.g. generation batch sizing. OLMo-2-7B OOMs during
# the construct phase at the default gen_micro_batch_size 768.
EXTRA_ARGS="${EXTRA_ARGS:-}"
# PHASE=construct|train|all. `train` reuses preference pairs already cached at
# --pref_data_path, so a run killed during DPO can be resumed without redoing
# the (expensive) generation phase.
PHASE="${PHASE:-all}"

# The construct phase generates with a DynamicCache that grows by torch.cat each
# decode step and fragments the allocator badly. Job 5876778 died at fact 14,032
# of 20,000 after 4h with only 68 GiB genuinely allocated but 19.95 GiB
# reserved-and-unusable on a 95 GiB card. expandable_segments lets the allocator
# grow segments in place rather than stranding them.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Sets REPO/PROJ/HF_HOME and activates the grpo conda env. Note HF_HOME now
# points at /projects (not ~/.cache/huggingface), and we no longer force
# HF_HUB_OFFLINE=1 — compute nodes on this cluster have outbound internet.
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh

echo "CM-Align DPO: model=$MODEL_ID tag=$TAG facts=$MAX_FACTS cand=$NUM_CANDIDATES on $(hostname) @ $(date -Is)"

python -u "$REPO/training/train_wikifact_cmalign_dpo.py" \
  --phase "$PHASE" \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --pref_data_path "$PROJ/datasets/cmalign_pref_${TAG}_n${MAX_FACTS}" \
  --output_dir "$PROJ/models/${TAG}-cmalign-n${MAX_FACTS}" \
  --run_name "${TAG}-cmalign-n${MAX_FACTS}" \
  --max_facts "$MAX_FACTS" \
  --num_candidates "$NUM_CANDIDATES" \
  --beta 0.1 \
  --nll_gamma 0.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 \
  $EXTRA_ARGS
