#!/bin/bash
# EN-pivot DPO (CM-Align) baseline on Isambard-AI.
# Runs both phases end-to-end (construct preference data -> DPO train) on one GPU.
#
# Submit for a given base model, e.g.:
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B,TAG=qwen cluster/cmalign_dpo.sbatch
#   sbatch --export=ALL,MODEL_ID=allenai/OLMo-2-1124-7B,TAG=olmo cluster/cmalign_dpo.sbatch
# Or run the same python line by hand inside an interactive allocation.
#SBATCH --job-name=cmalign-dpo
#SBATCH --account=brics.u6jh
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/cmalign-dpo-%j.out
#SBATCH --error=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/cmalign-dpo-%j.err

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
TAG="${TAG:-qwen}"
MAX_FACTS="${MAX_FACTS:-40000}"
NUM_CANDIDATES="${NUM_CANDIDATES:-4}"

PROJ=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Activate the grpo conda env (miniforge, no conda init on Isambard).
source ~/miniforge3/etc/profile.d/conda.sh
conda activate grpo

# LaBSE + base model are pre-cached in ~/.cache/huggingface; stay offline on the node.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "CM-Align DPO: model=$MODEL_ID tag=$TAG facts=$MAX_FACTS cand=$NUM_CANDIDATES on $(hostname) @ $(date -Is)"

python -u "$REPO/training/train_wikifact_cmalign_dpo.py" \
  --phase all \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/WIKI-FACT \
  --pref_data_path "$PROJ/datasets/cmalign_pref_${TAG}" \
  --output_dir "$PROJ/models/${TAG}-2.5-7b-cmalign-dpo" \
  --run_name "${TAG}-cmalign-dpo" \
  --max_facts "$MAX_FACTS" \
  --num_candidates "$NUM_CANDIDATES" \
  --beta 0.1 \
  --nll_gamma 0.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16
