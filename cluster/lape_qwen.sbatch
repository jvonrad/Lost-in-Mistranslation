#!/bin/bash
# LAPE language-specialised-neuron analysis across the OLMo post-training methods.
#
# Depends on the per-language TED files that the LAHIS job builds
# (mechanistic-interpretability/data/ted/ted_<lang>.json), so submit this after
# that array, or with --dependency on it.
#
# ~2M tokens x 12 languages = 24M forward tokens per model. Token files are
# cached per tokenizer family under results/lape/_tokens_olmo, so only the first
# task pays the tokenisation cost.
#SBATCH --job-name=lape-qwen
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/lape-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/lape-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

ENTRIES=(
  "Qwen/Qwen2.5-7B|qwen-base"
  "$PROJ/models/qwen-sft-n10000/merged|qwen-sft-10k"
  "$PROJ/models/qwen-dco-n10000-r128/merged|qwen-dco-10k"
  "$PROJ/models/qwen-cmalign-n10000/merged|qwen-cmalign-10k"
  "$PROJ/models/sweep/_merged_qwen-sweep-clip5|qwen-grpo"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"

echo "[lape] task=$SLURM_ARRAY_TASK_ID tag=$TAG @ $(date -Is)"
python -u mechanistic-interpretability/run_lape.py --model "$MODEL" --tag "$TAG"
STATUS=$?
echo "[lape] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
