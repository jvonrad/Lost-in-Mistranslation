#!/bin/bash
# MEXA representation alignment for the post-trained models, both families.
# Metric functions are imported from XScript-Pretraining; only the HF plumbing
# and the 12-language FLORES+ loading are new. RAW (uncentered) is the variant
# of interest -- see the script docstring for why centering would subtract the
# effect under test.
#SBATCH --job-name=mexa
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/mexa-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/mexa-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

ENTRIES=(
  "allenai/OLMo-2-1124-7B|olmo-base"
  "$PROJ/models/olmo-sft-n10000/merged|olmo-sft-10k"
  "$PROJ/models/olmo-dco-n10000-r128/merged|olmo-dco-10k"
  "$PROJ/models/olmo-cmalign-n10000/merged|olmo-cmalign-10k"
  # The OLMo GRPO row of the results table is olmo-2-7b-grpo-att-mlp-full, so the
  # mechanistic analysis must describe THAT checkpoint. The earlier run used
  # olmo-clip5.0 (the 1,000-step sweep arm), which is both a different model and
  # not data-matched to the 10k baselines it was being compared against.
  "jvonrad/olmo-2-7b-grpo-att-mlp-full|olmo-grpo-attmlp"
  "Qwen/Qwen2.5-7B|qwen-base"
  "$PROJ/models/qwen-sft-n10000/merged|qwen-sft-10k"
  "$PROJ/models/qwen-dco-n10000-r128/merged|qwen-dco-10k"
  "$PROJ/models/qwen-cmalign-n10000/merged|qwen-cmalign-10k"
  "$PROJ/models/sweep/_merged_qwen-sweep-clip5|qwen-grpo"
  # 10: CPT+GRPO -- the checkpoint the paper's LAPE figure used (English
  # 1,671 -> 2,310, +38.2%). Appended rather than inserted so the existing
  # task indices keep pointing at the same models.
  "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint|olmo-cpt-grpo"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
echo "[mexa] task=$SLURM_ARRAY_TASK_ID tag=$TAG @ $(date -Is)"
python -u evaluate/evaluate_mexa_alignment.py --model "$MODEL" --tag "$TAG" --centered
STATUS=$?
echo "[mexa] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
