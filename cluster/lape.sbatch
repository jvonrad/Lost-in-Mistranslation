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
#SBATCH --job-name=lape
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
  "allenai/OLMo-2-1124-7B|olmo-base"
  "$PROJ/models/olmo-sft-n10000/merged|olmo-sft-10k"
  "$PROJ/models/olmo-dco-n10000-r128/merged|olmo-dco-10k"
  "$PROJ/models/olmo-cmalign-n10000/merged|olmo-cmalign-10k"
  # The OLMo GRPO row of the results table is olmo-2-7b-grpo-att-mlp-full, so the
  # mechanistic analysis must describe THAT checkpoint. The earlier run used
  # olmo-clip5.0 (the 1,000-step sweep arm), which is both a different model and
  # not data-matched to the 10k baselines it was being compared against.
  "jvonrad/olmo-2-7b-grpo-att-mlp-full|olmo-grpo-attmlp"
  # 5: CPT+GRPO -- the checkpoint the PAPER's LAPE figure used. It reports
  # English 1,671 -> 2,310 (+38.2%); olmo-grpo-att-mlp-full gives only
  # 1,762 -> 1,898 (+7.7%) under the same pipeline, so the published effect
  # belongs to this model, not the plain-GRPO one.
  "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint|olmo-cpt-grpo"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"

echo "[lape] task=$SLURM_ARRAY_TASK_ID tag=$TAG @ $(date -Is)"
python -u mechanistic-interpretability/run_lape.py --model "$MODEL" --tag "$TAG"
STATUS=$?
echo "[lape] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
