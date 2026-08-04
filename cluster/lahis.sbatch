#!/bin/bash
# LAHIS attention-head importance across the OLMo post-training methods.
#
# Task 0 additionally builds the per-language TED files the whole array needs,
# and the rest wait on it via a dependency (see the submit line below), because
# twelve concurrent writers to the same ted_<lang>.json would race.
#SBATCH --job-name=lahis
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/lahis-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/lahis-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# model | tag   -- five OLMo checkpoints on ONE protocol. Base is recomputed
# rather than reusing the committed tensors, whose data_num/max_length were
# never recorded.
ENTRIES=(
  "allenai/OLMo-2-1124-7B|olmo-base"
  "$PROJ/models/olmo-sft-n10000/merged|olmo-sft-10k"
  "$PROJ/models/olmo-dco-n10000-r128/merged|olmo-dco-10k"
  "$PROJ/models/olmo-cmalign-n10000/merged|olmo-cmalign-10k"
  "$PROJ/models/sweep/_merged_olmo-clip5.0|olmo-grpo"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"

# GRPO is a LoRA adapter; merge it once if the merged copy is absent.
if [[ "$TAG" == "olmo-grpo" && ! -f "$MODEL/config.json" ]]; then
  python -u - "$PROJ/models/sweep/olmo-clip5.0" allenai/OLMo-2-1124-7B "$MODEL" <<'PY'
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ad, base, out = sys.argv[1:4]
m = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu"),
    ad, dtype=torch.bfloat16).merge_and_unload()
os.makedirs(out, exist_ok=True); m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged", out)
PY
fi

EXTRA=""
[[ "$SLURM_ARRAY_TASK_ID" == "0" ]] && EXTRA="--prepare-data"

echo "[lahis] task=$SLURM_ARRAY_TASK_ID tag=$TAG @ $(date -Is)"
python -u mechanistic-interpretability/run_lahis.py $EXTRA --model "$MODEL" --tag "$TAG"
STATUS=$?
echo "[lahis] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
