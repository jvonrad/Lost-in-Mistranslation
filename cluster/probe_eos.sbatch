#!/bin/bash
# Measure P(stop token | prompt) for the collapsed nobonus arm vs the base model.
#
# Merges the preserved adapter, probes, deletes the merged copy. See
# data_analysis/probe_eos_collapse.py for what is being tested and why.
#
# Submit: sbatch cluster/probe_eos.sbatch
#SBATCH --job-name=probe-eos
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/probe-eos-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/probe-eos-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

MERGED="$PROJ/models/final/_preserved/_merged_nobonus"

python -u - "$PROJ/models/final/_preserved/qwen-nobonus-s6000" Qwen/Qwen2.5-7B "$MERGED" <<'PY'
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
adapter, base_id, out = sys.argv[1:4]
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(base, adapter, dtype=torch.bfloat16).merge_and_unload()
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
try: AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
except Exception: AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
print("merged", flush=True)
PY

python -u data_analysis/probe_eos_collapse.py --model "$MERGED" --label "qwen-nobonus-s6000" \
  --output_json "$REPO/results/probe_eos_nobonus.json"
python -u data_analysis/probe_eos_collapse.py --model Qwen/Qwen2.5-7B --label "qwen-base (control)" \
  --output_json "$REPO/results/probe_eos_base.json"

STATUS=$?
rm -rf "$MERGED"
echo "[probe-eos] finished status $STATUS @ $(date -Is)"
exit $STATUS
