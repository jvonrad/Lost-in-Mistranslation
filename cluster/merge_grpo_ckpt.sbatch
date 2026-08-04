#!/bin/bash
# Merge a mid-run GRPO LoRA checkpoint into its base model so the eval scripts
# (which take a plain --model path, no adapter support) can load it.
#
# The trainer only writes a merged/ directory at the END of a run; these are
# checkpoints from runs stopped early (the cosine-LR bug — see CLAUDE.md), so
# the merge has to happen here.
#
# Merging on a login node is NOT possible: the 4 GiB user cgroup kills any 7B
# load with a bare "Killed". Hence a compute-node job.
#
# Submit: sbatch cluster/merge_grpo_ckpt.sbatch
#SBATCH --job-name=merge-grpo
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:20:00   # measured: ~1m20s per 7B merge; short = better backfill
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/merge-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/merge-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# <checkpoint dir> <base model id> <merged output dir>
ENTRIES=(
  "$PROJ/models/qwen-grpo-full-r128-lr1e-5/checkpoint-2400|Qwen/Qwen2.5-7B|$PROJ/models/merged/qwen-grpo-s2400"
  "$PROJ/models/olmo-grpo-full-r128-lr1e-5/checkpoint-2000|allenai/OLMo-2-1124-7B|$PROJ/models/merged/olmo-grpo-s2000"
)
IFS='|' read -r CKPT BASE OUT <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
echo "[merge] ckpt=$CKPT base=$BASE out=$OUT @ $(date -Is)"

python -u - "$CKPT" "$BASE" "$OUT" <<'PY'
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ckpt, base_id, out = sys.argv[1:4]
print(f"loading base {base_id} ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
print(f"applying adapter {ckpt} ...", flush=True)
model = PeftModel.from_pretrained(base, ckpt, dtype=torch.bfloat16)
model = model.merge_and_unload()

# Sanity: the merge must actually change the weights. A zero-init or silently
# unloaded adapter would produce a byte-identical copy of the base and the eval
# would quietly re-measure the baseline.
ref = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
deltas = [
    (n, (p - dict(ref.named_parameters())[n]).abs().max().item())
    for n, p in list(model.named_parameters())[:400]
    if n in dict(ref.named_parameters())
]
changed = [(n, d) for n, d in deltas if d > 0]
print(f"merge check: {len(changed)}/{len(deltas)} inspected tensors changed; "
      f"max |delta| = {max([d for _, d in deltas], default=0.0):.6f}", flush=True)
assert changed, "adapter merge was a no-op — refusing to write a base-model copy"

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
# Tokenizer: prefer the checkpoint's own copy (it carries any pad-token fixes
# the trainer made), fall back to the base.
try:
    AutoTokenizer.from_pretrained(ckpt).save_pretrained(out)
except Exception:
    AutoTokenizer.from_pretrained(base_id).save_pretrained(out)
print(f"wrote {out}", flush=True)
PY

echo "[merge] task $SLURM_ARRAY_TASK_ID finished status $? @ $(date -Is)"
