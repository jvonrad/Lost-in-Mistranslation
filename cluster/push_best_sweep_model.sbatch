#!/bin/bash
# Merge the best 1,500-step sweep checkpoint and push it to the HF hub.
#
# BEST ARM = qwen-sweep-clip5 (max_grad_norm 5.0, lr 1e-5, kl 0, all-or-nothing
# bonus). It is simultaneously the best arm on KLAR (+8.67) and on PolyFact
# TotCons (+2.35) of everything swept, and matches the paper's 20,000-step
# checkpoint on MCQ metrics while beating it ~4x on generation.
#
# AUTH REQUIRED. There is no token on this allocation. Before submitting, run
# ONE of these (in an interactive shell, not here):
#   huggingface-cli login                       # writes $HF_HOME/token
#   printf '%s' "<token>" > $HF_HOME/token      # HF_HOME is set by cluster/env.sh
# The token needs WRITE scope. The job aborts early with a clear message if it
# cannot authenticate, rather than failing after the 15 GB merge.
#
# Repo defaults to PRIVATE. Flip it on the hub, or pass PRIVATE=0, once you are
# happy for it to be public.
#
# Submit: sbatch cluster/push_best_sweep_model.sbatch
#         sbatch --export=ALL,PRIVATE=0,REPO_ID=jvonrad/Some-Other-Name cluster/push_best_sweep_model.sbatch
#SBATCH --job-name=push-hf
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/push-hf-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/push-hf-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

ADAPTER="${ADAPTER:-$PROJ/models/sweep/qwen-sweep-clip5}"
BASE="${BASE:-Qwen/Qwen2.5-7B}"
REPO_ID="${REPO_ID:-jvonrad/Qwen-2.5-7B-grpo-clip5-1500}"
PRIVATE="${PRIVATE:-1}"

echo "[push] adapter=$ADAPTER base=$BASE repo=$REPO_ID private=$PRIVATE @ $(date -Is)"

python -u - "$ADAPTER" "$BASE" "$REPO_ID" "$PRIVATE" <<'PY'
import sys, os, torch
from huggingface_hub import HfApi, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, base_id, repo_id, private = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"

# Fail fast on auth, BEFORE spending ~5 min merging 15 GB.
try:
    who = whoami()
    print(f"authenticated as: {who.get('name')}", flush=True)
except Exception as e:
    sys.exit(f"NOT AUTHENTICATED ({e!r}).\n"
             f"Run `huggingface-cli login` (write scope) or write a token to "
             f"{os.environ.get('HF_HOME','$HF_HOME')}/token, then resubmit.")

if not os.path.isfile(os.path.join(adapter, "adapter_model.safetensors")):
    sys.exit(f"no adapter at {adapter}")

print("merging ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter, dtype=torch.bfloat16).merge_and_unload()

# Never publish a base-model copy by accident.
ref = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
rp = dict(ref.named_parameters())
d = max((( p - rp[n]).abs().max().item() for n, p in list(model.named_parameters())[:400] if n in rp), default=0.0)
print(f"merge check: max |delta| = {d:.6f}", flush=True)
assert d > 0, "adapter merge was a no-op - refusing to push a base-model copy"
del ref, rp

tok = AutoTokenizer.from_pretrained(adapter)
api = HfApi(); api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

card = f"""---
base_model: {base_id}
license: apache-2.0
tags: [grpo, cross-lingual, factual-recall, consistency, polyfact]
---

# {repo_id.split('/')[-1]}

GRPO on **PolyFact-Clean** (`parallel`), 1,500 facts at 1 fact/optimizer step,
LoRA r128/alpha256 merged into {base_id}. ~3.3 GPU-hours on one GH200.

The distinguishing hyperparameter is **`max_grad_norm=5.0`**. With the more
common `1.0`, ~99% of steps are renormalised and free-form generation transfer
is suppressed; loosening it is worth roughly +10 points of KLAR accuracy on this
setup, at no cost to the multiple-choice metrics.

## Results

Deltas are vs the {base_id} base model, same evaluation pipeline.

| benchmark | metric | base | this model | delta |
|---|---|---|---|---|
| PolyFact-Clean test (n=2,039) | accuracy | 51.25 | 55.97 | **+4.72** |
| | total consistency | 5.35 | 7.70 | **+2.35** |
| | RankC@4 | 62.36 | 64.00 | **+1.63** |
| KLAR, 17 languages | accuracy | 40.54 | 49.21 | **+8.67** |
| | 10 unseen languages | 35.59 | 43.74 | **+8.15** |
| | decontaminated subset | 39.96 | 48.42 | **+8.46** |

Scoring is byte-normalised (lm-eval `acc_bytes`); cross-lingual option alignment
uses PolyFact-Clean's stored Wikidata `option_ids`, so it is exact rather than
inferred.

## Training configuration

- reward: number of correct languages (of 12) + 1.0 if all 12 correct, pooled
  over languages, advantage z-scored within each group of G=8 rollouts
- `max_grad_norm` 5.0, `learning_rate` 1e-5, `kl_coef` 0.0, G=8,
  `max_completion_length` 48, temperature 0.7, top_p 0.95, seed 42
- native-language prompt scaffold; answers matched to option text, with no
  credit for bare option letters

## Limitations

Single seed. Trained on the same distribution as the PolyFact-Clean test split,
so the PolyFact numbers are in-domain; KLAR is a separate dataset and is the
out-of-distribution evidence. Consistency gains are smaller than a
purely-consistency-driven objective achieves in-domain, and the accompanying
paper reports that comparison.
"""
os.makedirs("/tmp/_card", exist_ok=True)
open("/tmp/_card/README.md", "w").write(card)

print(f"pushing to {repo_id} (private={private}) ...", flush=True)
model.push_to_hub(repo_id, private=private)
tok.push_to_hub(repo_id, private=private)
api.upload_file(path_or_fileobj="/tmp/_card/README.md", path_in_repo="README.md",
                repo_id=repo_id, repo_type="model")
print(f"DONE: https://huggingface.co/{repo_id}", flush=True)
PY

STATUS=$?
echo "[push] finished status $STATUS @ $(date -Is)"
exit $STATUS
