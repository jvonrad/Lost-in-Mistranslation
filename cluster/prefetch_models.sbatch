#!/bin/bash
# Pre-download every paper checkpoint + dataset into $HF_HOME on Lustre.
#
# Why this is a Slurm job and not a login-node loop: Isambard-AI login nodes
# cap each user at a 4 GiB memory cgroup (/sys/fs/cgroup/user.slice/...
# /memory.max). Parallel HF downloads — especially with the Xet backend —
# blow straight through that and the process dies with a bare "Killed".
# Compute nodes have 460 GB and outbound internet, so the fetch runs there.
# A CPU-only (--gpus=0) job just pends on workq, hence --gpus=1.
#
# Submit: sbatch cluster/prefetch_models.sbatch
#SBATCH --job-name=prefetch
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/prefetch-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/prefetch-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

echo "prefetch: job ${SLURM_JOB_ID:-none} on $(hostname) @ $(date -Is)  HF_HOME=$HF_HOME"

python -u - <<'PY'
from huggingface_hub import snapshot_download

MODELS = [
    # --- OLMo-2-1124-7B family (README model list) ---
    "allenai/OLMo-2-1124-7B",                                          # Base
    "jvonrad/OLMo-2-1124-7B-TED",                                      # CPT
    "jvonrad/olmo-2-7b-wikifact-sft",                                  # SFT
    "jvonrad/olmo-2-7b-aligned-wikifact-sft",                          # CPT+SFT
    "jvonrad/olmo-2-7b-grpo-att-mlp-full",                             # GRPO (paper's)
    "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint",  # CPT+GRPO
    # --- Qwen2.5-7B family ---
    "Qwen/Qwen2.5-7B",                                                 # Base
    "jvonrad/Qwen-2.5-7B-TED",                                         # CPT
    "jvonrad/Qwen-2.5-7B-SFT-CE-random",                               # SFT
    "jvonrad/Qwen-2.5-7B-TED-SFT",                                     # CPT+SFT
    "jvonrad/Qwen-2.5-7B-grpo-consistent",                             # GRPO
    "jvonrad/Qwen-2.5-7B-TED-grpo",                                    # CPT+GRPO
    # --- cross-lingual encoder used for RankC option alignment ---
    "sentence-transformers/LaBSE",
]
DATASETS = ["jvonrad/PolyFact-Clean", "jvonrad/WIKI-FACT", "jvonrad/PolyFact"]

ok, bad = [], []
for m in MODELS:
    try:
        snapshot_download(m, ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.h5"],
                          max_workers=8)
        print(f"OK    model   {m}", flush=True); ok.append(m)
    except Exception as e:
        print(f"FAIL  model   {m}: {type(e).__name__}: {e}", flush=True); bad.append(m)
for d in DATASETS:
    try:
        snapshot_download(d, repo_type="dataset", max_workers=8)
        print(f"OK    dataset {d}", flush=True); ok.append(d)
    except Exception as e:
        print(f"FAIL  dataset {d}: {type(e).__name__}: {e}", flush=True); bad.append(d)

print(f"\nprefetch summary: {len(ok)} ok, {len(bad)} failed")
for b in bad:
    print("  FAILED:", b)
PY

echo "prefetch done @ $(date -Is); cache size:"
du -sh "$HF_HOME"
