#!/bin/bash
# Shared environment for Isambard-AI phase 2, project u6sg (`brics.u6sg`).
# Source this at the top of any sbatch payload or interactive step:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Ported from the previous u6jh allocation. Two directories, do not confuse
# them (this was a recurring bug in the u6jh scripts):
#   REPO — the git checkout, i.e. the CODE (training/, evaluate/, cluster/).
#   PROJ — Lustre data/output-only tree (datasets/, models/, logs/). It has
#          never contained code; a script that cd's to PROJ and calls
#          `python training/...` fails with "no such file", which used to get
#          misdiagnosed as a Lustre mount race.

export ACCOUNT=brics.u6sg
export REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
export PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
export SCRATCH="${SCRATCH:-/scratch/u6sg/jvonrad.u6sg}"

# Caches must stay off /home — it has a hard 101 GB quota (see CLAUDE.md).
export HF_HOME=/projects/u6sg/jvonrad.u6sg/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export TOKENIZERS_PARALLELISM=false
# wandb's local run dir defaults to ./wandb in CWD — which is the REPO on
# /home. Small per run, but it accumulates against the 101 GB quota; keep it
# on Lustre with everything else.
export WANDB_DIR="$PROJ/logs/wandb"
mkdir -p "$WANDB_DIR" 2>/dev/null || true

# Compute nodes on this cluster DO have outbound internet (verified: hf.co and
# pypi.org both reachable from nid*), so unlike the u6jh setup we do not force
# HF_HUB_OFFLINE=1. Set it yourself if you want to guarantee no hub traffic.

# miniforge, no `conda init` on Isambard.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "grpo" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate grpo
fi

mkdir -p "$PROJ/logs" "$PROJ/models" "$PROJ/datasets" "$TMPDIR"
