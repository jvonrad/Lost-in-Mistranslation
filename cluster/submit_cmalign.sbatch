#!/bin/bash
# Submit-safe wrapper for the CM-Align sharded construct+train orchestrator.
#
# Runs as its own tiny Slurm job (1 CPU, no GPU) instead of a login-node
# background process. This matters: login nodes here have systemd
# KillUserProcesses=yes, which tears down a user's ENTIRE session cgroup —
# including setsid-nohup'd background processes — when their SSH session
# ends. That's what killed the previous run (all srun --overlap steps died
# simultaneously when the launching login-node shell's session was reaped).
# A proper sbatch job lives in Slurm's own cgroup and is immune to that.
#
# The actual GPU work still executes inside the target INTERACTIVE
# allocation via `srun --jobid=$JOBID --overlap` (see run_cmalign_sharded.sh)
# — this job just needs to survive long enough to issue those commands and
# wait on them; it requests no GPUs of its own.
#
# Submit:
#   sbatch --export=ALL,JOBID=5635241,GPUS=0,1,TAG=qwen,MAX_FACTS=40000 \
#     cluster/submit_cmalign.sbatch
#SBATCH --job-name=cmalign-orch
#SBATCH --account=brics.u6jh
#SBATCH --partition=workq
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/cmalign-orch-%j.out
#SBATCH --error=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/cmalign-orch-%j.err

set -uo pipefail
REPO=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation
cd "$REPO"
echo "[submit_cmalign] orchestrator job $SLURM_JOB_ID on $(hostname), targeting allocation JOBID=$JOBID"
bash cluster/run_cmalign_sharded.sh
