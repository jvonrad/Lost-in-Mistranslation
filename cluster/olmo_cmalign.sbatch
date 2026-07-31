#!/bin/bash
# Self-contained OLMo-2-7B CM-Align / EN-pivot DPO run (construct -> merge ->
# DPO train, sharded across 2 GPUs). Requests its own fresh allocation so it
# doesn't compete with other jobs on the shared account for GPUs.
#
# The construct/merge/train orchestration IS the batch payload (calls
# run_cmalign_sharded.sh directly, targeting this job's own $SLURM_JOB_ID via
# --overlap steps) rather than a placeholder interactive shell launched
# separately from a login node -- runs under Slurm's own job cgroup for its
# whole lifetime, immune to the login-node session-kill issue (systemd
# KillUserProcesses=yes) that killed earlier runs launched via setsid nohup.
#
# Both phases now have full checkpoint/resume support (construct saves every
# --checkpoint_every facts; DPO train uses accelerator.save_state every
# --save_steps), so re-submitting this same script after a time-limit kill
# continues instead of restarting.
#
# Submit: sbatch cluster/olmo_cmalign.sbatch
#SBATCH --job-name=olmo-cmalign
#SBATCH --account=brics.u6jh
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/slurm-olmo-cmalign-%j.out
#SBATCH --error=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/slurm-olmo-cmalign-%j.err

set -uo pipefail
REPO=/home/u6jh/jvonrad.u6jh/Lost-in-Mistranslation
cd "$REPO"

echo "olmo-cmalign: job $SLURM_JOB_ID on $(hostname) @ $(date -Is)"

JOBID="$SLURM_JOB_ID" MODEL_ID=allenai/OLMo-2-1124-7B TAG=olmo GPUS=0,1,2,3 MAX_FACTS=40000 \
  bash cluster/run_cmalign_sharded.sh
