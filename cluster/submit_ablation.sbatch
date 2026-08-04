#!/bin/bash
# Submit-safe wrapper for the GRPO no-bonus ablation (see submit_cmalign.sbatch
# for why this needs to be its own Slurm job rather than a login-node
# background process: systemd KillUserProcesses=yes reaps the whole user
# session, including setsid-nohup'd children, when the SSH session ends).
#
# Submit:
#   sbatch --export=ALL,JOBID=5635241,GPUS=2,3 cluster/submit_ablation.sbatch
#SBATCH --job-name=ablation-orch
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ablation-orch-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/ablation-orch-%j.err

set -uo pipefail
REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
cd "$REPO"
echo "[submit_ablation] orchestrator job $SLURM_JOB_ID on $(hostname), targeting allocation JOBID=$JOBID"
bash cluster/run_grpo_ablation.sh
