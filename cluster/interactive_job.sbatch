#!/bin/bash
# One 24h interactive placeholder allocation on Isambard-AI.
# Attach to the running node with:
#   srun --jobid=<JOBID> --pty bash
# or ssh to the allocated node directly (`squeue -j <JOBID> -o '%N'`).
# Override GPU count at submit time (SBATCH directives don't expand shell vars):
#   sbatch --gpus=2 cluster/interactive_job.sbatch
#SBATCH --job-name=interactive
#SBATCH --account=brics.u6jh
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --gpus=4
#SBATCH --output=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/interactive-%j.out
#SBATCH --error=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/interactive-%j.err

echo "Interactive allocation $SLURM_JOB_ID started on $SLURMD_NODENAME at $(date -Is)"
echo "Attach with: srun --jobid=$SLURM_JOB_ID --pty bash"
sleep 86400
