#!/bin/bash
# Placeholder allocation that holds a whole node so work can be iterated on
# without re-queueing after every crash. Run steps into it from anywhere with:
#   srun --jobid=<JOBID> --overlap <command>
# or attach a shell:
#   srun --jobid=<JOBID> --pty bash
# (`squeue -j <JOBID> -o '%N'` shows the node.)
#
# SBATCH directives can't expand shell vars, so override at submit time:
#   sbatch --time=01:00:00 cluster/interactive_job.sbatch     # one hour
#   sbatch --gpus=2        cluster/interactive_job.sbatch
#
# NOTE: it sleeps for its whole wall time and is billed for it whether or not
# anything is running — scancel it as soon as you're done.
#SBATCH --job-name=interactive
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/interactive-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/interactive-%j.err

echo "Interactive allocation $SLURM_JOB_ID started on $SLURMD_NODENAME at $(date -Is)"
echo "Attach with: srun --jobid=$SLURM_JOB_ID --pty bash"
sleep 86400
