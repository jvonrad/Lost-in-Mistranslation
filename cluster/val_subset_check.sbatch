#!/bin/bash
# Is the in-loop eval SET the reason the 1,500-step sweep runs outscore the
# 10,000-step final runs at matched steps?
#
# The sweep arms ran --max_eval_wikifact 150, the final arms 150 -> 100. The val
# set is built as val_ds.select(range(N)), so the 100-fact set is the FIRST 100
# of the 150-fact set: the two evals are nested, not identical, and every
# sweep-vs-final metric comparison is therefore across different item pools.
#
# This measures the offset directly on the UNTRAINED base model: metrics over
# val[0:100] vs val[0:150] vs the 50-fact tail val[100:150]. Any gap here is a
# pure eval-set artifact present in every comparison, at every step, for free.
#
# Submit: sbatch cluster/val_subset_check.sbatch
#SBATCH --job-name=val-subset
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/val-subset-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/val-subset-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

python -u data_analysis/val_subset_offset.py --model Qwen/Qwen2.5-7B
STATUS=$?
echo "[val-subset] finished status $STATUS @ $(date -Is)"
exit $STATUS
