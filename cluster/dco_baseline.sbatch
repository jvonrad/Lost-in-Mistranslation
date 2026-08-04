#!/bin/bash
# DCO baseline (ConsistencyRL, ICML 2026) on PolyFact-Clean — reviewer-
# requested CLC baseline. See training/train_polyfact_dco.py for the port
# notes (loss verbatim + numerically verified; adapter-off reference; LoRA
# r128 family config; label-free consistency objective).
#
# Offline preference training, no rollouts: ~1-1.5 s/step at bs 4 on one
# GH200 -> 5k instances ≈ 0.5-1 h, 40k ≈ 3-5 h. 1 GPU each.
#
# Submit:
#   sbatch --export=ALL,N_INST=40000 cluster/dco_baseline.sbatch            (Qwen 40k)
#   sbatch --export=ALL,N_INST=5000  cluster/dco_baseline.sbatch            (Qwen 5k, their paper scale)
#   sbatch --export=ALL,N_INST=40000,MODEL_ID=allenai/OLMo-2-1124-7B,MTAG=olmo \
#          cluster/dco_baseline.sbatch                                      (OLMo 40k)
#SBATCH --job-name=dco-baseline
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-dco-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-dco-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MTAG="${MTAG:-qwen}"
N_INST="${N_INST:-40000}"
TAG="${MTAG}-dco-n${N_INST}-r128"

echo "dco-baseline: job $SLURM_JOB_ID on $(hostname) @ $(date -Is) (model=$MODEL_ID n=$N_INST)"

python -u training/train_polyfact_dco.py \
  --model_id "$MODEL_ID" \
  --output_dir "$PROJ/models/$TAG" \
  --run_name "$TAG" \
  --num_instances "$N_INST" \
  --beta 1.0 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 4 \
  --lora_r 128 \
  --seed 42 \
  --logging_steps 50 --save_steps 2000 \
  --report_to wandb
