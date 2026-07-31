#!/bin/bash
# Self-contained continuation job for the GRPO all_correct_bonus=0.0 ablation
# currently running inside job 5635241, which is about to hit its 24h wall
# limit. Same pattern as bonus5_ablation.sbatch: requests its own fresh
# 2-GPU allocation, training command IS the batch payload (immune to
# login-node session kill), --resume_from_checkpoint auto picks up from the
# latest accelerator.save_state checkpoint in the existing output_dir rather
# than restarting.
#
# Submit: sbatch cluster/nobonus_ablation_continue.sbatch
#SBATCH --job-name=grpo-nobonus-ablation
#SBATCH --account=brics.u6jh
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --output=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/slurm-grpo-nobonus-%j.out
#SBATCH --error=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation/logs/slurm-grpo-nobonus-%j.err

set -uo pipefail
REPO=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation

# See bonus5_ablation.sbatch for why: a freshly-scheduled node can start this
# script before its Lustre /projects mount is fully up.
for i in $(seq 1 30); do
    [[ -f "$REPO/training/train_wikifact_grpo_accelerate.py" ]] && break
    echo "[grpo-nobonus-ablation] waiting for $REPO to mount (attempt $i/30) ..."
    sleep 2
done
cd "$REPO"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate grpo
export TOKENIZERS_PARALLELISM=false

echo "grpo-nobonus-ablation: job $SLURM_JOB_ID on $(hostname) @ $(date -Is)"

accelerate launch --num_processes 2 --multi_gpu \
  training/train_wikifact_grpo_accelerate.py \
  --model_id Qwen/Qwen2.5-7B \
  --dataset_id jvonrad/WIKI-FACT \
  --output_dir "$REPO/models/qwen-2.5-7b-grpo-nobonus-ablation" \
  --run_name qwen-grpo-nobonus-ablation \
  --use_lora --bf16 \
  --kl_coef 0.0 \
  --all_correct_bonus 0.0 \
  --num_generations 8 \
  --learning_rate 1e-5 \
  --max_completion_length 48 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_train_samples 40000 \
  --num_train_epochs 1 \
  --eval_steps 200 \
  --max_eval_wikifact 100 \
  --gen_micro_batch_size 192 \
  --logprob_micro_batch_size 48 \
  --resume_from_checkpoint auto \
  --report_to wandb
