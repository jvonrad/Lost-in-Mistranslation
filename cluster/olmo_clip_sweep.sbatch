#!/bin/bash
# Minimal OLMo clipping probe: max_grad_norm 1.0 (current default) vs 5.0.
#
# WHY OLMo NEEDS ITS OWN PROBE: clipping bites very differently across families.
# Measured pre-clip gradient norms on the production runs were median 138.9
# (Qwen) vs 3.67 (OLMo). Both were clipped on ~99% of steps, but raising the
# threshold to 5.0 leaves MOST OLMo steps unclipped while Qwen stays clipped at
# every step. A clip value chosen on the Qwen sweep is therefore an
# extrapolation across a ~37x difference in how hard the intervention bites.
#
# 1,000 facts on 1 GPU = 1,000 optimizer steps at 1 fact/step (the paper's
# semantics). OLMo measures ~9.9 s/step, so ~2.8 h each; 2 arms in parallel,
# ~5.5 GPU-h total. Deliberately shorter than the 1,500-step Qwen sweep to
# protect wall-clock for the full run that follows.
#
# Rank on the SAVED final checkpoints against the full 2,039-fact PolyFact-Clean
# test, not the 150-fact in-loop eval (std err ~0.6pp vs ~2.1pp on TotCons).
#
# Submit: sbatch cluster/olmo_clip_sweep.sbatch
#SBATCH --job-name=olmo-clip
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo-clip-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo-clip-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

CLIPS=(1.0 5.0)
CLIP="${CLIPS[$SLURM_ARRAY_TASK_ID]}"
TAG="olmo-clip${CLIP}"
echo "[olmo-clip] task=$SLURM_ARRAY_TASK_ID max_grad_norm=$CLIP on $(hostname) @ $(date -Is)"

python -u training/train_wikifact_grpo_accelerate.py \
  --model_id allenai/OLMo-2-1124-7B \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/sweep/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --learning_rate 1e-5 \
  --kl_coef 0.0 \
  --max_grad_norm "$CLIP" \
  --reward_pooling group \
  --all_correct_bonus 1.0 \
  --num_generations 8 \
  --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
  --per_device_train_batch_size 1 \
  --length_bucketing --gen_micro_batch_size 32 --logprob_micro_batch_size 48 \
  --max_train_samples 1000 --num_train_epochs 1 \
  --seed 42 \
  --eval_steps 250 --max_eval_wikifact 150 --max_eval_mmlu 100 \
  --logging_steps 25 \
  --report_to wandb

STATUS=$?
echo "[olmo-clip] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
