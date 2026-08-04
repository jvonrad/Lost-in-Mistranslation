#!/bin/bash
# GRPO baseline WITHOUT cross-lingual reward pooling (meta-review request).
#
# --reward_pooling per_lang: each (fact, language) is its own GRPO group —
# own-language reward only, advantage z-scored within that language's G
# rollouts and applied only to its own generation. No pooled reward, no
# broadcast advantage, no all_correct_bonus (undefined without pooling; passed
# as 0.0 explicitly to keep the config self-documenting).
#
# Everything else matches the paper-family ablation runs (qwen nobonus/bonus5):
# same data volume (40k facts), G=8, lr 1e-5, kl 0, LoRA — the ONLY delta vs
# the nobonus ablation is the pooling structure, so together the three runs
# dissect: full method (pooling+bonus) vs pooling-only (nobonus) vs neither
# (this). NOTE the runs from 2026-08-02 onward additionally use PolyFact-Clean,
# the native prompt scaffold and the no-letter-credit matcher — so compare this
# baseline against a matched re-run of the method, not against the older
# WIKI-FACT/English-scaffold checkpoints.
#
# Rollout compute is identical to the pooled method (~3.9 s/fact bucketed);
# 40k facts / 2 GPUs ≈ 22 h wall ≈ 44 GPU-h. Fits one 24 h QOS window; if it
# spills, --resume_from_checkpoint auto + resubmit continues it.
#
# Watch polyfact/mcq_* and consistency/rankc_avg vs the pooled run, and expect
# a nonzero per-language degenerate-group fraction (hard languages with all-G
# identical rewards get zero gradient — the mechanism this baseline exposes).
#
# Submit: sbatch cluster/perlang_baseline.sbatch                       (Qwen)
#         sbatch --export=ALL,MODEL_ID=allenai/OLMo-2-1124-7B,MTAG=olmo \
#                cluster/perlang_baseline.sbatch                       (OLMo)
#SBATCH --job-name=grpo-perlang
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-perlang-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/slurm-grpo-perlang-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MTAG="${MTAG:-qwen}"
TAG="${MTAG}-grpo-perlang-r128-lr1e-5"
echo "grpo-perlang-baseline: job $SLURM_JOB_ID on $(hostname) @ $(date -Is) (model=$MODEL_ID)"

accelerate launch --num_processes 4 --multi_gpu \
  training/train_wikifact_grpo_accelerate.py \
  --model_id "$MODEL_ID" \
  --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
  --output_dir "$PROJ/models/$TAG" \
  --run_name "$TAG" \
  --use_lora --bf16 \
  --lora_r 128 \
  --kl_coef 0.0 \
  --reward_pooling per_lang \
  --all_correct_bonus 0.0 \
  --num_generations 8 \
  --learning_rate 1e-5 \
  --max_completion_length 48 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_train_samples 40000 \
  --num_train_epochs 1 \
  --length_bucketing \
  --gen_micro_batch_size 32 \
  --logprob_micro_batch_size 48 \
  --eval_steps 200 \
  --max_eval_wikifact 100 \
  --max_eval_mmlu 200 \
  --resume_from_checkpoint auto \
  --report_to wandb
