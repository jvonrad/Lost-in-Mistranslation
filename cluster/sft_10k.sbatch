#!/bin/bash
# Retrain the SFT rows on 10K facts so they are FACT-MATCHED to every other
# post-2026-08 arm (GRPO, DCO, CM-Align all use 10K PolyFact-Clean facts).
#
# WHY: the released SFT checkpoints were trained on 40K facts, so the current
# SFT rows compare a 4x-data method against 10K-data methods. That is the same
# confound we removed for DCO and CM-Align by retraining them at 10K.
#
# CAVEAT FOR THE WRITE-UP: this changes TWO things at once versus the released
# checkpoints -- fact count (40K -> 10K) AND dataset (the released ones predate
# the PolyFact-Clean switch). So the new rows are the right FAIR-COMPARISON
# rows, but the old-vs-new difference is not a clean data-scaling ablation.
#
# --consistency_weight 0.0 = PURE CROSS-ENTROPY. This is deliberate and is NOT
# the trainer's default (0.5, which is also what the module docstring's example
# passes). The SFT row exists as the supervised control: plain next-token
# supervision on the parallel data, with no explicit cross-lingual term. Leaving
# the auxiliary consistency loss on would put a consistency objective inside the
# baseline that GRPO's consistency reward is being compared against, which is
# exactly the contrast the paper is drawing. The released Qwen checkpoint name
# (`Qwen-2.5-7B-SFT-CE-random`) carries the same "CE" reading.
# Everything else matches the released recipe: LoRA r64/alpha128 (hardcoded in
# the trainer, not a flag), 1 epoch, lr 2e-5.
#
# The trainer writes the adapter to $output_dir and a merged model to
# $output_dir/merged, which is what the eval launcher consumes.
#
# Submit (4 arms, ~1 GPU-h each):
#   sbatch cluster/sft_10k.sbatch
#SBATCH --job-name=sft-10k
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-3
# N_FACTS is overridable: the data-scaling curve needs the same recipe at 1,500.
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/sft10k-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/sft10k-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# base_model | tag
# The CPT bases are the ones the user confirmed 2026-08-03: OLMo CPT is
# olmo-2-7b-finetranslations (NOT OLMo-2-1124-7B-TED), Qwen CPT is
# Qwen-2.5-7B-TED.
ENTRIES=(
  "allenai/OLMo-2-1124-7B|olmo-sft-n${N_FACTS:-10000}"
  "Qwen/Qwen2.5-7B|qwen-sft-n${N_FACTS:-10000}"
  "jvonrad/olmo-2-7b-finetranslations|olmo-cpt-sft-n${N_FACTS:-10000}"
  "jvonrad/Qwen-2.5-7B-TED|qwen-cpt-sft-n${N_FACTS:-10000}"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"

echo "[sft-10k] task=$SLURM_ARRAY_TASK_ID model=$MODEL tag=$TAG @ $(date -Is)"

python -u training/train_multilingual_consistency_lora_sft.py \
  --model_id "$MODEL" \
  --hf_dataset_id jvonrad/PolyFact-Clean --hf_dataset_config parallel \
  --max_train_facts "${N_FACTS:-10000}" \
  --num_epochs 1 \
  --facts_per_device_batch 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --consistency_weight 0.0 \
  --ckpt_dir "$SCRATCH/sft_ckpt_$TAG" \
  --output_dir "$PROJ/models/$TAG" \
  --run_name "$TAG" \
  --save_total_limit 1 \
  --report_to wandb

STATUS=$?
echo "[sft-10k] $TAG finished status $STATUS @ $(date -Is)"
exit $STATUS
