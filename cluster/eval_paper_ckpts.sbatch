#!/bin/bash
# Evaluate the PAPER checkpoints (CPT / SFT / CPT+SFT / CPT+GRPO rows) straight
# from the hub, on whichever benchmark is missing.
#
# WHY A SECOND LAUNCHER: eval_sweep_finals.sbatch resolves everything under
# $PROJ/models/sweep and merges a LoRA adapter first. These are full models on
# the hub, so there is nothing to merge and nothing to clean up -- forcing them
# through that script would mean faking adapter paths.
#
# The CPT/SFT/CPT+GRPO rows currently have NO BMLAMA-53 numbers at all, and no
# PolyFact-Clean / 17-lang KLAR JSONs in this checkout (the curated sweep that
# produced the table's cells ran on the Trainium box and its result files were
# not copied back). Re-running them here also puts every row of the table on one
# pipeline, which is the property the comparison needs.
#
# Submit:
#   sbatch --export=ALL,BENCHMARK=bmlama53   cluster/eval_paper_ckpts.sbatch
#   sbatch --export=ALL,BENCHMARK=polyfact   cluster/eval_paper_ckpts.sbatch
#   sbatch --export=ALL,BENCHMARK=klar       cluster/eval_paper_ckpts.sbatch
#SBATCH --job-name=eval-paper
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-paper-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-paper-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

# hub_id | result_tag   (tags match the existing results/*.json naming)
ENTRIES=(
  "jvonrad/OLMo-2-1124-7B-TED|OLMo-2-1124-7B-TED"
  "jvonrad/olmo-2-7b-wikifact-sft|olmo-2-7b-wikifact-sft"
  "jvonrad/olmo-2-7b-aligned-wikifact-sft|olmo-2-7b-aligned-wikifact-sft"
  "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint|olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
  "jvonrad/Qwen-2.5-7B-TED|Qwen-2.5-7B-TED"
  "jvonrad/Qwen-2.5-7B-SFT-CE-random|Qwen-2.5-7B-SFT-CE-random"
  "jvonrad/Qwen-2.5-7B-TED-SFT|Qwen-2.5-7B-TED-SFT"
  "jvonrad/Qwen-2.5-7B-TED-grpo|Qwen-2.5-7B-TED-grpo"
  # index 8: the OTHER OLMo CPT checkpoint. The table's CPT row uses
  # OLMo-2-1124-7B-TED (created 2026-05-21, same day as the Qwen TED model), but
  # CPT+SFT and CPT+GRPO were both built in April on top of
  # olmo-2-7b-finetranslations (2026-04-02 -> +GRPO 04-06 -> +SFT 04-15). If the
  # two differ, the OLMo CPT block is reporting a different base from the one its
  # own derived rows were trained on. Submit with --array=8 to settle it.
  "jvonrad/olmo-2-7b-finetranslations|olmo-finetranslations"
  # 9-10: the untrained bases. Needed for the free-form PolyFact table -- without
  # them every row there is an absolute with no delta.
  "Qwen/Qwen2.5-7B|qwen-base"
  "allenai/OLMo-2-1124-7B|olmo-base"
)
IFS='|' read -r MODEL TAG <<< "${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
BENCHMARK="${BENCHMARK:-bmlama53}"
echo "[eval-paper] task=$SLURM_ARRAY_TASK_ID tag=$TAG bench=$BENCHMARK @ $(date -Is)"

case "$BENCHMARK" in
  bmlama53)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark bmlama53 --split test --model "$MODEL" \
      --device cuda --batch_size 16 --max_length 192 --score_mode byte \
      --langs en,de,id,pt,ar,bn,es,ru,fr,ja,zh \
      --output_json "$REPO/results/${TAG}_bmlama53_consistency.json"
    ;;
  polyfact)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark polyfact --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel \
      --split test --model "$MODEL" \
      --device cuda --batch_size 32 --max_length 192 --score_mode byte \
      --output_json "$REPO/results/${TAG}_polyfact_clean_consistency.json"
    ;;
  polyfact_freeform)
    FFSUF=""; [[ "${N_SHOT:-0}" != "0" ]] && FFSUF="_${N_SHOT}shot"
    python -u evaluate/evaluate_polyfact_freeform.py \
      --model "$MODEL" \
      --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel --split test \
      --batch_size 32 --max_new_tokens 16 --n_shot "${N_SHOT:-0}" \
      --output_json "$REPO/results/${TAG}_polyfact_freeform${FFSUF}.json"
    ;;
  gmmlu_lite)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark global_mmlu_lite --split test --model "$MODEL" \
      --device cuda --batch_size 16 --max_length 768 --score_mode byte \
      --langs en,de,id,pt,ar,bn,sw,es,fr,ja,zh \
      --output_json "$REPO/results/${TAG}_gmmlu_lite_consistency.json"
    ;;
  klar)
    # --langs is nargs="+": space-separated. Comma-separated silently scores 0
    # samples and still exits 0.
    python -u evaluate/evaluate_klar.py \
      --model "$MODEL" --tokenizer "$MODEL" \
      --device cuda --klar-root datasets/KLAR-CLC \
      --langs ar ca el en es fa fr he hu ja ko nl ru tr uk vi zh \
      --batch-size 32 --max-length 448 \
      --contamination-labels evaluate/alignments/klar_polyfact_contamination.json \
      --output-json "$REPO/results/klar/${TAG}_klar_alllangs.json"
    ;;
  *) echo "unknown BENCHMARK=$BENCHMARK"; exit 2 ;;
esac
STATUS=$?
echo "[eval-paper] $TAG / $BENCHMARK finished status $STATUS @ $(date -Is)"
exit $STATUS
