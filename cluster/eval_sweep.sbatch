#!/bin/bash
# Evaluation sweep on Isambard-AI GH200 — one array task per model, 1 GPU each.
#
# The Trainium batch-size trap (silently corrupted logits above --batch_size 8)
# does NOT apply here: that was a Neuron runtime memory limit on the fp32
# logsumexp, not a model issue. On a 96 GB GH200 the same code runs correctly at
# much larger batches, so the defaults below are tuned for CUDA. The sanity
# check still stands though — if a run reports near-chance accuracy (~0.25) for
# `en`, stop and investigate before trusting the rest.
#
# Submit one benchmark for all models in cluster/models.txt:
#   sbatch --export=ALL,BENCHMARK=polyfact    cluster/eval_sweep.sbatch
#   sbatch --export=ALL,BENCHMARK=gmmlu_lite  cluster/eval_sweep.sbatch
#   sbatch --export=ALL,BENCHMARK=klar        cluster/eval_sweep.sbatch
# A subset:  sbatch --array=0,4,10 --export=ALL,BENCHMARK=polyfact cluster/eval_sweep.sbatch
#
#SBATCH --job-name=eval-sweep
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11%12
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-%x-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-%x-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

BENCHMARK="${BENCHMARK:-polyfact}"
RESULTS="$REPO/results"
mkdir -p "$RESULTS" "$RESULTS/klar"

# Pick this array task's model out of the shared list.
mapfile -t ENTRIES < <(grep -vE '^\s*(#|$)' cluster/models.txt)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#ENTRIES[@]} )); then
  echo "array index $IDX >= ${#ENTRIES[@]} models — nothing to do"; exit 0
fi
read -r MODEL BASE <<< "${ENTRIES[$IDX]}"

echo "[eval_sweep] benchmark=$BENCHMARK model=$MODEL base=$BASE"
echo "[eval_sweep] job ${SLURM_ARRAY_JOB_ID:-?}_${IDX} on $(hostname) @ $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "$BENCHMARK" in
  polyfact)
    # PolyFact-Clean is the curated, eval-facing release (see CLAUDE.md).
    #
    # DO NOT pass --alignment_cache here. That cache was built for WIKI-FACT;
    # curation resampled distractors and reshuffled options, so against
    # PolyFact-Clean only 1,503/2,523 fact_ids even appear in it and just
    # 2,142/18,036 (11.9%) of its mappings are correct — and it fails SILENTLY,
    # producing garbage RankC/AnsAgr. PolyFact-Clean's `parallel` config stores
    # per-language option_ids (Wikidata QIDs), so the evaluator derives the
    # alignment exactly via align_by_option_ids with no flag at all.
    #
    # --score_mode byte (the script's current default, passed explicitly so the
    # sweep is reproducible if that default ever moves) = lm-eval acc_bytes:
    # tokenizer-independent, which matters when comparing across languages.
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark polyfact \
      --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel --split test \
      --model "$MODEL" \
      --device cuda --batch_size 32 --max_length 192 \
      --score_mode byte \
      --output_json "$RESULTS/${BASE}_polyfact_clean_consistency.json"
    ;;

  gmmlu_lite)
    # Global-MMLU-Lite has NO Russian config -> 11 languages. Long Bengali tail;
    # --max_length 1024 leaves ~1.5% of prompts right-truncated (model-agnostic).
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark global_mmlu_lite --split test \
      --model "$MODEL" \
      --device cuda --batch_size 8 --max_length 1024 \
      --score_mode byte \
      --langs en,de,id,pt,ar,bn,sw,es,fr,ja,zh \
      --output_json "$RESULTS/${BASE}_gmmlu_lite_consistency.json"
    ;;

  klar)
    # Generation eval: 6 langs, 3-shot template 1, greedy 10 tokens. Matches the
    # protocol used for the Trainium 12-model run so numbers stay comparable.
    python -u evaluate/evaluate_klar.py \
      --model "$MODEL" --tokenizer "$MODEL" \
      --device cuda --klar-root datasets/KLAR-CLC \
      --batch-size 32 --max-length 448 \
      --contamination-labels evaluate/alignments/klar_polyfact_contamination.json \
      --output-json "$RESULTS/klar/${BASE}_klar.json"
    ;;

  *)
    echo "unknown BENCHMARK=$BENCHMARK (want polyfact|gmmlu_lite|klar)"; exit 2 ;;
esac

STATUS=$?
echo "[eval_sweep] $BASE / $BENCHMARK finished with status $STATUS @ $(date -Is)"
exit $STATUS
