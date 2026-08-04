#!/bin/bash
# Evaluate the two new-regime GRPO checkpoints on all four benchmarks, using
# exactly the protocols the CM-Align / DCO baselines were run under so the
# numbers drop straight into the comparison table.
#
# 2 models x 4 benchmarks = 8 array tasks, 1 GPU each.
#
# Protocol notes (all deliberate, see CLAUDE.md):
#  - polyfact : PolyFact-Clean `parallel` test (2,039 curated). NO --alignment_cache
#               (that cache is WIKI-FACT's and fails silently); option_ids give
#               the exact QID alignment. --score_mode byte.
#  - gmmlu     : Global-MMLU-Lite has no Russian -> 11 langs.
#  - klar      : ALL 17 languages, not evaluate_klar.py's 6-language default.
#               The default (es,fr,ru,zh,ja,ar — no English) is NOT comparable to
#               the paper's 7-language "Seen" split and previously produced a
#               false protocol-mismatch alarm.
#  - bmlama53  : 11 of our 12 languages (no `sw` config in BMLAMA-53).
#
# Submit after the merge job:
#   sbatch --dependency=afterok:<merge_jobid> cluster/eval_grpo_new.sbatch
#SBATCH --job-name=eval-grpo-new
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
# Measured runtimes: polyfact ~4 min, gmmlu ~5 min, bmlama ~8 min, klar-17lang
# ~14 min. 40 min is generous headroom. Do NOT inflate this — Slurm bills
# elapsed time so an over-request costs no budget, but it does cost QUEUE
# POSITION: a 40-minute job backfills into gaps a 5-hour job cannot.
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-grpo-new-%A_%a.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/eval-grpo-new-%A_%a.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

RESULTS="$REPO/results"
mkdir -p "$RESULTS" "$RESULTS/klar"

MODELS=("$PROJ/models/merged/qwen-grpo-s2400" "$PROJ/models/merged/olmo-grpo-s2000")
TAGS=("qwen-grpo-new-s2400" "olmo-grpo-new-s2000")
BENCHES=(polyfact gmmlu_lite klar bmlama53)

MI=$(( SLURM_ARRAY_TASK_ID / 4 ))
BI=$(( SLURM_ARRAY_TASK_ID % 4 ))
MODEL="${MODELS[$MI]}"; TAG="${TAGS[$MI]}"; BENCHMARK="${BENCHES[$BI]}"

echo "[eval] task=$SLURM_ARRAY_TASK_ID model=$TAG benchmark=$BENCHMARK on $(hostname) @ $(date -Is)"
if [[ ! -d "$MODEL" ]]; then echo "missing merged model $MODEL — did the merge job run?"; exit 3; fi

case "$BENCHMARK" in
  polyfact)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark polyfact \
      --hf_dataset jvonrad/PolyFact-Clean --hf_config parallel --split test \
      --model "$MODEL" \
      --device cuda --batch_size 32 --max_length 192 \
      --score_mode byte \
      --output_json "$RESULTS/${TAG}_polyfact_clean_consistency.json" ;;

  gmmlu_lite)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark global_mmlu_lite --split test \
      --model "$MODEL" \
      --device cuda --batch_size 8 --max_length 1024 \
      --score_mode byte \
      --langs en,de,id,pt,ar,bn,sw,es,fr,ja,zh \
      --output_json "$RESULTS/${TAG}_gmmlu_lite_consistency.json" ;;

  klar)
    python -u evaluate/evaluate_klar.py \
      --model "$MODEL" --tokenizer "$MODEL" \
      --device cuda --klar-root datasets/KLAR-CLC \
      --langs ar ca el en es fa fr he hu ja ko nl ru tr uk vi zh \
      --batch-size 32 --max-length 448 \
      --contamination-labels evaluate/alignments/klar_polyfact_contamination.json \
      --output-json "$RESULTS/klar/${TAG}_klar_alllangs.json" ;;

  bmlama53)
    python -u evaluate/evaluate_crosslingual_consistency.py \
      --benchmark bmlama53 --split test \
      --model "$MODEL" \
      --device cuda --batch_size 16 --max_length 192 \
      --score_mode byte \
      --langs en,de,id,pt,ar,bn,es,ru,fr,ja,zh \
      --output_json "$RESULTS/${TAG}_bmlama53_consistency.json" ;;

  *) echo "unknown BENCHMARK=$BENCHMARK"; exit 2 ;;
esac

echo "[eval] $TAG / $BENCHMARK finished status $? @ $(date -Is)"
