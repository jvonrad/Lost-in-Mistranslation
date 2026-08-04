#!/bin/bash
# GRPO rollout-efficiency probe: sweep --num_generations at FIXED batch size 1.
#
# Answers three things the batch-size sweep structurally cannot, all from
# logging the trainer already does (no extra instrumentation):
#
#  1. DEGENERATE-GROUP RATE. compute_group_advantages z-scores rewards within a
#     fact: advantages = (rewards - mean) / (std + 1e-6). If all G rollouts of a
#     fact score the same, std=0 -> every advantage is 0 -> that fact yields
#     EXACTLY ZERO gradient after paying full generation + fwd + bwd. At
#     per_device_train_batch_size=1 the trainer's printed `reward_std` IS that
#     within-group std (the rewards list is exactly this rank's one fact x G
#     rollouts), so the fraction of steps with reward_std ~ 0 is a direct
#     measurement of wasted optimizer steps. This is why bs is pinned to 1 here:
#     at bs>1 the logged std pools facts and the measurement is destroyed.
#
#  2. MARGINAL COST OF G. Generation scales linearly in G; if the degenerate
#     rate at G=8 is high, raising G buys gradient signal — this prices it.
#
#  3. COMPLETION-LENGTH WASTE. cumulative rollout_tok counts NON-PAD completion
#     tokens, so mean tokens/sequence vs --max_completion_length 48 shows how
#     much decode is spent on padding. HF generate runs every sequence in a
#     micro-batch out to the longest one, so if answers are ~10 tokens the cap
#     is ~4x oversized and shrinking it is a free generation speedup.
#
# Usage (inside an allocation):
#   CUDA_VISIBLE_DEVICES=2,3 bash cluster/gen_pilot.sh [outdir] [steps] [G...]
set -uo pipefail

source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

OUTDIR="${1:-$PROJ/logs/gen_pilot}"
STEPS="${2:-6}"
shift 2 2>/dev/null || true
GENS=("${@:-}")
[[ -z "${GENS[0]:-}" ]] && GENS=(4 8 16)

MODEL="${MODEL_ID:-Qwen/Qwen2.5-7B}"
NPROC=2
PORT="${MAIN_PROCESS_PORT:-29600}"   # must differ from the bs sweep's port
DEADLINE="${PILOT_DEADLINE:-0}"      # epoch seconds; 0 = no guard

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf "num_gen\tstatus\tstep_s\tdegenerate_rate\tmean_tok_per_seq\tpeak_mem_gb\n" > "$SUMMARY"

echo "=== gen pilot: model=$MODEL gpus=${CUDA_VISIBLE_DEVICES:-all} steps=$STEPS G=${GENS[*]} ==="

ANY_OK=0

for G in "${GENS[@]}"; do
  if [[ "$DEADLINE" != "0" ]] && (( $(date +%s) > DEADLINE )); then
    echo "!! deadline reached — SKIPPING remaining num_generations: $G onward"
    printf "%s\tSKIPPED_DEADLINE\tnan\tnan\tnan\tnan\n" "$G" >> "$SUMMARY"
    continue
  fi

  LOG="$OUTDIR/g${G}.log"
  SAMPLES=$(( 1 * NPROC * STEPS ))
  echo "--- num_generations=$G (bs=1, max_train_samples=$SAMPLES) -> $LOG"

  timeout 1800 accelerate launch --num_processes "$NPROC" --multi_gpu \
    --main_process_port "$PORT" \
    training/train_wikifact_grpo_accelerate.py \
    --model_id "$MODEL" \
    --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
    --output_dir "$OUTDIR/run_g${G}" \
    --run_name "gen-pilot-${G}" \
    --use_lora --bf16 \
    --kl_coef 0.0 \
    --num_generations "$G" \
    --learning_rate 1e-5 \
    --max_completion_length 48 \
    --temperature 0.7 --top_p 0.95 \
    --per_device_train_batch_size 1 \
    --max_train_samples "$SAMPLES" \
    --num_train_epochs 1 \
    --gen_micro_batch_size 192 \
    --logprob_micro_batch_size 48 \
    --logging_steps 1 \
    --eval_steps 0 \
    --max_eval_mmlu 1 --max_eval_flores 1 --max_eval_wikifact 1 \
    --resume_from_checkpoint none \
    --report_to none \
    > "$LOG" 2>&1
  RC=$?

  if grep -qiE "out of memory|torch.OutOfMemoryError" "$LOG"; then
    STATUS=OOM
  elif [[ $RC -eq 124 ]]; then
    STATUS=TIMEOUT
  elif [[ $RC -ne 0 ]]; then
    STATUS="FAIL_rc${RC}"
  else
    STATUS=OK
  fi

  read -r STEP_S DEGEN TOKSEQ PEAK < <(
    python - "$LOG" "$G" "$NPROC" <<'PY'
import ast, sys
log, G, nproc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for line in open(log, errors="ignore"):
    line = line.strip()
    if line.startswith("{") and "'step'" in line and "step_s" in line:
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            pass
rows = rows[2:]  # drop warmup
if not rows:
    print("nan nan nan nan"); raise SystemExit

n = len(rows)
step_s = sum(r["step_s"] for r in rows) / n
peak = max(r["peak_mem_gb"] for r in rows)

# A group is degenerate when all G rollouts scored identically -> zero gradient.
degen = sum(1 for r in rows if abs(r["reward_std"]) < 1e-9) / n

# rollout_tok is cumulative and summed over ranks; each step generates
# nproc ranks x 1 fact x 12 langs x G sequences.
seqs_per_step = nproc * 12 * G
deltas = [b["rollout_tok"] - a["rollout_tok"] for a, b in zip(rows, rows[1:])]
tok_per_seq = (sum(deltas) / len(deltas) / seqs_per_step) if deltas else float("nan")

print("%.3f %.3f %.2f %.2f" % (step_s, degen, tok_per_seq, peak))
PY
  )

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$G" "$STATUS" "$STEP_S" "$DEGEN" "$TOKSEQ" "$PEAK" >> "$SUMMARY"
  echo "    -> $STATUS  step_s=$STEP_S  degenerate=$DEGEN  tok/seq=$TOKSEQ (cap 48)  peak=${PEAK}GB"

  if [[ "$STATUS" == FAIL_* && "$ANY_OK" != "1" ]]; then
    echo "    !! first config failed with a non-OOM error — aborting sweep."
    grep -A15 -m1 "Traceback" "$LOG" || tail -15 "$LOG"
    break
  fi
  [[ "$STATUS" == "OK" ]] && ANY_OK=1

  rm -rf "$OUTDIR/run_g${G}"
done

echo
echo "=== summary ==="
column -t "$SUMMARY"
echo
echo "degenerate_rate = fraction of optimizer steps producing exactly zero gradient."
echo "mean_tok_per_seq vs --max_completion_length 48 = decode spent on padding."
