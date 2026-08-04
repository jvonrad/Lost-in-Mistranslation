#!/bin/bash
# GRPO per-device batch-size pilot: find the largest --per_device_train_batch_size
# that fits on a GH200 and measure seconds per optimizer step at each size.
#
# Runs on TWO GPUs (whichever CUDA_VISIBLE_DEVICES selects) so the DDP path,
# gradient sync and per-rank memory are all exercised the way a real run would
# be — a 1-GPU probe would over-report available memory.
#
# Each size is a separate short `accelerate launch`, because batch size is fixed
# at launch. Training state is thrown away: --eval_steps 0 disables BOTH the
# periodic benchmark eval and checkpoint writing, --report_to none keeps wandb
# out, --resume_from_checkpoint none guarantees a cold start every time.
#
# Model choice matters: peak memory is dominated by the [seqs, seqlen, vocab]
# logits/log_softmax tensor in compute_logprob_loss, so the ceiling is
# vocab-dependent. Default is Qwen (152k vocab) rather than OLMo (100k) because
# it is the tighter constraint — a batch size that fits Qwen also fits OLMo.
#
# Usage (inside an allocation):
#   CUDA_VISIBLE_DEVICES=0,1 bash cluster/bs_pilot.sh [outdir] [steps] [sizes...]
set -uo pipefail

source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

OUTDIR="${1:-$PROJ/logs/bs_pilot}"
STEPS="${2:-4}"          # optimizer steps per size; first 2 are discarded as warmup
shift 2 2>/dev/null || true
SIZES=("${@:-}")
[[ -z "${SIZES[0]:-}" ]] && SIZES=(1 2 3 4 6 8)

MODEL="${MODEL_ID:-Qwen/Qwen2.5-7B}"
NPROC=2
PORT="${MAIN_PROCESS_PORT:-29500}"
DEADLINE="${PILOT_DEADLINE:-0}"   # epoch seconds; 0 = no guard
PER_CFG_TIMEOUT="${PER_CFG_TIMEOUT:-600}"

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf "bs\tfacts_per_step\tstatus\tstep_s\ts_per_fact\tpeak_mem_gb\n" > "$SUMMARY"

echo "=== bs pilot: model=$MODEL gpus=${CUDA_VISIBLE_DEVICES:-all} steps=$STEPS sizes=${SIZES[*]} ==="

ANY_OK=0

for BS in "${SIZES[@]}"; do
  # Never start a config we don't have time to finish — and say so out loud
  # rather than silently truncating the sweep.
  if [[ "$DEADLINE" != "0" ]] && (( $(date +%s) > DEADLINE )); then
    echo "!! deadline reached — SKIPPING remaining batch sizes: $BS onward"
    printf "%s\t%s\tSKIPPED_DEADLINE\tnan\tnan\tnan\n" "$BS" "$(( BS * NPROC ))" >> "$SUMMARY"
    continue
  fi

  LOG="$OUTDIR/bs${BS}.log"
  # Enough facts for exactly $STEPS optimizer steps across both ranks.
  SAMPLES=$(( BS * NPROC * STEPS ))
  echo "--- per_device_train_batch_size=$BS (max_train_samples=$SAMPLES) -> $LOG"

  timeout "$PER_CFG_TIMEOUT" accelerate launch --num_processes "$NPROC" --multi_gpu \
    --main_process_port "$PORT" \
    training/train_wikifact_grpo_accelerate.py \
    --model_id "$MODEL" \
    --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
    --output_dir "$OUTDIR/run_bs${BS}" \
    --run_name "bs-pilot-${BS}" \
    --use_lora --bf16 \
    --kl_coef 0.0 \
    --num_generations 8 \
    --learning_rate 1e-5 \
    --max_completion_length 48 \
    --temperature 0.7 --top_p 0.95 \
    --per_device_train_batch_size "$BS" \
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

  # Pull the per-step metrics the trainer now prints, discarding the first two
  # steps (model warmup, allocator settling, first-touch of the static KV cache).
  read -r STEP_S PER_FACT PEAK < <(
    python - "$LOG" <<'PY'
import ast, sys
rows = []
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if line.startswith("{") and "'step'" in line and "step_s" in line:
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            pass
rows = rows[2:]  # drop warmup steps
if not rows:
    print("nan nan nan")
else:
    n = len(rows)
    print("%.3f %.3f %.2f" % (
        sum(r["step_s"] for r in rows) / n,
        sum(r["s_per_fact"] for r in rows) / n,
        max(r["peak_mem_gb"] for r in rows),
    ))
PY
  )

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$BS" "$(( BS * NPROC ))" "$STATUS" "$STEP_S" "$PER_FACT" "$PEAK" >> "$SUMMARY"
  echo "    -> $STATUS  step_s=$STEP_S  s/fact=$PER_FACT  peak_mem=${PEAK}GB"

  # Don't keep probing larger sizes once we've hit the memory wall.
  if [[ "$STATUS" == "OOM" ]]; then
    echo "    OOM at bs=$BS — stopping sweep (larger sizes cannot fit)."
    break
  fi

  # A crash before ANY size has succeeded means the trainer is broken, not that
  # we found the memory ceiling — abort rather than burn the allocation
  # repeating the same failure five more times.
  if [[ "$STATUS" == FAIL_* && "$ANY_OK" != "1" ]]; then
    echo "    !! first config failed with a non-OOM error — aborting sweep."
    grep -A15 -m1 "Traceback" "$LOG" || tail -15 "$LOG"
    break
  fi
  [[ "$STATUS" == "OK" ]] && ANY_OK=1

  rm -rf "$OUTDIR/run_bs${BS}"
done

echo
echo "=== summary ==="
column -t "$SUMMARY"
