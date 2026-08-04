#!/bin/bash
# Four single-GPU GRPO throughput variants, one per GPU of an allocated node.
#
# The bs sweep showed bigger batches do NOT buy throughput (2.84 -> 2.67 s/fact
# from bs=1 to bs=2): generation and the loss pass are both compute-saturated
# and linear in sequence count, so only fixed per-step overhead amortizes. The
# real waste is PADDING: tokenizer fertility spans bn ~307 tok vs en ~73 for
# the same fact, so unsorted micro-batches pad ~everything to the Bengali max
# (~445 tok vs ~164 mean). These variants attack that + the duplicated
# full-vocab log_softmax tensor.
#
#   A (GPU0) control        — current code, no new flags
#   B (GPU1) bucketing      — --length_bucketing --gen_micro_batch_size 32
#   C (GPU2) fused          — --fused_logprob + expandable_segments allocator
#   D (GPU3) all + mbs96    — B + C + --logprob_micro_batch_size 96
#
# All: 1 GPU, bs=1, G=8, native scaffold, STEPS optimizer steps (first 2
# discarded by the extractor as warmup).
#
# Usage (inside an allocation): bash cluster/throughput_pilot.sh [outdir] [steps]
set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

OUTDIR="${1:-$PROJ/logs/throughput_pilot}"
STEPS="${2:-7}"
MODEL="${MODEL_ID:-Qwen/Qwen2.5-7B}"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
printf "variant\tstatus\tstep_s\tpeak_mem_gb\n" > "$SUMMARY"

run_variant () {  # name gpu port alloc_conf extra_flags...
  local NAME="$1" GPU="$2" PORT="$3" ALLOC="$4"; shift 4
  local LOG="$OUTDIR/${NAME}.log"
  echo "--- variant $NAME on GPU $GPU (extra: $*)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF="$ALLOC" \
  timeout 900 accelerate launch --num_processes 1 --main_process_port "$PORT" \
    training/train_wikifact_grpo_accelerate.py \
    --model_id "$MODEL" \
    --output_dir "$OUTDIR/run_${NAME}" \
    --run_name "tp-${NAME}" \
    --use_lora --bf16 --kl_coef 0.0 \
    --num_generations 8 --learning_rate 1e-5 \
    --max_completion_length 48 --temperature 0.7 --top_p 0.95 \
    --per_device_train_batch_size 1 \
    --max_train_samples "$STEPS" --num_train_epochs 1 \
    --gen_micro_batch_size 192 --logprob_micro_batch_size 48 \
    --logging_steps 1 --eval_steps 0 \
    --max_eval_mmlu 1 --max_eval_flores 1 --max_eval_wikifact 1 \
    --resume_from_checkpoint none --report_to none \
    "$@" > "$LOG" 2>&1
  local RC=$?
  local STATUS=OK
  grep -qiE "out of memory|OutOfMemoryError" "$LOG" && STATUS=OOM
  [[ $RC -eq 124 ]] && STATUS=TIMEOUT
  [[ $RC -ne 0 && "$STATUS" == "OK" ]] && STATUS="FAIL_rc${RC}"
  read -r STEP_S PEAK < <(python - "$LOG" <<'PY'
import ast, sys
rows=[]
for line in open(sys.argv[1], errors="ignore"):
    line=line.strip()
    if line.startswith("{") and "'step'" in line and "step_s" in line:
        try: rows.append(ast.literal_eval(line))
        except Exception: pass
rows=rows[2:]
if not rows: print("nan nan")
else:
    n=len(rows)
    print("%.3f %.2f" % (sum(r["step_s"] for r in rows)/n, max(r["peak_mem_gb"] for r in rows)))
PY
  )
  printf "%s\t%s\t%s\t%s\n" "$NAME" "$STATUS" "$STEP_S" "$PEAK" >> "$SUMMARY"
  echo "    -> $NAME: $STATUS  step_s=$STEP_S  peak=${PEAK}GB"
  rm -rf "$OUTDIR/run_${NAME}"
}

# NOTE: gen_micro_batch_size 32 for the bucketing variants — at bs=1 there are
# only 96 prompts, so the default 192 puts them all in ONE generate() chunk and
# sorting cannot reduce its padding; 32 gives 3 chunks, each padded to its own
# language-cluster max.
run_variant A_control  0 29700 ""                         &
run_variant B_bucket   1 29710 ""                         --length_bucketing --gen_micro_batch_size 32 &
run_variant C_fused    2 29720 "expandable_segments:True" --fused_logprob &
run_variant D_all      3 29730 "expandable_segments:True" --length_bucketing --fused_logprob \
                                                          --gen_micro_batch_size 32 --logprob_micro_batch_size 96 &
wait

echo; echo "=== summary ==="; column -t "$SUMMARY"
