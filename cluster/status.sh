#!/bin/bash
# One-shot status of the CM-Align pipeline + GRPO ablation running in the
# interactive allocation. Run from the repo root: bash cluster/status.sh
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
JOBID="$(cut -d' ' -f1 cluster/state/current_job 2>/dev/null)"

line() { printf '%s\n' "------------------------------------------------------------"; }

echo "allocation: job $JOBID  ($(squeue -j "$JOBID" -h -o '%T on %N %M elapsed' 2>/dev/null))"
line
echo "CM-Align CONSTRUCT (preprocessing) — per GPU shard:"
for s in 0 1 2 3; do
  f="logs/cmalign_qwen_construct_shard${s}.log"
  [ -f "$f" ] || continue
  printf "  shard %s: " "$s"
  if grep -qa "\[construct\] done" "$f"; then
    echo "DONE — $(grep -a '\[construct\] done' "$f" | tail -1 | sed 's/\[construct\] //')"
  else
    tr '\r' '\n' < "$f" | grep -aE "construct\] fact" | tail -1 | sed 's/\[construct\] //'
  fi
done
line
echo "CM-Align orchestrator stage (construct -> merge -> DPO train):"
tail -3 logs/cmalign_qwen_orchestrator.log 2>/dev/null | sed 's/^/  /'
if [ -f logs/cmalign_qwen_train.log ]; then
  echo "  DPO train:"; tr '\r' '\n' < logs/cmalign_qwen_train.log | grep -aE "'step'|saving|saved merged" | tail -2 | sed 's/^/    /'
fi
line
echo "GRPO no-bonus ablation (GPU 2):"
tr '\r' '\n' < logs/grpo_nobonus_ablation.log 2>/dev/null | grep -aE "'step'|\[eval @|saving" | tail -2 | sed 's/^/  /'
line
echo "GPUs:"
srun --jobid="$JOBID" --overlap nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
echo
echo "wandb: https://wandb.ai/jonathan-von-rad/UnLock  (runs: qwen-grpo-nobonus-ablation, qwen-cmalign-dpo)"
