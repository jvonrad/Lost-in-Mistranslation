#!/bin/bash
# Full-node GRPO efficiency pilot — ~30 minutes of measurement, nothing idle.
#
#   GPU 0,1 -> per-device batch-size sweep   (cluster/bs_pilot.sh)
#              largest batch that fits + seconds per optimizer step
#   GPU 2,3 -> rollout-efficiency probe      (cluster/gen_pilot.sh)
#              degenerate-group rate, marginal cost of --num_generations,
#              and how much decode is spent on padding vs the 48-token cap
#   CPU     -> finish the HF model cache (no GPU needed)
#
# The two sweeps are complementary, not redundant: the first prices a bigger
# batch, the second measures whether the gradient signal is there to justify
# one. Both pin Qwen (152k vocab) because peak memory is dominated by the
# [seqs, seqlen, vocab] logits tensor — Qwen is the tighter constraint, so a
# batch size that fits it also fits OLMo (100k).
#
# Submit: sbatch cluster/pilot_node.sbatch
#SBATCH --job-name=pilot-node
#SBATCH --account=brics.u6sg
#SBATCH --partition=workq
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --output=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/pilot-node-%j.out
#SBATCH --error=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/pilot-node-%j.err

set -uo pipefail
source /home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/cluster/env.sh
cd "$REPO"

RUNDIR="$PROJ/logs/pilot-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUNDIR"
START=$(date +%s)
# Stop LAUNCHING new sweep configs 30 min in; the remaining 15 min of wall time
# is slack for the last in-flight config plus summaries.
export PILOT_DEADLINE=$(( START + 30*60 ))

echo "pilot-node: job ${SLURM_JOB_ID:-none} on $(hostname) @ $(date -Is)"
echo "sweep deadline: $(date -d @$PILOT_DEADLINE -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Phase 0 (CPU, blocking): Qwen must be cached before either sweep starts. ──
# The login node cannot do this — it enforces a 4 GiB per-user memory cgroup and
# SIGKILLs the download partway through (twice, at 6.6 GB and 291 MB). Compute
# nodes have 460 GB and outbound internet, so it succeeds here in a few minutes.
echo "[phase0] fetching Qwen base + dataset @ $(date -Is)"
python -u - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B",
                  ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.h5"], max_workers=16)
snapshot_download("jvonrad/PolyFact-Clean", repo_type="dataset", max_workers=8)
print("phase0 model+dataset cached", flush=True)
PY
echo "[phase0] done @ $(date -Is) (+$(( $(date +%s) - START ))s)"

# ── Phase 0b (CPU): warm the datasets .map cache ONCE. ───────────────────────
# build_grouped_fact_item is mapped over the full train split with num_proc=32
# at every launch; without this, each of the ~9 short launches below would pay
# that cost again instead of hitting the on-disk cache.
echo "[phase0b] warming dataset map cache @ $(date -Is)"
python -u - <<'PY'
import sys
sys.path.insert(0, "training")
from datasets import load_dataset
from train_wikifact_grpo_accelerate import build_grouped_fact_item
ds = load_dataset("jvonrad/PolyFact-Clean", "parallel", split="train")
ds = ds.map(build_grouped_fact_item, num_proc=32)
print("phase0b mapped rows:", len(ds), flush=True)
PY
echo "[phase0b] done @ $(date -Is) (+$(( $(date +%s) - START ))s)"

# ── Phase 1: both sweeps in parallel on disjoint GPU pairs ───────────────────
# Distinct --main_process_port per sweep, or the two accelerate rendezvous
# collide on the default 29500 and one hangs.
CUDA_VISIBLE_DEVICES=0,1 MAIN_PROCESS_PORT=29500 \
  bash cluster/bs_pilot.sh "$RUNDIR/bs_pilot" 4 1 2 3 4 6 8 \
  > "$RUNDIR/bs_pilot.log" 2>&1 &
PID_BS=$!

CUDA_VISIBLE_DEVICES=2,3 MAIN_PROCESS_PORT=29600 \
  bash cluster/gen_pilot.sh "$RUNDIR/gen_pilot" 5 4 8 16 \
  > "$RUNDIR/gen_pilot.log" 2>&1 &
PID_GEN=$!

# ── CPU only: finish the model cache for the later eval sweep ────────────────
python -u - > "$RUNDIR/prefetch_rest.log" 2>&1 <<'PY' &
from huggingface_hub import snapshot_download
REST = [
    "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint",
    "jvonrad/Qwen-2.5-7B-TED",
    "jvonrad/Qwen-2.5-7B-SFT-CE-random",
    "jvonrad/Qwen-2.5-7B-TED-SFT",
    "jvonrad/Qwen-2.5-7B-grpo-consistent",
    "jvonrad/Qwen-2.5-7B-TED-grpo",
    "sentence-transformers/LaBSE",
]
for m in REST:
    try:
        snapshot_download(m, ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.h5"],
                          max_workers=4)
        print("OK  ", m, flush=True)
    except Exception as e:
        print("FAIL", m, type(e).__name__, e, flush=True)
print("prefetch-rest done", flush=True)
PY
PID_FETCH=$!

wait $PID_BS;    RC_BS=$?
wait $PID_GEN;   RC_GEN=$?
wait $PID_FETCH; RC_FETCH=$?

echo
echo "=================== pilot-node results ==================="
echo "--- batch-size sweep (rc=$RC_BS) ---"
[[ -f "$RUNDIR/bs_pilot/summary.tsv" ]] && column -t "$RUNDIR/bs_pilot/summary.tsv"
echo
echo "--- rollout-efficiency probe (rc=$RC_GEN) ---"
[[ -f "$RUNDIR/gen_pilot/summary.tsv" ]] && column -t "$RUNDIR/gen_pilot/summary.tsv"
echo
echo "--- prefetch (rc=$RC_FETCH) ---"
tail -3 "$RUNDIR/prefetch_rest.log" 2>/dev/null
echo
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min; artifacts in $RUNDIR"
