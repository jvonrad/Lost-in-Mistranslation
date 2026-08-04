#!/bin/bash
# Finish the bespoke OLMo CM-Align run assembled from five non-uniform pieces:
#   shard0  [0,10000), shard0b [10000,15000), shard0c [15000,20000),
#   shard2 [20000,30000), shard3 [30000,40000).
#
# This script is intended to run inside a CPU-only --overlap step of the same
# interactive allocation as shard0b/0c. It waits for all checkpoints, validates
# and atomically merges them, then starts DPO in one combined two-GPU step.
# Keeping the wait inside a Slurm step makes it immune to login-session cleanup.
#
# Usage:
#   srun --jobid=<jobid> --overlap --nodes=1 --ntasks=1 \
#     bash cluster/finish_olmo_cmalign_5way.sh <jobid> <node>

set -euo pipefail

JOBID="${1:?usage: $0 <jobid> <node>}"
NODE="${2:?usage: $0 <jobid> <node>}"
REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
MERGED="$PROJ/datasets/cmalign_pref_olmo"
OUT="$PROJ/models/olmo-cmalign-dpo"

SHARD_DIRS=(
  "$PROJ/datasets/cmalign_pref_olmo_shard0"
  "$PROJ/datasets/cmalign_pref_olmo_shard0b"
  "$PROJ/datasets/cmalign_pref_olmo_shard0c"
  "$PROJ/datasets/cmalign_pref_olmo_shard2"
  "$PROJ/datasets/cmalign_pref_olmo_shard3"
)
EXPECTED_FACTS=(10000 5000 5000 10000 10000)

log() { echo "[$(date -Is)] [olmo-5way] $*"; }

source ~/miniforge3/etc/profile.d/conda.sh
conda activate grpo

checkpoint_position() {
  python - "$1/_checkpoint_meta.json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        print(int(json.load(f).get("next_batch_start", 0)))
except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
    print(0)
PY
}

log "job=$JOBID node=$NODE; waiting for five construct pieces"
while true; do
  all_ready=1
  status=()
  for i in "${!SHARD_DIRS[@]}"; do
    pos="$(checkpoint_position "${SHARD_DIRS[$i]}")"
    status+=("$pos/${EXPECTED_FACTS[$i]}")
    if (( pos < EXPECTED_FACTS[i] )); then
      all_ready=0
    fi
  done
  log "construct positions: ${status[*]}"
  (( all_ready == 1 )) && break
  sleep 60
done

cd "$REPO"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

log "validating and merging five datasets -> $MERGED"
python - "$MERGED" "${SHARD_DIRS[@]}" <<'PY'
import json
import os
import shutil
import sys
from datasets import concatenate_datasets, load_from_disk

merged_path, *dirs = sys.argv[1:]
expected_parts = [10000, 5000, 5000, 10000, 10000]
parts = []
fact_sets = []
summaries = []

for path, expected in zip(dirs, expected_parts):
    with open(os.path.join(path, "_checkpoint_meta.json")) as f:
        meta = json.load(f)
    position = int(meta.get("next_batch_start", 0))
    if position < expected:
        raise RuntimeError(f"incomplete shard {path}: {position}/{expected}")
    processed = int(meta.get("n_facts_used", 0))
    if processed != expected:
        raise RuntimeError(f"shard {path} processed {processed}/{expected} usable facts")
    ds = load_from_disk(path)
    required = {"fact_id", "lang", "prompt", "chosen", "rejected"}
    missing = required.difference(ds.column_names)
    if missing:
        raise RuntimeError(f"shard {path} missing columns: {sorted(missing)}")
    facts = set(ds["fact_id"])
    parts.append(ds)
    fact_sets.append(facts)
    summaries.append({
        "path": path, "rows": len(ds), "processed_facts": processed,
        "facts_with_pairs": len(facts),
    })

# The source dataset itself can contain the same fact_id at two different
# positional indices, so adjacent positional shards are not guaranteed to
# have disjoint fact IDs. Keep the first shard's preference rows for each fact
# and drop all rows for that fact from later shards. This is deterministic and
# prevents duplicated facts from receiving extra DPO weight.
seen_facts = set()
deduped_parts = []
duplicate_facts = []
for i, (ds, facts) in enumerate(zip(parts, fact_sets)):
    duplicates = facts.intersection(seen_facts)
    summaries[i]["rows_before_dedup"] = len(ds)
    summaries[i]["duplicate_facts_dropped"] = sorted(duplicates)
    if duplicates:
        ds = ds.filter(lambda row, drop=duplicates: row["fact_id"] not in drop)
        duplicate_facts.extend(sorted(duplicates))
    summaries[i]["rows"] = len(ds)
    deduped_parts.append(ds)
    seen_facts.update(facts)
parts = deduped_parts

manifest = {
    "parts": summaries,
    "rows": sum(len(p) for p in parts),
    "processed_facts": sum(p["processed_facts"] for p in summaries),
    "unique_facts_with_pairs": len(seen_facts),
    "duplicate_fact_count": len(duplicate_facts),
    "duplicate_facts_dropped": duplicate_facts,
}
if manifest["processed_facts"] != 40000:
    raise RuntimeError(f"expected 40000 processed facts, got {manifest['processed_facts']}")

manifest_path = os.path.join(merged_path, "_merge_manifest.json")
if os.path.isdir(merged_path) and os.path.isfile(manifest_path):
    with open(manifest_path) as f:
        existing = json.load(f)
    if existing == manifest:
        loaded = load_from_disk(merged_path)
        if len(loaded) == manifest["rows"]:
            print(f"[merge] existing validated dataset: {len(loaded)} rows")
            raise SystemExit(0)

tmp = merged_path.rstrip("/") + "_tmp_5way"
if os.path.exists(tmp):
    shutil.rmtree(tmp)
merged = concatenate_datasets(parts)
merged.save_to_disk(tmp)
with open(os.path.join(tmp, "_merge_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
if os.path.exists(merged_path):
    backup = merged_path.rstrip("/") + "_pre5way_backup"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    os.rename(merged_path, backup)
os.rename(tmp, merged_path)
print(
    f"[merge] wrote {len(merged)} rows from {manifest['processed_facts']} processed facts "
    f"({manifest['unique_facts_with_pairs']} unique facts produced pairs; "
    f"dropped {manifest['duplicate_fact_count']} duplicate facts)"
)
PY

log "launching two-process DPO train on physical GPUs 2,3 -> $OUT"
# Request the allocation's full GPU set and pin this process group to devices
# 2,3. Separate --overlap steps requesting gpu:2 can both be mapped to physical
# devices 0,1 on this cluster, which would collide with the bonus+5 step.
srun --jobid="$JOBID" --overlap -w "$NODE" --nodes=1 --ntasks=1 --gres=gpu:4 bash -c "
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate grpo
  cd '$REPO'
  export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3
  accelerate launch --num_processes 2 --multi_gpu \
    training/train_wikifact_cmalign_dpo.py \
    --phase train --model_id allenai/OLMo-2-1124-7B \
    --pref_data_path '$MERGED' --output_dir '$OUT' \
    --run_name olmo-cmalign-dpo \
    --beta 0.1 --nll_gamma 0.0 --learning_rate 5e-6 --num_train_epochs 1 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --resume_from_checkpoint auto \
    --bf16 --report_to wandb
" > "$REPO/logs/cmalign_olmo_train.log" 2>&1

log "DPO train finished; merged model is at $OUT/merged"
