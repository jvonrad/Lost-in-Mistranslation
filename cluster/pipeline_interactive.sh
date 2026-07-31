#!/bin/bash
# Keeps a rolling 24h interactive allocation alive on Isambard-AI by submitting
# the next job ahead of the current one's expiry, timed off the observed queue
# wait, so the new allocation lands close to when the old one runs out.
#
# Does NOT cancel jobs early or opportunistically grab idle nodes — each job
# runs its full 24h and is left to expire naturally; only submit timing is
# pipelined. Run this on a login node, e.g.:
#   setsid nohup cluster/pipeline_interactive.sh > cluster/pipeline.out 2>&1 < /dev/null &
#
# Current job id / node is kept in cluster/state/current_job for you to attach to:
#   srun --jobid=$(cat cluster/state/current_job) --pty bash
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$REPO_DIR/cluster/interactive_job.sbatch"
STATE_DIR="$REPO_DIR/cluster/state"
WAIT_HISTORY="$STATE_DIR/wait_history.log"
CURRENT_JOB_FILE="$STATE_DIR/current_job"
LOG="$REPO_DIR/cluster/pipeline.log"

WALLTIME_SECONDS=86400          # must match --time in interactive_job.sbatch
SAFETY_MARGIN_SECONDS=900       # submit this much earlier than the raw wait estimate
INITIAL_WAIT_ESTIMATE=3600      # guess used before we have any observed data
POLL_INTERVAL=30

mkdir -p "$STATE_DIR"
touch "$WAIT_HISTORY"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

submit_job() {
    sbatch --parsable "$SBATCH_SCRIPT"
}

# Polls until the job leaves PENDING (RUNNING, or gone/failed). Echoes the
# epoch time it started running, or "FAILED" if it never reached RUNNING.
wait_for_running() {
    local jobid="$1"
    while true; do
        local state
        state=$(squeue -j "$jobid" -h -o '%T' 2>/dev/null)
        if [[ "$state" == "RUNNING" ]]; then
            date +%s
            return 0
        elif [[ -z "$state" ]]; then
            # left the queue without ever running
            echo "FAILED"
            return 1
        fi
        sleep "$POLL_INTERVAL"
    done
}

record_wait() {
    echo "$1" >> "$WAIT_HISTORY"
}

# Conservative estimate: max of the last 5 observed waits, plus safety margin.
estimate_wait_seconds() {
    local n
    n=$(wc -l < "$WAIT_HISTORY")
    if [[ "$n" -eq 0 ]]; then
        echo $((INITIAL_WAIT_ESTIMATE + SAFETY_MARGIN_SECONDS))
        return
    fi
    local max_recent
    max_recent=$(tail -n 5 "$WAIT_HISTORY" | sort -n | tail -n 1)
    echo $((max_recent + SAFETY_MARGIN_SECONDS))
}

sleep_until() {
    local target_epoch="$1"
    local now
    now=$(date +%s)
    local delta=$((target_epoch - now))
    if [[ "$delta" -gt 0 ]]; then
        sleep "$delta"
    fi
}

log "pipeline starting"

submit_epoch=$(date +%s)
jobid=$(submit_job)
log "submitted job $jobid at $(date -Is)"

start_epoch=$(wait_for_running "$jobid")
if [[ "$start_epoch" == "FAILED" ]]; then
    log "job $jobid never started running; aborting"
    exit 1
fi
wait_seconds=$((start_epoch - submit_epoch))
record_wait "$wait_seconds"
node=$(squeue -j "$jobid" -h -o '%N')
echo "$jobid $node" > "$CURRENT_JOB_FILE"
log "job $jobid running on $node (queue wait ${wait_seconds}s)"

while true; do
    estimate=$(estimate_wait_seconds)
    trigger_epoch=$((start_epoch + WALLTIME_SECONDS - estimate))
    log "next submit at $(date -Is -d @"$trigger_epoch") (estimated wait ${estimate}s incl. margin)"
    sleep_until "$trigger_epoch"

    submit_epoch=$(date +%s)
    jobid=$(submit_job)
    log "submitted job $jobid at $(date -Is)"

    new_start_epoch=$(wait_for_running "$jobid")
    if [[ "$new_start_epoch" == "FAILED" ]]; then
        log "job $jobid never started running; will retry next loop without advancing state"
        continue
    fi
    wait_seconds=$((new_start_epoch - submit_epoch))
    record_wait "$wait_seconds"
    node=$(squeue -j "$jobid" -h -o '%N')
    echo "$jobid $node" > "$CURRENT_JOB_FILE"
    log "job $jobid running on $node (queue wait ${wait_seconds}s) — new current allocation"

    start_epoch="$new_start_epoch"
done
