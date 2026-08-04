#!/bin/bash
# Launch the OLMo-2-7B GRPO all_correct_bonus=5.0 ablation on ALL 4 GPUs of a
# standing interactive placeholder allocation (2 nodes x 2 GPUs = multi-node
# DDP). The counterpart of cluster/bonus5_in_interactive.sh (Qwen, 2 GPUs).
#
# 4-GPU multi-node: one srun task per node (--nodes=2 --ntasks-per-node=1
# --gres=gpu:2), each running `accelerate launch` with num_machines=2,
# num_processes=4, machine_rank=$SLURM_PROCID, rendezvousing on the head node.
# Verified rendezvous (DIST_OK world=4) before wiring this up. The whole srun
# is wrapped in an in-step retry loop, so a crash relaunches the entire 2-node
# job with --resume_from_checkpoint auto (both nodes restart together, no
# rendezvous desync).
#
# Usage: setsid nohup bash cluster/olmo_bonus5_interactive.sh <JOBID> [HEAD_IP] \
#          > /projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation/logs/olmo_bonus5_watcher.log 2>&1 &
set -uo pipefail
JOBID="${1:?usage: olmo_bonus5_interactive.sh <jobid> [head_ip]}"
REPO=/home/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
PROJ=/projects/u6sg/jvonrad.u6sg/Lost-in-Mistranslation
PORT=29501
# Head IP = first node's high-speed NIC (rank-0 / machine_rank 0 lands there
# because it is first in the allocation's nodelist). Pass explicitly to override.
HEAD_IP="${2:-$(scontrol show hostnames "$(squeue -j "$JOBID" -h -o %N)" 2>/dev/null | head -1)}"
# If we got a hostname (not an IP), resolve it below via srun; a bare nid name
# usually resolves fine for NCCL, but the numeric IP is safest.

echo "[olmo-bonus5] waiting for job $JOBID to be RUNNING ..."
while true; do
    st=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
    [[ "$st" == "RUNNING" ]] && break
    sleep 15
done
echo "[olmo-bonus5] job $JOBID RUNNING @ $(date -Is); head=$HEAD_IP; launching 4-GPU multi-node training"

for attempt in $(seq 1 20); do
    echo "[olmo-bonus5] attempt $attempt/20 @ $(date -Is)"
    srun --jobid="$JOBID" --overlap --nodes=2 --ntasks-per-node=1 --gres=gpu:2 bash -c "
        source ~/miniforge3/etc/profile.d/conda.sh
        conda activate grpo
        cd $REPO
        export TOKENIZERS_PARALLELISM=false
        echo \"[rank-launch] machine_rank=\$SLURM_PROCID host=\$(hostname) @ \$(date -Is)\"
        accelerate launch --num_machines 2 --num_processes 4 \
          --machine_rank \$SLURM_PROCID \
          --main_process_ip $HEAD_IP --main_process_port $PORT --multi_gpu \
          training/train_wikifact_grpo_accelerate.py \
          --model_id allenai/OLMo-2-1124-7B \
          --dataset_id jvonrad/PolyFact-Clean --dataset_config parallel \
          --output_dir '$PROJ/models/olmo-grpo-bonus50-ablation' \
          --run_name olmo-grpo-bonus50-ablation \
          --use_lora --bf16 \
          --kl_coef 0.0 \
          --all_correct_bonus 5.0 \
          --num_generations 8 \
          --learning_rate 1e-5 \
          --max_completion_length 48 \
          --temperature 0.7 \
          --top_p 0.95 \
          --max_train_samples 40000 \
          --num_train_epochs 1 \
          --eval_steps 200 \
          --max_eval_wikifact 100 \
          --gen_micro_batch_size 192 \
          --logprob_micro_batch_size 48 \
          --resume_from_checkpoint auto \
          --report_to wandb
    " > "$PROJ/logs/olmo_grpo_bonus5_interactive.log" 2>&1
    rc=$?
    echo "[olmo-bonus5] attempt $attempt exited rc=$rc @ $(date -Is)"
    [[ $rc -eq 0 ]] && break
    sleep 10
done
echo "[olmo-bonus5] watcher done @ $(date -Is)"
