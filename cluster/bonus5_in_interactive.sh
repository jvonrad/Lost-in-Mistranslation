#!/bin/bash
# Wait for the standing interactive placeholder job (4 GPUs, 24h) to start,
# then launch the GRPO all_correct_bonus=5.0 ablation inside it via a single
# srun step with its OWN in-step retry loop. The retry loop runs remotely
# under the job's cgroup (immune to login-node session kill), so a training
# crash (e.g. a Lustre mount race, or anything else) just relaunches
# --resume_from_checkpoint auto in place -- no requeue, no risk of landing on
# a fresh node with its own mount race, no lost queue position.
#
# Usage: setsid nohup bash cluster/bonus5_in_interactive.sh <JOBID> > logs/bonus5_watcher.log 2>&1 &
set -uo pipefail
JOBID="${1:?usage: bonus5_in_interactive.sh <jobid>}"
# REPO is the actual git checkout (code) -- lives under /home, NOT /projects.
# /projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation is a separate, pre-existing
# *data/output-only* directory (datasets/, models/, logs/) that has never
# contained training/, evaluate/, cluster/, etc. Every earlier "Lustre mount
# race" failure this session (nid011229, nid010545, nid010530, nid010320) was
# actually this same wrong-path bug, not a timing issue -- the file was never
# going to appear there no matter how long the retry loop waited.
REPO=/home/u6jh/jvonrad.u6jh/Lost-in-Mistranslation
PROJ=/projects/u6jh/jvonrad.u6jh/Lost-in-Mistranslation

echo "[bonus5-watcher] waiting for job $JOBID to start ..."
while true; do
    state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
    [[ "$state" == "RUNNING" ]] && break
    sleep 15
done
echo "[bonus5-watcher] job $JOBID is RUNNING @ $(date -Is), launching training (in-step retry loop, up to 10 attempts)"

srun --jobid="$JOBID" --overlap --nodes=1 --ntasks=1 --gres=gpu:2 bash -c "
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate grpo
  cd $REPO
  export TOKENIZERS_PARALLELISM=false

  for attempt in \$(seq 1 10); do
    echo \"[bonus5] attempt \$attempt/10 @ \$(date -Is) on \$(hostname)\"
    accelerate launch --num_processes 2 --multi_gpu \
      training/train_wikifact_grpo_accelerate.py \
      --model_id Qwen/Qwen2.5-7B \
      --dataset_id jvonrad/WIKI-FACT \
      --output_dir '$PROJ/models/qwen-grpo-bonus50-ablation' \
      --run_name qwen-grpo-bonus50-ablation \
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
    rc=\$?
    echo \"[bonus5] attempt \$attempt exited with code \$rc @ \$(date -Is)\"
    [[ \$rc -eq 0 ]] && break
    sleep 10
  done
" > "$REPO/logs/grpo_bonus5_ablation_interactive.log" 2>&1
echo "[bonus5-watcher] srun step returned (exit $?) @ $(date -Is)"
