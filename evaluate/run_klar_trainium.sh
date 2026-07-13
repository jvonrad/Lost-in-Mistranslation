#!/usr/bin/env bash
# KLAR eval (6 langs, greedy 10 tok, clean-vs-contaminated split) for all 12
# models, one logical NeuronCore pair each, setsid-detached.
set -uo pipefail
R=/home/ubuntu/Lost-in-Mistranslation/results/klar
mkdir -p "$R"
cd /home/ubuntu/Lost-in-Mistranslation

L () { # cores model basename
  local cores="$1" model="$2" base="$3"
  NEURON_RT_VISIBLE_CORES="$cores" NEURON_CC_FLAGS="--cache_dir=/mnt/nvme/neuron-cache" \
  setsid bash -c "source ~/neuron_venv/bin/activate && python -u evaluate/evaluate_klar.py \
    --model '$model' --tokenizer '$model' --device xla --klar-root datasets/KLAR-CLC \
    --batch-size 16 \
    --contamination-labels evaluate/alignments/klar_polyfact_contamination.json \
    --output-json '$R/${base}_klar.json'" > "$R/${base}_klar.log" 2>&1 < /dev/null &
}

L "0-1"   "allenai/OLMo-2-1124-7B"                 "OLMo-2-1124-7B"
L "2-3"   "jvonrad/OLMo-2-1124-7B-TED"             "OLMo-2-1124-7B-TED"
L "4-5"   "jvonrad/olmo-2-7b-wikifact-sft"         "olmo-2-7b-wikifact-sft"
L "6-7"   "jvonrad/olmo-2-7b-aligned-wikifact-sft" "olmo-2-7b-aligned-wikifact-sft"
L "8-9"   "jvonrad/olmo-2-7b-grpo-att-mlp-full"    "olmo-2-7b-grpo-att-mlp-full"
L "10-11" "jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint" "olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
L "12-13" "Qwen/Qwen2.5-7B"                        "Qwen-2.5-7B"
L "14-15" "jvonrad/Qwen-2.5-7B-TED"                "Qwen-2.5-7B-TED"
L "16-17" "jvonrad/Qwen-2.5-7B-SFT-CE-random"      "Qwen-2.5-7B-SFT-CE-random"
L "18-19" "jvonrad/Qwen-2.5-7B-TED-SFT"            "Qwen-2.5-7B-TED-SFT"
L "20-21" "jvonrad/Qwen-2.5-7B-grpo-consistent"    "Qwen-2.5-7B-grpo-consistent"
L "22-23" "jvonrad/Qwen-2.5-7B-TED-grpo"           "Qwen-2.5-7B-TED-grpo"
echo "LAUNCHED_KLAR_12"
