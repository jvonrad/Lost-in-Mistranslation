#!/usr/bin/env bash
set -e

# ---------- Models ----------
LLAMA_2_7B="meta-llama/Llama-2-7b-hf"
OLMO_7B="allenai/OLMo-7B-hf"
# ----------------------------

# --------- Fine-tuned Models ---------
LLAMA_2_7B_NO_TAGS="/data/jonathan/Lost-in-Mistranslation/models/llama2-ted2025-cpt-notags/final"
# --------------------------------------

# -------- Benchmarks ----------
GLOBAL="global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr,global_mmlu_full_de,global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja" #,global_mmlu_full_sw,global_mmlu_full_bn,global_mmlu_full_te
MGSM="mgsm_direct_bn,mgsm_direct_de,mgsm_direct_en,mgsm_direct_es,mgsm_direct_fr,mgsm_direct_ja,mgsm_direct_ru,mgsm_direct_sw,mgsm_direct_te,mgsm_direct_th,mgsm_direct_zh"
MGSM_COT="mgsm_cot_native_bn,mgsm_cot_native_de,mgsm_cot_native_en,mgsm_cot_native_es,mgsm_cot_native_fr,mgsm_cot_native_ja,mgsm_cot_native_ru,mgsm_cot_native_sw,mgsm_cot_native_te,mgsm_cot_native_th,mgsm_cot_native_zh"
# -------------------------------

MODEL="${OLMO_7B}" # Change this to the desired model

TIMESTAMP=$(date +%F_%H-%M-%S)
RUN_DIR="logs/mmlu_olmo_7b_${TIMESTAMP}"

mkdir -p "$RUN_DIR"


echo "Starting run at $TIMESTAMP"
echo "Run directory: $RUN_DIR"



# Run evaluation
lm_eval --model hf \
  --model_args "pretrained=$MODEL,device_map=auto,dtype=auto" \
  --tasks "$GLOBAL" \
  --batch_size auto \
  > >(tee "$RUN_DIR/stdout.log") \
  2> "$RUN_DIR/stderr.log"

echo "Run finished at $(date +%F_%H-%M-%S)"