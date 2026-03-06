#!/usr/bin/env bash
set -e

# --------- GPU Config -------
export CUDA_VISIBLE_DEVICES=6 # use gpu=x

# ---------- Models ----------
LLAMA_2_7B="meta-llama/Llama-2-7b-hf"
OLMO_7B="allenai/OLMo-7B-hf"
OLMO_3_7B="allenai/Olmo-3-7B-Instruct"
OLMO_3_7B_BASE="allenai/Olmo-3-1025-7B"
# ----------------------------

# --------- Fine-tuned Models ---------
LLAMA_2_7B_NO_TAGS="/data/jonathan/Lost-in-Mistranslation/models/llama2-ted2025-cpt-notags/final"
LLAMA_2_7B_60_STEPS_NO_TAGS="/data/jonathan/Lost-in-Mistranslation/models/Llama-2-7b-hf-ted2025-cpt-60steps-notags/final"
# --------------------------------------

# -------- Benchmarks ----------
# Knowledge
GLOBAL_FULL="global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr,global_mmlu_full_de,global_mmlu_full_id,global_mmlu_full_pt,global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja,global_mmlu_full_ar,global_mmlu_full_sw,global_mmlu_full_bn"
GLOBAL_1="global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr,global_mmlu_full_de"
GLOBAL_2="global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja,global_mmlu_full_ar"
GLOBAL_3="global_mmlu_full_sw,global_mmlu_full_bn,global_mmlu_full_pt,global_mmlu_full_id"

# Translation
FLORES_1="flores200:eng_Latn-deu_Latn,flores200:eng_Latn-spa_Latn,flores200:eng_Latn-fra_Latn,flores200:eng_Latn-por_Latn"
FLORES_2="flores200:eng_Latn-rus_Cyrl,flores200:eng_Latn-zho_Hans,flores200:eng_Latn-jpn_Jpan,flores200:eng_Latn-arb_Arab"
FLORES_3="flores200:eng_Latn-swh_Latn,flores200:eng_Latn-ben_Beng,flores200:eng_Latn-ind_Latn"

# Reasoning
MGSM="mgsm_direct_bn,mgsm_direct_de,mgsm_direct_en,mgsm_direct_es,mgsm_direct_fr,mgsm_direct_ja,mgsm_direct_ru,mgsm_direct_sw,mgsm_direct_te,mgsm_direct_th,mgsm_direct_zh"
MGSM_COT="mgsm_cot_native_bn,mgsm_cot_native_de,mgsm_cot_native_en,mgsm_cot_native_es,mgsm_cot_native_fr,mgsm_cot_native_ja,mgsm_cot_native_ru,mgsm_cot_native_sw,mgsm_cot_native_te,mgsm_cot_native_th,mgsm_cot_native_zh"
# -------------------------------

# --------- Run Config ---------
CURR_MODEL="allenai/OLMo-2-1124-7B" # <-- change this to the model you want to evaluate
MODEL_NAME=${CURR_MODEL//\//_}
TIMESTAMP=$(date +%F_%H-%M-%S)
RUN_DIR="logs/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
echo "Starting run at $TIMESTAMP"
echo "Run directory: $RUN_DIR"
# ----------------------------



# --------- LIGHTEVAL --------------
# needed packages:
# pip install lighteval
# pip install language_data
# pip install "datasets>=2.19,<4.0.0"

export VLLM_USE_V1=0
export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL"

# ---------- Translation ------ Flores 200 ---------
lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_3" --load-tasks-multilingual  \
  > >(tee "$RUN_DIR/stdout.log") \
  2> "$RUN_DIR/stderr.log"

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_2" --load-tasks-multilingual \
#   > >(tee "$RUN_DIR/stdout.log") \
#   2> "$RUN_DIR/stderr.log"

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_3" --load-tasks-multilingual \
#   > >(tee "$RUN_DIR/stdout.log") \
#   2> "$RUN_DIR/stderr.log"

# --------- LM EVAL HARNESS ---------

# # Run evaluation
# lm_eval --model hf \
#   --model_args "pretrained=$CURR_MODEL,device_map=auto,dtype=auto" \
#   --tasks "$GLOBAL_FULL" \
#   --batch_size auto \
#   > >(tee "$RUN_DIR/stdout.log") \
#   2> "$RUN_DIR/stderr.log"

# echo "Run finished at $(date +%F_%H-%M-%S)"


# # using vllm
# lm_eval --model vllm \
#     --model_args pretrained=$CURR_MODEL,dtype=auto \
#     --tasks $GLOBAL_3 \
#     --batch_size auto\
#   > >(tee "$RUN_DIR/stdout.log") \
#   2> "$RUN_DIR/stderr.log"


# with peft model
# lm_eval --model hf \
#   --model_args pretrained=$CURR_MODEL,peft=/data/jonathan/Lost-in-Mistranslation/models/OLMo-2-1124-7B-ted2025-cpt-lora-720steps-r32-multilingual/final_lora/ \
#   --tasks $GLOBAL_3 \
#   --batch_size auto \
#   --device cuda

# ------- Model Training --------

# export WANDB_PROJECT="UnLock"
# export WANDB_ENTITY="jonathan-von-rad"
# export WANDB_DISABLED=false

# -------- Train Tokenizer ----------

# python train_tokenizer.py \
#   --base_model allenai/OLMo-2-1124-7B \
#   --out_dir /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo2_tok_ext_mined_30k \
#   --langs ar bn ru ja zh \
#   --max_docs_per_lang 200000 \
#   --min_chars 200 \
#   --num_new_tokens 30000 \
#   --ngram_max 3 \
#   --max_script_chars_per_lang 30000000