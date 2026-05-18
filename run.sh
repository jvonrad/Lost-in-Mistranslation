#!/usr/bin/env bash
set -e

# --------- GPU Config -------
export CUDA_VISIBLE_DEVICES=2

# ---------- Models ----------
LLAMA_2_7B="meta-llama/Llama-2-7b-hf"
OLMO_7B="allenai/OLMo-7B-hf"
OLMO_3_7B="allenai/Olmo-3-7B-Instruct"
OLMO_3_7B_BASE="allenai/Olmo-3-1025-7B"
OLMO_2_7B_CULTURA="/data/jonathan/Lost-in-Mistranslation/models/olmo2-culturax-ar-bn-ru-cpt/merged"
# ----------------------------

# --------- Tokenizer-extended Models ----------------
OLMO_2_7B_NEW_TOK="/data/jonathan/Lost-in-Mistranslation/tokalign/olmo2_to_custom151k/TokAlign-Init-7B"

# --------- Fine-tuned Models ---------
LLAMA_2_7B_NO_TAGS="/data/jonathan/Lost-in-Mistranslation/models/llama2-ted2025-cpt-notags/final"
LLAMA_2_7B_60_STEPS_NO_TAGS="/data/jonathan/Lost-in-Mistranslation/models/Llama-2-7b-hf-ted2025-cpt-60steps-notags/final"
OLMO_2_7B_STRUCTURED_LORA="/data/jonathan/Lost-in-Mistranslation/models/olmo2-ted-structured-lora-final/merged"
OLMO_2_7B_BASE_FINETUNED="/data/jonathan/Lost-in-Mistranslation/models/OLMo-2-1124-7B-ted2025-multilingual-lora/"
OLMO_2_7B_BASE_SW_BN_TED_CULTURA="/data/jonathan/Lost-in-Mistranslation/models/olmo2-ted-cultura-sw-bn-structured-lora-final/merged"
OLMO_2_7B_BASE_FINETUNED_PRETOKENIZED="/data/jonathan/Lost-in-Mistranslation/models/OLMo-2-1124-7B-ted2025-cpt-fullft-300steps-multilingual-notags/final"
OLMO_2_7B_FINETRANSLATIONS_LORA="/data/jonathan/Lost-in-Mistranslation/models/olmo2-finetranslations-structured-lora-checkpoints/checkpoint-12400-merged"
OLMO_2_7B_WIKIFACT_SFT="/data/jonathan/Lost-in-Mistranslation/models/wikifact_sft_lora/merged"
OLMO_2_7B_TED_WIKIFACT_GRPO="/data/jonathan/Lost-in-Mistranslation/models/aligned-ted-wikifact-grpo/checkpoint-8400-merged"
OLMO_2_7B_FINE_WIKIFACT_GRPO="/data/jonathan/Lost-in-Mistranslation/models/aligned-finetranslations-wikifact-grpo/checkpoint-3200-merged"
OLMO_2_7B_FINETRANSLATION_WIKIFACT_GRPO_ATT_MLP="jonny-vr/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
OLMO_2_7B_FINETRANSLATION_WIKIFACT_GRPO_ATT_MLP_FINAL="/data/jonathan/Lost-in-Mistranslation/models/aligned-finetranslations-wikifact-grpo-att-mlp/checkpoint-10800-merged"
OLMO_2_7B_FINETRANSLATION_WIKIFACT_GRPO_ATT="/data/jonathan/Lost-in-Mistranslation/models/aligned-finetranslations-wikifact-grpo-att/checkpoint-2100-merged"
OLMO_2_7B_FINE_WIKIFACT_GRPO_ATT_FINAL="/data/jonathan/Lost-in-Mistranslation/models/aligned-finetranslations-wikifact-grpo-final/checkpoint-25000-merged"
# --------------------------------------

# -------- Benchmarks ----------
# Knowledge
GLOBAL_FULL="global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr,global_mmlu_full_de,global_mmlu_full_id,global_mmlu_full_pt,global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja,global_mmlu_full_ar,global_mmlu_full_sw,global_mmlu_full_bn"
GLOBAL_1="global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr,global_mmlu_full_de"
GLOBAL_2="global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja,global_mmlu_full_ar"
GLOBAL_3="global_mmlu_full_sw,global_mmlu_full_bn"
GLOBAL_4="global_mmlu_full_pt,global_mmlu_full_id"
GLOBAL_RU_BN_AR="global_mmlu_full_ru,global_mmlu_full_bn,global_mmlu_full_ar"
GLOBAL_KLAR="global_mmlu_full_ar,global_mmlu_full_en,global_mmlu_full_es,global_mmlu_full_fr"
GLOBAL_KLAR2="global_mmlu_full_ja,global_mmlu_full_ru,global_mmlu_full_zh"
  
# Translation
FLORES_1="flores200:eng_Latn-deu_Latn,flores200:eng_Latn-spa_Latn,flores200:eng_Latn-fra_Latn,flores200:eng_Latn-por_Latn"
FLORES_2="flores200:eng_Latn-rus_Cyrl,flores200:eng_Latn-zho_Hans,flores200:eng_Latn-jpn_Jpan,flores200:eng_Latn-arb_Arab"
FLORES_3="flores200:eng_Latn-swh_Latn,flores200:eng_Latn-ben_Beng,flores200:eng_Latn-ind_Latn"
FLORES_FULL="flores200:eng_Latn-deu_Latn,flores200:eng_Latn-spa_Latn,flores200:eng_Latn-fra_Latn,flores200:eng_Latn-por_Latn,flores200:eng_Latn-rus_Cyrl,flores200:eng_Latn-zho_Hans,flores200:eng_Latn-jpn_Jpan,flores200:eng_Latn-arb_Arab,flores200:eng_Latn-swh_Latn,flores200:eng_Latn-ben_Beng,flores200:eng_Latn-ind_Latn"

# Truthfulness
XTRUTHFUL="truthfulqa_mc1,truthfulqa_ar_mc1,truthfulqa_bn_mc1,truthfulqa_de_mc1,truthfulqa_es_mc1,truthfulqa_fr_mc1,truthfulqa_id_mc1,truthfulqa_pt_mc1,truthfulqa_ru_mc1,truthfulqa_zh_mc1"

# Reasoning
XNLI="xnli_ar,xnli_bg,xnli_de,xnli_el,xnli_en,xnli_es,xnli_fr,xnli_hi,xnli_ru,xnli_sw,xnli_th,xnli_tr,xnli_ur,xnli_vi,xnli_zh"
XCOPA="xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh"
MGSM="mgsm_direct_bn,mgsm_direct_de,mgsm_direct_en,mgsm_direct_es,mgsm_direct_fr,mgsm_direct_ja,mgsm_direct_ru,mgsm_direct_sw,mgsm_direct_te,mgsm_direct_th,mgsm_direct_zh"
MGSM_COT="mgsm_cot_native_bn,mgsm_cot_native_de,mgsm_cot_native_en,mgsm_cot_native_es,mgsm_cot_native_fr,mgsm_cot_native_ja,mgsm_cot_native_ru,mgsm_cot_native_sw,mgsm_cot_native_te,mgsm_cot_native_th,mgsm_cot_native_zh"
# -------------------------------

# ----------- Final Models -----------
BASELINE="allenai/OLMo-2-1124-7B"
ALIGNED_SFT="jonny-vr/olmo-2-7b-aligned-wikifact-sft"
ALIGNED_GRPO="jonny-vr/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
ONLY_GRPO="jonny-vr/olmo-2-7b-wikifact-grpo"
# ONLY_BASE_GRPO="/data/jonathan/Lost-in-Mistranslation/models/olmo-base-wikifact-grpo-lr-1e-5/checkpoint-2500-merged"
ONLY_SFT="jonny-vr/olmo-2-7B.wikifact-sft-consistent"
ONLY_ALIGNED="jonny-vr/olmo-2-7b-finetranslations"
TED_ALIGNED=""/data/jonathan/Lost-in-Mistranslation/models/olmo2-ted-structured-lora-final/merged""

# --------- Run Config ---------
CURR_MODEL="jvonrad/olmo-2-7b-wikifact-sft"
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

# export VLLM_USE_V1=0
# export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL,tokenizer=allenai/OLMo-2-1124-7B"

# ---------- Translation ------ Flores 200 ---------
# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_1" --load-tasks-multilingual 

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_2" --load-tasks-multilingual

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_3" --load-tasks-multilingual

# --------- LM EVAL HARNESS ---------


#using vllm
lm_eval --model vllm \
    --model_args pretrained=$CURR_MODEL,tokenizer=allenai/OLMo-2-1124-7B,dtype=auto \
    --tasks $GLOBAL_MMLU_FULL \
    --batch_size auto 
    
    
echo "Evaluation of $CURR_MODEL completed."
    
#     \
#   > >(tee "$RUN_DIR/stdout.log") \
#   2> "$RUN_DIR/stderr.log"


# lm_eval --model vllm   --model_args pretrained=/data/jonathan/Lost-in-Mistranslation/models/olmo2-klar-full/final,tokenizer=allenai/OLMo-2-1124-7B,dtype=auto,trust_remote_code=True   --tasks $GLOBAL_KLAR2   --batch_size auto


# ------- Model Training --------

# CUDA_VISIBLE_DEVICES=4,7 \
# torchrun --standalone --nproc_per_node=2 training/train_ted_structured_lora.py

# export WANDB_PROJECT="UnLock"
# export WANDB_ENTITY="jonathan-von-rad"
# export WANDB_DISABLED=false

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 training/train_culturax.py \
#   --base_model allenai/OLMo-2-1124-7B \
#   --tokenizer_path /data/jonathan/Lost-in-Mistranslation/tokenizers/olmo2_tok_ext_bn_ru \
#   --output_dir /data/jonathan/Lost-in-Mistranslation/models/olmo2-culturax-bn-ru-cpt \
#   --langs bn ru \
#   --max_steps 3000

####################
# Train for CL-Consistency
####################


# python generate_multilingual_mcq_bundles.py \
#   --seed_facts_jsonl seed_facts.jsonl \
#   --output_jsonl multilingual_mcq_bundles.jsonl \
#   --target_total_tokens 200000000

#   python pretokenize_multilingual_mcq_bundles.py \
#   --input_jsonl multilingual_mcq_bundles.jsonl \
#   --output_dir /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/multilingual_mcq_bundles_12lang \
#   --max_length 1024

#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 train_multilingual_consistency_lora.py \
#   --dataset_path /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/multilingual_mcq_bundles_12lang


# ------- Pretokenize ----------

# Klar dataset

# python pretokenize_klar.py \
#   --klar_root /data/jonathan/Lost-in-Mistranslation/datasets/KLAR-CLC \
#   --model_name allenai/OLMo-2-1124-7B \
#   --output_dir /data/jonathan/Lost-in-Mistranslation/datasets/tokenized/klar-olmo2 

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