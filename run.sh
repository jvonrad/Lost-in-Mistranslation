#!/usr/bin/env bash
set -e

# --------- GPU Config -------
export CUDA_VISIBLE_DEVICES=5
# GPUs used by the parallel benchmark sweep (one model per GPU, run concurrently)
BENCH_GPUS=(1 2 5)

# ---------- Models ----------
LLAMA_2_7B="meta-llama/Llama-2-7b-hf"


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
GLOBAL_LITE="global_mmlu_en,global_mmlu_es,global_mmlu_fr,global_mmlu_de,global_mmlu_zh,global_mmlu_ja,global_mmlu_ar,global_mmlu_sw,global_mmlu_bn,global_mmlu_pt,global_mmlu_id"

  
# Translation
FLORES_1="flores200:eng_Latn-deu_Latn,flores200:eng_Latn-spa_Latn,flores200:eng_Latn-fra_Latn,flores200:eng_Latn-por_Latn"
FLORES_2="flores200:eng_Latn-rus_Cyrl,flores200:eng_Latn-zho_Hans,flores200:eng_Latn-jpn_Jpan,flores200:eng_Latn-arb_Arab"
FLORES_3="flores200:eng_Latn-swh_Latn,flores200:eng_Latn-ben_Beng,flores200:eng_Latn-ind_Latn"
FLORES_FULL="flores200:eng_Latn-deu_Latn,flores200:eng_Latn-spa_Latn,flores200:eng_Latn-fra_Latn,flores200:eng_Latn-por_Latn,flores200:eng_Latn-rus_Cyrl,flores200:eng_Latn-zho_Hans,flores200:eng_Latn-jpn_Jpan,flores200:eng_Latn-arb_Arab,flores200:eng_Latn-swh_Latn,flores200:eng_Latn-ben_Beng,flores200:eng_Latn-ind_Latn"
FLORES_FULL_X2EN="flores200:deu_Latn-eng_Latn,flores200:spa_Latn-eng_Latn,flores200:fra_Latn-eng_Latn,flores200:por_Latn-eng_Latn,flores200:rus_Cyrl-eng_Latn,flores200:zho_Hans-eng_Latn,flores200:jpn_Jpan-eng_Latn,flores200:arb_Arab-eng_Latn,flores200:swh_Latn-eng_Latn,flores200:ben_Beng-eng_Latn,flores200:ind_Latn-eng_Latn"
FLORES_BIDIR="${FLORES_FULL},${FLORES_FULL_X2EN}"

# Truthfulness
XTRUTHFUL="truthfulqa_mc1,truthfulqa_ar_mc1,truthfulqa_bn_mc1,truthfulqa_de_mc1,truthfulqa_es_mc1,truthfulqa_fr_mc1,truthfulqa_id_mc1,truthfulqa_pt_mc1,truthfulqa_ru_mc1,truthfulqa_zh_mc1"

# Reasoning
XNLI="xnli_ar,xnli_bg,xnli_de,xnli_el,xnli_en,xnli_es,xnli_fr,xnli_hi,xnli_ru,xnli_sw,xnli_th,xnli_tr,xnli_ur,xnli_vi,xnli_zh"
XCOPA="xcopa_et,xcopa_ht,xcopa_id,xcopa_it,xcopa_qu,xcopa_sw,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi,xcopa_zh"
MGSM="mgsm_direct_bn,mgsm_direct_de,mgsm_direct_en,mgsm_direct_es,mgsm_direct_fr,mgsm_direct_ja,mgsm_direct_ru,mgsm_direct_sw,mgsm_direct_te,mgsm_direct_th,mgsm_direct_zh"
MGSM_COT="mgsm_cot_native_bn,mgsm_cot_native_de,mgsm_cot_native_en,mgsm_cot_native_es,mgsm_cot_native_fr,mgsm_cot_native_ja,mgsm_cot_native_ru,mgsm_cot_native_sw,mgsm_cot_native_te,mgsm_cot_native_th,mgsm_cot_native_zh"
# -------------------------------

# # ----------- Final Models -----------
# BASELINE="allenai/OLMo-2-1124-7B"
# ALIGNED_SFT="jonny-vr/olmo-2-7b-aligned-wikifact-sft"
# ALIGNED_GRPO="jonny-vr/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
# ONLY_GRPO="jonny-vr/olmo-2-7b-wikifact-grpo"
# # ONLY_BASE_GRPO="/data/jonathan/Lost-in-Mistranslation/models/olmo-base-wikifact-grpo-lr-1e-5/checkpoint-2500-merged"
# ONLY_SFT="jonny-vr/olmo-2-7B.wikifact-sft-consistent"
# ONLY_ALIGNED="jonny-vr/olmo-2-7b-finetranslations"
# TED_ALIGNED=""/data/jonathan/Lost-in-Mistranslation/models/olmo2-ted-structured-lora-final/merged""

# ---------- Base Models ----------
OLMO_2_7B="allenai/OLMo-2-1124-7B"
QWEN_2_5_7B="Qwen/Qwen2.5-7B"
LLAMA_3_1_8B="meta-llama/Llama-3.1-8B"

# ---------- TED Aligned Models ----------
OLMO_2_7B_TED="jvonrad/OLMo-2-1124-7B-TED"
QWEN_2_5_7B_TED="jvonrad/Qwen-2.5-7B-TED"
LLAMA_3_1_8B_TED="jvonrad/Llama-3.1-8B-TED"
GEMMA_2_9B_TED="jvonrad/Gemma-2-9B-TED"

# ---------- PolyFact GRPO Models --------
OLMO_2_7B_GRPO="jvonrad/OLMo-2-7B-grpo"
QWEN_2_5_7B_GRPO="jvonrad/Qwen-2.5-7B-grpo"
LLAMA_3_1_8B_GRPO="jvonrad/Llama-3.1-8B-grpo"
OLMO_2_7B_TED_GRPO="jvonrad/OLMo-2-7B-TED-grpo"
QWEN_2_5_7B_TED_GRPO="jvonrad/Qwen-2.5-7B-TED-grpo"
LLAMA_3_1_8B_TED_GRPO="jvonrad/Llama-3.1-8B-TED-grpo"




# --------- Run Config ---------
CURR_MODEL="jvonrad/olmo-2-7b-finetranslation-wikifact-grpo-att-mlp-checkpoint"
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
# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_1" --load-tasks-multilingual 

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_2" --load-tasks-multilingual

# lighteval accelerate "$LIGHTEVAL_CONFIG" "$FLORES_3" --load-tasks-multilingual

# --------- LM EVAL HARNESS ---------


#using vllm
lm_eval --model vllm \
    --model_args pretrained=$CURR_MODEL,tokenizer="allenai/OLMo-2-1124-7B",dtype=auto \
    --tasks $GLOBAL_LITE \
    --batch_size auto 

    
    
# echo "Evaluation of $CURR_MODEL completed."
    
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



# # =====================================================================
# # TED Benchmark Sweep
# # Languages: en, es, fr, de, id, pt, ru, zh, ja, ar, sw, bn
# # Benchmarks: belebele, xnli, xcopa (lm-eval-harness) + flores200 (lighteval)
# # =====================================================================

# # ----- Fine-tuned model -> base model pairs -----
# BENCH_PAIRS=(
#     "jvonrad/OLMo-2-1124-7B-TED|allenai/OLMo-2-1124-7B"
#     "jvonrad/Llama-3.1-8B-TED|meta-llama/Llama-3.1-8B"
#     "jvonrad/Gemma-2-9B-TED|google/gemma-2-9b"
#     "jvonrad/Qwen-2.5-7B-TED|Qwen/Qwen2.5-7B"
# )

# # ----- lm-eval-harness task sets (12-lang target, intersected with task availability) -----
# # belebele: full 12-lang coverage
# BELEBELE_TASKS="belebele_eng_Latn,belebele_spa_Latn,belebele_fra_Latn,belebele_deu_Latn,belebele_ind_Latn,belebele_por_Latn,belebele_rus_Cyrl,belebele_zho_Hans,belebele_jpn_Jpan,belebele_arb_Arab,belebele_swh_Latn,belebele_ben_Beng"
# # xnli: en, es, fr, de, ru, zh, ar, sw  (id/pt/ja/bn unavailable in lm-eval xnli)
# XNLI_TASKS_12="xnli_en,xnli_es,xnli_fr,xnli_de,xnli_ru,xnli_zh,xnli_ar,xnli_sw"
# # xcopa: id, sw, zh  (other target langs not in xcopa dataset)
# XCOPA_TASKS_12="xcopa_id,xcopa_sw,xcopa_zh"

# LM_EVAL_TASKS="$BELEBELE_TASKS,$XNLI_TASKS_12,$XCOPA_TASKS_12"

# # flores200 translation (lighteval) — FLORES_BIDIR covers eng_Latn->X and X->eng_Latn for all 11 langs

# # Set BENCH_ROOT_REUSE=logs/bench_<ts> to resume into an existing sweep dir;
# # otherwise a fresh timestamped dir is created.
# if [[ -n "${BENCH_ROOT_REUSE:-}" ]]; then
#     BENCH_ROOT="$BENCH_ROOT_REUSE"
#     echo "Reusing existing bench dir: $BENCH_ROOT"
# else
#     BENCH_TS=$(date +%F_%H-%M-%S)
#     BENCH_ROOT="logs/bench_${BENCH_TS}"
# fi
# mkdir -p "$BENCH_ROOT"

# # All benchmarks share one conda env.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$HOME/jonathan/envs/eval"

# # Returns 0 (success) if the given dir already has at least one results_*.json — used to skip
# # benchmarks that have already completed during a previous run.
# has_results() {
#     local D="$1"
#     [[ -d "$D" ]] || return 1
#     [[ -n "$(find "$D" -type f -name 'results_*.json' -print -quit 2>/dev/null)" ]]
# }

# # Run lm-eval (belebele/xnli/xcopa) then lighteval (flores200) for one model on one GPU.
# # Skips either phase if its output dir already contains results.
# bench_one_model() {
#     local GPU="$1"
#     local M="$2"
#     local SAFE="${M//\//_}"
#     local OUT_DIR="$BENCH_ROOT/$SAFE"
#     mkdir -p "$OUT_DIR"
#     {
#         echo "[$M] GPU=$GPU START $(date)"

#         if has_results "$OUT_DIR/lm_eval"; then
#             echo "[$M] lm_eval results already present — skipping"
#         else
#             CUDA_VISIBLE_DEVICES="$GPU" \
#                 lm_eval --model vllm \
#                     --model_args "pretrained=$M,dtype=auto,trust_remote_code=True,gpu_memory_utilization=0.85" \
#                     --tasks "$LM_EVAL_TASKS" \
#                     --batch_size auto \
#                     --output_path "$OUT_DIR/lm_eval"
#         fi

#         if has_results "$OUT_DIR/lighteval"; then
#             echo "[$M] lighteval results already present — skipping"
#         else
#             CUDA_VISIBLE_DEVICES="$GPU" \
#                 lighteval accelerate "model_name=$M" "$FLORES_BIDIR" \
#                     --load-tasks-multilingual \
#                     --output-dir "$OUT_DIR/lighteval"
#         fi

#         echo "[$M] GPU=$GPU DONE  $(date)"
#     } >>"$OUT_DIR/run.log" 2>&1
# }

# # Build the model list. Override with BENCH_MODELS (space-separated) to limit the sweep
# # to specific models, e.g.
# #   BENCH_MODELS="jvonrad/Llama-3.1-8B-TED allenai/OLMo-2-1124-7B" bash run.sh
# if [[ -n "${BENCH_MODELS:-}" ]]; then
#     read -r -a ALL_MODELS <<< "$BENCH_MODELS"
# else
#     ALL_MODELS=()
#     for PAIR in "${BENCH_PAIRS[@]}"; do
#         ALL_MODELS+=("${PAIR%%|*}" "${PAIR##*|}")
#     done
# fi

# # Don't let one failed model kill the rest of the sweep.
# set +e

# NGPU=${#BENCH_GPUS[@]}
# for i in "${!ALL_MODELS[@]}"; do
#     GPU="${BENCH_GPUS[$(( i % NGPU ))]}"
#     M="${ALL_MODELS[$i]}"
#     echo ">>> Launching: $M on GPU $GPU  (log: $BENCH_ROOT/${M//\//_}/run.log)"
#     bench_one_model "$GPU" "$M" &
#     # After every NGPU launches, wait for the batch to drain before starting the next.
#     if (( (i + 1) % NGPU == 0 )); then
#         wait
#     fi
# done
# wait

# set -e
# echo "All benchmarks complete. Results under: $BENCH_ROOT"
# # =====================================================================
