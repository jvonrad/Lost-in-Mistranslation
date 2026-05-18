#!/bin/sh
set -e

export MAIN_DIR="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/TokAlign"
export GLOVE_DIR="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/TokAlign/GloVe"

# Source model/tokenizer
export MODLE_PATH1="allenai/OLMo-2-1124-7B"
export TOKENIZER_PATH1="allenai/OLMo-2-1124-7B"

# Target tokenizer (custom)
export MODLE_PATH2="/data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k"
export TOKENIZER_PATH2="/data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k"

# GloVe training corpora
export GLOVE_TRAIN_PATH1="/data/jonathan/Lost-in-Mistranslation/datasets/mix-olmo-glove"
export GLOVE_TRAIN_PATH2="/data/jonathan/Lost-in-Mistranslation/datasets/mix-custom151k-glove"

# Alignment eval file (IMPORTANT — you forgot this before)
export MATRIX_EVAL_PATH="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/TokAlign/data/pretrain-dataset/olmo2-custom151k-glove-eval-mix"

# Output dirs/files
export ALIGN_OUT_DIR="/data/jonathan/Lost-in-Mistranslation/tokalign/olmo2_to_custom151k"
mkdir -p "${ALIGN_OUT_DIR}"
mkdir -p "${MAIN_DIR}/data/Vocab_count"

export GLOVE_VECTOR_PATH1="${ALIGN_OUT_DIR}/mix-olmo-vectors.txt"
export GLOVE_VECTOR_PATH2="${ALIGN_OUT_DIR}/mix-custom151k-vectors.txt"

export TGT_ID_2_SRC_ID_GOLD_PATH="${ALIGN_OUT_DIR}/custom151k_to_olmo_gold.json"
export TGT_ID_2_SRC_ID_RES_PATH="${ALIGN_OUT_DIR}/align_matrix.json"

# ========================
# Stage 1: train GloVe
# ========================
# cd "${GLOVE_DIR}"

# GLOVE_VECTOR_NAME1=$(basename "${GLOVE_VECTOR_PATH1}")
# GLOVE_VECTOR_NAME1="${GLOVE_VECTOR_NAME1%.*}"

# printf "\n### Train GloVe vector %s with %s ###\n\n" "${GLOVE_VECTOR_NAME1}" "${GLOVE_TRAIN_PATH1}"
# bash "${MAIN_DIR}/script/train_glove.sh" "${GLOVE_TRAIN_PATH1}" "${GLOVE_VECTOR_NAME1}"
# mv "${GLOVE_VECTOR_NAME1}.txt" "${GLOVE_VECTOR_PATH1}"

# GLOVE_VECTOR_NAME2=$(basename "${GLOVE_VECTOR_PATH2}")
# GLOVE_VECTOR_NAME2="${GLOVE_VECTOR_NAME2%.*}"

# printf "\n### Train GloVe vector %s with %s ###\n\n" "${GLOVE_VECTOR_NAME2}" "${GLOVE_TRAIN_PATH2}"
# bash "${MAIN_DIR}/script/train_glove.sh" "${GLOVE_TRAIN_PATH2}" "${GLOVE_VECTOR_NAME2}"
# mv "${GLOVE_VECTOR_NAME2}.txt" "${GLOVE_VECTOR_PATH2}"

# ========================
# Stage 2: alignment
# ========================
cd "${MAIN_DIR}"

export VOCAB_SIZE1=$(python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
print(len(tok))
PY
)

export VOCAB_SIZE2=$(python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k")
print(len(tok))
PY
)

printf "\n### Source vocab size: %s ###\n" "${VOCAB_SIZE1}"
printf "### Target vocab size: %s ###\n\n" "${VOCAB_SIZE2}"

# gold dictionary from tokenizer overlap
python src/count_dict.py \
    -s "${TOKENIZER_PATH1}" \
    -t "${TOKENIZER_PATH2}" \
    -o "${TGT_ID_2_SRC_ID_GOLD_PATH}"

# alignment matrix
python src/cal_trans_matrix.py \
    -s "${GLOVE_VECTOR_PATH1}" \
    -s1 "${VOCAB_SIZE1}" \
    -t "${GLOVE_VECTOR_PATH2}" \
    -s2 "${VOCAB_SIZE2}" \
    -r -n 300 \
    -g "${TGT_ID_2_SRC_ID_GOLD_PATH}" \
    -o "${TGT_ID_2_SRC_ID_RES_PATH}"

printf "\n### Alignment matrix saved to: %s ###\n" "${TGT_ID_2_SRC_ID_RES_PATH}"