#!/bin/sh

export MAIN_DIR="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/TokAlign"
cd "${MAIN_DIR}" || exit 1

export CACHE_DIR="/data/jonathan/cache"
#export TRAIN_FILE="/data/jonathan/Lost-in-Mistranslation/datasets/olmo_tokenizer_mix.txt"
export TRAIN_FILE="/data/jonathan/Lost-in-Mistranslation/datasets/olmo_tokenizer_mix.json"

# Source tokenizer/model
export MODLE_PATH1="allenai/OLMo-2-1124-7B"
export TOKENIZER_PATH1="allenai/OLMo-2-1124-7B"
export DATASET_PATH1="/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/olmo2_mix_tokenized"
export GLOVE_TRAIN_PATH1="/data/jonathan/Lost-in-Mistranslation/datasets/mix-olmo-glove"

# Target tokenizer/model
export MODLE_PATH2="allenai/OLMo-2-1124-7B"
export TOKENIZER_PATH2="/data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k"
export DATASET_PATH2="/data/jonathan/Lost-in-Mistranslation/datasets/tokenized/custom151k_tokenized"
export GLOVE_TRAIN_PATH2="/data/jonathan/Lost-in-Mistranslation/datasets/mix-custom151k-glove"

# Alignment eval output
export MATRIX_EVAL_PATH="${MAIN_DIR}/data/pretrain-dataset/olmo2-custom151k-glove-eval-mix"

export NUM_WORKERS=32

mkdir -p "${CACHE_DIR}"
mkdir -p "${DATASET_PATH1}" "${DATASET_PATH2}"
mkdir -p "$(dirname "${GLOVE_TRAIN_PATH1}")"
mkdir -p "$(dirname "${MATRIX_EVAL_PATH}")"

tokenize () {
  python -u src/process_dataset.py \
    --model_name_or_path "${MODLE_PATH}" \
    --tokenizer_name "${TOKENIZER_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --only_tokenize \
    --cache_dir "${CACHE_DIR}" \
    --dataset_path_in_disk "${DATASET_PATH}" \
    --preprocessing_num_workers "${NUM_WORKERS}" \
    --output_dir ./log 2>&1
}

# Stage 1: tokenize corpus with source tokenizer
MODLE_PATH="${MODLE_PATH1}"
TOKENIZER_PATH="${TOKENIZER_PATH1}"
DATASET_PATH="${DATASET_PATH1}"

printf "\n### Tokenize %s into token-ID corpus %s with tokenizer %s ###\n\n" \
  "${TRAIN_FILE}" "${DATASET_PATH1}" "${TOKENIZER_PATH1}"
tokenize

# Stage 1: tokenize corpus with target tokenizer
MODLE_PATH="${MODLE_PATH2}"
TOKENIZER_PATH="${TOKENIZER_PATH2}"
DATASET_PATH="${DATASET_PATH2}"

printf "\n### Tokenize %s into token-ID corpus %s with tokenizer %s ###\n\n" \
  "${TRAIN_FILE}" "${DATASET_PATH2}" "${TOKENIZER_PATH2}"
tokenize

MIN_LEN=0
MAX_LINE_TRAIN=1000000000
MAX_LINE_EVAL=1000

# Stage 2: extract token IDs for GloVe training
printf "\n### Extract token IDs from %s for GloVe training ###\n\n" "${DATASET_PATH1}"
python src/convert2glove_train.py \
  -s "${DATASET_PATH1}" \
  -k train \
  -m "${MIN_LEN}" \
  -l "${MAX_LINE_TRAIN}" \
  -o "${GLOVE_TRAIN_PATH1}"

printf "\n### Extract token IDs from %s for GloVe training ###\n\n" "${DATASET_PATH2}"
python src/convert2glove_train.py \
  -s "${DATASET_PATH2}" \
  -k train \
  -m "${MIN_LEN}" \
  -l "${MAX_LINE_TRAIN}" \
  -o "${GLOVE_TRAIN_PATH2}"

MIN_LEN=10

printf "\n### Extract aligned token IDs from source (%s) and target (%s) for matrix evaluation ###\n\n" \
  "${DATASET_PATH1}" "${DATASET_PATH2}"
python src/convert2glove_train.py \
  -s "${DATASET_PATH1}" \
  -t "${DATASET_PATH2}" \
  -k validation \
  -m "${MIN_LEN}" \
  -l "${MAX_LINE_EVAL}" \
  -o "${MATRIX_EVAL_PATH}"