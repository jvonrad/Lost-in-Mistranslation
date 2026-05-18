#!/bin/sh
set -e

export MAIN_DIR="/home/nvidia/jonathan/projects/Lost-in-Mistranslation/TokAlign"
cd "${MAIN_DIR}"

export TGT_ID_2_SRC_ID_RES_PATH="/data/jonathan/Lost-in-Mistranslation/tokalign/olmo2_to_custom151k/align_matrix.json"

export MODLE_PATH1="allenai/OLMo-2-1124-7B"
export TOKENIZER_PATH2="/data/jonathan/Lost-in-Mistranslation/tokenizers/olmo_12lang_bpe_151k"

export OUTPUT_PATH="/data/jonathan/Lost-in-Mistranslation/tokalign/olmo2_to_custom151k/TokAlign-Init-7B"

python src/convert.py \
    -m "${TGT_ID_2_SRC_ID_RES_PATH}" \
    -s "${MODLE_PATH1}" \
    -t "${TOKENIZER_PATH2}" \
    -o "${OUTPUT_PATH}"