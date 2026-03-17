#!/bin/sh

# Update your working directory
export MAIN_DIR="/Users/harry.odonnell/TokAlign/"
cd ${MAIN_DIR}

# Point to the real matrix you generated (comment out the demo)
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/olmo_to_gemma9b/align_matrix.json"
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2gemma/align_matrix_demo.json"

# Set the Source Model (OLMo)
export MODLE_PATH1="allenai/OLMo-2-1124-7B"

#Set the Target Tokenizer (Gemma)
export TOKENIZER_PATH2="google/gemma-2-9b"

# Define where the new hybrid model will be saved
export OUTPUT_PATH="${MAIN_DIR}/data/olmo_to_gemma9b/TokAlign-Init-7B"

python src/convert.py \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -s ${MODLE_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${OUTPUT_PATH}