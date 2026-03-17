#!/bin/sh

export MAIN_DIR="/Users/harry.odonnell/TokAlign/"
cd ${MAIN_DIR}

# The path of token alignment matrix
# export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/pythia2gemma/align_matrix.json"
# Point this to the output path you set in token_align.sh
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/olmo_to_gemma9b/align_matrix.json"

# This must match the MATRIX_EVAL_PATH from convert2glove_corpus.sh
export MATRIX_EVAL_DATA_PATH="${MAIN_DIR}/data/pretrain-dataset/pythia-2-gemma-glove-eval-mix"

# BLEU-1 evaluation
export EVAL_METHOD=bleu
export BLEU_WEIGHT="1,0,0,0"

# Bert-score evaluation
export EVAL_METHOD=bert-score
export BERT_SOCRE_EVAL_MODEL="all-mpnet-base-v2"
export TOKENIZER_PATH="allenai/OLMo-2-1124-7B"

python src/eval_matrix.py \
    -e ${EVAL_METHOD} \
    -m ${TGT_ID_2_SRC_ID_RES_PATH} \
    -f ${MATRIX_EVAL_DATA_PATH} \
    -t ${TOKENIZER_PATH} \
    -b ${BERT_SOCRE_EVAL_MODEL} \
    -w ${BLEU_WEIGHT}