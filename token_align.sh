#!/bin/sh

export MAIN_DIR="/Users/harry.odonnell/TokAlign/"
# git clone https://github.com/stanfordnlp/GloVe.git
export GLOVE_DIR="/Users/harry.odonnell/TokAlign/GloVe"

export MODLE_PATH1="allenai/OLMo-2-1124-7B"
export TOKENIZER_PATH1="allenai/OLMo-2-1124-7B"

export GLOVE_TRAIN_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-olmo-glove"
export GLOVE_TRAIN_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-gemma9b-glove"

export GLOVE_VECTOR_PATH1="${MAIN_DIR}/data/pretrain-dataset/mix-olmo-vectors.txt"
export GLOVE_VECTOR_PATH2="${MAIN_DIR}/data/pretrain-dataset/mix-gemma9b-vectors.txt"

export MODLE_PATH2="google/gemma-2-9b"
export TOKENIZER_PATH2="google/gemma-2-9b"

export TGT_ID_2_SRC_ID_GOLD_PATH="${MAIN_DIR}/data/Vocab_count/gemma9b_to_olmo.json"
export TGT_ID_2_SRC_ID_RES_PATH="${MAIN_DIR}/data/olmo_to_gemma9b/align_matrix.json"


# Stage-1: train glove vectors
cd ${GLOVE_DIR}
GLOVE_VECTOR_NAME1=$(basename ${GLOVE_VECTOR_PATH1})
GLOVE_VECTOR_NAME1="${GLOVE_VECTOR_NAME1%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME1} with ${GLOVE_TRAIN_PATH1}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH1} ${GLOVE_VECTOR_NAME1}
mv ${GLOVE_VECTOR_NAME1}.txt ${GLOVE_VECTOR_PATH1}

GLOVE_VECTOR_NAME2=$(basename ${GLOVE_VECTOR_PATH2})
GLOVE_VECTOR_NAME2="${GLOVE_VECTOR_NAME2%.*}"
printf "\n### Train GloVe vector ${GLOVE_VECTOR_NAME2} with ${GLOVE_TRAIN_PATH2}  ###\n\n"
bash ${MAIN_DIR}/script/train_glove.sh ${GLOVE_TRAIN_PATH2} ${GLOVE_VECTOR_NAME2}
mv ${GLOVE_VECTOR_NAME2}.txt ${GLOVE_VECTOR_PATH2}


# Stage-2: token ID align
cd ${MAIN_DIR}

export VOCAB_SIZE1=$(python src/count_vocab.py -m ${MODLE_PATH1})
export VOCAB_SIZE2=$(python src/count_vocab.py -m ${MODLE_PATH2})

python src/count_dict.py \
    -s ${TOKENIZER_PATH1} \
    -t ${TOKENIZER_PATH2} \
    -o ${TGT_ID_2_SRC_ID_GOLD_PATH}

python src/cal_trans_matrix.py \
    -s ${GLOVE_VECTOR_PATH1} \
    -s1 ${VOCAB_SIZE1} \
    -t ${GLOVE_VECTOR_PATH2} \
    -s2 ${VOCAB_SIZE2} \
    -r -n 300 \
    -g ${TGT_ID_2_SRC_ID_GOLD_PATH} \
    -o ${TGT_ID_2_SRC_ID_RES_PATH}
