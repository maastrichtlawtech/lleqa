#!/bin/bash

QUESTION_PATHS=(
    "data/lleqa/questions_train.json"
    "data/lleqa/questions_dev.json"
    "data/lleqa/questions_test.json"
)
CORPUS_PATH="data/lleqa/articles.json"
OUT_PATH="data/lleqa/negatives/"
TMP_PATH=$OUT_PATH/tmp_$(cat /proc/sys/kernel/random/uuid | tr -d '-' | head -c 8).json

MODEL="intfloat/multilingual-e5-large"
MAX_SEQ_LENGTH=512
SIM="cos_sim"
NUM_NEG=10

jq -s add "${QUESTION_PATHS[@]}" > $TMP_PATH && 
python src/retriever/biencoder_inference.py \
    --corpus_path $CORPUS_PATH \
    --queries_path $TMP_PATH \
    --output_dir $OUT_PATH \
    --model_name_or_path $MODEL \
    --max_seq_length $MAX_SEQ_LENGTH \
    --sim $SIM \
    --do_negatives_extraction \
    --num_negatives $NUM_NEG &&
rm $TMP_PATH
