#!/bin/bash

QUESTION_PATHS=(
    "data/lleqa/questions_train.json"
    "data/lleqa/questions_dev.json"
    "data/lleqa/questions_test.json"
)
CORPUS_PATH="data/lleqa/articles.json"
OUT_PATH="data/lleqa/negatives"
TMP_PATH=$OUT_PATH/tmp_$(cat /proc/sys/kernel/random/uuid | tr -d '-' | head -c 8).json

K1=2.5
B=0.2
NUM_NEG=10

jq -s add "${QUESTION_PATHS[@]}" > $TMP_PATH && 
python src/retriever/bm25.py \
    --corpus_path $CORPUS_PATH \
    --queries_path $TMP_PATH \
    --output_dir $OUT_PATH \
    --do_preprocessing \
    --k1 $K1 \
    --b $B \
    --do_negatives_extraction \
    --num_negatives $NUM_NEG &&
rm $TMP_PATH
