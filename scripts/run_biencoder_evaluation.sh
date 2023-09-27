#!/bin/bash

QUESTION_PATH="data/lleqa/questions_test.json"
CORPUS_PATH="data/lleqa/articles.json"
OUT_PATH="output/testing/"

MODEL="camembert-base"
MAX_SEQ_LENGTH=512
SIM="cos_sim"

python src/retriever/biencoder_inference.py \
    --corpus_path $CORPUS_PATH \
    --queries_path $QUESTION_PATH \
    --output_dir $OUT_PATH \
    --model_name_or_path $MODEL \
    --max_seq_length $MAX_SEQ_LENGTH \
    --sim $SIM \
    --do_eval