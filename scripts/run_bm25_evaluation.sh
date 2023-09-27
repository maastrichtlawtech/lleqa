#!/bin/bash

# Data paths.
QUERIES_PATH="data/lleqa/questions_dev.json"
CORPUS_PATH="data/lleqa/articles.json"
OUT_PATH="output/testing"

# Optimal BM25 parameters.
K1=2.5
B=0.2

python src/retriever/bm25.py \
    --corpus_path $CORPUS_PATH \
    --queries_path $QUERIES_PATH \
    --output_dir $OUT_PATH \
    --do_preprocessing \
    --k1 $K1 \
    --b $B \
    --do_evaluation
