#!/bin/bash


# Model.       
MODEL="camembert-base"
MAX_SEQ_LEN=384
POOL="mean"
SIM="cos_sim"

# Optimization.
EPOCHS=20
BATCH_SIZE=32
SCHEDULER="warmuplinear"
LR=2e-5
WARMUP_STEPS=60
OPTIMIZER="AdamW"
WD=0.01
FP16="--use_fp16"

# Actions.
EVAL_BEFORE_TRAINING="--eval_before_training"
EVAL_DURING_TRAINING="--eval_during_training"
LOG_STEPS=1
DO_TEST="--do_test"
DO_SAVE="--do_save"

# Runs.
SEEDS=(42) #(42 43 44 45 46)
for SEED in "${SEEDS[@]}"; do

    python src/retriever/biencoder_training.py \
        --model_name "$MODEL" \
        --max_seq_length $MAX_SEQ_LEN \
        --pooling "$POOL" \
        --sim "$SIM" \
        --epochs $EPOCHS \
        --train_batch_size $BATCH_SIZE \
        --scheduler "$SCHEDULER" \
        --lr $LR \
        --optimizer "$OPTIMIZER" \
        --wd $WD \
        --warmup_steps $WARMUP_STEPS \
        $FP16 \
        --seed $SEED \
        $EVAL_BEFORE_TRAINING \
        $EVAL_DURING_TRAINING \
        --log_steps $LOG_STEPS \
        $DO_SAVE \
        --output_dir "output/training/retriever" \
        $DO_TEST
done