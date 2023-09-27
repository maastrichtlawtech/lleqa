#!/bin/bash

#------------------------------------------#
#          ACTIONS TO PERFORM              #
#------------------------------------------#
DO_TRAIN="--do_train"
DO_VAL="" #"--do_val"
DO_TEST="--do_test"

#------------------------------------------#
#                   DATA                   #
#------------------------------------------#
TRAIN_QUESTIONS="data/lleqa/questions_train.json"
DEV_QUESTIONS="data/lleqa/questions_dev.json"
TEST_QUESTIONS="data/lleqa/questions_test.json"
CORPUS="data/lleqa/articles.json"
NEGATIVES="data/lleqa/negatives/negatives_intfloat_multilingual-e5-large.json"
OUT_DIR="output/training/generator"

#------------------------------------------#
#               GENERATOR                  #
#------------------------------------------#
MAX_CONTEXT_LEN=4096
MODELS=(
    "lmsys/vicuna-7b-v1.3"
    "TheBloke/wizardLM-7B-HF"
    "TheBloke/tulu-7B-fp16"
    "TheBloke/guanaco-7B-HF"
)

#------------------------------------------#
#                TRAINING                  #
#------------------------------------------#
EPOCHS=10
BS=1
ACC_STEPS=8
LR=2e-4
SCHEDULER="constant"
DEEPSPEED="" #"src/config/ds_zero3_offload.json"

#------------------------------------------#
#                TESTING                   #
#------------------------------------------#
TEMP=0.1
DECODING="nucleus-sampling"
EVIDENCE_RETRIEVER="intfloat/multilingual-e5-base" #"maastrichtlawtech/camembert-base-lleqa"

for MODEL in "${MODELS[@]}"
do
    if [[ $MODEL == *"t5"* ]]; then
        LR=3e-3
    fi

    ARGS=(
        --corpus_filepath "$CORPUS"
        --negatives_filepath "$NEGATIVES"
        --output_dir "$OUT_DIR"
        --model_name_or_path "$MODEL"
        --max_context_len $MAX_CONTEXT_LEN
        $DO_TRAIN
        --train_questions_filepath "$TRAIN_QUESTIONS"
        --num_epochs $EPOCHS
        --batch_size $BS
        --gradient_accumulation_steps $ACC_STEPS
        --scheduler "$SCHEDULER"
        --lr $LR
        --optimizer "adamw"
        --gradient_checkpointing
        $DO_VAL
        --dev_questions_filepath "$DEV_QUESTIONS"
        $DO_TEST
        --test_questions_filepath "$TEST_QUESTIONS"
        --decoding_strategy "$DECODING"
        --temperature $TEMP
        --evidence_retriever "$EVIDENCE_RETRIEVER"
    )

    RUN=("python")
    if [[ -n "$DEEPSPEED" ]]; then
        RUN=("deepspeed" "--num_gpus=1")
        ARGS=("--deepspeed" "$DEEPSPEED" "${ARGS[@]}")
    fi

    CMD=("${RUN[@]}" "src/generator/finetuning.py")
    "${CMD[@]}" "${ARGS[@]}"
done