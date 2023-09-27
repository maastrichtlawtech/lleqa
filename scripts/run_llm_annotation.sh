#!/bin/bash

EVAL="--do_evaluation"

if [ "$EVAL" == "--do_evaluation" ]; then
    QUESTIONS="data/lleqa/questions_test.json"
else
    QUESTIONS="data/lleqa/questions_train.json"
fi
ARTICLES="data/lleqa/articles.json"
OUT_DIR="output/annotation"

MODELS=(
    #---------------------------------------------------#
    #                BASELINES                          #
    #---------------------------------------------------#
    '{"provider":"localhost","model_name":"all"}'
    '{"provider":"localhost","model_name":"first"}'
    '{"provider":"localhost","model_name":"last"}'
    '{"provider":"localhost","model_name":"random"}'
    #---------------------------------------------------#
    #                PROPRIETARY LLMs                   #
    #---------------------------------------------------#
    '{"provider":"openai","temperature":0.0,"max_completion_tokens":100,"model_name":"gpt-3.5-turbo-16k-0613"}'
    '{"provider":"openai","temperature":0.0,"max_completion_tokens":100,"model_name":"gpt-4-0613"}'
    '{"provider":"anthropic","temperature":0.0,"max_completion_tokens":100"model_name":"claude-2.0"}'
    '{"provider":"anthropic","temperature":0.0,"max_completion_tokens":100"model_name":"claude-instant-1.1"}'
    '{"provider":"ai21","temperature":0.0,"max_completion_tokens":100"model_name":"j2-mid"}'
    '{"provider":"ai21","temperature":0.0,"max_completion_tokens":100"model_name":"j2-ultra"}'
    '{"provider":"cohere","temperature":0.0,"max_completion_tokens":100"model_name":"command-light"}'
    '{"provider":"cohere","temperature":0.0,"max_completion_tokens":100"model_name":"command"}'
    #---------------------------------------------------#
    #               OPEN-SOURCE LLMs                    #
    #---------------------------------------------------#
    '{"provider":"localhost","temperature":0.0,"max_completion_tokens":100,"model_name":"lmsys/vicuna-33b-v1.3"}'
    '{"provider":"localhost","temperature":0.0,"max_completion_tokens":100,"model_name":"TheBloke/WizardLM-30B-fp16"}'
    '{"provider":"localhost","temperature":0.0,"max_completion_tokens":100,"model_name":"timdettmers/guanaco-33b-merged"}'
    '{"provider":"localhost","temperature":0.0,"max_completion_tokens":100,"model_name":"TheBloke/tulu-30B-fp16"}'
)

for MODEL in "${MODELS[@]}"
do
    python src/data/annotation.py \
        --system "$MODEL" \
        $EVAL \
        --questions_filepath "$QUESTIONS" \
        --articles_filepath "$ARTICLES" \
        --output_dir "$OUT_DIR"

    sleep 2
done