#!/bin/bash


#------------------------------------------#
#                   DATA                   #
#------------------------------------------#
ARTICLES="data/lleqa/articles.json"
QUESTIONS="data/lleqa/questions_test.json"
NEGATIVES="data/lleqa/negatives/negatives_intfloat_multilingual-e5-large.json"
DEMONSTRATIONS="data/lleqa/questions_train.json"
OUT_DIR="output/fewshot"

#------------------------------------------#
#           EVIDENCE PARAMETERS            #
#------------------------------------------#
NUM_EVIDENCE=5
EVIDENCE_ORDER="most_relevant_first"
EVIDENCE_RETRIEVER="intfloat/multilingual-e5-base" #"maastrichtlawtech/camembert-base-lleqa"

#------------------------------------------#
#      IN-CONTEXT LEARNING PARAMETERS      #
#------------------------------------------#
INSTRUCTION='Please respond to the "QUESTION" based on the provided "CONTEXT". '\
'Your response should be written in French. '\
'Your response should start with an "ANSWER" that gives a comprehensive answer to the "QUESTION" in simple terms. '\
'Your response should end with "RATIONALES" as a list of comma-separated IDs of paragraphs from the "CONTEXT" that support your "ANSWER". '\
'Your response should be formatted in the following schema, including the leading ``` and trailing ```:\n'\
'```\n'\
'ANSWER: <Comprehensive answer to the "QUESTION" in simple terms>\n'\
'RATIONALES: <Comma-separated IDs of paragraphs from the "CONTEXT" that support your "ANSWER">\n'\
'```\n'
NUM_DEMOS=(0 1 2)
DEMO_TYPE="similar"
DEMO_RETRIEVER="dangvantuan/sentence-camembert-base"

#------------------------------------------#
#               LLM CONFIG                 #
#------------------------------------------#
TEMP=0.1
MAX_OUT_TOKENS=350
MODELS=(
    '{"provider":"localhost","temperature":'$TEMP',"max_completion_tokens":'$MAX_OUT_TOKENS',"model_name":"lmsys/vicuna-7b-v1.3"}'
    '{"provider":"localhost","temperature":'$TEMP',"max_completion_tokens":'$MAX_OUT_TOKENS',"model_name":"TheBloke/wizardLM-7B-HF"}'
    '{"provider":"localhost","temperature":'$TEMP',"max_completion_tokens":'$MAX_OUT_TOKENS',"model_name":"TheBloke/tulu-7B-fp16"}'
    '{"provider":"localhost","temperature":'$TEMP',"max_completion_tokens":'$MAX_OUT_TOKENS',"model_name":"TheBloke/guanaco-7B-HF"}'
)

#------------------------------------------#
#                EXPERIMENTS               #
#------------------------------------------#
for N in "${NUM_DEMOS[@]}"
do
    for MODEL in "${MODELS[@]}"
    do
        python src/generator/fewshot.py \
            --system "$MODEL" \
            --articles_filepath "$ARTICLES" \
            --questions_filepath "$QUESTIONS" \
            --negatives_filepath "$NEGATIVES" \
            --demonstration_filepath "$DEMONSTRATIONS" \
            --instruction "$INSTRUCTION" \
            --num_evidence $NUM_EVIDENCE \
            --evidence_order "$EVIDENCE_ORDER" \
            --evidence_retriever_model_name "$EVIDENCE_RETRIEVER" \
            --num_demonstrations $N \
            --demonstration_type "$DEMO_TYPE" \
            --demonstration_selector_model_name "$DEMO_RETRIEVER" \
            --output_dir "$OUT_DIR"
    done
done