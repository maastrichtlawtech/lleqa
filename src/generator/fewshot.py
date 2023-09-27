import os
import re
import sys
import json
import logging
import pathlib
import argparse
from tqdm import tqdm
from typing import List, Type
from collections import defaultdict
from os.path import join, splitext, basename, dirname

import evaluate
import numpy as np
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

try:
    from src.utils.loggers import LoggingHandler
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.loggers import LoggingHandler
from src.generator.inference import LLMTaskSolver
from src.utils.metrics import compute_precision_recall_f1
from src.data.lleqa import LLeQADatasetQALoader, AnswerOutputParser
    

def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    logging.info("Loading LLM model...")
    parser = AnswerOutputParser()
    model = LLMTaskSolver(llm_config=args.system, prompt="{input}", output_parser=parser)
    
    logging.info("Preparing dataset...")
    dataset = LLeQADatasetQALoader(
        questions_filepath=args.questions_filepath,
        articles_filepath=args.articles_filepath, 
        negatives_filepath=args.negatives_filepath,
        out_format="raw",
        num_evidence=args.num_evidence,
        evidence_retriever_model_name=args.evidence_retriever_model_name,
        evidence_order=args.evidence_order,
        instruction=args.instruction,
        num_demonstrations=args.num_demonstrations,
        demonstration_filepath=args.demonstration_filepath,
        demonstration_type=args.demonstration_type,
        text_similarity_model_name=args.demonstration_selector_model_name,
    ).run()

    logging.info("Running inference...") 
    results = []
    ignored = 0
    for sample in tqdm(dataset, desc='Samples'):
        response = model(input=sample['source'])
        if response:
            pred = response['output']
            gold = parser.parse(sample['target'])
            results.append({
                'id': sample['id'],
                'pred_answer': LLeQADatasetQALoader.postprocess(pred['full'])[0],
                'gold_answer': LLeQADatasetQALoader.postprocess(gold['full'])[0],
                'pred_rationales': pred['parsed_rationales'],
                'gold_rationales': gold['parsed_rationales'],
            })
        else:
            ignored += 1
    if ignored > 0:
        logging.info(f"!! NB !! {ignored} samples were ignored due to their length exceeding the maximum input length of {args.system.get('model_name')}.")

    logging.info("Computing scores...")
    scores = [compute_precision_recall_f1(gold=r['gold_rationales'], predicted=r['pred_rationales']) for r in results]
    scores = {f"rationales_{key}": np.mean(list(map(lambda x: x[key], scores))) for key in scores[0].keys()}
    scores.update(rouge.compute(predictions=list(map(lambda x: x['pred_answer'], results)), references=list(map(lambda x: x['gold_answer'], results)), use_stemmer=True))
    scores.update(meteor.compute(predictions=list(map(lambda x: x['pred_answer'], results)), references=list(map(lambda x: x['gold_answer'], results))))

    logging.info("Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    retriever = 'biencoder' if args.evidence_retriever_model_name not in ['oracle', 'random'] else args.evidence_retriever_model_name
    run_name = (
        f"{args.system['model_name'].replace('/', '-')}_"
        f"k={args.num_evidence}_{args.evidence_order}_{retriever}_"
        f"n={args.num_demonstrations}_{args.demonstration_type}_"
        f"temp={args.system['temperature']}"
    )
    with open(join(args.output_dir, f"scores_{run_name}.json"), "w") as f:
        json.dump(scores, f, indent=2)
    with open(join(args.output_dir, f"results_{run_name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=json.loads, help="The annotator system config.")
    parser.add_argument("--articles_filepath", type=str, help="The articles data file path.")
    parser.add_argument("--negatives_filepath", type=str, help="The negatives data file path.")
    parser.add_argument("--questions_filepath", type=str, help="The questions data file path.")
    parser.add_argument("--instruction", type=str, help="The instruction for the task.")
    parser.add_argument("--num_evidence", type=int, help="The number of evidence articles to use per sample.")
    parser.add_argument("--evidence_retriever_model_name", type=str, help="The model to use for retrieving evidence articles. Either 'oracle' or a HF model id.")
    parser.add_argument("--evidence_order", type=str, choices=['most_relevant_first', 'most_relevant_last', 'random'], help="The order of the evidence articles.")
    parser.add_argument("--num_demonstrations", type=int, help="The number of demonstrations to use per sample.")
    parser.add_argument("--demonstration_filepath", type=str, help="The demonstration data file path.")
    parser.add_argument("--demonstration_type", type=str, choices=["random", "similar"], help="The demonstration type.")
    parser.add_argument("--demonstration_selector_model_name", type=str, help="The HF model id to use for retrieving similar in-context examples.")
    parser.add_argument("--output_dir", type=str, help="The output directory.")
    args, _ = parser.parse_known_args()
    main(args)
