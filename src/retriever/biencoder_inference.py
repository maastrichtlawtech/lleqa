import os
import sys
import json
import pathlib
import argparse
from os.path import join, basename, splitext

import torch
import pandas as pd
from sentence_transformers import util

try:
    from src.data.lleqa import LLeQADatasetIRLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.lleqa import LLeQADatasetIRLoader
from src.utils.SentenceTransformer import SentenceTransformerCustom, InformationRetrievalEvaluatorCustom


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SentenceTransformer model from {args.model_name_or_path}...")
    model = SentenceTransformerCustom(args.model_name_or_path)
    model.max_seq_length = args.max_seq_length

    if args.do_eval:
        print("Loading documents and queries...")
        data = LLeQADatasetIRLoader(
            stage='eval',
            corpus_path_or_url="data/lleqa/articles.json",
            test_path_or_url="data/lleqa/questions_test.json",
        ).run()

        print("Evaluating...")
        test_evaluator = InformationRetrievalEvaluatorCustom(
            name=f'lleqa_test', queries=data['test_queries'], relevant_docs=data['test_labels'], corpus=data['corpus'],
            precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100],
            score_functions={args.sim: getattr(util, args.sim)},
            show_progress_bar=True,
            corpus_chunk_size=50000, batch_size=64,
        )
        model.evaluate(evaluator=test_evaluator, output_path=args.output_dir)
        print("Done.")
    else:
        print("Loading documents and queries...")
        dfC = pd.read_json(args.corpus_path, orient='records')
        dfQ = pd.read_json(args.queries_path, orient='records')

        print("Encoding queries...")
        q_embeddings = model.encode(
            sentences=dfQ['question'].tolist(),
            batch_size=32,
            output_value='sentence_embedding',
            convert_to_tensor=True,
            normalize_embeddings=False,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            show_progress_bar=True,
        )

        print("Encoding documents...")
        d_embeddings = model.encode(
            sentences=dfC['article'].tolist(),
            batch_size=32,
            output_value='sentence_embedding',
            convert_to_tensor=True,
            normalize_embeddings=False,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            show_progress_bar=True,
        )

        print("Computing cosine similarity scores...")
        retrieved_docs = util.semantic_search( # Returns a list with one entry for each query where each entry is a list of dictionaries {‘corpus_id’: x, ‘score’: y}, sorted by decreasing similarity scores.
            query_embeddings=q_embeddings, 
            corpus_embeddings=d_embeddings,
            top_k=500,
            score_function=getattr(util, args.sim),
        )
        # Extract the doc_id only (NB: +1 because article_id start at index 1 while semantic_search() returns indices from the given document list).
        retrieved_docs = [[res['corpus_id']+1 for res in results] for results in retrieved_docs]

        if args.do_negatives_extraction:
            print(f"Extracting top-{args.num_negatives} negatives for each question...")
            results = dict()
            for q_id, truths_i, preds_i in zip(dfQ['id'].tolist(), dfQ['article_ids'].tolist(), retrieved_docs):
                results[q_id] = [y for y in preds_i if y not in truths_i][:args.num_negatives]
            results = dict(sorted(results.items()))
            with open(join(args.output_dir, f"negatives_{args.model_name_or_path.replace('/', '_')}.json"), 'w') as f:
                json.dump(results, f, indent=2)
    
        print("Done.")
        return retrieved_docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, help="Path of the data file containing the corpus of documents.")
    parser.add_argument("--queries_path", type=str, help="Path of the data file containing the queries.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    parser.add_argument("--model_name_or_path", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--sim", type=str, choices=["cos_sim", "dot_score"], help="Similarity function for scoring query-document representation.")
    parser.add_argument("--do_eval", action='store_true', default=False, help="Whether we perform a model evaluation.")
    parser.add_argument("--do_negatives_extraction", action='store_true', default=False, help="Whether or not to extract negatives.")
    parser.add_argument("--num_negatives", type=int, default=10, help="Number of negatives to extract per question.")
    args, _ = parser.parse_known_args()
    main(args)
