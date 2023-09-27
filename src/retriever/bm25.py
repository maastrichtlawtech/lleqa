import os
import sys
import json
import pathlib
import argparse
import itertools
from tqdm import tqdm
from os.path import join
from typing import List, Type

import math
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean

try:
    from src.utils.common import log_step
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import log_step
from src.utils.metrics import Metrics
from src.data.text_processor import TextPreprocessor



class TFIDFRetriever:
    def __init__(self, retrieval_corpus):
        self.retrieval_corpus = retrieval_corpus
        self.N = len(retrieval_corpus)
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id,_ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        results = dict()
        for i, doc in enumerate(self.retrieval_corpus):
            results[i+1] = self.score(q, doc) #NB: '+1' because doc_ids in BSARD start at 1.
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def score(self, q, d):
        score = 0.0
        for t in q.split():
            score += self._compute_tfidf(t, d)
        return score

    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable([doc.lower().split() for doc in self.retrieval_corpus])))

    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word,_ in idfs.items():
            idfs[word] = self._compute_idf(word)
        return idfs

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10(self.N / (df + 1))

    def _compute_tf(self, t, d):
        return d.split().count(t)

    def _compute_tfidf(self, t, d):
        tf = self._compute_tf(t, d)
        idf = self.idfs[t] if t in self.idfs else math.log10(self.N)
        return tf * idf


class BM25Retriever(TFIDFRetriever):
    def __init__(self, retrieval_corpus, k1, b):
        super().__init__(retrieval_corpus)
        self.k1 = k1
        self.b = b
        self.avgdl = self._compute_avgdl()

    def update_params(self, k1, b):
        self.k1 = k1
        self.b = b
    
    def score(self, q, d):
        score = 0.0
        for t in q.split():
            tf = self._compute_tf(t, d)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5)/0.5)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(d.split())/self.avgdl))
        return score

    def _compute_avgdl(self):
        return mean([len(doc.split()) for doc in self.retrieval_corpus])

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10((self.N - df + 0.5) / (df + 0.5))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading documents and queries...")
    dfC = pd.read_json(args.corpus_path, orient='records')
    dfQ = pd.read_json(args.queries_path, orient='records')
    documents = dfC['article'].tolist()
    queries = dfQ['question'].tolist()
    
    if args.do_preprocessing:
        print("Preprocessing documents and queries (lemmatizing=True)...")
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        documents = cleaner.preprocess(documents, lemmatize=True)
        queries = cleaner.preprocess(queries, lemmatize=True)

    if args.do_hyperparameter_tuning:
        print("Starting hyperparameter tuning...")
        # Init evaluator and BM25 retriever module.
        evaluator = Metrics(recall_at_k=[100, 200, 500])
        retriever = BM25Retriever(retrieval_corpus=documents, k1=0., b=0.)

        # Create dataframe to store results.
        hyperparameters = ['k1', 'b']
        metrics = [f"recall@{k}" for k in evaluator.recall_at_k]
        grid_df = pd.DataFrame(columns=hyperparameters+metrics)

        # Create all possible combinations of hyperparamaters.
        k1_range = np.arange(0., 8.5, 0.5)
        b_range = np.arange(0., 1.1, 0.1)
        combinations = list(itertools.product(*[k1_range, b_range]))

        # Launch grid search runs.
        for i, (k1, b) in enumerate(combinations):
            print(f"\n\n({i+1}) Model: BM25 - k1={k1}, b={b}")
            retriever.update_params(k1, b)
            retrieved_docs = retriever.search_all(queries, top_k=500)
            scores = evaluator.compute_all_metrics(all_ground_truths=dfQ['article_ids'].tolist(), all_results=retrieved_docs)
            scores.update({**{'k1':k1, 'b':b}, **{f"{metric}@{k}": v for metric, results in scores.items() if isinstance(results, dict) for k,v in results.items()}})
            scores.pop('recall')
            grid_df = grid_df.append(scores, ignore_index=True)
            grid_df.to_csv(join(args.output_dir, 'bm25_tuning_results.csv'), sep=',', float_format='%.5f', index=False)
        
        # Plot heatmap.
        grid_df = grid_df.pivot_table(values='recall@100', index='k1', columns='b')[::-1] *100
        plot = sns.heatmap(grid_df, annot=True, cmap="YlOrBr", fmt='.1f', cbar=False, vmin=40, vmax=60)
        plot.get_figure().savefig(join(args.output_dir, "bm25_tuning_heatmap.pdf"))

    else:
        print("Initializing the BM25 retriever model...")
        retriever = BM25Retriever(retrieval_corpus=documents, k1=args.k1, b=args.b)

        print("Running BM25 model on queries...")
        retrieved_docs = retriever.search_all(queries, top_k=500)

        if args.do_evaluation:
            print("Computing the retrieval scores...")
            evaluator = Metrics(recall_at_k=[5,10,20,50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100])
            scores = evaluator.compute_all_metrics(all_ground_truths=dfQ['article_ids'].tolist(), all_results=retrieved_docs)
            with open(join(args.output_dir, f'scores_bm25.json'), 'w') as f:
                json.dump(scores, f, indent=2)

        if args.do_negatives_extraction:
            print(f"Extracting top-{args.num_negatives} negatives for each question...")
            results = dict()
            for q_id, truths_i, preds_i in zip(dfQ['id'].tolist(), dfQ['article_ids'].tolist(), retrieved_docs):
                results[q_id] = [y for y in preds_i if y not in truths_i][:args.num_negatives]
            results = dict(sorted(results.items()))
            with open(join(args.output_dir, f'negatives_bm25.json'), 'w') as f:
                json.dump(results, f, indent=2)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, help="Path of the data file containing the corpus of documents.")
    parser.add_argument("--queries_path", type=str, help="Path of the data file containing the queries.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    parser.add_argument("--do_preprocessing", action='store_true', default=False, help="Whether or not to pre-process the articles (lowercasing, lemmatization, and deletion of stopwords, punctuation, and numbers).")
    parser.add_argument("--k1", type=float, default=1.0, help="BM25 parameter.")
    parser.add_argument("--b", type=float, default=0.6, help="BM25 parameter.")
    parser.add_argument("--do_evaluation", action='store_true', default=False, help="Whether or not to perform evaluation.")
    parser.add_argument("--do_negatives_extraction", action='store_true', default=False, help="Whether or not to extract negatives.")
    parser.add_argument("--num_negatives", type=int, default=10, help="Number of negatives to extract per question.")
    parser.add_argument("--do_hyperparameter_tuning", action='store_true', default=False, help="Whether or not to perform hyperparameter tuning.")
    args, _ = parser.parse_known_args()
    main(args)
