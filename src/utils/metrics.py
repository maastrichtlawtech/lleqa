import numpy as np
from statistics import mean
from typing import List, Optional
from collections import defaultdict


def compute_precision_recall_f1(gold, predicted):
    if predicted is None:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    tp = len(set(gold) & set(predicted))
    fp = len(predicted) - tp
    fn = len(gold) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}


class Metrics:
    def __init__(self, recall_at_k: List[int], map_at_k: List[int] = [], mrr_at_k: List[int] = [], ndcg_at_k: List[int] = []):
        self.recall_at_k = recall_at_k
        self.map_at_k = map_at_k
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k

    def compute_all_metrics(self, all_ground_truths, all_results):
        scores = defaultdict(dict)
        for k in self.recall_at_k:
            scores['recall'][k] = self.compute_mean_score(self.recall, all_ground_truths, all_results, k)
        for k in self.map_at_k:
            scores['map'][k] = self.compute_mean_score(self.average_precision, all_ground_truths, all_results, k)
        for k in self.mrr_at_k:
            scores['mrr'][k] = self.compute_mean_score(self.reciprocal_rank, all_ground_truths, all_results, k)
        for k in self.ndcg_at_k:
            scores['ndcg'][k] = self.compute_mean_score(self.ndcg, all_ground_truths, all_results, k)
        scores['r-precision'] = self.compute_mean_score(self.r_precision, all_ground_truths, all_results)
        return scores

    def compute_mean_score(self, score_func, all_ground_truths: List[List[int]], all_results: List[List[int]],  k: int = None):
        return mean([score_func(truths, res, k) for truths, res in zip(all_ground_truths, all_results)])

    def average_precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        p_at_k = [self.precision(ground_truths, results, k=i+1) if d in ground_truths else 0 for i, d in enumerate(results[:k])]
        return sum(p_at_k)/len(ground_truths)

    def reciprocal_rank(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        return max([1/(i+1) if d in ground_truths else 0.0 for i, d in enumerate(results[:k])])

    def ndcg(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        dcg = relevances[0] + sum(relevances[i] / np.log2(i + 1) for i in range(1, len(relevances)))
        idcg = 1 + sum(1 / np.log2(i + 1) for i in range(1, len(ground_truths)))
        return (dcg / idcg) if idcg != 0 else 0

    def r_precision(self, ground_truths: List[int], results: List[int], R: int = None):
        R = len(ground_truths)
        relevances = [1 if d in ground_truths else 0 for d in results[:R]]
        return sum(relevances)/R

    def recall(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(ground_truths)

    def precision(self, ground_truths: List[int], results: List[int], k: int = None):
        k = len(results) if k is None else k
        relevances = [1 if d in ground_truths else 0 for d in results[:k]]
        return sum(relevances)/len(results[:k])

    def fscore(self, ground_truths: List[int], results: List[int], k: int = None):
        p = self.precision(ground_truths, results, k)
        r = self.recall(ground_truths, results, k)
        return (2*p*r)/(p+r) if (p != 0.0 or r != 0.0) else 0.0
