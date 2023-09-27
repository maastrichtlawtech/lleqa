import os
import csv
import json
import shutil
import string
import logging
from collections import OrderedDict
from tqdm.autonotebook import tqdm, trange
from typing import Dict, Tuple, Iterable, Callable, Type, List, Set, Union, Optional

import heapq
import numpy as np

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, __MODEL_HUB_ORGANIZATION__, __version__
from sentence_transformers.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import fullname, batch_to_device, cos_sim, dot_score, snapshot_download
from sentence_transformers.model_card_templates import ModelCardTemplate

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

class SentenceTransformerCustom(SentenceTransformer):
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            # New
            log_callback: Callable[[int, int, int, float, float], None] = None,
            log_every_n_steps: int = 0,
        ):
        """Custom .fit() function with an extra generic logging framework parameter.
        Source: https://github.com/UKPLab/sentence-transformers/pull/1606#issuecomment-1383608304
        """
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                    if log_callback is not None and log_every_n_steps > 0 and training_steps % log_every_n_steps == (log_every_n_steps - 1):
                        try:
                            log_callback(train_idx, epoch, global_step, scheduler.get_last_lr()[0], loss_value.item())
                        except Exception as e:
                            logger.warning(f"Logging error encountered: {e}. Ignoring..")

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        if evaluator is None and output_path is not None:
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None, epoch: int = -1, steps: int = -1):
        """Custom .evaluate() function with two extra 'epoch' and 'steps' parameters.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path=output_path, epoch=epoch, steps=steps)


class InformationRetrievalEvaluatorCustom(InformationRetrievalEvaluator):
    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim, 'dot_score': dot_score},
                 main_score_function: str = None,
                 # New
                 log_callback: Callable[[int, int, str, str, float], None] = None,
                 output_value: str = 'sentence_embedding', #[None, 'sentence_embedding', 'token_embeddings']
        ):
        """Custom init function that adds an extra generic logging framework 'log_callback' and an 'output_value' 
        parameter indicating if the model outputs one sentence embedding or all token embeddings.
        """
        super().__init__(
            queries, corpus, relevant_docs, corpus_chunk_size, 
            mrr_at_k, ndcg_at_k, accuracy_at_k, precision_recall_at_k, map_at_k,
            show_progress_bar, batch_size, name, write_csv, score_functions, main_score_function,
        )
        self.log_callback = log_callback
        self.output_value = output_value
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        """Custom call function that logs scores to the provided logging framework and has two extra parameters:
        'epoch' and 'steps'.
        """
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    val = scores[name]['accuracy@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/accuracy@{k}', val)

                for k in self.precision_recall_at_k:
                    p_val = scores[name]['precision@k'][k]
                    r_val = scores[name]['recall@k'][k]
                    output_data.append(p_val)
                    output_data.append(r_val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/precision@{k}', p_val)
                        self.log_callback(epoch, steps, f'{self.name}/recall@{k}', r_val)

                for k in self.mrr_at_k:
                    val = scores[name]['mrr@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/mrr@{k}', val)

                for k in self.ndcg_at_k:
                    val = scores[name]['ndcg@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/ndcg@{k}', val)

                for k in self.map_at_k:
                    val = scores[name]['map@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/map@{k}', val)

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['map@k'][max(self.map_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['map@k'][max(self.map_at_k)]


    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model
        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        query_embeddings = model.encode(self.queries, output_value=self.output_value, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        #Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], output_value=self.output_value, show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores

