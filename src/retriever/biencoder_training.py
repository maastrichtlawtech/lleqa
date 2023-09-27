import os
import sys
import pathlib
import logging
import argparse
from datetime import datetime

import torch
from torch import nn
from torch.optim import AdamW
from torch_optimizer import Adafactor
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

import wandb
import random
import numpy as np
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

from sentence_transformers import util, LoggingHandler
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import MultipleNegativesRankingLoss

try:
    from src.data.lleqa import LLeQADatasetIRLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.lleqa import LLeQADatasetIRLoader
from src.utils.shampoo import Shampoo
from src.utils.loggers import WandbLogger
from src.utils.common import set_seed, count_trainable_parameters
from src.utils.SentenceTransformer import SentenceTransformerCustom, InformationRetrievalEvaluatorCustom


def main(args):
    set_seed(args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    
    # Model.
    if args.use_pre_trained_model:
        logging.info("Using pretrained biencoder model...")
        model = SentenceTransformerCustom(args.model_name)
    else:
        logging.info("Creating new biencoder model...")
        embedding_model = Transformer(args.model_name, max_seq_length=args.max_seq_length, tokenizer_args={'model_max_length': args.max_seq_length})
        pooling_model = Pooling(embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
        model = SentenceTransformerCustom(modules=[embedding_model, pooling_model])
    args.model_params = count_trainable_parameters(model)

    # Loss.
    loss = MultipleNegativesRankingLoss(model=model, similarity_fct=getattr(util, args.sim))

    # Data.
    args.dataset = "lleqa"
    data = LLeQADatasetIRLoader(
        stage='fit',
        corpus_path_or_url="data/lleqa/articles.json",
        train_path_or_url="data/lleqa/questions_train.json",
        dev_path_or_url="data/lleqa/questions_dev.json",
        test_path_or_url="data/lleqa/questions_test.json",
        negatives_path_or_url="data/lleqa/negatives/negatives_intfloat_multilingual-e5-large.json",
    ).run()
    train_dataloader = DataLoader(data['train'], shuffle=True, batch_size=args.train_batch_size)

    # Logger.
    new_model_name = f'{args.model_name.replace("/", "-")}-{args.dataset}'
    out_path = os.path.join(args.output_dir, args.dataset, f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{new_model_name}')
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    logger = WandbLogger(
        project_name=args.dataset,
        run_name=new_model_name, 
        run_config=args, 
        log_dir=os.path.join(args.output_dir, 'logs'),
    )

    # Dev set evaluator.
    dev_evaluator = InformationRetrievalEvaluatorCustom(
        name=f'{args.dataset}_dev', queries=data['dev_queries'], relevant_docs=data['dev_labels'], corpus=data['corpus'],
        precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
        score_functions={args.sim: getattr(util, args.sim)},
        log_callback=logger.log_eval, show_progress_bar=True,
        corpus_chunk_size=50000, batch_size=64,
    )
    train_evaluator, eval_steps = (dev_evaluator, len(train_dataloader)) if args.eval_during_training else (None, 0)

    # Before training.
    if args.eval_before_training:
        model.evaluate(evaluator=dev_evaluator, output_path=out_path, epoch=0, steps=0)

    # Training.
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        use_amp=args.use_fp16,
        scheduler=args.scheduler, warmup_steps=args.warmup_steps,
        optimizer_class=getattr(sys.modules[__name__], args.optimizer), optimizer_params={"lr": args.lr}, weight_decay=args.wd, 
        log_every_n_steps=args.log_steps, log_callback=logger.log_training,
        evaluator=train_evaluator, evaluation_steps=eval_steps, output_path=out_path,
        checkpoint_path=out_path if args.do_save else None, checkpoint_save_steps=len(train_dataloader), checkpoint_save_total_limit=3,
        show_progress_bar=True,
    )

    # After training.
    if args.do_save:
        model.save(f"{out_path}/final")
    
    if args.eval_after_training:
        model.evaluate(evaluator=dev_evaluator, output_path=out_path, epoch=args.epochs, steps=args.epochs*len(train_dataloader))

    if args.do_test:
        test_evaluator = InformationRetrievalEvaluatorCustom(
            name=f'{args.dataset}_test', queries=data['test_queries'], relevant_docs=data['test_labels'], corpus=data['corpus'],
            precision_recall_at_k=[1, 5, 10, 20, 50, 100, 200, 500], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100],
            score_functions={args.sim: getattr(util, args.sim)},
            log_callback=logger.log_eval, show_progress_bar=True,
            corpus_chunk_size=50000, batch_size=64,
        )
        model.evaluate(evaluator=test_evaluator, output_path=out_path, epoch=args.epochs, steps=args.epochs*len(train_dataloader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model.
    parser.add_argument("--use_pre_trained_model", action="store_true", default=False, help="Whether to load pretrained biencoder or create new model.")
    parser.add_argument("--model_name", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--pooling", type=str, choices=["mean", "max", "cls"], help="Type of pooling to perform to get a passage representation.")
    parser.add_argument("--sim", type=str, choices=["cos_sim", "dot_score"], help="Similarity function for scoring query-document representation.")
    # Training.
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adafactor", "Shampoo"], help="Type of optimizer to use for training.")
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--wd", type=float, help="The weight decay to apply (if not zero) to all layers in AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, choices=["constantlr", "warmupconstant", "warmuplinear", "warmupcosine", "warmupcosinewithhardrestarts"], help="Type of learning rate scheduler to use for training.")
    parser.add_argument("--warmup_steps", type=int, help="Number of training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--use_fp16", action="store_true", default=False, help="Whether to use mixed precision during training.")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--do_save", action="store_true", default=False, help="Wether to save model checkpoints.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Evaluation.
    parser.add_argument("--log_steps", type=int, help="Log every k training steps.")
    parser.add_argument("--eval_before_training", action="store_true", default=False, help="Wether to perform dev evaluation before training.")
    parser.add_argument("--eval_during_training", action="store_true", default=False, help="Wether to perform dev evaluation before training.")
    parser.add_argument("--eval_after_training", action="store_true", default=False, help="Wether to perform dev evaluation before training.")
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform test evaluation after training.")
    args, _ = parser.parse_known_args()
    main(args)
