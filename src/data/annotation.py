import os
import re
import sys
import json
import time
import random
import logging
import pathlib
import argparse
from typing import List, Type
from more_itertools import sliced
from os.path import join, splitext, basename, dirname

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from langchain.schema import BaseOutputParser
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

try:
    from src.generator.inference import LLMTaskSolver
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.generator.inference import LLMTaskSolver
    from src.utils.loggers import LoggingHandler
    from src.utils.metrics import compute_precision_recall_f1


class DataPreparator:
    def __init__(self, questions_filepath: str, articles_filepath: str, comparison_w_gold: bool):
        self.dfA = pd.read_json(articles_filepath, orient='records')
        self.dfQ = pd.read_json(questions_filepath, orient='records')
        self.comparison_w_gold = comparison_w_gold

    def preprocess(self):
        if self.comparison_w_gold:
            return (self.dfQ
                .pipe(self._split_paragraph_ids)
                .pipe(self._drop_questions_w_paragraphs_not_in_corpus, self.dfA)
                .pipe(self._keep_only_necessary_data)
                .pipe(self._group_paragraph_ids)
                .pipe(self._add_article_paragraphs, self.dfA)
                .pipe(self._drop_and_reorder_columns)
            )
        else:
            return (self.dfQ
                .pipe(self._split_article_ids)
                .pipe(self._add_article_paragraphs, self.dfA)
                .pipe(self._drop_and_reorder_columns)
            )

    def postprocess(self, annotated_df, system_to_keep: str):
        return (annotated_df
            .pipe(self._select_annotations_from, system_to_keep)
            .pipe(self._group_predictions)
            .pipe(self._add_predictions_to_original_dataset)
            .pipe(self._put_predictions_column_last)
        )

    def _split_paragraph_ids(self, dfQ):
        dfQ = dfQ.explode('gold_paragraph_ids')
        dfQ[['article_id', 'paragraph_id']] = dfQ['gold_paragraph_ids'].str.split('§', expand=True)
        dfQ[['article_id', 'paragraph_id']] = dfQ[['article_id', 'paragraph_id']].astype(int)
        return dfQ

    def _split_article_ids(self, dfQ):
        dfQ = dfQ.explode('article_ids')
        dfQ = dfQ.rename(columns={'article_ids': 'article_id'})
        return dfQ

    def _drop_questions_w_paragraphs_not_in_corpus(self, dfQ, dfA):
        def check(row, paragraphs_per_article):
            return paragraphs_per_article.get(row['article_id'], 0) >= row['paragraph_id']
        paragraphs_per_article = dfA.set_index('id')['paragraphs'].apply(len).to_dict()
        result = dfQ.apply(lambda row: check(row, paragraphs_per_article), axis=1)
        return dfQ[result == True]

    def _keep_only_necessary_data(self, dfQ):
        dfQ = dfQ[['id', 'question', 'answer', 'article_id', 'paragraph_id']]
        dfQ = dfQ.sort_values(by=['question', 'article_id', 'paragraph_id'])
        dfQ = dfQ.reset_index(drop=True)
        return dfQ

    def _group_paragraph_ids(self, dfQ):
        dfQ = dfQ.groupby(dfQ.columns.difference(['paragraph_id']).tolist(), as_index=False).agg({'paragraph_id': list})
        dfQ = dfQ.rename(columns={'paragraph_id': 'gold_paragraph_ids'})
        return dfQ

    def _add_article_paragraphs(self, dfQ, dfA):
        dfQ['article_paragraphs'] = dfQ.apply(lambda row: dfA[dfA['id'] == row['article_id']]['paragraphs'].iloc[0], axis=1)
        return dfQ

    def _drop_and_reorder_columns(self, dfQ):
        cols = ['id', 'question', 'answer', 'article_id', 'article_paragraphs']
        if self.comparison_w_gold:
            cols.append('gold_paragraph_ids')
        return dfQ[cols]

    def _select_annotations_from(self, dataf, selected_system: str):
        def process_row(row):
            preds = row['predicted_paragraph_ids']
            if pd.isna(preds) or preds.get(selected_system) is None:
                return np.nan
            output = preds.get(selected_system).get('parsed_output', [])
            if not output:
                return np.nan
            return list(map(lambda x: str(row['article_id']) + '§' + str(x), output))
        dataf['predicted_paragraph_ids'] = dataf.apply(process_row, axis=1)
        return dataf

    def _group_predictions(self, dataf):
        dataf['predicted_paragraph_ids'] = dataf['predicted_paragraph_ids'].fillna({i: [] for i in dataf.index})
        return dataf.groupby('id').agg({'predicted_paragraph_ids': 'sum'}).reset_index()

    def _add_predictions_to_original_dataset(self, dataf):
        self.dfQ['order'] = range(self.dfQ.shape[0])
        return pd.merge(self.dfQ, dataf, on='id', how='outer').sort_values('order').drop(columns=['order'])

    def _put_predictions_column_last(self, dataf):
        data = dataf.pop('predicted_paragraph_ids')
        dataf['predicted_paragraph_ids'] = data
        return dataf


class DataAnnotator:
    class ParagraphMarkerOutputParser(BaseOutputParser):
        def parse(self, text: str):
            return {
                'full_output': text,
                'parsed_output': list(set(map(int, re.findall(r'\[P(\d+)\]', text)))) if text else [],
            }
    
    def __init__(self, system_config: dict, dataset: Type[pd.DataFrame]):
        self.df = dataset
        self.model_name = system_config['model_name'].replace('/', '-')
        if self.model_name not in ['all', 'first', 'last', 'random']:
            self.model = LLMTaskSolver(
                llm_config=system_config,
                prompt=(
                    "Given a QUESTION and its ANSWER, identify the PARAGRAPH(S) that was/were used to create some part of the ANSWER. "
                    "Your RESPONSE should be a list of comma separated paragraph identifiers only, eg: `[P2], [P4]`. " 
                    "If you think that none of the provided PARAGRAPHS were used to make part of the ANSWER, your RESPONSE should be `None`.\n\n"
                    "QUESTION: {question}\n"
                    "ANSWER: {answer}\n"
                    "PARAGRAPHS:\n{paragraphs}\n"
                    "RESPONSE: The paragraph(s) used to create the answer is/are:"
                ),
                output_parser=self.ParagraphMarkerOutputParser(),
            )
    
    def run(self, output_path: str, chunk_size: int = 10, skip_unique_paragraph: bool = False, override_prev: bool = False, save_while_running: bool = False):
        if self.model_name in ['all', 'random', 'first', 'last']:
            data = self.df.copy()
            data['predicted_paragraph_ids'] = data.apply(self.run_baseline, axis=1)
        else:
            data = pd.DataFrame()
            slices = sliced(seq=range(len(self.df)), n=chunk_size)
            for index in tqdm(slices, total=math.ceil(len(self.df)/chunk_size), desc="Batches"):
                chunk = self.df.iloc[index].copy()
                chunk['predicted_paragraph_ids'] = chunk.apply(lambda row: self.get_predictions(row, override_prev, skip_unique_paragraph), axis=1)
                data = pd.concat([data, chunk], axis=0, ignore_index=True, sort=False)
                if save_while_running:
                    (self.df
                        .merge(data, on=list(self.df.columns.difference(['article_paragraphs', 'predicted_paragraph_ids'])), how='outer')
                        .rename(columns=lambda x: x.replace('_x', ''))
                        .pipe(lambda x: x.drop(columns=[c for c in x.columns if c.endswith('_y')]))
                        .to_json(output_path, orient='records', force_ascii=False, indent=2)
                    )
        return data

    def run_baseline(self, row: Type[pd.Series]):
        baseline_type = self.model_name
        preds = row.get('predicted_paragraph_ids', {})
        num_paragraphs = len(row['article_paragraphs'])
        options = list(range(1, num_paragraphs+1))
        if baseline_type == "all":
            y = options
        elif baseline_type == "first":
            y = [options[0]]
        elif baseline_type == "last":
            y = [options[-1]]
        elif baseline_type == "random":
            random.seed(1)
            x = random.randint(0, num_paragraphs)
            y = random.sample(options, x)
        else:
            raise ValueError("Unkown baseline type. Please choose between 'all', 'first', 'last' and 'random'.")
        preds.update({baseline_type: {'parsed_output': y, 'full_output': '', 'estimated_cost': float('nan')}})
        return preds

    def get_predictions(self, row: Type[pd.Series], override_prev: bool, skip_unique_paragraph: bool):
        preds = row.to_dict().get('predicted_paragraph_ids', {})
        preds = {} if pd.isna(preds) else preds

        # If the system has already annotated sample, skip annotation.
        if not override_prev and preds.get(self.model_name):
            return preds

        # If article only has one paragraph, mark it as the relevant one to save an API call.
        if skip_unique_paragraph and len(row['article_paragraphs']) == 1:
            preds.update({self.model_name: {'parsed_output': [1], 'full_output': '', 'estimated_cost': float('nan')}})
            return preds

        # Handle limited number of requests per minute (RPM) or tokens per minute (TPM).
        cohere_cond = self.model.config['provider'] == 'cohere' and row.name % 4 == 0 #Free-tier allows 5 RPM.
        openai_cond = self.model.config['provider'] == 'openai' and "gpt-4" in self.model.config['model_name'] #GPT-4 allows 10K TPM.
        if cohere_cond or openai_cond:
            time.sleep(61)

        # Make API call.
        response = self.model(
            question=row['question'],
            answer=row['answer'],
            paragraphs=''.join([f'[P{k}] ' + re.sub(r'^§\s*\d+(er)?\.\s*', '', v) + ".\n" for k, v in row['article_paragraphs'].items()])
        )

        # Update dataframe with resulting output.
        if response is None:
            preds.update({self.model_name: float('nan')})
        else:
            preds.update({self.model_name: {**response['output'], 'estimated_cost': response['cost']}})
        return preds


class AnnotationEvaluator:
    def __init__(self, dataset: Type[pd.DataFrame] = None, annotations_filepath: str = None):
        if dataset is None and annotations_filepath is None:
            raise ValueError('df and all_annotations_filepath cannot be None at the same time')
        elif annotations_filepath is not None:
            self.df = pd.read_json(annotations_filepath, orient='records')
        else:
            self.df = dataset

    def run(self):
        metrics = []
        for _, row in self.df.iterrows():
            for system, prediction in row['predicted_paragraph_ids'].items():
                scores = compute_precision_recall_f1(gold=row['gold_paragraph_ids'], predicted=prediction.get('parsed_output'))
                cost = prediction['estimated_cost'] if prediction else float('nan')
                metrics.append({'annotator': system, 'cost': cost, **scores})
        dataf = pd.DataFrame(metrics)
        dataf['cost_bis'] = dataf['cost']
        dataf = (dataf
            .groupby('annotator')
            .agg({
                'precision': 'mean', 
                'recall': 'mean', 
                'f1': 'mean', 
                'cost': lambda x: x.sum() if not x.isna().all() else float('nan'),
                'cost_bis': 'mean',
            })
            .rename(columns={'cost': 'total_cost', 'cost_bis': 'avg_cost_per_sample'})
            .reset_index()
        )
        return dataf

def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = splitext(basename(args.questions_filepath))[0]
    all_annotations_filepath = os.path.join(args.output_dir, f'{base_filename}_all_annotations.json')
    annotation_scores_filepath = os.path.join(args.output_dir, f'annotation_scores_{base_filename}.csv')
    system_annotations_filepath = os.path.join(args.output_dir, f'{base_filename}_{args.system["model_name"]}_annotations.json')

    logging.info(f"Preparing data for annotation...")
    data_preparator = DataPreparator(args.questions_filepath, args.articles_filepath, args.do_evaluation)
    if os.path.exists(all_annotations_filepath):
        logging.info(f"Found existing annotated output file at {all_annotations_filepath}. Loading it...")
        df = pd.read_json(all_annotations_filepath, orient='records')
    else:
        df = data_preparator.preprocess()

    logging.info(f"Annotating data with annotator {args.system['model_name']}...")
    df = DataAnnotator(system_config=args.system, dataset=df).run(
        override_prev=args.do_evaluation,
        skip_unique_paragraph=not args.do_evaluation,
        save_while_running=not args.do_evaluation, 
        output_path=all_annotations_filepath,
    )
    df.to_json(all_annotations_filepath, orient='records', force_ascii=False, indent=2)

    if args.do_evaluation:
        logging.info(f"Evaluating systems' annotations...")
        scores = AnnotationEvaluator(annotations_filepath=all_annotations_filepath).run()
        scores.to_csv(annotation_scores_filepath, index=False, float_format='%.6f')
    else:
        logging.info(f"Transforming annotated data back to original format...")
        df = data_preparator.postprocess(df, system_to_keep=args.system['model_name'])
        df.to_json(system_annotations_filepath, orient='records', force_ascii=False, indent=2)
    
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=json.loads, help="The annotator system config.")
    parser.add_argument("--questions_filepath", type=str, help="The questions data file path to annotate.")
    parser.add_argument("--articles_filepath", type=str, help="The articles data file path.")
    parser.add_argument("--do_evaluation", action="store_true", default=False, help="Wether to evaluate the quality of the predicted LLM annotations compared to gold annotations (only for test set!).")
    parser.add_argument("--output_dir", type=str, help="The output directory for the annotations and scores.")
    args, _ = parser.parse_known_args()
    main(args)
