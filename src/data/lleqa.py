"""
The LLeQA dataset is a legal IR+QA dataset comprising of:
- a corpus of 27,941 law articles;
- a training set of 1,673 queries (with at least one relevant article from the corpus);
- a test set of 195 queries.
"""
import os
import sys
import json
import json
import random
import pathlib
from typing import Optional, List, Tuple, Dict, Set, Type

import re
import nltk
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from langchain.schema import BaseOutputParser
from datasets import Dataset as HFDataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, util, InputExample

try:
    from src.utils.common import read_json_file
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import read_json_file


def split_into_traindev(df: Type[pd.DataFrame], train_ratio: float = 0.9, seed: int = 42):
    train = df.sample(frac=train_ratio, random_state=seed)
    dev = df.drop(train.index).sample(frac=1.0, random_state=seed)
    return train.reset_index(drop=True), dev.reset_index(drop=True)


class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return {'full': text, 'parsed_rationales': re.findall(r'\[(\d+§\d+)\]', text)}


class LLeQADatasetQALoader:
    def __init__(self, 
                 questions_filepath: str, # Path to the questions file.
                 articles_filepath: str, #
                 negatives_filepath: str, #
                 out_format: str, # Output format of the dataset. Either 'raw' or 'tokenized'.
                 seed: int = 42, #
                 max_num_refs: int = 5, # The maximum number of references one question can have (if more, the question will be discarded).
                 num_evidence: int = 5, # The number of evidence articles to use per question.
                 evidence_retriever_model_name: str = "oracle", # The model to use for retrieving evidence articles. Either 'oracle' or a HF model id.
                 evidence_order: str = "most_relevant_first", # The order in which to return the evidence articles. Either 'most_relevant_first', 'most_relevant_last', or 'random'.
                 ### in-context learning only ###
                 instruction: str = None, # Optional instruction to use for the prompt.
                 num_demonstrations: int = 0, # The number of demonstrations to use per question.
                 demonstration_filepath: str = None, # The path to the file containing the demonstration examples (should correspond to the training set).
                 demonstration_type: str = "similar", # The type of demonstrations to use. Either 'random' or 'similar'.
                 text_similarity_model_name: str = None, # The model to use for retrieving similar question examples if demonstration_type is 'similar'.
                 ### 'tokenized' format only ###
                 tokenizer: Type[AutoTokenizer] = None, # Tokenizer (required when output format is set to 'tokenized').
                 task: str = None, # Type of language modeling (required when output format is set to 'tokenized'). Either "CAUSAL LM" or "SEQ_2_SEQ_LM".
                 stage: str = None, # Run stage (required when output format is set to 'tokenized'). Either 'fit' or 'eval'.
                 target_length_percentile: int = 95, # Percentile to use for the maximum target length (required when output format is set to 'tokenized'). Longer targets will be truncated.
                 context_size: int = None, # Maximum context size of the model (required when output format is set to 'tokenized'). Inputs will be truncated such that Input (+ Target if causalLM) < ContextSize.
        ):
        assert out_format in ['raw', 'tokenized'], f"The output format must be either 'raw' or 'tokenized'."
        assert max_num_refs in range(1, 21), "It is recommended to ignore questions with more than 20 references as evidence context becomes too large."
        assert num_evidence in range(0, 21), "The number of evidence articles must be between 0 and 20."
        assert evidence_order in ['most_relevant_first', 'most_relevant_last', 'random'], "The evidence order must be either 'most_relevant_first', 'most_relevant_last', or 'random'."
        self.config = {k: v for k, v in locals().items() if k not in ['self', 'tokenizer']}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer

    def run(self):
        # Load articles and negatives.
        self.dfA = pd.read_json(self.config['articles_filepath'], orient='records')
        with open(self.config['negatives_filepath'], "r") as f:
            self.negatives = json.load(f)

        # Setup the evidence retriever.
        if self.config['num_evidence'] > 0 and self.config['evidence_retriever_model_name'] not in ['oracle', 'random']:
            self.retriever = SentenceTransformer(self.config['evidence_retriever_model_name'])
            self.article_embeddings = self.retriever.encode(self.dfA['article'].tolist(), device=self.device)

        # Prepare demonstrations and example selector.
        if self.config['num_demonstrations'] > 0:
            assert self.config['demonstration_filepath'] is not None, "You did not provide a demonstration filepath."
            assert self.config['demonstration_type'] in ['random', 'similar'], "The demonstration type must be either 'random' or 'similar'."
            self.demonstrations = self._prepare_demonstrations(dataf=pd.read_json(self.config['demonstration_filepath'], orient='records'))
            if self.config['demonstration_type'] == 'similar':
                assert self.config['text_similarity_model_name'] is not None, "You did not provide a model name for retrieving similar in-context examples."
                self.selector = SentenceTransformer(self.config['text_similarity_model_name'])
                self.question_embeddings = self.selector.encode(self.demonstrations['question'].tolist(), device=self.device)

        # Prepare samples.
        data = self._prepare_samples(dataf=pd.read_json(self.config['questions_filepath'], orient='records'))
        if self.config['out_format'] == 'tokenized':
            assert self.tokenizer is not None, "You did not provide a tokenizer."
            assert self.config['task'] in ["SEQ_2_SEQ_LM", "CAUSAL_LM"], "The language modeling task must be either 'SEQ_2_SEQ_LM', or 'CAUSAL_LM'."
            assert self.config['stage'] in ['fit', 'eval'], f"The running stage must either be 'fit' or 'eval'."
            assert self.config['target_length_percentile'] in range(0, 101), "The target length percentile must be between 0 and 100."
            assert self.config['context_size'] is not None, "You did not provide a maximum context window size."
            self.config['max_output_len'] = self._calculate_percentile_length(dataset=data, text_column="target", percentile=self.config['target_length_percentile'])
            self.config['max_input_len'] = self.config['context_size'] - self.config['max_output_len'] if self.config['task'] == "CAUSAL_LM" else self.config['context_size']
            data = self._tokenize_dataset(data, source_column="source", target_column="target")
        return data

    def _prepare_demonstrations(self, dataf):
        return (dataf
            .pipe(self._start_pipeline)
            .pipe(self._drop_questions_w_too_many_article_references, log="demonstrations")
            .pipe(self._drop_questions_wo_paragraph_references, log="demonstrations")
            .pipe(self._add_negative_article_ids)
            .pipe(self._create_evidence_context, truncate=50)
            .pipe(self._clean_answer)
            .pipe(self._create_target_outputs)
        )

    def _prepare_samples(self, dataf):
        """ Transform the dataframe to samples in the form {'id': ..., 'source': ..., 'target': ...}.
        """
        return (dataf
            .pipe(self._start_pipeline)
            .pipe(self._drop_questions_w_too_many_article_references, log="samples")
            .pipe(self._drop_questions_wo_paragraph_references, log="samples")
            .pipe(self._add_negative_article_ids)
            .pipe(self._create_evidence_context)
            .pipe(self._clean_answer)
            .pipe(self._create_target_outputs)
            .pipe(self._create_source_inputs)
            .pipe(self._drop_unecessary_columns)
            .pipe(self._convert_to_hf)
        )

    def _calculate_percentile_length(self, dataset: Type[HFDataset], text_column: str, percentile: int):
        """Calculate the maximum sequence length for of a given text column and return the Xth percentile of each for better utilization.
        """
        tokenized_inputs = dataset.map(lambda x: self.tokenizer(x[text_column], truncation=False, padding=False), remove_columns=dataset.column_names)
        sequence_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
        return int(np.percentile(sequence_lenghts, percentile))

    def _tokenize_dataset(self, dataset: Type[HFDataset], source_column: str, target_column: str, instruction: Optional[str] = ""):
        return dataset.map(
            self._tokenize_function,
            fn_kwargs={
                "source_column": source_column,
                "target_column": target_column,
                "instruction": instruction,
            },
            remove_columns=dataset.column_names, 
            desc="Running tokenizer on dataset",
        )

    def _tokenize_function(self, sample, source_column: str, target_column: str, instruction: str):
        # Tokenize target text.
        targets = self.tokenizer(text_target=sample[target_column], padding=False, truncation=self.config['stage'] == 'fit', max_length=self.config['max_output_len'])

        # Calculate max input length. During training, truncate inputs according to the known length of the corresponding target. 
        # During inference, truncate according to a predefined fixed length as target is supposed to be unknown.
        max_input_len = self.config['max_input_len']
        if self.config['stage'] == 'fit' and self.config['task'] == "CAUSAL_LM":
            max_input_len = self.config['context_size'] - len(targets["input_ids"])
        
        # Tokenize input text.
        prompt = instruction + sample[source_column]
        sources = self.tokenizer(text=prompt, padding=False, truncation=True, max_length=max_input_len)

        # Prepare model inputs according to the running stage and language modeling task.
        if self.config['stage'] == 'fit' and self.config['task'] == 'CAUSAL_LM':
            # Append target tokens to source tokens for autoregressive language modeling using the whole sequence but only compute
            # the loss on target tokens by setting source tokens to -100 in the labels.
            inputs = {k: sources[k] + targets[k] for k in sources.keys()}
            inputs['labels'] = [-100] * len(sources['input_ids']) + targets["input_ids"]
        else:
            # For causal LM in test mode and seq2seq models in both training and test modes, return inputs and targets as is.
            inputs = sources.copy()
            inputs['labels'] = targets["input_ids"]
        return inputs

    def _start_pipeline(self, dataf):
        return dataf.copy()

    def _drop_questions_w_too_many_article_references(self, dfQ, log=""):
        dfQ_post = dfQ[dfQ['article_ids'].apply(lambda x: len(x) <= self.config['max_num_refs'])].reset_index(drop=True)
        if len(dfQ_post) < len(dfQ):
            print((
                f"{log.capitalize()}: "
                f"Dropped {len(dfQ) - len(dfQ_post)} questions with more than {self.config['max_num_refs']} referenced articles. "
                f"The resulting dataset contains {len(dfQ_post)} questions."
            ))
        return dfQ_post

    def _drop_questions_wo_paragraph_references(self, dfQ, log=""):
        col_name = self._get_paragraph_column_name(dfQ)
        dfQ_post = dfQ[dfQ[col_name].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        if len(dfQ_post) < len(dfQ):
            print((
                f"{log.capitalize()}: "
                f"Dropped {len(dfQ) - len(dfQ_post)} questions with no referenced paragraphs. "
                f"The resulting dataset contains {len(dfQ_post)} questions."
            ))
        return dfQ_post

    def _add_negative_article_ids(self, dfQ):
        dfQ['negative_ids'] = dfQ.apply(lambda row: self.negatives[str(row['id'])], axis=1)
        return dfQ

    def _create_evidence_context(self, dfQ, truncate=None):
        def format_paragraph(p):
            p = re.sub(r"^§\s?\d+(er)?\.", "", p).lstrip()
            if not p.endswith('.'):
                p += '.'
            if truncate is not None:
                p = ' '.join(p.split()[:truncate])
            return p

        def format_article(article_id):
            paragraphs = self.dfA.loc[self.dfA['id'] == article_id, 'paragraphs'].iloc[0]
            paragraphs = [f"[{article_id}§{p_id}] {format_paragraph(p_text)}" for p_id, p_text in paragraphs.items()]
            return '\n'.join(paragraphs)

        def get_evidence(row):
            if self.config['evidence_retriever_model_name'] == 'oracle':
                # An oracle retriever always returns the relevant articles in the top-k results' context. If k > num_rel, we add hard negatives.
                articles = [format_article(x) for x in row['article_ids']] #gold
                remaining = self.config['num_evidence'] - len(articles)
                if remaining > 0:
                    hard_neg_articles = [format_article(x) for x in row['negative_ids'][:remaining]]
                    articles = articles + hard_neg_articles
            elif self.config['evidence_retriever_model_name'] == 'random':
                article_ids = random.sample(self.dfA['id'].tolist(), k=self.config['num_evidence'])
                articles = [format_article(x) for x in article_ids]
            else:
                q_emb = self.retriever.encode(row['question'], device=self.device, show_progress_bar=False)
                hits = util.semantic_search(q_emb, self.article_embeddings, top_k=self.config['num_evidence'], score_function=util.cos_sim)
                hits = self.dfA.iloc[[h['corpus_id'] for h in hits[0]]]
                articles = [format_article(x) for x in hits['id']]
            
            if self.config['evidence_order'] == 'random':
                random.shuffle(articles)
            elif self.config['evidence_order'] == 'most_relevant_last':
                articles.reverse()
            return '\n'.join(articles)
        
        random.seed(self.config['seed'])
        dfQ['context'] = dfQ.apply(get_evidence, axis=1)
        return dfQ

    def _clean_answer(self, dfQ):
        def cleansing(text):
            #text = re.sub(r"Pour plus d['’]info[^.]+?\.\s?", '', text) #Sometimes they refer to other answers (irrelevant), but some other times to relevant external resources.
            text = re.sub(r"Avant d['’]aller plus loin\s?", '', text)
            text = re.sub(r"Cette fiche a été mise à jour il y a plus d['’]un an\.?\s?", '', text)
            return text

        dfQ['answer'] = dfQ['answer'].apply(cleansing)
        return dfQ

    def _create_target_outputs(self, dfQ):
        col_name = self._get_paragraph_column_name(dfQ)
        rationales = dfQ[col_name].apply(lambda r: ', '.join(f'[{x}]' for x in r))
        dfQ['target'] = "```\nANSWER: " + dfQ['answer'] + "\nRATIONALES: " + rationales + "\n```"
        return dfQ

    def _create_source_inputs(self, dfQ):
        I = self.config['instruction'] if self.config['instruction'] else ''
        C = "\nCONTEXT:\n" + dfQ['context'] + "\n"
        Q = "QUESTION: " + dfQ['question'] + "\n"
        A = "```\nANSWER: "
        D = ''
        if self.config['num_demonstrations'] > 0:
            D = dfQ.apply(self.get_demonstrations, axis=1)
        dfQ['source'] = I + D + C + Q + A
        return dfQ

    def get_demonstrations(self, row):
        if self.config['demonstration_type'] == "random":
            samples = self.demonstrations.sample(n=self.config['num_demonstrations'], random_state=self.config['seed'])
        elif self.config['demonstration_type'] == "similar":
            q_emb = self.selector.encode(row['question'], device=self.device, show_progress_bar=False)
            hits = util.semantic_search(q_emb, self.question_embeddings, top_k=len(self.demonstrations), score_function=util.cos_sim)
            hits = [h['corpus_id'] for h in hits[0] if h['score'] < 1.0][:self.config['num_demonstrations']]
            samples = self.demonstrations.iloc[hits]
        demos = [f"CONTEXT:\n{c}\nQUESTION: {q}\n{t}" for c, q, t in zip(samples['context'], samples['question'], samples['target'])]
        return "Here are some examples:\n\n" + "\n\n".join(demos) + "\n"

    def _drop_unecessary_columns(self, dfQ):
        return dfQ[['id', 'source', 'target']]

    def _convert_to_hf(self, dfQ):
        return HFDataset.from_pandas(dfQ)

    def _get_paragraph_column_name(self, dfQ):
        return dfQ.columns[dfQ.columns.str.endswith('paragraph_ids')][0]

    @staticmethod
    def postprocess(texts):
        if texts is None:
            return texts
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.replace("```", "").replace("ANSWER:", "").replace("RATIONALES:", "").replace("CONTEXT:", "") for t in texts]
        texts = [re.sub(r'\[\d+§\d+\],?\s?', '', t).strip() for t in texts] #Remove rationale identifiers from predicted answer.
        texts = ["\n".join(nltk.sent_tokenize(t, language="french")) for t in texts] #ROUGE-LSum expects newline after each sentence.
        return texts


class LLeQADatasetIRLoader:
    def __init__(self,
                 stage: str,
                 corpus_path_or_url: str,
                 train_path_or_url: Optional[str] = None,
                 dev_path_or_url: Optional[str] = None,
                 test_path_or_url: Optional[str] = None,
                 negatives_path_or_url: Optional[str] = None,
                 synthetic_path_or_url: Optional[str] = None,
                 synthetic_negatives_path_or_url: Optional[str] = None,
        ):
        assert stage in ['fit', 'eval'], f"The running stage must either be 'fit' or 'eval'."
        self.stage = stage
        self.corpus_path_or_url = corpus_path_or_url
        self.train_path_or_url = train_path_or_url
        self.dev_path_or_url = dev_path_or_url
        self.test_path_or_url = test_path_or_url
        self.negatives_path_or_url = negatives_path_or_url
        self.synthetic_path_or_url = synthetic_path_or_url
        self.synthetic_negatives_path_or_url = synthetic_negatives_path_or_url

    def run(self):
        # Load corpus of articles.
        corpus = pd.read_json(self.corpus_path_or_url)

        if self.stage == 'eval':
            test_samples = pd.read_json(self.test_path_or_url)
            test_dataset = LLeQADataset(test_samples, corpus, 'test', add_doc_title=True)
            return {
                'corpus': test_dataset.documents,
                'test_queries': test_dataset.queries,
                'test_labels': test_dataset.one_to_many_pairs,
            }
        
        # Load train/val sets. If no validation data is given, split training set into train/val sets with ratio 88/12 such that dev_size matches test_size.
        train_samples = pd.read_json(self.train_path_or_url)
        if self.dev_path_or_url is not None:
            dev_samples = pd.read_json(self.dev_path_or_url)
        else:
            train_samples, dev_samples = split_into_traindev(train_samples, train_ratio=0.88)
            train_samples.to_json("./train.json", orient='records', force_ascii=False, indent=2)
            dev_samples.to_json("./dev.json", orient='records', force_ascii=False, indent=2)
        
        # Load hard negatives, if any.
        negatives = None
        if self.negatives_path_or_url is not None:
            negatives = read_json_file(self.negatives_path_or_url)

        # Use extra synthetic samples for training, if any.
        if self.synthetic_path_or_url is not None:
            synthetic_samples = pd.read_json(self.synthetic_path_or_url)
            train_samples = pd.concat([train_samples, synthetic_samples], ignore_index=True)

            # Load hard negatives for synthetic samples, if any.
            synthetic_negatives = []
            if self.synthetic_negatives_path_or_url is not None:
                synthetic_negatives = read_json_file(self.synthetic_negatives_path_or_url)
                negatives = negatives + synthetic_negatives
            elif negatives is not None:
                raise ValueError("You must provide hard negatives for synthetic samples.")

        # Make sure that no dev sample is also in the (potentially augmented) train set.
        dup_questions = train_samples[train_samples['question'].isin(dev_samples['question'])]
        if len(dup_questions) > 0:
            print(f"Found {len(dup_questions)} questions that appear both in train and dev sets. Removing them from train set...")
            train_samples.drop(dup_questions.index, inplace=True)

        # Create the train/dev Pytorch datasets for use in dataloaders.
        train_dataset = LLeQADataset(train_samples, corpus, 'train', add_doc_title=True, hard_negatives=negatives)
        dev_dataset = LLeQADataset(dev_samples, corpus, 'dev', add_doc_title=True)

        # Load test set, if any.
        test_dataset = None
        if self.test_path_or_url is not None:
            test_samples = pd.read_json(self.test_path_or_url)
            test_dataset = LLeQADataset(test_samples, corpus, 'test', add_doc_title=True)

        # Return the datasets.
        return {
            'train': train_dataset, 
            'corpus': train_dataset.documents, 
            'dev_queries': dev_dataset.queries, 
            'dev_labels': dev_dataset.one_to_many_pairs,
            'test_queries': test_dataset.queries if test_dataset is not None else None,
            'test_labels': test_dataset.one_to_many_pairs if test_dataset is not None else None,
        }


class LLeQADataset(Dataset):
    def __init__(self, 
                 queries: pd.DataFrame, 
                 documents: pd.DataFrame,
                 stage: str, #train, dev, test
                 add_doc_title: Optional[bool] = False, #whether or not we should append the document title before its content.
                 hard_negatives: Optional[Dict[str, List[str]]] = None, #Dict[qid, List[neg_docid_i]].
                ):
        self.stage = stage
        self.add_doc_title = add_doc_title 
        self.hard_negatives = hard_negatives 
        self.queries = self.get_id_query_pairs(queries) #Dict[qid, query]
        self.documents = self.get_id_document_pairs(documents) #Dict[docid, document]
        self.one_to_one_pairs = self.get_one_to_one_relevant_pairs(queries) #List[(qid, pos_docid_i)]
        self.one_to_many_pairs =  self.get_one_to_many_relevant_pairs(queries) #Dict[qid, List[pos_docid_i]]

    def __len__(self):
        if self.stage == "train":
            return len(self.one_to_one_pairs)
        return len(self.one_to_many_pairs)

    def __getitem__(self, idx):
        pos_id, neg_id, pos_doc, neg_doc = (None, None, None, None)
        if self.stage == "train":
            # Get query and positive document.
            qid, pos_id = self.one_to_one_pairs[idx]
            query = self.queries[qid]
            pos_doc = self.documents[pos_id]
            if self.hard_negatives is not None:
                # Get one hard negative for the query (by poping the first one in the list and adding it back at the end).
                neg_id = self.hard_negatives[str(qid)].pop(0)
                neg_doc = self.documents[neg_id]
                self.hard_negatives[str(qid)].append(neg_id)
                return InputExample(texts=[query, pos_doc, neg_doc])
            return InputExample(texts=[query, pos_doc])
        else:
            qid, query = list(self.queries.items())[idx]
            return InputExample(texts=[query])

    def get_id_query_pairs(self, queries: pd.DataFrame) -> Dict[str, str]:
        return queries.set_index('id')['question'].astype('str').to_dict()

    def get_id_document_pairs(self, documents: pd.DataFrame) -> Dict[str, str]:
        if self.add_doc_title:
            documents['article'] = documents['description'].apply(lambda x: x[0] + " [SEP] " if len(x) > 0 else None).fillna('') + documents['article']
        return documents.set_index('id')['article'].astype('str').fillna('').to_dict()

    def get_one_to_one_relevant_pairs(self, queries: pd.DataFrame) -> List[Tuple[int, int]]:
        return (queries
            .explode('article_ids')
            .drop(columns=['question', 'answer', 'regions', 'topics'], errors='ignore')
            .drop(columns=queries.filter(regex='paragraph_ids$').columns)
            .rename(columns={'article_ids':'article_id','id':'question_id'})
            .apply(pd.to_numeric)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
            .to_records(index=False)
        )

    def get_one_to_many_relevant_pairs(self, queries: pd.DataFrame) -> Dict[str, List[str]]:
        return queries.set_index('id')['article_ids'].to_dict()
