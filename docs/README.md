# Documentation

### Setup

This repository is tested on Python 3.8+. First, you should install a virtual environment:

```bash
python3 -m venv .venv/lleqa
source .venv/lleqa/bin/activate
```

Then, you can install all dependencies:

```bash
pip install -r requirements.txt
```

## LLeQA: The Long-form Legal Question Answering Dataset

We provide access to LLeQA on [🤗 Datasets](https://huggingface.co/datasets/maastrichtlawtech/lleqa) in restricted access for research purposes only. To load the dataset, you simply need to run:

```python
from datasets import load_dataset

repo = "maastrichtlawtech/lleqa"

# Load corpus.
articles = load_dataset(repo, name="corpus")

# Load questions.
train_questions = load_dataset(repo, name="questions", split="train")
dev_questions = load_dataset(repo, name="questions", split="dev")
test_negatives = load_dataset(repo, name="questions", split="test")

# Load negatives (needed for training).
bm25_negatives = load_dataset(repo, name="negatives", split="bm25")
#me5_negatives = load_dataset(repo, name="negatives", split="me5")
```

To run the following experiments, we recommend downloading the dataset and placing the files inside a ``*data/lleqa/*'' folder at the root of this repository.

## "Retrieve-then-read" pipeline

We use the popular retrieve-then-read pipeline, which first leverages a retriever over a large evidence corpus to fetch a set of relevant legislative articles, and then employs a reader to peruse these articles and formulate a comprehensive answer.

### Retriever

Our retriever relies on a lightweight [CamemBERT](https://huggingface.co/camembert-base)-based bi-encoder model, wich enables fast and effective retrieval.

#### Hard negatives

The bi-encoder model is optimized using a contrastive learning strategy, which requires the inclusion of negative samples. Besides in-batch negatives, we sample hard negatives using two different systems: BM25 and DPR-based.

To create the BM25 negatives, you can run the following command:

```bash
bash scripts/run_negatives_generation_bm25.sh
```

whose script comes with the following variables (to be adapted to your needs):

* `K1` (float, default=2.5): BM25 parameter *k1*;
* `B` (float, default=0.2): BM25 parameter *b*;
* `NUM_NEG` (int, default=10): number of negatives to generate per query.

To generate the DPR-based negatives, you can run:

```bash
bash scripts/run_negatives_generation_biencoder.sh
```

whose script comes with the following variables:

* `MODEL` (str, default="[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)"): model checkpoint ID on Hugging Face;
* `MAX_SEQ_LENGTH` (int, default=512): maximum input length (longer inputs are truncated);
* `SIM` (str, default="cos_sim"): similarity function for the question and article encoders (either "cos_sim" or "dot_product");
* `NUM_NEG` (int, default=10): number of negatives to generate per query.

#### Training

You can train the bi-encoder model by running:

```bash
bash scripts/run_biencoder_training.sh
```

whose script comes with the following variables:

* `MODEL` (str, default="[camembert-base](https://huggingface.co/camembert-base)"): Transformers checkpoint ID from Hugging Face;
* `MAX_SEQ_LEN` (int, default=384): maximum input length (longer inputs are truncated);
* `POOL` (str, default="mean"): pooling strategy for the question and article encoders (either "mean", "max", or "cls");
* `SIM` (str, default="cos_sim"): similarity function for the question and article encoders (either "cos_sim" or "dot_product");
* `EPOCHS` (int, default=20): number of training epochs;
* `BATCH_SIZE` (int, default=32): batch size;
* `SCHEDULER` (str, default="warmuplinear"): learning rate scheduler (either "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", or "warmupcosinewithhardrestarts");
* `LR` (float, default=2e-5): learning rate;
* `WARMUP_STEPS` (int, default=60): number of warmup steps;
* `OPTTIMIZER` (str, default="AdamW"): optimizer (either "AdamW" or "Adafacor");
* `WD` (float, default=0.01): weight decay;
* `FP16` (bool, default=True): whether to use mixed precision training;
* `EVAL_BEFORE_TRAINING` (bool, default=True): whether to evaluate the model before training;
* `EVAL_DURING_TRAINING` (bool, default=True): whether to evaluate the model during training;
* `LOG_STEPS` (int, default=1): number of steps between each evaluation;
* `DO_TEST` (bool, default=True): whether to evaluate the model on the test set after training;
* `DO_SAVE` (bool, default=True): whether to save the model after training.

#### Evaluation

You can evaluate the performance of a biencoder model on LLeQA by running:

```bash
bash scripts/run_biencoder_evaluation.sh
```

whose script comes with the following variables:

* `MODEL` (str, default="[camembert-base-lleqa](https://huggingface.co/maastrichtlawtech/camembert-base-lleqa)"): SentenceTransformer checkpoint ID from Hugging Face;
* `MAX_SEQ_LEN` (int, default=512): maximum input length (longer inputs are truncated);
* `SIM` (str, default="cos_sim"): similarity function for the question and article encoders (either "cos_sim" or "dot_product").

As a baseline, we also report the performance of BM25, which can be obtained by running:

```bash
bash scripts/run_bm25_evaluation.sh
```

whose script comes with the following variables:

* `K1` (float, default=2.5): BM25 parameter *k1*;
* `B` (float, default=0.2): BM25 parameter *b*.

### Reader

For our reader, we use a large language model (LLM) that we adapt to our task via two distinct learning strategies: *in-context learning*, wherein the model learns from instructions and a set of contextually provided examples; and *parameter-efficient finetuning*, where a small number of extra parameters are optimized on a downstream dataset while the base model's weights are quantized and remain unchanged.

#### In-context learning

You can evaluate the in-context performance of some LLMs on LLeQA test set by running:

```bash
bash scripts/run_llm_fewshot.sh
```

whose script comes with the following variables:

* `NUM_EVIDENCE` (int, default=5): number of evidence paragraphs to retrieve that will be used as context;
* `EVIDENCE_ORDER` (str, default="most_relevant_first"): order in which evidence paragraphs are passed to the model (either "most_relevant_first" or "least_relevant_first");
* `EVIDENCE_RETRIEVER` (str, default="[camembert-base-lleqa](https://huggingface.co/maastrichtlawtech/camembert-base-lleqa)"): SentenceTransformer checkpoint ID from Hugging Face;
* `INSTRUCTION` (str, default="..."): instruction to the model;
* `NUM_DEMOS` (list, default=[0, 1, 2]): number of examples to provide to the model;
* `DEMO_TYPE` (str, default="similar"): type of examples to provide to the model (either "similar" or "random");
* `DEMO_RETRIEVER` (str, default="[sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base)"): SentenceTransformer checkpoint ID from Hugging Face;
* `TEMP` (float, default=0.1): temperature for sampling;
* `MAX_OUT_TOKENS` (int, default=350): maximum number of tokens to generate;
* `MODELS` (list, default=["[vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)", "[wizardLM-7B](https://huggingface.co/TheBloke/wizardLM-7B-HF)", "[tulu-7B](https://huggingface.co/TheBloke/tulu-7B-fp16)", "[guanaco-7B"](https://huggingface.co/TheBloke/guanaco-7B-HF)]): list of LLMs to evaluate.

#### Parameter-efficient finetuning

You can efficiently finetune LLMs on LLeQA training set by running:

```bash
bash scripts/run_llm_finetuning.sh
```

whose script comes with the following variables:

* `DO_TRAIN` (bool, default=True): whether to perform finetuning;
* `DO_VAL` (bool, default=True): whether to evaluate the model on the validation set during finetuning;
* `DO_TEST` (bool, default=True): whether to evaluate the model on the test set after finetuning;
* `MAX_CONTEXT_LEN` (int, default=4096): extended context length for LLMs using RoPE;
* `MODELS` (list, default=["[vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)", "[wizardLM-7B](https://huggingface.co/TheBloke/wizardLM-7B-HF)", "[tulu-7B](https://huggingface.co/TheBloke/tulu-7B-fp16)", "[guanaco-7B"](https://huggingface.co/TheBloke/guanaco-7B-HF)]): list of LLMs to finetune;
* `EPOCHS` (int, default=10): number of finetuning epochs;
* `BS` (int, default=1): batch size;
* `ACC_STEPS` (int, default=8): number of accumulation steps;
* `LR` (float, default=2e-4): learning rate;
* `SCHEDULER` (str, default="constant"): learning rate scheduler (either "linear", "cosine", "cosine_w_restarts", "polynomial", "constant", or "constant_w_warmup");
* `DEEPSPEED` (str, default=""): path to the DeepSpeed configuration file.
* `TEMP` (float, default=0.1): temperature for sampling;
* `DECODING` (str, default="nucleus-sampling"): decoding strategy (either 'greedy-search', 'beam-search', 'random-sampling', 'topk-sampling', 'nucleus-sampling', or 'topk-nucleus-sampling');
* `EVIDENCE_RETRIEVER` (str, default="[camembert-base-lleqa](https://huggingface.co/maastrichtlawtech/camembert-base-lleqa)"): SentenceTransformer checkpoint ID from Hugging Face.
