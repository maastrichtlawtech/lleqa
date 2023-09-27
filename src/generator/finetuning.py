import os
import sys
import json
import logging
import pathlib
import argparse
from tqdm import tqdm
from os.path import join
from datetime import datetime

import torch
import wandb
import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
    AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    AutoTokenizer, AutoConfig, GenerationConfig,
    BitsAndBytesConfig, set_seed,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, PeftConfig, PeftModel, get_peft_model,
)
try:
    from src.utils.loggers import LoggingHandler
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.loggers import LoggingHandler
from src.utils.common import count_trainable_parameters
from src.utils.metrics import compute_precision_recall_f1
from src.data.lleqa import LLeQADatasetQALoader, AnswerOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
OPTIMIZERS = {
    "adafactor": "adafactor",
    "adamw": "paged_adamw_8bit",
    "lion": "paged_lion_8bit",
}

def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    set_seed(args.seed)
    
    logging.info("Setting up model configuration...")
    model = None
    try:
        # Load the base model configuration.
        model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        base_model_name = args.model_name_or_path
    except OSError:
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        base_model_name = peft_config.base_model_name_or_path
        model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Depending on the provided model, define the type of task and architecture as well as the context size and maximum input length.
    is_seq2seq = any(m in base_model_name.lower() for m in {"t5", "t0", "ul2"})
    if is_seq2seq:
        task = "SEQ_2_SEQ_LM"
        ModelType = AutoModelForSeq2SeqLM
        context_size = getattr(model_config, 'n_positions')
    else:
        task = "CAUSAL_LM"
        ModelType = AutoModelForCausalLM
        context_size = getattr(model_config, 'max_position_embeddings', getattr(model_config, 'max_seq_len', None))

    # Setup dynamic NTK-aware scaled RoPE, if possible.
    rope_config = {}
    rope_scaling_supported = model_config.architectures[0] in ["LlamaForCausalLM", "GPTNeoXForCausalLM"]
    if rope_scaling_supported and args.max_context_len and context_size < args.max_context_len:
        scale_factor = args.max_context_len / context_size
        rope_config = {'rope_scaling': {"type": "dynamic", "factor": scale_factor}}
        context_size = args.max_context_len

    # Setup the quantization configuration for the base model.
    fp16_supported = all(m not in base_model_name.lower() for m in {"t5"})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if fp16_supported else torch.float32,
    )

    # Setup the model loading configuration.
    load_config = {
        'pretrained_model_name_or_path': base_model_name,
        'quantization_config': bnb_config,
        'device_map': 'auto',
        'trust_remote_code': True,
        'use_cache': False if args.do_train and args.gradient_checkpointing else True,
    }
    load_config.update(rope_config)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, legacy=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Setup the data configuration for the processors.
    data_config = {
        'articles_filepath': args.corpus_filepath,
        'negatives_filepath': args.negatives_filepath,
        'out_format': 'tokenized',
        'num_evidence': 5,
        'evidence_retriever_model_name': 'oracle',
        'evidence_order': 'most_relevant_first',
        'tokenizer': tokenizer,
        'task': task,
        'target_length_percentile': 75,
        'context_size': context_size,
    }
    
    # Setup the decoding strategy.
    strategies = {
        'greedy-search': {},
        'beam-search': {'num_beams': 4, 'early_stopping': True},
        'random-sampling': {'do_sample': True, 'top_k' : 0, 'temperature': args.temperature},
        'topk-sampling': {'do_sample': True, 'top_k' : args.top_k, 'temperature': args.temperature},
        'nucleus-sampling': {'do_sample': True, 'top_k': 0, 'top_p': args.top_p, 'temperature': args.temperature},
        'topk-nucleus-sampling': {'do_sample': True, 'top_k': args.top_k, 'top_p': args.top_p, 'temperature': args.temperature},
    }
    decoding_strategy = GenerationConfig(**strategies[args.decoding_strategy])

    # Define metrics computation.
    def prepare_compute_metrics(return_outputs: bool):
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        parser = AnswerOutputParser()
        def compute_metrics(pred_sample):
            preds, targets = pred_sample
            if isinstance(preds, tuple):
                preds = preds[0]
            targets = np.where(targets != -100, targets, tokenizer.pad_token_id) # Change -100 values in targets to PAD token for decoding.
            decoded_preds, decoded_targets = (tokenizer.batch_decode(samples, skip_special_tokens=True) for samples in [preds, targets]) # Decode token sequences.
            pred_rationales, gold_rationales = ([parser.parse(x)['parsed_rationales'] for x in samples] for samples in [decoded_preds, decoded_targets]) # Parse rationales.
            pred_answers, gold_answers = (LLeQADatasetQALoader.postprocess(x) for x in [decoded_preds, decoded_targets]) # Clean decoded sequences a bit.
            # Compute precision, recall, and F1 scores for rationales extraction.
            scores = [compute_precision_recall_f1(predicted=p, gold=g) for p, g in zip(pred_rationales, gold_rationales)]
            scores = {f"rationales_{key}": np.mean(list(map(lambda x: x[key], scores))) for key in scores[0].keys()}
            # Compute ROUGE and METEOR for predicted answers.
            scores.update(rouge.compute(predictions=pred_answers, references=gold_answers, use_stemmer=True))
            scores.update(meteor.compute(predictions=pred_answers, references=gold_answers))
            # Log the average output length.
            scores["out_length"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
            if not return_outputs:
                return scores
            else:
                results = [
                    {'pred_answer': pa, 'gold_answer': ga, 'pred_rationales': pr, 'gold_rationales': gr} 
                    for pa, ga, pr, gr in zip(pred_answers, gold_answers, pred_rationales, gold_rationales)
                ]
                return scores, results
        return compute_metrics
    
    if args.do_train:
        logging.info("Setting up training configuration...")
        new_model_name = f'{base_model_name.replace("/", "-")}-LLeQA'
        out_path = join(args.output_dir, f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{new_model_name}')
        logs_path = join(args.output_dir, 'logs')
        os.makedirs(logs_path, exist_ok=True)
        wandb.init(
            project="llm-finetuning", 
            name=new_model_name,
            config=args,
            dir=logs_path,
        )

        # Load base model.
        model = ModelType.from_pretrained(**load_config)
        model = prepare_model_for_kbit_training(model)

        # Create the trainable PEFT adapters.
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            task_type=task,
        )
        model = get_peft_model(model, peft_config)
        count_trainable_parameters(model)

        # Load the train/dev sets.
        train_preparator = LLeQADatasetQALoader(**data_config, stage="fit", questions_filepath=args.train_questions_filepath)
        dev_preparator = LLeQADatasetQALoader(**data_config, stage="eval", questions_filepath=args.dev_questions_filepath)
        train_dataset, dev_dataset = train_preparator.run(), dev_preparator.run()

        # Set a limit for completion tokens when generating dev sample targets.
        decoding_strategy.max_new_tokens = train_preparator.config['max_output_len']

        # Load the data collator.
        collator = DataCollatorForSeq2Seq(
            tokenizer, 
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100, #Set [PAD] tokens to -100 to ignore those in the loss computation
            return_tensors="pt",
        ) 

        # Define the training arguments.
        training_args = Seq2SeqTrainingArguments(
            # Epochs
            num_train_epochs=args.num_epochs,
            # Effective batch size
            auto_find_batch_size=False, 
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # Learning rate
            lr_scheduler_type=args.scheduler,
            learning_rate=args.lr,
            warmup_ratio=0.01,
            # Optimization
            optim=OPTIMIZERS.get(args.optimizer), 
            adam_beta1=0.9, 
            adam_beta2=0.999, 
            adam_epsilon=1e-07, 
            weight_decay=0.01,
            gradient_checkpointing=args.gradient_checkpointing,
            fp16=fp16_supported,
            max_grad_norm=1.0,
            deepspeed=args.deepspeed,
            # Logging
            logging_strategy='steps', 
            logging_steps=1, 
            log_level='passive', 
            report_to='wandb', 
            logging_dir=logs_path,
            # Evaluation
            evaluation_strategy='epoch' if args.do_val else 'no', 
            per_device_eval_batch_size=args.batch_size,
            # Saving
            save_strategy='epoch',
            save_total_limit=2,
            output_dir=out_path,
            # HuggingFace Hub
            push_to_hub=args.push_to_hub,
            hub_strategy='end',
            hub_model_id=f"{args.hf_username}/{base_model_name}-LLeQA".replace("/", "_"),
            hub_token=os.getenv("HF", None),
            # Generation
            predict_with_generate=True,
            generation_config=decoding_strategy,
        )

        # Create the Trainer instance.
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            compute_metrics=prepare_compute_metrics(return_outputs=False),
        )
        
        logging.info("Starting model fine-tuning...")
        trainer.train()

        logging.info("Saving the adapter weigths...")
        trainer.save_model(out_path)
        if args.push_to_hub:
            trainer.create_model_card()
            trainer.push_to_hub(
                f"{args.hf_username}/"
                + f"lleqa_{base_model_name}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
                use_auth_token=True,
            )
        logging.info("Done.")
    
    if args.do_test:
        logging.info("Setting up testing configuration...")
        if model is None:
            # Load the base quantized model together with the trained PEFT adapters.
            model = ModelType.from_pretrained(**load_config)
            model = PeftModel.from_pretrained(model, args.model_name_or_path, device_map="auto")
        model.config.use_cache = True
        model.eval()

        # Load the test set.
        data_config.update({'evidence_retriever_model_name': args.evidence_retriever})
        test_preparator = LLeQADatasetQALoader(**data_config, stage="eval", questions_filepath=args.test_questions_filepath)
        test_dataset = test_preparator.run().with_format("torch")

        logging.info(f"Running evaluation...")
        decoding_strategy.max_new_tokens = 350
        outputs, targets = [], []
        for sample in tqdm(test_dataset, desc="Testing"):
            with torch.no_grad():
                p = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), generation_config=decoding_strategy)[0]
                if task == "CAUSAL_LM":
                    p = p[sample["input_ids"].size(0):]
            outputs.append(p.detach().cpu().numpy())
            targets.append(sample["labels"].numpy())

        # Pad and stack gold answer tokens (needed for metrics calculation).
        max_target_len = max(len(x) for x in targets)
        targets = [np.pad(x, (0, max_target_len - len(x)), 'constant', constant_values=-100) for x in targets]
        targets = np.stack(targets)

        logging.info("Computing and saving results...")
        compute_metrics = prepare_compute_metrics(return_outputs=True)
        scores, results = compute_metrics((outputs, targets))
        with open(join(args.output_dir, f"test_scores_{args.model_name_or_path.replace('/', '_')}.json"), "w") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        with open(join(args.output_dir, f"test_results_{args.model_name_or_path.replace('/', '_')}.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info("Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Actions.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--do_val", action="store_true", default=False, help="Wether to perform validation during training.")
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform testing after training.")
    # Data.
    parser.add_argument("--train_questions_filepath", type=str, help="The path to the training questions file.")
    parser.add_argument("--dev_questions_filepath", type=str, help="The path to the dev questions file.")
    parser.add_argument("--test_questions_filepath", type=str, help="The path to the testing questions file.")
    parser.add_argument("--negatives_filepath", type=str, help="The path to the negatives file for all questions.")
    parser.add_argument("--corpus_filepath", type=str, help="The path to the corpus file.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Model.
    parser.add_argument("--model_name_or_path", type=str, help="The model id (HF) or path.")
    parser.add_argument("--max_context_len", type=int, choices=[512, 1024, 2048, 4096, 8192, 16384], help="")
    parser.add_argument("--decoding_strategy", type=str, choices=['greedy-search','beam-search','random-sampling','topk-sampling','nucleus-sampling','topk-nucleus-sampling'], help="")
    parser.add_argument("--num_beams", type=int, default=4, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--top_p", type=float, default=0.95, help="")
    parser.add_argument("--temperature", type=float, default=0.1, help="")
    # Training.
    parser.add_argument("--num_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="This parameter multiplied by the batch_size gives you the effective batch size.")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "adafactor", "lion"], help="Type of optimizer to use for training.")
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, choices=["linear","cosine","cosine_w_restarts","polynomial","constant","constant_w_warmup"], help="Type of learning rate scheduler to use for training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Whether to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    # Other.
    parser.add_argument("--evidence_retriever", type=str, help="")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Wether to push the trained model to Hugging Face Hub.")
    parser.add_argument("--hf_username", type=str, default="", help="The username to use when pushing the model to Hugging Face Hub.")
    args, _ = parser.parse_known_args()
    main(args)
