import os
import re
import sys
import time
import pathlib
import textwrap
import itertools
from getpass import getpass
from dotenv import load_dotenv
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Set, Type, Any, Union

import torch
import numpy as np
from fastchat.model import get_conversation_template
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

# LLMs imports.
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts.example_selector import LengthBasedExampleSelector#, NGramOverlapExampleSelector
from langchain.llms import OpenAI, Anthropic, Cohere, AlephAlpha, AI21, HuggingFaceHub, HuggingFacePipeline

# Chat imports.
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder

# Embedding imports.
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector

# Utility imports.
from langchain.prompts import load_prompt
from langchain.llms.loading import load_llm
from langchain.callbacks import get_openai_callback
from langchain.callbacks.tracers import WandbTracer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Type-based imports.
from langchain.prompts.example_selector.base import BaseExampleSelector

# Custom imports.
try:
    from src.utils.common import catchtime
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.common import catchtime
from src.utils.FastChat import register_custom_conversations, register_custom_adapters
register_custom_conversations()
register_custom_adapters()

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CONTEXT_SIZE = {
    "gpt-3.5-turbo-16k-0613": 16384, "gpt-4-0613": 8192,
    "claude-instant-1.1": 100000, "claude-2.0": 100000,
    "j2-light": 8192, "j2-mid": 8192, "j2-ultra": 8192,
    "command-light": 4096, "command": 4096,
    "luminous-base": 2048, "luminous-extended": 2048, "luminous-supreme": 2048, "luminous-supreme-control": 2048,
}
EXT_LENGTH = 8192 #8192,16384

def compute_llm_api_cost(model_name: str, tokens: Dict[str, int]) -> float:
    # OpenAI models: https://openai.com/pricing
    if "gpt-3.5-turbo-16k" in model_name:
        prompt_pricing = 0.003 / 1e3
        completion_pricing = 0.004 / 1e3
    elif "gpt-3.5-turbo" in model_name:
        prompt_pricing = 0.0015 / 1e3
        completion_pricing = 0.002 / 1e3
    elif "gpt-4-32k" in model_name:
        prompt_pricing = 0.06 / 1e3
        completion_pricing = 0.12 / 1e3
    elif "gpt-4" in model_name:
        prompt_pricing = 0.03 / 1e3
        completion_pricing = 0.06 / 1e3
    # Anthropic models: https://www-files.anthropic.com/production/images/model_pricing_july2023.pdf
    elif "claude-instant" in model_name:
        prompt_pricing = 1.63 / 1e6
        completion_pricing = 5.51 / 1e6
    elif "claude" in model_name:
        prompt_pricing = 11.02 / 1e6
        completion_pricing = 32.68 / 1e6
    # Cohere models: https://cohere.com/pricing
    elif "command-light" in model_name:
        prompt_pricing = float('nan')
        completion_pricing = float('nan')
    elif "command" in model_name:
        prompt_pricing = 15 / 1e6
        completion_pricing = 15 / 1e6
    # AI21 Labs models: https://www.ai21.com/studio/pricing-v2
    elif "j2-light" in model_name:
        prompt_pricing = 0.003 / 1e3
        completion_pricing = 0.003 / 1e3
    elif "j2-mid" in model_name:
        prompt_pricing = 0.01 / 1e3
        completion_pricing = 0.01 / 1e3
    elif "j2-ultra" in model_name:
        prompt_pricing = 0.015 / 1e3
        completion_pricing = 0.015 / 1e3
    # Aleph-Alpha models: https://www.aleph-alpha.com/pricing
    elif model_name == "luminous-base":
        prompt_pricing = 0.006 / 1e3
        completion_pricing = 1.1 * prompt_pricing
    elif model_name == "luminous-extended":
        prompt_pricing = 0.009 / 1e3
        completion_pricing = 1.1 * prompt_pricing
    elif model_name == "luminous-supreme":
        prompt_pricing = 0.035 / 1e3
        completion_pricing = 1.1 * prompt_pricing
    elif model_name == "luminous-supreme-control":
        prompt_pricing = 0.045 / 1e3
        completion_pricing = 1.1 * prompt_pricing
    else:
        return float('nan')
    return tokens['prompt'] * prompt_pricing + tokens['completion'] * completion_pricing


class RandomExampleSelector(BaseExampleSelector, BaseModel):
    """ Custom Pydantic example selector that selects in-context examples randomly.
    """
    examples: List[Dict[str, str]]
    k: int = 1
    
    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=self.k, replace=False)


class LLMTaskSolver:
    def __init__(self, 
                 llm_config : Dict[str, object], 
                 prompt: str, 
                 examples: List[Dict[str, str]] = None, 
                 output_parser: Type[BaseOutputParser] = None,
                 log_to_wandb: bool = False,
        ):
        self.out_parser = output_parser
        self.llm = self.init_llm(**llm_config)
        self.prompt = self.create_prompt_template(prompt=prompt, examples=examples)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.tracer = [WandbTracer({"project": "prompt-design", "name": llm_config['model_name']})] if log_to_wandb else None

    def __call__(self, **kwargs):
        tokens = dict()
        tokens['prompt'] = self.get_num_tokens(self.prompt.format(**kwargs))
        if tokens['prompt'] > self.config.get('max_input_len', 1e6):
            print(f"Prompt is {tokens['prompt']} tokens whereas maximum input length is set to {self.config.get('max_input_len', None)}. Ignoring sample...")
            return None
        with catchtime() as t:
            output = self.chain.run(**kwargs, callbacks=self.tracer)
        tokens['completion'] = self.get_num_tokens(output)
        tokens['total'] = tokens['prompt'] + tokens['completion']
        return {
            'output': self.out_parser.parse(output) if self.out_parser else output.strip(),
            'input': self.prompt.format(**kwargs),
            'cost': compute_llm_api_cost(self.config.get('model_name', ''), tokens),
            'time': t.time,
            **{k + '_tokens': v for k, v in tokens.items()},
        }

    def init_llm(self, 
                 provider: str, 
                 model_name: str, 
                 temperature: float = 0.7, 
                 max_completion_tokens: int = 256, 
                 streaming: bool = False, 
                 **kwargs,
        ):
        assert provider in ["openai", "anthropic", "cohere", "ai21", "alephalpha", "huggingface", "localhost"], "Unkown provider."
        self.config = {k: v for k, v in locals().items() if k not in ['self', 'kwargs']}
        self.config.update(kwargs)
        self.config['max_input_len'] = CONTEXT_SIZE.get(model_name)
        if provider == "openai":
            params = {
                "openai_api_key": os.getenv(provider.upper()), 
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_completion_tokens,
                "streaming": streaming, 
                "verbose": streaming, 
                "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
            } 
            return OpenAI(**params) if any(s in model_name for s in ['davinci', 'curie', 'babbage', 'ada']) else ChatOpenAI(**params)
        elif provider == "anthropic":
            return Anthropic(
                anthropic_api_key=os.getenv(provider.upper()),
                model=model_name, 
                temperature=temperature, 
                max_tokens_to_sample=max_completion_tokens,
                streaming=streaming, 
                verbose=streaming, 
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
        elif provider == "cohere":
            return Cohere(
                cohere_api_key=os.getenv(provider.upper()),
                model=model_name,
                temperature=temperature, 
                max_tokens=max_completion_tokens,
            )
        elif provider == "ai21":
            return AI21(
                ai21_api_key=os.getenv(provider.upper()),
                model=model_name,
                temperature=temperature, 
                maxTokens=max_completion_tokens,
            )
        elif provider == "alephalpha":
            return AlephAlpha(
                aleph_alpha_api_key=os.getenv(provider.upper()),
                model=model_name, 
                temperature=temperature, 
                maximum_tokens=max_completion_tokens, 
                stop_sequences=["###"],
            )
        elif provider == "hugging-face":
            return HuggingFaceHub(
                huggingfacehub_api_token=os.getenv(provider.upper()),
                repo_id=model_name,
                model_kwargs={
                    "temperature": temperature, 
                    "max_new_tokens": max_completion_tokens,
                }
            )
        elif provider == "localhost":
            # Load tokenizer and model configuration.
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Depending on the provided model, define the type of task and architecture as well as the context size and maximum input length.
            is_seq2seq = any(m in model_name.lower() for m in {"t5", "t0", "ul2"})
            if is_seq2seq:
                task = "text2text-generation"
                ModelType = AutoModelForSeq2SeqLM
                context_size = self.config['max_input_len'] = getattr(model_config, 'n_positions')
            else:
                task = "text-generation"
                ModelType = AutoModelForCausalLM
                context_size = getattr(model_config, 'max_position_embeddings', getattr(model_config, 'max_seq_len', None))
                self.config['max_input_len'] = context_size - max_completion_tokens

            # Setup dynamic NTK-aware scaled RoPE, if possible.
            rope_config = {}
            rope_scaling_supported = model_config.architectures[0] in ["LlamaForCausalLM", "GPTNeoXForCausalLM"]
            if rope_scaling_supported and context_size < EXT_LENGTH:
                scale_factor = EXT_LENGTH / context_size
                rope_config = {'rope_scaling': {"type": "dynamic", "factor": scale_factor}}
                self.config['max_input_len'] = EXT_LENGTH - max_completion_tokens if not is_seq2seq else self.config['max_input_len']

            # Setup the quantization configuration for the base model.
            fp16_supported = all(m not in model_name.lower() for m in {"t5"})
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if fp16_supported else torch.float32,
            )

            # Setup the model loading configuration.
            load_config = {
                'pretrained_model_name_or_path': model_name,
                'quantization_config': bnb_config,
                'device_map': 'auto',
                'trust_remote_code': True,
            }
            load_config.update(rope_config)

            # Load model.
            model = ModelType.from_pretrained(**load_config)

            # Setup the decoding strategy.
            decoding = {
                'greedy-search': {},
                'beam-search': {'num_beams': 4, 'early_stopping': True},
                'random-sampling': {'do_sample': True, 'top_k' : 0, 'temperature': temperature},
                'topk-sampling': {'do_sample': True, 'top_k' : 50, 'temperature': temperature},
                'nucleus-sampling': {'do_sample': True, 'top_k': 0, 'top_p': 0.95, 'temperature': temperature},
                'topk-nucleus-sampling': {'do_sample': True, 'top_k': 50, 'top_p': 0.95, 'temperature': temperature},
            }
            decoding_strategy = decoding['nucleus-sampling'] if temperature > 0 else decoding['greedy-search']

            # Create generation pipeline.
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer, 
                max_new_tokens=max_completion_tokens,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                **decoding_strategy
            )
            return HuggingFacePipeline(pipeline=pipe)


    def create_prompt_template(self, prompt: str, examples: List[Dict[str, str]]):
        temp = get_conversation_template(self.config.get('model_name', ''))
        if examples:
            instruction, user_input = prompt.split(':', maxsplit=1)
            examples = RandomExampleSelector(examples=examples, k=min(3, len(examples))).select_examples()
            fewshot_template = '\n\n'.join('\n'.join(f"{k.upper()}: {v}" for k, v in ex.items()) for ex in examples)
            user_input = f"{list(examples[0].keys())[0].upper()}: {user_input}\n"
            prompt = '\n\n'.join([instruction, fewshot_template, user_input])
        temp.append_message(temp.roles[0], prompt)
        temp.append_message(temp.roles[1], None)
        return PromptTemplate(template=temp.get_prompt(), input_variables=re.findall(r'\{(\w+)\}', prompt))


    def get_num_tokens(self, text: str):
        if self.config.get('provider', '') == "localhost":
            tokenizer = self.llm.pipeline.tokenizer
        else:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return len(tokenizer.encode(text))
