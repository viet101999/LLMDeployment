from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import time
import psutil
import os
import numpy as np
# import deepspeed
import torch.nn.functional as F
import evaluate
from prometheus_client import start_http_server, Summary, Gauge, Counter, generate_latest
from starlette.responses import Response
import threading

from common.common_keys import *
from data_model.model import llm
from modules.base import BaseModule


class ModelLoad(BaseModule):
    def __init__(self, config_app: dict):
        super(ModelLoad, self).__init__()

        self.device = config_app[DEVICE] if torch.cuda.is_available() else "cpu"

        self.model_name = config_app[LLM][MODEL_NAME]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enables 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for 4-bit computation
            bnb_4bit_use_double_quant=True,  # Enables double quantization for 4-bit
            bnb_4bit_quant_type="nf4",  # Uses "nf4" quantization type
        )

        self.logger.info(f"Loading {self.model_name} model...")
        start = time.time()

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.llm_model.eval()
        self.logger.info(f"{self.model_name} model loaded successfully in %fs" % (time.time() - start))

    def generate_text(self, prompt: str, max_length: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate output tokens
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                # temperature=0.01,  # Sampling temperature
                # top_p=1.0,  # Top-p (nucleus) sampling
                # top_k=1,  # Top-k sampling
                # repetition_penalty=1.0,
                # length_penalty=1.0,
                # use_beam_search=False,
                # min_tokens=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # do_sample=True,
                # top_k=50, 
                # top_p=0.9,
            )

        # Decode and calculate log probabilities

        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # input_token_logprobs = logprobs_from_prompt(request.prompt, tokenizer, model)
        # output_token_logprobs = logprobs_from_prompt(output_text, tokenizer, model)

        # input_ids = inputs["input_ids"]
        # output = model(input_ids=input_ids)

        # # neglecting the first token, since we make no prediction about it
        # shift_labels = input_ids[..., 1:].contiguous()
        # shift_logits = output.logits[..., :-1, :].contiguous()

        # for label_id,logit in zip(shift_labels[0].tolist(), shift_logits[0]):
        #     logprob = F.log_softmax(logit, dim=0).tolist()[label_id]
        #     print(tokenizer.decode(label_id)," : ", logprob)
        # output = {
        #     "generated_text": output_text, 
        #     # "input_token_logprobs": input_token_logprobs, 
        #     # "output_token_logprobs": output_token_logprobs
        # }
        return output
        
    def measure_speed(self, prompt, num_iterations=10):
        """Measures inference speed in tokens per second."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        total_tokens = 0
        total_time = 0

        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.llm_model.generate(**input_ids, max_new_tokens=100)
            end_time = time.time()
            total_time += (end_time - start_time)
            total_tokens += outputs.shape[1]

        avg_time_per_iteration = total_time / num_iterations
        tokens_per_second = total_tokens / total_time
        return tokens_per_second, avg_time_per_iteration