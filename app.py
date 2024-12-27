from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import time
import psutil
import os
import numpy as np
import deepspeed
import torch.nn.functional as F

def get_vram_usage():
    """Returns VRAM usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        return vram
    else:
        return None

def get_cpu_ram_usage():
    """Returns CPU RAM usage in GB."""
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / (1024 ** 3) # Convert to GB
    return ram

def measure_speed(model, tokenizer, prompt, num_iterations=10):
    """Measures inference speed in tokens per second."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    total_tokens = 0
    total_time = 0

    for _ in range(num_iterations):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**input_ids)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_tokens += outputs.shape[1]

    avg_time_per_iteration = total_time / num_iterations
    tokens_per_second = total_tokens / total_time
    return tokens_per_second, avg_time_per_iteration

def calculate_log_probs(model, tokenizer, prompt):
    """Calculates log probabilities of generated tokens."""
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids) # labels are needed to get loss/logits
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        # Get log probs of generated tokens (excluding padding and input tokens)
        generated_token_log_probs = log_probs[0, input_ids.shape[1]-1:-1, :] #Select only generated token
        return generated_token_log_probs

def logprobs_from_prompt(prompt, tokenizer, model):
      encoded = tokenizer(prompt, return_tensors="pt").to("cpu")
      input_ids = encoded["input_ids"]
      output = model(input_ids=input_ids)
      shift_labels = input_ids[..., 1:].contiguous()
      shift_logits = output.logits[..., :-1, :].contiguous()
      log_probs = []
      log_probs.append((tokenizer.decode(input_ids[0].tolist()[0]), None))
      for idx, (label_id, logit) in enumerate(zip(shift_labels[0].tolist(), shift_logits[0])):
            logprob = F.log_softmax(logit, dim=0).tolist()[label_id]
            log_probs.append((tokenizer.decode(label_id), float(logprob)))
      return log_probs
  
def evaluate_performance(model, tokenizer, prompts):
    """Evaluates performance by comparing generated text to expected outputs."""
    correct = 0
    total = len(prompts)
    for prompt, expected_output in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=len(expected_output.split()))
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if expected_output.lower() in generated_text.lower(): # Basic string matching for evaluation
            correct += 1
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Define request and response schemas
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

class CompletionResponse(BaseModel):
    completion: str
    logprobs: list = None

# Initialize FastAPI
app = FastAPI()

@app.post("/generate", response_model=CompletionResponse)
async def generate(request: CompletionRequest):
    try:
        # Tokenize the input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

        # Generate output tokens
        outputs = model.generate(
            inputs["input_ids"],
            max_length=request.max_tokens + len(inputs["input_ids"][0]),
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode and calculate log probabilities
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        token_logprobs = torch.nn.functional.log_softmax(model(inputs["input_ids"]).logits, dim=-1)
        
        return CompletionResponse(
            completion=output_text,
            logprobs=token_logprobs.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_text(request: CompletionRequest):
    try:
        # Tokenize the input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)

        # Generate output tokens
        outputs = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=request.max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # do_sample=True,
            # top_k=50, 
            # top_p=0.9,
            # temperature=request.temperature
        )

        # Decode and calculate log probabilities

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

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
        output = {
            "generated_text": output_text, 
            # "input_token_logprobs": input_token_logprobs, 
            # "output_token_logprobs": output_token_logprobs
        }
        return output
    except Exception as e:
        print(str(e))

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
model_name = "bigscience/bloomz-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",  # Automatically map to GPU
    # load_in_8bit=True,  # Use 4-bit quantization
)
model.eval()

# Example usage
prompts_for_eval = [
    ("Translate 'Hello, how are you?' to French.", "Bonjour, comment allez-vous ?"),
    ("Write a short story about a robot learning to love.", "A robot named R-3X") # Partial match example
]

prompt = "Write a short story about a robot learning to love."

# Measure VRAM usage
# vram_before = get_vram_usage()
# ram_before = get_cpu_ram_usage()
# print(f"VRAM usage before inference: {vram_before:.2f} GB")
# print(f"RAM usage before inference: {ram_before:.2f} GB")

# Measure Speed
tokens_per_second, avg_time = measure_speed(model, tokenizer, prompt)
print(f"Inference speed: {tokens_per_second:.2f} tokens/second")
print(f"Average time per iteration: {avg_time:.4f} seconds")

# Calculate Log Probabilities
# log_probs = calculate_log_probs(model, tokenizer, prompt)

request = CompletionRequest(prompt=prompt, max_tokens=789, temperature=0.7)
output = generate_text(request)
# if output["log_probs"] is not None:
#     print(f"Shape of log probabilities: {output['log_probs'].shape}")
#     print(f"Example log probabilities: {output['log_probs'][0, :5, :5]}") # Print first 5 tokens and first 5 possible next token
# else:
#     print("Log probability calculation failed")

# Evaluate Performance
# accuracy = evaluate_performance(model, tokenizer, prompts_for_eval)
# print(f"Performance accuracy: {accuracy:.2f}%")

# vram_after = get_vram_usage()
# ram_after = get_cpu_ram_usage()
# print(f"VRAM usage after inference: {vram_after:.2f} GB")
# print(f"RAM usage after inference: {ram_after:.2f} GB")

# print(f"VRAM peak usage : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB") if torch.cuda.is_available() else print("No CUDA device detected")
# print(f"RAM peak usage : {psutil.Process(os.getpid()).memory_full_info().rss / (1024 ** 3):.2f} GB")


# request = CompletionRequest(prompt="Hello", max_tokens=50, temperature=0.7)
# generate(request)