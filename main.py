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
            outputs = model.generate(**input_ids, max_new_tokens=100)
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
  
def evaluate_performance(model, quantized_model, tokenizer, prompts):
    """Evaluates performance by comparing generated text to expected outputs."""
    correct = 0
    quantized_correct = 0
    total = len(prompts)
    for prompt, expected_output in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=len(expected_output.split()))
            quantized_generated_ids = quantized_model.generate(**input_ids, max_new_tokens=len(expected_output.split()))
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        quantized_generated_text = tokenizer.decode(quantized_generated_ids[0], skip_special_tokens=True)
        if expected_output.lower() in generated_text.lower(): # Basic string matching for evaluation
            correct += 1
        if expected_output.lower() in quantized_generated_text.lower(): # Basic string matching for evaluation
            quantized_correct += 1
    accuracy = (correct / total) * 100 if total > 0 else 0
    quantized_accuracy = (quantized_correct / total) * 100 if total > 0 else 0
    return {"accuracy": accuracy, "quantized_accuracy": quantized_accuracy}

def eval_summary(references: list, predictions: list):
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    # BLEU expects plain text inputs
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_results['bleu'] * 100:.2f}")
    # 
    # ROUGE expects plain text inputs
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    # Access ROUGE scores (no need for indexing into the result)
    print(f"ROUGE-1 F1 Score: {rouge_results['rouge1']:.2f}")
    print(f"ROUGE-L F1 Score: {rouge_results['rougeL']:.2f}")
    return {"bleu": bleu_results, "rouge": rouge_results}
    
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
async def generate(request: CompletionRequest, model):
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

# # Metrics
# vram_usage_gauge = Gauge('vram_usage_gb', 'VRAM usage in GB')
# cpu_ram_usage_gauge = Gauge('cpu_ram_usage_gb', 'CPU RAM usage in GB')
# inference_speed_gauge = Gauge('inference_speed_tokens_per_second', 'Inference speed in tokens per second')
# requests_counter = Counter('http_requests_total', 'Total HTTP requests received')
# request_duration = Summary('request_duration_seconds', 'Request duration in seconds')

# # Expose metrics endpoint
# @app.get("/metrics")
# def metrics():
#     return Response(generate_latest(), media_type="text/plain")

# # Start Prometheus metrics server in a separate thread
# def start_metrics_server():
#     start_http_server(8001)  # Expose metrics on a separate port
# threading.Thread(target=start_metrics_server, daemon=True).start()

# @app.middleware("http")
# async def count_requests(request: Request, call_next):
#     requests_counter.inc()
#     with request_duration.time():
#         response = await call_next(request)
#     return response

# @app.on_event("startup")
# async def monitor_resources():
#     while True:
#         # Update VRAM and CPU usage
#         vram_usage = get_vram_usage()
#         cpu_ram_usage = get_cpu_ram_usage()
#         if vram_usage is not None:
#             vram_usage_gauge.set(vram_usage)
#         cpu_ram_usage_gauge.set(cpu_ram_usage)

#         # Sleep for 5 seconds before updating metrics again
#         time.sleep(5)
        
def generate_text(request: CompletionRequest, model, quantized_model):
    try:
        # Tokenize the input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True)
        # Move input tensors to the appropriate device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate output tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                temperature=0.01,  # Sampling temperature
                top_p=1.0,  # Top-p (nucleus) sampling
                top_k=1,  # Top-k sampling
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
                length_penalty=1.0,
                # do_sample=True,
                # top_k=50, 
                # top_p=0.9,
                # temperature=request.temperature
            )
        
            quantized_outputs = quantized_model.generate(
                **inputs,
                max_length=1024,
                temperature=0.01,  # Sampling temperature
                top_p=1.0,  # Top-p (nucleus) sampling
                top_k=1,  # Top-k sampling
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
                length_penalty=1.0,
            )

        # Decode and calculate log probabilities

        # output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        quantized_output_text = tokenizer.batch_decode(quantized_outputs, skip_special_tokens=True)[0]

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
        # return output
        print("DONE")
    except Exception as e:
        print(str(e))

def load_transcript(transcript_path: str):
    with open(transcript_path, 'r') as r:
        data_all = r.readlines()
        video_metadata = []
        metadata = []
        for data in data_all:
            start = data.split("]  ")[0].split(" - ")[0].replace("[", "").strip()
            end = data.split("]  ")[0].split(" - ")[-1].replace("]", "").strip()
            text = data.split("]  ")[-1].strip()
            segment = {
                "start": start,
                "end": end,
                "text": text
            }
            metadata.append(segment)

        video_metadata.extend(metadata)
    return video_metadata
            
# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enables 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for 4-bit computation
    bnb_4bit_use_double_quant=True,  # Enables double quantization for 4-bit
    bnb_4bit_quant_type="nf4",  # Uses "nf4" quantization type
)

# Load model and tokenizer
model_name = "bigscience/bloomz-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically map to GPU
)
quantized_model.eval()

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto", 
#     device_map="auto"
# )
# model.eval()

# Example usage
prompts_for_eval = [
    ("Translate 'Hello, how are you?' to French.", "Bonjour, comment allez-vous ?"),
    ("Write a short story about a robot learning to love.", "A robot named R-3X") # Partial match example
]

# question = "what is the prize?"
# transcript_path = "/TMTAI/AI_MemBer/workspace/vietdh/llm_deployment/survive100daystrappedwin500000.txt"
# video_metadata = load_transcript(transcript_path)
# transcript = " ".join([f"[{entry['start']} - {entry['end']}] {entry['text']}" for entry in video_metadata])
# content = "Below is a transcript of a video/audio\n{transcript}\n\nplease answer the user's question as accurately and concisely as possible\nQuestion: {query}"
# prompt = content.replace("{transcript}", transcript).replace("{query}", question)
story = 'Martin Luther King Jr., the renowned civil rights leader, was assassinated on April 4, 1968, at the Lorraine Motel in Memphis, Tennessee. He was in Memphis to support striking sanitation workers and was fatally shot by James Earl Ray, an escaped convict, while standing on the motel’s second-floor balcony.' 
prompt = f'This is a story {story}. Who is Martin Luther King Jr.\n'

# Measure VRAM usage
vram_before = get_vram_usage()
ram_before = get_cpu_ram_usage()
print(f"VRAM usage before inference: {vram_before:.2f} GB")
print(f"RAM usage before inference: {ram_before:.2f} GB")

# Measure Speed
tokens_per_second, avg_time = measure_speed(quantized_model, tokenizer, prompt)
print(f"Inference speed: {tokens_per_second:.2f} tokens/second")
print(f"Average time per iteration: {avg_time:.4f} seconds")

# Calculate Log Probabilities
# log_probs = calculate_log_probs(model, tokenizer, prompt)

# request = CompletionRequest(prompt=prompt, max_tokens=100, temperature=0.7)
# output = generate_text(request, model, quantized_model)
# if output["output_token_logprobs"] is not None:
#     print(f"Shape of log probabilities: {output['output_token_logprobs'].shape}")
#     print(f"Example log probabilities: {output['output_token_logprobs'][0, :5, :5]}") # Print first 5 tokens and first 5 possible next token
# else:
#     print("Log probability calculation failed")

# Evaluate Performance
# accuracy = evaluate_performance(model, quantized_model, tokenizer, prompts_for_eval)
# print(f"Performance accuracy: {accuracy['accuracy']:.2f}%")
# print(f"Performance accuracy: {accuracy['quantized_accuracy']:.2f}%")

vram_after = get_vram_usage()
ram_after = get_cpu_ram_usage()
print(f"VRAM usage after inference: {vram_after:.2f} GB")
print(f"RAM usage after inference: {ram_after:.2f} GB")

print(f"VRAM peak usage : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB") if torch.cuda.is_available() else print("No CUDA device detected")
print(f"RAM peak usage : {psutil.Process(os.getpid()).memory_full_info().rss / (1024 ** 3):.2f} GB")


# request = CompletionRequest(prompt="Hello", max_tokens=50, temperature=0.7)
# generate(request)