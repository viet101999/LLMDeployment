from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the router
router = APIRouter()

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bigscience/bloomz-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,
    device_map="auto",
)
quantized_model.eval()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

class CompletionResponse(BaseModel):
    completion: str
    logprobs: list = None

@router.post("/generate", response_model=CompletionResponse)
async def generate(request: CompletionRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        outputs = quantized_model.generate(
            inputs["input_ids"],
            max_length=request.max_tokens + len(inputs["input_ids"][0]),
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return CompletionResponse(completion=output_text, logprobs=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
