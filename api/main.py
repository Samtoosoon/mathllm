from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
import os

app = FastAPI(title="Math LLM for Hindi Medium Students API")

# Setup Global models
tokenizer = None
model = None

class MathRequest(BaseModel):
    question: str
    temperature: float = 0.2
    max_new_tokens: int = 512

class MathResponse(BaseModel):
    solution: str
    steps: list[str]
    confidence: float

@app.on_event("startup")
def load_models():
    global tokenizer, model
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    peft_model_id = os.path.join(os.path.dirname(__file__), "../checkpoints/mistral-math-hi/final_model")
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Loading PEFT adapters...")
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load PEFT model ({e}). Ensure training was completed. Running with base model for now.")
        # Fallback for dev purposes if checkpoint not yet created
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()

def parse_steps(solution_text: str) -> list[str]:
    """Basic parser to extract steps separated by line breaks or step indicators."""
    steps = solution_text.split('\n')
    steps = [s.strip() for s in steps if len(s.strip()) > 5]
    return steps

@app.post("/solve", response_model=MathResponse)
async def solve_math_problem(req: MathRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load.")
        
    prompt = f"[INST] नीचे दिए गए गणित के प्रश्न को हल करें:\n{req.question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Mock confidence score
    confidence = 0.95
    steps = parse_steps(generated_text)
    
    return MathResponse(
        solution=generated_text.strip(),
        steps=steps,
        confidence=confidence
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
