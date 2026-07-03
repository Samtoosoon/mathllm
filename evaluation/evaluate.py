import os
import json
import torch
import argparse
import evaluate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_answer(text):
    """
    Very basic heuristic to extract the final answer from math solving steps.
    Assuming the final answer is after 'उत्तर' or at the end.
    """
    if "उत्तर" in text:
        return text.split("उत्तर")[-1].strip()
    return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../data/metamathqa_hindi.json")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--peft_model", type=str, default="../checkpoints/mistral-math-hi/final_model")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    print("Loading metrics...")
    try:
        rouge = evaluate.load('rouge')
        bleu = evaluate.load('bleu')
        exact_match = evaluate.load('exact_match')
    except Exception as e:
        print(f"Warning: Ensure 'evaluate', 'rouge_score', and 'sacrebleu' are installed. Error: {e}")
        return
    
    print("Loading data...")
    try:
        data = load_data(args.test_data)
        # Assuming we evaluate on a subset of the test data for speed
        test_samples = data[-args.num_samples:] 
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, args.peft_model)
        model.eval()
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        return

    predictions = []
    references = []
    exact_predictions = []
    exact_references = []
    
    print(f"Evaluating {len(test_samples)} samples...")
    for sample in tqdm(test_samples):
        prompt = f"[INST] {sample['instruction']}\n{sample['input']} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        predictions.append(generated_text)
        references.append(sample['output'])
        
        # Heuristic exact match
        exact_predictions.append(extract_answer(generated_text))
        exact_references.append(extract_answer(sample['output']))
        
    print("Calculating metrics...")
    
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    em_results = exact_match.compute(predictions=exact_predictions, references=exact_references)
    
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"ROUGE: {rouge_results}")
    print(f"BLEU: {bleu_results}")
    print(f"Exact Match (Heuristic): {em_results['exact_match']}")
    print("="*30)
    
    with open("eval_results.json", "w") as f:
        json.dump({
            "rouge": rouge_results,
            "bleu": bleu_results,
            "exact_match": em_results['exact_match']
        }, f, indent=2)

if __name__ == "__main__":
    main()
