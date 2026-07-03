import os
import yaml
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_instruction(sample):
    """Formats the instruction for Mistral-7B."""
    prompt = f"[INST] {sample['instruction']}\n{sample['input']} [/INST] {sample['output']}</s>"
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/qlora_config.yaml")
    parser.add_argument("--dataset_path", type=str, default="../data/metamathqa_hindi.json")
    args = parser.parse_args()

    config = load_config(args.config)
    
    print("Loading dataset...")
    try:
        dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    except Exception as e:
        print(f"Failed to load dataset from {args.dataset_path}. Please run dataset preparation first.")
        return
    
    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = dataset['train']
    val_data = dataset['test']
    
    # Map formatting
    def generate_prompt(data_point):
        return {"text": format_instruction(data_point)}
        
    train_data = train_data.map(generate_prompt)
    val_data = val_data.map(generate_prompt)
    
    print("Initializing BitsAndBytes config for 4-bit loading...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['use_4bit'],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name_or_path'], 
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name_or_path'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False # Required for gradient checkpointing
    
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        
    model = prepare_model_for_kbit_training(model)
    
    print("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        optim=config['training']['optim'],
        save_strategy=config['training']['save_strategy'],
        eval_strategy=config['training']['evaluation_strategy'], # Use eval_strategy for newer transformers
        learning_rate=float(config['training']['learning_rate']),
        bf16=config['training']['bf16'],
        max_grad_norm=config['training']['max_grad_norm'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        logging_steps=config['training']['logging_steps'],
        group_by_length=True,
    )
    
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['training']['max_seq_length'],
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.model.save_pretrained(os.path.join(config['training']['output_dir'], "final_model"))
    tokenizer.save_pretrained(os.path.join(config['training']['output_dir'], "final_model"))
    
    print("Training complete!")

if __name__ == "__main__":
    main()
