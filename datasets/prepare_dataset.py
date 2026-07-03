import os
import json
import argparse
from datasets import load_dataset
from translator import MathTranslator

def main():
    parser = argparse.ArgumentParser(description="Prepare and translate MetaMathQA dataset.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of records to translate")
    parser.add_argument("--output_file", type=str, default="../data/metamathqa_hindi.json", help="Output JSON file")
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print("Loading MetaMathQA dataset...")
    try:
        dataset = load_dataset("meta-math/MetaMathQA", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # Shuffle and select a sample
    dataset = dataset.shuffle(seed=42).select(range(min(args.sample_size, len(dataset))))
    
    # Convert to list of dicts
    data = [item for item in dataset]
    
    print(f"Initializing translator...")
    translator = MathTranslator()
    
    print(f"Starting translation process for {len(data)} items...")
    translated_data = translator.process_dataset(data, limit=args.sample_size)
    
    print(f"Saving translated dataset to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
