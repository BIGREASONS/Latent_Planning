import os
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import load_checkpoint, load_model, load_tokenizer

def extract_hidden_states(model, tokenizer, dataset, target_layer: int, output_file: str):
    """Extracts and saves hidden states from a specified layer."""
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    extracted_data = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Extracting hidden states")):
            # Format prompt based on dataset structure
            prompt = f"Problem: Given the numbers {sample['numbers']}, reach the target {sample['target']}.\nSolution:\n{sample['cot']}"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Forward pass asking for hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # hidden_states is a tuple of (layers + 1)
            # 0 is the embedding layer, 1 to L are transformer layers
            layer_hidden_states = outputs.hidden_states[target_layer].squeeze(0) # Shape: (seq_len, hidden_size)
            
            # Store per-token hidden states
            for pos, token_id in enumerate(inputs.input_ids[0]):
                extracted_data.append({
                    "sample_id": i,
                    "position": pos,
                    "token_id": token_id.item(),
                    "token_str": tokenizer.decode(token_id),
                    # Storing tensor as list for JSON serialization (could use torch.save for pure tensors later)
                    "hidden_state": layer_hidden_states[pos].cpu().float().tolist()
                })
                
    # Save to file
    # Note: For large datasets, torch.save(list_of_dicts, output_file.pt) is much more efficient than JSON.
    # We will use torch.save if the output file ends in .pt
    if output_file.endswith(".pt"):
        torch.save(extracted_data, output_file)
    else:
        with open(output_file, 'w') as f:
            for item in extracted_data:
                f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract Hidden States")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint (or None for base model)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_file", type=str, default="reports/hidden_states.pt", help="Output file path (.pt recommended)")
    parser.add_argument("--layer", type=int, default=-1, help="Layer number to extract (-1 for last layer)")
    args = parser.parse_args()

    # Load Model
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model, tokenizer = load_checkpoint(args.model_path)
    else:
        print("Loading base model...")
        model = load_model()
        tokenizer = load_tokenizer()

    # Load Dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset = load_dataset("json", data_files={"test": args.data_file})["test"]

    print(f"Extracting hidden states from layer {args.layer}...")
    extract_hidden_states(model, tokenizer, dataset, args.layer, args.output_file)
    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()
