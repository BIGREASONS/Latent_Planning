import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Evaluate teacher model accuracy on Countdown problems."""

import argparse
import os
import sys

import torch
from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, format_header
from data_processing.action_parser import parse_solution

def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher accuracy")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--data_file", type=str, default="data/val.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading problems from {args.data_file}...")
    problems = load_problems(args.data_file)[:args.limit]
    
    print(f"Loading model {args.model}...")
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    model.eval()
    
    correct = 0
    total = len(problems)
    
    print(f"Evaluating {total} problems...")
    for i, prob in enumerate(problems):
        header = format_header(prob["numbers"], prob["target"])
        inputs = tokenizer(header, return_tensors="pt").to(model.device)
        
        # We expect a solution to be reasonably short, e.g. < 100 tokens.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Parse lines
        lines = [line.strip() for line in gen_text.split("\n") if line.strip()]
        
        try:
            actions = parse_solution(lines)
            # The final result is usually the output of the last action.
            # In parse_solution, it doesn't return the result.
            # We can parse the last line manually: "A op B = C"
            if not lines:
                continue
            last_line = lines[-1]
            parts = last_line.split("=")
            if len(parts) == 2:
                result = float(parts[1].strip())
                if abs(result - prob["target"]) < 1e-5:
                    correct += 1
        except Exception:
            pass
            
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{total} | Current Accuracy: {correct / (i+1) * 100:.2f}%")

    accuracy = correct / total * 100
    print(f"\nFinal Teacher Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if accuracy < 80.0:
        print("WARNING: Teacher accuracy is below 80%. Hidden states may contain flawed reasoning.")

if __name__ == "__main__":
    main()
