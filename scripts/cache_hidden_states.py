import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Cache frozen hidden states for a dataset to avoid repeated HF model invocations."""

import argparse
import os
import sys

import torch
from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, build_trajectories, save_trajectories

def main():
    parser = argparse.ArgumentParser(description="Cache hidden states for trajectories")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--data_file", type=str, required=True, help="Input JSONL problems file")
    parser.add_argument("--output_file", type=str, required=True, help="Output .pt trajectories file")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden layer to extract")
    args = parser.parse_args()

    print(f"Loading problems from {args.data_file}...")
    problems = load_problems(args.data_file)
    
    print(f"Loading model {args.model}...")
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    
    print(f"Extracting hidden states for {len(problems)} problems...")
    trajectories = build_trajectories(model, tokenizer, problems, layer=args.layer)
    
    print(f"Successfully built {len(trajectories)} trajectories. Saving to {args.output_file}...")
    save_trajectories(trajectories, args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()
