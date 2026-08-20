import os
import json
import random
import argparse
from typing import List, Dict, Any


def generate_countdown_problem() -> Dict[str, Any]:
    """Generates a simple Countdown numbers game problem."""
    # Choose 6 random numbers (traditionally 1-4 large, rest small, but we simplify)
    large_numbers = [25, 50, 75, 100]
    small_numbers = list(range(1, 11)) * 2

    # Pick randomly
    pool = random.sample(large_numbers, random.randint(1, 4))
    pool += random.sample(small_numbers, 6 - len(pool))

    # For now, we will generate a target that is strictly reachable by simple ops
    # to guarantee a valid solution for the dataset.
    num_ops = random.randint(2, 5)
    selected = random.sample(pool, num_ops + 1)

    target = selected[0]
    ops_history = []
    current_val = target

    for val in selected[1:]:
        op = random.choice(["+", "-", "*"])
        if op == "+":
            ops_history.append(f"{current_val} + {val} = {current_val + val}")
            current_val += val
        elif op == "-":
            # Avoid negative targets for simplicity
            if current_val - val > 0:
                ops_history.append(f"{current_val} - {val} = {current_val - val}")
                current_val -= val
            else:
                ops_history.append(f"{val} - {current_val} = {val - current_val}")
                current_val = val - current_val
        elif op == "*":
            ops_history.append(f"{current_val} * {val} = {current_val * val}")
            current_val *= val

    # The final current_val is our target
    target = current_val
    cot = "\n".join(ops_history)

    return {"numbers": pool, "target": target, "solution": ops_history, "cot": cot}


def generate_dataset(num_samples: int, output_file: str):
    """Generates a dataset of Countdown problems and saves to JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            problem = generate_countdown_problem()
            f.write(json.dumps(problem) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Countdown Dataset")
    parser.add_argument(
        "--num_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--num_val", type=int, default=100, help="Number of validation samples"
    )
    parser.add_argument(
        "--num_test", type=int, default=100, help="Number of testing samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )

    args = parser.parse_args()

    print(f"Generating datasets in {args.output_dir}...")
    generate_dataset(args.num_train, os.path.join(args.output_dir, "train.jsonl"))
    generate_dataset(args.num_val, os.path.join(args.output_dir, "val.jsonl"))
    generate_dataset(args.num_test, os.path.join(args.output_dir, "test.jsonl"))
    print("Done!")


if __name__ == "__main__":
    main()
