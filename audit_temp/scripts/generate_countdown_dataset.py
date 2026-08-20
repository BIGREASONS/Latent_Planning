import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = list(range(1, 11)) * 2

MIN_TARGET = 100
MAX_TARGET = 999


def _build_walk(is_ood: bool, max_target: int) -> Tuple[List[int], int, List[str]]:
    """One forward random walk. Division is integer-only; the walk is biased to
    stay bounded once it exceeds ``max_target``."""
    pool = random.sample(LARGE_NUMBERS, random.randint(1, 4))
    pool += random.sample(SMALL_NUMBERS, 6 - len(pool))

    # IID: 2-4 ops, OOD: exactly 5 ops
    num_ops = 5 if is_ood else random.randint(2, 4)
    selected = random.sample(pool, num_ops + 1)

    ops_history: List[str] = []
    current_val = selected[0]

    for val in selected[1:]:
        # Valid ops: division only when it yields a positive whole number.
        valid = ["+", "-", "*"]
        if val != 0 and current_val % val == 0 and current_val // val > 0:
            valid.append("/")

        # Keep the walk bounded: once large, prefer reducing ops so the final
        # target lands in a real Countdown-style range.
        if current_val > max_target:
            reduced = [o for o in valid if o in ("-", "/")]
            if reduced:
                valid = reduced

        op = random.choice(valid)

        if op == "+":
            new = current_val + val
            ops_history.append(f"{current_val} + {val} = {new}")
            current_val = new
        elif op == "*":
            new = current_val * val
            ops_history.append(f"{current_val} * {val} = {new}")
            current_val = new
        elif op == "/":
            new = current_val // val
            ops_history.append(f"{current_val} / {val} = {new}")
            current_val = new
        else:  # '-' : keep the result strictly positive
            if current_val - val > 0:
                a, b, new = current_val, val, current_val - val
            else:
                a, b, new = val, current_val, val - current_val
            ops_history.append(f"{a} - {b} = {new}")
            current_val = new

    return pool, current_val, ops_history


def generate_countdown_problem(
    is_ood: bool = False,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
    max_attempts: int = 4000,
) -> Dict[str, Any]:
    """Generates a Countdown numbers-game problem with a bounded integer target.

    Uses rejection sampling so the final target falls in ``[min_target, max_target]``.
    Operations are +, -, *, and integer / (remainder-free, matching apply_op)."""
    pool, target, ops_history = _build_walk(is_ood, max_target)
    attempts = 1
    while not (min_target <= target <= max_target) and attempts < max_attempts:
        pool, target, ops_history = _build_walk(is_ood, max_target)
        attempts += 1

    cot = "\n".join(ops_history)
    return {
        "numbers": pool,
        "target": target,
        "solution": ops_history,
        "cot": cot,
    }


def generate_dataset(
    num_samples: int,
    output_file: str,
    seed: int = 0,
    is_ood: bool = False,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
):
    """Generates a dataset of Countdown problems and saves to JSONL."""
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            problem = generate_countdown_problem(
                is_ood=is_ood, min_target=min_target, max_target=max_target
            )
            f.write(json.dumps(problem) + "\n")


def generate_multi_solution_dataset(
    num_samples: int,
    output_file: str,
    solutions_per_problem: int = 4,
    seed: int = 0,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
    max_attempts_per_solution: int = 200,
):
    """Generate Countdown problems each carrying several DISTINCT valid solutions.

    Used by the V5 permutation-robustness analysis: encoding each solution as a
    separate hidden-state trajectory lets us test whether *equivalent* symbolic
    states (reached via different reasoning orders) map to the *same* discrete
    code. Output records carry a ``solutions: List[List[str]]`` field (one list
    of step strings per distinct solution) in addition to the canonical
    ``solution`` / ``cot`` fields (set to the first solution for back-compat).

    Solutions are deduplicated by their canonical step-string tuple, so the
    returned list may be shorter than ``solutions_per_problem`` if fewer
    distinct walks are found within the attempt budget.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            seen: set = set()
            solutions: List[List[str]] = []
            last_pool, last_target = None, None
            for _ in range(max_attempts_per_solution):
                if len(solutions) >= solutions_per_problem:
                    break
                pool, target, ops_history = _build_walk(False, max_target)
                if not (min_target <= target <= max_target):
                    continue
                key = tuple(ops_history)
                if key in seen:
                    continue
                seen.add(key)
                solutions.append(ops_history)
                last_pool, last_target = pool, target

            if not solutions:
                continue
            record = {
                "numbers": last_pool,
                "target": last_target,
                "solutions": solutions,
                # Back-compat fields (first solution), so existing loaders work.
                "solution": solutions[0],
                "cot": "\n".join(solutions[0]),
            }
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Countdown Dataset")
    parser.add_argument(
        "--num_train", type=int, default=5000, help="Number of training samples"
    )
    parser.add_argument(
        "--num_val", type=int, default=500, help="Number of validation samples"
    )
    parser.add_argument(
        "--num_test", type=int, default=500, help="Number of testing samples"
    )
    parser.add_argument(
        "--num_test_ood", type=int, default=500, help="Number of OOD testing samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (each split is seeded deterministically off this)",
    )
    parser.add_argument(
        "--min_target",
        type=int,
        default=MIN_TARGET,
        help="Inclusive lower bound on the target",
    )
    parser.add_argument(
        "--max_target",
        type=int,
        default=MAX_TARGET,
        help="Inclusive upper bound on the target",
    )
    parser.add_argument(
        "--multi_solution",
        action="store_true",
        help="V5 permutation-robustness mode: write data/train_multi.jsonl with multiple distinct solutions per problem",
    )
    parser.add_argument(
        "--num_multi",
        type=int,
        default=2000,
        help="Number of multi-solution problems (only with --multi_solution)",
    )
    parser.add_argument(
        "--solutions_per_problem",
        type=int,
        default=4,
        help="Target number of distinct solutions per problem (only with --multi_solution)",
    )

    args = parser.parse_args()

    print(
        f"Generating datasets in {args.output_dir} (seed={args.seed}, target in [{args.min_target},{args.max_target}])..."
    )
    # Each split gets an independent, reproducible seed offset.
    generate_dataset(
        args.num_train,
        os.path.join(args.output_dir, "train.jsonl"),
        seed=args.seed,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    generate_dataset(
        args.num_val,
        os.path.join(args.output_dir, "val.jsonl"),
        seed=args.seed + 1,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    generate_dataset(
        args.num_test,
        os.path.join(args.output_dir, "test.jsonl"),
        seed=args.seed + 2,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    generate_dataset(
        args.num_test_ood,
        os.path.join(args.output_dir, "test_ood.jsonl"),
        seed=args.seed + 3,
        is_ood=True,
        min_target=args.min_target,
        max_target=args.max_target,
    )

    if args.multi_solution:
        print(
            f"Generating multi-solution dataset ({args.solutions_per_problem} solutions/problem)..."
        )
        generate_multi_solution_dataset(
            args.num_multi,
            os.path.join(args.output_dir, "train_multi.jsonl"),
            solutions_per_problem=args.solutions_per_problem,
            seed=args.seed + 4,
            min_target=args.min_target,
            max_target=args.max_target,
        )
    print("Done!")


if __name__ == "__main__":
    main()
