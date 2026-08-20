"""Game-of-24 reasoning-trace generator for the V5.4 additional-domain study.

Game of 24: combine four numbers with +, -, *, / (each number used once) to
reach **24**. Solutions are emitted as the *same* ``a op b = c`` step format as
Countdown, so the existing :mod:`data_processing.action_parser` and
``build_trajectories`` pipeline consume them **unchanged** — every step yields a
real symbolic action (``op_id`` + operands), which is what Experiment A's
``MLP(z,a) − ActionBigram(z,a)`` contrast requires. (A free-form domain like
GSM8K would have no action grammar and would collapse that contrast.)

To stay compatible with the Countdown parser/``apply_op`` we restrict to
**integer-only** intermediate values: division must be remainder-free and every
intermediate result is a positive integer. We therefore generate only puzzles
that have an integer-only solution (the classic game also allows fractions; we
intentionally exclude those for pipeline compatibility) and emit one such
solution as the reasoning trace.

Record schema matches the Countdown generator exactly::

    {"numbers": [...], "target": 24, "solution": ["a op b = c", ...], "cot": "..."}

Pool statistics (cards 1–13, integer-only, positive intermediates)::

    Total unique 4-card multisets: 1,820  (= C(16,4))
    Solvable:                      1,346  (74.0 %)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

TARGET = 24
CARD_MIN, CARD_MAX = 1, 13  # classic 4-card range
NUM_CARDS = 4

# (value, steps) item: a number on the table plus the steps that produced it.
Item = Tuple[int, List[str]]


def _apply(sym: str, a: int, b: int) -> Optional[int]:
    """Integer op mirroring action_parser.apply_op; None if illegal here.

    Results must stay positive integers (matches the Countdown distribution and
    keeps `a op b = c` parseable with positive operands)."""
    if sym == "+":
        return a + b
    if sym == "-":
        return a - b if a - b > 0 else None
    if sym == "*":
        return a * b
    if sym == "/":
        if b != 0 and a % b == 0 and a // b > 0:
            return a // b
        return None
    return None


def _solve(items: List[Item], target: int) -> Optional[List[str]]:
    """Return a valid ordered step list reducing ``items`` to ``target``, or None.

    Recursively picks two table values, combines them with an op, and recurses
    on the smaller multiset. Step ordering (left subtree, right subtree, then the
    combine) is a valid linearization because both operands are produced before
    they are combined."""
    if len(items) == 1:
        return items[0][1] if items[0][0] == target else None
    n = len(items)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_val, a_steps = items[i]
            b_val, b_steps = items[j]
            rest = [items[k] for k in range(n) if k != i and k != j]
            for sym in ("+", "-", "*", "/"):
                res = _apply(sym, a_val, b_val)
                if res is None:
                    continue
                step = f"{a_val} {sym} {b_val} = {res}"
                merged = a_steps + b_steps + [step]
                sol = _solve(rest + [(res, merged)], target)
                if sol is not None:
                    return sol
    return None


# --------------------------------------------------------------------------- #
# Exhaustive solvable-pool enumeration
# --------------------------------------------------------------------------- #


def enumerate_solvable_pool() -> List[Tuple[int, ...]]:
    """Return every unique solvable 4-card multiset from {1..13}.

    A hand is 'solvable' if the four numbers can be combined with +, -, *, /
    (each used exactly once, integer-only positive intermediates) to reach 24.

    Returns a sorted list of tuples for deterministic ordering.
    """
    pool: List[Tuple[int, ...]] = []
    for hand in combinations_with_replacement(range(CARD_MIN, CARD_MAX + 1), NUM_CARDS):
        items = [(n, []) for n in hand]
        if _solve(items, TARGET) is not None:
            pool.append(hand)
    return pool


def generate_game24_problem(
    rng: random.Random, max_attempts: int = 2000
) -> Dict[str, Any]:
    """Sample four cards with an integer-only solution to 24; return the record.

    This function is retained for backward compatibility with existing tests and
    the smoke_4bit_extraction script. For dataset generation, prefer
    :func:`generate_dataset` which uses exhaustive pool partitioning."""
    for _ in range(max_attempts):
        numbers = [rng.randint(CARD_MIN, CARD_MAX) for _ in range(NUM_CARDS)]
        sol = _solve([(n, []) for n in numbers], TARGET)
        if sol is not None:
            return {
                "numbers": numbers,
                "target": TARGET,
                "solution": sol,
                "cot": "\n".join(sol),
            }
    raise RuntimeError("No solvable Game-of-24 hand found within attempt budget.")


def _hand_to_record(hand: Tuple[int, ...], rng: random.Random) -> Dict[str, Any]:
    """Convert a canonical hand tuple to a problem record with a solved trace.

    The hand is randomly permuted so the model doesn't see cards in sorted order
    (which would be an artificial regularity absent from the real game)."""
    numbers = list(hand)
    rng.shuffle(numbers)
    sol = _solve([(n, []) for n in numbers], TARGET)
    assert sol is not None, f"hand {hand} was in solvable pool but solver returned None"
    return {
        "numbers": numbers,
        "target": TARGET,
        "solution": sol,
        "cot": "\n".join(sol),
    }


def generate_dataset(
    num_samples: int,
    output_file: str,
    seed: int = 0,
    is_ood: bool = False,
) -> None:
    """Generate ``num_samples`` unique Game-of-24 problems and save to JSONL.

    Problems are drawn **without replacement** from the exhaustively enumerated
    pool of 1,346 solvable hands. ``num_samples`` must not exceed the pool size.

    Parameters
    ----------
    is_ood : bool
        Accepted for API compatibility with the Countdown generator. Game24 has
        a fixed card range {1..13} with no meaningful OOD variant; if ``True``,
        a warning is emitted and the parameter is otherwise ignored.
    """
    if is_ood:
        warnings.warn(
            "Game24 has a fixed card range {1..13}; is_ood=True is accepted "
            "for API compatibility but has no effect.",
            stacklevel=2,
        )
    pool = enumerate_solvable_pool()
    if num_samples > len(pool):
        raise ValueError(
            f"Requested {num_samples} unique Game24 hands but only "
            f"{len(pool)} solvable hands exist. Reduce num_samples or use "
            f"the Countdown domain for larger datasets."
        )
    rng = random.Random(seed)
    selected = list(pool)
    rng.shuffle(selected)
    selected = selected[:num_samples]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for hand in selected:
            f.write(json.dumps(_hand_to_record(hand, rng)) + "\n")


def generate_partitioned_splits(
    output_dir: str,
    seed: int = 0,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, int]:
    """Partition the full solvable pool into mutually exclusive train/val/test.

    Returns a dict with the actual split sizes.
    """
    pool = enumerate_solvable_pool()
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets the remainder — guarantees no rounding loss
    train_hands = shuffled[:n_train]
    val_hands = shuffled[n_train : n_train + n_val]
    test_hands = shuffled[n_train + n_val :]

    os.makedirs(output_dir, exist_ok=True)
    sizes = {}
    for split_name, hands in [
        ("train", train_hands),
        ("val", val_hands),
        ("test", test_hands),
    ]:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for hand in hands:
                f.write(json.dumps(_hand_to_record(hand, rng)) + "\n")
        sizes[split_name] = len(hands)

    return sizes


def main():
    parser = argparse.ArgumentParser(description="Generate Game-of-24 dataset")
    parser.add_argument("--output_dir", type=str, default="data/game24")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of solvable pool for training (default 0.8)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of solvable pool for validation (default 0.1)",
    )
    args = parser.parse_args()

    pool = enumerate_solvable_pool()
    print(
        f"Game-of-24 solvable pool: {len(pool)} unique hands "
        f"(out of 1820 multisets = {len(pool)/1820:.1%})"
    )

    sizes = generate_partitioned_splits(
        args.output_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    print(f"Partitioned into mutually exclusive splits:")
    for split, n in sizes.items():
        print(f"  {split}: {n}")
    print("Done!")


if __name__ == "__main__":
    main()
