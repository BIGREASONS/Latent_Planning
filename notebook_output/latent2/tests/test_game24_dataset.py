"""Game-of-24 generator: every emitted solution must be a valid, parseable,
integer-only reduction of the four numbers to 24 — the contract the downstream
Countdown pipeline (action_parser + build_trajectories) relies on.

Also verifies that the exhaustive pool partitioning produces zero-overlap splits
and that the pool is complete (every solvable hand is included)."""

import json
import os
import random
import warnings
from collections import Counter
from itertools import combinations_with_replacement

from data_processing.action_parser import parse_step
from scripts.generate_game24_dataset import (
    _solve,
    enumerate_solvable_pool,
    generate_dataset,
    generate_game24_problem,
    generate_partitioned_splits,
    CARD_MIN,
    CARD_MAX,
    TARGET,
)


def _simulate(numbers, actions):
    """Replay the steps as a multiset reduction; return the final lone value."""
    available = Counter(numbers)
    for a in actions:
        assert available[a.arg1] > 0, f"operand {a.arg1} not on the table"
        available[a.arg1] -= 1
        assert available[a.arg2] > 0, f"operand {a.arg2} not on the table"
        available[a.arg2] -= 1
        available[a.result] += 1
    remaining = list(available.elements())
    assert len(remaining) == 1, f"expected one value left, got {remaining}"
    return remaining[0]


def test_generated_problems_are_valid():
    rng = random.Random(0)
    for _ in range(200):
        prob = generate_game24_problem(rng)
        assert prob["target"] == 24
        assert len(prob["numbers"]) == 4
        # Every step parses AND is arithmetically valid (validate=True raises otherwise).
        actions = [parse_step(s, validate=True) for s in prob["solution"]]
        assert len(actions) == 3  # four numbers -> three binary ops
        assert _simulate(prob["numbers"], actions) == 24


def test_determinism():
    a = generate_game24_problem(random.Random(42))
    b = generate_game24_problem(random.Random(42))
    assert a == b


# --------------------------------------------------------------------------- #
# Pool enumeration tests
# --------------------------------------------------------------------------- #


def test_pool_exhaustive():
    """Every solvable multiset in {1..13}^4 is in the pool, and no unsolvable
    hand is included."""
    pool = enumerate_solvable_pool()
    pool_set = set(pool)

    # Pool elements must be sorted tuples (canonical form)
    for hand in pool:
        assert hand == tuple(sorted(hand)), f"hand {hand} is not sorted"

    # Verify no duplicates
    assert len(pool_set) == len(pool), "pool contains duplicate hands"

    # Cross-check a sample: verify every pool entry is solvable
    for hand in pool:
        items = [(n, []) for n in hand]
        assert _solve(items, TARGET) is not None, f"pool entry {hand} is NOT solvable"

    # Verify no solvable hand is missing (exhaustive negative check)
    for hand in combinations_with_replacement(range(CARD_MIN, CARD_MAX + 1), 4):
        items = [(n, []) for n in hand]
        is_solvable = _solve(items, TARGET) is not None
        in_pool = hand in pool_set
        assert (
            is_solvable == in_pool
        ), f"hand {hand}: solvable={is_solvable} but in_pool={in_pool}"


def test_pool_size():
    """The pool should have exactly 1346 solvable hands."""
    pool = enumerate_solvable_pool()
    assert len(pool) == 1346, f"expected 1346 solvable hands, got {len(pool)}"


# --------------------------------------------------------------------------- #
# Split overlap tests
# --------------------------------------------------------------------------- #


def test_no_split_overlap(tmp_path):
    """Train, val, test splits share zero puzzles."""
    sizes = generate_partitioned_splits(str(tmp_path), seed=0)

    # Load back the JSONL files
    splits = {}
    for split_name in ("train", "val", "test"):
        hands = set()
        path = os.path.join(str(tmp_path), f"{split_name}.jsonl")
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                hands.add(tuple(sorted(rec["numbers"])))
        splits[split_name] = hands
        assert (
            len(hands) == sizes[split_name]
        ), f"{split_name}: expected {sizes[split_name]} unique hands, got {len(hands)}"

    # Zero overlap between all pairs
    assert len(splits["train"] & splits["val"]) == 0, "train ∩ val is non-empty"
    assert len(splits["train"] & splits["test"]) == 0, "train ∩ test is non-empty"
    assert len(splits["val"] & splits["test"]) == 0, "val ∩ test is non-empty"

    # Union covers entire pool
    pool = set(enumerate_solvable_pool())
    union = splits["train"] | splits["val"] | splits["test"]
    assert (
        union == pool
    ), f"splits don't cover full pool: missing {len(pool - union)}, extra {len(union - pool)}"


def test_no_within_split_duplicates(tmp_path):
    """Each split contains only unique hands (no resampling)."""
    generate_partitioned_splits(str(tmp_path), seed=7)

    for split_name in ("train", "val", "test"):
        path = os.path.join(str(tmp_path), f"{split_name}.jsonl")
        hands = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                hands.append(tuple(sorted(rec["numbers"])))
        assert len(hands) == len(set(hands)), f"{split_name} contains duplicate hands"


def test_generate_dataset_rejects_oversized_request(tmp_path):
    """Requesting more samples than the pool size raises ValueError."""
    try:
        generate_dataset(2000, os.path.join(str(tmp_path), "too_big.jsonl"))
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "1346" in str(e) or "solvable" in str(e).lower()


def test_is_ood_accepted_with_warning(tmp_path):
    """is_ood=True is accepted but emits a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        generate_dataset(10, os.path.join(str(tmp_path), "ood.jsonl"), is_ood=True)
        assert len(w) == 1
        assert (
            "is_ood" in str(w[0].message).lower()
            or "game24" in str(w[0].message).lower()
        )
