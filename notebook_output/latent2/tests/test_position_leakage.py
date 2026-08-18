import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.position_leakage import evaluate_position_leakage


def _disc_trajs(n=30, code_fn=None, max_steps=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        N = int(torch.randint(2, max_steps + 1, (1,), generator=g).item())
        if code_fn is None:
            codes = torch.randint(0, 8, (N + 1,), generator=g, dtype=torch.int64)
        else:
            codes = torch.tensor([code_fn(i) for i in range(N + 1)], dtype=torch.int64)
        trajs.append(DiscreteTrajectory(
            codes=codes, op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2), numbers=[1, 2, 3], target=100,
        ))
    return trajs


def test_position_leaking_codes_detected():
    """Code = position -> classifier achieves ~1.0 accuracy."""
    train = _disc_trajs(40, code_fn=lambda i: min(i, 7))
    test = _disc_trajs(20, code_fn=lambda i: min(i, 7))
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert m["position_predictability_score"] > 0.80
    assert m["n_train"] > 0
    assert m["n_test"] > 0


def test_position_independent_codes_near_chance():
    """Random codes -> score near majority-class chance."""
    train = _disc_trajs(40)
    test = _disc_trajs(20)
    m = evaluate_position_leakage(train, test, num_codes=8, seed=0)
    # Score should be close to (but may exceed slightly due to noise).
    assert m["position_predictability_score"] < 0.5
    assert m["chance_accuracy"] > 0.0


def test_permutation_null_lower_than_real():
    """Shuffled labels should not outperform the real classifier."""
    train = _disc_trajs(60, code_fn=lambda i: min(i, 7))
    test = _disc_trajs(30, code_fn=lambda i: min(i, 7))
    m = evaluate_position_leakage(train, test, num_codes=8, seed=0)
    assert m["permutation_null_accuracy"] <= m["position_predictability_score"] + 0.05


def test_metrics_bounded():
    train = _disc_trajs(20)
    test = _disc_trajs(10)
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert 0.0 <= m["position_predictability_score"] <= 1.0
    assert 0.0 <= m["permutation_null_accuracy"] <= 1.0
    assert 0.0 <= m["chance_accuracy"] <= 1.0


def test_empty_inputs_nan():
    train = []
    test = []
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert m["position_predictability_score"] != m["position_predictability_score"]  # NaN
