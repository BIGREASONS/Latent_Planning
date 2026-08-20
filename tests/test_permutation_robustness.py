import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.permutation_robustness import evaluate_permutation_robustness


def _make_group(n_solutions, shared_start_code):
    """n_solutions trajectories for one problem, all sharing code at depth 0."""
    trajs = []
    for s in range(n_solutions):
        N = 3
        codes = torch.tensor(
            [
                shared_start_code,
                shared_start_code + s + 1,
                shared_start_code + s + 2,
                shared_start_code + s + 3,
            ],
            dtype=torch.int64,
        )
        trajs.append(
            DiscreteTrajectory(
                codes=codes,
                op_ids=torch.zeros(N, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[25, 50, 75, 100, 3, 5],
                target=100,
            )
        )
    return trajs


def test_perfect_consistency():
    """All solutions share the same code at depth 0 -> consistency = 1.0."""
    groups = [
        _make_group(3, shared_start_code=10),
        _make_group(2, shared_start_code=20),
    ]
    m = evaluate_permutation_robustness(groups, num_codes=64, seed=0)
    assert m["cross_consistency"] == 1.0
    assert m["n_pairs_cross"] > 0
    assert m["uniform_random_baseline"] == 1.0 / 64


def test_baseline_below_random():
    """Random codes -> consistency near uniform_random_baseline."""
    import random

    rng = random.Random(0)
    groups = []
    for _ in range(5):
        group = []
        for _ in range(3):
            N = 3
            codes = torch.tensor(
                [rng.randint(0, 16) for _ in range(N + 1)], dtype=torch.int64
            )
            group.append(
                DiscreteTrajectory(
                    codes=codes,
                    op_ids=torch.zeros(N, dtype=torch.long),
                    operands=torch.zeros(N, 2),
                    numbers=[25, 50, 75, 100, 3, 5],
                    target=100,
                )
            )
        groups.append(group)
    m = evaluate_permutation_robustness(groups, num_codes=16, seed=0)
    # Cross consistency should be near the uniform baseline (1/16 ≈ 0.0625).
    assert m["cross_consistency"] < 0.3
    assert m["n_problems_total"] == 5


def test_single_solution_no_cross_pairs():
    """One solution per problem -> zero cross pairs, consistency NaN."""
    groups = [[_make_group(1, 5)[0]] for _ in range(3)]
    m = evaluate_permutation_robustness(groups, num_codes=8, seed=0)
    assert m["n_pairs_cross"] == 0
    assert m["cross_consistency"] != m["cross_consistency"]  # NaN


def test_metrics_in_dict():
    groups = [_make_group(2, 10)]
    m = evaluate_permutation_robustness(groups, num_codes=32, seed=0)
    expected_keys = {
        "cross_consistency",
        "within_consistency",
        "uniform_random_baseline",
        "empirical_random_baseline",
        "n_pairs_cross",
        "n_problems_with_match",
        "n_problems_total",
        "mean_solutions_per_problem",
    }
    assert expected_keys.issubset(m.keys())
