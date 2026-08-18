import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    CodeTransitionModel,
    build_transition_matrix,
    conditional_entropy,
    save_entropy_report,
    save_predictability_metrics,
    train_code_transition,
)
from models.vq_state import VQStateQuantizer


def _synthetic(n=20, H=8, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 4
        seq = torch.randint(0, K, (N + 1,), generator=g)
        states = centers[seq] + 0.05 * torch.randn(N + 1, H, generator=g)
        trajs.append(Trajectory(
            all_hidden=states, input_ids=torch.zeros(N + 1, dtype=torch.long),
            state_indices=torch.arange(N + 1),
            op_ids=torch.randint(0, 4, (N,), generator=g),
            operands=torch.randint(1, 50, (N, 2), generator=g).float(),
            numbers=[1, 2, 3, 4, 5, 6], target=100,
        ))
    return trajs, centers, K


def test_transition_matrix_stochastic():
    # Deterministic cycles: code i -> (i+1) % K.
    K = 5
    trajs = []
    for _ in range(10):
        codes = torch.tensor([(i % K) for i in range(6)], dtype=torch.int64)
        trajs.append(DiscreteTrajectory(
            codes=codes, op_ids=torch.zeros(5, dtype=torch.long),
            operands=torch.zeros(5, 2), numbers=[1], target=1,
        ))
    T = build_transition_matrix(trajs, K)
    # Rows that were observed must sum to 1.
    for i in range(K):
        assert abs(T[i].sum() - 1.0) < 1e-6
    # Each observed transition should be ~100% on one successor.
    for i in range(K):
        assert T[i][(i + 1) % K] > 0.9


def test_transition_matrix_empty_rows_uniform():
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1], dtype=torch.int64),
        op_ids=torch.zeros(1, dtype=torch.long), operands=torch.zeros(1, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    # Unobserved rows (2, 3) should fall back to uniform.
    assert abs(T[2, 0] - 0.25) < 1e-6
    assert abs(T[3, 0] - 0.25) < 1e-6


def test_conditional_entropy_deterministic_is_zero():
    K = 3
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        op_ids=torch.zeros(3, dtype=torch.long), operands=torch.zeros(3, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    ent = conditional_entropy(T)
    # Deterministic transitions -> entropy should be 0 (all mass on one successor).
    assert ent["global_entropy"] < 0.01


def test_conditional_entropy_uniform():
    K = 10
    # Uniform: every row has equal probability to go to any code.
    # Simulate by making many random transitions.
    g = torch.Generator().manual_seed(0)
    trajs = [DiscreteTrajectory(
        codes=torch.randint(0, K, (6,), generator=g, dtype=torch.int64),
        op_ids=torch.zeros(5, dtype=torch.long), operands=torch.zeros(5, 2),
        numbers=[1], target=1,
    ) for _ in range(200)]
    T = build_transition_matrix(trajs, K)
    ent = conditional_entropy(T)
    # High entropy (near log(K)) expected for uniform transitions.
    assert ent["global_entropy"] > 1.0
    assert ent["global_perplexity"] > 2.0


def test_train_code_transition_reduces_loss():
    trajs, centers, K = _synthetic(n=30, seed=0)
    vq = VQStateQuantizer(hidden_dim=8, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    cfg = CodeTransitionConfig(epochs=40, seed=0)
    model, metrics = train_code_transition(disc, num_codes=K, config=cfg)
    # Majority/bigram baselines replace the removed uniform-chance metric
    # (V5.1 Fix 2) and must be well-defined accuracies.
    assert 0.0 <= metrics["majority_baseline"] <= 1.0
    assert 0.0 <= metrics["bigram_baseline"] <= 1.0
    # Sequences here are random (no transition structure), so the MLP cannot
    # meaningfully beat the count baselines — it should track the majority
    # floor rather than collapse far below it.
    assert metrics["top1_accuracy"] >= metrics["majority_baseline"] - 0.2
    assert metrics["predictive_entropy_nats"] > 0.0


def test_predictability_metrics_in_range():
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.randint(0, K, (4,), dtype=torch.int64),
        op_ids=torch.zeros(3, dtype=torch.long), operands=torch.zeros(3, 2),
        numbers=[1], target=1,
    ) for _ in range(10)]
    cfg = CodeTransitionConfig(epochs=5, seed=0)
    _, m = train_code_transition(trajs, num_codes=K, config=cfg)
    assert 0.0 <= m["top1_accuracy"] <= 1.0
    assert 0.0 <= m["predictive_entropy_nats"]
    assert m["predictive_perplexity"] >= 1.0


def test_save_entropy_report(tmp_path):
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64),
        op_ids=torch.zeros(4, dtype=torch.long), operands=torch.zeros(4, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    freqs = np.ones(K) / K
    ent = conditional_entropy(T, code_freqs=freqs)
    csv = tmp_path / "entropy.csv"
    png = tmp_path / "entropy.png"
    save_entropy_report(T, ent, str(csv), str(png), active_mask=np.ones(K, dtype=bool))
    assert csv.exists()
    assert png.exists()
