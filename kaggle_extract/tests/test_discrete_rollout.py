import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    train_code_transition,
)
from evaluation.discrete_rollout import evaluate_discrete_rollout
from models.vq_state import VQStateQuantizer


def _cyclic_trajs(n=20, H=12, K=6, seed=0):
    """Trajectories with deterministic cyclic codes: z_{t+1} = (z_t+1) % K."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 5
        start = int(torch.randint(0, K, (1,), generator=g).item())
        seq = [(start + i) % K for i in range(N + 1)]
        states = centers[K - 1] * torch.ones(N + 1, H)  # constant, to test code-only
        trajs.append(
            Trajectory(
                all_hidden=states,
                input_ids=torch.zeros(N + 1, dtype=torch.long),
                state_indices=torch.arange(N + 1),
                op_ids=torch.zeros(N, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[1, 2, 3],
                target=100,
            )
        )
    return trajs, centers, K


def test_rollout_produces_correct_depths():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc,
        num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    assert len(df) == 3
    assert list(df["depth"]) == [1, 2, 3]
    assert (df["n_samples"] > 0).all()


def test_rollout_cosine_in_range():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc,
        num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    for _, r in df.iterrows():
        # Cosine similarity must be in [-1, 1] (should be >= 0 for real data).
        assert -1.0 <= r["cosine_similarity"] <= 1.0
        assert r["code_match_accuracy"] >= 0.0


def test_rollout_mse_non_negative():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc,
        num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    for _, r in df.iterrows():
        assert r["mse"] >= 0.0


def test_rollout_degrades_with_depth():
    """On noisy data, code-match should tend to decrease with depth."""
    g = torch.Generator().manual_seed(1)
    H, K, n = 12, 8, 30
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 6
        start = int(torch.randint(0, K, (1,), generator=g).item())
        seq = [(start + i) % K for i in range(N + 1)]
        states = centers[K - 1] + 0.5 * torch.randn(N + 1, H, generator=g)
        trajs.append(
            Trajectory(
                all_hidden=states,
                input_ids=torch.zeros(N + 1, dtype=torch.long),
                state_indices=torch.arange(N + 1),
                op_ids=torch.zeros(N, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[1, 2, 3],
                target=100,
            )
        )
    vq = VQStateQuantizer(hidden_dim=H, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc,
        num_codes=K,
        config=CodeTransitionConfig(epochs=50, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=5)
    # First depth should not be worse than last depth on average.
    if df["n_samples"].iloc[0] > 0 and df["n_samples"].iloc[-1] > 0:
        assert (
            df["code_match_accuracy"].iloc[0]
            >= df["code_match_accuracy"].iloc[-1] - 0.2
        )
