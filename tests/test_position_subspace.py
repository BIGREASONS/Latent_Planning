import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from evaluation.position_subspace import (
    apply_scrub_to_trajectories,
    build_position_dataset,
    inlp_position_projection,
    train_position_probe,
)


def _planted(M=800, D=24, n_cls=4, noise=0.15, seed=0):
    """Linearly-separable class signal along planted directions + noise."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_cls, size=M)
    dirs = rng.randn(n_cls, D) * 3.0
    X = (dirs[y] + noise * rng.randn(M, D)).astype(np.float32)
    cut = int(0.7 * M)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_probe_recovers_planted_signal():
    Xtr, ytr, Xte, yte = _planted()
    res = train_position_probe(Xtr, ytr, Xte, yte)
    # A linear probe should nearly perfectly recover a linearly planted signal.
    assert res["position_accuracy"] > 0.9
    assert res["majority_baseline"] < 0.4


def test_inlp_removes_planted_signal():
    Xtr, ytr, Xte, yte = _planted()
    before = train_position_probe(Xtr, ytr, Xte, yte)["position_accuracy"]
    P, info = inlp_position_projection(Xtr, ytr, Xte, yte, num_iters=8)

    # P is a symmetric projection (P == P^2 == P^T).
    assert np.allclose(P, P.T, atol=1e-6)
    assert np.allclose(P @ P, P, atol=1e-5)

    # After scrubbing, a fresh probe collapses toward the majority baseline.
    assert info["before_accuracy"] > 0.9
    assert before > 0.9
    assert info["after_accuracy"] <= info["majority_baseline"] + 0.1
    assert info["dims_removed"] >= 1


def test_scrub_trajectories_roundtrip_and_kills_signal():
    # Build trajectories whose state content is dominated by relative position.
    rng = np.random.RandomState(1)
    H = 16
    stage_dirs = torch.tensor(rng.randn(4, H) * 3.0, dtype=torch.float32)
    trajs = []
    for _ in range(120):
        N = 4  # 5 states -> bins 0,1,2,3,3
        n_states = N + 1
        bins = [min(int(i / N * 4), 3) for i in range(n_states)]
        states = stage_dirs[bins] + 0.1 * torch.randn(n_states, H)
        trajs.append(
            Trajectory(
                all_hidden=states,
                input_ids=torch.zeros(n_states, dtype=torch.long),
                state_indices=torch.arange(n_states),
                op_ids=torch.zeros(N, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[1, 2, 3],
                target=6,
            )
        )

    X, y = build_position_dataset(trajs)
    assert X.shape[0] == y.shape[0] == sum(t.num_steps + 1 for t in trajs)
    assert set(np.unique(y).tolist()).issubset({0, 1, 2, 3})

    mean = X.mean(axis=0)
    P, _ = inlp_position_projection(X - mean, y, X - mean, y, num_iters=6)
    scrubbed = apply_scrub_to_trajectories(trajs, mean, P)

    # Structural roundtrip: same shapes, metadata preserved, states changed.
    assert len(scrubbed) == len(trajs)
    for orig, sc in zip(trajs, scrubbed):
        assert sc.states.shape == orig.states.shape
        assert sc.num_steps == orig.num_steps
        assert sc.hidden_dim == orig.hidden_dim

    # Position is no longer linearly decodable from the scrubbed states.
    Xs, ys = build_position_dataset(scrubbed)
    cut = int(0.7 * Xs.shape[0])
    res = train_position_probe(Xs[:cut], ys[:cut], Xs[cut:], ys[cut:])
    assert res["position_accuracy"] <= res["majority_baseline"] + 0.15
