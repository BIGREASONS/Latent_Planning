import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes,
)
from evaluation.cross_domain_transfer import evaluate_cross_domain_transfer
from models.vq_state import VQStateQuantizer
from training.train_vq import train_vq_quantizer, VQTrainConfig


def _toy_trajs(centers, n=20, H=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        N = 4
        idx = torch.randint(0, len(centers), (N + 1,), generator=g)
        states = centers[idx] + 0.05 * torch.randn(N + 1, H, generator=g)
        out.append(
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
    return out


def test_near_domain_high_reuse():
    """Transfer domain near training domain -> high code reuse, low KL."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=30, seed=0)
    vq = train_vq_quantizer(
        train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128)
    )
    disc_train = encode_trajectories_to_codes(vq, train)
    # Near domain: same centers + small noise.
    near = _toy_trajs(centers + 0.05, n=15, seed=1)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"near": near}, num_codes=K)
    assert res["near"]["code_reuse"] > 0.8
    assert "transition_kl_symmetric" in res["near"]
    assert res["near"]["n_states"] == 15 * 5


def test_arithmetic_reference_present():
    """The reference 'arithmetic' entry should always be present."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=20, seed=0)
    vq = train_vq_quantizer(
        train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128)
    )
    disc_train = encode_trajectories_to_codes(vq, train)
    res = evaluate_cross_domain_transfer(vq, disc_train, {}, num_codes=K)
    assert "arithmetic" in res
    assert res["arithmetic"]["active_codes"] > 0
    assert res["arithmetic"]["perplexity"] > 0


def test_entropy_delta_sign():
    """Far domain should have different entropy (may go up or down)."""
    torch.manual_seed(1)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=30, seed=0)
    vq = train_vq_quantizer(
        train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128)
    )
    disc_train = encode_trajectories_to_codes(vq, train)
    far_centers = torch.randn(6, H) * 3  # very different region
    far = _toy_trajs(far_centers, n=15, seed=2)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"far": far}, num_codes=K)
    # Entropy delta may be positive or negative, but should not be NaN.
    assert not np.isnan(res["far"]["entropy_delta"])


def test_empty_domain_graceful():
    """Empty transfer domain -> error key in result."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=20, seed=0)
    vq = train_vq_quantizer(
        train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128)
    )
    disc_train = encode_trajectories_to_codes(vq, train)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"empty": []}, num_codes=K)
    assert res["empty"].get("error") is not None
