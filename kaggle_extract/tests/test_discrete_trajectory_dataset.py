import os
import tempfile

import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTransitionDataset,
    all_codes,
    encode_trajectories_to_codes,
    save_discrete_trajectories,
    load_discrete_trajectories,
)
from models.vq_state import VQStateQuantizer


def _toy_trajectories(n=6, H=8, steps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        all_hidden = torch.randn(steps + 1, H, generator=g)
        trajs.append(
            Trajectory(
                all_hidden=all_hidden,
                input_ids=torch.zeros(steps + 1, dtype=torch.long),
                state_indices=torch.arange(steps + 1),
                op_ids=torch.randint(0, 4, (steps,), generator=g),
                operands=torch.randint(1, 50, (steps, 2), generator=g).float(),
                numbers=[1, 2, 3, 4, 5, 6],
                target=100,
            )
        )
    return trajs


def test_encode_produces_int_codes():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    assert len(disc) == len(trajs)
    for d, t in zip(disc, trajs):
        assert d.codes.shape[0] == t.num_steps + 1
        assert d.codes.dtype == torch.int64
        assert 0 <= int(d.codes.min()) < 12
        assert 0 <= int(d.codes.max()) < 12
        # Action metadata preserved.
        assert torch.equal(d.op_ids, t.op_ids)


def test_transition_dataset_pairs():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    ds = DiscreteTransitionDataset(disc)
    expected = sum(t.num_steps for t in trajs)
    assert len(ds) == expected
    item = ds[0]
    assert set(item.keys()) == {"z_t", "z_next"}
    assert item["z_t"].dtype == torch.long


def test_all_codes_concatenates():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    codes = all_codes(disc)
    expected = sum(t.num_steps + 1 for t in trajs)
    assert codes.shape[0] == expected
    assert all_codes([]).shape[0] == 0


def test_save_load_roundtrip():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "disc.pt")
        save_discrete_trajectories(disc, path)
        loaded = load_discrete_trajectories(path)
        assert len(loaded) == len(disc)
        for a, b in zip(loaded, disc):
            assert torch.equal(a.codes, b.codes)
