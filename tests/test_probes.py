import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from evaluation.probes import extract_probe_data, run_probes


def test_extract_probe_data_labels():
    traj = Trajectory(
        all_hidden=torch.zeros(4, 5),
        input_ids=torch.zeros(4, dtype=torch.long),
        state_indices=torch.arange(4),
        op_ids=torch.tensor([0, 1, 2]),  # ADD, SUB, MUL
        operands=torch.tensor([[25.0, 3.0], [28.0, 50.0], [78.0, 75.0]]),
        numbers=[25, 50, 75, 100, 3, 4],
        target=10,
    )
    data = extract_probe_data([traj])
    # Three usable states s_0, s_1, s_2 (distance 3, 2, 1).
    assert data["B"].tolist() == [3, 2, 1]
    assert data["C"].tolist() == [0, 1, 2]
    assert data["D"].tolist() == [0, 1, 1]  # distance <= 2
    # Remaining large numbers {25,50,75,100} as they get consumed.
    assert data["A"].tolist() == [
        [1, 1, 1, 1],  # nothing used yet
        [2, 1, 1, 1],  # 25 used
        [2, 2, 1, 1],  # 25, 50 used
    ]


def _encoded_trajectories(n=60, H=12, steps=4, seed=0):
    """States linearly encode next-op (dims 0-3) and distance (dims 4-7)."""
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        all_hidden = torch.zeros(steps + 1, H)
        for i in range(steps):
            dist = steps - i  # 4..1
            all_hidden[i, op_ids[i]] = 1.0
            all_hidden[i, 3 + dist] = 1.0
        all_hidden += 0.01 * torch.randn(steps + 1, H, generator=g)
        operands = torch.randint(1, 100, (steps, 2), generator=g).float()
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.zeros(steps + 1, dtype=torch.long),
            state_indices=torch.arange(steps + 1),
            op_ids=op_ids,
            operands=operands,
            numbers=[25, 50, 75, 100, 3, 4],
            target=10,
        ))
    return trajs


def test_probes_recover_encoded_signal():
    train = _encoded_trajectories(n=60, seed=0)
    test = _encoded_trajectories(n=30, seed=1)
    df, details, fitted_probes = run_probes(train, test)

    assert list(df["probe"].str[0]) == ["A", "B", "C", "D"]
    assert details["n_train"] > 0 and details["n_test"] > 0

    by = {r["probe"][0]: r for _, r in df.iterrows()}
    # Distance (B) and next-op (C) are explicitly encoded -> high accuracy.
    assert by["B"]["accuracy"] > 0.85
    assert by["C"]["accuracy"] > 0.85
    # All metrics within valid ranges.
    for _, r in df.iterrows():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0
