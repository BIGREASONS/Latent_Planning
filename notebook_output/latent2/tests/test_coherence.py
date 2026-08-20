import math
import torch
import torch.nn as nn

from data_processing.trajectory_dataset import Trajectory
from evaluation.coherence import evaluate_coherence


class _PerfectTransition(nn.Module):
    """Reproduces synthetic dynamics h_{t+1} = h_t + shift[op] exactly."""

    def __init__(self, shifts):
        super().__init__()
        self.shifts = shifts

    def forward(self, h, op_id, operands):
        return h + self.shifts.to(h.device)[op_id]


def _toy_trajectories(n=5, H=8, steps=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    shifts = torch.randn(4, H, generator=g)
    trajs = []
    for _ in range(n):
        T = steps + 2
        all_hidden = torch.zeros(T, H)
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        s = torch.randn(H, generator=g)
        all_hidden[0] = s
        for i in range(steps):
            s = s + shifts[op_ids[i]]
            all_hidden[i + 1] = s
        trajs.append(
            Trajectory(
                all_hidden=all_hidden,
                input_ids=torch.arange(T),
                state_indices=torch.arange(steps + 1),
                op_ids=op_ids,
                operands=torch.zeros(steps, 2),
                numbers=[1, 2, 3, 4, 5, 6],
                target=10,
            )
        )
    return trajs, shifts


def test_perfect_model_is_coherent():
    trajs, shifts = _toy_trajectories()
    model = _PerfectTransition(shifts)
    df = evaluate_coherence(model, trajs, probe_c=None, max_depth=8)

    assert list(df["depth"]) == list(range(1, 9))
    # Data only supports depth 4; deeper rows are empty.
    populated = df[df["n_samples"] > 0]
    assert set(populated["depth"]) == {1, 2, 3, 4}
    # Perfect dynamics -> ~0 MSE, ~1 cosine at every populated depth.
    assert populated["mse"].max() < 1e-8
    assert populated["cosine_similarity"].min() > 1 - 1e-5
    # No decoder -> operator accuracy is NaN.
    assert all(math.isnan(x) for x in populated["operator_accuracy"])
    assert all(math.isnan(x) for x in populated["state_probe_accuracy"])


def test_depth_limited_by_data():
    trajs, shifts = _toy_trajectories(steps=2)
    df = evaluate_coherence(_PerfectTransition(shifts), trajs, max_depth=8)
    populated = df[df["n_samples"] > 0]
    assert set(populated["depth"]) == {1, 2}
