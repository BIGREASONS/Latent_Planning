"""Discrete code trajectories.

Converts a list of continuous :class:`~data_processing.trajectory_dataset.Trajectory`
objects into discrete **code trajectories** by quantizing each symbolic-aligned
state ``s_0..s_N`` through a trained :class:`~models.vq_state.VQStateQuantizer`.

    h_1, h_2, h_3, h_4   -->   z_1, z_2, z_3, z_4

Each :class:`DiscreteTrajectory` retains the action metadata (op ids /
operands / numbers / target) needed by the permutation-robustness and
position-leakage analyses, but the *state* is now an int64 code id per step.

A :class:`DiscreteTransitionDataset` exposes the flat ``(z_t, z_{t+1})`` view
used to train / evaluate the discrete transition model. It is deliberately
**action-blind** (no op/operand conditioning): the Version-5 question is
whether discrete state dynamics exist *autonomously*, not whether they can be
predicted given the symbolic action.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from data_processing.trajectory_dataset import Trajectory
from models.vq_state import VQStateQuantizer


@dataclass
class DiscreteTrajectory:
    """One trajectory through *discrete* code space for a single problem."""

    codes: torch.Tensor       # (N+1,) int64 code id per state s_0..s_N
    op_ids: torch.Tensor      # (N,) action op id for steps 1..N
    operands: torch.Tensor    # (N, 2) float operands for steps 1..N
    numbers: List[int]
    target: int

    @property
    def num_steps(self) -> int:
        return int(self.op_ids.shape[0])

    @property
    def length(self) -> int:
        """Number of states (== ``num_steps + 1``)."""
        return int(self.codes.shape[0])


@torch.no_grad()
def encode_trajectories_to_codes(
    vq: VQStateQuantizer,
    trajectories: List[Trajectory],
    batch_size: int = 1024,
) -> List[DiscreteTrajectory]:
    """Quantize every trajectory's aligned states into discrete code ids.

    Args:
        vq: a *trained* :class:`VQStateQuantizer` (its codebook is read-only).
        trajectories: continuous teacher trajectories.
        batch_size: states quantized per forward pass (memory control).

    Returns a list of :class:`DiscreteTrajectory` parallel to the input.
    """
    vq.eval()
    # VQ codebook lives in a buffer (no Parameters), so fall back to buffers.
    try:
        device = next(vq.parameters()).device
    except StopIteration:
        device = next(vq.buffers()).device

    out: List[DiscreteTrajectory] = []
    for traj in trajectories:
        states = traj.states.to(device)  # (N+1, H)
        codes = []
        for i in range(0, states.shape[0], batch_size):
            chunk = states[i : i + batch_size]
            codes.append(vq.encode(chunk).cpu())
        codes_t = torch.cat(codes, dim=0).to(torch.int64)
        out.append(
            DiscreteTrajectory(
                codes=codes_t,
                op_ids=traj.op_ids,
                operands=traj.operands,
                numbers=list(traj.numbers),
                target=int(traj.target),
            )
        )
    return out


def save_discrete_trajectories(
    discrete_trajs: List[DiscreteTrajectory], path: str
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(discrete_trajs, path)


def load_discrete_trajectories(path: str) -> List[DiscreteTrajectory]:
    return torch.load(path, weights_only=False)


def all_codes(discrete_trajs: List[DiscreteTrajectory]) -> torch.Tensor:
    """Concatenate every code id across all trajectories into a 1-D int64 tensor."""
    if not discrete_trajs:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat([d.codes for d in discrete_trajs], dim=0)


class DiscreteTransitionDataset(Dataset):
    """Flat action-blind ``(z_t, z_{t+1})`` view over discrete trajectories.

    Each item is a ``(current_code, next_code)`` int64 pair. The transition
    model trained on this view answers: *given the current discrete state, is
    the next discrete state predictable?*
    """

    def __init__(self, discrete_trajs: List[DiscreteTrajectory]):
        self.z_t: List[int] = []
        self.z_next: List[int] = []
        for traj in discrete_trajs:
            for i in range(traj.num_steps):
                self.z_t.append(int(traj.codes[i].item()))
                self.z_next.append(int(traj.codes[i + 1].item()))

    def __len__(self) -> int:
        return len(self.z_t)

    def __getitem__(self, idx: int):
        return {
            "z_t": torch.tensor(self.z_t[idx], dtype=torch.long),
            "z_next": torch.tensor(self.z_next[idx], dtype=torch.long),
        }
