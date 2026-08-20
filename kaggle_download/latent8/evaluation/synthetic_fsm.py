"""Synthetic finite-state environment for the V5.2 positive control.

Calibration target: V5.1/Experiment-1 found that, on Countdown, conditioning on
the action gives H(z'|z,op) ≈ 0.99 nats and a 0.11 deterministic mass. Is that
*meaningful* discrete-state structure, or near-noise? To answer, we need the
ceiling: run the exact same VQ + analysis pipeline on an environment that has
**known, genuinely reusable, deterministic states** and see what the pipeline
reports.

Environment
-----------
A deterministic FSM with ``num_states`` states and ``num_actions`` actions and a
random (fixed) transition table ``T[s, a] -> s'``. Trajectories are random
walks: a uniform start state, then random actions. Because the start is uniform
and the table is a random map, the state marginal is ~uniform at every step, so
**state is decorrelated from position** (unlike Countdown) — which also lets the
position-leakage metric be calibrated.

Each visited state ``s`` is rendered as a continuous "hidden state"
``proto[s] + noise`` (``structured=True``), mimicking an LM hidden state that
encodes a genuine reusable state. The **noise floor** (``structured=False``)
replaces the embedding with pure noise of the same scale, carrying no state
information — the lower anchor.

The trajectories are returned as standard
:class:`~data_processing.trajectory_dataset.Trajectory` objects (actions stored
in ``op_ids``) so the entire existing pipeline — VQ training, code encoding,
codebook usage, position leakage, action-blind and action-conditioned
transition analysis — runs on them unchanged. Ground-truth state ids are
returned alongside for purity / AMI scoring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory


def generate_fsm(num_states: int, num_actions: int, seed: int = 0) -> np.ndarray:
    """Deterministic transition table ``T[s, a] -> s'`` (shape K×A).

    Each action is a random **permutation** of the states. This keeps the
    machine deterministic and reusable while guaranteeing a well-mixed,
    ~uniform stationary distribution (a plain random table tends to funnel all
    mass into a few recurrent states, which would make the "states" degenerate).
    """
    rng = np.random.RandomState(seed)
    T = np.zeros((num_states, num_actions), dtype=np.int64)
    for a in range(num_actions):
        T[:, a] = rng.permutation(num_states)
    return T


def make_prototypes(
    num_states: int, hidden_dim: int, proto_scale: float = 1.0, seed: int = 0
) -> np.ndarray:
    """The environment's fixed ``state -> hidden vector`` map (K×H prototypes).

    This geometry must be **shared across train/val/test splits**: a VQ trained
    on the train split learns codes at these prototype locations, so the eval
    splits have to render the same state at the same location or the VQ cannot
    encode them (every eval state collapses onto one code, saturating the
    train->eval predictability metrics). Generate once, pass to every split.
    """
    rng = np.random.RandomState(seed)
    return (rng.randn(num_states, hidden_dim) * proto_scale).astype(np.float32)


def generate_dataset(
    T: np.ndarray,
    num_traj: int,
    hidden_dim: int,
    lengths: Tuple[int, ...] = (3, 4, 5),
    noise: float = 0.3,
    proto_scale: float = 1.0,
    structured: bool = True,
    seed: int = 0,
    protos: np.ndarray = None,
) -> Tuple[List[Trajectory], List[np.ndarray]]:
    """Generate FSM random-walk trajectories with continuous state embeddings.

    Returns ``(trajectories, true_states)`` where ``true_states[i]`` is the
    ground-truth state id per visited state of trajectory ``i`` (aligned with
    that trajectory's ``codes`` after VQ encoding).

    Pass ``protos`` (from :func:`make_prototypes`) to share one prototype
    geometry across splits; if omitted, prototypes are drawn from ``seed`` (each
    split then gets its own geometry — only valid for a single self-contained
    split).
    """
    num_states, num_actions = T.shape
    rng = np.random.RandomState(seed)
    if protos is None:
        # Self-contained split: prototypes drawn from this split's own seed.
        protos = (rng.randn(num_states, hidden_dim) * proto_scale).astype(np.float32)

    trajs: List[Trajectory] = []
    true_states: List[np.ndarray] = []
    for _ in range(num_traj):
        n_states = int(rng.choice(lengths))
        s = int(rng.randint(num_states))
        states = [s]
        actions = []
        for _ in range(n_states - 1):
            a = int(rng.randint(num_actions))
            s = int(T[s, a])
            actions.append(a)
            states.append(s)
        states_arr = np.asarray(states, dtype=np.int64)
        actions_arr = np.asarray(actions, dtype=np.int64)

        if structured:
            proto_idx = states_arr  # embedding tracks the true state
        else:  # noise floor: same well-separated prototypes, but assigned at
            # random per occurrence — codes stay spread (no VQ collapse) yet
            # carry no information about the walk.
            proto_idx = rng.randint(0, num_states, size=n_states)
        emb = (
            protos[proto_idx]
            + rng.randn(n_states, hidden_dim).astype(np.float32) * noise
        )

        trajs.append(
            Trajectory(
                all_hidden=torch.from_numpy(emb),
                input_ids=torch.zeros(n_states, dtype=torch.long),
                state_indices=torch.arange(n_states, dtype=torch.long),
                op_ids=torch.from_numpy(actions_arr).long(),
                operands=torch.zeros(max(n_states - 1, 0), 2),
                numbers=[0],
                target=0,
            )
        )
        true_states.append(states_arr)
    return trajs, true_states


def flatten_true_states(true_states: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-trajectory ground-truth state ids (aligns with all_codes)."""
    if not true_states:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(true_states, axis=0)


def oracle_transitions(true_states: List[np.ndarray], trajs: List[Trajectory]):
    """Ground-truth ``(s_t, a_t, s_{t+1})`` arrays for oracle-dynamics checks."""
    st, at, sn = [], [], []
    for s, tr in zip(true_states, trajs):
        a = tr.op_ids.numpy()
        for i in range(len(a)):
            st.append(int(s[i]))
            at.append(int(a[i]))
            sn.append(int(s[i + 1]))
    return np.asarray(st), np.asarray(at), np.asarray(sn)
