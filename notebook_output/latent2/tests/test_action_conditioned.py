import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.action_conditioned import (
    ActionMLPConfig,
    conditional_structure,
    extract_action_transitions,
    action_bigram_top1,
    state_bigram_top1,
    train_action_mlp,
)


def _planted_traj(n, K=5, N=6, seed=0):
    """Env where op0: c->(c+1)%K, op1: c->(c+2)%K.

    z_t alone is ambiguous (two equally likely successors); (z_t, op) is
    deterministic.
    """
    rng = np.random.RandomState(seed)
    trajs = []
    for _ in range(n):
        c = int(rng.randint(0, K))
        codes = [c]
        ops = []
        for _ in range(N):
            op = int(rng.randint(0, 2))
            c = (c + (1 if op == 0 else 2)) % K
            codes.append(c)
            ops.append(op)
        trajs.append(
            DiscreteTrajectory(
                codes=torch.tensor(codes, dtype=torch.int64),
                op_ids=torch.tensor(ops, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[1],
                target=1,
            )
        )
    return trajs


def test_action_conditioning_recovers_planted_structure():
    K = 5
    train = extract_action_transitions(_planted_traj(300, K=K, seed=0))
    ev = extract_action_transitions(_planted_traj(150, K=K, seed=1))

    # Lookups: state ambiguous (~0.5), action deterministic (~1.0).
    sb = state_bigram_top1(train, ev)
    ab = action_bigram_top1(train, ev)
    assert 0.4 <= sb <= 0.65, sb
    assert ab > 0.95, ab

    # MLPs: action-conditioned recovers the rule; state-only cannot.
    d = train_action_mlp(
        train, ev, K, ActionMLPConfig(use_state=True, use_op=False, epochs=40)
    )
    e = train_action_mlp(
        train, ev, K, ActionMLPConfig(use_state=True, use_op=True, epochs=40)
    )
    assert d["top1"] < 0.65, d
    assert e["top1"] > 0.9, e

    # Structure: H(z'|z) high, H(z'|z,op) ~ 0; determinism flips on conditioning.
    s_state = conditional_structure(train.z_t.tolist(), train.z_next.tolist())
    s_act = conditional_structure(
        list(zip(train.z_t.tolist(), train.op.tolist())), train.z_next.tolist()
    )
    assert s_state["global_entropy"] > 0.5
    assert s_act["global_entropy"] < 0.1
    assert s_state["det_frac_mass"] < 0.1
    assert s_act["det_frac_mass"] > 0.9


def test_extract_action_transitions_shapes():
    trajs = _planted_traj(10, K=4, N=5, seed=2)
    tr = extract_action_transitions(trajs)
    assert tr.z_t.shape == tr.z_next.shape == tr.op.shape
    assert tr.operands.shape == (tr.z_t.shape[0], 2)
    assert tr.z_t.shape[0] == 10 * 5
