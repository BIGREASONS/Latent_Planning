"""Action-conditioned transition analysis (V5.2 Experiment 1).

V5.1 tested only the *action-blind* transition ``z_t -> z_{t+1}`` and found it
no more predictable than a bigram. But a *planning state* is defined by
action-conditioned dynamics ``(z_t, a_t) -> z_{t+1}`` (as in MuZero/Dreamer):
the next state need not be a function of the current state alone. This module
asks whether conditioning on the symbolic Countdown action exposes structure
the action-blind test could not see.

Models (all evaluated on a held-out split):

* **A. Majority** — predict the global mode of ``z_{t+1}``.
* **B. Bigram** — lookup ``z_t -> mode z_{t+1}``.
* **C. Action bigram** — lookup ``(z_t, op) -> mode z_{t+1}`` (fallback: state
  bigram, then global majority).
* **D. MLP(z_t)** — the V5.1 action-blind MLP.
* **E. MLP(z_t, op)** — action-conditioned MLP.
* **E2. MLP(z_t, op, operands)** — adds the numeric operands of the action.
* **F. MLP(op, operands)** — *action-only control*: ignores ``z_t`` entirely.
  If F ≈ E2, the current discrete state contributes nothing and the codes are
  not functioning as states.

Decisive contrasts: C vs B and E vs D (does the action help at all?), E vs C
(does a learned model beat the action-conditioned lookup?), F vs E2 (does the
state matter, or does the action alone dictate the next code?).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory

NUM_OPS = 4  # ADD/SUB/MUL/DIV id space (DIV unused in the data)


# --------------------------------------------------------------------------- #
# Transition extraction
# --------------------------------------------------------------------------- #
@dataclass
class ActionTransitions:
    z_t: np.ndarray  # (M,) current code
    z_next: np.ndarray  # (M,) next code
    op: np.ndarray  # (M,) op id of the action
    operands: np.ndarray  # (M, 2) raw [arg1, arg2]


def extract_action_transitions(disc: List[DiscreteTrajectory]) -> ActionTransitions:
    zt, zn, ops, opnd = [], [], [], []
    for tr in disc:
        codes = tr.codes
        for i in range(tr.num_steps):
            zt.append(int(codes[i].item()))
            zn.append(int(codes[i + 1].item()))
            ops.append(int(tr.op_ids[i].item()))
            opnd.append(
                [float(tr.operands[i][0].item()), float(tr.operands[i][1].item())]
            )
    return ActionTransitions(
        z_t=np.asarray(zt, dtype=np.int64),
        z_next=np.asarray(zn, dtype=np.int64),
        op=np.asarray(ops, dtype=np.int64),
        operands=np.asarray(opnd, dtype=np.float32).reshape(-1, 2),
    )


# --------------------------------------------------------------------------- #
# Lookup baselines (A, B, C)
# --------------------------------------------------------------------------- #
def majority_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    mode = Counter(train.z_next.tolist()).most_common(1)[0][0]
    return float((ev.z_next == mode).mean())


def _build_lookup(keys, nexts):
    tbl = defaultdict(Counter)
    for k, n in zip(keys, nexts):
        tbl[k][n] += 1
    return {k: c.most_common(1)[0][0] for k, c in tbl.items()}


def state_bigram_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    pred = _build_lookup(train.z_t.tolist(), train.z_next.tolist())
    g = Counter(train.z_next.tolist()).most_common(1)[0][0]
    yhat = np.array([pred.get(z, g) for z in ev.z_t])
    return float((yhat == ev.z_next).mean())


def action_bigram_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    """(z_t, op) lookup with fallback to state bigram then global majority."""
    key_tr = list(zip(train.z_t.tolist(), train.op.tolist()))
    pred_za = _build_lookup(key_tr, train.z_next.tolist())
    pred_z = _build_lookup(train.z_t.tolist(), train.z_next.tolist())
    g = Counter(train.z_next.tolist()).most_common(1)[0][0]
    yhat = []
    for z, op in zip(ev.z_t.tolist(), ev.op.tolist()):
        if (z, op) in pred_za:
            yhat.append(pred_za[(z, op)])
        elif z in pred_z:
            yhat.append(pred_z[z])
        else:
            yhat.append(g)
    return float((np.asarray(yhat) == ev.z_next).mean())


# --------------------------------------------------------------------------- #
# Conditional structure (entropy + deterministic fraction), measured on train
# --------------------------------------------------------------------------- #
def conditional_structure(keys, nexts, min_count: int = 10) -> Dict:
    """Frequency-weighted H(z'|key) and deterministic mass fraction.

    A "deterministic" conditioning is one whose successor distribution has
    entropy < 0.5 nats; we only credit conditionings seen at least
    ``min_count`` times so single-sample pairs cannot look spuriously
    deterministic. ``det_frac_mass`` is the share of transition *mass* coming
    from such conditionings.
    """
    tbl = defaultdict(Counter)
    for k, n in zip(keys, nexts):
        tbl[k][n] += 1
    total = sum(sum(c.values()) for c in tbl.values())
    gH, det_mass, reliable = 0.0, 0, 0
    for k, c in tbl.items():
        cnt = sum(c.values())
        p = np.array(list(c.values()), dtype=np.float64) / cnt
        H = float(-(p * np.log(p)).sum())
        gH += H * cnt
        if cnt >= min_count:
            reliable += cnt
            if H < 0.5:
                det_mass += cnt
    return {
        "global_entropy": gH / max(total, 1),
        "det_frac_mass": det_mass / max(total, 1),
        "n_conditions": len(tbl),
        "reliable_mass_frac": reliable / max(total, 1),
    }


# --------------------------------------------------------------------------- #
# MLP models (D, E, E2, F)
# --------------------------------------------------------------------------- #
class ActionTransitionMLP(nn.Module):
    """Next-code MLP optionally conditioned on op id, operands, and/or z_t."""

    def __init__(
        self,
        num_codes,
        num_ops=NUM_OPS,
        embed_dim=32,
        op_dim=8,
        hidden_dim=128,
        use_state=True,
        use_op=False,
        use_operands=False,
    ):
        super().__init__()
        self.use_state, self.use_op, self.use_operands = use_state, use_op, use_operands
        in_dim = 0
        if use_state:
            self.code_embed = nn.Embedding(num_codes, embed_dim)
            in_dim += embed_dim
        if use_op:
            self.op_embed = nn.Embedding(num_ops, op_dim)
            in_dim += op_dim
        if use_operands:
            in_dim += 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_codes)
        )

    def forward(self, z, op, operands):
        feats = []
        if self.use_state:
            feats.append(self.code_embed(z))
        if self.use_op:
            feats.append(self.op_embed(op))
        if self.use_operands:
            feats.append(operands)
        return self.net(torch.cat(feats, dim=-1))


@dataclass
class ActionMLPConfig:
    use_state: bool = True
    use_op: bool = False
    use_operands: bool = False
    embed_dim: int = 32
    op_dim: int = 8
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    seed: int = 0
    history: List[dict] = field(default_factory=list)


def _operand_features(
    operands: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Signed-log transform then standardize (operands span 1..thousands)."""
    sl = np.sign(operands) * np.log1p(np.abs(operands))
    return (sl - mean) / std


def _fit_operand_norm(operands: np.ndarray):
    sl = np.sign(operands) * np.log1p(np.abs(operands))
    mean = sl.mean(axis=0)
    std = sl.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _tensors(tr: ActionTransitions, op_mean, op_std):
    z = torch.tensor(tr.z_t, dtype=torch.long)
    op = torch.tensor(tr.op, dtype=torch.long)
    opnd = torch.tensor(
        _operand_features(tr.operands, op_mean, op_std), dtype=torch.float32
    )
    y = torch.tensor(tr.z_next, dtype=torch.long)
    return TensorDataset(z, op, opnd, y)


def train_action_mlp(
    train: ActionTransitions,
    ev: ActionTransitions,
    num_codes: int,
    config: ActionMLPConfig,
) -> Dict[str, float]:
    """Train one MLP variant; return held-out top1 + predictive entropy."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op_mean, op_std = _fit_operand_norm(train.operands)

    train_loader = DataLoader(
        _tensors(train, op_mean, op_std), batch_size=config.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        _tensors(ev, op_mean, op_std), batch_size=config.batch_size
    )

    model = ActionTransitionMLP(
        num_codes,
        use_state=config.use_state,
        use_op=config.use_op,
        use_operands=config.use_operands,
        embed_dim=config.embed_dim,
        op_dim=config.op_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for _ in range(config.epochs):
        model.train()
        for z, op, opnd, y in train_loader:
            z, op, opnd, y = z.to(device), op.to(device), opnd.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(z, op, opnd), y)
            loss.backward()
            opt.step()

    model.eval()
    correct, ent_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for z, op, opnd, y in eval_loader:
            z, op, opnd, y = z.to(device), op.to(device), opnd.to(device), y.to(device)
            logits = model(z, op, opnd)
            probs = F.softmax(logits, dim=-1)
            ent_sum += (
                (-(probs * torch.log(probs.clamp_min(1e-10))).sum(-1)).sum().item()
            )
            correct += (logits.argmax(-1) == y).sum().item()
            n += z.shape[0]
    return {"top1": correct / max(n, 1), "pred_entropy": ent_sum / max(n, 1)}


def run_all_models(
    train: ActionTransitions,
    ev: ActionTransitions,
    num_codes: int,
    epochs: int = 30,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run A-F and the two structural conditionings. Returns a results dict."""
    res: Dict[str, Dict] = {}
    res["A_majority"] = {"top1": majority_top1(train, ev), "cond": "—"}
    res["B_bigram"] = {"top1": state_bigram_top1(train, ev), "cond": "z_t"}
    res["C_action_bigram"] = {"top1": action_bigram_top1(train, ev), "cond": "z_t, op"}

    variants = {
        "D_mlp_z": ActionMLPConfig(
            use_state=True, use_op=False, use_operands=False, epochs=epochs, seed=seed
        ),
        "E_mlp_z_op": ActionMLPConfig(
            use_state=True, use_op=True, use_operands=False, epochs=epochs, seed=seed
        ),
        "E2_mlp_z_op_operands": ActionMLPConfig(
            use_state=True, use_op=True, use_operands=True, epochs=epochs, seed=seed
        ),
        "F_mlp_action_only": ActionMLPConfig(
            use_state=False, use_op=True, use_operands=True, epochs=epochs, seed=seed
        ),
    }
    conds = {
        "D_mlp_z": "z_t",
        "E_mlp_z_op": "z_t, op",
        "E2_mlp_z_op_operands": "z_t, op, operands",
        "F_mlp_action_only": "op, operands",
    }
    for name, cfg in variants.items():
        m = train_action_mlp(train, ev, num_codes, cfg)
        m["cond"] = conds[name]
        res[name] = m

    # Structural transition entropy / determinism (measured on train).
    res["_struct_state"] = conditional_structure(
        train.z_t.tolist(), train.z_next.tolist()
    )
    res["_struct_action"] = conditional_structure(
        list(zip(train.z_t.tolist(), train.op.tolist())), train.z_next.tolist()
    )
    return res
