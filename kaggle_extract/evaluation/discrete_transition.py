"""Discrete transition predictability and entropy analysis.

Two related diagnostics over the discrete code trajectory cache:

**C3 — Transition Predictability.** Train a small action-blind MLP
``z_t -> z_{t+1}`` and measure top-1 accuracy, predictive entropy and
perplexity. Above-chance accuracy is the minimal evidence that discrete state
dynamics *exist* (i.e. the next state is a function of the current one).

**C4 — Transition Entropy Analysis.** Build the empirical transition matrix
``T[i, j] = P(z_next = j | z_current = i)`` and report:

* the per-state conditional entropy ``H(Z_next | Z_current = i)``, and
* the global conditional entropy ``H(Z_next | Z_current)``.

A *planning state* has low conditional entropy (a near-deterministic
successor); a *compression bucket* has high entropy (many possible
successors, i.e. the code is lossy). The split between these two regimes is
the substantive scientific output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    DiscreteTransitionDataset,
)


# --------------------------------------------------------------------------- #
# C3 — Transition predictability (small MLP)
# --------------------------------------------------------------------------- #
class CodeTransitionModel(nn.Module):
    """Tiny embedding-MLP next-code predictor: ``z_t -> logits over codes``."""

    def __init__(self, num_codes: int, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_codes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_codes),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(z_t))


@dataclass
class CodeTransitionConfig:
    embed_dim: int = 32
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval_predictability(
    model: CodeTransitionModel, loader: DataLoader, num_codes: int
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    ce_sum, correct, entropy_sum, n = 0.0, 0, 0.0, 0
    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_next = batch["z_next"].to(device)
        logits = model(z_t)
        probs = F.softmax(logits, dim=-1)
        entropy_sum += (-(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)).sum().item()
        ce_sum += F.cross_entropy(logits, z_next, reduction="sum").item()
        correct += (logits.argmax(-1) == z_next).sum().item()
        n += z_t.shape[0]
    loss = ce_sum / max(n, 1)
    return {
        "loss": loss,
        "top1_accuracy": correct / max(n, 1),
        "predictive_entropy_nats": entropy_sum / max(n, 1),
        # Perplexity of the model's predicted next-state distribution.
        "predictive_perplexity": float(np.exp(loss)),
    }


def transition_baselines(
    train_disc: List[DiscreteTrajectory],
    eval_disc: List[DiscreteTrajectory],
    num_codes: int,
) -> Dict[str, float]:
    """Non-parametric next-code baselines, evaluated on ``eval_disc``.

    These replace the old uniform ``1 / num_codes`` "chance" baseline, which
    badly understated the real floor whenever code usage is skewed (it made an
    MLP look strong merely for predicting a frequent code). The two honest
    references are:

    * **Majority** (Baseline A): always predict the single most frequent next
      code seen in training. With a collapsed/skewed codebook this alone can be
      large.
    * **Bigram** (Baseline B): predict ``argmax_j count(i -> j)`` for the
      current code ``i`` (its most frequent training successor), falling back to
      the global majority for codes unseen in training. This is the first-order
      Markov count model. The MLP only demonstrates *learned* structure if it
      clears the bigram; matching it means the MLP captured nothing beyond
      pairwise counts.
    """
    train_ds = DiscreteTransitionDataset(train_disc)
    eval_ds = DiscreteTransitionDataset(eval_disc)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        return {"majority_baseline": float("nan"), "bigram_baseline": float("nan")}

    zt_tr = np.asarray(train_ds.z_t, dtype=np.int64)
    zn_tr = np.asarray(train_ds.z_next, dtype=np.int64)
    zt_ev = np.asarray(eval_ds.z_t, dtype=np.int64)
    zn_ev = np.asarray(eval_ds.z_next, dtype=np.int64)

    # Majority: global mode of training next-codes.
    majority_code = int(np.bincount(zn_tr, minlength=num_codes).argmax())
    majority_acc = float((zn_ev == majority_code).mean())

    # Bigram: most frequent successor per current code.
    counts = np.zeros((num_codes, num_codes), dtype=np.int64)
    np.add.at(counts, (zt_tr, zn_tr), 1)
    bigram_pred = counts.argmax(axis=1)               # (num_codes,)
    bigram_pred[counts.sum(axis=1) == 0] = majority_code  # unseen i -> majority
    bigram_acc = float((zn_ev == bigram_pred[zt_ev]).mean())

    return {"majority_baseline": majority_acc, "bigram_baseline": bigram_acc}


def train_code_transition(
    train_disc: List[DiscreteTrajectory],
    val_disc: Optional[List[DiscreteTrajectory]] = None,
    num_codes: int = 256,
    config: Optional[CodeTransitionConfig] = None,
) -> Tuple[CodeTransitionModel, Dict[str, float]]:
    """Train the action-blind discrete transition model.

    Returns ``(model, metrics)`` where ``metrics`` is evaluated on ``val_disc``
    if provided, else on ``train_disc``, and includes the majority/bigram
    baselines alongside the MLP's top-1 accuracy.
    """
    config = config or CodeTransitionConfig()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DiscreteTransitionDataset(train_disc)
    if len(train_ds) == 0:
        raise ValueError("No discrete transitions to train on.")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    eval_disc = val_disc if val_disc else train_disc
    eval_loader = DataLoader(DiscreteTransitionDataset(eval_disc), batch_size=config.batch_size)

    model = CodeTransitionModel(num_codes, config.embed_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_next = batch["z_next"].to(device)
            optimizer.zero_grad()
            logits = model(z_t)
            loss = F.cross_entropy(logits, z_next)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * z_t.shape[0]
            n += z_t.shape[0]
        record = {"epoch": epoch, "step": epoch, "loss": loss_sum / max(n, 1)}
        ev = _eval_predictability(model, eval_loader, num_codes)
        record["eval_top1_accuracy"] = ev["top1_accuracy"]
        record["eval_loss"] = ev["loss"]
        config.history.append(record)

    metrics = _eval_predictability(model, eval_loader, num_codes)
    metrics.update(transition_baselines(train_disc, eval_disc, num_codes))
    return model, metrics


def save_predictability_metrics(metrics: Dict[str, float], csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


# --------------------------------------------------------------------------- #
# C4 — Empirical transition matrix + conditional entropy
# --------------------------------------------------------------------------- #
def build_transition_matrix(
    discrete_trajs: List[DiscreteTrajectory], num_codes: int, smoothing: float = 0.0
) -> np.ndarray:
    """Empirical ``T[i, j] = P(z_next = j | z_current = i)``.

    Rows with no observed successors fall back to a uniform distribution so
    the matrix is always a valid stochastic matrix (this also avoids
    ``log(0)`` downstream). Optional Laplace ``smoothing`` (default 0) adds
    pseudo-counts to every cell.
    """
    T = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
    for traj in discrete_trajs:
        codes = traj.codes.numpy()
        for i in range(traj.num_steps):
            T[codes[i], codes[i + 1]] += 1.0

    row_sums = T.sum(axis=1, keepdims=True)
    # Rows that were never observed (sum 0) -> uniform.
    zero = (row_sums.squeeze(-1)) == 0
    T[zero] = 1.0 / num_codes
    row_sums[zero] = 1.0
    T /= row_sums
    return T


def conditional_entropy(T: np.ndarray, code_freqs: Optional[np.ndarray] = None) -> Dict:
    """Conditional entropy of the transition matrix.

    Args:
        T: ``(K, K)`` stochastic transition matrix.
        code_freqs: optional ``(K,)`` marginal ``P(z_current = i)`` used to
            weight per-state entropies into the global ``H(Z_next | Z_current)``.
            If omitted, the unweighted mean of per-state entropies is reported.

    Returns a dict with per-state entropy array and the (weighted) global
    conditional entropy / perplexity. All entropies are in nats.
    """
    K = T.shape[0]
    safe = np.clip(T, 1e-12, None)
    per_state = -(safe * np.log(safe)).sum(axis=1)  # (K,)

    if code_freqs is None:
        weight = np.full(K, 1.0 / K)
        global_entropy = float(per_state.mean())
    else:
        w = np.asarray(code_freqs, dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        weight = w
        global_entropy = float((per_state * w).sum())

    return {
        "per_state_entropy": per_state,
        "global_entropy": global_entropy,
        "global_perplexity": float(np.exp(global_entropy)),
        # Fraction of *used* states whose successor is near-deterministic
        # (entropy < 0.5 nats ~= top-1 prob > ~0.78). These are "planning" codes.
        "deterministic_state_frac": float(
            ((per_state < 0.5) & (weight > 0)).sum() / max(int((weight > 0).sum()), 1)
        ),
    }


def save_entropy_report(
    T: np.ndarray,
    ent: Dict,
    csv_path: str,
    png_path: Optional[str] = None,
    active_mask: Optional[np.ndarray] = None,
) -> None:
    """Write a per-state entropy CSV (active states only) and the plot."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    per_state = ent["per_state_entropy"]
    rows = []
    mask = active_mask if active_mask is not None else np.ones_like(per_state, dtype=bool)
    for i in np.where(mask)[0]:
        rows.append(
            {
                "code": int(i),
                "entropy_nats": float(per_state[i]),
                "top1_successor_prob": float(T[i].max()),
                "top1_successor_code": int(T[i].argmax()),
            }
        )
    summary = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "entropy_nats", "top1_successor_prob", "top1_successor_code"]
    )
    summary.to_csv(csv_path, index=False)

    if png_path is not None:
        _plot_entropy(per_state, ent, png_path, active_mask)


def _plot_entropy(per_state, ent, png_path, active_mask=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    mask = active_mask if active_mask is not None else np.ones_like(per_state, dtype=bool)
    vals = per_state[mask]
    if vals.size:
        ax1.hist(vals, bins=min(40, max(8, int(np.sqrt(vals.size)))),
                 color="tab:purple", alpha=0.8)
    ax1.axvline(ent["global_entropy"], color="tab:red", linestyle="--",
                label=f"global H = {ent['global_entropy']:.3f}")
    ax1.set_xlabel("H(Z_next | Z_current = i)  [nats]")
    ax1.set_ylabel("# codes")
    ax1.set_title("Per-state transition entropy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(["global entropy", "global perplexity", "det. state frac"],
            [ent["global_entropy"], ent["global_perplexity"], ent["deterministic_state_frac"]],
            color=["tab:blue", "tab:orange", "tab:green"])
    ax2.set_title("Transition entropy summary")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
