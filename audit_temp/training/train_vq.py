"""Training loop for the VQ discrete-state quantizer.

Trains :class:`models.vq_state.VQStateQuantizer` on the *symbolic-aligned*
states ``s_0..s_N`` (``traj.states``) of the frozen-LM trajectory cache. The
result is a codebook that converts any continuous hidden state into a discrete
code id — the "state" of the Version-5 research question.

Only the aligned states are quantized (not every token's hidden state) so the
learned codes correspond to reasoning-step boundaries, matching the rest of the
pipeline's notion of a latent state.

History (per-epoch commitment loss, perplexity, active-code count) is recorded
on ``config.history`` for the standard training-curve plot.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.trajectory_dataset import (
    Trajectory,
    load_trajectories,
)
from models.vq_state import VQStateQuantizer


@dataclass
class VQTrainConfig:
    num_codes: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    lr: float = 3e-4
    epochs: int = 50
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


def _aligned_states_dataset(trajectories: List[Trajectory]) -> TensorDataset:
    """Stack the symbolic-aligned states ``s_0..s_N`` from every trajectory."""
    states = []
    for traj in trajectories:
        states.append(traj.states)  # (N+1, H)
    if not states:
        raise ValueError("No trajectories provided.")
    X = torch.cat(states, dim=0)  # (M, H)
    return TensorDataset(X)


def _init_codebook_from_data(
    model: VQStateQuantizer, train_ds: TensorDataset, device: torch.device, seed: int = 0
) -> None:
    """Data-dependent codebook initialization (V5.1 Fix 1).

    The default ``randn * 0.02`` init leaves the codebook two orders of
    magnitude smaller in norm than the hidden states (``||c|| ~ 0.9`` vs
    ``||h|| ~ 85``). On the very first batch nearly every state then maps to
    whichever code happens to be marginally closest, and the EMA update plus
    dead-code revival cannot recover — the codebook collapses to ~6/32 active
    codes (the invalidated V5 result). Seeding every code from a randomly
    sampled training state places all codes inside the data manifold from step
    zero, so EMA refines real clusters instead of fighting a scale mismatch.
    """
    X = train_ds.tensors[0]  # (M, H) aligned hidden states
    K = model.num_codes
    g = torch.Generator().manual_seed(seed)
    if X.shape[0] >= K:
        idx = torch.randperm(X.shape[0], generator=g)[:K]
    else:  # fewer states than codes — sample with replacement
        idx = torch.randint(0, X.shape[0], (K,), generator=g)
    pts = X[idx].to(device).clone()
    with torch.no_grad():
        model.codebook.copy_(pts)
        model.ema_w.copy_(pts)
        # Start every code with a non-trivial cluster size so revival does not
        # immediately re-kill the freshly seeded codes on the first step.
        model.ema_cluster_size.fill_(1.0)


def _model_device(model: nn.Module) -> torch.device:
    """Device of a module that works even when it has no Parameters (only buffers)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


@torch.no_grad()
def _eval(model: VQStateQuantizer, loader: DataLoader) -> dict:
    """Mean commitment loss + perplexity / active codes on held-out states."""
    model.eval()
    device = _model_device(model)
    commit_sum, perplexity_sum, active_sum, n = 0.0, 0.0, 0.0, 0
    for (h,) in loader:
        h = h.to(device)
        _, _, info = model(h, training=False)
        b = h.shape[0]
        commit_sum += info["commitment_loss"].item() * b
        perplexity_sum += info["perplexity"].item() * b
        active_sum += info["active_codes"].item()
        n += b
    return {
        "commitment_loss": commit_sum / max(n, 1),
        "perplexity": perplexity_sum / max(n, 1),
        "active_codes": active_sum / max(1, len(loader)),
    }


def train_vq_quantizer(
    train_trajs: List[Trajectory],
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[VQTrainConfig] = None,
) -> VQStateQuantizer:
    """Train a VQ codebook on symbolic-aligned hidden states. Returns the model."""
    config = config or VQTrainConfig()
    torch.manual_seed(config.seed)

    hidden_dim = train_trajs[0].hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = _aligned_states_dataset(train_trajs)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = _aligned_states_dataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = VQStateQuantizer(
        hidden_dim=hidden_dim,
        num_codes=config.num_codes,
        commitment_cost=config.commitment_cost,
        ema_decay=config.ema_decay,
    ).to(device)
    # Seed the codebook from real data BEFORE any EMA step (V5.1 Fix 1).
    _init_codebook_from_data(model, train_ds, device, seed=config.seed)
    # The codebook is a buffer updated by EMA inside forward(); the quantizer
    # has no trainable parameters. We still build an optimizer over any
    # *future* trainable params (e.g. an encoder net) so this loop generalizes;
    # if there are none, training is pure EMA and the optimizer is a no-op.
    trainable = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
        if trainable
        else None
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        commit_sum, perplexity_sum, active_sum, n = 0.0, 0.0, 0.0, 0
        for (h,) in train_loader:
            h = h.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            _, _, info = model(h, training=True)
            loss = config.commitment_cost * info["commitment_loss"]
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            b = h.shape[0]
            commit_sum += info["commitment_loss"].item() * b
            perplexity_sum += info["perplexity"].item() * b
            active_sum += info["active_codes"].item()
            n += b

        record = {
            "epoch": epoch,
            "step": epoch,
            "loss": commit_sum / max(n, 1),
            "perplexity": perplexity_sum / max(n, 1),
            "active_codes": active_sum / max(1, len(train_loader)),
        }
        if val_loader is not None:
            ev = _eval(model, val_loader)
            record["eval_loss"] = ev["commitment_loss"]
            record["eval_perplexity"] = ev["perplexity"]
            record["eval_active_codes"] = ev["active_codes"]
        config.history.append(record)

    return model


def save_history_csv(history: List[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train VQ discrete-state quantizer")
    parser.add_argument(
        "--train_traj", type=str, required=True, help="Input .pt trajectories file"
    )
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default="checkpoints/vq_state.pt"
    )
    parser.add_argument("--log_csv", type=str, default="reports/vq_train_log.csv")
    parser.add_argument("--num_codes", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None

    config = VQTrainConfig(
        num_codes=args.num_codes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        commitment_cost=args.commitment_cost,
        ema_decay=args.ema_decay,
    )
    model = train_vq_quantizer(train_trajs, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "num_codes": model.num_codes,
            "config": config.__dict__,
        },
        args.output,
    )
    save_history_csv(config.history, args.log_csv)
    final = config.history[-1]
    msg = (
        f"Done. Final train commitment={final['loss']:.5f} "
        f"perplexity={final['perplexity']:.1f}/{config.num_codes} "
        f"active={int(final['active_codes'])}"
    )
    if "eval_loss" in final:
        msg += (
            f" | val commitment={final['eval_loss']:.5f} "
            f"perplexity={final['eval_perplexity']:.1f} "
            f"active={int(final['eval_active_codes'])}"
        )
    print(msg)


if __name__ == "__main__":
    main()
