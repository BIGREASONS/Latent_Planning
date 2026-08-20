"""Training loop for the latent transition model."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from data_processing.trajectory_dataset import (
    Trajectory,
    TransitionDataset,
    load_trajectories,
)
from models.transition_model import TransitionModel, transition_loss


@dataclass
class TransitionTrainConfig:
    mlp_hidden_dim: int = 512
    op_embed_dim: int = 16
    predict_delta: bool = True
    use_action: bool = True
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval_loss(model: TransitionModel, loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["h_t"], batch["op_id"], batch["operands"])
        loss = transition_loss(pred, batch["h_next"], reduction="sum")
        total += loss.item()
        n += batch["h_t"].shape[0] * batch["h_t"].shape[1]
    return total / max(n, 1)


def train_transition_model(
    train_trajs: List[Trajectory],
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[TransitionTrainConfig] = None,
) -> TransitionModel:
    """Train a transition model on teacher trajectories. Returns the model.

    Training history (per-epoch train/val MSE) is recorded on ``config.history``.
    """
    config = config or TransitionTrainConfig()
    torch.manual_seed(config.seed)

    train_ds = TransitionDataset(train_trajs)
    if len(train_ds) == 0:
        raise ValueError("No transitions in training trajectories.")

    operands_tensor = torch.stack([x for x in train_ds.operands])
    operand_mean = operands_tensor.mean().item()
    operand_std = operands_tensor.std().item() + 1e-8

    hidden_dim = train_trajs[0].hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = TransitionDataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = TransitionModel(
        hidden_dim=hidden_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        op_embed_dim=config.op_embed_dim,
        predict_delta=config.predict_delta,
        use_action=config.use_action,
        operand_mean=operand_mean,
        operand_std=operand_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        epoch_sum, epoch_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch["h_t"], batch["op_id"], batch["operands"])
            loss = transition_loss(pred, batch["h_next"])
            loss.backward()
            optimizer.step()
            bs = batch["h_t"].shape[0]
            epoch_sum += loss.item() * bs
            epoch_n += bs
        train_loss = epoch_sum / max(epoch_n, 1)

        record = {"epoch": epoch, "step": epoch, "loss": train_loss}
        if val_loader is not None:
            record["eval_loss"] = _eval_loss(model, val_loader)
        config.history.append(record)

    return model


def save_history_csv(history: List[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train latent transition model")
    parser.add_argument("--train_traj", type=str, required=True)
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/transition_model.pt")
    parser.add_argument(
        "--log_csv", type=str, default="reports/transition_train_log.csv"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument(
        "--no_delta", action="store_true", help="Predict h_next directly"
    )
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None

    config = TransitionTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        predict_delta=not args.no_delta,
    )
    model = train_transition_model(train_trajs, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "config": config.__dict__,
        },
        args.output,
    )
    save_history_csv(config.history, args.log_csv)
    final = config.history[-1]
    print(
        f"Done. Final train MSE={final['loss']:.5f}"
        + (f" val MSE={final['eval_loss']:.5f}" if "eval_loss" in final else "")
    )


if __name__ == "__main__":
    main()
