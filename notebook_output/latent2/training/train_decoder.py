"""Training loop for the diagnostic next-token decoder."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from data_processing.trajectory_dataset import (
    Trajectory,
    DecoderDataset,
    load_trajectories,
)
from models.diagnostic_decoder import DiagnosticDecoder


@dataclass
class DecoderTrainConfig:
    mlp_hidden_dim: int = 0
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval(model: DiagnosticDecoder, loader: DataLoader) -> dict:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["hidden"])
        loss_sum += ce(logits, batch["target"]).item()
        correct += (logits.argmax(-1) == batch["target"]).sum().item()
        n += batch["target"].shape[0]
    return {"loss": loss_sum / max(n, 1), "accuracy": correct / max(n, 1)}


def train_decoder_model(
    train_trajs: List[Trajectory],
    vocab_size: int,
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[DecoderTrainConfig] = None,
) -> DiagnosticDecoder:
    """Train the diagnostic decoder on teacher hidden states. Returns the model."""
    config = config or DecoderTrainConfig()
    torch.manual_seed(config.seed)

    train_ds = DecoderDataset(train_trajs)
    if len(train_ds) == 0:
        raise ValueError("No (hidden, next_token) pairs in training trajectories.")
    hidden_dim = train_trajs[0].hidden_dim

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = DecoderDataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiagnosticDecoder(hidden_dim, vocab_size, config.mlp_hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    ce = torch.nn.CrossEntropyLoss()

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for step_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["hidden"])
            loss = ce(logits, batch["target"])
            loss.backward()
            optimizer.step()
            bs = batch["target"].shape[0]
            loss_sum += loss.item() * bs
            correct += (logits.argmax(-1) == batch["target"]).sum().item()
            n += bs

            if step_idx % 10 == 0:
                print(
                    f"  [Epoch {epoch+1}/{config.epochs} | Step {step_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        record = {
            "epoch": epoch,
            "step": epoch,
            "loss": loss_sum / max(n, 1),
            "accuracy": correct / max(n, 1),
        }
        if val_loader is not None:
            ev = _eval(model, val_loader)
            record["eval_loss"] = ev["loss"]
            record["eval_accuracy"] = ev["accuracy"]
        config.history.append(record)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train diagnostic next-token decoder")
    parser.add_argument("--train_traj", type=str, required=True)
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument(
        "--output", type=str, default="checkpoints/diagnostic_decoder.pt"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None
    config = DecoderTrainConfig(epochs=args.epochs, lr=args.lr)
    model = train_decoder_model(train_trajs, args.vocab_size, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "vocab_size": model.vocab_size,
        },
        args.output,
    )
    final = config.history[-1]
    print(
        f"Done. Final train acc={final['accuracy']:.4f}"
        + (f" val acc={final['eval_accuracy']:.4f}" if "eval_accuracy" in final else "")
    )


if __name__ == "__main__":
    main()
