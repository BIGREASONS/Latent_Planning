"""Latent transition model: ``T(h_t, a_t) -> h_{t+1}``.

The model learns to advance a frozen LM hidden state by one symbolic reasoning
step. An action is encoded as a learned op embedding concatenated with its
(normalized) numeric operands; this action vector is concatenated with the
current hidden state and passed through a two-layer MLP.

Architecture (as specified):

    action_embedding = [ Embedding(op) ; arg1/scale ; arg2/scale ]
    x = [ h_t ; action_embedding ]
    h_next = Linear -> ReLU -> Linear (x)            (+ h_t, if predict_delta)

By default the MLP predicts a *residual* (delta) that is added to ``h_t``. The
identity mapping is a strong prior for latent dynamics and materially improves
multi-step coherence; the core network is still exactly Linear->ReLU->Linear.
Set ``predict_delta=False`` to predict ``h_{t+1}`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data_processing.action_parser import OP_TO_ID

NUM_OPS = len(OP_TO_ID)


class ActionEncoder(nn.Module):
    """Encodes a symbolic action into a dense vector."""

    def __init__(
        self,
        op_embed_dim: int = 16,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
    ):
        super().__init__()
        self.op_embedding = nn.Embedding(NUM_OPS, op_embed_dim)
        self.register_buffer(
            "operand_mean", torch.tensor(operand_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "operand_std", torch.tensor(operand_std, dtype=torch.float32)
        )
        self.output_dim = op_embed_dim + 2  # + arg1, arg2

    def forward(self, op_id: torch.Tensor, operands: torch.Tensor) -> torch.Tensor:
        op_vec = self.op_embedding(op_id)  # (B, op_embed_dim)
        operand_vec = (operands - self.operand_mean) / self.operand_std  # (B, 2)
        return torch.cat([op_vec, operand_vec], dim=-1)


class TransitionModel(nn.Module):
    """Two-layer MLP transition model over [hidden_state ; action]."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,
        op_embed_dim: int = 16,
        predict_delta: bool = True,
        use_action: bool = True,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_delta = predict_delta
        self.use_action = use_action

        if self.use_action:
            self.action_encoder = ActionEncoder(op_embed_dim, operand_mean, operand_std)
            in_dim = hidden_dim + self.action_encoder.output_dim
        else:
            self.action_encoder = None
            in_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        op_id: torch.Tensor = None,
        operands: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_action:
            if op_id is None or operands is None:
                raise ValueError("op_id and operands are required when use_action=True")
            action = self.action_encoder(op_id, operands)
            x = torch.cat([h_t, action], dim=-1)
        else:
            x = h_t

        out = self.net(x)
        if self.predict_delta:
            return h_t + out
        return out


def transition_loss(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Mean-squared error between predicted and target hidden states."""
    return nn.functional.mse_loss(pred, target, reduction=reduction)
