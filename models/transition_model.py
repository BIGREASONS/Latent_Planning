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

    def __init__(self, op_embed_dim: int = 16, operand_mean: float = 0.0, operand_std: float = 1.0):
        super().__init__()
        self.op_embedding = nn.Embedding(NUM_OPS, op_embed_dim)
        self.register_buffer("operand_mean", torch.tensor(operand_mean, dtype=torch.float32))
        self.register_buffer("operand_std", torch.tensor(operand_std, dtype=torch.float32))
        self.output_dim = op_embed_dim + 2  # + arg1, arg2

    def forward(self, op_id: torch.Tensor, operands: torch.Tensor) -> torch.Tensor:
        op_vec = self.op_embedding(op_id)                # (B, op_embed_dim)
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


class LinearTransitionModel(nn.Module):
    """Low-rank linear projection transition model over [hidden_state ; action].
    Matches the parameter count of the MLP baseline perfectly."""

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
            nn.Linear(mlp_hidden_dim, hidden_dim)
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


class TransformerTransitionModel(nn.Module):
    """Bottleneck Transformer transition model for parameter matching."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,  # Used as d_model for the transformer bottleneck
        op_embed_dim: int = 16,
        predict_delta: bool = True,
        use_action: bool = True,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
        num_layers: int = 2,
        nhead: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_delta = predict_delta
        self.use_action = use_action
        self.d_model = mlp_hidden_dim // 2
        
        if self.use_action:
            self.action_encoder = ActionEncoder(op_embed_dim, operand_mean, operand_std)
            self.action_proj = nn.Linear(self.action_encoder.output_dim, self.d_model)
        else:
            self.action_encoder = None
            
        self.state_proj_in = nn.Linear(hidden_dim, self.d_model)
        
        # Positional embeddings for STATE (idx 0) and ACTION (idx 1)
        self.pos_emb = nn.Embedding(2, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.state_proj_out = nn.Linear(self.d_model, hidden_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        op_id: torch.Tensor = None,
        operands: torch.Tensor = None,
    ) -> torch.Tensor:
        B = h_t.size(0)
        
        # Project state
        state_emb = self.state_proj_in(h_t)  # (B, d_model)
        state_emb = state_emb.unsqueeze(1)   # (B, 1, d_model)
        
        if self.use_action:
            if op_id is None or operands is None:
                raise ValueError("op_id and operands are required when use_action=True")
            action_raw = self.action_encoder(op_id, operands)
            action_emb = self.action_proj(action_raw)  # (B, d_model)
            action_emb = action_emb.unsqueeze(1)       # (B, 1, d_model)
            
            # Sequence: [STATE, ACTION]
            seq = torch.cat([state_emb, action_emb], dim=1)  # (B, 2, d_model)
            positions = torch.arange(2, device=h_t.device).unsqueeze(0).expand(B, 2)
            seq = seq + self.pos_emb(positions)
        else:
            seq = state_emb
            positions = torch.zeros(B, 1, dtype=torch.long, device=h_t.device)
            seq = seq + self.pos_emb(positions)
            
        # Transformer pass
        out_seq = self.transformer(seq)  # (B, SeqLen, d_model)
        
        # Extract STATE token (index 0)
        state_out = out_seq[:, 0, :]  # (B, d_model)
        
        # Project back to full hidden_dim
        out = self.state_proj_out(state_out)  # (B, hidden_dim)
        
        if self.predict_delta:
            return h_t + out
        return out
