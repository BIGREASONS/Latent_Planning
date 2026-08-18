"""Diagnostic decoder: ``hidden_state -> next reasoning token``.

This is a *diagnostic probe*, not a component of the planner. Its sole purpose
is to measure how much next-token information survives in (a) teacher hidden
states and (b) hidden states produced by rolling out the transition model. It is
deliberately lightweight — a single linear read-out by default — so that any
decodable signal reflects the representation, not decoder capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosticDecoder(nn.Module):
    """Maps a hidden state to a distribution over the token vocabulary.

    With ``mlp_hidden_dim=0`` (default) this is a single linear layer. A small
    hidden layer can be enabled but is discouraged for diagnostic use.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, mlp_hidden_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if mlp_hidden_dim and mlp_hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, vocab_size),
            )
        else:
            self.net = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)

    @torch.no_grad()
    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Argmax next-token prediction for a batch of hidden states."""
        return self.forward(hidden).argmax(dim=-1)
