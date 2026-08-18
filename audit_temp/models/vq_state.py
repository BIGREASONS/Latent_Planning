"""Vector-quantized discrete state discovery for frozen LM hidden states.

This is the **only** component that converts a continuous hidden state ``h`` into
a discrete state id ``z``. It is a VQ-VAE style quantizer (van den Oord et al.,
2017) with the following design choices:

* **EMA codebook updates** (no gradient through the codebook), which are far
  more stable than the original gradient-based codebook update and avoid the
  optimizer destabilizing the embedding table.
* **Straight-through estimator** (STE): the quantized output passes the decoder
  gradient straight through to the encoder input, scaled by a commitment loss
  that pulls the encoder output toward its assigned code.
* **Dead-code revival**: at every step, codes whose EMA cluster size falls
  below ``epsilon`` are re-initialized to a randomly sampled input vector. This
  directly counteracts the codebook collapse that :mod:`evaluation.codebook_usage`
  is built to detect.

This module answers the Version-5 research question's first half:

    continuous hidden state ``h``  ->  discrete state id ``z``

Outputs are a ``(num_codes, hidden_dim)`` codebook that, once trained, can
quantize any frozen LM hidden state in a single ``forward`` call.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQStateQuantizer(nn.Module):
    """Vector-quantize a hidden state into a discrete code.

    Args:
        hidden_dim: dimension of the input hidden state ``h``.
        num_codes: size of the discrete codebook (the state-space cardinality).
        commitment_cost: weight on the commitment loss
            ``||sg(z_q) - h||^2`` that pulls the encoder output toward its code.
        ema_decay: EMA decay factor for codebook / cluster-size updates.
        epsilon: numerical stabilizer for the EMA cluster-size normalization,
            also used as the dead-code-revival threshold.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_codes: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Codebook initialized with small-variance gaussian (the EMA update
        # quickly overrides the init once data flows through).
        codebook = torch.randn(num_codes, hidden_dim) * 0.02
        self.register_buffer("codebook", codebook)
        # EMA tracking buffers (van den Oord et al. formulation).
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", codebook.clone())

    # ------------------------------------------------------------------ #
    def _quantize(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbor lookup. Returns (z_q, indices).

        ``h`` is (..., H); we flatten the leading dims for the distance
        computation and reshape back at the end.
        """
        lead_shape = h.shape[:-1]
        h_flat = h.reshape(-1, self.hidden_dim)  # (M, H)

        # Squared L2 distance ||h - c||^2 = ||h||^2 + ||c||^2 - 2 h·c
        # computed without materializing the (M, num_codes, H) tensor.
        dist = (
            h_flat.pow(2).sum(dim=-1, keepdim=True)               # (M, 1)
            + self.codebook.pow(2).sum(dim=-1, keepdim=False)     # (num_codes,)
            - 2.0 * h_flat @ self.codebook.t()                    # (M, num_codes)
        )
        indices = dist.argmin(dim=-1)                              # (M,)
        z_q_flat = self.codebook[indices]                          # (M, H)
        z_q = z_q_flat.reshape(*lead_shape, self.hidden_dim)
        return z_q, indices, z_q_flat

    # ------------------------------------------------------------------ #
    def _ema_update(self, h_flat: torch.Tensor, indices_flat: torch.Tensor) -> None:
        """Update EMA cluster size and codebook vectors (in-place, no grad).

        Implements dead-code revival: codes with cluster size below
        ``epsilon`` are re-initialized to a random input from the current batch.
        """
        with torch.no_grad():
            one_hot = F.one_hot(indices_flat, self.num_codes).type_as(h_flat)  # (M, K)
            cluster_size = one_hot.sum(dim=0)                                   # (K,)
            # Sum of encoder outputs assigned to each code.
            dw = one_hot.t() @ h_flat                                          # (K, H)

            # Laplace smoothing on the EMA cluster size.
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1.0 - self.ema_decay
            )
            n = self.ema_cluster_size.sum()
            # Laplace-smoothed cluster size, rescaled back to the *count* domain
            # (sums to ``n``). Dividing the running per-code input SUM by this
            # count yields the running per-code input MEAN = the new codebook.
            # (Without the ``* n`` rescale we'd divide by a probability and the
            # codebook would blow up by ~1/batch_size.)
            smoothed_count = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_codes * self.epsilon)
                * n
            )
            # EMA of the per-code sum of encoder outputs.
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)
            # Normalize to get the new codebook = running mean of assigned inputs.
            self.codebook.copy_(self.ema_w / smoothed_count.unsqueeze(-1))

            # Dead-code revival: any code whose smoothed cluster size is below
            # epsilon gets re-pointed to a random batch vector. This is the
            # primary defense against collapse (measured by codebook_usage).
            dead = self.ema_cluster_size < self.epsilon
            if dead.any() and h_flat.shape[0] > 0:
                n_dead = int(dead.sum().item())
                repl = h_flat[torch.randint(0, h_flat.shape[0], (n_dead,), device=h_flat.device)]
                self.codebook[dead] = repl
                # Reset the EMA accumulator so the revived code is not
                # immediately re-killed by a stale near-zero cluster size.
                self.ema_w[dead] = repl
                self.ema_cluster_size[dead] = 0.0

    # ------------------------------------------------------------------ #
    def forward(
        self, h: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize ``h`` and return ``(z_q_st, indices, info)``.

        Args:
            h: ``(B, H)`` or ``(..., H)`` continuous hidden states.
            training: when ``True`` the EMA codebook update is applied. Pass
                ``training=False`` for inference-only quantization (used by the
                discrete-trajectory encoder and the rollout test).

        ``info`` carries diagnostic scalars: ``commitment_loss`` (for the
        training objective), ``codebook_loss`` (zero under EMA, kept for API
        symmetry with gradient codebooks), ``perplexity`` (effective number of
        active codes) and ``active_codes`` (count of codes used this batch).
        """
        z_q, indices, _ = self._quantize(h)

        # Straight-through estimator: forward passes z_q, gradient flows to h.
        z_q_st = h + (z_q - h).detach()

        # Diagnostics + EMA update operate on the flattened assignment.
        h_flat = h.reshape(-1, self.hidden_dim)
        idx_flat = indices.reshape(-1)
        info = self._diagnostics(idx_flat)
        if training and self.training:
            self._ema_update(h_flat, idx_flat)

        # Commitment loss pulls the encoder output toward its (stop-grad) code.
        commitment_loss = F.mse_loss(h_flat, z_q.reshape(-1, self.hidden_dim).detach())
        info["commitment_loss"] = commitment_loss
        info["codebook_loss"] = torch.zeros((), device=h.device)  # 0 under EMA

        return z_q_st, indices, info

    # ------------------------------------------------------------------ #
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Inference-only quantization: returns integer code ids ``z``."""
        self.eval()
        with torch.no_grad():
            _, indices, _ = self.forward(h, training=False)
            return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up the continuous codebook vector for a batch of code ids."""
        with torch.no_grad():
            return self.codebook[indices]

    # ------------------------------------------------------------------ #
    def _diagnostics(self, idx_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perplexity (= effective number of codes used) and active-code count."""
        device = idx_flat.device
        counts = torch.bincount(idx_flat, minlength=self.num_codes).type_as(self.codebook)
        total = counts.sum().clamp_min(1.0)
        avg_probs = counts / total
        entropy = -(avg_probs * torch.log(avg_probs.clamp_min(1e-10))).sum()
        perplexity = torch.exp(entropy)
        active = (counts > 0).sum()
        return {"perplexity": perplexity, "active_codes": active}
