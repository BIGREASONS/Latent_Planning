"""Cross-domain transfer metrics for the V5 discrete codebook.

Given a codebook trained on the **arithmetic** (Countdown) domain, does it
transfer to other reasoning domains? For each transfer domain we encode its
hidden states with the frozen arithmetic VQ and measure:

* **code reuse** — fraction of transfer states assigned to a code that is
  *active* in the arithmetic domain (vs falling onto dead codes / novel modes).
* **entropy delta** — ``H_transfer - H_arithmetic`` over the code distribution.
  A large positive delta means transfer states are spread (the codebook does
  not cover them well); a negative delta means they collapse onto few codes.
* **transition stability** — symmetric KL divergence between the arithmetic
  empirical transition matrix ``T_arith`` and the transfer-empirical
  ``T_transfer`` (restricted to codes active in both). Low divergence means the
  same discrete dynamics govern both domains.

These three numbers summarize whether the discrete-state structure discovered
in arithmetic is a *general* property of the frozen LM's hidden-state
trajectories or an artifact of one domain.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
    all_codes,
)
from data_processing.trajectory_dataset import Trajectory
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.discrete_transition import build_transition_matrix
from models.vq_state import VQStateQuantizer


def _code_distribution(codes: torch.Tensor, num_codes: int) -> np.ndarray:
    counts = np.bincount(codes.detach().cpu().numpy(), minlength=num_codes).astype(np.float64)
    s = counts.sum()
    return counts / s if s > 0 else counts


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()) if nz.size else 0.0


def _symmetric_kl(P: np.ndarray, Q: np.ndarray, active_mask: np.ndarray) -> float:
    """Symmetric KL between two transition distributions, restricted to ``active`` rows.

    Each row is treated as a categorical over next-codes; we add a tiny floor to
    avoid log(0) and average the per-row symmetrized KL over active rows.
    """
    eps = 1e-12
    mask = active_mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    P = P[mask] + eps
    Q = Q[mask] + eps
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    kl_pq = (P * np.log(P / Q)).sum(axis=1)
    kl_qp = (Q * np.log(Q / P)).sum(axis=1)
    return float((kl_pq + kl_qp).mean() / 2.0)


def evaluate_cross_domain_transfer(
    vq: VQStateQuantizer,
    arith_disc: List[DiscreteTrajectory],
    transfer_trajs_by_domain: Dict[str, List[Trajectory]],
    num_codes: int,
) -> Dict[str, Dict]:
    """Compute transfer metrics for every domain in ``transfer_trajs_by_domain``.

    Args:
        vq: the arithmetic-trained VQ quantizer (frozen).
        arith_disc: arithmetic discrete trajectories (the in-domain reference).
        transfer_trajs_by_domain: ``{domain_name: [Trajectory, ...]}`` to evaluate.
        num_codes: codebook cardinality.

    Returns ``{domain: {code_reuse, entropy, entropy_delta, transition_kl,
    active_codes_in_domain, n_states}}`` plus an ``arithmetic`` reference entry.
    """
    # Arithmetic reference distribution + transition matrix.
    arith_codes = all_codes(arith_disc)
    arith_stats = analyze_codebook_usage(arith_codes, num_codes)
    arith_freq = arith_stats["freqs"]
    arith_H = arith_stats["entropy"]
    arith_active_mask = arith_freq > 0
    T_arith = build_transition_matrix(arith_disc, num_codes)

    results: Dict[str, Dict] = {
        "arithmetic": {
            "entropy": arith_H,
            "active_codes": int(arith_active_mask.sum()),
            "perplexity": arith_stats["perplexity"],
            "collapse_score_gini": arith_stats["collapse_score"],
            "n_states": int(arith_codes.shape[0]),
        }
    }

    for domain, trajs in transfer_trajs_by_domain.items():
        if not trajs:
            results[domain] = {"error": "no trajectories"}
            continue
        disc = encode_trajectories_to_codes(vq, trajs)
        codes = all_codes(disc)
        stats = analyze_codebook_usage(codes, num_codes)
        H = stats["entropy"]

        # Code reuse: fraction of transfer states landing on codes that are
        # active in the arithmetic domain.
        freq = stats["freqs"]
        reuse = float(freq[arith_active_mask].sum()) if arith_active_mask.any() else 0.0

        # Transition stability: symmetric KL between arithmetic and transfer
        # transition matrices over codes active in BOTH domains.
        T_trans = build_transition_matrix(disc, num_codes)
        both_active = arith_active_mask & (freq > 0)
        tkl = _symmetric_kl(T_arith, T_trans, both_active)

        results[domain] = {
            "code_reuse": reuse,
            "entropy": H,
            "entropy_delta": H - arith_H,
            "transition_kl_symmetric": tkl,
            "active_codes_in_domain": int(stats["used_codes"]),
            "perplexity": stats["perplexity"],
            "collapse_score_gini": stats["collapse_score"],
            "n_states": int(codes.shape[0]),
        }
    return results


def save_transfer_report(results: Dict[str, Dict], csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows = []
    for domain, m in results.items():
        row = {"domain": domain}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
