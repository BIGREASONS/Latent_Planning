"""Codebook usage / VQ collapse diagnostics.

Measures whether the trained VQ codebook is actually *used*: how many codes
are active, how skewed the usage distribution is, and how many codes are dead
(receive zero assignments). These are the standard VQ-VAE collapse signals.

Key outputs:

* **active code count** — codes used at least ``min_freq_frac`` of the time.
* **frequency distribution** — per-code empirical probability.
* **dead codes** — codes with zero (or near-zero) usage.
* **collapse score** — the Gini coefficient of the usage distribution. A Gini
  near 1 means a tiny subset of codes dominates (severe collapse); a Gini near
  ``0`` means perfectly uniform usage.
* **perplexity** — ``exp(H(code distribution))``, the effective number of
  codes used (mirrors the VQ-VAE paper's perplexity diagnostic).

These numbers are the first gate of the Version-5 verdict: a collapsed
codebook cannot support reusable discrete states.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector (0 = uniform, 1 = collapsed)."""
    arr = np.sort(counts.astype(np.float64))
    if arr.sum() <= 0:
        return 0.0
    n = arr.shape[0]
    idx = np.arange(1, n + 1)
    # Standard Gini formula on sorted values.
    return float((2.0 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


def analyze_codebook_usage(
    codes: torch.Tensor,
    num_codes: int,
    min_freq_frac: float = 1e-4,
) -> Dict:
    """Compute usage statistics from a 1-D tensor of integer code ids.

    Args:
        codes: int64 tensor of assigned codes (e.g. over all train states).
        num_codes: size of the codebook these ids index into.
        min_freq_frac: a code is "active" if its empirical frequency is at
            least this fraction of the total (default: ~1 per 10k).

    Returns a dict with: ``counts`` (per-code raw counts),
    ``freqs`` (per-code probabilities), ``active_codes``, ``dead_codes``,
    ``collapse_score`` (Gini), ``perplexity``, ``entropy``,
    ``mean_freq_when_used``.
    """
    if isinstance(codes, torch.Tensor):
        codes_np = codes.detach().cpu().to(torch.int64).numpy()
    else:
        codes_np = np.asarray(codes, dtype=np.int64)

    total = codes_np.shape[0]
    if total == 0:
        return {
            "counts": np.zeros(num_codes, dtype=np.int64),
            "freqs": np.zeros(num_codes, dtype=np.float64),
            "active_codes": 0,
            "dead_codes": num_codes,
            "collapse_score": 0.0,
            "perplexity": 0.0,
            "entropy": 0.0,
            "mean_freq_when_used": 0.0,
        }

    counts = np.bincount(codes_np, minlength=num_codes).astype(np.int64)
    freqs = counts.astype(np.float64) / total

    # Entropy in nats; perplexity = effective number of codes used.
    nonzero = freqs[freqs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    perplexity = float(np.exp(entropy))

    used = counts > 0
    active = freqs >= min_freq_frac
    dead = num_codes - int(used.sum())

    mean_freq_when_used = float(nonzero.mean()) if nonzero.size else 0.0

    return {
        "counts": counts,
        "freqs": freqs,
        "active_codes": int(active.sum()),
        "used_codes": int(used.sum()),
        "dead_codes": dead,
        "collapse_score": _gini(counts.astype(np.float64)),
        "perplexity": perplexity,
        "entropy": entropy,
        "mean_freq_when_used": mean_freq_when_used,
    }


def save_codebook_usage_report(
    stats: Dict, csv_path: str, png_path: str = None
) -> None:
    """Write a one-row summary CSV and (optionally) a log-frequency bar plot."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    row = {
        "active_codes": stats["active_codes"],
        "used_codes": stats["used_codes"],
        "dead_codes": stats["dead_codes"],
        "collapse_score_gini": round(stats["collapse_score"], 4),
        "perplexity": round(stats["perplexity"], 2),
        "entropy_nats": round(stats["entropy"], 4),
        "mean_freq_when_used": round(stats["mean_freq_when_used"], 5),
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    if png_path is not None:
        _plot_usage(stats, png_path)


def _plot_usage(stats: Dict, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = stats["counts"].astype(np.float64)
    order = np.argsort(counts)[::-1]  # most-used first
    freqs = stats["freqs"][order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(freqs.shape[0]), freqs + 1e-9, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Code rank (most-used first)")
    ax.set_ylabel("Empirical frequency (log)")
    ax.set_title(
        f"Codebook usage (active={stats['active_codes']}, "
        f"dead={stats['dead_codes']}, Gini={stats['collapse_score']:.3f}, "
        f"perplexity={stats['perplexity']:.1f})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
