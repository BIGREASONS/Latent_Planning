"""Oracle Transition coherence evaluation.

The Oracle Transition is a scientific control, not a model. It returns the
exact teacher hidden state at each depth, establishing the theoretical maximum
coherence achievable under perfect transitions.

For a trajectory ``h0 -> a0 -> h1 -> a1 -> h2 -> ...``, the Oracle returns::

    Oracle(h_t, a_t) = teacher_h_{t+1}

No learning. No prediction. No approximation.

This isolates the question:

    Is the Phase A coherence bottleneck in the learned dynamics,
    or in the hidden-state representation itself?

Output schema matches ``coherence.py`` exactly so CSVs are directly comparable.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from data_processing.trajectory_dataset import Trajectory


@torch.no_grad()
def evaluate_oracle_coherence(
    trajectories: List[Trajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    probe_d=None,
    max_depth: Optional[int] = None,
    domain: str = "countdown",
):
    """Evaluate coherence using the Oracle (perfect) transition.

    The Oracle simply looks up the true teacher state at each depth.
    This establishes the representation ceiling: even with perfect dynamics,
    how much task information survives in the frozen hidden states?

    Returns a pandas DataFrame with the same schema as ``evaluate_coherence``.
    """
    import pandas as pd
    import numpy as np

    if max_depth is None:
        lengths = [t.num_steps for t in trajectories]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    from evaluation.probes import get_probe_a_targets

    # Accumulators per depth — same structure as coherence.py.
    agg = {
        d: {
            "mse": [],
            "cos": [],
            "op_acc": [],
            "teacher_op_acc": [],
            "state_acc": [],
            "teacher_state_acc": [],
            "id_mse": [],
            "id_cos": [],
            "id_op_acc": [],
            "id_state_acc": [],
            "probe_b_acc": [],
            "teacher_probe_b_acc": [],
            "id_probe_b_acc": [],
            "probe_d_acc": [],
            "teacher_probe_d_acc": [],
            "id_probe_d_acc": [],
        }
        for d in range(1, max_depth + 1)
    }

    for traj in trajectories:
        states = traj.states  # (N+1, H)
        N = traj.num_steps
        h_identity = states[0:1]  # frozen at s_0
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            # ---- ORACLE TRANSITION: use the exact teacher state ----
            h = states[d : d + 1]  # Oracle output
            teacher = states[d : d + 1]  # Ground truth (same tensor)

            # MSE and cosine vs teacher — should be exactly 0 / 1.0
            agg[d]["mse"].append(F.mse_loss(h, teacher).item())
            agg[d]["cos"].append(F.cosine_similarity(h, teacher, dim=-1).item())

            # Identity baseline metrics (s_0 vs s_d)
            agg[d]["id_mse"].append(F.mse_loss(h_identity, teacher).item())
            agg[d]["id_cos"].append(
                F.cosine_similarity(h_identity, teacher, dim=-1).item()
            )

            # --- Probe evaluations ---
            h_np = h.cpu().numpy()
            teacher_np = teacher.cpu().numpy()
            identity_np = h_identity.cpu().numpy()

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                agg[d]["op_acc"].append(
                    1.0 if int(probe_c.predict(h_np)[0]) == target_op else 0.0
                )
                agg[d]["teacher_op_acc"].append(
                    1.0 if int(probe_c.predict(teacher_np)[0]) == target_op else 0.0
                )
                agg[d]["id_op_acc"].append(
                    1.0 if int(probe_c.predict(identity_np)[0]) == target_op else 0.0
                )

            if probe_b is not None:
                target_dist = N - d
                agg[d]["probe_b_acc"].append(
                    1.0 if int(probe_b.predict(h_np)[0]) == target_dist else 0.0
                )
                agg[d]["teacher_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(teacher_np)[0]) == target_dist else 0.0
                )
                agg[d]["id_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(identity_np)[0]) == target_dist else 0.0
                )

            if probe_d is not None:
                target_reach = 1 if (N - d) <= 2 else 0
                agg[d]["probe_d_acc"].append(
                    1.0 if int(probe_d.predict(h_np)[0]) == target_reach else 0.0
                )
                agg[d]["teacher_probe_d_acc"].append(
                    1.0 if int(probe_d.predict(teacher_np)[0]) == target_reach else 0.0
                )
                agg[d]["id_probe_d_acc"].append(
                    1.0 if int(probe_d.predict(identity_np)[0]) == target_reach else 0.0
                )

            if probe_a is not None:
                a_row = get_probe_a_targets(traj, d, domain)

                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t) if pv == tv) / len(t)

                pred_a = probe_a.predict(h_np)[0]
                agg[d]["state_acc"].append(_avg_match(pred_a, a_row))

                t_pred_a = probe_a.predict(teacher_np)[0]
                agg[d]["teacher_state_acc"].append(_avg_match(t_pred_a, a_row))

                id_pred_a = probe_a.predict(identity_np)[0]
                agg[d]["id_state_acc"].append(_avg_match(id_pred_a, a_row))

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for d in range(1, max_depth + 1):
        rows.append(
            {
                "depth": d,
                "mse": _mean(agg[d]["mse"]),
                "cosine_similarity": _mean(agg[d]["cos"]),
                "operator_accuracy": _mean(agg[d]["op_acc"]),
                "teacher_operator_accuracy": _mean(agg[d]["teacher_op_acc"]),
                "state_probe_accuracy": _mean(agg[d]["state_acc"]),
                "teacher_state_probe_accuracy": _mean(agg[d]["teacher_state_acc"]),
                "dist_probe_accuracy": _mean(agg[d]["probe_b_acc"]),
                "teacher_dist_probe_accuracy": _mean(agg[d]["teacher_probe_b_acc"]),
                "reach_probe_accuracy": _mean(agg[d]["probe_d_acc"]),
                "teacher_reach_probe_accuracy": _mean(agg[d]["teacher_probe_d_acc"]),
                "identity_mse": _mean(agg[d]["id_mse"]),
                "identity_cosine_similarity": _mean(agg[d]["id_cos"]),
                "identity_operator_accuracy": _mean(agg[d]["id_op_acc"]),
                "identity_state_probe_accuracy": _mean(agg[d]["id_state_acc"]),
                "identity_dist_probe_accuracy": _mean(agg[d]["id_probe_b_acc"]),
                "identity_reach_probe_accuracy": _mean(agg[d]["id_probe_d_acc"]),
                "semantic_gain_state": (
                    _mean(agg[d]["state_acc"]) - _mean(agg[d]["id_state_acc"])
                    if len(agg[d]["state_acc"])
                    else float("nan")
                ),
                "semantic_gain_dist": (
                    _mean(agg[d]["probe_b_acc"]) - _mean(agg[d]["id_probe_b_acc"])
                    if len(agg[d]["probe_b_acc"])
                    else float("nan")
                ),
                "semantic_gain_reach": (
                    _mean(agg[d]["probe_d_acc"]) - _mean(agg[d]["id_probe_d_acc"])
                    if len(agg[d]["probe_d_acc"])
                    else float("nan")
                ),
                "n_samples": len(agg[d]["mse"]),
            }
        )
    return pd.DataFrame(rows)


def save_oracle_coherence(df, csv_path: str, png_path: Optional[str] = None) -> None:
    """Write the Oracle coherence CSV and optional plot."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_oracle(df, png_path)


def _plot_oracle(df, png_path: str) -> None:
    """Standalone Oracle coherence plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Rollout depth")
    ax1.set_ylabel("Cosine similarity / Probe accuracy")
    ax1.plot(
        valid["depth"],
        valid["cosine_similarity"],
        marker="o",
        color="tab:blue",
        label="Oracle Cosine",
    )
    ax1.plot(
        valid["depth"],
        valid["operator_accuracy"],
        marker="s",
        color="tab:green",
        label="Oracle Op Acc",
    )
    ax1.plot(
        valid["depth"],
        valid["state_probe_accuracy"],
        marker="v",
        color="tab:purple",
        label="Oracle State Acc",
    )
    ax1.plot(
        valid["depth"],
        valid["identity_cosine_similarity"],
        marker="x",
        linestyle=":",
        color="tab:blue",
        alpha=0.5,
        label="Identity Cosine",
    )
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Hidden-state MSE", color="tab:red")
    ax2.plot(
        valid["depth"], valid["mse"], marker="d", color="tab:red", label="Oracle MSE"
    )
    ax2.plot(
        valid["depth"],
        valid["identity_mse"],
        marker="*",
        linestyle=":",
        color="tab:red",
        alpha=0.5,
        label="Identity MSE",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize="small",
    )
    ax1.set_title("Oracle Transition Coherence vs Depth")
    ax1.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
# Overlay comparison plot
# ------------------------------------------------------------------ #
def plot_comparison_overlay(
    oracle_df,
    action_df,
    blind_df,
    png_path: str,
) -> None:
    """Four-way overlay: Oracle vs Action-Conditioned vs Action-Blind vs Identity.

    All on the same axes for direct visual comparison.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Phase B: Oracle Transition Comparison", fontsize=14, fontweight="bold"
    )

    def _valid(df):
        return df[df["n_samples"] > 0]

    ov = _valid(oracle_df)
    av = _valid(action_df)
    bv = _valid(blind_df)

    # --- Panel 1: Cosine Similarity ---
    ax = axes[0, 0]
    ax.set_title("Cosine Similarity")
    ax.plot(
        ov["depth"],
        ov["cosine_similarity"],
        "o-",
        color="gold",
        linewidth=2,
        label="Oracle",
    )
    ax.plot(
        av["depth"],
        av["cosine_similarity"],
        "s-",
        color="tab:blue",
        label="Action-Cond.",
    )
    ax.plot(
        bv["depth"],
        bv["cosine_similarity"],
        "^-",
        color="tab:orange",
        label="Action-Blind",
    )
    ax.plot(
        av["depth"],
        av["identity_cosine_similarity"],
        "x:",
        color="gray",
        alpha=0.6,
        label="Identity",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 2: MSE ---
    ax = axes[0, 1]
    ax.set_title("Hidden-State MSE")
    ax.plot(ov["depth"], ov["mse"], "o-", color="gold", linewidth=2, label="Oracle")
    ax.plot(av["depth"], av["mse"], "s-", color="tab:blue", label="Action-Cond.")
    ax.plot(bv["depth"], bv["mse"], "^-", color="tab:orange", label="Action-Blind")
    ax.plot(
        av["depth"], av["identity_mse"], "x:", color="gray", alpha=0.6, label="Identity"
    )
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 3: Operator Accuracy (Probe C) ---
    ax = axes[1, 0]
    ax.set_title("Operator Accuracy (Probe C)")
    ax.plot(
        ov["depth"],
        ov["operator_accuracy"],
        "o-",
        color="gold",
        linewidth=2,
        label="Oracle",
    )
    ax.plot(
        av["depth"],
        av["operator_accuracy"],
        "s-",
        color="tab:blue",
        label="Action-Cond.",
    )
    ax.plot(
        bv["depth"],
        bv["operator_accuracy"],
        "^-",
        color="tab:orange",
        label="Action-Blind",
    )
    ax.plot(
        av["depth"],
        av["identity_operator_accuracy"],
        "x:",
        color="gray",
        alpha=0.6,
        label="Identity",
    )
    ax.plot(
        ov["depth"],
        ov["teacher_operator_accuracy"],
        "d--",
        color="green",
        alpha=0.5,
        label="Teacher Ceiling",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 4: State Probe Accuracy (Probe A) ---
    ax = axes[1, 1]
    ax.set_title("State Probe Accuracy (Probe A)")
    ax.plot(
        ov["depth"],
        ov["state_probe_accuracy"],
        "o-",
        color="gold",
        linewidth=2,
        label="Oracle",
    )
    ax.plot(
        av["depth"],
        av["state_probe_accuracy"],
        "s-",
        color="tab:blue",
        label="Action-Cond.",
    )
    ax.plot(
        bv["depth"],
        bv["state_probe_accuracy"],
        "^-",
        color="tab:orange",
        label="Action-Blind",
    )
    ax.plot(
        av["depth"],
        av["identity_state_probe_accuracy"],
        "x:",
        color="gray",
        alpha=0.6,
        label="Identity",
    )
    ax.plot(
        ov["depth"],
        ov["teacher_state_probe_accuracy"],
        "d--",
        color="green",
        alpha=0.5,
        label="Teacher Ceiling",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
