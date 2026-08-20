"""Discrete rollout coherence test.

The Version-5 analog of the continuous coherence rollout
(:mod:`evaluation.coherence`). Starting from the *teacher* code ``z_0`` we
repeatedly apply the action-blind discrete transition model
(:class:`evaluation.discrete_transition.CodeTransitionModel`) to roll forward
in code space. At each depth ``d`` we:

1. decode the predicted code back to a continuous hidden state via the VQ
   codebook (``h_decoded = codebook[z_pred]``),
2. run the **existing** linear probes (A remaining numbers / B distance / C
   next-op) on ``h_decoded``,
3. compare against the teacher code at the same depth.

The output schema mirrors ``coherence_depth.csv`` so the discrete rollout can
be plotted alongside (or against) the continuous one.

This test answers: *do the discrete dynamics remain coherent when rolled out
multiple steps, or does the code sequence collapse / diverge past depth 1?*
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from models.vq_state import VQStateQuantizer
from evaluation.discrete_transition import CodeTransitionModel

LARGE_NUMBERS = [25, 50, 75, 100]


@torch.no_grad()
def evaluate_discrete_rollout(
    transition_model: CodeTransitionModel,
    vq: VQStateQuantizer,
    discrete_trajs: List[DiscreteTrajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    max_depth: Optional[int] = None,
):
    """Roll out the discrete transition model and measure coherence vs depth.

    Returns a pandas DataFrame with one row per depth (1..max_depth):
    ``depth, code_match_accuracy, cosine_similarity, mse, operator_accuracy,
    dist_probe_accuracy, state_probe_accuracy, teacher_*``, ``n_samples``.
    """
    import pandas as pd

    if max_depth is None:
        lengths = [t.num_steps for t in discrete_trajs]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    # The transition model lives on whichever device it was trained on; the VQ
    # codebook is read-only, so we can do all rollout on the model's device and
    # move tiny tensors CPU-side for the sklearn probes.
    try:
        device = next(transition_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    transition_model.eval()
    codebook = vq.codebook.to(device)

    agg = {
        d: {
            "code_match": [],
            "cos": [],
            "mse": [],
            "op_acc": [],
            "teacher_op_acc": [],
            "probe_b_acc": [],
            "teacher_probe_b_acc": [],
            "state_acc": [],
            "teacher_state_acc": [],
            "n": 0,
        }
        for d in range(1, max_depth + 1)
    }

    for traj in discrete_trajs:
        N = traj.num_steps
        if N == 0:
            continue
        z = traj.codes[0:1].to(device)  # (1,) teacher start code
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            logits = transition_model(z)  # (1, K)
            z_pred = logits.argmax(dim=-1)  # (1,)
            teacher_code = traj.codes[d : d + 1].to(device)

            agg[d]["code_match"].append(float((z_pred == teacher_code).item()))
            h_pred = codebook[z_pred]  # (1, H)
            h_teacher = codebook[teacher_code]
            agg[d]["cos"].append(F.cosine_similarity(h_pred, h_teacher, dim=-1).item())
            agg[d]["mse"].append(F.mse_loss(h_pred, h_teacher).item())

            # Advance the rollout by the predicted code (autoregressive).
            z = z_pred

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["op_acc"].append(
                    1.0 if int(probe_c.predict(hp)[0]) == target_op else 0.0
                )
                agg[d]["teacher_op_acc"].append(
                    1.0 if int(probe_c.predict(ht)[0]) == target_op else 0.0
                )

            if probe_b is not None:
                target_dist = N - d
                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["probe_b_acc"].append(
                    1.0 if int(probe_b.predict(hp)[0]) == target_dist else 0.0
                )
                agg[d]["teacher_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(ht)[0]) == target_dist else 0.0
                )

            if probe_a is not None:
                used = []
                for i in range(d):
                    used.extend(traj.operands[i].tolist())
                a_row = []
                for v in LARGE_NUMBERS:
                    if v not in traj.numbers:
                        a_row.append(0)
                    elif v not in used:
                        a_row.append(1)
                    else:
                        a_row.append(2)

                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t) if pv == tv) / len(t)

                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["state_acc"].append(_avg_match(probe_a.predict(hp)[0], a_row))
                agg[d]["teacher_state_acc"].append(
                    _avg_match(probe_a.predict(ht)[0], a_row)
                )

            agg[d]["n"] += 1

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for d in range(1, max_depth + 1):
        rows.append(
            {
                "depth": d,
                "code_match_accuracy": _mean(agg[d]["code_match"]),
                "cosine_similarity": _mean(agg[d]["cos"]),
                "mse": _mean(agg[d]["mse"]),
                "operator_accuracy": _mean(agg[d]["op_acc"]),
                "teacher_operator_accuracy": _mean(agg[d]["teacher_op_acc"]),
                "dist_probe_accuracy": _mean(agg[d]["probe_b_acc"]),
                "teacher_dist_probe_accuracy": _mean(agg[d]["teacher_probe_b_acc"]),
                "state_probe_accuracy": _mean(agg[d]["state_acc"]),
                "teacher_state_probe_accuracy": _mean(agg[d]["teacher_state_acc"]),
                "n_samples": agg[d]["n"],
            }
        )
    return pd.DataFrame(rows)


def save_discrete_rollout(df, csv_path: str, png_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_rollout(df, png_path)


def _plot_rollout(df, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Rollout depth")
    ax.set_ylabel("Accuracy / similarity")

    curves = [
        ("code_match_accuracy", "Code match", "tab:blue", "o"),
        ("cosine_similarity", "Cosine (decoded)", "tab:cyan", "x"),
        ("operator_accuracy", "Probe C (op)", "tab:green", "s"),
        ("dist_probe_accuracy", "Probe B (dist)", "tab:orange", "*"),
        ("state_probe_accuracy", "Probe A (state)", "tab:purple", "v"),
    ]
    for col, label, color, marker in curves:
        if col in valid.columns:
            ax.plot(valid["depth"], valid[col], marker=marker, color=color, label=label)

    # Teacher ceilings as dashed lines.
    for col, label, color in [
        ("teacher_operator_accuracy", "Teacher C", "tab:green"),
        ("teacher_state_probe_accuracy", "Teacher A", "tab:purple"),
    ]:
        if col in valid.columns:
            ax.plot(
                valid["depth"],
                valid[col],
                linestyle="--",
                color=color,
                alpha=0.5,
                label=label,
            )

    ax.set_ylim(0, 1.05)
    ax.set_title("Discrete rollout coherence vs depth")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
