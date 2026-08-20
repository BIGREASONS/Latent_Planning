"""Latent coherence evaluation.

Measures how far the transition model can carry a frozen hidden state before it
drifts from the teacher trajectory. Starting from the *teacher* state ``s_0``, we
repeatedly apply the transition model with the **ground-truth symbolic actions**
and, at each depth ``d``, compare the rolled-out state ``ĥ_d`` against the teacher
state ``s_d``:

* hidden-state MSE
* cosine similarity
* diagnostic-decoder next-token accuracy (does ĥ_d still decode to the correct
  next reasoning token?)

The teacher-state decoder accuracy is reported alongside as an upper bound — it
isolates representation error (teacher) from accumulated dynamics error (rollout).

Output: ``coherence_depth.csv`` (depth, mse, cosine_similarity, token_accuracy,
plus n_samples and teacher_token_accuracy) and ``coherence_depth.png``.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from data_processing.trajectory_dataset import Trajectory, IGNORE_TOKEN


@torch.no_grad()
def evaluate_coherence(
    transition_model,
    trajectories: List[Trajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    probe_d=None,
    max_depth: Optional[int] = None,
    domain: str = "countdown",
    rollout_mode: str = "true",
):
    """Rollout the transition model and measure coherence vs depth.

    rollout_mode can be "true", "shuffled", or "constant".

    If max_depth is None, sets it dynamically to the 95th percentile of trajectory lengths.
    Returns a pandas DataFrame indexed by depth (1..max_depth).
    """
    import pandas as pd
    import numpy as np

    if max_depth is None:
        lengths = [t.num_steps for t in trajectories]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition_model = transition_model.to(device)
    transition_model.eval()

    # Compute random-pair cosine baseline
    if len(trajectories) >= 2:
        t_idx1 = torch.randint(0, len(trajectories), (10000,))
        t_idx2 = torch.randint(0, len(trajectories), (10000,))
        valid = t_idx1 != t_idx2
        t_idx1, t_idx2 = t_idx1[valid], t_idx2[valid]

        s1 = torch.cat(
            [
                trajectories[i.item()].states[
                    torch.randint(0, trajectories[i.item()].num_steps, (1,)).item() :
                ][:1]
                for i in t_idx1
            ],
            dim=0,
        ).to(device)
        s2 = torch.cat(
            [
                trajectories[i.item()].states[
                    torch.randint(0, trajectories[i.item()].num_steps, (1,)).item() :
                ][:1]
                for i in t_idx2
            ],
            dim=0,
        ).to(device)

        random_pair_cos = F.cosine_similarity(s1, s2, dim=-1).mean().item()
    else:
        random_pair_cos = 0.0

    # Accumulators per depth.
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

    from evaluation.probes import get_probe_a_targets

    for traj in trajectories:
        states = traj.states  # (N+1, H)
        N = traj.num_steps
        h = states[0:1].to(device)  # (1, H) teacher start
        h_identity = h.clone().to(device)
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            op = traj.op_ids[d - 1 : d].to(device)
            operands = traj.operands[d - 1 : d].to(device)

            if rollout_mode == "shuffled":
                rand_traj = trajectories[torch.randint(len(trajectories), (1,)).item()]
                rand_d = torch.randint(rand_traj.num_steps, (1,)).item()
                op = rand_traj.op_ids[rand_d : rand_d + 1].to(device)
                operands = rand_traj.operands[rand_d : rand_d + 1].to(device)
            elif rollout_mode == "constant":
                const_traj = trajectories[0]
                op = const_traj.op_ids[0:1].to(device)
                operands = const_traj.operands[0:1].to(device)

            h = transition_model(h, op, operands)  # predicted s_d
            teacher = states[d : d + 1].to(device)

            agg[d]["mse"].append(F.mse_loss(h, teacher).item())
            agg[d]["cos"].append(F.cosine_similarity(h, teacher, dim=-1).item())

            # Identity baseline metrics
            agg[d]["id_mse"].append(F.mse_loss(h_identity, teacher).item())
            agg[d]["id_cos"].append(
                F.cosine_similarity(h_identity, teacher, dim=-1).item()
            )

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                agg[d]["op_acc"].append(
                    1.0
                    if int(probe_c.predict(h.cpu().numpy())[0]) == target_op
                    else 0.0
                )
                agg[d]["teacher_op_acc"].append(
                    1.0
                    if int(probe_c.predict(teacher.cpu().numpy())[0]) == target_op
                    else 0.0
                )
                agg[d]["id_op_acc"].append(
                    1.0
                    if int(probe_c.predict(h_identity.cpu().numpy())[0]) == target_op
                    else 0.0
                )

            if probe_b is not None:
                target_dist = N - d
                agg[d]["probe_b_acc"].append(
                    1.0
                    if int(probe_b.predict(h.cpu().numpy())[0]) == target_dist
                    else 0.0
                )
                agg[d]["teacher_probe_b_acc"].append(
                    1.0
                    if int(probe_b.predict(teacher.cpu().numpy())[0]) == target_dist
                    else 0.0
                )
                agg[d]["id_probe_b_acc"].append(
                    1.0
                    if int(probe_b.predict(h_identity.cpu().numpy())[0]) == target_dist
                    else 0.0
                )

            if probe_d is not None:
                target_reach = 1 if (N - d) <= 2 else 0
                agg[d]["probe_d_acc"].append(
                    1.0
                    if int(probe_d.predict(h.cpu().numpy())[0]) == target_reach
                    else 0.0
                )
                agg[d]["teacher_probe_d_acc"].append(
                    1.0
                    if int(probe_d.predict(teacher.cpu().numpy())[0]) == target_reach
                    else 0.0
                )
                agg[d]["id_probe_d_acc"].append(
                    1.0
                    if int(probe_d.predict(h_identity.cpu().numpy())[0]) == target_reach
                    else 0.0
                )

            if probe_a is not None:
                a_row = get_probe_a_targets(traj, d, domain)

                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t) if pv == tv) / len(t)

                pred_a = probe_a.predict(h.cpu().numpy())[0]
                agg[d]["state_acc"].append(_avg_match(pred_a, a_row))

                t_pred_a = probe_a.predict(teacher.cpu().numpy())[0]
                agg[d]["teacher_state_acc"].append(_avg_match(t_pred_a, a_row))

                id_pred_a = probe_a.predict(h_identity.cpu().numpy())[0]
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
                "random_pair_cosine": random_pair_cos,
                "mean_centered_cosine": (
                    _mean(agg[d]["cos"]) - random_pair_cos
                    if agg[d]["cos"]
                    else float("nan")
                ),
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
    df = pd.DataFrame(rows)
    # Identity-Normalized Transition Score
    df["dynamics_gain"] = df["identity_mse"] / df["mse"]
    return df


def save_coherence(df, csv_path: str, png_path: Optional[str] = None) -> None:
    """Write the coherence CSV and (optionally) the depth plot."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_coherence(df, png_path)


def _plot_coherence(df, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Rollout depth")
    ax1.set_ylabel("Cosine similarity / Probe accuracy")
    (l1,) = ax1.plot(
        valid["depth"],
        valid["cosine_similarity"],
        marker="o",
        color="tab:blue",
        label="Cosine similarity",
    )
    (l2,) = ax1.plot(
        valid["depth"],
        valid["operator_accuracy"],
        marker="s",
        color="tab:green",
        label="Probe C (op) acc",
    )
    (l3,) = ax1.plot(
        valid["depth"],
        valid["state_probe_accuracy"],
        marker="v",
        color="tab:purple",
        label="Probe A (state) acc",
    )
    (l4,) = ax1.plot(
        valid["depth"],
        valid["dist_probe_accuracy"],
        marker="*",
        color="tab:orange",
        label="Probe B (dist) acc",
    )
    (l5,) = ax1.plot(
        valid["depth"],
        valid["reach_probe_accuracy"],
        marker="p",
        color="tab:cyan",
        label="Probe D (reach) acc",
    )

    (l6,) = ax1.plot(
        valid["depth"],
        valid["teacher_operator_accuracy"],
        marker="s",
        linestyle="--",
        color="tab:green",
        alpha=0.5,
        label="Teacher Probe C",
    )
    (l7,) = ax1.plot(
        valid["depth"],
        valid["teacher_state_probe_accuracy"],
        marker="v",
        linestyle="--",
        color="tab:purple",
        alpha=0.5,
        label="Teacher Probe A",
    )
    (l8,) = ax1.plot(
        valid["depth"],
        valid["teacher_dist_probe_accuracy"],
        marker="*",
        linestyle="--",
        color="tab:orange",
        alpha=0.5,
        label="Teacher Probe B",
    )
    (l9,) = ax1.plot(
        valid["depth"],
        valid["teacher_reach_probe_accuracy"],
        marker="p",
        linestyle="--",
        color="tab:cyan",
        alpha=0.5,
        label="Teacher Probe D",
    )

    (l_id_cos,) = ax1.plot(
        valid["depth"],
        valid["identity_cosine_similarity"],
        marker="x",
        linestyle=":",
        color="tab:blue",
        alpha=0.5,
        label="Cosine similarity (Identity)",
    )
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Hidden-state MSE", color="tab:red")
    (l_mse,) = ax2.plot(
        valid["depth"],
        valid["mse"],
        marker="d",
        color="tab:red",
        label="Hidden-state MSE",
    )
    (l_id_mse,) = ax2.plot(
        valid["depth"],
        valid["identity_mse"],
        marker="*",
        linestyle=":",
        color="tab:red",
        alpha=0.5,
        label="Hidden-state MSE (Identity)",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = [l1, l2, l3, l4, l5, l_id_cos, l6, l7, l8, l9, l_mse, l_id_mse]
    ax1.legend(
        lines,
        [ln.get_label() for ln in lines],
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize="small",
    )
    ax1.set_title("Latent coherence vs rollout depth")
    ax1.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
