"""Phase B orchestrator: Oracle Transition Diagnostic.

Runs after Phase A. Loads saved trajectories and fitted probes, then:

  1. Evaluates Oracle (perfect) transition coherence
  2. Loads Phase A action-conditioned and action-blind CSVs for comparison
  3. Generates the 4-way overlay plot
  4. Generates the Phase B Oracle report with gain tables and diagnosis

Usage (from repo root after Phase A completes)::

    python scripts/run_phase_b.py --reports_dir reports

Or on Kaggle, appended after Phase A in the same notebook.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import pandas as pd
import numpy as np

from data_processing.trajectory_dataset import load_trajectories
from evaluation.oracle_coherence import (
    evaluate_oracle_coherence,
    save_oracle_coherence,
    plot_comparison_overlay,
)
from evaluation.probes import run_probes


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #
def generate_phase_b_report(
    oracle_df: pd.DataFrame,
    action_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    md_path: str,
    domain: str = "countdown",
) -> None:
    """Generate the Phase B Oracle diagnostic report."""

    def _v(df):
        return df[df["n_samples"] > 0]

    ov = _v(oracle_df)
    av = _v(action_df)
    bv = _v(blind_df)

    lines = []
    lines.append("# Phase B Report — Oracle Transition Diagnostic\n")
    lines.append(
        "The Oracle Transition returns the **exact teacher hidden state** at "
        "each depth. It performs no learning and no prediction. It establishes "
        "the theoretical maximum coherence achievable under perfect dynamics.\n"
    )

    # ---- Sanity check ----
    lines.append("## 0. Sanity Check\n")
    oracle_cos_vals = ov["cosine_similarity"].tolist()
    oracle_mse_vals = ov["mse"].tolist()
    all_cos_one = all(c > 0.999 for c in oracle_cos_vals)
    all_mse_zero = all(m < 1e-6 for m in oracle_mse_vals)
    if all_cos_one and all_mse_zero:
        lines.append(
            "✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all "
            "depths. The Oracle is correctly returning the exact teacher state.\n"
        )
    else:
        lines.append(
            "⚠️ **FAIL**: Oracle cosine or MSE deviates from expected values. "
            "There may be a bug in the Oracle implementation.\n"
        )
        lines.append(f"  Cosine values: {oracle_cos_vals}")
        lines.append(f"  MSE values: {oracle_mse_vals}\n")

    # ---- Question 1: How much coherence survives? ----
    lines.append("## 1. How much coherence survives under perfect transitions?\n")
    lines.append(
        "Since the Oracle returns the exact teacher state, its cosine and MSE "
        "are trivially perfect. The informative metric is **probe accuracy**: "
        "how much task information do the *teacher states themselves* contain "
        "at each depth?\n"
    )
    lines.append("| Depth | Oracle State Acc | Oracle Op Acc | n_samples |")
    lines.append("|---|---|---|---|")
    for _, r in ov.iterrows():
        lines.append(
            f"| {int(r['depth'])} | {r['state_probe_accuracy']:.3f} | "
            f"{r['operator_accuracy']:.3f} | {int(r['n_samples'])} |"
        )
    lines.append("")

    # ---- Question 2: Transition error vs representation limitation ----
    lines.append("## 2. Transition Learning Error vs Representation Limitations\n")
    lines.append(
        "The **Oracle Gain** at each depth measures how much coherence the "
        "learned transition model leaves on the table.\n"
    )

    # Cosine gain table
    lines.append("### Cosine Similarity Gain\n")
    lines.append("| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_cos = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        a_cos = av.loc[av["depth"] == d, "cosine_similarity"].values
        b_cos = bv.loc[bv["depth"] == d, "cosine_similarity"].values
        if len(o_cos) and len(a_cos) and len(b_cos):
            gain = o_cos[0] - a_cos[0]
            lines.append(
                f"| {d} | {o_cos[0]:.3f} | {a_cos[0]:.3f} | "
                f"{b_cos[0]:.3f} | {gain:+.3f} |"
            )
    lines.append("")

    # State probe gain table
    lines.append("### State Probe Accuracy Gain (Probe A)\n")
    if domain == "game24":
        lines.append(
            "*Probe A Definition: Game24 (Remaining card multiset encoding)*\n"
        )
    else:
        lines.append(
            "*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*\n"
        )

    lines.append("| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_s = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        a_s = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        b_s = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values
        if len(o_s) and len(a_s) and len(b_s):
            gain = o_s[0] - a_s[0]
            lines.append(
                f"| {d} | {o_s[0]:.3f} | {a_s[0]:.3f} | "
                f"{b_s[0]:.3f} | {gain:+.3f} |"
            )
    lines.append("")

    # Operator accuracy gain table
    lines.append("### Operator Accuracy Gain (Probe C)\n")
    lines.append("| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_op = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        a_op = av.loc[av["depth"] == d, "operator_accuracy"].values
        b_op = bv.loc[bv["depth"] == d, "operator_accuracy"].values
        if len(o_op) and len(a_op) and len(b_op):
            gain = o_op[0] - a_op[0]
            lines.append(
                f"| {d} | {o_op[0]:.3f} | {a_op[0]:.3f} | "
                f"{b_op[0]:.3f} | {gain:+.3f} |"
            )
    lines.append("")

    # ---- Action Gain ----
    lines.append("## 2b. Does the Action Vector actually drive dynamics?\n")
    lines.append(
        "The **Action Gain** measures how much of the theoretically available "
        "planning signal (Oracle - Blind) is captured by the action conditioning. "
        "Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.\n"
    )
    lines.append(
        "| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |"
    )
    lines.append("|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        bc = bv.loc[bv["depth"] == d, "cosine_similarity"].values

        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        bs_ = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values

        oo = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        ao = av.loc[av["depth"] == d, "operator_accuracy"].values
        bo = bv.loc[bv["depth"] == d, "operator_accuracy"].values

        def calc_gain(o, a, b):
            if not (len(o) and len(a) and len(b)):
                return float("nan")
            denom = o[0] - b[0]
            if abs(denom) < 1e-4:
                return float("nan")
            return (a[0] - b[0]) / denom

        c_gain = calc_gain(oc, ac, bc)
        s_gain = calc_gain(os_, as_, bs_)
        o_gain = calc_gain(oo, ao, bo)

        c_str = f"{c_gain:+.3f}" if not np.isnan(c_gain) else "n/a"
        s_str = f"{s_gain:+.3f}" if not np.isnan(s_gain) else "n/a"
        o_str = f"{o_gain:+.3f}" if not np.isnan(o_gain) else "n/a"

        if any(not np.isnan(g) for g in [c_gain, s_gain, o_gain]):
            lines.append(f"| {d} | {c_str} | {s_str} | {o_str} |")
    lines.append("")

    # ---- Question 3: Full comparison ----
    lines.append("## 3. Full Depth-by-Depth Comparison\n")
    lines.append(
        "| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | "
        "Oracle State | Action State | Blind State | Identity State |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        bc = bv.loc[bv["depth"] == d, "cosine_similarity"].values
        ic = av.loc[av["depth"] == d, "identity_cosine_similarity"].values
        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        bs_ = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values
        is_ = av.loc[av["depth"] == d, "identity_state_probe_accuracy"].values
        if all(len(x) for x in [oc, ac, bc, ic, os_, as_, bs_, is_]):
            lines.append(
                f"| {d} | {oc[0]:.3f} | {ac[0]:.3f} | {bc[0]:.3f} | "
                f"{ic[0]:.3f} | {os_[0]:.3f} | {as_[0]:.3f} | "
                f"{bs_[0]:.3f} | {is_[0]:.3f} |"
            )
    lines.append("")

    # ---- Diagnosis ----
    lines.append("## 4. Bottleneck Diagnosis\n")

    # Compute average gains across depths
    shared_depths = sorted(
        set(ov["depth"].tolist())
        & set(av["depth"].tolist())
        & set(bv["depth"].tolist())
    )
    cos_gains, state_gains, op_gains = [], [], []
    for d in shared_depths:
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        oo = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        ao = av.loc[av["depth"] == d, "operator_accuracy"].values
        if len(oc) and len(ac):
            cos_gains.append(oc[0] - ac[0])
        if len(os_) and len(as_):
            state_gains.append(os_[0] - as_[0])
        if len(oo) and len(ao):
            op_gains.append(oo[0] - ao[0])

    avg_cos_gain = np.mean(cos_gains) if cos_gains else 0
    avg_state_gain = np.mean(state_gains) if state_gains else 0
    avg_op_gain = np.mean(op_gains) if op_gains else 0

    # Check oracle representation ceiling
    oracle_state_accs = ov["state_probe_accuracy"].dropna().tolist()
    oracle_op_accs = ov["operator_accuracy"].dropna().tolist()
    avg_oracle_state = np.mean(oracle_state_accs) if oracle_state_accs else 0
    avg_oracle_op = np.mean(oracle_op_accs) if oracle_op_accs else 0

    lines.append(f"**Average Oracle Gain (cosine):** {avg_cos_gain:+.4f}")
    lines.append(f"**Average Oracle Gain (state probe):** {avg_state_gain:+.4f}")
    lines.append(f"**Average Oracle Gain (operator):** {avg_op_gain:+.4f}\n")
    lines.append(f"**Average Oracle State Probe Accuracy:** {avg_oracle_state:.4f}")
    lines.append(f"**Average Oracle Operator Accuracy:** {avg_oracle_op:.4f}\n")

    # Case analysis
    oracle_degrades = avg_oracle_state < 0.6 or avg_oracle_op < 0.4
    gain_large = avg_cos_gain > 0.05 or avg_state_gain > 0.10
    gain_small = avg_cos_gain < 0.02 and avg_state_gain < 0.05

    lines.append("### Interpretation\n")
    if oracle_degrades:
        lines.append(
            "**Case C — Representation Instability.** Even under perfect "
            "(Oracle) transitions, the teacher hidden states yield low probe "
            "accuracy. The frozen hidden state is not a stable planning state. "
            "The bottleneck is in the **representation itself**, not the "
            "learned dynamics.\n"
        )
    elif gain_small:
        lines.append(
            "**Case A — Transition model is NOT the bottleneck.** Oracle and "
            "action-conditioned coherence are comparable. The learned "
            "transition model already captures most of the available dynamics. "
            "The remaining degradation comes from **representation limitations** "
            "— the hidden states themselves lose task information as depth "
            "increases.\n"
        )
    elif gain_large:
        lines.append(
            "**Case B — Transition model IS the bottleneck.** The Oracle "
            "significantly outperforms the action-conditioned transition. The "
            "representation contains usable planning information that the "
            "learned dynamics fail to preserve. Improving the transition model "
            "(or its training) is the highest-leverage intervention.\n"
        )
    else:
        lines.append(
            "**Mixed result.** The Oracle gain is moderate, suggesting "
            "contributions from both transition learning error and "
            "representation limitations. Further investigation is needed.\n"
        )

    lines.append("---\n")
    lines.append(
        "_This report was generated by the Phase B Oracle Transition "
        "Diagnostic. The Oracle performs no learning; it simply returns the "
        "teacher state at each depth. It answers one question: is latent "
        "planning limited by the learned dynamics, or by the representation "
        "itself?_\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Phase B: Oracle Transition Diagnostic"
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory containing Phase A outputs",
    )
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--domain", type=str, default="countdown")
    args = parser.parse_args()

    out = args.reports_dir

    # ---- Load Phase A artifacts ----
    print("[Phase B] Loading Phase A trajectories and probes...")

    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    test_traj_path = os.path.join(out, "trajectories", "test.pt")

    if not os.path.exists(train_traj_path) or not os.path.exists(test_traj_path):
        print("ERROR: Phase A trajectories not found. Run Phase A first.")
        print(f"  Expected: {train_traj_path}")
        print(f"  Expected: {test_traj_path}")
        sys.exit(1)

    train_trajs = load_trajectories(train_traj_path)
    test_trajs = load_trajectories(test_traj_path)
    print(f"  Loaded {len(train_trajs)} train, {len(test_trajs)} test trajectories.")

    # Re-fit probes (same as Phase A) to get fitted probe objects
    print("[Phase B] Fitting probes on train data...")
    probe_df, probe_details, fitted_probes = run_probes(
        train_trajs, test_trajs, domain=args.domain
    )
    probe_a = fitted_probes.get("A")
    probe_c = fitted_probes.get("C")

    # ---- Run Oracle evaluation ----
    print("[Phase B] Evaluating Oracle transition coherence...")
    oracle_df = evaluate_oracle_coherence(
        test_trajs,
        probe_a=probe_a,
        probe_c=probe_c,
        max_depth=args.max_depth,
        domain=args.domain,
    )
    save_oracle_coherence(
        oracle_df,
        os.path.join(out, "coherence_oracle_depth.csv"),
        os.path.join(out, "coherence_oracle_depth.png"),
    )
    print("  Oracle coherence saved.")

    # ---- Sanity check ----
    valid_oracle = oracle_df[oracle_df["n_samples"] > 0]
    cos_vals = valid_oracle["cosine_similarity"].tolist()
    mse_vals = valid_oracle["mse"].tolist()
    print(f"  Sanity check: Oracle cosine = {cos_vals}")
    print(f"  Sanity check: Oracle MSE    = {mse_vals}")
    if all(c > 0.999 for c in cos_vals) and all(m < 1e-6 for m in mse_vals):
        print("  ✅ Oracle sanity check PASSED.")
    else:
        print("  ⚠️ Oracle sanity check FAILED — investigate!")

    # ---- Load Phase A comparison CSVs ----
    action_csv = os.path.join(out, "coherence_action_depth.csv")
    blind_csv = os.path.join(out, "coherence_blind_depth.csv")

    if not os.path.exists(action_csv) or not os.path.exists(blind_csv):
        print(
            "WARNING: Phase A coherence CSVs not found. "
            "Skipping comparison overlay and report."
        )
        return

    action_df = pd.read_csv(action_csv)
    blind_df = pd.read_csv(blind_csv)
    print(
        f"  Loaded Phase A action ({len(action_df)} rows) "
        f"and blind ({len(blind_df)} rows) CSVs."
    )

    # ---- Overlay plot ----
    print("[Phase B] Generating 4-way comparison overlay...")
    plot_comparison_overlay(
        oracle_df,
        action_df,
        blind_df,
        os.path.join(out, "coherence_comparison_overlay.png"),
    )

    # ---- Report ----
    print("[Phase B] Generating Phase B Oracle report...")
    generate_phase_b_report(
        oracle_df,
        action_df,
        blind_df,
        os.path.join(out, "phase_b_oracle_report.md"),
        args.domain,
    )

    print("\n[Phase B] Done. Artifacts:")
    for name in [
        "coherence_oracle_depth.csv",
        "coherence_oracle_depth.png",
        "coherence_comparison_overlay.png",
        "phase_b_oracle_report.md",
    ]:
        print(f"  - {os.path.join(out, name)}")


if __name__ == "__main__":
    main()
