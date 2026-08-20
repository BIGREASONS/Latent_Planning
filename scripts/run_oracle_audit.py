"""Oracle Ceiling Audit.

Cleanly separates *representation interchangeability* from *dynamics
robustness* so that a failed transition model cannot masquerade as a failed
representation.

Oracle hierarchy (see project spec):

* Oracle 0 — Teacher Ceiling: probes A/B/C/D on the exact teacher state h_t.
* Oracle 1 — True-State Transition Baseline: T(h_t, a_t) -> ĥ_{t+1}, Probe A.
* Oracle 2A — State-Swap Representation Audit: for an anchor state h, find a
  partner h' with the *same target + same symbolic state* but a *different
  trajectory and different action history*; compare probes on h vs h' directly
  (no transition model).
* Oracle 2B — State-Swap Dynamics Audit: feed the anchor's intended action a_t
  to the swapped state, T(h', a_t) -> ĥ'_{t+1}, Probe A. The primary signal is
  ``delta_transition_accuracy = Oracle1 - Oracle2B`` on matched pairs.

This is strictly an evaluation audit. It runs no training and adds no new
experiments. Probes are fit on the train split; the audit (anchors + swap
partners) is drawn from a held-out split so partners are also held out. The
optional ``--audit train_test`` mode instead pools the train and test
trajectories as the anchor/partner pool (probe fitting is unchanged: probes
are still fit on the train split). Trajectories are concatenated, so every
trajectory keeps a distinct index and the "different trajectory" matching rule
is unchanged.

Outputs (exactly):
    reports/oracle_audit_metrics.csv
    reports/oracle_audit_coverage.md
    reports/oracle_audit_summary.md
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states
from evaluation.probes import (
    MultiLabelProbe,
    _make_probe,
    extract_probe_data,
)

LARGE_NUMBERS = [25, 50, 75, 100]
from models.transition_model import (
    TransitionModel,
    LinearTransitionModel,
    TransformerTransitionModel,
)


# --------------------------------------------------------------------------- #
# Probe fitting (reuses evaluation.probes infrastructure)
# --------------------------------------------------------------------------- #
class _ConstPredictor:
    """Degenerate predictor for a single-class training target."""

    def __init__(self, const: int):
        self.const = int(const)

    def predict(self, X):
        return np.full((X.shape[0],), self.const, dtype=np.int64)


def fit_all_probes(train_trajs):
    """Fit linear probes A (multi-label), B, C, D on teacher train states."""
    tr = extract_probe_data(train_trajs)
    X = tr["X"]

    probe_a = MultiLabelProbe()
    probe_a.fit(X, tr["A"])

    def _fit_single(y):
        classes = np.unique(y)
        if classes.shape[0] < 2:
            return _ConstPredictor(int(classes[0]) if classes.shape[0] else 0)
        clf = _make_probe()
        clf.fit(X, y)
        return clf

    return {
        "A": probe_a,
        "B": _fit_single(tr["B"]),
        "C": _fit_single(tr["C"]),
        "D": _fit_single(tr["D"]),
    }


# --------------------------------------------------------------------------- #
# Ground-truth Probe-A label (matches evaluation.probes.extract_probe_data)
# --------------------------------------------------------------------------- #
def a_label_at(numbers, operands, d):
    """Probe-A multi-label vector at depth ``d`` for {25,50,75,100}.

    0 = value never in the problem, 1 = in pool / not yet used as an operand,
    2 = already used as an operand in steps 0..d-1.
    """
    used = set()
    for i in range(d):
        used.add(int(operands[i][0].item()))
        used.add(int(operands[i][1].item()))
    row = []
    for v in LARGE_NUMBERS:
        if v not in numbers:
            row.append(0)
        elif v not in used:
            row.append(1)
        else:
            row.append(2)
    return row


# --------------------------------------------------------------------------- #
# State records + symbolic-state grouping
# --------------------------------------------------------------------------- #
def build_state_records(trajs):
    """Per-depth state instances with symbolic key, history, gt labels, action.

    Returns ``(records, groups)`` where ``groups`` maps a symbolic state to the
    list of record indices that share it. Depths invalidated by a teacher
    arithmetic mistake (``get_symbolic_states`` returns ``None``) are skipped.
    """
    records = []
    for ti, traj in enumerate(trajs):
        sym_list = get_symbolic_states(traj)  # len N+1; (sym, hist) or None
        N = traj.num_steps
        numbers = list(traj.numbers)
        operands = traj.operands  # (N, 2)
        states = traj.states  # (N+1, H)
        for d in range(N + 1):
            info = sym_list[d]
            if info is None:
                continue
            sym, hist = info
            has_next = d < N and sym_list[d + 1] is not None
            rec = {
                "traj_idx": ti,
                "depth": d,
                "sym": sym,
                "hist": hist,
                "h": states[d].numpy().astype(np.float32),
                "gtA": a_label_at(numbers, operands, d),
                "gtB": int(N - d),
                "gtC": int(traj.op_ids[d].item()) if d < N else None,
                "gtD": 1 if (N - d) <= 2 else 0,
                "has_next": has_next,
            }
            if has_next:
                rec["next_op"] = int(traj.op_ids[d].item())
                rec["next_operands"] = operands[d].numpy().astype(np.float32)
                rec["next_gtA"] = a_label_at(numbers, operands, d + 1)
            else:
                rec["next_op"] = None
                rec["next_operands"] = None
                rec["next_gtA"] = None
            records.append(rec)

    groups = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        groups[rec["sym"]].append(idx)
    return records, groups


# --------------------------------------------------------------------------- #
# Agreement / metric primitives
# --------------------------------------------------------------------------- #
def a_exact(p, r):
    return 1.0 if list(p) == list(r) else 0.0


def a_per_label(p, r):
    return sum(1.0 for a, b in zip(p, r) if a == b) / len(r)


def a_jaccard(p, r):
    """Jaccard over the set of still-available numbers (label == 1)."""
    P = {v for v, l in zip(LARGE_NUMBERS, p) if l == 1}
    R = {v for v, l in zip(LARGE_NUMBERS, r) if l == 1}
    union = P | R
    if not union:
        return 1.0
    return len(P & R) / len(union)


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _stratify(pairs):
    """pairs: list of (delta_depth, value) -> {'all': mean, dd: mean, ...}."""
    out = {"all": _mean([v for _, v in pairs])}
    by = collections.defaultdict(list)
    for dd, v in pairs:
        by[dd].append(v)
    for dd in sorted(by):
        out[dd] = _mean(by[dd])
    return out


def _counts_by_delta(pairs):
    by = collections.Counter(dd for _, _, dd in pairs)
    return {"all": len(pairs), **{dd: by[dd] for dd in sorted(by)}}


# --------------------------------------------------------------------------- #
# Confidence intervals (reporting only; no metric definition is changed)
# --------------------------------------------------------------------------- #
_Z95 = 1.959963984540054  # standard normal quantile for a two-sided 95% CI


def _mean_ci(values):
    """Mean and 95% CI of the mean over a 1-D sample (normal approximation).

    Used for the pair-level Oracle 2A / 2B metrics. The CI is computed on the
    sample of per-pair values; the point estimate is identical to ``_mean`` so
    no reported metric changes.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_lo": mean, "ci_hi": mean, "n": 1}
    se = float(arr.std(ddof=1)) / np.sqrt(n)
    return {
        "mean": mean,
        "ci_lo": mean - _Z95 * se,
        "ci_hi": mean + _Z95 * se,
        "n": int(n),
    }


def _proportion_ci(k, n):
    """Wilson 95% CI for a binomial proportion (used for coverage rate)."""
    if n == 0:
        return {
            "rate": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 0,
        }
    p = k / n
    denom = 1.0 + _Z95 * _Z95 / n
    center = (p + _Z95 * _Z95 / (2 * n)) / denom
    half = (_Z95 * np.sqrt(p * (1 - p) / n + _Z95 * _Z95 / (4 * n * n))) / denom
    return {"rate": p, "ci_lo": center - half, "ci_hi": center + half, "n": int(n)}


# --------------------------------------------------------------------------- #
# Transition model
# --------------------------------------------------------------------------- #
def load_transition_model(
    ckpt_path: str, transition_arch: str = "mlp"
) -> torch.nn.Module:
    """Load a transition model from checkpoint, predicting delta states."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Try to load config dict
    cfg = ckpt.get("config", {})
    hidden_dim = int(
        ckpt.get("hidden_dim")
        or cfg.get("hidden_dim")
        or sd.get(
            "net.2.weight", sd.get("net.weight", sd.get("state_proj_in.weight"))
        ).shape[0]
    )
    mlp_hidden_dim = cfg.get("mlp_hidden_dim", 512)
    predict_delta = cfg.get("predict_delta", True)
    use_action = cfg.get("use_action", any(k.startswith("action_encoder") for k in sd))
    op_embed_dim = cfg.get("op_embed_dim", 16)
    operand_mean = cfg.get("operand_mean", 0.0)
    operand_std = cfg.get("operand_std", 1.0)

    if transition_arch == "linear":
        model = LinearTransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )
    elif transition_arch == "transformer":
        model = TransformerTransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )
    else:
        model = TransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )

    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def transition_predict_a(model, probe_a, h_mat, op_arr, operands_arr, chunk=4096):
    """Apply T(h, a) over batched inputs, return Probe-A predictions (m, 4)."""
    preds = []
    for s in range(0, h_mat.shape[0], chunk):
        ht = torch.tensor(h_mat[s : s + chunk], dtype=torch.float32)
        op = torch.tensor(op_arr[s : s + chunk], dtype=torch.long)
        oper = torch.tensor(operands_arr[s : s + chunk], dtype=torch.float32)
        out = model(ht, op, oper).cpu().numpy()
        preds.append(probe_a.predict(out))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Oracle 0
# --------------------------------------------------------------------------- #
def oracle0_metrics(probes, test_trajs):
    te = extract_probe_data(test_trajs)
    X = te["X"]
    predA = probes["A"].predict(X)
    predB = probes["B"].predict(X)
    predC = probes["C"].predict(X)
    predD = probes["D"].predict(X)
    return {
        "n": int(X.shape[0]),
        "probeA_exact": _mean(
            [a_exact(predA[i], te["A"][i]) for i in range(len(predA))]
        ),
        "probeA_per_label": _mean(
            [a_per_label(predA[i], te["A"][i]) for i in range(len(predA))]
        ),
        "probeA_jaccard": _mean(
            [a_jaccard(predA[i], te["A"][i]) for i in range(len(predA))]
        ),
        "probeB_accuracy": float(np.mean(predB == te["B"])),
        "probeC_accuracy": float(np.mean(predC == te["C"])),
        "probeD_accuracy": float(np.mean(predD == te["D"])),
    }


# --------------------------------------------------------------------------- #
# Audit core (Oracle 1, 2A, 2B)
# --------------------------------------------------------------------------- #
def run_audit(
    records, groups, probes, transition_model, max_partners, smoke_pairs=False
):
    H = records[0]["h"].shape[0] if records else 0

    # Precompute Probe predictions on every state instance (one batched call).
    if records:
        h_mat = np.stack([r["h"] for r in records])
        predA = probes["A"].predict(h_mat)
        predB = probes["B"].predict(h_mat)
        predC = probes["C"].predict(h_mat)
        predD = probes["D"].predict(h_mat)
    else:
        predA = np.zeros((0, 4))
        predB = predC = predD = np.zeros((0,))
    for i, r in enumerate(records):
        r["predA"] = predA[i]
        r["predB"] = int(predB[i])
        r["predC"] = int(predC[i])
        r["predD"] = int(predD[i])

    anchors = [i for i, r in enumerate(records) if r["has_next"]]

    # ---- Coverage (shared pool for 2A and 2B) ----
    partners_of = {}
    for ai in anchors:
        a = records[ai]
        parts = [
            pi
            for pi in groups[a["sym"]]
            if records[pi]["traj_idx"] != a["traj_idx"]
            and records[pi]["hist"] != a["hist"]
        ]
        partners_of[ai] = parts
    with_partner = [ai for ai in anchors if partners_of[ai]]
    cov_ci = _proportion_ci(len(with_partner), len(anchors))
    coverage = {
        "total_states": len(anchors),
        "states_with_swap_partner": len(with_partner),
        "states_without_swap_partner": len(anchors) - len(with_partner),
        "coverage_rate": (len(with_partner) / len(anchors)) if anchors else 0.0,
        "coverage_rate_ci_lo": cov_ci["ci_lo"],
        "coverage_rate_ci_hi": cov_ci["ci_hi"],
        "total_state_instances": len(records),
    }

    # ---- Oracle 1 (true-state transition, full has_next pool) ----
    o1 = {
        "probeA_exact": float("nan"),
        "probeA_per_label": float("nan"),
        "probeA_jaccard": float("nan"),
        "n": len(anchors),
        "probeA_exact_ci_lo": float("nan"),
        "probeA_exact_ci_hi": float("nan"),
    }
    o1_by_anchor = {}  # ai -> (exact, per_label, jaccard) for matched delta
    if anchors and transition_model is not None:
        hm = np.stack([records[ai]["h"] for ai in anchors])
        opm = np.array([records[ai]["next_op"] for ai in anchors], dtype=np.int64)
        oprm = np.stack([records[ai]["next_operands"] for ai in anchors])
        pa = transition_predict_a(transition_model, probes["A"], hm, opm, oprm)
        ex, pl, jc = [], [], []
        for k, ai in enumerate(anchors):
            tgt = records[ai]["next_gtA"]
            e = a_exact(pa[k], tgt)
            p = a_per_label(pa[k], tgt)
            j = a_jaccard(pa[k], tgt)
            o1_by_anchor[ai] = (e, p, j)
            ex.append(e)
            pl.append(p)
            jc.append(j)
        o1 = {
            "probeA_exact": _mean(ex),
            "probeA_per_label": _mean(pl),
            "probeA_jaccard": _mean(jc),
            "n": len(anchors),
        }
        o1_ci = _mean_ci(ex)
        o1["probeA_exact_ci_lo"] = o1_ci["ci_lo"]
        o1["probeA_exact_ci_hi"] = o1_ci["ci_hi"]

    # ---- Build the valid swap-pair list (capped per anchor) ----
    pairs = []  # (anchor_idx, partner_idx, delta_depth)
    for ai in with_partner:
        a = records[ai]
        for pi in partners_of[ai][:max_partners]:
            p = records[pi]
            pairs.append((ai, pi, abs(a["depth"] - p["depth"])))

    if smoke_pairs:
        counts = _counts_by_delta(pairs)
        print(f"[Smoke Pairs] Built {len(pairs)} pairs.")
        print(f"[Smoke Pairs] Delta counts: {counts}")
        import sys

        sys.exit(0)

    # ---- Oracle 2A (representation agreement on h vs h') ----
    a2 = {}  # metric -> stratified dict
    acc = collections.defaultdict(list)  # metric -> [(dd, value)]
    for ai, pi, dd in pairs:
        a, p = records[ai], records[pi]
        acc["probeA_pred_exact"].append((dd, a_exact(a["predA"], p["predA"])))
        acc["probeA_pred_per_label"].append((dd, a_per_label(a["predA"], p["predA"])))
        acc["probeA_pred_jaccard"].append((dd, a_jaccard(a["predA"], p["predA"])))
        acc["probeB_pred_abs_diff"].append((dd, abs(a["predB"] - p["predB"])))
        acc["probeB_pred_agreement"].append(
            (dd, 1.0 if a["predB"] == p["predB"] else 0.0)
        )
        acc["probeC_pred_agreement"].append(
            (dd, 1.0 if a["predC"] == p["predC"] else 0.0)
        )
        acc["probeD_pred_agreement"].append(
            (dd, 1.0 if a["predD"] == p["predD"] else 0.0)
        )
        # Ground-truth label comparison for the same symbolic state
        acc["gtA_exact"].append((dd, a_exact(a["gtA"], p["gtA"])))
        acc["gtA_per_label"].append((dd, a_per_label(a["gtA"], p["gtA"])))
        acc["gtA_jaccard"].append((dd, a_jaccard(a["gtA"], p["gtA"])))
        acc["gtB_abs_diff"].append((dd, abs(a["gtB"] - p["gtB"])))
        acc["gtB_agreement"].append((dd, 1.0 if a["gtB"] == p["gtB"] else 0.0))
        if a["gtC"] is not None and p["gtC"] is not None:
            acc["gtC_agreement"].append((dd, 1.0 if a["gtC"] == p["gtC"] else 0.0))
        acc["gtD_agreement"].append((dd, 1.0 if a["gtD"] == p["gtD"] else 0.0))
    for m, lst in acc.items():
        a2[m] = _stratify(lst)
    a2_counts = _counts_by_delta(pairs)
    a2_ci = {m: _mean_ci([v for _, v in lst]) for m, lst in acc.items()}

    # ---- Oracle 2B (dynamics on swapped state) + matched delta ----
    b2 = {}
    b2_acc = collections.defaultdict(list)
    delta_acc = collections.defaultdict(list)
    b2_ci = {}
    if pairs and transition_model is not None:
        hm = np.stack([records[pi]["h"] for _, pi, _ in pairs])
        opm = np.array([records[ai]["next_op"] for ai, _, _ in pairs], dtype=np.int64)
        oprm = np.stack([records[ai]["next_operands"] for ai, _, _ in pairs])
        pa = transition_predict_a(transition_model, probes["A"], hm, opm, oprm)
        for k, (ai, pi, dd) in enumerate(pairs):
            tgt = records[ai]["next_gtA"]
            e = a_exact(pa[k], tgt)
            pl = a_per_label(pa[k], tgt)
            jc = a_jaccard(pa[k], tgt)
            b2_acc["probeA_exact"].append((dd, e))
            b2_acc["probeA_per_label"].append((dd, pl))
            b2_acc["probeA_jaccard"].append((dd, jc))
            o1e = o1_by_anchor.get(ai, (float("nan"),) * 3)[0]
            delta_acc["oracle1_paired_exact"].append((dd, o1e))
            delta_acc["delta_transition_accuracy"].append((dd, o1e - e))
        for m, lst in b2_acc.items():
            b2[m] = _stratify(lst)
        for m, lst in delta_acc.items():
            b2[m] = _stratify(lst)
        for m, lst in b2_acc.items():
            b2_ci[m] = _mean_ci([v for _, v in lst])
        for m, lst in delta_acc.items():
            # delta is defined only where the paired Oracle 1 exact is finite.
            b2_ci[m] = _mean_ci([v for _, v in lst if not np.isnan(v)])
    b2_counts = _counts_by_delta(pairs)

    return {
        "coverage": coverage,
        "oracle1": o1,
        "oracle2a": a2,
        "oracle2a_counts": a2_counts,
        "oracle2a_ci": a2_ci,
        "oracle2b": b2,
        "oracle2b_counts": b2_counts,
        "oracle2b_ci": b2_ci,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6f}"
    return str(v)


def write_metrics_csv(o0, res, path):
    import csv

    rows = [("oracle", "metric", "delta_depth", "value", "n", "ci_lo", "ci_hi")]

    for m in [
        "probeA_exact",
        "probeA_per_label",
        "probeA_jaccard",
        "probeB_accuracy",
        "probeC_accuracy",
        "probeD_accuracy",
    ]:
        rows.append(("oracle0", m, "all", o0[m], o0["n"], "", ""))

    # Coverage rate with its 95% CI (Wilson interval).
    c = res["coverage"]
    rows.append(
        (
            "coverage",
            "coverage_rate",
            "all",
            c["coverage_rate"],
            c["total_states"],
            c["coverage_rate_ci_lo"],
            c["coverage_rate_ci_hi"],
        )
    )

    o1 = res["oracle1"]
    for m in ["probeA_exact", "probeA_per_label", "probeA_jaccard"]:
        lo = o1["probeA_exact_ci_lo"] if m == "probeA_exact" else ""
        hi = o1["probeA_exact_ci_hi"] if m == "probeA_exact" else ""
        rows.append(("oracle1", m, "all", o1[m], o1["n"], lo, hi))

    a2, a2c = res["oracle2a"], res["oracle2a_counts"]
    a2ci = res.get("oracle2a_ci", {})
    for m in sorted(a2):
        for dd, val in a2[m].items():
            ci = a2ci.get(m, {}) if dd == "all" else {}
            rows.append(
                (
                    "oracle2a",
                    m,
                    dd,
                    val,
                    a2c.get(dd, ""),
                    ci.get("ci_lo", ""),
                    ci.get("ci_hi", ""),
                )
            )

    b2, b2c = res["oracle2b"], res["oracle2b_counts"]
    b2ci = res.get("oracle2b_ci", {})
    for m in sorted(b2):
        for dd, val in b2[m].items():
            ci = b2ci.get(m, {}) if dd == "all" else {}
            rows.append(
                (
                    "oracle2b",
                    m,
                    dd,
                    val,
                    b2c.get(dd, ""),
                    ci.get("ci_lo", ""),
                    ci.get("ci_hi", ""),
                )
            )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(
                [
                    r[0],
                    r[1],
                    r[2],
                    _fmt(r[3]) if isinstance(r[3], float) else r[3],
                    r[4],
                    _fmt(r[5]) if isinstance(r[5], float) else r[5],
                    _fmt(r[6]) if isinstance(r[6], float) else r[6],
                ]
            )


def write_coverage_md(res, path):
    c = res["coverage"]
    a2c = res["oracle2a_counts"]
    lines = ["# Oracle Audit — Coverage\n"]
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| total_states | {c['total_states']} |")
    lines.append(f"| states_with_swap_partner | {c['states_with_swap_partner']} |")
    lines.append(
        f"| states_without_swap_partner | {c['states_without_swap_partner']} |"
    )
    lines.append(f"| coverage_rate | {_fmt(c['coverage_rate'])} |")
    lines.append(
        f"| coverage_rate_95ci | [{_fmt(c['coverage_rate_ci_lo'])}, "
        f"{_fmt(c['coverage_rate_ci_hi'])}] |"
    )
    lines.append(f"| total_state_instances | {c['total_state_instances']} |")
    lines.append("")
    lines.append("## Swap pairs by delta_depth\n")
    lines.append("| delta_depth | n_pairs |")
    lines.append("|---|---|")
    for dd in [k for k in a2c if k != "all"]:
        lines.append(f"| {dd} | {a2c[dd]} |")
    lines.append(f"| all | {a2c.get('all', 0)} |")
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _kv_table(title, items):
    out = [f"## {title}\n", "| metric | value |", "|---|---|"]
    for k, v in items:
        out.append(f"| {k} | {_fmt(v)} |")
    out.append("")
    return out


def _strat_table(title, strat, metrics):
    dds = sorted({dd for m in metrics for dd in strat.get(m, {}) if dd != "all"})
    header = "| metric | all | " + " | ".join(f"dd={dd}" for dd in dds) + " |"
    sep = "|---|---|" + "---|" * len(dds)
    out = [f"## {title}\n", header, sep]
    for m in metrics:
        s = strat.get(m, {})
        cells = [_fmt(s.get("all", float("nan")))]
        cells += [_fmt(s.get(dd, float("nan"))) for dd in dds]
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _ci_table(title, ci, metrics):
    """95% CI table (all-pairs aggregate) for the given metrics."""
    out = [
        f"## {title}\n",
        "| metric | mean | ci_lo | ci_hi | n |",
        "|---|---|---|---|---|",
    ]
    for m in metrics:
        c = ci.get(m, {})
        out.append(
            f"| {m} | {_fmt(c.get('mean', float('nan')))} | "
            f"{_fmt(c.get('ci_lo', float('nan')))} | "
            f"{_fmt(c.get('ci_hi', float('nan')))} | {c.get('n', 0)} |"
        )
    out.append("")
    return out


def write_summary_md(o0, res, path):
    c = res["coverage"]
    o1 = res["oracle1"]
    a2 = res["oracle2a"]
    b2 = res["oracle2b"]
    a2ci = res.get("oracle2a_ci", {})
    b2ci = res.get("oracle2b_ci", {})

    lines = ["# Oracle Audit — Summary\n"]

    # 1. Coverage
    lines += _kv_table(
        "1. Coverage",
        [
            ("total_states", c["total_states"]),
            ("states_with_swap_partner", c["states_with_swap_partner"]),
            ("states_without_swap_partner", c["states_without_swap_partner"]),
            ("coverage_rate", c["coverage_rate"]),
            ("coverage_rate_ci_lo", c["coverage_rate_ci_lo"]),
            ("coverage_rate_ci_hi", c["coverage_rate_ci_hi"]),
            ("total_state_instances", c["total_state_instances"]),
        ],
    )

    # 2. Oracle 0
    lines += _kv_table(
        "2. Oracle 0 — Teacher Ceiling",
        [
            ("probeA_exact", o0["probeA_exact"]),
            ("probeA_per_label", o0["probeA_per_label"]),
            ("probeA_jaccard", o0["probeA_jaccard"]),
            ("probeB_accuracy", o0["probeB_accuracy"]),
            ("probeC_accuracy", o0["probeC_accuracy"]),
            ("probeD_accuracy", o0["probeD_accuracy"]),
            ("n", o0["n"]),
        ],
    )

    # 3. Oracle 1
    lines += _kv_table(
        "3. Oracle 1 — True-State Transition",
        [
            ("probeA_exact", o1["probeA_exact"]),
            ("probeA_exact_ci_lo", o1["probeA_exact_ci_lo"]),
            ("probeA_exact_ci_hi", o1["probeA_exact_ci_hi"]),
            ("probeA_per_label", o1["probeA_per_label"]),
            ("probeA_jaccard", o1["probeA_jaccard"]),
            ("n", o1["n"]),
        ],
    )

    # 4. Oracle 2A agreement (stratified)
    lines += _strat_table(
        "4. Oracle 2A — Representation Agreement",
        a2,
        [
            "probeA_pred_exact",
            "probeA_pred_per_label",
            "probeA_pred_jaccard",
            "probeB_pred_abs_diff",
            "probeB_pred_agreement",
            "probeC_pred_agreement",
            "probeD_pred_agreement",
            "gtA_exact",
            "gtA_per_label",
            "gtA_jaccard",
            "gtB_abs_diff",
            "gtB_agreement",
            "gtC_agreement",
            "gtD_agreement",
        ],
    )

    # 4b. Oracle 2A agreement — 95% CI (all pairs)
    lines += _ci_table(
        "4b. Oracle 2A — Representation Agreement (95% CI, all pairs)",
        a2ci,
        [
            "probeA_pred_exact",
            "probeA_pred_per_label",
            "probeA_pred_jaccard",
            "probeB_pred_abs_diff",
            "probeB_pred_agreement",
            "probeC_pred_agreement",
            "probeD_pred_agreement",
            "gtA_exact",
            "gtA_per_label",
            "gtA_jaccard",
            "gtB_abs_diff",
            "gtB_agreement",
            "gtC_agreement",
            "gtD_agreement",
        ],
    )

    # 5. Oracle 2B transition metrics (stratified)
    lines += _strat_table(
        "5. Oracle 2B — Swapped-State Transition",
        b2,
        ["probeA_exact", "probeA_per_label", "probeA_jaccard"],
    )

    # 6. Oracle 2B delta metrics (stratified)
    lines += _strat_table(
        "6. Oracle 2B — Delta",
        b2,
        ["oracle1_paired_exact", "delta_transition_accuracy"],
    )

    # 6b. Oracle 2B transition + delta — 95% CI (all pairs)
    lines += _ci_table(
        "6b. Oracle 2B — Transition + Delta (95% CI, all pairs)",
        b2ci,
        [
            "probeA_exact",
            "probeA_per_label",
            "probeA_jaccard",
            "oracle1_paired_exact",
            "delta_transition_accuracy",
        ],
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def _resolve_audit_paths(reports_dir, audit):
    traj = os.path.join(reports_dir, "trajectories")
    if audit == "all":
        return [os.path.join(traj, f"{n}.pt") for n in ("train", "val", "test")]
    if audit == "train_test":
        return [os.path.join(traj, f"{n}.pt") for n in ("train", "test")]
    return [os.path.join(traj, f"{audit}.pt")]


def main():
    ap = argparse.ArgumentParser(description="Oracle Ceiling Audit")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument(
        "--transition_arch",
        type=str,
        default="mlp",
        choices=["mlp", "linear", "transformer"],
        help="Architecture of the transition model",
    )
    ap.add_argument(
        "--train_traj",
        default=None,
        help="Trajectory file used to fit probes (default <reports>/trajectories/train.pt)",
    )
    ap.add_argument(
        "--audit",
        choices=["test", "val", "train", "all", "train_test"],
        default="test",
        help="Split providing audit anchors + swap partners (held out from "
        "probe fit). 'train_test' pools the train and test trajectories "
        "as the anchor/partner pool.",
    )
    ap.add_argument(
        "--transition_ckpt",
        default=None,
        help="Action-conditioned transition checkpoint "
        "(default <reports>/transition_model_action.pt)",
    )
    ap.add_argument(
        "--max_partners",
        type=int,
        default=50,
        help="Cap on swap partners per anchor (bounds pair count)",
    )
    ap.add_argument(
        "--smoke_pairs",
        action="store_true",
        help="Lightweight validation: build pairs, run count, then exit.",
    )
    args = ap.parse_args()

    train_path = args.train_traj or os.path.join(
        args.reports_dir, "trajectories", "train.pt"
    )
    audit_paths = _resolve_audit_paths(args.reports_dir, args.audit)
    ckpt_path = args.transition_ckpt or os.path.join(
        args.reports_dir, "transition_model_action.pt"
    )

    if not os.path.exists(train_path):
        print(f"ERROR: train trajectories not found: {train_path}")
        sys.exit(1)
    for p in audit_paths:
        if not os.path.exists(p):
            print(f"ERROR: audit trajectories not found: {p}")
            sys.exit(1)

    print(f"[Oracle Audit] Fitting probes on {train_path}")
    train_trajs = load_trajectories(train_path)
    probes = fit_all_probes(train_trajs)

    print(f"[Oracle Audit] Loading audit trajectories: {audit_paths}")
    audit_trajs = []
    for p in audit_paths:
        audit_trajs.extend(load_trajectories(p))
    print(f"  {len(audit_trajs)} audit trajectories")

    transition_model = None
    if os.path.exists(ckpt_path):
        print(f"[Oracle Audit] Loading transition model: {ckpt_path}")
        transition_model = load_transition_model(ckpt_path, args.transition_arch)
    else:
        print(
            f"WARNING: transition checkpoint missing ({ckpt_path}); "
            "Oracle 1 / 2B metrics will be nan."
        )

    print("[Oracle Audit] Oracle 0 (teacher ceiling)...")
    o0 = oracle0_metrics(probes, audit_trajs)

    print("[Oracle Audit] Building state records and swap pairs...")
    records, groups = build_state_records(audit_trajs)
    res = run_audit(
        records, groups, probes, transition_model, args.max_partners, args.smoke_pairs
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "oracle_audit_metrics.csv")
    cov_path = os.path.join(args.out_dir, "oracle_audit_coverage.md")
    sum_path = os.path.join(args.out_dir, "oracle_audit_summary.md")

    write_metrics_csv(o0, res, csv_path)
    write_coverage_md(res, cov_path)
    write_summary_md(o0, res, sum_path)

    c = res["coverage"]
    print("\n[Oracle Audit] Done.")
    print(
        f"  coverage: {c['states_with_swap_partner']}/{c['total_states']} anchors "
        f"with a swap partner (rate={c['coverage_rate']:.3f})"
    )
    for pth in (csv_path, cov_path, sum_path):
        print(f"  - {pth}")


if __name__ == "__main__":
    main()
