"""V5.2 Experiment 3 — layer-sweep aggregation and reporting (pure logic).

All V5.2 results so far probed a single layer (TinyLlama's final, ``-1``). This
module holds the depth-independent parts of Experiment 3: place each swept
layer on the Experiment-2 floor→ceiling axes (det-frac / entropy / MLP), pick
the layer where Countdown's action-conditioned structure is strongest, and
render the comparison report. The GPU extraction + per-layer VQ/analysis lives
in ``scripts/run_layer_sweep.py``; everything here is plain numpy/stdlib so it
is cheap to unit-test.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def _frac(flo: float, ceil: float, x: float) -> float:
    """Fraction of the way from ``flo`` (Exp2 noise floor) to ``ceil`` (Exp2
    positive-control ceiling) that ``x`` sits. For entropy the floor is high
    and the ceiling low, so passing ``flo``/``ceil`` in that order yields the
    fraction of the way *down* toward the ceiling. NaN if the anchors coincide.
    """
    return (x - flo) / (ceil - flo) if abs(ceil - flo) > 1e-9 else float("nan")


def summarize_sweep(rows: List[Dict], floor: Dict, ceiling: Dict) -> Dict:
    """Locate the layer with the strongest action-conditioned structure.

    Args:
        rows: per-layer metric dicts, each with ``layer``, ``det_action``,
            ``H_action`` and ``mlp_z_op``.
        floor, ceiling: Experiment-2 anchors (noise floor / positive control),
            each with ``det_action``, ``H_action`` and ``mlp_z_op``.

    Returns each layer's position on the three floor→ceiling axes, the best
    layer (highest det-frac, tie-broken by MLP(z,op)), and whether that best
    layer reaches the reusable-state regime (Exp1's det-frac ≥ 0.20 bar *and*
    at least halfway to the ceiling).
    """
    per_layer = [
        {
            "layer": r["layer"],
            "pos_det": _frac(
                floor["det_action"], ceiling["det_action"], r["det_action"]
            ),
            "pos_mlp": _frac(floor["mlp_z_op"], ceiling["mlp_z_op"], r["mlp_z_op"]),
            "pos_H": _frac(floor["H_action"], ceiling["H_action"], r["H_action"]),
        }
        for r in rows
    ]
    best = max(rows, key=lambda r: (r["det_action"], r["mlp_z_op"]))
    best_pos = next(o for o in per_layer if o["layer"] == best["layer"])
    reaches = best["det_action"] >= 0.20 and best_pos["pos_det"] >= 0.5
    return {
        "per_layer": per_layer,
        "best_layer": best["layer"],
        "best_det_action": best["det_action"],
        "best_pos_det": best_pos["pos_det"],
        "reaches_ceiling": bool(reaches),
    }


def _f(v, p=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}f}"


def build_markdown(
    rows: List[Dict],
    summary: Dict,
    floor: Dict,
    ceiling: Dict,
    observed: Optional[Dict],
    config: Dict,
) -> str:
    """Assemble the Experiment-3 markdown report from collected per-layer rows."""
    pos_by_layer = {o["layer"]: o for o in summary["per_layer"]}
    L = []
    L.append("# V5.2 · Experiment 3 — Layer Sweep\n")
    L.append(
        "> Every V5.2 result so far probed TinyLlama's **final** layer "
        "(`-1`). A planning-state representation need not live there — "
        "mid-stack residuals often carry more abstract structure. We "
        "re-extract hidden states at several depths and run the *identical* "
        "VQ + action-conditioned pipeline (Exp1) at each, to find whether "
        "any layer pushes Countdown toward the Exp2 reusable-state "
        "ceiling.\n"
    )

    L.append("\n## Setup\n")
    L.append(
        f"- **Model:** `{config['model']}` ({config['num_layers']} layers). "
        f"Layers swept: {', '.join(str(x) for x in config['layers'])} "
        "(index into `hidden_states`; the final layer = "
        f"{config['num_layers']}).\n"
    )
    L.append(
        f"- **Data:** capped Countdown — {config['cap_train']} train / "
        f"{config['cap_val']} val problems (val de-duplicated against "
        "train by `(target, sorted numbers)`).\n"
    )
    L.append(
        f"- **Per layer:** VQ K={config['num_codes']} (data-dependent init, "
        "raw states) → action-conditioned models A–F on held-out val + "
        "transition structure on train, plus codebook usage and position "
        "leakage. Identical to Experiment 1.\n"
    )
    L.append(
        "- **Anchors:** Experiment-2 noise floor and positive-control "
        "ceiling (same K, pipeline) bound each action-conditioned axis.\n"
    )

    L.append("\n## Per-layer metrics\n")
    L.append(
        "| Layer | Active | Perplex. | Pos-leak | Bigram(z) | MLP(z) | "
        "Act-bigram | MLP(z,op) | H(z'\\|z) | H(z'\\|z,op) | Det(z,op) | "
        "→ceiling |"
    )
    L.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |"
    )
    for r in rows:
        p = pos_by_layer[r["layer"]]
        L.append(
            f"| {r['layer']} | {r['active_codes']} | {_f(r['perplexity'], 1)} | "
            f"{_f(r['position_leakage'])} | {_f(r['bigram_z'])} | {_f(r['mlp_z'])} | "
            f"{_f(r['action_bigram'])} | {_f(r['mlp_z_op'])} | {_f(r['H_state'])} | "
            f"{_f(r['H_action'])} | {_f(r['det_action'])} | "
            f"{_f(p['pos_det'] * 100, 0)}% |"
        )

    L.append(
        "\n_Det(z,op) = share of transition mass from near-deterministic "
        "`(z,op)` conditionings (H<0.5 nats, seen ≥10×). →ceiling = "
        "position on the det-frac axis between the Exp2 noise floor and "
        "positive-control ceiling._\n"
    )

    L.append("\n## Anchors (Experiment 2)\n")
    L.append("| | Det(z,op) | H(z'\\|z,op) | MLP(z,op) |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append(
        f"| Noise floor | {_f(floor['det_action'])} | {_f(floor['H_action'])} | "
        f"{_f(floor['mlp_z_op'])} |"
    )
    L.append(
        f"| Positive-control ceiling | {_f(ceiling['det_action'])} | "
        f"{_f(ceiling['H_action'])} | {_f(ceiling['mlp_z_op'])} |"
    )
    if observed is not None and observed.get("det_action") is not None:
        L.append(
            f"| Cached layer −1 (Exp1, full data) | {_f(observed['det_action'])} | "
            f"{_f(observed['H_action'])} | {_f(observed['mlp_z_op'])} |"
        )

    best = summary["best_layer"]
    best_row = next(r for r in rows if r["layer"] == best)
    L.append("\n---\n## Verdict\n")
    if summary["reaches_ceiling"]:
        L.append(
            f"**Depth matters.** Layer **{best}** reaches the reusable-state "
            f"regime: det-frac(z,op) = {_f(best_row['det_action'])} "
            f"({_f(summary['best_pos_det'] * 100, 0)}% of the way to the "
            f"positive-control ceiling), H(z'|z,op) = {_f(best_row['H_action'])} "
            f"nats. The Exp1/Exp2 conclusion — drawn at the final layer — "
            f"understated the structure available deeper in the stack.\n"
        )
    else:
        L.append(
            f"**The conclusion is robust to layer choice.** The strongest "
            f"layer ({best}) still only reaches det-frac(z,op) = "
            f"{_f(best_row['det_action'])} "
            f"({_f(summary['best_pos_det'] * 100, 0)}% of the way from the "
            f"noise floor to the positive-control ceiling) — short of the "
            f"reusable-state regime. No probed depth turns Countdown's "
            f"hidden-state trajectory into crisp, reusable discrete "
            f"dynamics; the partial, graded structure found at the final "
            f"layer (Exp1/Exp2) is representative of the whole stack.\n"
        )

    L.append(
        "\n_Caveats: single model/task/seed; capped data (numbers shift "
        "slightly vs full-data Exp1); op-type action; layer subset, not "
        "every layer. det-frac is a strict threshold metric — see Exp2 for "
        "why the soft entropy/MLP axes can read higher._\n"
    )
    L.append("\n_Artifacts: `reports/layer_sweep.json`, `reports/layer_sweep.png`._\n")
    return "\n".join(L)


def build_plot(rows: List[Dict], floor: Dict, ceiling: Dict, png_path: str) -> None:
    """Plot the headline action-conditioned metrics against layer depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in rows]
    det = [r["det_action"] for r in rows]
    H = [r["H_action"] for r in rows]
    mlp = [r["mlp_z_op"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(layers, det, "o-", color="tab:green", label="det-frac(z,op)")
    ax1.plot(layers, mlp, "s-", color="tab:blue", label="MLP(z,op) top1")
    ax1.axhline(
        ceiling["det_action"],
        color="tab:green",
        ls="--",
        alpha=0.6,
        label=f"ceiling det {ceiling['det_action']:.2f}",
    )
    ax1.axhline(
        floor["det_action"],
        color="tab:gray",
        ls=":",
        alpha=0.6,
        label=f"floor det {floor['det_action']:.2f}",
    )
    ax1.axhline(ceiling["mlp_z_op"], color="tab:blue", ls="--", alpha=0.4)
    ax1.axhline(floor["mlp_z_op"], color="tab:blue", ls=":", alpha=0.4)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("det-frac / accuracy")
    ax1.set_title("Action-conditioned determinism vs depth")
    ax1.legend(fontsize=8)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.plot(layers, H, "o-", color="tab:red", label="H(z'|z,op)")
    ax2.axhline(
        ceiling["H_action"],
        color="tab:red",
        ls="--",
        alpha=0.6,
        label=f"ceiling H {ceiling['H_action']:.2f}",
    )
    ax2.axhline(
        floor["H_action"],
        color="tab:gray",
        ls=":",
        alpha=0.6,
        label=f"floor H {floor['H_action']:.2f}",
    )
    ax2.set_xlabel("layer")
    ax2.set_ylabel("H(z'|z,op)  [nats]")
    ax2.set_title("Action-conditioned entropy vs depth")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
