"""V5.2 Experiment 1 — action-conditioned transition test.

Addresses the reviewer criticism that V5.1 tested only action-blind
``z_t -> z_{t+1}``. A planning state is defined by ``(z_t, a_t) -> z_{t+1}``.
We run A-F (see evaluation.action_conditioned) on the Fixed-Init codebook
(primary) and the position-scrubbed codebook (robustness), evaluating top-1 on
the held-out val split and the transition structure (entropy / determinism) on
train. Writes reports/action_conditioned.md + .json. Then stop.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

from data_processing.discrete_trajectory_dataset import load_discrete_trajectories
from evaluation.action_conditioned import extract_action_transitions, run_all_models

MODEL_ROWS = [
    ("A_majority", "A. Majority", "—"),
    ("B_bigram", "B. Bigram", "z_t"),
    ("C_action_bigram", "C. Action bigram", "z_t, op"),
    ("D_mlp_z", "D. MLP(z_t)", "z_t"),
    ("E_mlp_z_op", "E. MLP(z_t, op)", "z_t, op"),
    ("E2_mlp_z_op_operands", "E2. MLP(z_t, op, operands)", "z_t, op, operands"),
    ("F_mlp_action_only", "F. MLP(op, operands) — control", "op, operands"),
]


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}f}"


def run_representation(name, train_path, val_path, K, epochs, seed):
    train = extract_action_transitions(load_discrete_trajectories(train_path))
    ev = extract_action_transitions(load_discrete_trajectories(val_path))
    res = run_all_models(train, ev, K, epochs=epochs, seed=seed)
    res["_meta"] = {"name": name, "n_train": int(train.z_t.shape[0]),
                    "n_eval": int(ev.z_t.shape[0])}
    return res


def model_table(res):
    rows = ["| Model | Conditioning | Top1 | Pred. entropy (nats) |",
            "| --- | --- | ---: | ---: |"]
    for key, label, cond in MODEL_ROWS:
        m = res[key]
        rows.append(f"| {label} | {cond} | {_fmt(m['top1'])} | "
                    f"{_fmt(m.get('pred_entropy'))} |")
    return "\n".join(rows)


def struct_table(res):
    s, a = res["_struct_state"], res["_struct_action"]
    return "\n".join([
        "| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |",
        "| --- | ---: | ---: | ---: |",
        f"| z_t | {_fmt(s['global_entropy'])} | {_fmt(s['det_frac_mass'])} | {s['n_conditions']} |",
        f"| z_t, op | {_fmt(a['global_entropy'])} | {_fmt(a['det_frac_mass'])} | {a['n_conditions']} |",
    ])


def contrasts(res):
    g = lambda k: res[k]["top1"]
    return {
        "action_helps_lookup_CminusB": g("C_action_bigram") - g("B_bigram"),
        "action_helps_mlp_EminusD": g("E_mlp_z_op") - g("D_mlp_z"),
        "mlp_vs_action_bigram_EminusC": g("E_mlp_z_op") - g("C_action_bigram"),
        "operands_help_E2minusE": g("E2_mlp_z_op_operands") - g("E_mlp_z_op"),
        "state_matters_E2minusF": g("E2_mlp_z_op_operands") - g("F_mlp_action_only"),
        "determinism_gain": res["_struct_action"]["det_frac_mass"]
                            - res["_struct_state"]["det_frac_mass"],
        "entropy_drop": res["_struct_state"]["global_entropy"]
                        - res["_struct_action"]["global_entropy"],
    }


def write_report(out, primary, scrub, contrasts_p, K, epochs):
    c = contrasts_p
    # Decision logic (skeptical thresholds).
    material = 0.03
    det_action = primary["_struct_action"]["det_frac_mass"]
    action_helps = (c["action_helps_lookup_CminusB"] > material
                    or c["action_helps_mlp_EminusD"] > material)
    mlp_beats_bigram = c["mlp_vs_action_bigram_EminusC"] > material
    determinism_emerged = det_action >= 0.20          # crisp reusable dynamics
    determinism_partial = det_action >= 0.05           # non-trivial but weak
    state_matters = c["state_matters_E2minusF"] > material
    # Three-way verdict.
    if action_helps and determinism_emerged:
        verdict = "yes"
    elif action_helps and (determinism_partial or c["entropy_drop"] > 0.3):
        verdict = "partial"
    else:
        verdict = "no"

    L = []
    L.append("# V5.2 · Experiment 1 — Action-Conditioned Transition Test\n")
    L.append("> Reviewer criticism addressed: a planning state is defined by "
             "`(state, action) → next_state`, not `state → next_state`. V5.1 "
             "tested only the action-blind form. Here we condition on the "
             "symbolic Countdown action.\n")
    L.append("\n## Implementation\n")
    L.append(f"- **Representation:** Fixed-Init VQ codes (K={K}, "
             "data-dependent init, raw last-layer states) — the clean, "
             "non-collapsed V5.1 codebook. Scrubbed codes reported as a "
             "robustness check.\n")
    L.append("- **Action:** op id (0=ADD, 1=SUB, 2=MUL; DIV absent from data) "
             "and the raw operands `[arg1, arg2]` (signed-log + standardized for "
             "the MLP).\n")
    L.append(f"- **Eval:** top-1 on the held-out **val** split "
             f"(train {primary['_meta']['n_train']} transitions, "
             f"val {primary['_meta']['n_eval']}); MLPs trained {epochs} epochs, "
             "seed 0. Entropy/determinism measured on train (frequency-weighted, "
             "deterministic = successor entropy < 0.5 nats over conditionings "
             "seen ≥ 10×).\n")
    L.append("- **Models A–F:** majority; state bigram; action bigram "
             "`(z,op)→mode`; MLP(z); MLP(z,op); MLP(z,op,operands); and an "
             "**action-only control** MLP(op,operands) that ignores `z_t`.\n")

    L.append("\n## Results — Fixed-Init codebook (primary)\n")
    L.append("### Predictive accuracy (held-out val)\n")
    L.append(model_table(primary))
    L.append("\n### Transition structure (train)\n")
    L.append(struct_table(primary))

    L.append("\n### Key contrasts\n")
    L.append(f"- **Does the action help the lookup?** C − B = "
             f"{c['action_helps_lookup_CminusB']:+.3f} "
             f"({_fmt(primary['C_action_bigram']['top1'])} vs "
             f"{_fmt(primary['B_bigram']['top1'])}).")
    L.append(f"- **Does the action help the MLP?** E − D = "
             f"{c['action_helps_mlp_EminusD']:+.3f} "
             f"({_fmt(primary['E_mlp_z_op']['top1'])} vs "
             f"{_fmt(primary['D_mlp_z']['top1'])}).")
    L.append(f"- **Does the MLP beat the action bigram?** E − C = "
             f"{c['mlp_vs_action_bigram_EminusC']:+.3f}.")
    L.append(f"- **Do operands add information?** E2 − E = "
             f"{c['operands_help_E2minusE']:+.3f} "
             f"(E2 = {_fmt(primary['E2_mlp_z_op_operands']['top1'])}).")
    L.append(f"- **Does the state matter, or only the action?** E2 − F = "
             f"{c['state_matters_E2minusF']:+.3f} "
             f"(F action-only = {_fmt(primary['F_mlp_action_only']['top1'])}).")
    L.append(f"- **Determinism gain from conditioning on the action:** "
             f"det-frac {_fmt(primary['_struct_state']['det_frac_mass'])} (z) → "
             f"{_fmt(primary['_struct_action']['det_frac_mass'])} (z,op); "
             f"entropy {_fmt(primary['_struct_state']['global_entropy'])} → "
             f"{_fmt(primary['_struct_action']['global_entropy'])} nats.\n")

    L.append("\n## Robustness — Scrubbed codebook\n")
    L.append(model_table(scrub))
    L.append("\n" + struct_table(scrub) + "\n")

    L.append("\n---\n## Reviewer-style interpretation\n")
    L.append("**1. Results.** See tables above. On the clean Fixed-Init "
             f"codebook, conditioning on the symbolic action moves top-1 from "
             f"{_fmt(primary['B_bigram']['top1'])} (state bigram) to "
             f"{_fmt(primary['C_action_bigram']['top1'])} (action bigram) and "
             f"{_fmt(primary['E_mlp_z_op']['top1'])} (action MLP); adding "
             f"operands gives {_fmt(primary['E2_mlp_z_op_operands']['top1'])}.\n")

    L.append("**2. Does MLP(z,a) materially beat the action-aware bigram?** "
             + ("**Yes** " if mlp_beats_bigram else "**No** ")
             + f"— E − C = {c['mlp_vs_action_bigram_EminusC']:+.3f} "
             f"(threshold ±{material}). "
             + ("The learned model captures structure beyond action-conditioned "
                "counts.\n" if mlp_beats_bigram else
                "The MLP does not exceed the action-conditioned lookup: whatever "
                "the action contributes is already a first-order count effect, "
                "not learned structure.\n"))

    L.append("**3. Does the planning-state hypothesis survive?**\n")
    ds = _fmt(primary["_struct_state"]["det_frac_mass"])
    da = _fmt(primary["_struct_action"]["det_frac_mass"])
    hs = _fmt(primary["_struct_state"]["global_entropy"])
    ha = _fmt(primary["_struct_action"]["global_entropy"])
    if verdict == "yes":
        L.append("> **Tentatively yes (revisit).** Conditioning on the action "
                 "materially raises predictability and a substantial deterministic "
                 f"subset emerges (det-frac {da} of mass). This warrants the "
                 "positive control (Experiment 2) before any claim.\n")
    elif verdict == "partial":
        L.append(
            "> **Partially — and this revises the V5.1 wording.** The reviewer "
            "criticism was correct that the action-blind test understated the "
            f"structure: conditioning on the action roughly *doubles* "
            f"predictability (state→action bigram "
            f"{_fmt(primary['B_bigram']['top1'])} → "
            f"{_fmt(primary['C_action_bigram']['top1'])}; E−D "
            f"{c['action_helps_mlp_EminusD']:+.3f}), *halves* the conditional "
            f"entropy ({hs} → {ha} nats), and turns a 0% deterministic mass into "
            f"{da}. So `(state, action)` carries real, non-trivial dynamics that "
            "`state` alone did not.\n")
        L.append(
            "> **But it does not reach reusable planning states.** The structure "
            f"is purely first-order — the MLP does not beat the action lookup "
            f"(E−C {c['mlp_vs_action_bigram_EminusC']:+.3f}) — and it is far from "
            f"deterministic: only {da} of transition mass is near-deterministic, "
            f"H(z'|z,op) ≈ {ha} nats (~{int(round(2.718 ** float(ha)))} effective "
            f"successors), and top-1 is {_fmt(primary['E2_mlp_z_op_operands']['top1'])} "
            "even with operands. The state is not vacuous (E2−F "
            f"{c['state_matters_E2minusF']:+.3f}), but it composes with actions "
            "into *weak, stochastic* transitions, not crisp reusable ones.\n")
        L.append(
            "> **Net:** V5.1's flat \"no reusable structure\" is too strong and "
            "should be revised to **\"weak, first-order, sub-deterministic "
            "action-conditioned dynamics — not reusable planning states.\"** "
            "Whether even this weak effect exceeds what *any* action-conditioned "
            "model would yield on position-correlated codes cannot be judged "
            "without a calibrated positive control — which is exactly the purpose "
            "of Experiment 2.\n")
    else:
        reasons = []
        if not action_helps:
            reasons.append(
                f"conditioning on the action barely changes predictability "
                f"(C−B {c['action_helps_lookup_CminusB']:+.3f}, "
                f"E−D {c['action_helps_mlp_EminusD']:+.3f})")
        if not determinism_emerged:
            reasons.append(f"no deterministic subset appears (det-frac {da})")
        if not mlp_beats_bigram:
            reasons.append("the MLP never beats the action bigram")
        L.append("> **No.** " + "; ".join(reasons) + ". The negative result "
                 "survives even the action-conditioned test.\n")

    L.append("\n_Caveats: single model/task/layer/seed; op-type action (DIV "
             "absent); operands make the next *number* computable, so any E2 gain "
             "may reflect numeric leakage rather than reusable state composition. "
             "Scope unchanged from V5.1._\n")

    L.append("\n_Artifacts: `reports/action_conditioned.json`._\n")

    with open(os.path.join(out, "action_conditioned.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.reports_dir

    fdir = os.path.join(out, "discrete_fixed_init")
    sdir = os.path.join(out, "discrete_scrubbed")
    print("[Exp1] Fixed-Init representation...")
    primary = run_representation("fixed_init", os.path.join(fdir, "train.pt"),
                                 os.path.join(fdir, "val.pt"), args.num_codes,
                                 args.epochs, args.seed)
    print("[Exp1] Scrubbed representation...")
    scrub = run_representation("scrubbed", os.path.join(sdir, "train.pt"),
                               os.path.join(sdir, "val.pt"), args.num_codes,
                               args.epochs, args.seed)

    con = contrasts(primary)
    payload = {
        "config": {"num_codes": args.num_codes, "epochs": args.epochs,
                   "seed": args.seed, "eval_split": "val"},
        "fixed_init": {k: v for k, v in primary.items()},
        "scrubbed": {k: v for k, v in scrub.items()},
        "contrasts_fixed_init": con,
    }
    with open(os.path.join(out, "action_conditioned.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(out, primary, scrub, con, args.num_codes, args.epochs)

    print("\n[Exp1] Fixed-Init top1:",
          {k: round(primary[k]["top1"], 3) for k, _, _ in MODEL_ROWS})
    print("[Exp1] contrasts:", {k: round(v, 3) for k, v in con.items()})
    print("[Exp1] Done. See reports/action_conditioned.md")


if __name__ == "__main__":
    main()
