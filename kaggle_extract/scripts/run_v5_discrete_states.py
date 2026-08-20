"""V5 orchestrator: do frozen LM hidden states admit a reusable discrete state?

Single entry point (mirrors :mod:`scripts.run_phase_a`) that runs the core V5
analysis on the cached trajectory files produced by Phase A:

    reports/trajectories/{train,val,test}.pt
        -> (C1) load / train the VQ codebook
        -> (C2) encode every trajectory to discrete codes
        -> (C5) codebook usage / collapse diagnostics
        -> (C3) action-blind transition predictability (top-1, entropy, ppl)
        -> (C4) empirical transition matrix + conditional entropy
        -> (C6) position-leakage test (codes as step indices?)
        -> (C9) discrete rollout coherence, decoded through probes A/B/C
        -> master report  reports/v5_discrete_state_report.md

C7 (permutation robustness) and C8 (cross-domain transfer) need their own data
extraction and are run from their own scripts; this orchestrator references
their CSV outputs in the master report if present.

The verdict logic is intentionally conservative: "reusable discrete states
exist" requires *all* of (a) non-collapsed codebook, (b) above-chance
transition predictability, (c) low entropy for a non-trivial code subset,
(d) low position leakage, (e) coherent multi-step rollout. Failing any one
gate is reported honestly as evidence *against* the hypothesis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from data_processing.trajectory_dataset import load_trajectories
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes,
    save_discrete_trajectories,
    all_codes,
)
from models.vq_state import VQStateQuantizer
from training.train_vq import train_vq_quantizer, VQTrainConfig
from evaluation.codebook_usage import (
    analyze_codebook_usage,
    save_codebook_usage_report,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    train_code_transition,
    build_transition_matrix,
    conditional_entropy,
    save_predictability_metrics,
    save_entropy_report,
)
from evaluation.position_leakage import (
    evaluate_position_leakage,
    save_position_leakage_report,
)
from evaluation.discrete_rollout import evaluate_discrete_rollout, save_discrete_rollout


def _load_vq(
    checkpoint_path: str, hidden_dim: int, num_codes: int, device
) -> VQStateQuantizer:
    """Load a VQ from checkpoint, or raise a clear error if missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VQ checkpoint not found: {checkpoint_path}. "
            f"Run training/train_vq.py first (see V5_READINESS.md)."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQStateQuantizer(
        hidden_dim=int(ckpt.get("hidden_dim", hidden_dim)),
        num_codes=int(ckpt.get("num_codes", num_codes)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_master_report(
    reports_dir: str,
    num_codes: int,
    usage_stats: dict,
    predict_metrics: dict,
    entropy: dict,
    position: dict,
    rollout_df,
    hidden_dim: int,
    n_train: int,
    n_test: int,
    smoke: bool,
) -> None:
    """Write reports/v5_discrete_state_report.md with the verdict."""
    # --- Gates (each must pass for a positive verdict) ----------------------
    # (a) Non-collapsed codebook: Gini < 0.9 AND active codes >= 10% of K.
    collapse_ok = usage_stats["collapse_score"] < 0.9
    active_ok = usage_stats["active_codes"] >= max(4, 0.1 * num_codes)
    # (b) Transition predictability beyond first-order counts: the MLP must
    # clear the bigram baseline (uniform chance was a misleadingly low bar).
    predict_ok = (
        predict_metrics["top1_accuracy"] > predict_metrics["bigram_baseline"] + 0.02
    )
    # (c) Low entropy for a non-trivial subset of states (>= 20% deterministic).
    entropy_ok = entropy["deterministic_state_frac"] >= 0.20
    # (d) Low position leakage: score not far above null + chance.
    leakage_ok = (
        position["position_predictability_score"]
        <= max(position["permutation_null_accuracy"], position["chance_accuracy"])
        + 0.10
    )
    # (e) Coherent rollout: code-match at depth 1 > 2 * chance.
    if len(rollout_df) and rollout_df["n_samples"].iloc[0] > 0:
        rollout_ok = rollout_df["code_match_accuracy"].iloc[0] > max(
            2.0 / num_codes, 0.20
        )
    else:
        rollout_ok = False

    gates = {
        "non_collapse": collapse_ok and active_ok,
        "predictable": predict_ok,
        "low_entropy_subset": entropy_ok,
        "low_position_leakage": leakage_ok,
        "coherent_rollout": rollout_ok,
    }
    n_pass = sum(gates.values())

    lines = ["# V5 Report — Reusable Discrete States?\n"]
    lines.append(
        "Research question: _do hidden-state trajectories of frozen language "
        "models admit a compact, predictive, reusable discrete state "
        "representation?_\n"
    )
    lines.append(f"- **Codebook size (K):** {num_codes}")
    lines.append(f"- **Hidden dim:** {hidden_dim}")
    lines.append(f"- **Trajectories:** train={n_train} test={n_test}")
    if smoke:
        lines.append(
            "\n> ⚠️ **Smoke run** — small N / synthetic data. Validate the "
            "pipeline, then re-run on the full Phase A cache for real numbers."
        )
    lines.append("\n---\n")

    lines.append("## 1. Codebook usage (collapse check)\n")
    lines.append(
        f"- Active codes: **{usage_stats['active_codes']}/{num_codes}** "
        f"(used: {usage_stats['used_codes']}, dead: {usage_stats['dead_codes']})\n"
        f"- Collapse score (Gini): **{usage_stats['collapse_score']:.3f}** "
        "(0 = uniform, 1 = fully collapsed)\n"
        f"- Perplexity: **{usage_stats['perplexity']:.1f}** "
        f"(effective codes used, max {num_codes})\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['non_collapse'] else 'FAIL'} "
        f"(Gini<0.9 AND active>=10% of K).\n"
    )

    lines.append("## 2. Transition predictability\n")
    lines.append(
        f"- Top-1 next-code accuracy (MLP): **{predict_metrics['top1_accuracy']:.3f}** "
        f"(majority = {predict_metrics['majority_baseline']:.3f}, "
        f"bigram = {predict_metrics['bigram_baseline']:.3f})\n"
        f"- Predictive entropy: **{predict_metrics['predictive_entropy_nats']:.3f} nats** "
        f"(perplexity {predict_metrics['predictive_perplexity']:.2f})\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['predictable'] else 'FAIL'} "
        f"(MLP top-1 > bigram + 0.02).\n"
    )

    lines.append("## 3. Transition entropy (planning state vs bucket)\n")
    lines.append(
        f"- Global H(Z_next | Z_current): **{entropy['global_entropy']:.3f} nats** "
        f"(perplexity {entropy['global_perplexity']:.2f})\n"
        f"- Fraction of states with near-deterministic successor "
        f"(H<0.5): **{entropy['deterministic_state_frac']*100:.1f}%**\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['low_entropy_subset'] else 'FAIL'} "
        f"(>= 20% deterministic states).\n"
    )

    lines.append("## 4. Position leakage (codes as step indices?)\n")
    lines.append(
        f"- Position predictability: **{position['position_predictability_score']:.3f}**\n"
        f"  - chance: {position['chance_accuracy']:.3f}, "
        f"permutation null: {position['permutation_null_accuracy']:.3f}\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['low_position_leakage'] else 'FAIL'} "
        f"(score <= max(null, chance) + 0.10). High leakage => codes are step "
        f"indices, not reusable states.\n"
    )

    lines.append("## 5. Discrete rollout coherence\n")
    if len(rollout_df) and rollout_df["n_samples"].iloc[0] > 0:
        lines.append(
            "| depth | code match | cosine | op acc (Probe C) | state acc (Probe A) |"
        )
        lines.append("|---|---|---|---|---|")
        for _, r in rollout_df.iterrows():
            if r["n_samples"] > 0:
                lines.append(
                    f"| {int(r['depth'])} | {r['code_match_accuracy']:.3f} | "
                    f"{r['cosine_similarity']:.3f} | {r['operator_accuracy']:.3f} | "
                    f"{r['state_probe_accuracy']:.3f} |"
                )
        lines.append(
            f"\nGate: {'PASS' if gates['coherent_rollout'] else 'FAIL'} "
            f"(depth-1 code match > max(2/K, 0.20)).\n"
        )
    else:
        lines.append("_No multi-step trajectories available to evaluate._\n")

    lines.append("\n---\n")
    lines.append("## Verdict\n")
    if n_pass == 5:
        verdict = (
            "**Reusable discrete states are supported.** All five gates pass: "
            "the codebook is non-collapsed, the next state is predictable, a "
            "non-trivial subset of states has deterministic successors, codes "
            "are not merely step indices, and the discrete rollout stays "
            "coherent beyond depth 1."
        )
    elif n_pass >= 3:
        verdict = (
            f"**Mixed evidence ({n_pass}/5 gates pass).** See the failing "
            f"gates above ({', '.join(k for k,v in gates.items() if not v)}) "
            f"for the specific failure mode."
        )
    else:
        verdict = (
            f"**Evidence AGAINST reusable discrete states ({n_pass}/5 gates "
            f"pass).** The codebook does not support a compact, predictive, "
            f"reusable discrete representation under this configuration."
        )
    lines.append(verdict)
    lines.append(
        "\n_Failure modes: high Gini => VQ collapse; low top-1 => "
        "unpredictable dynamics; high position leakage => codes are step "
        "indices; low deterministic-state fraction => every state is a "
        "compression bucket._\n"
    )

    # Reference C7/C8 outputs if present.
    perm_csv = os.path.join(reports_dir, "v5_permutation_robustness.csv")
    transfer_csv = os.path.join(reports_dir, "v5_cross_domain_transfer.csv")
    extras = []
    if os.path.exists(perm_csv):
        extras.append("v5_permutation_robustness.csv (C7)")
    if os.path.exists(transfer_csv):
        extras.append("v5_cross_domain_transfer.csv (C8)")
    if extras:
        lines.append(
            "\n## Supplementary (run separately)\n"
            + "\n".join(f"- `{e}`" for e in extras)
            + "\n"
        )

    out_path = os.path.join(reports_dir, "v5_discrete_state_report.md")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMaster report: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the V5 discrete-state analysis")
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Holds trajectories/{train,val,test}.pt; receives v5_* outputs",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Holds (or receives) vq_state.pt",
    )
    parser.add_argument(
        "--vq_checkpoint",
        type=str,
        default=None,
        help="VQ checkpoint filename (default: vq_state.pt in checkpoints_dir)",
    )
    parser.add_argument("--num_codes", type=int, default=256)
    parser.add_argument(
        "--train_vq",
        action="store_true",
        help="Train the VQ from trajectories/train.pt if no checkpoint exists "
        "(otherwise the checkpoint is required)",
    )
    parser.add_argument("--vq_epochs", type=int, default=50)
    parser.add_argument("--vq_batch_size", type=int, default=256)
    parser.add_argument("--transition_epochs", type=int, default=30)
    parser.add_argument(
        "--rollout_max_depth",
        type=int,
        default=8,
        help="0 = dynamic (95th pct of trajectory lengths)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Mark the run as a smoke test in the report",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.reports_dir
    disc_dir = os.path.join(out, "discrete_trajectories")
    os.makedirs(out, exist_ok=True)
    os.makedirs(disc_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    print("[V5] Loading cached trajectories...")
    trajs_train = load_trajectories(os.path.join(out, "trajectories", "train.pt"))
    trajs_val = load_trajectories(os.path.join(out, "trajectories", "val.pt"))
    trajs_test = load_trajectories(os.path.join(out, "trajectories", "test.pt"))
    if not trajs_train:
        raise RuntimeError("train.pt is empty — run Phase A first.")
    hidden_dim = trajs_train[0].hidden_dim

    # ------------------------------------------------------------------ #
    # C1 — load or train the VQ codebook.
    ckpt_name = args.vq_checkpoint or "vq_state.pt"
    ckpt_path = os.path.join(args.checkpoints_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"[V5/C1] Loading VQ from {ckpt_path}")
        vq = _load_vq(ckpt_path, hidden_dim, args.num_codes, device)
    elif args.train_vq:
        print("[V5/C1] Training VQ from train.pt...")
        vq = train_vq_quantizer(
            trajs_train,
            trajs_val,
            config=VQTrainConfig(
                num_codes=args.num_codes,
                epochs=args.vq_epochs,
                batch_size=args.vq_batch_size,
                seed=args.seed,
            ),
        )
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": vq.state_dict(),
                "hidden_dim": hidden_dim,
                "num_codes": vq.num_codes,
            },
            ckpt_path,
        )
    else:
        raise FileNotFoundError(
            f"No VQ checkpoint at {ckpt_path}. Re-run with --train_vq, or run "
            f"training/train_vq.py first."
        )

    # ------------------------------------------------------------------ #
    # C2 — encode every split to discrete codes and cache them.
    print("[V5/C2] Encoding trajectories to discrete codes...")
    disc_train = encode_trajectories_to_codes(vq, trajs_train)
    disc_val = encode_trajectories_to_codes(vq, trajs_val)
    disc_test = encode_trajectories_to_codes(vq, trajs_test)
    for name, d in [("train", disc_train), ("val", disc_val), ("test", disc_test)]:
        save_discrete_trajectories(d, os.path.join(disc_dir, f"{name}.pt"))
    n_train_states = int(all_codes(disc_train).shape[0])
    n_test_states = int(all_codes(disc_test).shape[0])
    print(f"  train states: {n_train_states}, test states: {n_test_states}")

    # ------------------------------------------------------------------ #
    # C5 — codebook usage.
    print("[V5/C5] Codebook usage analysis...")
    codes_train = all_codes(disc_train)
    usage = analyze_codebook_usage(codes_train, vq.num_codes)
    save_codebook_usage_report(
        usage,
        os.path.join(out, "v5_codebook_usage.csv"),
        os.path.join(out, "v5_codebook_usage.png"),
    )
    print(
        f"  active={usage['active_codes']}/{vq.num_codes} "
        f"gini={usage['collapse_score']:.3f} perplexity={usage['perplexity']:.1f}"
    )

    # ------------------------------------------------------------------ #
    # C3 — transition predictability.
    print("[V5/C3] Transition predictability (action-blind MLP)...")
    _, predict = train_code_transition(
        disc_train,
        disc_val,
        num_codes=vq.num_codes,
        config=CodeTransitionConfig(epochs=args.transition_epochs, seed=args.seed),
    )
    save_predictability_metrics(
        predict, os.path.join(out, "v5_transition_predictability.csv")
    )
    print(
        f"  top1={predict['top1_accuracy']:.3f} "
        f"(majority={predict['majority_baseline']:.3f}, "
        f"bigram={predict['bigram_baseline']:.3f}) "
        f"entropy={predict['predictive_entropy_nats']:.3f}"
    )

    # ------------------------------------------------------------------ #
    # C4 — transition entropy.
    print("[V5/C4] Transition entropy analysis...")
    T = build_transition_matrix(disc_train, vq.num_codes)
    code_freqs = usage["freqs"]
    entropy = conditional_entropy(T, code_freqs=code_freqs)
    active_mask = code_freqs > 0
    save_entropy_report(
        T,
        entropy,
        os.path.join(out, "v5_transition_entropy.csv"),
        os.path.join(out, "v5_transition_entropy.png"),
        active_mask=active_mask,
    )
    print(
        f"  global H={entropy['global_entropy']:.3f} "
        f"det_frac={entropy['deterministic_state_frac']*100:.1f}%"
    )

    # ------------------------------------------------------------------ #
    # C6 — position leakage.
    print("[V5/C6] Position leakage test...")
    position = evaluate_position_leakage(
        disc_train, disc_test, vq.num_codes, seed=args.seed
    )
    save_position_leakage_report(position, os.path.join(out, "v5_position_leakage.csv"))
    print(
        f"  score={position['position_predictability_score']:.3f} "
        f"(null={position['permutation_null_accuracy']:.3f}, "
        f"chance={position['chance_accuracy']:.3f})"
    )

    # ------------------------------------------------------------------ #
    # C9 — discrete rollout. Fit linear probes A/B/C on the continuous train
    # states so we can decode the rolled-out code -> continuous -> probes.
    print("[V5/C9] Discrete rollout coherence...")
    from evaluation.probes import run_probes

    _, _, fitted = run_probes(trajs_train, trajs_test)
    probe_a, probe_b, probe_c = fitted.get("A"), fitted.get("B"), fitted.get("C")

    # Re-train the transition model on discrete codes (cheap; reuses C3 model).
    code_model, _ = train_code_transition(
        disc_train,
        disc_val,
        num_codes=vq.num_codes,
        config=CodeTransitionConfig(epochs=args.transition_epochs, seed=args.seed),
    )
    max_depth = args.rollout_max_depth if args.rollout_max_depth > 0 else None
    rollout_df = evaluate_discrete_rollout(
        code_model,
        vq,
        disc_test,
        probe_a=probe_a,
        probe_b=probe_b,
        probe_c=probe_c,
        max_depth=max_depth,
    )
    save_discrete_rollout(
        rollout_df,
        os.path.join(out, "v5_discrete_rollout.csv"),
        os.path.join(out, "v5_discrete_rollout.png"),
    )

    # ------------------------------------------------------------------ #
    generate_master_report(
        reports_dir=out,
        num_codes=vq.num_codes,
        usage_stats=usage,
        predict_metrics=predict,
        entropy=entropy,
        position=position,
        rollout_df=rollout_df,
        hidden_dim=hidden_dim,
        n_train=len(trajs_train),
        n_test=len(trajs_test),
        smoke=args.smoke,
    )
    print("\n[V5] Done. v5_* artifacts in", out)


if __name__ == "__main__":
    main()
