"""Phase A orchestrator: the full latent-planning viability diagnostic.

Runs the complete pipeline end to end:

    generate/load problems
      -> extract teacher trajectories (frozen LM hidden states)
      -> train latent transition model      T(h_t, a_t) -> h_{t+1}
      -> train diagnostic next-token decoder
      -> coherence rollout evaluation        (coherence_depth.csv / .png)
      -> linear representation probes         (probe_results.csv / probe_report.md)
      -> master report                        (reports/phase_a_report.md)

The pipeline is model-agnostic. Use ``--model`` to switch between the full
TinyLlama teacher (real results) and a tiny random model (fast smoke test of the
machinery). All numeric artifacts are tagged with the model that produced them.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import (
    build_trajectories,
    load_problems,
    save_trajectories,
)
from training.train_transition import (
    train_transition_model,
    TransitionTrainConfig,
    save_history_csv,
)
from training.train_decoder import train_decoder_model, DecoderTrainConfig
from evaluation.coherence import evaluate_coherence, save_coherence
from evaluation.probes import (
    run_probes,
    save_probe_results,
    generate_probe_report,
)
from evaluation.plotter import plot_training_curves

DEFAULT_TEACHER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


# --------------------------------------------------------------------------- #
def maybe_generate_data(data_dir: str, sizes, domain: str = "countdown", base_seed: int = 42) -> None:
    """Generate data if the split files are missing/too small.

    Parameters
    ----------
    domain : str
        ``"countdown"`` (default, V5.3-compatible) or ``"game24"``.
    base_seed : int
        Base seed for generation to ensure disjoint splits.
    """
    if domain == "game24":
        from scripts.generate_game24_dataset import generate_dataset
    else:
        from scripts.generate_countdown_dataset import generate_dataset

    # sizes = [train, val, test, test_ood]
    files = {"train": sizes[0], "val": sizes[1], "test": sizes[2]}
    split_seeds = {
        "train": base_seed,
        "val": base_seed + 1,
        "test": base_seed + 2,
    }
    for split, n in files.items():
        path = os.path.join(data_dir, f"{split}.jsonl")
        existing = 0
        if os.path.exists(path):
            with open(path) as f:
                existing = sum(1 for _ in f)
        if existing < n:
            print(f"  generating {n} '{split}' problems -> {path}")
            generate_dataset(n, path, seed=split_seeds[split])
            
    if len(sizes) > 3:
        n_ood = sizes[3]
        path_ood = os.path.join(data_dir, "test_ood.jsonl")
        existing_ood = 0
        if os.path.exists(path_ood):
            with open(path_ood) as f:
                existing_ood = sum(1 for _ in f)
        if existing_ood < n_ood:
            print(f"  generating {n_ood} 'test_ood' problems -> {path_ood}")
            generate_dataset(n_ood, path_ood, is_ood=True, seed=base_seed + 3)


def build_split_trajectories(model, tokenizer, data_dir, layer, cap, out_dir, batch_size=32):
    trajs = {}
    splits = ["train", "val", "test"]
    if os.path.exists(os.path.join(data_dir, "test_ood.jsonl")):
        splits.append("test_ood")
        
    for split in splits:
        problems = load_problems(os.path.join(data_dir, f"{split}.jsonl"))
        if cap:
            problems = problems[:cap]
        print(f"  [{split}] building trajectories from {len(problems)} problems...")
        t = build_trajectories(model, tokenizer, problems, layer=layer, batch_size=batch_size)
        save_trajectories(t, os.path.join(out_dir, "trajectories", f"{split}.pt"))
        n_trans = sum(tr.num_steps for tr in t)
        print(f"  [{split}] {len(t)} trajectories, {n_trans} transitions")
        trajs[split] = t
        
    print("\n[1.5] Verifying Dataset Disjointness (Leakage Check & De-duplication)")
    def get_problem_hash(t):
        return (t.target, tuple(sorted(t.numbers)))
        
    train_hashes = set(get_problem_hash(t) for t in trajs.get("train", []))
    
    # Filter val
    if "val" in trajs:
        orig_val = len(trajs["val"])
        trajs["val"] = [t for t in trajs["val"] if get_problem_hash(t) not in train_hashes]
        val_hashes = set(get_problem_hash(t) for t in trajs["val"])
        dropped_val = orig_val - len(trajs["val"])
        if dropped_val > 0:
            print(f"  [WARNING] Dropped {dropped_val} leaking trajectories from Val set.")
            
    # Filter test
    if "test" in trajs:
        orig_test = len(trajs["test"])
        trajs["test"] = [t for t in trajs["test"] if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]
        dropped_test = orig_test - len(trajs["test"])
        if dropped_test > 0:
            print(f"  [WARNING] Dropped {dropped_test} leaking trajectories from Test set.")
            
    # Filter test_ood
    if "test_ood" in trajs:
        orig_ood = len(trajs["test_ood"])
        trajs["test_ood"] = [t for t in trajs["test_ood"] if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]
        dropped_ood = orig_ood - len(trajs["test_ood"])
        if dropped_ood > 0:
            print(f"  [WARNING] Dropped {dropped_ood} leaking trajectories from Test OOD set.")
            
    print(f"train: {len(trajs.get('train', []))}")
    print(f"val: {len(trajs.get('val', []))}")
    print(f"test: {len(trajs.get('test', []))}")
    print(f"test_ood: {len(trajs.get('test_ood', []))}")

    print("  [OK] Datasets de-duplicated.\n")
    
    return trajs


# --------------------------------------------------------------------------- #
def generate_master_report(
    coherence_df, probe_df, probe_details, transition_history,
    meta: dict, md_path: str,
) -> None:
    import numpy as np

    cov = coherence_df[coherence_df["n_samples"] > 0]

    # Coherence summary.
    def _depth_where(col, thresh, above=True):
        if col not in cov.columns:
            return 0
        ok = cov[cov[col] >= thresh] if above else cov[cov[col] <= thresh]
        return int(ok["depth"].max()) if len(ok) else 0

    cos_horizon = _depth_where("cosine_similarity", 0.9)
    op_horizon = _depth_where("operator_accuracy", 0.5)
    probe_horizon = _depth_where("state_probe_accuracy", 0.5)
    max_depth_avail = int(cov["depth"].max()) if len(cov) else 0
    final_val_mse = (transition_history[-1].get("eval_loss")
                     if transition_history else None)

    # Probe summary: which quantities are linearly decodable.
    chance = probe_details["chance"]
    chance_map = {row["probe"][0]: c for row, c in
                  zip([r for _, r in probe_df.iterrows()],
                      [chance[k] for k in ["A", "B", "C", "D"]])}
    encoded = []
    for _, r in probe_df.iterrows():
        k = r["probe"][0]
        base = chance_map[k]
        auc = r["auc"]
        if (not np.isnan(auc) and auc >= 0.65) or (r["accuracy"] - base >= 0.10):
            encoded.append(r["probe"])

    survives_beyond_3 = op_horizon > 3 or cos_horizon > 3
    info_present = len(encoded) > 0

    lines = []
    lines.append("# Phase A Report — Is Latent Planning Viable?\n")
    lines.append(f"- **Teacher model:** `{meta['model']}`")
    lines.append(f"- **Hidden layer probed:** {meta['layer']}")
    lines.append(f"- **Hidden dim:** {meta['hidden_dim']}")
    lines.append(
        f"- **Trajectories:** train={meta['n_train']} val={meta['n_val']} "
        f"test={meta['n_test']}")
    if meta.get("smoke"):
        lines.append(
            "\n> ⚠️ **Smoke-test run** with a randomly-initialized model. "
            "The numbers below validate the *pipeline*, not the science. "
            "Re-run with the TinyLlama teacher for real conclusions.")
    lines.append("\n---\n")

    lines.append("## 1. Do hidden states contain reasoning information?\n")
    if info_present:
        lines.append(
            "**Yes.** The following quantities are linearly decodable from the "
            "frozen hidden state (above majority-class baseline):")
        for p in encoded:
            lines.append(f"- {p}")
    else:
        lines.append(
            "**Not clearly.** No probed quantity was linearly decodable above "
            "baseline. See `probe_report.md`.")
    lines.append(f"\nSee `probe_results.csv` and `probe_report.md` for full metrics.\n")

    lines.append("## 2. How quickly does coherence decay?\n")
    if max_depth_avail:
        lines.append(
            f"Rollouts were evaluated to depth {max_depth_avail} "
            "(limited by solution length in the data).")
        lines.append("\n| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in cov.iterrows():
            lines.append(
                f"| {int(r['depth'])} | {r['cosine_similarity']:.3f} | "
                f"{r['operator_accuracy']:.3f} | {r['teacher_operator_accuracy']:.3f} | "
                f"{r['state_probe_accuracy']:.3f} | {r['teacher_state_probe_accuracy']:.3f} | "
                f"{r['mse']:.4f} |")
        lines.append(
            f"\nCosine similarity stays >= 0.90 through depth **{cos_horizon}**; "
            f"rollout operator accuracy stays >= 0.50 through depth **{op_horizon}**; "
            f"rollout state probe accuracy stays >= 0.50 through depth **{probe_horizon}**.")
    else:
        lines.append("No multi-step trajectories were available to evaluate.")
    lines.append("\nSee `coherence_depth.csv` / `coherence_depth.png`.\n")

    lines.append("## 3. What is the effective planning horizon?\n")
    horizon = max(cos_horizon, op_horizon, probe_horizon)
    lines.append(
        f"Taking the depth at which the rolled-out latent still resembles "
        f"the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) "
        f"and state information (acc >= 0.5), the **effective horizon is ~{horizon} step(s)**.")
    if final_val_mse is not None:
        lines.append(f"\nSingle-step transition validation MSE: {final_val_mse:.4f}.")
    lines.append("")

    lines.append("## 4. Is latent planning worth pursuing?\n")
    if info_present and survives_beyond_3:
        verdict = ("**Promising.** Hidden states encode task-relevant structure "
                   "*and* coherence survives beyond depth 3 — the regime where "
                   "search would operate. Proceed to Phase B.")
    elif info_present and not survives_beyond_3:
        verdict = ("**Mixed / representation-limited bottleneck.** The "
                   "representation encodes reasoning information, but the learned "
                   "dynamics lose coherence by depth 3. The bottleneck is "
                   "*dynamics*, not representation — worth pursuing only with a "
                   "stronger transition model.")
    elif not info_present and survives_beyond_3:
        verdict = ("**Inconclusive.** Latents stay self-consistent under rollout "
                   "but probes find little decodable task information — the model "
                   "may be coasting on uninformative directions. Investigate "
                   "representation before pursuing planning.")
    else:
        verdict = ("**Not yet.** Hidden states show weak task information and "
                   "coherence decays quickly. Latent planning is unlikely to work "
                   "with this representation/dynamics pair as-is.")
    lines.append(verdict)
    lines.append(
        "\n**Bottleneck diagnosis:** "
        + ("representation" if (not info_present and survives_beyond_3)
           else "dynamics" if (info_present and not survives_beyond_3)
           else "neither (proceed)" if (info_present and survives_beyond_3)
           else "both") + ".")
    lines.append(
        "\n_Decision rule: 'viable' requires both (a) task information linearly "
        "present in the representation and (b) rollout coherence surviving beyond "
        "depth 3._\n")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run the Phase A diagnostic pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--domain", type=str, choices=["countdown", "game24"],
                        default="countdown",
                        help="Problem domain (default: countdown for V5.3 compat)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--cap", type=int, default=None,
                        help="Max problems per split (for quick runs)")
    parser.add_argument("--generate", nargs="+", type=int, metavar="SIZE",
                        default=[5000, 500, 1000, 500], help="Generate data if missing. E.g. 5000 500 1000 500 (train val test test_ood)")
    parser.add_argument("--transition_epochs", type=int, default=100)
    parser.add_argument("--decoder_epochs", type=int, default=30)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extracting hidden states")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load the teacher in 4-bit NF4 (inference-only; for 7B/8B models)")
    parser.add_argument("--smoke", action="store_true",
                        help="Mark outputs as a smoke test (random model)")
    parser.add_argument("--extract_only", action="store_true", help="Only generate trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    if args.generate:
        print("[0/6] Ensuring data exists...")
        maybe_generate_data(args.data_dir, args.generate, domain=args.domain, base_seed=args.seed)

    print(f"[1/6] Loading frozen teacher: {args.model}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype,
                       load_in_4bit=args.load_in_4bit)
    tokenizer = load_tokenizer(model_id=args.model)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))

    print("[2/6] Extracting teacher trajectories...")
    trajs = build_split_trajectories(
        model, tokenizer, args.data_dir, args.layer, args.cap, out, args.batch_size)
    hidden_dim = trajs["train"][0].hidden_dim

    if args.extract_only:
        print("\n[*] Extract only mode. Done.")
        return

    print("[3/6] Training latent transition models (Action-Conditioned + Action-Blind)...")
    tcfg = TransitionTrainConfig(epochs=args.transition_epochs)
    
    # Train Action-Conditioned
    print("  -> Action-Conditioned")
    tcfg.use_action = True
    tmodel_action = train_transition_model(trajs["train"], trajs["val"], tcfg)
    torch.save({"state_dict": tmodel_action.state_dict(), "hidden_dim": hidden_dim},
               os.path.join(out, "transition_model_action.pt"))
    save_history_csv(tcfg.history, os.path.join(out, "transition_action_log.csv"))
    plot_training_curves(os.path.join(out, "transition_action_log.csv"),
                         os.path.join(out, "transition_action_curve.png"))
                         
    # Train Action-Blind
    print("  -> Action-Blind")
    tcfg_blind = TransitionTrainConfig(epochs=args.transition_epochs, use_action=False)
    tmodel_blind = train_transition_model(trajs["train"], trajs["val"], tcfg_blind)
    torch.save({"state_dict": tmodel_blind.state_dict(), "hidden_dim": hidden_dim},
               os.path.join(out, "transition_model_blind.pt"))
    save_history_csv(tcfg_blind.history, os.path.join(out, "transition_blind_log.csv"))
    plot_training_curves(os.path.join(out, "transition_blind_log.csv"),
                         os.path.join(out, "transition_blind_curve.png"))

    print("[4/6] Training diagnostic decoder...")
    dcfg = DecoderTrainConfig(epochs=args.decoder_epochs)
    decoder = train_decoder_model(trajs["train"], vocab_size, trajs["val"], dcfg)

    print("[5/6] Representation probes + reports...")
    probe_df, probe_details, fitted_probes = run_probes(trajs["train"], trajs["test"])
    save_probe_results(probe_df, os.path.join(out, "probe_results.csv"))
    generate_probe_report(probe_df, probe_details, os.path.join(out, "probe_report.md"))

    print("[6/7] Coherence rollout evaluation...")
    probe_a = fitted_probes.get("A")
    probe_b = fitted_probes.get("B")
    probe_c = fitted_probes.get("C")
    probe_d = fitted_probes.get("D")
    
    coh_df = evaluate_coherence(
        tmodel_action, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth
    )
    save_coherence(coh_df, os.path.join(out, "coherence_action_depth.csv"),
                   os.path.join(out, "coherence_action_depth.png"))
                   
    coh_blind_df = evaluate_coherence(
        tmodel_blind, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth
    )
    save_coherence(coh_blind_df, os.path.join(out, "coherence_blind_depth.csv"),
                   os.path.join(out, "coherence_blind_depth.png"))
                   
    if "test_ood" in trajs:
        coh_ood_df = evaluate_coherence(
            tmodel_action, trajs["test_ood"], 
            probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
            max_depth=args.max_depth
        )
        save_coherence(coh_ood_df, os.path.join(out, "coherence_ood_depth.csv"),
                       os.path.join(out, "coherence_ood_depth.png"))

    print("[7/7] Phase B: Oracle Transition Diagnostic...")
    from evaluation.oracle_coherence import (
        evaluate_oracle_coherence,
        save_oracle_coherence,
        plot_comparison_overlay,
    )
    oracle_df = evaluate_oracle_coherence(
        trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d,
        max_depth=args.max_depth,
    )
    save_oracle_coherence(
        oracle_df,
        os.path.join(out, "coherence_oracle_depth.csv"),
        os.path.join(out, "coherence_oracle_depth.png"),
    )
    # Sanity check (ignore empty rollout bins)
    valid_oracle = oracle_df[
        (oracle_df["n_samples"] > 0)
        & oracle_df["cosine_similarity"].notna()
        & oracle_df["mse"].notna()
    ]

    cos_vals = valid_oracle["cosine_similarity"].tolist()
    mse_vals = valid_oracle["mse"].tolist()

    if len(valid_oracle) == 0:
        print("  [WARNING] Oracle sanity check skipped (no valid rollout depths).")
    elif all(c > 0.999 for c in cos_vals) and all(m < 1e-6 for m in mse_vals):
        print("  [PASS] Oracle sanity check.")
    else:
        print("  [FAIL] Oracle sanity check - investigate!")
        print(f"    Cosine: {cos_vals}")
        print(f"    MSE:    {mse_vals}")

    # 4-way overlay plot
    plot_comparison_overlay(
        oracle_df, coh_df, coh_blind_df,
        os.path.join(out, "coherence_comparison_overlay.png"),
    )

    # Phase B report
    from scripts.run_phase_b import generate_phase_b_report
    generate_phase_b_report(
        oracle_df, coh_df, coh_blind_df,
        os.path.join(out, "phase_b_oracle_report.md"),
    )

    meta = {
        "model": args.model, "layer": args.layer, "hidden_dim": hidden_dim,
        "n_train": len(trajs["train"]), "n_val": len(trajs["val"]),
        "n_test": len(trajs["test"]), "smoke": args.smoke,
    }
    # Master report uses the IID Action-Conditioned dataframe
    generate_master_report(coh_df, probe_df, probe_details, tcfg.history, meta,
                           os.path.join("reports", "phase_a_report.md"))

    print("\nDone. Artifacts written to:")
    for name in ["coherence_action_depth.csv", "coherence_action_depth.png",
                 "coherence_blind_depth.png", "coherence_oracle_depth.csv",
                 "coherence_comparison_overlay.png",
                 "phase_b_oracle_report.md",
                 "probe_results.csv", "probe_report.md",
                 "transition_action_curve.png"]:
        print(f"  - {os.path.join(out, name)}")
    print("  - reports/phase_a_report.md")


if __name__ == "__main__":
    main()
