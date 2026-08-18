"""V5.2 Experiment 3 — layer sweep (orchestrator).

Re-extracts TinyLlama hidden-state trajectories at several depths and runs the
*identical* Experiment-1 pipeline (VQ → action-conditioned models A–F →
transition structure → codebook usage → position leakage) at each layer, then
calibrates every layer against the Experiment-2 floor/ceiling. Answers: does
the action-conditioned discrete-state structure depend on which layer we probe,
and does any depth reach the reusable-state ceiling? Writes
``reports/layer_sweep.{json,md,png}``. Then stop.

The per-layer analysis reuses already-tested components unchanged; the only new
logic (axis placement + reporting) lives in ``evaluation.layer_sweep`` and is
unit-tested separately.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, build_trajectories
from training.train_vq import train_vq_quantizer, VQTrainConfig
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes, all_codes,
)
from evaluation.action_conditioned import extract_action_transitions, run_all_models
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.position_leakage import evaluate_position_leakage
from evaluation.layer_sweep import summarize_sweep, build_markdown, build_plot

DEFAULT_TEACHER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def _problem_hash(p):
    return (p["target"], tuple(sorted(p["numbers"])))


def analyze_layer(model, tokenizer, train_problems, val_problems, layer,
                  K, vq_epochs, tr_epochs, batch_size, seed):
    """Full single-layer pipeline -> one metrics row."""
    train = build_trajectories(model, tokenizer, train_problems, layer=layer,
                               batch_size=batch_size)
    val = build_trajectories(model, tokenizer, val_problems, layer=layer,
                             batch_size=batch_size)

    vq = train_vq_quantizer(train, val, config=VQTrainConfig(
        num_codes=K, epochs=vq_epochs, batch_size=256, seed=seed))
    d_tr = encode_trajectories_to_codes(vq, train)
    d_va = encode_trajectories_to_codes(vq, val)

    usage = analyze_codebook_usage(all_codes(d_tr), K)
    leak = evaluate_position_leakage(d_tr, d_va, K, seed=seed)
    res = run_all_models(extract_action_transitions(d_tr),
                         extract_action_transitions(d_va), K,
                         epochs=tr_epochs, seed=seed)
    return {
        "layer": layer,
        "n_train_states": int(all_codes(d_tr).shape[0]),
        "active_codes": int(usage["active_codes"]),
        "perplexity": float(usage["perplexity"]),
        "gini": float(usage["collapse_score"]),
        "position_leakage": float(leak["position_predictability_score"]),
        "leakage_majority": float(leak["chance_accuracy"]),
        "majority": float(res["A_majority"]["top1"]),
        "bigram_z": float(res["B_bigram"]["top1"]),
        "action_bigram": float(res["C_action_bigram"]["top1"]),
        "mlp_z": float(res["D_mlp_z"]["top1"]),
        "mlp_z_op": float(res["E_mlp_z_op"]["top1"]),
        "mlp_z_op_operands": float(res["E2_mlp_z_op_operands"]["top1"]),
        "mlp_action_only": float(res["F_mlp_action_only"]["top1"]),
        "H_state": float(res["_struct_state"]["global_entropy"]),
        "det_state": float(res["_struct_state"]["det_frac_mass"]),
        "H_action": float(res["_struct_action"]["global_entropy"]),
        "det_action": float(res["_struct_action"]["det_frac_mass"]),
    }


def load_anchors(reports_dir):
    """Exp2 floor/ceiling and the cached layer −1 (Exp1) for cross-check."""
    floor = ceiling = observed = None
    pc_path = os.path.join(reports_dir, "positive_control.json")
    if os.path.exists(pc_path):
        pc = json.load(open(pc_path, encoding="utf-8"))
        floor, ceiling = pc["noise_floor"], pc["positive_control"]
    ac_path = os.path.join(reports_dir, "action_conditioned.json")
    if os.path.exists(ac_path):
        ac = json.load(open(ac_path, encoding="utf-8"))["fixed_init"]
        observed = {"det_action": ac["_struct_action"]["det_frac_mass"],
                    "H_action": ac["_struct_action"]["global_entropy"],
                    "mlp_z_op": ac["E_mlp_z_op"]["top1"]}
    return floor, ceiling, observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default=DEFAULT_TEACHER)
    ap.add_argument("--layers", default="4,8,12,16,20,22")
    ap.add_argument("--cap_train", type=int, default=1500)
    ap.add_argument("--cap_val", type=int, default=750)
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--vq_epochs", type=int, default=50)
    ap.add_argument("--tr_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.reports_dir
    layers = [int(x) for x in args.layers.split(",")]

    print(f"[Exp3] Loading frozen teacher: {args.model}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype)
    tokenizer = load_tokenizer(model_id=args.model)
    num_layers = int(model.config.num_hidden_layers)

    train_problems = load_problems(os.path.join(args.data_dir, "train.jsonl"))[:args.cap_train]
    val_problems = load_problems(os.path.join(args.data_dir, "val.jsonl"))[:args.cap_val]
    # De-duplicate val against train (same leakage guard as Phase A).
    train_h = {_problem_hash(p) for p in train_problems}
    val_problems = [p for p in val_problems if _problem_hash(p) not in train_h]
    print(f"[Exp3] {len(train_problems)} train / {len(val_problems)} val problems; "
          f"layers {layers} (model has {num_layers}).")

    rows = []
    for layer in layers:
        print(f"\n[Exp3] === layer {layer} ===")
        rows.append(analyze_layer(model, tokenizer, train_problems, val_problems,
                                  layer, args.num_codes, args.vq_epochs,
                                  args.tr_epochs, args.batch_size, args.seed))
        r = rows[-1]
        print(f"[Exp3] layer {layer}: det(z,op)={r['det_action']:.3f} "
              f"H(z'|z,op)={r['H_action']:.3f} MLP(z,op)={r['mlp_z_op']:.3f} "
              f"pos-leak={r['position_leakage']:.3f}")

    floor, ceiling, observed = load_anchors(out)
    config = {"model": args.model, "num_layers": num_layers, "layers": layers,
              "cap_train": len(train_problems), "cap_val": len(val_problems),
              "num_codes": args.num_codes, "vq_epochs": args.vq_epochs,
              "tr_epochs": args.tr_epochs, "seed": args.seed}

    payload = {"config": config, "layers": rows,
               "floor": floor, "ceiling": ceiling, "observed_layer_-1": observed}
    if floor is not None and ceiling is not None:
        summary = summarize_sweep(rows, floor, ceiling)
        payload["summary"] = summary
        with open(os.path.join(out, "layer_sweep.md"), "w", encoding="utf-8") as f:
            f.write(build_markdown(rows, summary, floor, ceiling, observed, config))
        build_plot(rows, floor, ceiling, os.path.join(out, "layer_sweep.png"))
        print(f"\n[Exp3] best layer = {summary['best_layer']} "
              f"(det {summary['best_det_action']:.3f}, "
              f"{summary['best_pos_det']*100:.0f}% to ceiling); "
              f"reaches ceiling: {summary['reaches_ceiling']}")
    else:
        print("\n[Exp3] WARNING: reports/positive_control.json not found — "
              "writing metrics JSON only (no calibrated report).")

    with open(os.path.join(out, "layer_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[Exp3] Done. See reports/layer_sweep.md")


if __name__ == "__main__":
    main()
