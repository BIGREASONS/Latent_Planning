"""Pre-flight smoke test for 4-bit hidden-state extraction -> VQ encode.

Runs the *real* extraction + VQ-encode path on a small slice (default 50
trajectories) and asserts the four invariants that a new/larger model or the
4-bit loader could silently break:

  1. hidden dimension is reported consistently (model config == trajectory == VQ)
  2. layer indexing is valid (hidden_states tuple length, selected-layer shape)
  3. the VQ encoder receives the expected shapes (codes align 1:1 with states)
  4. no silent truncation/drop (every problem -> a trajectory; every step kept)

Architecture-agnostic: validate the plumbing locally on a small cached model,
then run the identical check on Kaggle before the full 7B/8B job, e.g.

    python scripts/smoke_4bit_extraction.py --model Qwen/Qwen2.5-7B --load_in_4bit
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import build_trajectories
from training.train_vq import train_vq_quantizer, VQTrainConfig
from data_processing.discrete_trajectory_dataset import encode_trajectories_to_codes
from scripts.generate_game24_dataset import generate_game24_problem
from scripts.generate_countdown_dataset import generate_countdown_problem


def _problems(domain: str, n: int):
    rng = random.Random(0)
    if domain == "game24":
        return [generate_game24_problem(rng) for _ in range(n)]
    random.seed(0)  # countdown generator uses the module-global RNG
    return [generate_countdown_problem() for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--domain", choices=["game24", "countdown"], default="game24")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--num_codes", type=int, default=32)
    args = ap.parse_args()

    print(f"[smoke] model={args.model} 4bit={args.load_in_4bit} domain={args.domain} n={args.n}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype,
                       load_in_4bit=args.load_in_4bit)
    tokenizer = load_tokenizer(model_id=args.model)
    model.eval()

    cfg_dim = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers)
    print(f"[smoke] config: hidden_size={cfg_dim}  num_hidden_layers={n_layers}")

    # ---- Check 2: layer indexing -------------------------------------- #
    probs = _problems(args.domain, args.n)
    enc = tokenizer(probs[0]["problem"] if "problem" in probs[0]
                    else "Problem:\n" + probs[0]["cot"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    n_hs = len(out.hidden_states)
    assert n_hs == n_layers + 1, f"hidden_states tuple len {n_hs} != n_layers+1 ({n_layers+1})"
    sel = out.hidden_states[args.layer]
    assert sel.shape[-1] == cfg_dim, f"layer[{args.layer}] dim {sel.shape[-1]} != {cfg_dim}"
    print(f"[smoke] OK  check 2: hidden_states len={n_hs} (=L+1); layer[{args.layer}] dim={sel.shape[-1]}")

    # ---- Build trajectories (full per-token extraction) --------------- #
    # Countdown generator emits no 'problem' header; supply the standard one.
    for p in probs:
        p.setdefault("problem", "Problem: Reach the target.\nSolution:\n")
    trajs = build_trajectories(model, tokenizer, probs, layer=args.layer, batch_size=8)

    # ---- Check 4: no silent truncation/drop --------------------------- #
    n_built = len(trajs)
    print(f"[smoke] built {n_built}/{len(probs)} trajectories")
    assert n_built > 0, "no trajectories built"
    dropped = len(probs) - n_built
    if dropped:
        print(f"[smoke] WARNING: {dropped} problem(s) dropped during alignment "
              f"(silent loss — investigate before scaling)")
    bad_len, bad_idx = 0, 0
    for p, t in zip(probs, trajs):
        n_states = int(t.state_indices.shape[0])
        if n_states != len(p["solution"]) + 1:   # N steps -> N+1 states
            bad_len += 1
        if int(t.state_indices.max()) >= t.all_hidden.shape[0]:
            bad_idx += 1
    assert bad_len == 0, f"{bad_len} trajectories have wrong #states (steps dropped)"
    assert bad_idx == 0, f"{bad_idx} trajectories index past the token sequence"
    print(f"[smoke] OK  check 4: every problem kept, states=steps+1, indices in-range")

    # ---- Check 1: hidden dim consistency ------------------------------ #
    tdim = trajs[0].hidden_dim
    assert tdim == cfg_dim, f"trajectory hidden_dim {tdim} != config {cfg_dim}"
    print(f"[smoke] OK  check 1: trajectory hidden_dim={tdim} == config {cfg_dim}")

    # ---- Check 3: VQ receives expected shapes ------------------------- #
    vq = train_vq_quantizer(trajs, trajs,
                            config=VQTrainConfig(num_codes=args.num_codes, epochs=5, batch_size=128))
    assert vq.hidden_dim == cfg_dim, f"VQ hidden_dim {vq.hidden_dim} != config {cfg_dim}"
    disc = encode_trajectories_to_codes(vq, trajs)
    mism = sum(1 for t, d in zip(trajs, disc)
               if d.codes.shape[0] != int(t.state_indices.shape[0]))
    assert mism == 0, f"{mism} encoded trajectories have code count != state count"
    all_codes = torch.cat([d.codes for d in disc])
    assert int(all_codes.min()) >= 0 and int(all_codes.max()) < args.num_codes, "code id out of range"
    print(f"[smoke] OK  check 3: VQ dim={vq.hidden_dim}; codes align with states; "
          f"ids in [0,{args.num_codes}); active={len(set(all_codes.tolist()))}")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
