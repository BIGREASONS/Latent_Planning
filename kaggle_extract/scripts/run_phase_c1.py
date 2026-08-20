import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    out = args.reports_dir
    os.makedirs(out, exist_ok=True)

    print("[Phase C.1] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")

    if not os.path.exists(train_traj_path):
        print("ERROR: Trajectories not found. Run Phase A first.")
        sys.exit(1)

    trajs = load_trajectories(train_traj_path)
    print(f"Loaded {len(trajs)} trajectories.")

    # We want to group by:
    # 1. Symbolic State
    # 2. Depth
    # Data structure: states_by_sym[sym] = [(t_idx, depth, h)]
    # We also keep a flat list for random sampling

    states_by_sym = {}
    states_flat = []

    print("[Phase C.1] Grouping states by symbolic state and depth...")
    for t_idx, traj in enumerate(trajs):
        states_info = get_symbolic_states(traj)
        for d, info in enumerate(states_info):
            if info is not None:
                sym, hist = info
                h = traj.states[d : d + 1]  # (1, H)
                if sym not in states_by_sym:
                    states_by_sym[sym] = []
                states_by_sym[sym].append((t_idx, d, h))
                states_flat.append((sym, t_idx, d, h))

    # Define pairs to compute:
    # 1. Same state, Same depth (different traj)
    # 2. Same state, Different depth (different traj)
    # 3. Different state, Same depth (different traj)
    # 4. Different state, Different depth (different traj)

    results = {
        "Same state, Same depth": [],
        "Same state, Different depth": [],
        "Different state, Same depth": [],
        "Different state, Different depth": [],
    }

    print("[Phase C.1] Sampling pairs and computing cosines...")
    MAX_PAIRS_PER_CAT = 10000

    # 1 & 2: Same state
    for sym, elements in states_by_sym.items():
        n = len(elements)
        if n < 2:
            continue

        # Sample limited number of pairs per state to avoid quadratic explosion
        pairs_to_sample = min(100, n * (n - 1) // 2)
        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(indices)
        indices = indices[:pairs_to_sample]

        for i, j in indices:
            t1, d1, h1 = elements[i]
            t2, d2, h2 = elements[j]
            if t1 != t2:  # Must be different trajectories
                cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1).item()
                if d1 == d2:
                    if len(results["Same state, Same depth"]) < MAX_PAIRS_PER_CAT:
                        results["Same state, Same depth"].append(cos)
                else:
                    if len(results["Same state, Different depth"]) < MAX_PAIRS_PER_CAT:
                        results["Same state, Different depth"].append(cos)

    # 3 & 4: Different state
    # Shuffle flat list
    random.shuffle(states_flat)

    sampled_diff = 0
    while sampled_diff < MAX_PAIRS_PER_CAT and len(states_flat) >= 2:
        # Pick 2 random states
        idx1 = random.randint(0, len(states_flat) - 1)
        idx2 = random.randint(0, len(states_flat) - 1)

        sym1, t1, d1, h1 = states_flat[idx1]
        sym2, t2, d2, h2 = states_flat[idx2]

        if sym1 != sym2 and t1 != t2:
            cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1).item()
            if d1 == d2:
                if len(results["Different state, Same depth"]) < MAX_PAIRS_PER_CAT:
                    results["Different state, Same depth"].append(cos)
                    sampled_diff += 1
            else:
                if len(results["Different state, Different depth"]) < MAX_PAIRS_PER_CAT:
                    results["Different state, Different depth"].append(cos)
                    sampled_diff += 1

    print("[Phase C.1] Generating report...")
    md_path = os.path.join(out, "phase_c1_anisotropy_report.md")

    with open(md_path, "w") as f:
        f.write("# Phase C.1 — Anisotropy Audit\n\n")
        f.write(
            "This audit decouples symbolic state similarity from token position (reasoning depth) similarity to identify if the latent space is dominated by position.\n\n"
        )

        f.write("## Results\n\n")
        f.write("| Pair Type | Cosine | N |\n")
        f.write("| --- | --- | --- |\n")

        metrics = {}
        for cat, vals in results.items():
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                metrics[cat] = mean_val
                f.write(f"| {cat} | {mean_val:.3f} ± {std_val:.3f} | {len(vals)} |\n")
            else:
                f.write(f"| {cat} | N/A | 0 |\n")
                metrics[cat] = 0.0

        f.write("\n## Verdict\n\n")
        diff_state_same_depth = metrics.get("Different state, Same depth", 0.0)
        same_state_same_depth = metrics.get("Same state, Same depth", 0.0)

        if diff_state_same_depth > 0.8:
            f.write("**POSITION DOMINATES (Anisotropy Confound).**\n\n")
            f.write(
                "Hidden states at the same reasoning depth are highly similar regardless of the symbolic task state. The previously reported 'within-state' similarity was likely an artifact of comparing states at the same depth, not true symbolic alignment. The representation is fragmented by token position.\n"
            )
        elif same_state_same_depth - diff_state_same_depth > 0.15:
            f.write("**SYMBOLIC STATE ENCODED (Position Controlled).**\n\n")
            f.write(
                "States sharing the same symbolic task state are significantly closer than different states at the same depth. Position does not dominate the representation.\n"
            )
        else:
            f.write("**INCONCLUSIVE.**\n\n")
            f.write("Differences are marginal. Further investigation required.\n")

    print(f"Report written to {md_path}")

    # Save raw CSV
    rows = []
    for cat, vals in results.items():
        for v in vals:
            rows.append({"category": cat, "cosine": v})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out, "anisotropy_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")


if __name__ == "__main__":
    main()
