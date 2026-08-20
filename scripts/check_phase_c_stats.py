import os
import sys
import torch
import collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import gather_state_groups


def main():
    out_dir = "reports"
    train_traj_path = os.path.join(out_dir, "trajectories", "train.pt")

    if not os.path.exists(train_traj_path):
        print("ERROR: train.pt not found.")
        sys.exit(1)

    print("Loading trajectories...")
    trajs = load_trajectories(train_traj_path)
    print(f"Loaded {len(trajs)} trajectories.")

    print("Gathering state groups...")
    # gather_state_groups maps sym -> list of (traj_idx, hist, h)
    state_groups = gather_state_groups(trajs)

    unique_states = len(state_groups)

    # We only care about cross-history collisions
    cross_history_groups = {}
    total_positive_pairs = 0
    cluster_sizes = []

    for sym, elements in state_groups.items():
        # group by history to see how many distinct histories reached this state
        histories = collections.defaultdict(list)
        for t_idx, hist, h in elements:
            histories[hist].append((t_idx, h))

        if len(histories) > 1:
            # calculate cross-history pairs
            # Number of pairs = sum(len(h1) * len(h2) for all h1!=h2)
            # which is (sum(len)^2 - sum(len^2)) / 2
            sizes = [len(v) for v in histories.values()]
            total = sum(sizes)
            pairs = (total**2 - sum(s**2 for s in sizes)) // 2

            if pairs > 0:
                cross_history_groups[sym] = elements
                total_positive_pairs += pairs
                cluster_sizes.append(total)

    num_cross_history_states = len(cross_history_groups)

    if num_cross_history_states > 0:
        avg_cluster = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster = max(cluster_sizes)
    else:
        avg_cluster = 0
        max_cluster = 0

    lines = [
        "# Phase C Dataset Statistics",
        "",
        f"- **Total Trajectories loaded**: {len(trajs)}",
        f"- **Total Unique Symbolic States**: {unique_states}",
        f"- **Cross-History States** (>=2 different histories): {num_cross_history_states}",
        f"- **Average Cluster Size** (for cross-history states): {avg_cluster:.2f}",
        f"- **Maximum Cluster Size**: {max_cluster}",
        f"- **Total Possible Positive Pairs**: {total_positive_pairs}",
        "",
    ]

    if num_cross_history_states < 500 or total_positive_pairs < 5000:
        lines.append("> [!WARNING]")
        lines.append(
            "> The dataset may be underpowered for contrastive learning. Consider generating more trajectories."
        )
    else:
        lines.append("> [!PASS]")
        lines.append(
            "> Dataset appears sufficiently powered for contrastive canonicalization."
        )

    md_path = os.path.join(out_dir, "state_statistics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Stats written to {md_path}")
    print(f"  Cross-History States: {num_cross_history_states}")
    print(f"  Total Positive Pairs: {total_positive_pairs}")


if __name__ == "__main__":
    main()
