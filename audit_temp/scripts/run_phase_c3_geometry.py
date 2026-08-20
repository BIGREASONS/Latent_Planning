import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.trajectory_dataset import load_trajectories

LARGE_NUMBERS = [25, 50, 75, 100]


def extract_all_targets(trajectories):
    X, A, B, C, D, Depth = [], [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        operands = traj.operands.tolist()
        used = set()
        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i  # steps remaining

            a_row = []
            for v in LARGE_NUMBERS:
                if v not in traj.numbers:
                    a_row.append(0)
                elif v not in used:
                    a_row.append(1)
                else:
                    a_row.append(2)
            A.append(a_row)
            B.append(dist)
            C.append(ops[i])
            D.append(1 if dist <= 2 else 0)
            Depth.append(i)  # True depth!

            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))

    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "B": np.array(B, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "D": np.array(D, dtype=np.int64),
        "Depth": np.array(Depth, dtype=np.int64),
    }


class MultiLabelProbe:
    def __init__(self):
        self.clfs = []
        self.constants = []

    def fit(self, X, Y):
        for j in range(Y.shape[1]):
            classes = np.unique(Y[:, j])
            if classes.shape[0] < 2:
                self.clfs.append(None)
                self.constants.append(int(classes[0]) if classes.shape[0] else 0)
            else:
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X, Y[:, j])
                self.clfs.append(clf)
                self.constants.append(None)

    def predict(self, X):
        preds = []
        for clf, const in zip(self.clfs, self.constants):
            if clf is None:
                preds.append(np.full((X.shape[0],), const))
            else:
                preds.append(clf.predict(X))
        return np.column_stack(preds)


def evaluate_subset(Xtr, ytr_dict, Xte, yte_dict):
    results = {}

    # Depth (Logistic Regression for classification acc)
    clf_depth = LogisticRegression(max_iter=2000)
    clf_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth Acc"] = accuracy_score(yte_dict["Depth"], clf_depth.predict(Xte))

    # Depth (Linear Regression for R^2)
    reg_depth = LinearRegression()
    reg_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth R2"] = r2_score(yte_dict["Depth"], reg_depth.predict(Xte))

    # Probe C (Next Op)
    clf_c = LogisticRegression(max_iter=2000)
    clf_c.fit(Xtr, ytr_dict["C"])
    results["Next Op (Probe C)"] = accuracy_score(yte_dict["C"], clf_c.predict(Xte))

    # Probe D (Reachable in 2)
    clf_d = LogisticRegression(max_iter=2000)
    clf_d.fit(Xtr, ytr_dict["D"])
    results["Reachable in 2 (Probe D)"] = accuracy_score(
        yte_dict["D"], clf_d.predict(Xte)
    )

    # Probe B (Distance to solution)
    clf_b = LogisticRegression(max_iter=2000)
    clf_b.fit(Xtr, ytr_dict["B"])
    results["Dist to Sol (Probe B)"] = accuracy_score(yte_dict["B"], clf_b.predict(Xte))

    # Probe A (Remaining Numbers) - Multi-label
    clf_a = MultiLabelProbe()
    clf_a.fit(Xtr, ytr_dict["A"])
    pred_a = clf_a.predict(Xte)

    accs = []
    for j in range(ytr_dict["A"].shape[1]):
        accs.append(accuracy_score(yte_dict["A"][:, j], pred_a[:, j]))

    exact_acc = float(np.mean(np.all(yte_dict["A"] == pred_a, axis=1)))
    hamming_acc = float(np.mean(yte_dict["A"] == pred_a))

    results["Rem Nums (A) Exact Match"] = exact_acc
    results["Rem Nums (A) Per-Label"] = np.mean(accs)
    results["Rem Nums (A) Hamming"] = hamming_acc

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    out = args.reports_dir
    os.makedirs(out, exist_ok=True)

    print("[Phase C.3] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    trajs = load_trajectories(train_traj_path)[:10000]

    print("[Phase C.3] Extracting targets...")
    data = extract_all_targets(trajs)
    X = data["X"]

    print(f"Dataset size: {len(X)} states.")

    # Train test split on indices
    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=42)

    # Standardize X before PCA (fit on train only)
    print("[Phase C.3] Standardizing & fitting PCA...")
    scaler = StandardScaler().fit(X[idx_tr])
    X_scaled = scaler.transform(X)

    # Extract up to 100 components, plus keeping all 2048
    pca = PCA(n_components=100)
    pca.fit(X_scaled[idx_tr])
    X_pca = pca.transform(X_scaled)

    print(f"Explained Variance (PC1-100): {np.sum(pca.explained_variance_ratio_):.3f}")

    ytr_dict = {k: v[idx_tr] for k, v in data.items() if k != "X"}
    yte_dict = {k: v[idx_te] for k, v in data.items() if k != "X"}

    # Define Subsets
    subsets = [1, 5, 10, 20, 50, 100]
    all_results = []

    print("[Phase C.3] Training Probes on PC Subsets...")
    for k in subsets:
        print(f"  Fitting PC 1-{k}...")
        Xtr_sub = X_pca[idx_tr, :k]
        Xte_sub = X_pca[idx_te, :k]
        res = evaluate_subset(Xtr_sub, ytr_dict, Xte_sub, yte_dict)
        res["PC Subset"] = f"PC 1-{k}"
        all_results.append(res)

    print("  Fitting Baseline (All 2048)...")
    res_baseline = evaluate_subset(
        X_scaled[idx_tr], ytr_dict, X_scaled[idx_te], yte_dict
    )
    res_baseline["PC Subset"] = "All 2048"
    all_results.append(res_baseline)

    df = pd.DataFrame(all_results)

    # Reorder columns
    cols = [
        "PC Subset",
        "Depth R2",
        "Depth Acc",
        "Rem Nums (A) Exact Match",
        "Rem Nums (A) Hamming",
        "Rem Nums (A) Per-Label",
        "Next Op (Probe C)",
        "Dist to Sol (Probe B)",
        "Reachable in 2 (Probe D)",
    ]
    df = df[cols]

    csv_path = os.path.join(out, "phase_c3_geometry_metrics.csv")
    df.to_csv(csv_path, index=False)

    report_path = os.path.join(out, "phase_c3_geometry_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase C.3 — Representation Geometry Audit\n\n")
        f.write(
            "This audit measures how target information (Depth and Planning Probes) is distributed across the top Principal Components of the hidden state.\n\n"
        )

        f.write("## Probe Accuracies across PC Subsets\n\n")
        f.write(
            "| PC Subset | Depth $R^2$ | Depth Acc | Probe A (Exact) | Probe A (Hamming) | Probe A (Per-Label) | Probe C (Next Op) | Probe B (Dist to Sol) | Probe D (Reachable) |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for _, r in df.iterrows():
            f.write(
                f"| {r['PC Subset']} | {r['Depth R2']:.3f} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact Match']*100:.1f}% | {r['Rem Nums (A) Hamming']*100:.1f}% | {r['Rem Nums (A) Per-Label']*100:.1f}% | {r['Next Op (Probe C)']*100:.1f}% | {r['Dist to Sol (Probe B)']*100:.1f}% | {r['Reachable in 2 (Probe D)']*100:.1f}% |\n"
            )

        # Verdict logic
        depth_r2_pc5 = df[df["PC Subset"] == "PC 1-5"]["Depth R2"].iloc[0]
        depth_r2_all = df[df["PC Subset"] == "All 2048"]["Depth R2"].iloc[0]

        f.write("\n## Verdict\n\n")
        if depth_r2_pc5 > 0.90 * depth_r2_all:
            f.write(
                "**CONCENTRATED.**\nDepth is tightly concentrated in the very first few principal components (e.g. PC1-5 explains nearly all Depth variance)."
            )
        else:
            f.write(
                "**DISTRIBUTED.**\nDepth is not a single simple axis. It requires many dimensions to decode accurately, meaning it is fundamentally woven throughout the representation manifold."
            )

    print(f"\n[Phase C.3] Done. Report written to {report_path}")


if __name__ == "__main__":
    main()
