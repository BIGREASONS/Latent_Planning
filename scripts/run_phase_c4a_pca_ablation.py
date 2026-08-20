import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states

LARGE_NUMBERS = [25, 50, 75, 100]
# Fixed vocabulary of numbers that can appear in a prompt (small 1-10 + large).
# Used to build the "raw operand" baseline: what is literally listed in the prompt.
BAG_VOCAB = list(range(1, 11)) + [25, 50, 75, 100]


def extract_all_targets(trajectories):
    X, A, C, Depth, Sym, Hist, Bag = [], [], [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        operands = traj.operands.tolist()
        used = set()

        # Bag-of-prompt-numbers: constant per trajectory (what the probe could
        # read straight off the prompt text without any computation).
        bag_row = [1 if v in traj.numbers else 0 for v in BAG_VOCAB]

        # Get exact symbolic states for retrieval
        states_info = get_symbolic_states(traj)

        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i

            a_row = []
            for v in LARGE_NUMBERS:
                if v not in traj.numbers:
                    a_row.append(0)
                elif v not in used:
                    a_row.append(1)
                else:
                    a_row.append(2)
            A.append(a_row)
            C.append(ops[i])
            Depth.append(i)
            Bag.append(bag_row)

            info = states_info[i] if i < len(states_info) else None
            if info is not None:
                sym, hist = info
                Sym.append(str(sym))
                # Action history (prefix of steps taken). Identical history with an
                # identical header => byte-identical hidden state under the frozen
                # causal LM, i.e. a trivial prefix collision.
                Hist.append(str(hist))
            else:
                Sym.append("NONE")
                Hist.append("NONE")

            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))

    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "Depth": np.array(Depth, dtype=np.int64),
        "Sym": np.array(Sym),
        "Hist": np.array(Hist),
        "Bag": np.array(Bag, dtype=np.float32),
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


def evaluate_subset(Xtr, ytr_dict, Xte, yte_dict, dup_exclude=None):
    """Evaluate probes + retrieval on a feature subset.

    dup_exclude: optional list of length n_test; entry i is an array of TRAIN
    column indices to exclude from retrieval for test query i (prefix collisions,
    i.e. train states whose raw hidden state is byte-identical to the query).
    """
    results = {}

    # Depth (Logistic Regression)
    clf_depth = LogisticRegression(max_iter=2000)
    clf_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth Acc"] = accuracy_score(yte_dict["Depth"], clf_depth.predict(Xte))

    # Probe C (Next Op)
    clf_c = LogisticRegression(max_iter=2000)
    clf_c.fit(Xtr, ytr_dict["C"])
    results["Next Op (Probe C)"] = accuracy_score(yte_dict["C"], clf_c.predict(Xte))

    # Probe A (Remaining Numbers) - Multi-label
    clf_a = MultiLabelProbe()
    clf_a.fit(Xtr, ytr_dict["A"])
    pred_a = clf_a.predict(Xte)

    exact_acc = float(np.mean(np.all(yte_dict["A"] == pred_a, axis=1)))
    hamming_acc = float(np.mean(yte_dict["A"] == pred_a))

    results["Rem Nums (A) Exact"] = exact_acc
    results["Rem Nums (A) Hamming"] = hamming_acc

    # Retrieval
    # Normalize features for cosine similarity
    Xte_norm = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    Xtr_norm = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)

    batch_size = 1000
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    valid_queries = 0

    within_cosines = []
    between_cosines = []

    train_syms = ytr_dict["Sym"]
    train_syms_set = set(train_syms)

    # Vectorized computation batch by batch
    for i in range(0, Xte_norm.shape[0], batch_size):
        end = min(i + batch_size, Xte_norm.shape[0])
        Xte_batch = Xte_norm[i:end]
        yte_sym_batch = yte_dict["Sym"][i:end]

        sims = np.dot(Xte_batch, Xtr_norm.T)

        # Prefix mask: push byte-identical (prefix-collision) train neighbors to
        # the bottom so they can neither be retrieved nor inflate within-cosine.
        if dup_exclude is not None:
            for r in range(end - i):
                ex = dup_exclude[i + r]
                if ex is not None and len(ex) > 0:
                    sims[r, ex] = -1.0

        # Get top 10 indices
        top10_idx = np.argsort(sims, axis=1)[:, -10:][:, ::-1]

        for idx_in_batch in range(end - i):
            query_sym = yte_sym_batch[idx_in_batch]
            if query_sym == "NONE" or query_sym not in train_syms_set:
                continue

            valid_queries += 1

            # Within vs Between similarity
            excl_mask = np.zeros(train_syms.shape[0], dtype=bool)
            if dup_exclude is not None:
                ex = dup_exclude[i + idx_in_batch]
                if ex is not None and len(ex) > 0:
                    excl_mask[ex] = True
            same_mask = (train_syms == query_sym) & ~excl_mask
            diff_mask = ~(train_syms == query_sym) & (train_syms != "NONE") & ~excl_mask

            if np.any(same_mask):
                within_cosines.append(sims[idx_in_batch, same_mask].mean())
            if np.any(diff_mask):
                between_cosines.append(sims[idx_in_batch, diff_mask].mean())

            # Top hits
            hit_indices = [
                train_syms[top10_idx[idx_in_batch, k]] == query_sym for k in range(10)
            ]

            if hit_indices[0]:
                top1_hits += 1
            if any(hit_indices[:5]):
                top5_hits += 1
            if any(hit_indices[:10]):
                top10_hits += 1

    results["Retrieval Top-1"] = top1_hits / valid_queries if valid_queries > 0 else 0.0
    results["Retrieval Top-5"] = top5_hits / valid_queries if valid_queries > 0 else 0.0
    results["Retrieval Top-10"] = (
        top10_hits / valid_queries if valid_queries > 0 else 0.0
    )

    mean_within = np.mean(within_cosines) if within_cosines else 0.0
    mean_between = np.mean(between_cosines) if between_cosines else 1.0
    results["Within/Between Ratio"] = (
        mean_within / mean_between if mean_between > 0 else float("inf")
    )

    return results


def compute_prefix_collisions(Xtr_raw, Xte_raw, decimals=5):
    """Find prefix collisions: train states whose raw hidden vector is
    byte-identical to a test query's.

    Under a frozen, deterministic, causal LM, byte-identical hidden states can
    only arise from byte-identical token prefixes (same header + same step text).
    Such neighbors make retrieval trivially correct, so we exclude them.

    Returns (dup_exclude, n_queries_with_dup, n_pairs):
      dup_exclude[i] = np.array of TRAIN indices identical to test query i.
    """

    def key(v):
        return np.round(v, decimals).astype(np.float32).tobytes()

    train_map = {}
    for j in range(Xtr_raw.shape[0]):
        train_map.setdefault(key(Xtr_raw[j]), []).append(j)

    dup_exclude = []
    n_queries_with_dup = 0
    n_pairs = 0
    for i in range(Xte_raw.shape[0]):
        hits = train_map.get(key(Xte_raw[i]), [])
        if hits:
            n_queries_with_dup += 1
            n_pairs += len(hits)
        dup_exclude.append(np.array(hits, dtype=np.int64))
    return dup_exclude, n_queries_with_dup, n_pairs


def permute_labels(data_dict, seed=0):
    """Return a copy of the label arrays with rows permuted (features untouched
    elsewhere), decoupling labels from the representation: a permutation null."""
    rng = np.random.RandomState(seed)
    n = len(data_dict["Depth"])
    perm = rng.permutation(n)
    out = {}
    for k, v in data_dict.items():
        out[k] = v[perm] if k != "X" else v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
        help="Directory holding trajectories/ and where outputs are written (e.g. /kaggle/working/reports)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a quick smoke test on subset of data"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=10000,
        help="Cap on trajectories loaded from train.pt; 0 = use all (set high for large-scale Kaggle runs)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of trajectories used as the retrieval database (train); remainder is the query set (test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the permutation-null control (the rest of the pipeline is deterministic)",
    )
    args = parser.parse_args()

    out = args.reports_dir
    os.makedirs(out, exist_ok=True)

    print("[Phase C.4A] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    trajs = load_trajectories(train_traj_path)

    if args.smoke:
        print("[Phase C.4A] SMOKE TEST: Limiting to 500 trajectories.")
        trajs = trajs[:500]
    elif args.max_trajectories and len(trajs) > args.max_trajectories:
        print(
            f"[Phase C.4A] Capping to {args.max_trajectories} trajectories (of {len(trajs)})."
        )
        trajs = trajs[: args.max_trajectories]

    # Split completely by trajectory
    split_idx = int(len(trajs) * args.train_frac)
    trajs_train = trajs[:split_idx]
    trajs_test = trajs[split_idx:]

    print("[Phase C.4A] Extracting train targets...")
    train_data = extract_all_targets(trajs_train)
    Xtr_raw = train_data["X"]

    print("[Phase C.4A] Extracting test targets...")
    test_data = extract_all_targets(trajs_test)
    Xte_raw = test_data["X"]

    print(f"Train states: {len(Xtr_raw)}, Test states: {len(Xte_raw)}")

    # Measure overlap
    train_unique_syms = set([s for s in train_data["Sym"] if s != "NONE"])
    test_unique_syms = set([s for s in test_data["Sym"] if s != "NONE"])
    overlap = len(train_unique_syms.intersection(test_unique_syms))
    coverage = overlap / len(test_unique_syms) if len(test_unique_syms) > 0 else 0.0
    print(
        f"Test Coverage: {coverage*100:.1f}% ({overlap}/{len(test_unique_syms)} test states exist in train)"
    )

    # Standardize & PCA STRICTLY on train
    print("[Phase C.4A] Standardizing & fitting PCA on TRAIN only...")
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_scaled = scaler.transform(Xtr_raw)
    Xte_scaled = scaler.transform(Xte_raw)

    n_comp = min(2048, Xtr_scaled.shape[0], Xtr_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    Xtr_pca = pca.fit_transform(Xtr_scaled)
    Xte_pca = pca.transform(Xte_scaled)

    # Prefix mask: exclude byte-identical (prefix-collision) train neighbors from
    # retrieval. Computed on the RAW hidden states (the ground truth for "same
    # text"); identical raw states stay identical under any PCA ablation.
    print("[Phase C.4A] Detecting prefix collisions (deterministic-LM confound)...")
    dup_exclude, n_dup_q, n_dup_pairs = compute_prefix_collisions(Xtr_raw, Xte_raw)
    dup_q_frac = n_dup_q / len(Xte_raw) if len(Xte_raw) > 0 else 0.0
    print(
        f"  {n_dup_q}/{len(Xte_raw)} test states ({dup_q_frac*100:.1f}%) have >=1 "
        f"byte-identical train neighbor; {n_dup_pairs} colliding pairs masked."
    )

    # Ablation subsets
    ablations = [0, 1, 2, 5, 10, 20]
    all_results = []

    print("[Phase C.4A] Running Ablation Sweep...")
    for k in ablations:
        print(f"  Removing top {k} PCs...")
        # Drop the first k PCs
        Xtr_abl = Xtr_pca[:, k:]
        Xte_abl = Xte_pca[:, k:]

        res = evaluate_subset(
            Xtr_abl, train_data, Xte_abl, test_data, dup_exclude=dup_exclude
        )
        res["Removed PCs"] = k
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # ----------------------------------------------------------------------- #
    # Controls (computed at k=0, i.e. full PCA features)
    # ----------------------------------------------------------------------- #
    print("[Phase C.4A] Running controls (permutation null + raw-operand baseline)...")

    # (1) Permutation null: shuffle TRAIN labels, keep representation fixed.
    #     Any metric above this null is signal beyond label base rates / geometry.
    train_perm = permute_labels(train_data, seed=args.seed)
    null_res = evaluate_subset(
        Xtr_pca, train_perm, Xte_pca, test_data, dup_exclude=dup_exclude
    )

    # (2) Raw-operand baseline: probes/retrieval on the bag-of-prompt-numbers
    #     feature only (no hidden state). Shows what is decodable straight from
    #     the operands listed in the prompt, with zero computation.
    base_res = evaluate_subset(
        train_data["Bag"], train_data, test_data["Bag"], test_data
    )

    controls = pd.DataFrame(
        [
            {
                "control": "PERMUTED null",
                **{
                    c: null_res[c]
                    for c in [
                        "Depth Acc",
                        "Rem Nums (A) Exact",
                        "Next Op (Probe C)",
                        "Retrieval Top-1",
                        "Retrieval Top-5",
                    ]
                },
            },
            {
                "control": "Raw-operand baseline",
                **{
                    c: base_res[c]
                    for c in [
                        "Depth Acc",
                        "Rem Nums (A) Exact",
                        "Next Op (Probe C)",
                        "Retrieval Top-1",
                        "Retrieval Top-5",
                    ]
                },
            },
        ]
    )
    controls.to_csv(os.path.join(out, "phase_c4a_controls.csv"), index=False)

    # Compute Gains relative to Removed PCs = 0
    base_top1 = df[df["Removed PCs"] == 0]["Retrieval Top-1"].iloc[0]
    df["Top-1 Gain"] = df["Retrieval Top-1"] / base_top1 if base_top1 > 0 else 1.0

    # phase_c4a_pca_ablation_metrics.csv
    metrics_cols = [
        "Removed PCs",
        "Depth Acc",
        "Rem Nums (A) Exact",
        "Rem Nums (A) Hamming",
        "Next Op (Probe C)",
    ]
    df[metrics_cols].to_csv(
        os.path.join(out, "phase_c4a_pca_ablation_metrics.csv"), index=False
    )

    # phase_c4a_retrieval.csv
    retrieval_cols = [
        "Removed PCs",
        "Retrieval Top-1",
        "Retrieval Top-5",
        "Retrieval Top-10",
        "Top-1 Gain",
    ]
    df[retrieval_cols].to_csv(os.path.join(out, "phase_c4a_retrieval.csv"), index=False)

    # phase_c4a_geometry.csv
    geometry_cols = ["Removed PCs", "Within/Between Ratio"]
    df[geometry_cols].to_csv(os.path.join(out, "phase_c4a_geometry.csv"), index=False)

    # coverage_report.md
    coverage_path = os.path.join(out, "coverage_report.md")
    with open(coverage_path, "w") as f:
        f.write("# Coverage Report\n\n")
        f.write(f"- Train state count: {len(train_unique_syms)}\n")
        f.write(f"- Test state count: {len(test_unique_syms)}\n")
        f.write(f"- Overlap count: {overlap}\n")
        f.write(f"- Overlap %: {coverage*100:.2f}%\n")
        f.write(
            f"- Prefix collisions (test states with >=1 byte-identical train neighbor): "
            f"{n_dup_q}/{len(Xte_raw)} ({dup_q_frac*100:.2f}%), {n_dup_pairs} pairs masked\n"
        )

    report_path = os.path.join(out, "phase_c4a_pca_ablation_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase C.4A — PCA Ablation Sweep Report\n\n")
        f.write(
            "This audit explicitly removes the top (highest-variance) principal components to test if Depth is a low-rank nuisance variable that can be sliced off without harming the distributed symbolic state. The PCA was strictly fit on the Train trajectories.\n\n"
        )
        f.write(
            f"**Train-Test State Coverage**: {coverage*100:.1f}% ({overlap}/{len(test_unique_syms)} unique symbolic states in Test exist in Train)\n\n"
        )
        f.write(
            f"**Prefix collisions masked**: {n_dup_q}/{len(Xte_raw)} test states ({dup_q_frac*100:.1f}%) had >=1 byte-identical train neighbor (deterministic-LM prefix confound); these are excluded from retrieval.\n\n"
        )

        f.write("## Controls\n\n")
        f.write(
            "Retrieval/probe scores must be read against these. The PERMUTED null shuffles train labels (representation fixed) - anything at this level is noise. The raw-operand baseline probes only the bag of numbers listed in the prompt (zero computation).\n\n"
        )
        f.write(
            "| Control | Depth Acc | Probe A (Exact) | Probe C | Retrieval (Top-1) | Retrieval (Top-5) |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for _, r in controls.iterrows():
            f.write(
                f"| {r['control']} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact']*100:.1f}% | {r['Next Op (Probe C)']*100:.1f}% | {r['Retrieval Top-1']*100:.1f}% | {r['Retrieval Top-5']*100:.1f}% |\n"
            )
        f.write("\n## Ablation Results\n\n")
        f.write(
            "| Removed PCs | Depth Acc | Probe A (Exact) | Probe A (Hamming) | Retrieval (Top-1) | Gain | Retrieval (Top-5) | Retrieval (Top-10) | Within/Between |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for _, r in df.iterrows():
            f.write(
                f"| {int(r['Removed PCs'])} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact']*100:.1f}% | {r['Rem Nums (A) Hamming']*100:.1f}% | {r['Retrieval Top-1']*100:.1f}% | {r['Top-1 Gain']:.2f}x | {r['Retrieval Top-5']*100:.1f}% | {r['Retrieval Top-10']*100:.1f}% | {r['Within/Between Ratio']:.3f} |\n"
            )

        # Interpretation
        d_base = df[df["Removed PCs"] == 0]["Depth Acc"].iloc[0]
        d_5 = df[df["Removed PCs"] == 5]["Depth Acc"].iloc[0]
        a_base = df[df["Removed PCs"] == 0]["Rem Nums (A) Exact"].iloc[0]
        a_5 = df[df["Removed PCs"] == 5]["Rem Nums (A) Exact"].iloc[0]
        gain_5 = df[df["Removed PCs"] == 5]["Top-1 Gain"].iloc[0]

        f.write("\n## Verdict\n\n")
        if d_5 < 0.50 and a_5 > a_base * 0.8 and gain_5 > 1.1:
            f.write(
                "**NUISANCE FACTOR CONFIRMED.**\nRemoving the top PCs collapsed the depth signal, preserved the symbolic state, and substantially increased retrieval gain. The canonical planning state can be explicitly recovered by orthogonal projection."
            )
        elif d_5 < 0.50 and a_5 < a_base * 0.5:
            f.write(
                "**FATAL ENTANGLEMENT.**\nRemoving the top PCs successfully killed depth, but it also destroyed the symbolic state. They share the same high-variance directions."
            )
        else:
            f.write(
                "**PERSISTENT.**\nThe ablation had unexpected effects; further investigation is required."
            )

    print(f"\n[Phase C.4A] Done. Report written to {report_path}")


if __name__ == "__main__":
    main()
