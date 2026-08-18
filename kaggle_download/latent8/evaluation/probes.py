"""Linear representation probes.

We fit *linear* probes (logistic regression on standardized features) on frozen
teacher hidden states to ask: what task-relevant information is linearly
decodable from the representation? Linear probes are deliberately weak so that a
high score implies the information is explicitly present, not reconstructed by a
powerful probe.

Probes (evaluated at state ``s_i``, i.e. after ``i`` reasoning steps, looking
ahead to step ``i+1``):

* **A — remaining numbers**: for each large number in {25, 50, 75, 100}, is it
  still in the pool and not yet used as an operand? (multi-label, 4 labels)
* **B — distance-to-solution**: number of reasoning steps remaining (multiclass)
* **C — next symbolic operation**: the op of the next step (4-class)
* **D — reachable within 2 steps**: is distance-to-solution <= 2? (binary)

Probe A uses the fixed-value set {25,50,75,100} so each output dimension has a
consistent meaning across problems (a requirement for a single linear probe).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from data_processing.trajectory_dataset import Trajectory

LARGE_NUMBERS = [25, 50, 75, 100]
PROBE_NAMES = {
    "A": "remaining_numbers",
    "B": "distance_to_solution",
    "C": "next_operation",
    "D": "reachable_within_2",
}


def extract_probe_data(trajectories: List[Trajectory]) -> Dict[str, np.ndarray]:
    """Build the feature matrix and probe labels from trajectories.

    Features are the hidden states s_0..s_{N-1} (states that have a next step).
    """
    X, A, B, C, D = [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        operands = traj.operands.tolist()
        used = set()
        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i  # steps remaining from s_i
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
            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))
    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "B": np.array(B, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "D": np.array(D, dtype=np.int64),
    }


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def _make_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def _fit_eval_single(
    Xtr, ytr, Xte, yte, binary: bool
) -> Dict[str, float]:
    """Fit one logistic-regression probe and return accuracy/f1/auc."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    classes = np.unique(ytr)
    if classes.shape[0] < 2:
        # Degenerate target: predict the single observed class.
        const = int(classes[0]) if classes.shape[0] else 0
        pred = np.full_like(yte, const)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="binary" if binary else "macro",
                      zero_division=0)
        return {"accuracy": float(acc), "f1": float(f1), "auc": float("nan")}

    clf = _make_probe()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="binary" if binary else "macro",
                  zero_division=0)

    auc = float("nan")
    try:
        proba = clf.predict_proba(Xte)
        if binary:
            auc = roc_auc_score(yte, proba[:, 1])
        else:
            auc = roc_auc_score(yte, proba, multi_class="ovr",
                                average="macro", labels=clf.named_steps["clf"].classes_)
    except ValueError:
        auc = float("nan")  # e.g. a class missing from the test set
    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc), "clf": clf}


def _fit_eval_multilabel(Xtr, Ytr, Xte, Yte) -> Dict[str, float]:
    """Average per-label multiclass probe metrics over all label columns."""
    accs, f1s, aucs = [], [], []
    for j in range(Ytr.shape[1]):
        res = _fit_eval_single(Xtr, Ytr[:, j], Xte, Yte[:, j], binary=False)
        accs.append(res["accuracy"])
        f1s.append(res["f1"])
        if not np.isnan(res["auc"]):
            aucs.append(res["auc"])
    return {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
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
                clf = _make_probe()
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

def _fit_eval_multilabel_joint(Xtr, Ytr, Xte, Yte):
    from sklearn.metrics import accuracy_score
    clf = MultiLabelProbe()
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xte)
    
    # average per-label accuracy
    accs = []
    for j in range(Ytr.shape[1]):
        accs.append(accuracy_score(Yte[:, j], pred[:, j]))
    # exact match accuracy
    exact_acc = float(np.mean(np.all(Yte == pred, axis=1)))
    
    return {
        "accuracy": float(np.mean(accs)),
        "exact_accuracy": exact_acc,
        "f1": 0.0, # not critical
        "auc": float("nan"),
        "clf": clf
    }


def _chance_accuracy(y: np.ndarray) -> float:
    """Majority-class baseline accuracy."""
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_probes(
    train_trajs: List[Trajectory],
    test_trajs: List[Trajectory],
):
    """Fit probes A-D on train states, evaluate on test states.

    Returns ``(results_df, details, fitted_probes)`` where ``results_df`` has columns
    [probe, accuracy, f1, auc], ``details`` holds chance baselines / sizes,
    and ``fitted_probes`` is a dict of {"A": clfA, "C": clfC}.
    """
    import pandas as pd

    tr = extract_probe_data(train_trajs)
    te = extract_probe_data(test_trajs)

    if len(te["X"]) == 0:
        raise RuntimeError(
            "No probe samples were extracted from the test split. "
            "Likely causes: deduplication removed every trajectory or parser produced no labels."
        )

    rows, details = [], {}

    # A: multi-label (joint for coherence evaluation)
    resA = _fit_eval_multilabel_joint(tr["X"], tr["A"], te["X"], te["A"])
    # B, C: multiclass ; D: binary
    resB = _fit_eval_single(tr["X"], tr["B"], te["X"], te["B"], binary=False)
    resC = _fit_eval_single(tr["X"], tr["C"], te["X"], te["C"], binary=False)
    resD = _fit_eval_single(tr["X"], tr["D"], te["X"], te["D"], binary=True)

    for key, res in [("A", resA), ("B", resB), ("C", resC), ("D", resD)]:
        row = {
            "probe": f"{key}:{PROBE_NAMES[key]}",
            "accuracy": res["accuracy"],
            "f1": res["f1"],
            "auc": res["auc"],
        }
        if "exact_accuracy" in res:
            row["exact_accuracy"] = res["exact_accuracy"]
        rows.append(row)

    details["n_train"] = int(tr["X"].shape[0])
    details["n_test"] = int(te["X"].shape[0])
    details["chance"] = {
        "A": float(np.mean([_chance_accuracy(te["A"][:, j])
                            for j in range(te["A"].shape[1])])),
        "B": _chance_accuracy(te["B"]),
        "C": _chance_accuracy(te["C"]),
        "D": _chance_accuracy(te["D"]),
    }
    fitted_probes = {
        "A": resA.get("clf"),
        "B": resB.get("clf"),
        "C": resC.get("clf"),
    }
    return pd.DataFrame(rows), details, fitted_probes


def save_probe_results(df, csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)


def generate_probe_report(df, details: Dict, md_path: str) -> None:
    """Write probe_report.md explaining what is / is not encoded."""
    import pandas as pd  # noqa: F401

    chance = details["chance"]
    chance_by_key = {f"{k}:{PROBE_NAMES[k]}": v for k, v in chance.items()}

    lines = []
    lines.append("# Representation Probe Report\n")
    lines.append(
        f"Linear probes fit on **{details['n_train']}** teacher states, "
        f"evaluated on **{details['n_test']}** held-out states.\n")
    lines.append(
        "Each probe is a logistic regression on standardized hidden states "
        "(linear only). A score well above the majority-class baseline means "
        "the information is *linearly decodable* from the frozen representation.\n")

    lines.append("\n## Results\n")
    lines.append("| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    encoded, not_encoded = [], []
    for _, r in df.iterrows():
        base = chance_by_key.get(r["probe"], float("nan"))
        auc = r["auc"]
        # "Encoded" if clearly above chance and AUC indicates real signal.
        is_encoded = (
            (not np.isnan(auc) and auc >= 0.65)
            or (r["accuracy"] - base >= 0.10)
        )
        verdict = "encoded" if is_encoded else "weak / not encoded"
        (encoded if is_encoded else not_encoded).append(r["probe"])
        auc_str = "n/a" if np.isnan(auc) else f"{auc:.3f}"
        
        exact_str = f"{r['exact_accuracy']:.3f}" if "exact_accuracy" in r and not pd.isna(r["exact_accuracy"]) else "n/a"
        
        lines.append(
            f"| {r['probe']} | {r['accuracy']:.3f} | {exact_str} | {base:.3f} | "
            f"{r['f1']:.3f} | {auc_str} | {verdict} |")

    lines.append("\n## Interpretation\n")
    if encoded:
        lines.append("**Linearly encoded in the hidden state:**")
        for p in encoded:
            lines.append(f"- {p}")
    else:
        lines.append("**Linearly encoded:** none of the probed quantities "
                     "cleared the bar.")
    lines.append("")
    if not_encoded:
        lines.append("**Weak or not linearly encoded:**")
        for p in not_encoded:
            lines.append(f"- {p}")
    lines.append("")
    lines.append(
        "_Note: probe A (remaining numbers) is restricted to the fixed set "
        "{25,50,75,100} so each label has a consistent meaning across problems. "
        "AUC is reported as macro one-vs-rest for multiclass probes and averaged "
        "over labels for the multi-label probe; 'n/a' indicates a degenerate or "
        "missing class in the evaluation split._\n")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
