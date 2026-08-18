"""Position leakage test for the discrete codebook.

Asks the simplest adversarial question about the discovered codes: are they
just **trajectory position (step index)** in disguise? If a linear classifier
can predict "this state is step ``i`` of the trajectory" from the code id
alone, the codes are not capturing reusable state content — they are
memorizing where in the chain they sit.

Method
------
For every state in every discrete trajectory we form the label
``position = i`` (``0``-indexed depth within its trajectory) and a one-hot
feature vector over the codebook. We then fit a logistic regression
``code_id -> position`` on the train split and evaluate accuracy on the held-out
split, against:

* a **majority-class chance** baseline, and
* a **permutation null** (train labels shuffled) — matching the Phase C.4A
  control discipline. Any real signal must clear both.

The output ``position_predictability_score`` is the test accuracy; if it is
high (close to 1) and well above both baselines, the codes are position
indices, not reusable discrete states.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory


def _flatten_positions(
    discrete_trajs: List[DiscreteTrajectory], num_codes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (one-hot code features, position labels) over all states."""
    rows, positions = [], []
    for traj in discrete_trajs:
        n = traj.codes.shape[0]
        for i in range(n):
            rows.append(int(traj.codes[i].item()))
            positions.append(i)
    if not rows:
        return (
            np.zeros((0, num_codes), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    # One-hot via advanced indexing (sparse feature; logistic regression reads
    # it directly without a dense (M, num_codes) copy in memory).
    codes = np.asarray(rows, dtype=np.int64)
    X = np.zeros((codes.shape[0], num_codes), dtype=np.float32)
    X[np.arange(codes.shape[0]), codes] = 1.0
    return X, np.asarray(positions, dtype=np.int64)


def _chance_accuracy(y: np.ndarray) -> float:
    """Majority-class baseline accuracy."""
    if y.shape[0] == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def evaluate_position_leakage(
    train_disc: List[DiscreteTrajectory],
    test_disc: List[DiscreteTrajectory],
    num_codes: int,
    seed: int = 0,
) -> Dict:
    """Fit ``code -> position`` classifier; return accuracy + null controls.

    Returns a dict with ``position_predictability_score`` (test acc),
    ``chance_accuracy`` (majority class), ``permutation_null_accuracy``
    (train labels shuffled), and ``n_train`` / ``n_test`` sample counts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    Xtr, ytr = _flatten_positions(train_disc, num_codes)
    Xte, yte = _flatten_positions(test_disc, num_codes)

    if ytr.shape[0] == 0 or yte.shape[0] == 0:
        return {
            "position_predictability_score": float("nan"),
            "chance_accuracy": float("nan"),
            "permutation_null_accuracy": float("nan"),
            "n_train": 0,
            "n_test": 0,
        }

    chance = _chance_accuracy(yte)

    # Real fit. max_iter raised because one-hot features can be slow to converge.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    score = float(accuracy_score(yte, clf.predict(Xte)))

    # Permutation null: decouple train labels from codes.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ytr.shape[0])
    null_clf = LogisticRegression(max_iter=2000)
    null_clf.fit(Xtr, ytr[perm])
    null_score = float(accuracy_score(yte, null_clf.predict(Xte)))

    return {
        "position_predictability_score": score,
        "chance_accuracy": chance,
        "permutation_null_accuracy": null_score,
        "n_train": int(ytr.shape[0]),
        "n_test": int(yte.shape[0]),
    }


def save_position_leakage_report(metrics: Dict, csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
