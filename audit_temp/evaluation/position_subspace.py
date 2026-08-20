"""Position subspace estimation and removal (V5.1 Steps 1-3).

The invalidated-init audit left one substantive open question: are the
discovered discrete codes capturing *reusable* state content, or merely
**trajectory progression** (how far along the chain a state sits)? The V5
position-leakage test answered this for the *codes*; here we attack the
*continuous* hidden states directly.

Pipeline
--------
1. **Position probe** — fit a regularized logistic classifier
   ``hidden_state -> position bin`` and report held-out accuracy. The label is
   the **relative** depth ``i / num_steps`` of a state within its trajectory,
   quantized into 4 bins. Relative (not absolute) depth is used because the
   Countdown trajectories have only 3-5 states, so absolute step indices cannot
   form four populated bins, whereas relative depth captures "trajectory stage"
   uniformly across lengths.

2. **INLP** (Iterative Nullspace Projection, Ravfogel et al. 2020) — repeatedly
   train a linear position classifier and project the data onto the nullspace
   of its weight row-space, accumulating a single projection ``P`` that removes
   the *linearly position-predictive* subspace. We use the right singular
   vectors (``Vt``) of the stacked classifier weights to span each row-space.

3. **Scrub** — apply ``x -> (x - mean) @ P`` to every aligned state. Centering
   is a pure translation and therefore a no-op for every downstream discrete
   metric (nearest-neighbour assignment and the data-dependent codebook init
   shift together), so the scrubbed states differ from the raw states *only*
   by removal of the position subspace — a clean attribution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory


# --------------------------------------------------------------------------- #
# Position labels
# --------------------------------------------------------------------------- #
def build_position_dataset(
    trajectories: List[Trajectory], n_bins: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack aligned states and label each by its relative-depth quartile.

    Returns ``(X, y)`` where ``X`` is ``(M, H)`` float32 and ``y`` is ``(M,)``
    int64 bin ids in ``[0, n_bins)``. A state at step ``i`` of an ``N``-step
    trajectory gets ``bin = min(int(i / N * n_bins), n_bins - 1)`` (``i = 0``
    for single-state degenerate trajectories maps to bin 0).
    """
    Xs, ys = [], []
    for traj in trajectories:
        states = traj.states.float().cpu().numpy()  # (N+1, H)
        n_states = states.shape[0]
        N = max(n_states - 1, 1)
        for i in range(n_states):
            b = min(int(i / N * n_bins), n_bins - 1)
            ys.append(b)
        Xs.append(states)
    if not Xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def _majority_accuracy(y: np.ndarray) -> float:
    if y.shape[0] == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
# Step 1 — position probe
# --------------------------------------------------------------------------- #
def train_position_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Fit ``hidden_state -> position bin`` and report held-out accuracy.

    ``C`` is the inverse L2 regularization strength (smaller = stronger
    regularization); the default ``1.0`` is sklearn's regularized default.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    return {
        "position_accuracy": acc,
        "majority_baseline": _majority_accuracy(yte),
        "n_classes": int(np.unique(ytr).shape[0]),
        "n_train": int(ytr.shape[0]),
        "n_test": int(yte.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Step 2 — INLP
# --------------------------------------------------------------------------- #
def _rowspace_projection(W: np.ndarray) -> np.ndarray:
    """Projection ``(D, D)`` onto the row space of ``W`` ``(C, D)``.

    Uses the right singular vectors ``Vt`` with non-negligible singular values
    as an orthonormal basis of ``row(W)``; the projection is ``B.T @ B``.
    """
    W = np.atleast_2d(W)
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    if S.size == 0:
        return np.zeros((W.shape[1], W.shape[1]))
    tol = float(S.max()) * 1e-6
    rank = int((S > tol).sum())
    if rank == 0:
        return np.zeros((W.shape[1], W.shape[1]))
    B = Vt[:rank]  # (rank, D) orthonormal rows
    return B.T @ B


def inlp_position_projection(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    num_iters: int = 8,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """Iterative Nullspace Projection of the linear position subspace.

    Returns ``(P, info)`` where ``P`` is the ``(D, D)`` cumulative nullspace
    projection (symmetric) and ``info`` holds the per-iteration held-out
    accuracy trace plus the ``before`` / ``after`` probe accuracies and the
    number of dimensions removed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    D = Xtr.shape[1]
    weights: List[np.ndarray] = []

    def cumulative_nullspace() -> np.ndarray:
        if not weights:
            return np.eye(D)
        Wall = np.vstack(weights)  # (sum_C, D)
        return np.eye(D) - _rowspace_projection(Wall)

    trace: List[float] = []
    for _ in range(num_iters):
        P = cumulative_nullspace()
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(Xtr @ P, ytr)
        trace.append(float(accuracy_score(yte, clf.predict(Xte @ P))))
        weights.append(np.atleast_2d(clf.coef_))

    P_final = cumulative_nullspace()
    # "after": a fresh probe trying to recover position from fully scrubbed data.
    after_clf = LogisticRegression(C=C, max_iter=max_iter)
    after_clf.fit(Xtr @ P_final, ytr)
    after_acc = float(accuracy_score(yte, after_clf.predict(Xte @ P_final)))

    dims_removed = int(round(D - np.trace(P_final)))
    info = {
        "before_accuracy": trace[0] if trace else float("nan"),
        "after_accuracy": after_acc,
        "majority_baseline": _majority_accuracy(yte),
        "accuracy_trace": trace,
        "dims_removed": dims_removed,
        "num_iters": num_iters,
    }
    return P_final, info


# --------------------------------------------------------------------------- #
# Step 3 — scrub
# --------------------------------------------------------------------------- #
def apply_scrub_to_trajectories(
    trajectories: List[Trajectory], mean: np.ndarray, P: np.ndarray
) -> List[Trajectory]:
    """Return new trajectories whose aligned states are ``(s - mean) @ P``.

    The returned objects expose the scrubbed states via ``.states`` (with
    ``all_hidden`` set to the scrubbed aligned states and ``state_indices`` an
    identity range) so the existing VQ trainer and code-encoder consume them
    unchanged. ``input_ids`` is a placeholder (token-level rollout is not run
    on scrubbed states); all action metadata is preserved.
    """
    mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    P_t = torch.as_tensor(np.asarray(P), dtype=torch.float32)
    out: List[Trajectory] = []
    for traj in trajectories:
        s = traj.states.float()  # (N+1, H)
        s_scrub = (s - mean_t) @ P_t  # (N+1, H), P symmetric
        n_states = s_scrub.shape[0]
        out.append(
            Trajectory(
                all_hidden=s_scrub,
                input_ids=torch.zeros(n_states, dtype=torch.long),
                state_indices=torch.arange(n_states, dtype=torch.long),
                op_ids=traj.op_ids,
                operands=traj.operands,
                numbers=list(traj.numbers),
                target=int(traj.target),
            )
        )
    return out
