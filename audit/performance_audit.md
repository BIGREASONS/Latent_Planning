# Phase 8 — Performance Audit

## P1 — O(M²) pure-Python double loops in Phase C retrieval — CRITICAL (at scale)

`evaluation/intrinsic_noise.py:209-216`:
```python
M = len(all_h)                       # capped at 10,000
same_hist_mask = torch.zeros((M, M), dtype=torch.bool)
for i in range(M):
    for j in range(M):
        if all_hists[i] == all_hists[j]:   # tuple comparison
            same_hist_mask[i, j] = True
```
At the M=10,000 cap that is **10⁸ Python-level tuple comparisons** for the mask alone. Then `:229-233`:
```python
for i in range(len(all_syms)):          # up to 10,000
    for j in range(len(all_syms)):      # nested
        if ... all_syms[i]==all_syms[j]: ...
```
another **10⁸** iterations to compute `has_valid_match`. Combined ~2×10⁸ Python ops → minutes-to-hours, single-threaded, dominating Phase C wall-clock entirely. **Fix:** vectorize the history mask via integer-encoded history ids (`np.unique(..., return_inverse=True)` then broadcast equality), and precompute per-sym counts once instead of the O(M²) `has_valid_match` scan.

## P2 — Per-sample sklearn `.predict()` + CPU sync inside the rollout loop — HIGH

`evaluation/coherence.py:90-132`: for **every** trajectory × depth × {A,B,C,D} × {pred, teacher, identity} it calls `probe.predict(h.cpu().numpy())[0]`. That is:
- a `.cpu()` device→host sync per call,
- a `.numpy()` copy,
- a full sklearn `Pipeline.predict` (StandardScaler + LogisticRegression) on a **single row**.

For N_test trajectories at scale (thousands) × 8 depths × ~10 predict calls each = tens of thousands of single-row sklearn calls with a sync apiece. Batch all states per depth and call `predict` once on a stacked array.

## P3 — `.item()` / `.cpu()` inside hot loops — MEDIUM
- `coherence.py:83-88` `.item()` on `mse`, `cos`, `id_mse`, `id_cos` every depth (forces sync).
- `intrinsic_noise.py:144-163` `.item()` per pair inside the sampling loop.
- These are unavoidable-ish given the metric design but compound P1/P2.

## P4 — DataLoaders run single-process, no pinning/AMP — MEDIUM
`train_transition.py:72`, `train_decoder.py:62`: `DataLoader(..., shuffle=True)` with **default `num_workers=0`** and no `pin_memory`. Training data are small in-memory tensors so this is tolerable, but on the full extraction the trajectory build (`build_trajectories`) is the bottleneck and is already batched (good). No mixed precision in training despite `kaggle.yaml` advertising `bf16` (the yaml is unused — see portability §2.3).

## P5 — Redundant recomputation across phases — MEDIUM
`run_phase_b.py:332` re-runs `run_probes` (fits 4 logistic regressions on all train states) that Phase A already fit and could have serialized. Each Phase C/D script reloads and re-extracts the full trajectory set from scratch. Phase A trains **two** transition models + decoder + 3 coherence evals + oracle sequentially with no checkp-skip.

## P6 — Full M×M similarity matrices materialized — MEDIUM
`intrinsic_noise.py:200` `sim_matrix = H_mat @ H_mat.T` is M×M (10⁴×10⁴ = 10⁸ floats ≈ 400 MB fp32) held in RAM alongside two M×M boolean masks. `run_phase_c1_advanced.py:158` and `:167-194` build **three** more M×M masks (M capped at 5,000 → ~25M each). Memory-heavy but bounded.

## What's already good
- Hidden-state extraction is properly **batched** with padding+attention masks (`trajectory_dataset.py:167-278`).
- Phase C4a retrieval is **vectorized** in NumPy batches of 1000 (`run_phase_c4a:133-167`) — the correct pattern that Phase C (`intrinsic_noise.py`) fails to follow.

## Verdict
The training/extraction path is fine. The **diagnostic** path is the problem: Phase C's `compute_intrinsic_noise` has two ~10⁸ Python loops (P1) and the coherence rollout does tens of thousands of single-row sklearn predictions with device syncs (P2). On anything beyond the 52-state smoke set these dominate runtime and would make the "full 50k" Stage-3 plan in `PROJECT_STATUS.md` impractical without rewrites.
