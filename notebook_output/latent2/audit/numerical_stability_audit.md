# Phase 7 — Numerical Stability Audit

Overall the code is **defensively written** for divide-by-zero; the real numerical issues are subtler (scale mismatch, silent NaN propagation in means).

## Guards that are present (verified OK)
- `operand_std + 1e-8` (`train_transition.py:67`); normalization in `ActionEncoder` (`transition_model.py:42`).
- L2 norm `+ 1e-8` in c4a retrieval (`run_phase_c4a_pca_ablation.py:117-118`) and `F.normalize` (which has internal eps) in `intrinsic_noise.py:197`, `run_phase_c2:55`.
- `max(n, 1)` denominators in all training/eval loss reductions (`train_transition.py:46,106`, `train_decoder.py:44`).
- AUC wrapped in `try/except ValueError` for missing classes (`probes.py:116-124`).
- PCA `n_components=min(2048, n_samples, n_features)` (`c4a:229`) avoids over-rank request.
- Division denominators checked: `noise_ratio` (`run_phase_c.py:43`), `dynamics_gain` only divides by `mse` which is ≥0 and effectively never 0.

## Issues found

### NS1 — `mean()` over lists containing NaN silently poisons aggregates — MEDIUM
`coherence.py:_mean` (`:134`) returns `nan` for empty lists (fine), but the **per-row** means feed `df["dynamics_gain"] = identity_mse / mse` (`:161`). If any depth has `n_samples==0`, `mse` is `nan`, and `nan` propagates into the ratio and into every downstream `cov[...]` comparison in `generate_master_report` (`run_phase_a.py:147-155`), where `_depth_where` uses `>=` against NaN (always False) — silently yielding horizon=0 rather than erroring. A reader cannot distinguish "no signal" from "no data."

### NS2 — `cosine_similarity` on (near-)identical vectors → exactly 1.0, treated as signal — MEDIUM
Not a stability *crash*, but per leakage_audit L3, prefix-identical hidden states give cosine exactly 1.0. The within/between KDE plots (`intrinsic_noise.py:262`) and ratios are dominated by these degenerate 1.0s. Numerically "stable," scientifically misleading.

### NS3 — Rolled-out states are fed to a scaler fit on teacher states — MEDIUM
`coherence.py` passes predicted `h` (which can drift to large norm under repeated `+delta`) into probes whose `StandardScaler` used **teacher** mean/std. Far-from-train inputs produce extreme standardized values → logistic predictions saturate. No clipping. At large depth this can manifest as abrupt accuracy collapse that is an artifact of scaling, not representation.

### NS4 — `roc_auc_score` multiclass can still raise / return NaN that is averaged out — LOW
`_fit_eval_multilabel:135` skips NaN AUCs when averaging, so a label that always errors is dropped silently; the reported mean AUC then reflects a subset of labels without disclosure.

### NS5 — Adversarial λ schedule fine; GRL stable — LOW
`run_phase_c2:125` `lambda = 2/(1+e^{-10p}) − 1` is bounded [0,1); gradient reversal (`:24-36`) is standard. No instability. `F.normalize(z)` before heads (`:55`) prevents logit blow-up. OK.

### NS6 — No `nan`/`inf` assertions on hidden states at extraction — LOW
`build_trajectories` (`trajectory_dataset.py:231`) casts fp16 model outputs to fp32 and stores them with no `torch.isfinite` check. A fp16 overflow in the frozen LM would propagate undetected into every downstream metric.

## Verdict
No live divide-by-zero or `log(0)` (there are no `log` calls on data). The genuine numerical risks are **silent NaN propagation through `_mean`/`dynamics_gain`/`_depth_where`** (NS1) and **scale mismatch between rolled-out states and teacher-fit scalers** (NS3) — both of which corrupt reported metrics quietly rather than crashing. Add `isfinite` assertions at extraction and at each aggregation, and clip/renormalize rollout states before probing.
