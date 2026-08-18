# Executive Red-Team Report — Latent Planning Repository

**Auditor roles:** Senior ML Systems Eng · Senior PyTorch Eng · NeurIPS Reproducibility Reviewer · Production Reliability Eng
**Method:** full read of all 60 tracked `.py` files; ran `pytest` (29 pass / 1 skip) and inspected raw artifacts in `reports/`. Conclusions are traced to `file:line` or raw CSV; anything not executed end-to-end is labelled **UNVERIFIED** in the per-phase reports.

## TL;DR
The **engineering core is clean and tested**, but the **experiment as currently designed and reported cannot support its headline claim**. The shipped numbers come from a 52/63-sample smoke run, two phases contain confirmed train-on-test leakage, the two "positive" probes are position detectors, and the project's own identity-baseline CSV shows the transition model **losing to doing nothing by depth 3**. The good news is buried and points against the hypothesis.

## What's genuinely good
- Coherent, documented core library; **29/30 tests pass**.
- Strong controls *exist*: identity baseline, action-blind ablation, Oracle representation-ceiling.
- Hidden-state extraction is correctly batched; Phase C.4a does PCA leakage correctly.
- Dataset dedup at problem level is present.

## The five findings that matter
1. **Teacher doesn't plan** — base TinyLlama reads dataset-provided solutions; no planner to recover (`run_phase_a.py:301`). *(validity E1)*
2. **Data isn't reasoning** — unseeded random-walk "Countdown" with distractors, targets like 5003, no division (`generate_countdown_dataset.py`). *(E2)*
3. **Results are a smoke run** — 52 train / 63 test states, n_samples=20, cosine 0.43 — yet categorical "encoded"/"horizon" verdicts are emitted. *(metric M0)*
4. **Confirmed leakage** — C.3 fits PCA on full set pre-split; Phase D trains the projection on train+test then scores on the same states. *(leakage L1/L2)*
5. **The model loses to identity** — `dynamics_gain<1` by depth 3 in `coherence_action_depth.csv`; Probe A "encoded" with **F1=0.000**. *(M1/M4)*

## Scores (/10)

| Dimension | Score | Basis |
|---|---|---|
| **Repository Quality** | 4 | Clean core, but committed mutation scripts, hardcoded personal path, 3× duplicated probes, dead fine-tuning stack, stale docs/artifacts |
| **Engineering Quality** | 6 | Tests pass, batched extraction, good guards — but unconditional `.cuda()` crash in Phase D, O(M²) loops, split dependency contract |
| **Reproducibility** | 2 | Unseeded generator; all data/artifacts/results gitignored; no CIs/seeds; results untraceable to any commit |
| **Scientific Validity** | 2 | Non-planning teacher on non-reasoning data; position-confounded probes; leakage in C.3/D; no negative controls |
| **Performance Quality** | 5 | Training fine; diagnostics have 10⁸ Python loops and per-row sklearn-with-sync in rollout |
| **Risk Level** | 8 | High risk that any "promising" conclusion is an artifact |

## Final recommendation: **MAJOR REVISIONS REQUIRED — current research conclusions are UNVERIFIED / not supported**

Do **not** quote any number currently in `reports/`. Before any scientific claim:

**Must-fix to make results meaningful**
1. Use a real reasoning setup: either fine-tune a teacher and extract **its own** generations, or drop the "planning" framing and call it a representation-geometry study.
2. Replace the data generator with genuine Countdown (search-solved, all 4 ops, bounded targets, distractor control) **and seed it**.
3. Run at documented scale (≥5k/500) with **≥3 seeds and confidence intervals**; add a **permutation-label control** and a **raw-operand-token-embedding probe baseline**.
4. Fix the two leaks (C.3 PCA-before-split; Phase D train-on-test).
5. Partial out **depth/position** before interpreting Probe B/D; mask shared **prefixes** in Phase C similarity/retrieval.

**Must-fix for the repo to be shippable**
6. Delete `inject_sys_path*.py`, `patch_run_phase_batch.py`, `append_walkthrough*.py`, profiler dumps; `pip install -e .` instead of `sys.path` hacks.
7. Guard CUDA in `run_phase_d.py`; reconcile `requirements.txt` vs `pyproject.toml`; lazy-import bitsandbytes.
8. Vectorize `intrinsic_noise.compute_intrinsic_noise`; batch probe predictions in `coherence.py`.
9. Commit a results manifest with data/checkpoint hashes; fix the `coherence_depth.csv` filename drift.

**Honest reframing of what the data already shows:** the identity baseline beating the transition model (depth 3) and the F1=0 Probe A are real, useful negative results. Leaning into them — "latent dynamics in a frozen base LM on this toy do **not** beat the trivial copy baseline" — is a defensible, publishable finding. The current "latent planning is promising" direction is not.

---
*Per-phase detail: `repository_integrity.md`, `portability_audit.md`, `import_audit.md`, `path_audit.md`, `leakage_audit.md`, `metric_inflation_audit.md`, `numerical_stability_audit.md`, `performance_audit.md`, `experimental_validity_audit.md`, `failure_modes.md`.*
