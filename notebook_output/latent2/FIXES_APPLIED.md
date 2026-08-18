# FIXES_APPLIED.md

Changes made during the build/verify pass (2026-06-14). All deletions are of
git-tracked files and are recoverable with `git checkout -- <file>`. Nothing was
committed or pushed.

## A. `scripts/run_phase_c4a_pca_ablation.py`

1. **Removed dead imports** — `import random` and
   `from sklearn.metrics.pairwise import cosine_similarity` were unused.
2. **Added Kaggle CLI arguments** (replacing hardcoded behavior):
   - `--max_trajectories` (default 10000, `0` = use all) — replaces the
     hardcoded `if len(trajs) >= 10000: trajs = trajs[:10000]` cap that would
     silently truncate a large-scale run.
   - `--train_frac` (default 0.8) — exposes the previously hardcoded
     train/test (DB/query) split fraction.
   - `--seed` (default 0) — wired into the permutation-null control
     (`permute_labels`); the rest of the pipeline is already deterministic.
   - `--reports_dir` help text clarified for Kaggle (`/kaggle/working/reports`).
3. No change to the scientific logic (prefix mask, permutation null, raw-operand
   baseline are unchanged from their prior implementation).

## B. `scripts/generate_countdown_dataset.py`

No new changes this pass — verified the previously-applied fixes are intact:
seeded generation (`--seed`), bounded targets (`--min_target/--max_target`,
default 100–999, rejection sampling), and integer division support.
(`generate_dataset(seed=0)` default kept for backward-compatible test call.)

## C. Stale / footgun scripts removed (git-tracked, unreferenced)

| File | Why removed |
|---|---|
| `inject_sys_path.py` | One-off mutator that **rewrites every `scripts/*.py`** to inject `sys.path`. Already applied; re-running corrupts the tree. |
| `inject_sys_path_fix.py` | Same footgun (the "fix" variant). |
| `patch_run_phase_batch.py` | One-off mutator that **rewrites `run_phase_a.py`** to add `batch_size`. Already applied. |
| `append_walkthrough.py` | Appends to a **hardcoded personal path** `C:\Users\singh\.gemini\...\walkthrough.md`; crashes off that machine. |
| `append_walkthrough_c.py` | Same hardcoded personal Windows path. |

Confirmed unreferenced: `grep -r` across `*.py` found no imports/uses of any of
these. Removal eliminates the only hardcoded Windows paths and the codebase-
mutation footguns.

## D. Profiler dumps removed (git-tracked cruft)

| File | Why removed |
|---|---|
| `cpu_hotspots_raw.txt` | Raw cProfile output containing machine-specific absolute paths (`C:\Users\singh\...`). |
| `gpu_hotspots_raw.txt` | Same; raw profiling artifact, not used by any code. |

## E. Not changed (deliberately out of scope)

- `run_phase_d.py` unconditional `.cuda()` — documented in BUILD_STATUS §4, not
  on the C.4A path.
- `run_phase_c3_geometry.py` PCA-before-split leakage — documented, not on the
  C.4A path.
- `.claude/settings.local.json` shows as modified from tooling, not an
  intentional edit.

## Summary

- **Modified:** `scripts/run_phase_c4a_pca_ablation.py`
- **Deleted:** 5 stale scripts + 2 profiler dumps (all recoverable via git)
- **Added:** `BUILD_STATUS.md`, `FIXES_APPLIED.md`, `KAGGLE_READINESS.md`
