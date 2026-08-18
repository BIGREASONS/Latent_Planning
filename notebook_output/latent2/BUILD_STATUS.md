# BUILD_STATUS.md

Build/verify pass to prepare the repo for a clean large-scale **Phase C.4A** run.
No models were downloaded; no trajectories generated; no training, Phase A, or
Phase C.4A executed. Verification was limited to `py_compile`, import checks, and
model-free smoke validation.

Date: 2026-06-14

## 1. Fix verification (all six present)

| Fix | Location | Status |
|---|---|---|
| Seeded generator | `generate_countdown_dataset.py` `--seed`, `random.seed()` per split | VERIFIED |
| Bounded targets | `--min_target/--max_target` (default 100–999) + rejection sampling | VERIFIED |
| Division support | `_build_walk` adds `'/'` when remainder-free; `apply_op` enforces it | VERIFIED |
| Prefix-masked retrieval | `compute_prefix_collisions()` + `dup_exclude` in `evaluate_subset` | VERIFIED |
| Permutation null | `permute_labels()` + control row in report/CSV | VERIFIED |
| Raw-operand baseline | bag-of-prompt-numbers (`BAG_VOCAB`/`Bag`) through same probes | VERIFIED |

Generator smoke (model-free) confirmed: determinism (same seed → identical
output), all targets in [100,999], division present, every step parses and is
arithmetically valid.

## 2. Repository consistency audit

| Check | Result |
|---|---|
| `py_compile` / `compileall` (scripts, data_processing, evaluation, models, training, tests) | PASS |
| Import check `run_phase_c4a_pca_ablation` | PASS (no model load on import) |
| Import check `generate_countdown_dataset` | PASS |
| Hardcoded absolute paths in `.py` | NONE (only existed in deleted profiler dumps) |
| Windows-specific paths in code | NONE (all paths via `os.path.join`, repo-relative) |
| Trajectory loading paths repo-relative | YES — `os.path.join(--reports_dir, "trajectories", "train.pt")` |
| Stale / footgun scripts | REMOVED (see FIXES_APPLIED.md) |
| Dead imports in C.4A | REMOVED (`random`, `cosine_similarity`) |
| Kaggle config (`configs/kaggle.yaml`) paths | repo-relative, OK |

## 3. Verification commands run

- `python -m compileall scripts data_processing evaluation models training tests` → OK
- `python -c "import scripts.run_phase_c4a_pca_ablation, scripts.generate_countdown_dataset"` → OK
- `python scripts/run_phase_c4a_pca_ablation.py --help` → shows new CLI args
- Generator smoke (200 samples ×2 seeds) → determinism/bounds/division/parse all OK
- `pytest tests/test_dataset.py tests/test_action_parser.py` → **13 passed** (model-free only)

NOT run (per task constraints): TinyLlama, trajectory extraction, Phase A, Phase
C.4A, any training, and the model-invoking tests (`test_pipeline`,
`test_model_loading`, etc.).

## 4. Known issues (NOT in the Phase C.4A path — documented, not changed)

1. **`scripts/run_phase_d.py` uses unconditional `.cuda()`** (lines 81, 90–92,
   172, 181). Will crash on a CPU-only box. Out of scope for C.4A; fix before any
   Phase D run by guarding with `torch.device("cuda" if torch.cuda.is_available()
   else "cpu")`.
2. **`scripts/run_phase_c3_geometry.py` fits PCA/scaler before the train/test
   split** (leakage). C.4A does this correctly (train-only); C.3 remains
   uncorrected and should not be quoted.
3. **Test suite teardown noise**: a `datasets`/`multiprocess` `ResourceTracker`
   `AttributeError` prints at interpreter shutdown on Windows. Cosmetic; does not
   affect test outcomes (banner shows pass counts before it).
4. **`pytest` summary banner can be swallowed when output is piped** due to (3);
   run test files directly (no pipe) to see the banner.

## 5. Bottom line

The Phase C.4A run path (`generate_countdown_dataset.py` →
`reports/trajectories/*.pt` → `run_phase_c4a_pca_ablation.py`) compiles, imports
cleanly, has repo-relative configurable paths, and exposes the CLI arguments
needed for Kaggle. The repo is build-clean for a large-scale C.4A run once
trajectories are extracted at scale (not done here, by design).
