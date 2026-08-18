# Phase 1 — Repository Integrity Audit

**Method:** `git ls-files`, line counts, and per-file reads. Every claim below is traceable to a file/line. Items I could not execute end-to-end are labelled **UNVERIFIED**.

## 1.1 Inventory (60 tracked Python files, ~6,168 LOC)

| Area | Files | Used by pipeline? |
|---|---|---|
| Data | `data_processing/action_parser.py`, `trajectory_dataset.py` | yes |
| Data gen | `scripts/generate_countdown_dataset.py` | yes (via `run_phase_a.maybe_generate_data`) |
| Models | `models/transition_model.py`, `diagnostic_decoder.py`, `model_loader.py` | partial (see 1.3) |
| Training | `training/train_transition.py`, `train_decoder.py` | yes |
| Eval | `evaluation/coherence.py`, `oracle_coherence.py`, `probes.py`, `intrinsic_noise.py`, `plotter.py` | yes |
| Orchestration | `scripts/run_phase_a.py` … `run_phase_d.py` (+ c1/c1_advanced/c2/c3/c4a) | entry points |
| Tests | `tests/test_*.py` (9 files) | yes; **29 pass, 1 skip** (verified) |

## 1.2 Dead code / orphan files (committed but unreachable)

These should not be in a reproducibility artifact. All confirmed by grep (zero call sites) or by inspection:

1. **`inject_sys_path.py`** — one-shot script that *rewrites every `scripts/*.py`* to prepend a `sys.path.insert`. Already applied; running it again re-mutates source. **Dead + dangerous.**
2. **`inject_sys_path_fix.py`** — a second one-shot "fix" for the damage done by #1 (it even contains an abandoned half-written loop, lines 19–33, that does nothing). **Dead.**
3. **`patch_run_phase_batch.py`** — string-replacement patcher that edits `run_phase_a.py` in place. Already applied. **Dead + dangerous.**
4. **`append_walkthrough.py` / `append_walkthrough_c.py`** — append prose to a **hardcoded personal path** `C:\Users\singh\.gemini\antigravity\brain\<uuid>\walkthrough.md` (see portability_audit.md). **Dead + non-portable.**
5. **`cpu_hotspots_raw.txt`, `gpu_hotspots_raw.txt`** — raw profiler dumps committed to root.
6. **`build_trajectory()`** (singular) in `trajectory_dataset.py:110` — **zero call sites** (grep). Superseded by `build_trajectories()` (batched). Note: it also lacks the dedup/skip logic the batched version has.
7. **`evaluation/plotter.py:33 plot_evaluation_metric`** — **zero call sites** (grep). Only `plot_training_curves` is used.
8. **Legacy single-model artifacts** in `reports/`: `coherence_depth.csv`, `coherence_depth.png`, `transition_model.pt`, `transition_train_log.csv`, `transition_train_curve.png` coexist with the current `*_action`/`*_blind` artifacts. They are outputs of a previous code version and are stale (the current `run_phase_a.py` never writes `coherence_depth.csv`).

## 1.3 Unused subsystems (the fine-tuning path is never exercised)

`scripts/run_phase_a.py` and `run_phase_b.py` call **`load_model` (base) only** — grep confirms **no** use of `load_checkpoint`, `load_qlora_model`, `is_peft`, or `train_teacher` anywhere in the phase pipeline.

Consequently the entire QLoRA/teacher-training stack is dead relative to the experiment that actually runs:
- `models/model_loader.py`: `load_qlora_model`, `save_checkpoint`, `load_checkpoint` (lines 40–114)
- `scripts/train_teacher.py`, `scripts/evaluate_teacher.py`, `scripts/cache_hidden_states.py`, `scripts/extract_hidden_states.py`

This matters scientifically: the "teacher" is **base TinyLlama-1.1B reading the dataset's provided ground-truth solution text**, not a model trained to solve Countdown. See experimental_validity_audit.md §E2.

## 1.4 Duplicate / superseded implementations

`MultiLabelProbe` is **re-implemented three times** with drift:
- `evaluation/probes.py:143`
- `scripts/run_phase_c4a_pca_ablation.py:65`
- `scripts/run_phase_c3_geometry.py:55`

`extract_all_targets` / `extract_probe_data` likewise re-implemented in c3, c4a, and `probes.py` with subtly different label semantics (c3 uses `Depth=i`; probes use `B=N−i`). Divergence risk: a fix to one is silently absent from the others.

## 1.5 Stale documentation

- `README.md` lists `datasets/` and `notebooks/` directories. **`datasets/` does not exist**; `.gitignore` still references it. `notebooks/` exists but is empty of tracked files.
- `README.md` "Usage" points to `train_teacher.py` / `extract_hidden_states.py` as the workflow — but the real workflow is `run_phase_a.py` (not mentioned).
- `evaluation/coherence.py:17` docstring and `run_phase_a.py:223` master-report text tell the reader to consult `coherence_depth.csv`, a filename the pipeline no longer produces.

## 1.6 Verdict

The *core* library (data_processing, models, training, evaluation) is coherent and tested. The *repository around it* is littered with one-shot mutation scripts, a hardcoded personal path, profiler dumps, three copies of the same probe, an unused fine-tuning stack, and stale artifacts/docs. This is a working scratch tree, not a clean reproducibility package.
