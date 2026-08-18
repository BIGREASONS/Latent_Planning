# Phase 2 — Portability Audit

## 2.1 Hardcoded user-specific path (CRASH off this machine) — HIGH

```
append_walkthrough.py:19    path = r"C:\Users\singh\.gemini\antigravity\brain\04b26a60-...\walkthrough.md"
append_walkthrough_c.py:18  path = r"C:\Users\singh\.gemini\antigravity\brain\04b26a60-...\walkthrough.md"
```
Opening this path in append mode crashes (`FileNotFoundError`) on any other machine/user. These are dev-logging scripts that should never have been committed. **Fix:** delete both files.

## 2.2 Unconditional `.cuda()` (CRASH on CPU-only / non-NVIDIA) — HIGH

`scripts/run_phase_d.py` assumes CUDA at 6 sites:
```
81   model.cuda()
90-92 .cuda() on h1, h2, y
172  tm.cuda()
181  bx, bid, bop, by = bx.cuda(), ...
```
There is **no `torch.cuda.is_available()` guard**. On a CPU-only box (and this audit machine is CPU-only — `pin_memory` warning in pytest confirms no accelerator) Phase D dies immediately with `RuntimeError: ... no CUDA`. Every other script uses the correct `device = torch.device("cuda" if ... else "cpu")` idiom — Phase D is the lone offender. **Fix:** route through a `device` variable like the rest.

## 2.3 Kaggle / environment coupling — MEDIUM

- `requirements.txt:1-2` comments "torch provided by Kaggle / torchvision provided by Kaggle" — torch is **not** in `requirements.txt`, so `pip install -r requirements.txt` yields an environment that cannot import torch off Kaggle. `pyproject.toml` *does* pin `torch==2.8.0`, so the two installers disagree. **Fix:** make one canonical; add torch to requirements with a guard.
- `pyproject.toml` pins exact versions (`transformers==4.57.1`, `peft==0.17.1`, `accelerate==1.11.0`, `datasets==4.4.1`) while `requirements.txt` pins floors (`>=`). Two different dependency contracts in one repo.
- `configs/kaggle.yaml` hardcodes relative output dirs (`data/…`, `checkpoints`, `runs`) — fine, but note **the config is not read by any phase script** (grep: no yaml loader in the run_phase_* pipeline). It is decorative. UNVERIFIED whether anything consumes it.

## 2.4 Hardcoded model dimension (breaks on any non-2048 model) — MEDIUM

The pipeline is advertised as "model-agnostic" (`run_phase_a.py:13`), but several Phase C/D scripts hardcode TinyLlama's hidden size:
```
run_phase_d.py:19,80,171,264   ProjectionNet(2048,...) / TransitionModel(hidden_dim=2048) / build_transition_data(...,256)
run_phase_c1_advanced.py        (PCA/classifier infer dim, OK)
run_phase_c2_depth_removal.py:42,112  AdversarialProjector(2048,256,...)
run_phase_c3_geometry.py        PCA n_components=100 (OK), but report text hardcodes "2048"
run_phase_c4a_pca_ablation.py:229  n_comp = min(2048, ...)
```
Switch `--model` to anything but a 2048-d model and Phase C.2/D throw shape errors. Phase A itself reads `hidden_dim` dynamically (good); the downstream phases regress on this.

## 2.5 Windows/OneDrive specifics — LOW

- Working tree lives under `OneDrive\Documents` → file locking / sync races possible during long runs (artifacts written mid-sync). Not a code bug but a reproducibility hazard.
- All file paths in code use `os.path.join` / `os.makedirs` correctly — **no** forward/back-slash hardcoding inside the library. Good.

## 2.6 Stale pytest rootdir — LOW

`pytest` reported the skipped test as `..\..\..\.gemini\antigravity\scratch\latent_planning\tests\test_model_loading.py` — a path that **does not exist** on disk. The repo was developed under `.gemini/antigravity/scratch/` and moved to OneDrive; a stale `.pytest_cache` or rootdir inference is surfacing the old location. Harmless to results, confusing to readers. **Fix:** clear `.pytest_cache`.

## 2.7 Verdict

Two guaranteed crashes off the author's machine (`append_walkthrough*.py` hardcoded path; `run_phase_d.py` unconditional CUDA), a split dependency contract (requirements vs pyproject), and a "model-agnostic" claim that is false for Phase C.2/D. None affect Phase A on the original Kaggle box, but the repo is not portable as shipped.
