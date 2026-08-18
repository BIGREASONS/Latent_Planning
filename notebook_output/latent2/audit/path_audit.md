# Phase 4 — Path Audit

**Method:** traced every `open`, `torch.save/load`, `pd.read_csv`, `to_csv`, `savefig`, `os.makedirs`.

## 4.1 Pickle deserialization of trajectories — SECURITY / MEDIUM

```
data_processing/trajectory_dataset.py:298  torch.load(path, weights_only=False)
```
`weights_only=False` executes arbitrary pickle on load. Every phase after A does `load_trajectories("reports/trajectories/*.pt")`. If a `.pt` came from an untrusted source it is RCE. For a single-author repo the risk is low, but `weights_only=False` is the unsafe default and the trajectories are **not** pure tensors (they are dataclass objects), so it cannot simply be flipped to `True` without a custom `safe_globals` registration. Flag as a known hazard.

## 4.2 Cross-phase artifact contract is implicit and order-dependent — MEDIUM

Phases B/C/C1/C2/C3/C4a/D all hard-depend on files written by Phase A:
```
reports/trajectories/{train,test,val,test_ood}.pt
reports/coherence_action_depth.csv, coherence_blind_depth.csv
```
Most scripts guard with `os.path.exists(...) → sys.exit(1)` (good: `run_phase_b.py:320`, `run_phase_c.py:104`, `run_phase_c1.py:26`, `run_phase_d.py:210`). **But the guards are inconsistent:**
- `run_phase_c2_depth_removal.py:82-85`, `run_phase_c3_geometry.py:136`, `run_phase_c4a_pca_ablation.py:189` load `train.pt` **with no existence check** → raw `FileNotFoundError`/pickle error instead of the friendly message. Minor but inconsistent.
- `run_phase_b.py` re-derives probes by re-running `run_probes` (line 332) instead of loading Phase A's fitted probes; if the trajectory files were regenerated between A and B with a different seed, B's probes silently differ from A's. No checksum ties them together.

## 4.3 Stale-artifact dependency / filename drift — MEDIUM

- The master report (`run_phase_a.py:223`) and `coherence.py:17` reference **`coherence_depth.csv`**, but Phase A writes `coherence_action_depth.csv` / `coherence_blind_depth.csv`. A reader following the report looks for a file the current pipeline doesn't produce; the only `coherence_depth.csv` present is a **legacy artifact from older code** (see repository_integrity.md §1.2). Anyone re-deriving conclusions from `coherence_depth.csv` is reading stale numbers.
- `reports/` accumulates outputs from multiple code generations with no run-id namespacing or cleanup, so "the CSV in reports/" is ambiguous.

## 4.4 `os.path.dirname(os.path.abspath(path))` on bare filenames — LOW

Pattern used widely (e.g. `coherence.py:167`, `probes.py:252`). If a caller passes a bare filename (no directory), `os.path.dirname(os.path.abspath("x.csv"))` returns the cwd, `makedirs(cwd, exist_ok=True)` is a no-op — safe. No bug, just noting it was checked.

## 4.5 Data generation overwrites unconditionally — LOW

`scripts/generate_countdown_dataset.py:62` opens splits in `'w'` mode. `run_phase_a.maybe_generate_data` only regenerates when `existing < n` (line 68), so a *smaller* existing file is silently overwritten and a *larger* one is kept. If you run with `--generate 5000 ...` over a directory that already has a 52-line smoke `train.jsonl`, it regenerates; but if it has 6000 lines it keeps them — so the actual N used can silently differ from the requested N. No record of which is which is written to the report.

## 4.6 Reports are gitignored — REPRODUCIBILITY / MEDIUM

`.gitignore`: `reports/*` (except `.gitkeep`), `data/*`, `*.pt`, `checkpoints/`. So **none of the artifacts, datasets, checkpoints, or CSVs are version-controlled.** The numbers in `reports/*.md` cannot be tied to a commit. Combined with the non-deterministic dataset generator (no seed — see leakage/validity), re-running produces *different data and different numbers*, and nothing pins the reported results to code. This is the single biggest reproducibility gap.

## 4.7 Verdict

Paths themselves are mostly well-formed (consistent `os.path.join`, `makedirs(exist_ok=True)`). The real issues are **contractual**: unsafe pickle, inconsistent existence guards, filename drift to a stale `coherence_depth.csv`, and the fact that every artifact is gitignored so no reported number is reproducible from the committed tree.
