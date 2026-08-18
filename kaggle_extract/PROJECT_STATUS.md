# Project Status

## Pending Before First Scientific Run:
- [x] Decoder ceiling metrics (tracked during Coherence Rollout)
- [x] Identity baseline (tracked during Coherence Rollout)
- [x] Teacher accuracy check (`scripts/evaluate_teacher.py`)
- [x] OOD test split (`scripts/generate_countdown_dataset.py`)

## Kaggle Execution Plan
### Stage 0: Smoke Test
- Run `generate_countdown_dataset.py` with very small sample sizes.
- Run `evaluate_teacher.py` (200 samples) to verify target reasoning.
- Run full caching, phase A training, and coherence.

### Stage 1: Full Artifact Generation
- Train size: 5,000 | Val: 500 | Test: 500 (IID) | Test: 500 (OOD)
- Generate coherence curves, decoder ceilings, identity baselines.

### Stage 2: Action-Blind Rollouts
- Rerun evaluation with `use_action=False`.
- Generate overlapping curves to check if the latent state tracks algorithmic progression unconditionally.

### Stage 3: Large-Scale Run
- If Stage 1 & 2 results prove robust latent planning potential, scale generation to `50,000` trajectories and cache hidden states for rigorous phase B.
