# Changelog

## [Unreleased] Kaggle Readiness Refactoring (Phase A)

### Added
- **Action-Blind Transition Support**: Modified `TransitionModel` architecture to support `use_action=False`, allowing autonomous state rollouts without explicit action conditioning.
- **Kaggle Configuration**: Created `configs/kaggle.yaml` containing large-scale hyperparameter defaults, dataset paths, mixed precision settings (`bf16`), and artifact logging paths.
- **Hidden State Caching**: Implemented `scripts/cache_hidden_states.py` to batch generate and save latent trajectory tensors to disk, preventing repeated Hugging Face model inferences during training.
- **Artifact Directories**: Created standard directory structure (`artifacts/`, `runs/`, `checkpoints/`) for organized output management.

### Changed
- **Coherence Metric Overhaul**: Replaced the misleading `token_accuracy` metric with `operator_accuracy`, evaluating the linear decoding of the exact next mathematical operation. Also integrated `state_probe_accuracy` to track algorithmic state-tracking robustness via Probe A across rollouts.
- **Dynamic Evaluation Depth**: `evaluate_coherence` now dynamically bounds the rollout limit based on the 95th percentile of evaluation trajectory lengths, preventing erratic low-sample artifact spikes at depths 6-8.
- **Probe Label Granularity**: Enhanced Probe A to predict a 3-class target representing resource consumption (0 = absent, 1 = present unused, 2 = present used), replacing the previous binary target that conflated missing resources with consumed resources.
- **Dataset Scale Defaults**: Set `scripts/generate_countdown_dataset.py` generation defaults to 5,000 (train), 500 (val), and 500 (test) as a validation limit before doing the massive 50k generation.
- **Operand Normalization**: Replaced `nn.LayerNorm(2)` with a far more stable dataset-wide Z-score scaling strategy (`operand_mean` and `operand_std` injected dynamically during `train_transition.py`).
