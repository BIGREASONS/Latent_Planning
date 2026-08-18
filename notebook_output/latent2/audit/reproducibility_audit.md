# Reproducibility Audit

## Environment Setup & Requirements
- `requirements.txt` includes: `transformers>=4.40`, `datasets>=2.16`, `accelerate>=0.28`, `peft>=0.10`, `wandb`, `scikit-learn`, `matplotlib`, `pandas>=2.0.0`, `python-dotenv`, `bitsandbytes`, `tqdm`.
- `pyproject.toml` contains identical dependencies and adds development dependencies (`pytest`, `black`, `flake8`, `isort`).

## Scripts Needed to Reproduce V5.3
- `scripts/run_v5_discrete_states.py`: Main orchestrator for the V5 discrete state pipeline.
- `training/train_vq.py`: Standalone script used to train the VQ state quantizer.
- `scripts/generate_countdown_dataset.py` & `scripts/generate_transfer_datasets.py`: For underlying dataset generation.

## Missing Files & Broken References
- **Missing Files**: No missing code files were detected, but **V5.2 reports** are completely missing from the `reports/` and `reports_qwen/` directories.
- **Hardcoded Paths**: None found. All paths appear to be constructed relatively (e.g., `os.path.join`).

## Undocumented Dependencies
- **`pytest`**: Used in test instructions (`pytest tests/` in `V5_READINESS.md`) but missing from `requirements.txt` (it is only available in `pyproject.toml` dev group).
- **`pyyaml`**: Mentioned in `V5_READINESS.md` setup instructions (`pip install ... pyyaml`) but absent from both `requirements.txt` and `pyproject.toml`.
- **`scipy`**: Emits deprecation warnings according to `V5_READINESS.md`. It is an implicit dependency of `scikit-learn` but not explicitly pinned.
