# Git Hygiene Audit

## Repository Inventory Summary
- **File Count**: 324 files
- **Total Repository Size**: ~5.38 GB
- **Largest Files**: 
  1. `reports_qwen/trajectories/train_full.pt` (2.15 GB)
  2. `reports/trajectories/train.pt` (606 MB)
  3. `reports/trajectories/test.pt` (605 MB)

## Commit Recommendations

**Files that SHOULD be committed:**
- Source code (`*.py` in `scripts/`, `models/`, `training/`, `evaluation/`, `data_processing/`, `tests/`)
- Configurations (`configs/*.yaml`, `pyproject.toml`, `requirements.txt`)
- Markdown documentation and reports (`*.md`)
- Summarized JSON and CSV metrics (`reports/*.json`, `reports/*.csv`)
- Small plots and figures (`reports/*.png`)

**Files that should NOT be committed:**
- Checkpoints (`checkpoints/`, `checkpoints_qwen/`, `*.pt`)
- Cached hidden states and trajectories (`reports/trajectories/*.pt`, `reports/discrete_trajectories/*.pt`, `reports_qwen/trajectories/*.pt`)
- Raw Datasets (`data/*.jsonl`)
- Scratch logs (`scratch_logs/*.log`, `run_real.log`)
- Temporary and compiled files (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/`)

## Recommended `.gitignore` Entries
```text
# Python caches
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
*.egg-info/

# PyTorch artifacts
*.pt
*.pth

# Large datasets
data/
*.jsonl

# Logs
scratch_logs/
*.log

# Checkpoints
checkpoints/
checkpoints_qwen/

# Specific large directories
reports/trajectories/
reports_qwen/trajectories/
reports/discrete_trajectories/
reports_qwen/discrete_trajectories/
reports/scrubbed/
reports_qwen/scrubbed/
```
