# Publication Readiness

## Research Artifact Audit
- [x] **V5.1 reports**: Present (`V5_1_FINAL.md`, `V5_1_FINAL.json`, `V5_1_comparison.png`)
- [ ] **V5.2 reports**: **Missing** (Not found in `reports/` or `reports_qwen/`)
- [x] **V5.3 reports**: Present (`V5_3_VERDICT.md`, `v5_discrete_state_report.md`)
- [x] **Positive control**: Present (`positive_control.md`, `positive_control.json`, `qwen_positive_control.md`)
- [x] **Action conditioned experiments**: Present (`action_conditioned.md`, `action_conditioned.json`, `qwen_action_conditioned.md`)
- [x] **Layer validation**: Present (`layer_sweep.md`, `qwen_layer_validation.md`)
- [x] **Qwen replication**: Present (`reports_qwen/` directory exists with outputs)

## Publication Readiness Answers

**Is repository safe to push privately?**
No. While there are no security risks (no API keys, tokens, or personal info), the repository contains multiple files exceeding GitHub's hard 100MB file size limit (e.g., `train_full.pt` is 2.15 GB). A push to a private repository will fail and be rejected by GitHub unless Git LFS is used.

**Is repository safe to push publicly?**
No. In addition to the file size limits preventing the push entirely, the repository currently has poor Git hygiene. It includes model checkpoints, cached tensors, raw datasets, and scratch logs which should not be open-sourced as part of the source code repository.

**What must be removed before public release?**
- All `.pt` files (e.g., `reports/trajectories/*.pt`, `checkpoints/*.pt`)
- All `.jsonl` dataset files (e.g., `data/*.jsonl`)
- All scratch logs (`scratch_logs/*.log`, `run_real.log`, etc.)
- Python compilation artifacts (`__pycache__`, `.pytest_cache`, and `latent_planning.egg-info/` directories).

## RECOMMENDATION

**B. Safe only after cleanup**
