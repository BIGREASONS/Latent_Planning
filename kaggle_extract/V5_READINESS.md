# V5 Readiness Checklist

> **Research question (Version 5):** Do hidden-state trajectories of frozen
> language models admit a compact, predictive, reusable discrete state
> representation?

All nine components (C1–C9) are implemented, tested, and orchestrated. This
document lists every file change, the folder layout, the execution plan for
Kaggle, estimated GPU cost, and known risks.

---

## 1. New Files (20)

### Models & Training
| File | Component | Purpose |
|------|-----------|---------|
| `models/vq_state.py` | C1 | VQ-VAE quantizer with EMA codebook updates, straight-through estimator, dead-code revival |
| `training/train_vq.py` | C1 | Train VQ quantizer on cached Phase-A trajectory hidden states |

### Data Processing
| File | Component | Purpose |
|------|-----------|---------|
| `data_processing/discrete_trajectory_dataset.py` | C2 | Encode trajectories to discrete codes; build (z_t, z_{t+1}) transition pairs |
| `data_processing/transfer_trajectory.py` | C8 | Extract hidden-state trajectories from transfer-domain problems (algebra / logic / graph) |

### Evaluation
| File | Component | Purpose |
|------|-----------|---------|
| `evaluation/codebook_usage.py` | C5 | Gini, perplexity, active/dead/used code counts, collapse score |
| `evaluation/discrete_transition.py` | C3+C4 | Action-blind transition MLP training, transition matrix, conditional entropy |
| `evaluation/position_leakage.py` | C6 | Logistic regression: can position predict code? Permutation null |
| `evaluation/discrete_rollout.py` | C9 | Autoregressive z_t→z_{t+1} rollout, decode via codebook, evaluate with probes A/B/C |
| `evaluation/permutation_robustness.py` | C7 | Cross/within consistency of codes across multiple valid solutions |
| `evaluation/cross_domain_transfer.py` | C8 | Code reuse, entropy delta, symmetric KL across domains |

### Scripts
| File | Component | Purpose |
|------|-----------|---------|
| `scripts/generate_transfer_datasets.py` | C8 | Generate algebra, logic, and graph problem datasets |
| `scripts/run_v5_discrete_states.py` | Orchestrator | End-to-end C1→C5→C2→C3→C4→C6→C9 pipeline with 5-gate verdict |

### Config
| File | Purpose |
|------|---------|
| `configs/v5.yaml` | All V5 hyperparameters and paths |

### Tests (8 files, 38 tests)
| File | Tests | Component |
|------|-------|-----------|
| `tests/test_vq_state.py` | 5 | C1 — shapes, STE gradient, encode/decode roundtrip, EMA convergence, diagnostics |
| `tests/test_discrete_trajectory_dataset.py` | 4 | C2 — integer codes, transition pairs, concatenation, save/load |
| `tests/test_codebook_usage.py` | 5 | C5 — Gini extremes, active/dead counts, collapse detection, empty, CSV output |
| `tests/test_discrete_transition.py` | 7 | C3/C4 — stochastic matrix, empty rows, conditional entropy, training, metrics |
| `tests/test_position_leakage.py` | 5 | C6 — leaking codes, independent codes, permutation null, bounds, empty |
| `tests/test_discrete_rollout.py` | 4 | C9 — depths, cosine, MSE, degradation with depth |
| `tests/test_permutation_robustness.py` | 4 | C7 — perfect consistency, baselines, single-solution, dict keys |
| `tests/test_cross_domain_transfer.py` | 4 | C8 — near-domain reuse, arithmetic ref, entropy sign, empty domain |

---

## 2. Modified Files (2)

| File | Change |
|------|--------|
| `scripts/generate_countdown_dataset.py` | Added `generate_multi_solution_dataset()` function and `--multi_solution` / `--num_multi` / `--solutions_per_problem` CLI args for C7 |
| `evaluation/probes.py` | Added `"B": resB.get("clf")` to `fitted_probes` dict in `run_probes()` so the orchestrator can use probe B in discrete rollout decode (C9) |

---

## 3. New Folder Structure

```
latent_planning/
├── configs/
│   └── v5.yaml                          # V5 hyperparameters
├── models/
│   └── vq_state.py                      # NEW — C1
├── training/
│   └── train_vq.py                      # NEW — C1 training
├── data_processing/
│   ├── discrete_trajectory_dataset.py   # NEW — C2
│   └── transfer_trajectory.py           # NEW — C8
├── evaluation/
│   ├── codebook_usage.py                # NEW — C5
│   ├── discrete_transition.py          # NEW — C3+C4
│   ├── position_leakage.py             # NEW — C6
│   ├── discrete_rollout.py              # NEW — C9
│   ├── permutation_robustness.py        # NEW — C7
│   ├── cross_domain_transfer.py         # NEW — C8
│   └── probes.py                        # EDITED — added probe B to fitted_probes
├── scripts/
│   ├── generate_countdown_dataset.py    # EDITED — multi-solution generator
│   ├── generate_transfer_datasets.py    # NEW — C8
│   └── run_v5_discrete_states.py        # NEW — orchestrator
└── tests/
    ├── test_vq_state.py                 # NEW
    ├── test_discrete_trajectory_dataset.py  # NEW
    ├── test_codebook_usage.py           # NEW
    ├── test_discrete_transition.py     # NEW
    ├── test_position_leakage.py         # NEW
    ├── test_discrete_rollout.py         # NEW
    ├── test_permutation_robustness.py   # NEW
    └── test_cross_domain_transfer.py    # NEW
```

---

## 4. New Configs

| Config | File | Description |
|--------|------|-------------|
| V5 | `configs/v5.yaml` | All V5 hyperparameters: VQ (num_codes=256, commitment_cost=0.25, ema_decay=0.99, epochs=50, batch_size=256), discrete transition (embed=32, hidden=128, lr=0.001, epochs=30), rollout (max_depth=8), multi-solution (2000 problems, 4 solutions each), transfer (2000 per domain), seed=0 |

---

## 5. Training & Evaluation Scripts

| Script | Usage | Component |
|--------|-------|-----------|
| `python training/train_vq.py --reports_dir reports --checkpoints_dir checkpoints --num_codes 256 --epochs 50 --batch_size 256` | Standalone VQ training | C1 |
| `python scripts/generate_transfer_datasets.py --domains algebra logic graph --num_samples 2000 --output_dir data/transfer` | Generate transfer datasets | C8 |
| `python scripts/run_v5_discrete_states.py --reports_dir reports --checkpoints_dir checkpoints` | **Full end-to-end pipeline** (loads or trains VQ → runs C5, C2, C3, C4, C6, C9 → prints 5-gate verdict) | All |

The orchestrator (`run_v5_discrete_states.py`) is the primary entry point. It
mirrors `run_phase_a.py`'s structure and produces:

```
reports/
├── v5_codebook_usage.csv
├── v5_codebook_usage_logfreq.png
├── v5_predictability_metrics.csv
├── v5_entropy_report.csv
├── v5_entropy_report.png
├── v5_position_leakage.csv
├── v5_discrete_rollout.csv
├── v5_discrete_rollout.png
└── v5_master_report.json    # 5-gate verdict + all metrics
```

---

## 6. Kaggle Execution Plan (7 Steps)

All steps run on a single GPU (T4×2 or P100). No multi-GPU required.

### Step 0 — Environment Setup
```bash
pip install torch transformers sklearn matplotlib pyyaml
git clone <repo> && cd latent_planning
```

### Step 1 — Phase A Trajectories (already built, ~2–4 hr GPU)
Run the existing Phase A pipeline to produce `reports/trajectories/{train,val,test}.pt`.
If these files already exist in the working directory, skip this step.

### Step 2 — VQ Training (C1, ~10–15 min GPU)
```bash
python training/train_vq.py \
    --reports_dir reports \
    --checkpoints_dir checkpoints \
    --num_codes 256 --epochs 50 --batch_size 256
```
Output: `checkpoints/vq_state.pt`

### Step 3 — Codebook Usage Analysis (C5, ~1 min CPU)
```bash
# Handled by orchestrator, or standalone via Python:
#   from evaluation.codebook_usage import analyze_codebook_usage, save_codebook_usage_report
```

### Step 4 — Discrete Trajectory Encoding + Transition Analysis (C2→C3→C4, ~5 min GPU)
```bash
# Handled by orchestrator. Encodes all trajectories to codes,
# trains action-blind transition MLP, builds transition matrix,
# computes conditional entropy.
```

### Step 5 — Position Leakage (C6, ~30s CPU)
```bash
# Handled by orchestrator. Logistic regression + permutation null.
```

### Step 6 — Discrete Rollout Coherence (C9, ~2 min GPU)
```bash
# Handled by orchestrator. Requires fitted probes A/B/C from
# evaluation/probes.py. Autoregressive rollout, decode, probe eval.
```

### Step 7 — Run Orchestrator (single command for Steps 3–6)
```bash
python scripts/run_v5_discrete_states.py \
    --reports_dir reports \
    --checkpoints_dir checkpoints \
    --train_vq \
    --vq_epochs 50 \
    --vq_batch_size 256 \
    --transition_epochs 30 \
    --rollout_max_depth 8
```

### Optional: C7 — Permutation Robustness
```bash
python scripts/generate_countdown_dataset.py --multi_solution --num_multi 2000 --solutions_per_problem 4 --output_dir data/multi_solution
# Then use the orchestrator's C7 section or call:
#   from evaluation.permutation_robustness import evaluate_permutation_robustness
```

### Optional: C8 — Cross-Domain Transfer
```bash
python scripts/generate_transfer_datasets.py --domains algebra logic graph --num_samples 2000 --output_dir data/transfer
# Then extract trajectories and evaluate:
#   from evaluation.cross_domain_transfer import evaluate_cross_domain_transfer, save_transfer_report
```

---

## 7. Estimated GPU Usage

| Phase | Duration | GPU Required |
|-------|----------|-------------|
| Phase A trajectories (if not cached) | 2–4 hr | Yes |
| VQ training (C1) | 10–15 min | Yes (but minimal — no parameter gradients, only forward pass) |
| Transition MLP (C3) | 2–5 min | Yes |
| Discrete rollout (C9) | 2 min | Yes |
| All analyses (C2, C4, C5, C6, C7, C8) | < 2 min | CPU only |
| **Total new V5 work** | **~20 min GPU** | |

---

## 8. 5-Gate Verdict Logic

The orchestrator produces a binary verdict on whether the LM's hidden states
support reusable discrete representations. All five gates must PASS:

| Gate | Component | Pass Criterion |
|------|-----------|---------------|
| **Non-collapse** | C5 | Gini < 0.5 AND perplexity > 0.3 × num_codes |
| **Predictable** | C3 | Top-1 transition accuracy > 2 × chance |
| **Low-entropy subset** | C4 | Fraction of deterministic states (entropy < 0.1 nats) > 0.2 |
| **Low position leakage** | C6 | Position predictability < 0.5 (on [0,1] scale) |
| **Coherent rollout** | C9 | Average cosine similarity > 0.5 at rollout depth 1 |

**Possible verdicts:**
- All 5 PASS → *"Reusable discrete states are supported."*
- 3–4 PASS → *"Partial evidence for reusable discrete states (X/5 gates pass)."*
- 0–2 PASS → *"Evidence AGAINST reusable discrete states (X/5 gates pass)."*

---

## 9. Risks

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| 1 | **Codebook collapse** — VQ codebook maps all states to a few codes, making C3/C4/C9 trivially uninformative | Medium | Dead-code revival in `_ema_update()` + Gini/perplexity monitoring (C5). If collapse persists, reduce `num_codes` or increase `commitment_cost`. |
| 2 | **Phase A trajectory quality** — V5 inherits whatever hidden states Phase A produces. Bad trajectories = garbage in, garbage out. | Medium | Validate trajectories exist and have non-zero hidden states before V5 training. The orchestrator checks for this. |
| 3 | **Transfer dataset mismatch** — Algebra/logic/graph generators produce simple problems; the LM may not activate relevant reasoning circuitry. | Low | Transfer is optional; arithmetic results (C1–C6, C9) are the primary deliverable. |
| 4 | **Probe A/B/C dependence** — Discrete rollout (C9) depends on probes trained in Phase A. If probes are weak, rollout metrics are noisy. | Low | Probe accuracy is reported alongside rollout metrics. The orchestrator guards `if probe is not None`. |
| 5 | **Position leakage false positive** — Short trajectories (< 5 steps) have few distinct positions, making logistic regression unstable. | Low | The permutation null (shuffled labels) provides an empirical baseline. If both real and null are near chance, the gate still passes. |

---

## 10. Remaining Blockers

| Blocker | Status | Action |
|---------|--------|--------|
| Phase A trajectory files (`reports/trajectories/{train,val,test}.pt`) | **Not in repo** — must be generated on Kaggle | Run `run_phase_a.py` or `run_phase_a_orchestrator.py` before V5 |
| HuggingFace model access (for frozen LM hidden states) | **Requires HF token** | Set `HF_TOKEN` env var on Kaggle; add to Secrets |

---

## 11. Test Suite Status

```
38 passed, 8 warnings in 16.02s
```

All 38 tests across 8 test files pass. The 8 warnings are `DeprecationWarning`
from `scipy.optimize` in sklearn's logistic regression (position leakage tests) —
harmless and upstream.

```bash
pytest tests/test_vq_state.py \
       tests/test_discrete_trajectory_dataset.py \
       tests/test_codebook_usage.py \
       tests/test_discrete_transition.py \
       tests/test_position_leakage.py \
       tests/test_discrete_rollout.py \
       tests/test_permutation_robustness.py \
       tests/test_cross_domain_transfer.py -v
```

---

## 12. What Was NOT Built (Explicit Exclusions)

Per the V5 specification, the following were explicitly excluded:

- **No MuZero / MCTS / GFlowNets / RLHF / PRMs / agents / search** — V5
  answers a pure analysis question, not an optimization one.
- **No encoder neural network** — the VQ quantizer operates directly on the
  frozen LM's hidden states; there is no learned encoder projection.
- **No project redesign** — all new code follows existing patterns
  (dataclasses for config, `torch.save`/`torch.load` for checkpoints, CSV
  reports, `argparse` CLI, etc.).
- **No additional dependencies** — only `torch`, `sklearn`, `matplotlib`,
  `pyyaml` (all already in the project).

---

*Generated: 2026-06-22*
