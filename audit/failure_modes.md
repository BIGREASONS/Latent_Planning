# Phase 10 — Catastrophic Failure Modes (Top 20)

Severity × Probability. "Prob" = chance this is currently affecting results/usage. Each has a code anchor.

| # | Failure | Sev | Prob | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | **Teacher isn't planning** — base LM reads provided solutions; no planner exists to recover | Fatal | Certain | `run_phase_a.py:301` base model; no `load_checkpoint`/`train_teacher` use | Re-frame as representation study, or fine-tune a real teacher and extract its *own* generations |
| 2 | **Synthetic data isn't Countdown / isn't reasoning** — random forward walk, distractors, targets like 5003, no DIV | Fatal | Certain | `generate_countdown_dataset.py:11-57`; `data/train.jsonl` | Use real Countdown with search-based solver; include all 4 ops; bounded targets |
| 3 | **Conclusions drawn from a 52/63-sample smoke run** | Critical | Certain (as shipped) | `reports/probe_report.md`; coherence `n_samples=20` | Run at the documented 5k/500 scale; add CIs/seeds before any claim |
| 4 | **Transition model loses to identity baseline by depth 3** (`dynamics_gain<1`) | Critical | Certain (on current data) | `coherence_action_depth.csv` depth3=0.960 | This *is* a result — report it honestly; don't bury under "encoded" |
| 5 | **PCA/scaler fit on full set before split (C.3)** — train-on-test | Critical | Certain when C.3 run | `run_phase_c3_geometry.py:147-157` | Fit on train slice only (copy C.4a) |
| 6 | **Phase D projection trained on train+test, scored on same** | Critical | Certain when D run | `run_phase_d.py:220-257` | Hold out a disjoint eval set the projection never sees |
| 7 | **"Encoded" verdict fires on majority-class noise (F1=0)** | High | Certain | `probes.py:281`; Probe A F1=0.000 "encoded" | Require AUC *and* F1 *and* permutation test |
| 8 | **Probe B/D measure token position, sold as reasoning** | High | Certain | `probes.py:57,66`; Phase A report counts them | Partial out depth; add position-only baseline |
| 9 | **Deterministic-prefix identity inflates all Phase C similarity/retrieval** | High | Certain | causal LM + `intrinsic_noise.py:209` mask ignores prefixes | Mask shared prefixes; or use distinct-text same-state pairs only |
| 10 | **`run_phase_d.py` crashes on CPU/non-CUDA** (unconditional `.cuda()`) | High | Certain off-GPU | `run_phase_d.py:81,90-92,172,181` | Use `device` variable |
| 11 | **Hardcoded personal path** in `append_walkthrough*.py` | High | Certain off-author-machine | `:19`/`:18` `C:\Users\singh\.gemini\...` | Delete files |
| 12 | **No result is version-controlled / reproducible** — reports, data, ckpts gitignored; generator unseeded | High | Certain | `.gitignore`; `generate_countdown_dataset.py` (no seed) | Seed the generator; commit artifacts or a manifest+hashes |
| 13 | **`O(M²)` Python loops make Phase C intractable at scale** | High | Likely at 5k+ | `intrinsic_noise.py:209-216,229-233` | Vectorize masks |
| 14 | **Rolled-out states scaled by teacher-fit StandardScaler (distribution shift)** | Med | Likely | `coherence.py:92` + probe pipeline | Renormalize/clip; refit-free calibration |
| 15 | **bitsandbytes/peft imported at module top gate Phase A** | Med | Possible | `model_loader.py:9-16` | Lazy import |
| 16 | **DIV op never in data → Probe C 4-class is really 3-class; chance miscomputed** | Med | Certain | generator omits `/`; `_chance_accuracy` over observed only | Generate DIV; or document 3-class |
| 17 | **Master report points to non-existent `coherence_depth.csv`; stale legacy CSV present** | Med | Certain | `run_phase_a.py:223`; `reports/coherence_depth.csv` | Fix filename; clear stale artifacts |
| 18 | **Silent NaN propagation through `_mean`/`dynamics_gain`/`_depth_where`** hides "no data" as "no signal" | Med | Possible | `coherence.py:134,161`; `run_phase_a.py:147` | Assert finite; distinguish empty depth |
| 19 | **Phase B re-fits probes independently of A; no checksum linking trajectories↔results** | Med | Possible | `run_phase_b.py:332` | Serialize fitted probes from A; pin trajectory hashes |
| 20 | **One-shot mutation scripts committed** (`inject_sys_path*.py`, `patch_run_phase_batch.py`) re-edit source if re-run | Med | Possible | those files | Delete from repo |

## Single most likely way the whole project is "wrong"
Failures **1 + 2 + 8** compound: a non-planning model reads non-reasoning data, and the probes that look positive are decoding **token position and literal operand tokens already present in the prompt** — not any latent computation. Everything downstream (coherence, oracle, canonicalization) inherits this, so a "promising" verdict would be an artifact, while the honest internal signals (identity baseline beating the model, F1=0 on Probe A) already point the other way.
