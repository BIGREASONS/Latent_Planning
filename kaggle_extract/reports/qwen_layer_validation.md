# V5.3 · Experiment B — Heavy Layer Validation (Qwen2.5-1.5B)

> **Question:** Experiment A probed only the final layer. Does a *middle*-layer
> peak rescue the planning-state hypothesis in a stronger model — i.e. is
> mid-stack discrete structure more deterministic / more predictable than the
> final layer?

## Setup

- **Model:** `Qwen/Qwen2.5-1.5B` (28 layers). Swept **layer 14** (middle, 50%
  depth — the closest equivalent to TinyLlama's layer 12 of 22) and **layer 28**
  (final, = TinyLlama's layer 22). Two layers only, per the Experiment-B spec.
- **Pipeline:** the identical V5.2 layer-sweep (`scripts/run_layer_sweep.py`) —
  fresh hidden-state extraction at each depth → VQ K=32 (50 epochs, data-dependent
  init) → action-conditioned models A–F (30 epochs) + codebook usage + position
  leakage. `cap_train=1500`, full val split (497 after de-dup). Seed 0.
- **Calibration:** a Qwen-matched positive control (1536-d noise floor + FSM
  ceiling, `reports_qwen/positive_control.json`) was subsequently generated, so the
  layer rows are placed on the floor→ceiling axes (`→ceiling` columns below).

## Per-layer metrics

| Layer | Active | Perplx | Pos-leak | Act-bigram | **MLP(z,op)** | **H(z'\|z,op)** | **Det(z,op)** | **E−C** | →ceiling det / MLP / H |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 (middle) | 32 | 29.2 | 0.604 | 0.226 | **0.233** | **2.12** | **0.000** | **+0.007** | 0% / +9% / +23% |
| 28 (final) | 32 | 28.3 | 0.799 | 0.441 | **0.434** | **1.50** | **0.000** | **−0.007** | 0% / +44% / +59% |

_(Final-layer row reproduces Experiment A's final-layer result up to the
cap_train 1500-vs-1000 / VQ-reinit difference: MLP(z,op) 0.434 vs 0.464, E−C
−0.007 vs +0.000, det ≈ 0 vs 0.006 — internally consistent.)_

## Verdict — **No middle-layer advantage. Conclusion #6 replicates.**

On every axis the middle layer is the **weaker**, not the stronger, planning-state
candidate:

- **Determinism:** det(z,op) = **0.000 at both depths.** No near-deterministic
  reusable subset appears anywhere in the stack.
- **Predictability:** MLP(z,op) is **0.233 (middle) vs 0.434 (final)** — the
  middle layer is *worse* by −0.201.
- **Entropy:** H(z'|z,op) is **2.12 (middle) vs 1.50 (final)** nats — the middle
  layer is *more* stochastic (~8.3 vs ~4.5 effective successors).
- **MLP(z,a) ≈ action-bigram at both depths:** E−C = +0.007 (middle), −0.007
  (final). The learned model captures nothing beyond the action-conditioned count
  at *any* probed depth.

The final layer is the strongest discrete representation of the two, and even it
carries no reusable dynamics. On the Qwen-matched floor→ceiling axes, the middle
layer reaches only +9% (MLP) / +23% (entropy) of the way to the FSM ceiling and
the final layer +44% / +59% — **0% on the strict det-frac axis at both depths**,
and `reaches_ceiling = False`. The TinyLlama finding — *"middle layers were not
stronger than the final layer"* — holds on Qwen.

_Caveats: two layers only (14, 28), single seed, op-type action. Calibration uses
a clean synthetic FSM (upper-bound ceiling); see `qwen_positive_control.md`._

_Artifacts: `reports_qwen/layer_sweep.json`, `reports_qwen/layer_sweep.png`,
`scratch_logs/qwen_layer_sweep.log`._
