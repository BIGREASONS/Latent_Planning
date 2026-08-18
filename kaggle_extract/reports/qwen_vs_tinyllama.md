# V5.3 · TinyLlama vs Qwen2.5-1.5B — Comparison

Scale-matched replication (identical Countdown problems, 3003 train / 1507 val
transitions, K=32, last layer, seed 0). Pipeline unchanged between runs.

## Headline comparison (requested metrics)

| Metric | TinyLlama-1.1B | Qwen2.5-1.5B | Replicates? |
| --- | ---: | ---: | :---: |
| **MLP(z,a) top1** — E. MLP(z,op) | 0.558 | 0.464 | ✓ (Qwen weaker) |
| **Action-aware bigram** — C. (z,op) | 0.553 | 0.464 | ✓ (Qwen weaker) |
| **MLP(z,a) − action-bigram** (E − C) | **+0.005** | **+0.000** | ✓ MLP never beats lookup |
| **Entropy** H(z'\|z,op) [nats] | 0.993 | 1.341 | ✓ (Qwen *more* stochastic) |
| **Deterministic fraction** det(z,op) | 0.110 | **0.006** | ✓ both ≪ 0.20; Qwen lower |
| **Position leakage** (fixed-init) | 0.790 | 0.779 | ✓ near-identical, ≫ floor 0.25 |
| det-frac(z) action-blind | 0.000 | 0.000 | ✓ |
| state matters (E2 − F) | +0.217 | +0.232 | ✓ codes non-vacuous |
| action helps lookup (C − B) | +0.290 | +0.149 | ✓ same sign, smaller |

## Per-model A–F (held-out val top1)

| Model | TinyLlama | Qwen |
| --- | ---: | ---: |
| A. Majority | 0.116 | 0.106 |
| B. Bigram(z) | 0.263 | 0.316 |
| C. Action-bigram(z,op) | 0.553 | 0.464 |
| D. MLP(z) | 0.266 | 0.317 |
| E. MLP(z,op) | 0.558 | 0.464 |
| E2. MLP(z,op,operands) | 0.622 | 0.509 |
| F. MLP(op,operands) control | 0.405 | 0.277 |

## Reading

Every signature that defined the V5.2 conclusion reproduces on Qwen:

- **`MLP(z,a) ≈ action-aware bigram`** — the load-bearing finding — reproduces
  almost exactly (E−C: +0.005 → +0.000; scrubbed −0.005). No learned dynamics
  beyond a first-order action-conditioned count, in either model.
- **Deterministic fraction ≈ 0** — reproduces and is *stronger* on Qwen
  (0.110 → 0.006). No reusable low-entropy planning states emerge.
- **Position leakage** — reproduces (0.79 in both): discrete codes track
  trajectory stage, far above the majority/null floor, in both models.

Where the two models differ, **Qwen is the weaker planner-state candidate**, not
the stronger one: lower action-conditioned accuracy, higher conditional entropy,
near-zero determinism. A 36% larger, substantially stronger, more recent model
does **not** convert Countdown hidden states into crisper, reusable discrete
dynamics at the final layer.

## Experiment B — layer sweep (middle vs final)

Identical sweep config in both runs (`cap_train=1500`, full val, K=32, VQ 50 ep,
transition 30 ep, seed 0). TinyLlama swept layers 12/22 (of 22); Qwen 14/28 (of 28).

| | TinyLlama L12 (mid) | TinyLlama L22 (final) | Qwen L14 (mid) | Qwen L28 (final) |
| --- | ---: | ---: | ---: | ---: |
| MLP(z,op) top1 | 0.361 | 0.542 | 0.233 | 0.434 |
| E − C | −0.006 | −0.002 | +0.007 | −0.007 |
| H(z'\|z,op) nats | 1.69 | 1.14 | 2.12 | 1.50 |
| **det(z,op)** | **0.000** | **0.000** | **0.000** | **0.000** |
| position leakage | 0.681 | 0.778 | 0.604 | 0.799 |
| best layer / reaches ceiling | final (22) / **No** | | final (28) / **No** | |

**Same pattern in both models:** the middle layer is *weaker* than the final
(lower MLP, higher entropy); det(z,op) = 0 at every depth; MLP ≈ action-bigram
(E−C ≈ 0) everywhere; the best layer is the final one and it does **not** reach
the reusable-state regime. TinyLlama's conclusion *"middle layers were not
stronger than the final layer"* replicates on Qwen.

## Calibration — each model on its OWN floor/ceiling

Closing the "carried-over calibration" nit: a Qwen-matched synthetic-FSM control
(1536-d vs TinyLlama's 2048-d) gives Qwen its own noise floor and reusable-state
ceiling. Final-layer Countdown placement (% of the way from floor to ceiling):

| Axis | TinyLlama (2048-d anchors) | Qwen (1536-d anchors) |
| --- | ---: | ---: |
| Det-frac(z,a) | 19% | **1%** |
| MLP(z,a) acc | 47% | **49%** |
| H(z'\|z,a) | 68% | **68%** |
| control AMI (codes vs true states) | 0.935 | 0.815 |

Both models: **above the noise floor, well short of the FSM ceiling**, with the
same axis-disagreement (strict det-frac low, soft axes ~half). Qwen is
indistinguishable on the soft axes and *lower* on strict determinism — the larger
model is, if anything, slightly further from reusable states. The calibration
carries over; the conclusion does not change. See `qwen_positive_control.md`.

_All requested rows are now populated; Experiment A (final layer), Experiment B
(middle vs final), and the Qwen-matched calibration are complete._
