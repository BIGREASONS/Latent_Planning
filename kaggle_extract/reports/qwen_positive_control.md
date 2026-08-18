# V5.3 · Qwen Positive Control — Methodology Calibration (Qwen-matched anchors)

> **Why this exists.** Experiments A and B compared Qwen's Countdown numbers
> qualitatively against the *TinyLlama* floor/ceiling. A reviewer could object:
> *"you carried the TinyLlama calibration over to Qwen."* This closes that nit by
> rebuilding the noise floor and reusable-state ceiling at **Qwen's** hidden
> dimensionality (1536-d), then placing Qwen's observed Countdown structure
> between them. It is a calibration of the *pipeline*, not a new LM experiment —
> the floor/ceiling come from a synthetic FSM, not from any model.

## Setup

Identical to V5.2 Experiment 2, only `hidden_dim` changed 2048 → **1536** to match
Qwen. Deterministic FSM (32 states, 3 actions), each state rendered as
`proto[s] + noise(0.3)`. **Ceiling** = structured embeddings (genuine reusable
`(s,a)→s'`); **floor** = same walks, structureless embeddings. K=32, VQ 50 ep,
transition 30 ep, seed 0. Qwen's observed Countdown (final layer) is read from
`reports_qwen/action_conditioned.json` + `reports_qwen/V5_1_FINAL.json`.

## Methodology validated for Qwen's dimensionality

The pipeline recovers known structure when it exists and reads ~noise when it
does not — it is not biased toward either answer at 1536-d:

| | Noise floor | **Countdown (Qwen, final)** | FSM ceiling | Oracle (true states) |
| --- | ---: | ---: | ---: | ---: |
| AMI(codes, true states) | 0.001 | — | **0.815** | 1.000 |
| MLP(z, a) top1 | 0.178 | 0.464 | 0.765 | — |
| Action bigram (z, a) | 0.174 | 0.464 | 0.765 | — |
| H(z'\|z, a) nats | 2.526 | 1.341 | 0.782 | 0.000 |
| **Det-frac (z, a)** | **0.000** | **0.006** | **0.586** | **1.000** |
| Position leakage | 0.238 | 0.779 | 0.236 | — |

AMI 0.815 (≫ 0; floor 0.001) confirms the VQ codes recover the ground-truth
states on the ceiling; det-frac jumps 0.000 → 0.586 and entropy falls 2.53 → 0.78
nats. The methodology is sound at Qwen's dimensionality.

## Where Qwen's Countdown lands (Qwen-matched anchors)

Placed on each floor→ceiling axis:

| Axis | floor → ceiling | Qwen Countdown | % of the way to ceiling | TinyLlama (its own anchors) |
| --- | ---: | ---: | ---: | ---: |
| **Det-frac(z,a)** | 0.000 → 0.586 | 0.006 | **1%** | 19% |
| MLP(z,a) acc | 0.178 → 0.765 | 0.464 | **49%** | 47% |
| H(z'\|z,a) | 2.526 → 0.782 | 1.341 | **68%** (down) | 68% |

## Calibrated conclusion (unchanged from TinyLlama, now on Qwen's own anchors)

Qwen's action-conditioned structure is **real and clearly above the noise floor**
(every metric beats the floor: MLP 0.464 vs 0.178; H 1.341 vs 2.526 nats) but
falls **well short of the reusable-state ceiling** (MLP 0.765, H 0.782, det 0.586).
The axes disagree exactly as for TinyLlama: the strict det-frac bar puts Countdown
~1% of the way up, the soft accuracy/entropy axes ~49–68%. The honest reading is
**graded, partial action-conditioned dependence — not reusable planning states.**

Relative to TinyLlama on its own anchors, Qwen is **indistinguishable on the soft
axes** (MLP 49% vs 47%, H 68% vs 68%) and **lower on the strict determinism axis**
(1% vs 19%) — i.e. the larger model is, if anything, slightly *further* from the
reusable-state ceiling. The calibration carries over; the conclusion does not change.

_Note: the auto-generated `reports_qwen/positive_control.md` reuses the V5.2 report
template, whose title prose hardcodes TinyLlama's example "0.99 / 0.11" — the data
rows and percentages in that file are Qwen's; this curated report states Qwen's
actual observed numbers (1.341 / 0.006)._

_Caveats: clean well-separated FSM = an upper-bound ceiling; a harder control
would lower it. Qwen's position leakage (0.78) shows its codes remain far more
position-bound than the control's (~chance 0.24). Single seed._

_Artifacts: `reports_qwen/positive_control.{json,md}`,
`reports_qwen/layer_sweep.{json,md,png}` (now calibrated)._
