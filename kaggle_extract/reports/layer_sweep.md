# V5.2 · Experiment 3 — Heavy Layer Validation (Layer 12 vs Layer 22)

> **Curated final report.** Headline = the heavy run (serious settings) on the two decisive
> depths. A breadth screen across six depths (smoke settings) is folded in below as supporting
> evidence. Raw machine artifacts: `reports/layer_sweep.json` / `.png` (heavy, 12 vs 22) and
> `reports/layer_sweep_smoke6.json` / `.png` (breadth, 6 depths). Note: re-running
> `scripts/run_layer_sweep.py` overwrites `layer_sweep.{md,json,png}` with a generic templated
> report — keep this curated copy if you hand-edit it.


## Why this experiment

A reviewer raised the standard objection: every V5.2 result probes TinyLlama's **final** layer
(index 22 / `-1`), yet planning-relevant structure often lives in **middle** layers. If so, our
"no reusable discrete dynamics" conclusion could be an artifact of *where* we probe rather than a
property of the model.

We test the strongest form of that objection directly. We re-extract hidden states at **layer 12**
— the strongest mid-stack candidate — and **layer 22** — our baseline — and run the *identical*
Experiment-1 pipeline at each, at the **same serious settings as every other V5.2 experiment** (no
smoke shortcuts). This is a targeted two-layer validation, not a full-stack sweep; the breadth
screen below covers the rest of the stack at smoke fidelity.


## Setup

- **Model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (22 layers). Index into
  `hidden_states`; the final layer = 22.
- **Data:** Countdown — **1500 train / 497 val** problems (val de-duplicated against train by
  `(target, sorted numbers)`).
- **Settings (identical to Exp1):** VQ **K=32** (data-dependent init, raw states), **50 VQ
  epochs**, **30 transition epochs**, seed 0. Per layer: VQ → action-conditioned models A–F on
  held-out val + transition structure on train + codebook usage + position leakage.
- **Anchors:** Experiment-2 noise floor and positive-control ceiling (same K, same pipeline) bound
  each action-conditioned axis.


## Per-layer metrics (heavy run)

| Layer | Active | Perplex. | Pos-leak | Bigram(z) | MLP(z) | Act-bigram | MLP(z,op) | H(z'\|z) | H(z'\|z,op) | Det(z,op) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 (middle) | 32 | 29.0 | 0.681 | 0.215 | 0.211 | 0.367 | 0.361 | 2.330 | 1.689 | 0.000 |
| 22 (final)  | 32 | 27.4 | 0.778 | 0.256 | 0.255 | 0.544 | 0.542 | 2.151 | 1.137 | 0.000 |

**Position of each layer between the Exp2 noise floor and the reusable-state ceiling** (0% = at
noise floor, 100% = at positive-control ceiling):

| Axis | Layer 12 | Layer 22 |
| --- | ---: | ---: |
| Det(z,op) — strict reusable-state metric | **0%** (0.000) | **0%** (0.000) |
| MLP(z,op) — soft predictability | **−2%** (0.361, *at the floor*) | **43%** (0.542) |
| H(z'\|z,op) — soft determinism (lower = more structure) | **30%** (1.689) | **60%** (1.137) |

_Det(z,op) = share of transition mass from near-deterministic `(z,op)` conditionings (H<0.5 nats,
seen ≥10×). It is a strict threshold metric; on this run's codebook neither layer crosses it, so
the comparative signal lives on the soft axes (MLP, H) — both of which favour the **final** layer._


## Anchors (Experiment 2)

| | Det(z,op) | H(z'\|z,op) | MLP(z,op) |
| --- | ---: | ---: | ---: |
| Noise floor | 0.000 | 2.259 | 0.370 |
| Positive-control ceiling | 0.593 | 0.386 | 0.772 |
| Cached layer −1 (Exp1, full data) | 0.110 | 0.993 | 0.558 |

---
## Verdict — Case 1: the middle-layer objection is refuted

**Layer 12 is not richer than layer 22 — it is poorer, and neither reaches the reusable-state
regime.**

- On the strict **Det(z,op)** axis both layers sit at **0.000 (0% to ceiling)**: neither produces
  crisp, reusable discrete dynamics.
- On both *soft* axes the **final** layer leads. Layer 12's **MLP(z,op) = 0.361 sits *at the noise
  floor* (0.370)** — no action-conditioned predictability above chance — while layer 22 retains the
  partial structure seen in Exp1 (MLP 0.542 ≈ cached 0.558; 43% to ceiling, vs 30%→60% on the
  entropy axis).

So the strongest mid-stack candidate is, if anything, *closer to noise* than the layer we already
report. There is **no middle-layer peak** (no "Case 2"). The layer-choice criticism is resolved:
our conclusion does not depend on probing the final layer, and probing a middle layer would have
made the structure *weaker*, not stronger.


## Breadth screen (6 depths, smoke settings) — supporting

A wider but lighter screen across `4, 8, 12, 16, 20, 22` (60 train / 28 val, 2 VQ / 1 transition
epoch) returns **Det(z,op) = 0.000 at every depth** with no mid-stack peak. At smoke fidelity only
the strict-axis result is trustworthy; its sole takeaway is the absence of any layer where reusable
dynamics emerge. The heavy run above is what licenses the *comparative* (12 vs 22) claim.

| Layer | 4 | 8 | 12 | 16 | 20 | 22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Det(z,op) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

_Source: `reports/layer_sweep_smoke6.json`._

---
_Caveats: single model / task / seed; op-type action; two depths at heavy fidelity (rest at smoke).
Det(z,op) is a strict threshold metric — see Exp2 for why the soft entropy/MLP axes can read higher.
The heavy run re-trains VQ on this data, so layer-22 Det reads 0.000 here vs the 0.110 cached in
Exp1's full-data codebook; the comparison that matters (12 vs 22, same codebook regime) is unaffected._

_Artifacts: `reports/layer_sweep.json`, `reports/layer_sweep.png` (heavy);
`reports/layer_sweep_smoke6.json`, `reports/layer_sweep_smoke6.png` (breadth)._
