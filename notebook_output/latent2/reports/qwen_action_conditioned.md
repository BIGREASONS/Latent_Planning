# V5.3 · Experiment A — Action-Conditioned Validation (Qwen2.5-1.5B)

> **Replication question:** Are the V5.2 action-conditioned conclusions specific
> to TinyLlama, or do they hold on a stronger, more modern model?
> This runs the **exact** V5.2 Experiment-1 pipeline
> (`scripts/run_action_conditioned.py` → `evaluation/action_conditioned.py`)
> with the frozen teacher swapped from
> `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` to `Qwen/Qwen2.5-1.5B`.
> No metrics, baselines, or thresholds were changed.

## Setup & adjustments (documented)

- **Model:** `Qwen/Qwen2.5-1.5B` (28 layers, hidden 1536, fp16), frozen. Last
  layer (`-1`), exactly as V5.2.
- **Data:** identical Countdown splits. The hidden-state extraction, VQ
  (K=32, data-dependent init, EMA), INLP position-scrub, and A–F evaluation are
  the unchanged V5.2 code paths.
- **One adjustment — train scale-matching.** The cached TinyLlama trajectories
  were extracted with `cap=1000` train problems (1000 trajectories / **3003**
  transitions). The fresh Qwen extraction defaulted to all 5000. To isolate the
  *model* as the only variable, Qwen train was capped to the **same first 1000
  problems**; this reproduces TinyLlama's transition count to the unit
  (**3003 train / 1507 val** transitions), confirming the two runs see the
  identical Countdown problems. The full 5000-problem Qwen extraction is retained
  (`reports_qwen/trajectories/train_full.pt`) for an optional data-scale
  sensitivity check.
- **Codebook health (no collapse):** Qwen fixed-init VQ uses **32/32** codes
  (Gini 0.356, perplexity 25.8); scrubbed **32/32** (Gini 0.271). Comparable to
  TinyLlama — the comparison is not confounded by collapse.

## Results — Fixed-Init codebook (primary), held-out val

| Model | Conditioning | TinyLlama top1 | **Qwen top1** | Qwen pred-entropy |
| --- | --- | ---: | ---: | ---: |
| A. Majority | — | 0.116 | **0.106** | — |
| B. Bigram | z | 0.263 | **0.316** | — |
| C. Action bigram | z, op | 0.553 | **0.464** | — |
| D. MLP(z) | z | 0.266 | **0.317** | 1.896 |
| E. MLP(z, op) | z, op | 0.558 | **0.464** | 1.493 |
| E2. MLP(z, op, operands) | z, op, operands | 0.622 | **0.509** | 1.380 |
| F. MLP(op, operands) — control | op, operands | 0.405 | **0.277** | 2.006 |

### Transition structure (train, frequency-weighted)

| Conditioning | TinyLlama H (nats) | **Qwen H** | TinyLlama det-frac | **Qwen det-frac** |
| --- | ---: | ---: | ---: | ---: |
| z | 1.977 | **1.820** | 0.000 | **0.000** |
| z, op | 0.993 | **1.341** | 0.110 | **0.006** |

### Decisive contrasts (Qwen)

| Contrast | Meaning | TinyLlama | **Qwen** |
| --- | --- | ---: | ---: |
| C − B | does the action help the lookup? | +0.290 | **+0.149** |
| E − D | does the action help the MLP? | +0.292 | **+0.148** |
| **E − C** | **does MLP(z,a) beat the action bigram?** | **+0.005** | **+0.000** |
| E2 − E | do operands add information? | +0.064 | **+0.044** |
| E2 − F | does the state matter (not just the action)? | +0.217 | **+0.232** |
| det-frac gain (z→z,op) | reusable dynamics from the action | +0.110 | **+0.006** |

### Robustness — Scrubbed codebook (Qwen)

After INLP removal of the linearly-decodable position subspace (300/1536 dims,
position decodability 0.905 → 0.528), the action-conditioned MLP again **fails to
beat** the action bigram: **E − C = −0.005**, det-frac(z,op) = **0.000**. The
negative result is not an artifact of position leakage.

## Interpretation (skeptical)

**The two decisive V5.2 signatures replicate — and on Qwen they are if anything
*cleaner*:**

1. **MLP(z,a) ≈ action-aware bigram.** TinyLlama E−C = +0.005; Qwen E−C =
   **+0.000** (scrubbed −0.005). The learned next-code model captures **nothing**
   beyond an action-conditioned count lookup, in both models. This was the
   central V5.2 finding ("no learned dynamics beyond first order"). It holds.

2. **No reusable / deterministic planning states.** The near-deterministic
   transition mass is det-frac(z,op) = **0.006** on Qwen — below even the 0.05
   "partial" bar, and lower than TinyLlama's already-sub-threshold 0.110.
   H(z'|z,op) = **1.341 nats** (~3.8 effective successors) is *higher* (more
   stochastic) than TinyLlama's 0.993. The stronger model produces **less**
   deterministic action-conditioned structure, not more.

**The shared first-order effect is present in both.** Conditioning on the action
raises predictability (C−B, E−D both positive) and the state is not vacuous
(E2−F = +0.232 > 0): codes carry information beyond the action. But on Qwen the
action effect is *smaller* (C−B +0.149 vs +0.290), and it composes into weaker,
more stochastic transitions.

**Verdict (Experiment A):** the pipeline's own three-way logic labels Qwen
"partial" — but for the same reason it did for TinyLlama: the first-order action
effect plus an entropy drop, **not** reusable states. Read against the
reusable-state criteria (MLP > bigram; a deterministic low-entropy subset), Qwen
is a clean **negative**, matching TinyLlama. The bigger, more modern model does
not rescue the planning-state hypothesis at the final layer.

_Caveats: single seed (0), last layer only, op-type action (DIV absent). The
final-layer-only scope is exactly what Experiment B (layer sweep) would address._

_Artifacts: `reports_qwen/action_conditioned.{md,json}`,
`reports_qwen/V5_1_FINAL.json`, `reports_qwen/discrete_fixed_init/`,
`reports_qwen/discrete_scrubbed/`._
