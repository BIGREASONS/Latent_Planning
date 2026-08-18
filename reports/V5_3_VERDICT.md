# V5.3 — Cross-Model Replication Verdict (Qwen2.5-1.5B)

**Status: Experiments A, B, and the Qwen positive-control calibration complete.**

## What was asked

Does the V5.2 conclusion replicate on a stronger, more modern model, or is it a
TinyLlama artifact? The V5.2 findings were:

1. Action-conditioned structure exists.
2. Structure is above the calibrated noise floor.
3. Structure is weaker than a known finite-state machine.
4. MLP(z,a) ≈ action-aware bigram.
5. No reusable planning states emerged.
6. Middle layers were not stronger than the final layer.

## Verdict → **Outcome 1: the negative result replicates.**

Both experiments used the unchanged V5.2 pipeline with only the frozen teacher
swapped (`TinyLlama-1.1B` → `Qwen2.5-1.5B`), scale-matched to the identical
Countdown problems (Exp A: 3003/1507 transitions; Exp B: cap_train=1500, full val).

| V5.2 finding | Qwen replication | Evidence |
| --- | :---: | --- |
| 1. Action-conditioned structure exists | ✓ | C−B +0.149, E−D +0.148 (action raises predictability) |
| 2. Above the noise floor | ✓ (calibrated) | Qwen 1536-d anchors: MLP 49%, H 68% of floor→ceiling; all metrics beat the floor |
| 3. Weaker than a known FSM | ✓ (calibrated) | short of the FSM ceiling on every axis; det 0.006 vs ceiling 0.586 (1% up) |
| **4. MLP(z,a) ≈ action-aware bigram** | **✓** | **E−C = +0.000 (final), −0.007 / +0.007 across layers** |
| **5. No reusable planning states** | **✓ (stronger)** | **det(z,op) = 0.006 (Exp A), 0.000 at both swept layers** |
| **6. No middle-layer advantage** | **✓** | **middle weaker than final in both models; best = final, no ceiling** |

**All six replicate.** #2 and #3 are now calibrated against a **Qwen-matched
(1536-d) noise floor and FSM ceiling**, not carried over from TinyLlama: Qwen's
Countdown sits 1% (det-frac) / 49% (MLP) / 68% (entropy) of the way from floor to
ceiling — above noise, well short of reusable states — essentially identical to
TinyLlama on its own anchors (19% / 47% / 68%), and *lower* on the strict
determinism axis. The control itself recovers known FSM states (AMI 0.815),
validating the pipeline at Qwen's dimensionality. **No finding flips.**

### Why this is Outcome 1, not Outcome 2 or 3

- **Outcome 3 (hypothesis survives)** is rejected: no deterministic subset
  (det ≈ 0), MLP never beats the action bigram (E−C ≈ 0), no middle-layer peak.
- **Outcome 2 (scaling effect, still sub-deterministic)** is *not* supported:
  Qwen does not exceed TinyLlama. On every axis where the two differ, the larger
  model is the **weaker** planning-state candidate — lower action-conditioned
  accuracy (E 0.464 vs 0.558), higher conditional entropy (1.34 vs 0.99 nats),
  near-zero determinism (0.006 vs 0.110).
- **Outcome 1** is satisfied on its own terms: `MLP(z,a) ≈ action-aware bigram`,
  `deterministic fraction ≈ 0`, and `no middle-layer advantage` — all three.

## Bottom line for the paper

**The V5.2 negative is not a small-model artifact.** A 36%-larger, materially
stronger, more recent language model (Qwen2.5-1.5B) produces *no more* — and on
the determinism axis *less* — reusable discrete planning structure in its
Countdown hidden states than TinyLlama-1.1B. Under both models the learned
next-code dynamics collapse to a first-order action-conditioned count, no
deterministic reusable subset appears, and no probed depth changes this. This
cross-model replication **substantially strengthens** the manuscript's claim.

## Scope / honest limits

- Two models, one task (Countdown), one seed (0), op-type action (DIV absent).
- Two layers in Exp B (middle 14, final 28), not the full stack.
- Calibration is anchored on a clean, well-separated synthetic FSM (an *upper*
  bound on the ceiling); a harder control would lower it. The Qwen control uses
  hidden_dim 1536 / noise 0.3 / 32 states / 3 actions to match the Countdown
  setting; other control geometries were not swept.
- The retained full 5000-problem Qwen extraction
  (`reports_qwen/trajectories/train_full.pt`) allows a data-scale sensitivity
  check if desired; Experiment A was scale-matched to TinyLlama deliberately.

## Deliverables

- `reports/qwen_action_conditioned.md` — ✅ Experiment A.
- `reports/qwen_layer_validation.md` — ✅ Experiment B (calibrated).
- `reports/qwen_positive_control.md` — ✅ Qwen-matched floor/ceiling calibration.
- `reports/qwen_vs_tinyllama.md` — ✅ full comparison (A + B + calibration).
- `reports/V5_3_VERDICT.md` — ✅ this file.
