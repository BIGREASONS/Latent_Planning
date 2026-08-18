# V5.2 · Experiment 1 — Action-Conditioned Transition Test

> Reviewer criticism addressed: a planning state is defined by `(state, action) → next_state`, not `state → next_state`. V5.1 tested only the action-blind form. Here we condition on the symbolic Countdown action.


## Implementation

- **Representation:** Fixed-Init VQ codes (K=32, data-dependent init, raw last-layer states) — the clean, non-collapsed V5.1 codebook. Scrubbed codes reported as a robustness check.

- **Action:** op id (0=ADD, 1=SUB, 2=MUL; DIV absent from data) and the raw operands `[arg1, arg2]` (signed-log + standardized for the MLP).

- **Eval:** top-1 on the held-out **val** split (train 3003 transitions, val 1507); MLPs trained 30 epochs, seed 0. Entropy/determinism measured on train (frequency-weighted, deterministic = successor entropy < 0.5 nats over conditionings seen ≥ 10×).

- **Models A–F:** majority; state bigram; action bigram `(z,op)→mode`; MLP(z); MLP(z,op); MLP(z,op,operands); and an **action-only control** MLP(op,operands) that ignores `z_t`.


## Results — Fixed-Init codebook (primary)

### Predictive accuracy (held-out val)

| Model | Conditioning | Top1 | Pred. entropy (nats) |
| --- | --- | ---: | ---: |
| A. Majority | — | 0.116 | — |
| B. Bigram | z_t | 0.263 | — |
| C. Action bigram | z_t, op | 0.553 | — |
| D. MLP(z_t) | z_t | 0.266 | 2.049 |
| E. MLP(z_t, op) | z_t, op | 0.558 | 1.107 |
| E2. MLP(z_t, op, operands) | z_t, op, operands | 0.622 | 0.975 |
| F. MLP(op, operands) — control | op, operands | 0.405 | 1.471 |

### Transition structure (train)

| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |
| --- | ---: | ---: | ---: |
| z_t | 1.977 | 0.000 | 32 |
| z_t, op | 0.993 | 0.110 | 91 |

### Key contrasts

- **Does the action help the lookup?** C − B = +0.290 (0.553 vs 0.263).
- **Does the action help the MLP?** E − D = +0.292 (0.558 vs 0.266).
- **Does the MLP beat the action bigram?** E − C = +0.005.
- **Do operands add information?** E2 − E = +0.064 (E2 = 0.622).
- **Does the state matter, or only the action?** E2 − F = +0.217 (F action-only = 0.405).
- **Determinism gain from conditioning on the action:** det-frac 0.000 (z) → 0.110 (z,op); entropy 1.977 → 0.993 nats.


## Robustness — Scrubbed codebook

| Model | Conditioning | Top1 | Pred. entropy (nats) |
| --- | --- | ---: | ---: |
| A. Majority | — | 0.189 | — |
| B. Bigram | z_t | 0.218 | — |
| C. Action bigram | z_t, op | 0.434 | — |
| D. MLP(z_t) | z_t | 0.216 | 2.586 |
| E. MLP(z_t, op) | z_t, op | 0.439 | 1.595 |
| E2. MLP(z_t, op, operands) | z_t, op, operands | 0.561 | 1.329 |
| F. MLP(op, operands) — control | op, operands | 0.531 | 1.444 |

| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |
| --- | ---: | ---: | ---: |
| z_t | 2.494 | 0.000 | 32 |
| z_t, op | 1.456 | 0.023 | 92 |


---
## Reviewer-style interpretation

**1. Results.** See tables above. On the clean Fixed-Init codebook, conditioning on the symbolic action moves top-1 from 0.263 (state bigram) to 0.553 (action bigram) and 0.558 (action MLP); adding operands gives 0.622.

**2. Does MLP(z,a) materially beat the action-aware bigram?** **No** — E − C = +0.005 (threshold ±0.03). The MLP does not exceed the action-conditioned lookup: whatever the action contributes is already a first-order count effect, not learned structure.

**3. Does the planning-state hypothesis survive?**

> **Partially — and this revises the V5.1 wording.** The reviewer criticism was correct that the action-blind test understated the structure: conditioning on the action roughly *doubles* predictability (state→action bigram 0.263 → 0.553; E−D +0.292), *halves* the conditional entropy (1.977 → 0.993 nats), and turns a 0% deterministic mass into 0.110. So `(state, action)` carries real, non-trivial dynamics that `state` alone did not.

> **But it does not reach reusable planning states.** The structure is purely first-order — the MLP does not beat the action lookup (E−C +0.005) — and it is far from deterministic: only 0.110 of transition mass is near-deterministic, H(z'|z,op) ≈ 0.993 nats (~3 effective successors), and top-1 is 0.622 even with operands. The state is not vacuous (E2−F +0.217), but it composes with actions into *weak, stochastic* transitions, not crisp reusable ones.

> **Net:** V5.1's flat "no reusable structure" is too strong and should be revised to **"weak, first-order, sub-deterministic action-conditioned dynamics — not reusable planning states."** Whether even this weak effect exceeds what *any* action-conditioned model would yield on position-correlated codes cannot be judged without a calibrated positive control — which is exactly the purpose of Experiment 2.


_Caveats: single model/task/layer/seed; op-type action (DIV absent); operands make the next *number* computable, so any E2 gain may reflect numeric leakage rather than reusable state composition. Scope unchanged from V5.1._


_Artifacts: `reports/action_conditioned.json`._
