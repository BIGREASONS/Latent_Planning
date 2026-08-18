# V5 Report — Reusable Discrete States?

Research question: _do hidden-state trajectories of frozen language models admit a compact, predictive, reusable discrete state representation?_

- **Codebook size (K):** 32
- **Hidden dim:** 2048
- **Trajectories:** train=1000 test=1000

---

## 1. Codebook usage (collapse check)

- Active codes: **6/32** (used: 6, dead: 26)
- Collapse score (Gini): **0.854** (0 = uniform, 1 = fully collapsed)
- Perplexity: **5.3** (effective codes used, max 32)

Gate: PASS (Gini<0.9 AND active>=10% of K).

## 2. Transition predictability

- Top-1 next-code accuracy: **0.431** (chance = 0.031)
- Predictive entropy: **1.294 nats** (perplexity 3.61)

Gate: PASS (top-1 > max(2*chance, chance+0.02)).

## 3. Transition entropy (planning state vs bucket)

- Global H(Z_next | Z_current): **1.197 nats** (perplexity 3.31)
- Fraction of states with near-deterministic successor (H<0.5): **0.0%**

Gate: FAIL (>= 20% deterministic states).

## 4. Position leakage (codes as step indices?)

- Position predictability: **0.588**
  - chance: 0.251, permutation null: 0.090

Gate: FAIL (score <= max(null, chance) + 0.10). High leakage => codes are step indices, not reusable states.

## 5. Discrete rollout coherence

| depth | code match | cosine | op acc (Probe C) | state acc (Probe A) |
|---|---|---|---|---|
| 1 | 0.353 | 0.973 | 0.328 | 0.383 |
| 2 | 0.304 | 0.947 | 0.353 | 0.332 |
| 3 | 0.485 | 0.961 | 0.310 | 0.356 |
| 4 | 0.955 | 0.996 | nan | 0.383 |

Gate: PASS (depth-1 code match > max(2/K, 0.20)).


---

## Verdict

**Mixed evidence (3/5 gates pass).** See the failing gates above (low_entropy_subset, low_position_leakage) for the specific failure mode.

_Failure modes: high Gini => VQ collapse; low top-1 => unpredictable dynamics; high position leakage => codes are step indices; low deterministic-state fraction => every state is a compression bucket._
