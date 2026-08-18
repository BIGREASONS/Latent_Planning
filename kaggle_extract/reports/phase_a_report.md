# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20

> ⚠️ **Smoke-test run** with a randomly-initialized model. The numbers below validate the *pipeline*, not the science. Re-run with the TinyLlama teacher for real conclusions.

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_numbers
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 3 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.435 | 0.500 | 0.150 | 0.600 | 0.625 | 4.1354 |
| 2 | 0.448 | 0.412 | 0.588 | 0.463 | 0.475 | 4.3052 |
| 3 | 0.443 | 0.167 | 0.500 | 0.382 | 0.456 | 5.0114 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **1**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 2.1630.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._
