# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20

> ⚠️ **Smoke-test run** with a randomly-initialized model. The numbers below validate the *pipeline*, not the science. Re-run with the TinyLlama teacher for real conclusions.

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.359 | 0.400 | 0.150 | 0.715 | 0.758 | 4.7603 |
| 2 | 0.349 | 0.412 | 0.588 | 0.742 | 0.823 | 5.1674 |
| 3 | 0.352 | 0.167 | 0.667 | 0.751 | 0.833 | 6.0896 |
| 4 | 0.349 | nan | nan | 0.769 | 0.846 | 7.5153 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **4**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~4 step(s)**.

Single-step transition validation MSE: 2.8268.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._
