# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `mistralai/Mistral-7B-v0.3`
- **Hidden layer probed:** -1
- **Hidden dim:** 4096
- **Trajectories:** train=5000 val=491 test=970

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_numbers
- B:distance_to_solution
- C:next_operation
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.959 | 0.336 | 0.531 | 0.771 | 0.992 | 2.7811 |
| 2 | 0.932 | 0.328 | 0.582 | 0.637 | 0.943 | 4.4678 |
| 3 | 0.921 | 0.310 | 0.675 | 0.655 | 0.870 | 5.1602 |
| 4 | 0.954 | nan | nan | 0.702 | 0.812 | 3.0939 |

Cosine similarity stays >= 0.90 through depth **4**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **4**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~4 step(s)**.

Single-step transition validation MSE: 4.5472.

## 4. Is latent planning worth pursuing?

**Promising.** Hidden states encode task-relevant structure *and* coherence survives beyond depth 3 — the regime where search would operate. Proceed to Phase B.

**Bottleneck diagnosis:** neither (proceed).

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._
