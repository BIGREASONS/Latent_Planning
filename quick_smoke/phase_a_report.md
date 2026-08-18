# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Hidden layer probed:** -1
- **Hidden dim:** 896
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
| 1 | 0.587 | 0.200 | 0.300 | 0.723 | 0.792 | 83.7861 |
| 2 | 0.617 | 0.118 | 0.471 | 0.712 | 0.788 | 76.7216 |
| 3 | 0.625 | 0.500 | 0.500 | 0.679 | 0.814 | 79.1445 |
| 4 | 0.606 | nan | nan | 0.705 | 0.885 | 92.1147 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **3**; rollout state probe accuracy stays >= 0.50 through depth **4**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~4 step(s)**.

Single-step transition validation MSE: 36.0618.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._
