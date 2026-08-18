# Missing Controls

Ranked by importance. "Present?" = exists in repo or extracted results.

| Rank | Control | Why it matters | Present? |
|---|---|---|---|
| 1 | **Anisotropy floor / random-pair baseline** | Random hidden-state pairs cosine at 0.664. Without this line, between-state 0.69 looks meaningful when it is the floor. Every cosine claim is uninterpretable without it. | **No** (I had to compute it) |
| 2 | **Same-step / position control** | Same-step different-state cosine = 0.889. Distance-to-solution and retrieval are decodable from position alone. Mandatory to regress out step index. | **No** |
| 3 | **Identity transition baseline `h_{t+1}=h_t`** | With state-to-state cosine ~0.9, copying wins. Any transition result is meaningless without `gain = identity_mse/model_mse`. | **No** (no transition model at all) |
| 4 | **Shuffled-action transition** | Breaks action→next-state causality; if coherence unchanged, the model ignores actions. | **No** |
| 5 | **Mean-centering / whitening of hidden states** | Removes anisotropy before cosine/L2/retrieval. Changes between-state cosine from 0.66 to 0.03. | **No** |
| 6 | **Problem-disjoint splits (dedup + seed)** | 24 (target,numbers) problems leak train∩test; no seed. Probes/transition can memorize. | **No** |
| 7 | **Prefix-only extraction (exclude the step's own result token)** | Full `cot` is in the prompt; result tokens may be the indexed states → probes "see the future." | **No** |
| 8 | **Majority/random/exact-match probe baselines** | Next-op majority ≈ 33.7%; any probe must beat its trivial baseline with a CI. | **No** (no probes) |
| 9 | **Random-projection representation control** | Project hidden states to random subspace; if probe/retrieval barely drops, the "signal" is generic geometry, not learned content. | **No** |
| 10 | **Shuffled-label probe control** | Train probe on permuted labels; non-trivial accuracy ⇒ leakage/overfitting. | **No** |
| 11 | **Blind/zeroed-history extraction** | Extract with only the problem (no solution) in context to test genuine autoregressive planning vs teacher-forced readout. | **No** |
| 12 | **Oracle transition (later phase)** | To substantiate "Oracle can't save it," actually run it. | **No** (and forbidden this phase) |
| 13 | **Decoder majority-token baseline** | token_accuracy in coherence must beat predicting the most frequent next token. | **No** (no decoder) |
| 14 | **Per-trajectory normalized MSE** | Raw MSE confounds magnitude drift (L2 varies ±43%) with direction error. | **No** |
| 15 | **Bootstrap CIs + significance tests on every metric** | None reported anywhere. | **No** |

## Summary

**Zero of the 15 controls are present.** The two most load-bearing absences — anisotropy floor (#1) and same-step/position control (#2) — are exactly the ones that, when I added them, collapsed most of the apparent within/between contrast and exposed 85.7% position contamination in retrieval. The identity baseline (#3) would gut any future transition claim. This is not a project with weak controls; it is a project with **no controls.**
