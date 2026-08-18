# Coherence Audit

## Status: NO COHERENCE EXPERIMENT EXISTS

Spec Part 5 requires rollout coherence evaluation over depths 1..8 producing `reports/coherence_depth.csv` (columns: depth, mse, cosine_similarity, token_accuracy) and `coherence_depth.png`. The stated project objective is literally: *"determine whether latent planning survives beyond depth 3."*

Verified absent:
- `MISSING: reports/coherence_depth.csv`
- `MISSING: reports/coherence_depth.png`

There is **no coherence curve, no horizon measurement, no decay rate** anywhere in the repo or the extracted results. The project's defining question — the effective planning horizon — has **no answer and no supporting artifact.**

## Why the eventual coherence metric will be misleading (pre-registered objections)

Even when this is built, the metrics as specified are confounded:

1. **Cosine is misleading under anisotropy.** Random unrelated hidden states already cosine at **0.664** (measured floor). A rollout that has totally diverged from the truth can still report cosine ≈ 0.7–0.9. **Cosine similarity will overstate coherence at every depth.** The CSV must report mean-centered cosine and an anisotropy floor line, or it is uninterpretable.

2. **MSE is scale-confounded.** L2 norms of state vectors vary ±43% (within-state L2 28.8 ± 12.5). Raw MSE mixes direction error with magnitude drift. Per-trajectory normalized error is required.

3. **token_accuracy depends on the (missing) diagnostic decoder**, which itself is untrained and unvalidated. A decoder that predicts the majority next-token can inflate token_accuracy independent of whether the latent rollout is coherent.

4. **No statistical treatment is specified.** The spec asks for a point estimate per depth. Without per-depth bootstrap CIs and a significance test for the depth-3 drop, "survives beyond depth 3" is a vibe, not a result. With the underpowered state data (max cluster size 3, ~200 within-state pairs), CIs will be wide.

## Answers to required questions

- **Are probe drops statistically significant?** No probes, no drops, no tests → cannot be significant.
- **Are horizons real?** No horizon was measured.
- **Are confidence intervals overlapping?** No CIs were computed for any coherence metric.
- **Is cosine misleading?** Yes, demonstrably — the 0.664 anisotropy floor means cosine cannot distinguish "coherent" from "random" without correction.

## Verdict

**Coherence is unmeasured.** The single most important deliverable of the project does not exist. When built, the specified metrics (raw cosine, raw MSE, decoder accuracy) are exactly the ones most vulnerable to anisotropy and decoder artifacts, so the design as written is predisposed to **overstate** the planning horizon. Coherence claims are currently **0% supported**.
