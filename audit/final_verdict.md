# Final Verdict

All scores 0–10. Uncharitable; evidence only.

## Scores

**Evidence FOR latent planning: 1 / 10**
Only a small residual within>between state signal after anisotropy correction (centered gap 0.70; raw gap CI [0.258,0.288]). No planning, transition, coherence, or probe artifact exists. Almost nothing supports planning.

**Evidence AGAINST latent planning: 7 / 10**
Project's own verdict: "highly history-dependent (Case B), true bottleneck." Retrieval matches step-position 85.7%; within-state advantage over position confound only +0.04; benchmark isn't a real planning task; assumptions A1/A5/A7/A8 falsified. Held below 9 only because the negative evidence is itself confounded/underpowered.

**Representation quality: 2 / 10**
Severe anisotropy (random-pair cosine 0.664), strong position dependence, history-laden (full solution in prompt), underpowered clustering (max cluster 3). Weak residual state signal only.

**Dynamics quality: 0 / 10**
No transition model exists. No MSE, no cosine, no identity-baseline gain. Unmeasurable.

**Evaluation reliability: 1 / 10**
No seeds, no CIs/SE/significance tests, missing generating code (irreproducible), no anisotropy/position/identity controls, key-dependent retrieval, unrecorded extraction layer, broken plotter.

**Risk of hidden confounds: 9 / 10**
Anisotropy + token position confirmed to drive the headline geometry; full-solution-in-prompt risks future-information leakage; train/test problem leakage (24 cases). Confounds are pervasive and demonstrated, not hypothetical. (High = bad.)

**Publication readiness: 0 / 10**
Core experiments unbuilt; sole experiment confounded and irreproducible; benchmark invalid. Not submittable.

## Final judgement

### Probably False

The corpus provides **no positive demonstration of latent planning**: the transition model, coherence-horizon curve, and probes do not exist, so the central research questions are unanswered by artifacts. The single completed experiment (intrinsic noise) concludes *against* the necessary precondition — a clean, Markovian representation — and even that conclusion is built on metrics I reproduced to be **confounded by anisotropy (random-pair cosine 0.664) and token position (same-step cosine 0.889; 85.7% same-step retrieval)**, computed on a self-admittedly **underpowered** sample (max cluster size 3, ~200 pairs, no CIs), using **code that is not in the repository** (irreproducible). The benchmark itself is **not a valid planning task** (no division, targets to 10^8, no operand-use-once, solution-by-construction).

A faint residual state signal survives anisotropy correction, which is the only reason this is not "Refuted." But honest diagnosis and real infrastructure do not constitute evidence for planning. As it stands, **the results do not demonstrate latent planning, and the available evidence leans against the representation being adequate for it.**

## Minimum bar to revisit this verdict
1. Commit the generating code with seeds; make Phase C reproducible end-to-end.
2. Report all geometry metrics against the **anisotropy floor** and **same-step** baselines (mean-centered).
3. Regress out / stratify by step position before any state or retrieval claim.
4. Build the transition model and report `gain = identity_mse/model_mse` with a shuffled-action control.
5. Re-extract hidden states from **problem-only prompts** (no solution in context) to test genuine planning.
6. Fix the benchmark (real Countdown rules, bounded targets, DIV, operand-use-once, cross-split dedup).
7. Power up the state clusters (≥30 per state) and report bootstrap CIs on every number.

Until all seven hold, the verdict remains **Probably False**.
