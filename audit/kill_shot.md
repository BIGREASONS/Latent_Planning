# Kill Shot

## The strongest argument that these results do NOT demonstrate latent planning (evidence only)

1. **No planning artifact exists.** The deliverables that would demonstrate planning — transition model, coherence-vs-depth curve, probes — are all missing (`models/transition_model.py`, `reports/coherence_depth.csv`, `evaluation/probes.py`: verified absent). The project's defining objective ("does latent planning survive beyond depth 3") has **no measurement.** You cannot demonstrate planning with zero planning experiments.

2. **The one real experiment concludes against the premise.** `intrinsic_noise_report.md` final verdict: *"Representation appears highly history-dependent (Case B)... This is the true bottleneck."* The project's own output says the representation does not cleanly encode task state.

3. **Even that negative result is propped up by confounds — meaning there is no clean signal in either direction.** Reproduced from the raw tensors: random unrelated hidden states cosine at **0.664** (anisotropy floor); same-step different-state pairs cosine at **0.889**; within-state pairs at **0.929**. The within-state advantage over the position confound is only **+0.04**. Retrieval's nearest neighbor matches reasoning **step position 85.7%** of the time. So the geometry is explained by *anisotropy + token position*, not symbolic state.

4. **The headline retrieval number is not reproducible.** Top-1 24.6% cannot be regenerated from the supplied tensors because the labeling code is absent; under a defensible key only 7–380 queries are even eligible. An irreproducible, key-dependent, position-contaminated metric is not evidence.

5. **The benchmark is not a planning task.** The generator constructs "solutions" by random forward arithmetic, never divides, allows targets up to 2.8×10^8, and enforces no Countdown invariants. There is no search, so there is no plan to recover.

6. **Underpowered and unseeded.** Max symbolic-state cluster size 3; 198 positive pairs; their own `[!WARNING]` flags underpowering; no seed anywhere; extraction layer unrecorded. Nothing here would survive a rerun.

**Conclusion of the kill shot:** There is no positive evidence for latent planning, the sole experiment argues against it, and that argument is itself confounded — so the corpus supports *no* reliable claim. It does not demonstrate latent planning; it demonstrates an unfinished, confounded pipeline.

---

## The strongest rebuttal (evidence only)

1. **A residual state signal does survive confound-correction.** After mean-centering to remove anisotropy, within-state cosine (0.722) still exceeds between-state cosine (0.025) by **0.70**, and the raw within−between gap has a bootstrap 95% CI of **[0.258, 0.288]** that excludes zero. So identical symbolic states are *measurably* closer than random states — there is a non-null representation signal to build on.

2. **The data infrastructure is real and substantial.** 5000+ trajectories with aligned hidden states, op IDs, operands, and state indices (4.1 GB) exist and load correctly. This is a usable substrate for the missing experiments; the representation is not random noise.

3. **The negative framing is honest, not inflated.** Unlike the failure mode of a hype paper, the report concludes *against* its own hopes ("Case B", "true bottleneck"). There is no metric inflation in the stated conclusion — if anything it under-claims. That is the opposite of a false positive.

4. **History-dependence is diagnostic, not fatal.** Phase A is explicitly a *diagnostic* to decide whether to pursue planning. Finding that the frozen representation is noisy/history-dependent is a *valid result of the diagnostic*, correctly steering effort toward representation cleanup before dynamics — which is exactly what the report recommends.

**Conclusion of the rebuttal:** A real, reproducible-in-direction representation signal exists above the anisotropy floor; the infrastructure is sound; and the project's negative, non-inflated conclusion is an appropriate diagnostic outcome rather than a failed planning claim.

---

## Adjudication

The rebuttal establishes that (a) there is a small real state signal and (b) the project is honestly negative. It does **not** establish latent planning — it concedes the representation is the bottleneck and that no planning experiment was run. The kill shot stands on the central question: **no artifact demonstrates planning, and the only experiment argues against the necessary precondition.** The residual signal (rebuttal #1) lowers the verdict from "Refuted" to "Inconclusive/Probably False," but cannot raise it toward "Supported."
