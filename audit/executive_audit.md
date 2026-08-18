# Executive Audit — Latent Planning Repository

**Auditor stance:** hostile NeurIPS reviewer + reproducibility auditor + ML scientist + systems engineer + statistician. Every claim treated as false until proven. Only evidence counts.

**Scope of what exists.** Two disjoint artifacts:
1. **Repo** (`C:/Users/singh/Documents/latent_planning`): scaffolding only — dataset generator, model loader, extraction script, a broken plotter, three tests. The `prompts/phase_a_implementation.md` spec defines an entire diagnostic apparatus (action parser, trajectory dataset, transition model, diagnostic decoder, 4 probes, coherence eval, reports). **None of it exists in the repo.**
2. **Results** (`C:/Users/singh/Downloads/extracted_results/reports`): a *different* experiment — "Phase C — Intrinsic State Noise Diagnostic." Contains `intrinsic_noise.csv`, a report, `state_statistics.md`, and 4.1 GB of trajectory tensors (`train/val/test/test_ood.pt`, 5000/500/1000/500 trajectories of TinyLlama 2048-dim hidden states).

**The code that generated the Phase C results is not present in either location.** `intrinsic_noise_report.md` cites `scripts/run_phase_c.py`; `state_statistics.md` and the trajectory pickles reference `data_processing/trajectory_dataset.py` and an action parser. None of these files exist. **The headline experiment is not reproducible from supplied code.**

---

## Top findings (ranked by severity)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | **Headline experiment is not reproducible**: generating code (`run_phase_c.py`, `data_processing/*`, action parser) is absent. Results exist only as opaque 4.1 GB pickles. | Critical | Confirmed (files missing) |
| 2 | **Within-state cosine similarity is largely a position/anisotropy artifact.** Same-step different-state pairs already cosine at **0.889**; within-state is **0.929**. Global random-pair cosine floor is **0.664** (severe anisotropy). Only ~0.04 of the within-state similarity is attributable to symbolic-state matching, not the apparent 0.27 gap. | Critical | Reproduced from tensors |
| 3 | **Retrieval claim (Top-1 24.6%) not reproducible.** Under a defensible (target, numbers, step) symbolic key only 379/20022 states even have a cross-trajectory positive; my reconstruction gives n=7 eligible queries — cannot recover 24.6%. The label-construction code is missing, so the metric's denominator/positive-set is unauditable. | Critical | Could not reproduce |
| 4 | **Nearest-neighbor retrieval is contaminated by token position.** Top-1 NN shares the same reasoning *step index* 85.7% of the time → NN matches "how far into the trajectory" not "which symbolic state." | High | Reproduced |
| 5 | **Self-admitted underpowering.** `state_statistics.md`: 240 unique states, 181 cross-history, **max cluster size 3**, 198 positive pairs, with a `[!WARNING]` that the dataset is underpowered. Within-state cosine is computed on n≈200 pairs. | High | From their own files |
| 6 | **Train/test problem leakage in the real trajectory data.** 24 (target, numbers) problems shared train∩test, 12 train∩val, 4 train∩ood; train duplicate-problem rate 1.1%. Small but nonzero, and uncontrolled (no dedup, no seed). | Medium | Reproduced |
| 7 | **Dataset generator is scientifically broken** (see data_audit): targets explode to 10^8 (no division), 36% of targets >1000, "solution" is generation-by-construction not search, DIV never appears though spec requires it. Number-multiset collision rate 27.8%. | High | Reproduced |
| 8 | **No seeding anywhere.** `grep seed *.py` → nothing. Generation, splits, and any training are nondeterministic. | High | Confirmed |
| 9 | **No transition model, no probes, no coherence eval exist.** Audit areas 3, 4, 5, 6 (transition, coherence, probes) cannot be evaluated because nothing was built. The project's central claims (planning horizon, coherence decay, dynamics) have **zero supporting artifacts**. | Critical | Confirmed |
| 10 | **`evaluation/plotter.py` does not import** (`import matplotlib.pyplot.subplots` → ModuleNotFoundError). The only plotting utility is dead. | Low | Reproduced |

---

## Bottom line

The project has **not produced evidence for latent planning**. It has produced one diagnostic (intrinsic noise) whose own conclusion is negative ("representation appears highly history-dependent, Case B"), and even that negative conclusion rests on metrics confounded by anisotropy and token position, computed on an admittedly underpowered sample, using code that is not in the repository.

There is no transition model, no coherence horizon measurement, no probe — so the four research questions ("do hidden states contain reasoning info / can a transition model preserve it / what is the coherence horizon / is the bottleneck representation or dynamics") are **unanswered by artifacts**. The only answered question (intrinsic noise) answers *against* the representation being clean.

See `final_verdict.md` for scores. Headline: **Inconclusive trending Probably False**, and **not publication-ready**.
