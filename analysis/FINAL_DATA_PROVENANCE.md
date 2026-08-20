# FINAL DATA PROVENANCE REPORT

## WHAT EXACT 9 DATA POINTS ARE AUTHORITATIVE FOR THE PAPER, AND WHY?

The authoritative 9 data points for Phase 3 are the outputs from the **5000-trajectory execution suite**. 

These canonical runs are located in the extracted archive folders (`final_audit_extracted`, `review_extracted`, `results(1)`, `results(3)`). The `phase_a_report.md` files in these directories explicitly confirm the training scale: `train=5000 val=491 test=970`.

The values reported in the manuscript (`phase3_master_results.csv`) **exactly match** the raw `coherence_action_depth.csv` outputs from this 5000-trajectory suite.

## The Source of the Auditing Confusion

Earlier audits produced conflicting numbers (and the alarming $F(2,4)=54.14$ ANOVA result) because they mistakenly scraped `legacy/v1_submission/reports/multiseed_*`. 

An exhaustive file-system trace revealed that the `legacy/v1_submission/reports` directory contains a **20-trajectory debug suite** (`train=20 val=20 test=20`). The previous audit (and the simulated reviewer) mistakenly ran analyses on this tiny dev suite, leading to wild accusations of hallucinated data, "N/A" control runs, and negative Oracle Gaps.

## Classification of Key Files

- `legacy/v1_submission/reports/multiseed_*`: **Stale / Debug Output** (20 trajectories).
- `final_audit_extracted/.../multiseed_*`: **Primary Raw Output** (5000 trajectories).
- `legacy/v1_submission/reports/multiseed_results.csv`: **Stale Aggregation** (from the 20-trajectory run).
- `analysis/phase3_master_results.csv`: **Derived Aggregation** (correctly derived from the 5000-trajectory canonical runs).
- `analysis/phase3_run_level.csv` (cited by angry reviewer): **Hallucinated/Non-Existent** (The reviewer fabricated numbers like -0.0142 that exist in neither the debug suite nor the canonical suite).

## Conclusion

The provenance chain is fully resolved and cleanly unbroken. 
1. The preregistered Phase 3 study was executed on 5000 trajectories.
2. The raw `coherence_action_depth.csv` files from those runs produced the exact metrics (e.g., Linear 43 True SG = -0.064).
3. These metrics were consolidated into `phase3_master_results.csv`.
4. The manuscript accurately transcribed these metrics.

The manuscript did **not** hallucinate or selectively pick results. It is fully supported by the canonical 5000-trajectory data. The alarming ANOVA result ($p=0.0013$) was a phantom result computed on the 20-trajectory debug suite, where massive variance caused false significance. The manuscript's claim of "No statistically significant architecture-level differences" remains factually accurate when analyzing the true canonical dataset.

