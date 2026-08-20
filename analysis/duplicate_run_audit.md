# Duplicate Run Audit

A comprehensive search of the repository and its extracted ZIP archives revealed that there are exactly **two** instances of the `multiseed_*` experiments for the 9 architectural conditions:

## Instance A: The Debug/Dev Suite
- **Path:** `legacy/v1_submission/reports/multiseed_*`
- **Scale:** `train=20 val=20 test=20`
- **Purpose:** Fast development loop, pipeline smoke testing.
- **Timestamp:** 2026-07-22 ~06:00
- **Signatures:** Tiny trajectory counts, noisy probe accuracies (~0.60), negative Oracle Gaps due to small-sample variance.

## Instance B: The Preregistered Phase 3 Study (Canonical)
- **Path:** Scattered across multiple extracted ZIP folders (e.g., `final_audit_extracted/`, `review_extracted/`, `results(1)/`, `results(3)/`).
- **Scale:** `train=5000 val=491 test=970`
- **Purpose:** The actual preregistered 5000-trajectory scientific execution.
- **Timestamp:** Extracted timestamps vary (e.g., 2026-07-24, 2026-07-27, 2026-08-05), but represent the final computed artifacts.
- **Signatures:** Full trajectory counts, stable probe accuracies (~0.93), mathematically consistent large positive Oracle Gaps.

## Conclusion
The conflicting numbers observed in previous audits were entirely caused by the presence of Instance A (the 20-trajectory debug suite) in the main, unzipped repository tree. The manuscript values were cleanly derived from Instance B (the canonical 5000-trajectory execution), which was later uploaded/stored as distributed ZIP archives. The manuscript did not fabricate data; it correctly cited Instance B.

