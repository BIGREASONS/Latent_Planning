# Metric Trace: Linear Seed 43

## 1. The 20-Trajectory Debug Run (Incorrectly Audited Before)
**Location:** `legacy/v1_submission/reports/multiseed_linear_seed43`
**Data Size:** train=20, val=20, test=20
**Trace:**
- `probe_results.csv`: A:remaining_operands exact_accuracy = 0.238, accuracy = 0.607 (very noisy due to n=20)
- `coherence_action_depth.csv` (depth=1):
  - `state_probe_accuracy` = 0.5125
  - `identity_state_probe_accuracy` = 0.600
  - **True SG** = 0.5125 - 0.600 = **-0.0875**
- `phase_a_report.md` logs True SG as **-0.088** (rounded).

## 2. The 5000-Trajectory Canonical Run (Correct Paper Values)
**Location:** `final_audit_extracted/phase3_linear_all/.../reports/multiseed_linear_seed43`
**Data Size:** train=5000, val=491, test=970
**Trace:**
- `probe_results.csv`: A:remaining_operands exact_accuracy = 0.800, accuracy = 0.935 (reliable)
- `coherence_action_depth.csv` (depth=1):
  - `state_probe_accuracy` = 0.8113
  - `identity_state_probe_accuracy` = 0.8758
  - **True SG** = 0.8113 - 0.8758 = **-0.0645** (stored precisely as -0.0644 in float)
- `phase_a_report.md` logs True SG as **-0.064**.

**Conclusion:** The manuscript accurately cited the 5000-trajectory run. The previous mechanical audit was mistakenly analyzing the 20-trajectory debug run.

