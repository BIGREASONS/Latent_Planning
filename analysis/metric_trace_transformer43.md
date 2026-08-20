# Metric Trace: Transformer Seed 43

## 1. The 20-Trajectory Debug Run
**Location:** `legacy/v1_submission/reports/multiseed_transformer_seed43`
**Data Size:** test=20 trajectories (very few state evaluations)
**Trace:**
- `coherence_action_depth.csv` (depth=1):
  - `state_probe_accuracy` = 0.600
  - `teacher_state_probe_accuracy` (Oracle) = 0.550
- **Oracle Gap** = Oracle - Predicted = 0.550 - 0.600 = **-0.050**
- *Why is it negative?* With a tiny test set, the predicted latent vector accidentally crossed the linear decision boundary for a few states where the frozen teacher state was barely on the wrong side. The probe was weakly trained (on 20 trajectories), resulting in high noise and an artificial "prediction > oracle" anomaly.

## 2. The 5000-Trajectory Canonical Run
**Location:** `final_audit_extracted/results(3)/.../reports/multiseed_transformer_seed43`
**Data Size:** test=970 trajectories
**Trace:**
- `coherence_action_depth.csv` (depth=1):
  - `state_probe_accuracy` = 0.6907
  - `teacher_state_probe_accuracy` (Oracle) = 0.8753
- **Oracle Gap** = 0.8753 - 0.6907 = **0.1846**
- *Conclusion:* Once evaluated on a statistically sound sample size, the negative Oracle Gap anomaly completely vanishes. The Transformer transition strictly underperforms the teacher representation, resulting in a large positive Oracle Gap (0.185) exactly as reported in the manuscript.

