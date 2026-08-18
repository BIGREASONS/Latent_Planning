# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 19874 |
| states_with_swap_partner | 225 |
| states_without_swap_partner | 19649 |
| coverage_rate | 0.011321 |
| coverage_rate_ci_lo | 0.009942 |
| coverage_rate_ci_hi | 0.012890 |
| total_state_instances | 26374 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.957130 |
| probeA_per_label | 0.986502 |
| probeA_jaccard | 0.974011 |
| probeB_accuracy | 0.669065 |
| probeC_accuracy | 0.653014 |
| probeD_accuracy | 0.838080 |
| n | 19874 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.417379 |
| probeA_exact_ci_lo | 0.410523 |
| probeA_exact_ci_hi | 0.424236 |
| probeA_per_label | 0.814984 |
| probeA_jaccard | 0.653127 |
| n | 19874 |

## 4. Oracle 2A — Representation Agreement

| metric | all | dd=0 |
|---|---|---|
| probeA_pred_exact | 0.336170 | 0.336170 |
| probeA_pred_per_label | 0.698936 | 0.698936 |
| probeA_pred_jaccard | 0.755319 | 0.755319 |
| probeB_pred_abs_diff | 0.255319 | 0.255319 |
| probeB_pred_agreement | 0.787234 | 0.787234 |
| probeC_pred_agreement | 0.753191 | 0.753191 |
| probeD_pred_agreement | 0.957447 | 0.957447 |
| gtA_exact | 0.404255 | 0.404255 |
| gtA_per_label | 0.743617 | 0.743617 |
| gtA_jaccard | 0.846809 | 0.846809 |
| gtB_abs_diff | 0.255319 | 0.255319 |
| gtB_agreement | 0.791489 | 0.791489 |
| gtC_agreement | 0.857143 | 0.857143 |
| gtD_agreement | 0.906383 | 0.906383 |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | 0.336170 | 0.275643 | 0.396697 | 235 |
| probeA_pred_per_label | 0.698936 | 0.662492 | 0.735380 | 235 |
| probeA_pred_jaccard | 0.755319 | 0.708327 | 0.802312 | 235 |
| probeB_pred_abs_diff | 0.255319 | 0.188100 | 0.322538 | 235 |
| probeB_pred_agreement | 0.787234 | 0.734796 | 0.839672 | 235 |
| probeC_pred_agreement | 0.753191 | 0.697949 | 0.808434 | 235 |
| probeD_pred_agreement | 0.957447 | 0.931585 | 0.983309 | 235 |
| gtA_exact | 0.404255 | 0.341377 | 0.467133 | 235 |
| gtA_per_label | 0.743617 | 0.709292 | 0.777942 | 235 |
| gtA_jaccard | 0.846809 | 0.808407 | 0.885210 | 235 |
| gtB_abs_diff | 0.255319 | 0.187068 | 0.323570 | 235 |
| gtB_agreement | 0.791489 | 0.739439 | 0.843540 | 235 |
| gtC_agreement | 0.857143 | 0.811215 | 0.903070 | 224 |
| gtD_agreement | 0.906383 | 0.869060 | 0.943706 | 235 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all | dd=0 |
|---|---|---|
| probeA_exact | 0.123404 | 0.123404 |
| probeA_per_label | 0.561702 | 0.561702 |
| probeA_jaccard | 0.527660 | 0.527660 |

## 6. Oracle 2B — Delta

| metric | all | dd=0 |
|---|---|---|
| oracle1_paired_exact | 0.306383 | 0.306383 |
| delta_transition_accuracy | 0.182979 | 0.182979 |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | 0.123404 | 0.081263 | 0.165545 | 235 |
| probeA_per_label | 0.561702 | 0.527772 | 0.595633 | 235 |
| probeA_jaccard | 0.527660 | 0.474205 | 0.581114 | 235 |
| oracle1_paired_exact | 0.306383 | 0.247318 | 0.365448 | 235 |
| delta_transition_accuracy | 0.182979 | 0.114498 | 0.251460 | 235 |
