
# Mechanical Reconciliation Report

This report automatically parses the actual raw run artifacts and compares them to the values presented in the current manuscript/appendix.

## 1. True Action Semantic Gain (Primary Metric)

| Condition | Manuscript (Table II / VI) | Raw Output (`multiseed_results.csv`) | Raw Output (`phase_a_report.md` / `coherence_action_depth.csv`) | MATCH? |
|---|---|---|---|---|
| Linear Seed 42 | -0.0195 | -0.0750 | -0.0750 | ? |
| Linear Seed 43 | -0.0640 | -0.0875 | -0.0875 | ? |
| Linear Seed 44 | -0.0177 | -0.1000 | -0.1000 | ? |
| MLP Seed 42 | -0.0255 | -0.0625 | -0.0625 | ? |
| MLP Seed 43 | -0.0149 | -0.0375 | -0.0375 | ? |
| MLP Seed 44 | -0.0144 | -0.0750 | -0.0750 | ? |
| Transformer Seed 42 | -0.0206 | 0.0000 | 0.0000 | ? |
| Transformer Seed 43 | -0.0237 | 0.0125 | 0.0125 | ? |
| Transformer Seed 44 | -0.0314 | 0.0000 | 0.0000 | ? |

## 2. Oracle Gap

| Condition | Manuscript (Table II / VI) | Raw Output (`multiseed_results.csv`) | MATCH? |
|---|---|---|---|
| Linear Seed 42 | 0.1798 | 0.0375 | ? |
| Linear Seed 43 | 0.2250 | 0.0500 | ? |
| Linear Seed 44 | 0.1783 | 0.0625 | ? |
| MLP Seed 42 | 0.1860 | 0.0250 | ? |
| MLP Seed 43 | 0.1755 | 0.0000 | ? |
| MLP Seed 44 | 0.1750 | 0.0375 | ? |
| Transformer Seed 42 | 0.1814 | -0.0375 | ? |
| Transformer Seed 43 | 0.1845 | -0.0500 | ? |
| Transformer Seed 44 | 0.1920 | -0.0375 | ? |

## 3. Seed-Blocked ANOVA (True Data)

Running a standard Seed-Blocked ANOVA on the **true raw Oracle Gap values** yields:
**F(2, 4) = 54.14, p = 0.0013**

*Conclusion:* The true data actually **DOES** show statistically significant architecture-level differences, directly contradicting the manuscript's claim of "no statistically significant architecture-level differences."

