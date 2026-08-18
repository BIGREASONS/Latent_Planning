# Statistical Analysis Report

## 1. Summary Statistics (Oracle Gap)
Note: 95% Confidence Intervals are calculated using the t-distribution (df=2) due to n=3.

| Architecture   |   N |     Mean |         SD |         SE |   95% CI Half-Width |   CI Lower |   CI Upper |
|:---------------|----:|---------:|-----------:|-----------:|--------------------:|-----------:|-----------:|
| Linear         |   3 | 0.194367 | 0.0265398  | 0.0153228  |           0.0659286 |   0.128438 |   0.260295 |
| Mlp            |   3 | 0.178833 | 0.00621155 | 0.00358624 |           0.0154303 |   0.163403 |   0.194264 |
| Transformer    |   3 | 0.185967 | 0.00545008 | 0.0031466  |           0.0135387 |   0.172428 |   0.199505 |

## 2. Welch's ANOVA
* **H0**: Linear, MLP, Transformer have equal mean Oracle Gap.
* **H1**: At least one differs.
* **F-statistic**: 1.1748
* **Degrees of Freedom**: (2.00, 3.59)
* **p-value**: 0.4050

## 3. Pairwise Welch t-tests
Multiple comparisons corrected using Holm-Bonferroni.

| Comparison            |   t-statistic |      df |   p-value (uncorrected) |   p-value (Holm corrected) |   Cohen's d |
|:----------------------|--------------:|--------:|------------------------:|---------------------------:|------------:|
| Mlp vs Transformer    |     -1.49515  | 3.93348 |                0.210367 |                   0.6311   |   -1.22079  |
| Linear vs Mlp         |      0.987067 | 2.21845 |                0.418759 |                   0.837519 |    0.805937 |
| Linear vs Transformer |      0.536997 | 2.16838 |                0.641309 |                   0.837519 |    0.438457 |

## 4. Conclusion
The Welch's ANOVA does NOT indicate a statistically significant difference in Oracle Gap among the architectures (p > 0.05). 

**Interpretation:** Due to the extremely low statistical power (n=3), this result is technically inconclusive rather than proving equivalence. The effect sizes (Cohen's d) between Linear vs MLP and Linear vs Transformer are large, suggesting a potentially meaningful difference that fails to reach statistical significance purely due to the small sample size.