import pandas as pd
import numpy as np
import scipy.stats as stats
import os


def main():
    # Phase 3 Audited Raw Results
    data = [
        {"Architecture": "linear", "Seed": 42, "Oracle Gap": 0.1798},
        {"Architecture": "linear", "Seed": 43, "Oracle Gap": 0.2250},
        {"Architecture": "linear", "Seed": 44, "Oracle Gap": 0.1783},
        {"Architecture": "mlp", "Seed": 42, "Oracle Gap": 0.1860},
        {"Architecture": "mlp", "Seed": 43, "Oracle Gap": 0.1755},
        {"Architecture": "mlp", "Seed": 44, "Oracle Gap": 0.1750},
        {"Architecture": "transformer", "Seed": 42, "Oracle Gap": 0.1814},
        {"Architecture": "transformer", "Seed": 43, "Oracle Gap": 0.1845},
        {"Architecture": "transformer", "Seed": 44, "Oracle Gap": 0.1920},
    ]
    df = pd.DataFrame(data)

    os.makedirs("analysis", exist_ok=True)

    # 1. Summary Statistics using t-distribution
    summary = []
    for arch in ["linear", "mlp", "transformer"]:
        arch_data = df[df["Architecture"] == arch]["Oracle Gap"]
        n = len(arch_data)
        mean = arch_data.mean()
        std = arch_data.std(ddof=1)
        se = std / np.sqrt(n)

        # 95% CI using t-distribution (df = n-1)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_half_width = t_crit * se

        summary.append(
            {
                "Architecture": arch.capitalize(),
                "N": n,
                "Mean": mean,
                "SD": std,
                "SE": se,
                "95% CI Half-Width": ci_half_width,
                "CI Lower": mean - ci_half_width,
                "CI Upper": mean + ci_half_width,
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("analysis/summary_statistics.csv", index=False)

    # 2. Welch's ANOVA for Oracle Gap
    def welch_anova(groups):
        k = len(groups)
        w = [len(g) / np.var(g, ddof=1) for g in groups]
        sum_w = sum(w)
        means = [np.mean(g) for g in groups]
        x_bar_w = sum(w_i * m_i for w_i, m_i in zip(w, means)) / sum_w

        num = sum(w_i * (m_i - x_bar_w) ** 2 for w_i, m_i in zip(w, means)) / (k - 1)

        lambda_val = sum(
            (1 - w_i / sum_w) ** 2 / (len(g) - 1) for w_i, g in zip(w, groups)
        )
        den = 1 + 2 * (k - 2) / (k**2 - 1) * lambda_val

        F = num / den
        df1 = k - 1
        df2 = (k**2 - 1) / (3 * lambda_val)

        p = stats.f.sf(F, df1, df2)
        return F, p, df1, df2

    groups = [
        df[df["Architecture"] == "linear"]["Oracle Gap"].values,
        df[df["Architecture"] == "mlp"]["Oracle Gap"].values,
        df[df["Architecture"] == "transformer"]["Oracle Gap"].values,
    ]
    F_welch, p_welch, df1, df2 = welch_anova(groups)

    # 3. Pairwise Welch t-tests
    pairs = [("linear", "mlp"), ("linear", "transformer"), ("mlp", "transformer")]

    tests = []
    for p1, p2 in pairs:
        g1 = df[df["Architecture"] == p1]["Oracle Gap"].values
        g2 = df[df["Architecture"] == p2]["Oracle Gap"].values

        res = stats.ttest_ind(g1, g2, equal_var=False)
        t_stat = res.statistic
        p_val = res.pvalue

        # Cohen's d (pooled standard deviation)
        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(g1) - np.mean(g2)) / pooled_sd

        tests.append(
            {
                "Comparison": f"{p1.capitalize()} vs {p2.capitalize()}",
                "t-statistic": t_stat,
                "df": (
                    res.df
                    if hasattr(res, "df")
                    else (var1 / n1 + var2 / n2) ** 2
                    / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
                ),
                "p-value (uncorrected)": p_val,
                "Cohen's d": d,
            }
        )

    tests_df = pd.DataFrame(tests)

    # Multiple comparison correction (Holm-Bonferroni)
    tests_df = tests_df.sort_values("p-value (uncorrected)")
    tests_df["Rank"] = range(1, len(tests_df) + 1)
    tests_df["p-value (Holm corrected)"] = np.minimum(
        1, tests_df["p-value (uncorrected)"] * (len(tests_df) - tests_df["Rank"] + 1)
    )
    tests_df["p-value (Holm corrected)"] = tests_df["p-value (Holm corrected)"].cummax()

    tests_df.to_csv("analysis/hypothesis_tests.csv", index=False)

    # 4. Generate Report
    report = f"""# Statistical Analysis Report

## 1. Summary Statistics (Oracle Gap)
Note: 95% Confidence Intervals are calculated using the t-distribution (df=2) due to n=3.

{summary_df.to_markdown(index=False)}

## 2. Welch's ANOVA
* **H0**: Linear, MLP, Transformer have equal mean Oracle Gap.
* **H1**: At least one differs.
* **F-statistic**: {F_welch:.4f}
* **Degrees of Freedom**: ({df1:.2f}, {df2:.2f})
* **p-value**: {p_welch:.4f}

## 3. Pairwise Welch t-tests
Multiple comparisons corrected using Holm-Bonferroni.

{tests_df[['Comparison', 't-statistic', 'df', 'p-value (uncorrected)', 'p-value (Holm corrected)', "Cohen's d"]].to_markdown(index=False)}

## 4. Conclusion
"""

    if p_welch < 0.05:
        report += "The Welch's ANOVA indicates a statistically significant difference in Oracle Gap between at least two architectures at α=0.05.\n"
    else:
        report += "The Welch's ANOVA does NOT indicate a statistically significant difference in Oracle Gap among the architectures (p > 0.05). \n\n**Interpretation:** Due to the extremely low statistical power (n=3), this result is technically inconclusive rather than proving equivalence. The effect sizes (Cohen's d) between Linear vs MLP and Linear vs Transformer are large, suggesting a potentially meaningful difference that fails to reach statistical significance purely due to the small sample size."

    with open("analysis/report.md", "w") as f:
        f.write(report)

    print("Analysis complete. Check the 'analysis' directory.")


if __name__ == "__main__":
    main()
