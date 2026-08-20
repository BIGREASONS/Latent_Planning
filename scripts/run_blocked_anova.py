import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\singh\Documents\latent_planning\legacy\v1_submission\reports\multiseed_results.csv"
)
# Y = Oracle Gap
k = 3  # architectures
n = 3  # seeds
N = k * n

y = df["Oracle Gap"].values
y_mean = y.mean()

# SS_Total
SST = np.sum((y - y_mean) ** 2)

# SS_Treatments (Architecture)
arch_means = df.groupby("Architecture")["Oracle Gap"].mean()
SSTr = n * np.sum((arch_means - y_mean) ** 2)

# SS_Blocks (Seed)
seed_means = df.groupby("Seed")["Oracle Gap"].mean()
SSB = k * np.sum((seed_means - y_mean) ** 2)

# SS_Error
SSE = SST - SSTr - SSB

# DF
df_Tr = k - 1
df_B = n - 1
df_E = (k - 1) * (n - 1)

# MS
MSTr = SSTr / df_Tr
MSE = SSE / df_E

# F and p
F = MSTr / MSE
import scipy.stats as stats

p_val = 1 - stats.f.cdf(F, df_Tr, df_E)

print(f"F({df_Tr}, {df_E}) = {F:.4f}, p = {p_val:.4f}")

import os

os.makedirs("analysis", exist_ok=True)
with open("analysis/blocked_anova.md", "w") as f:
    f.write(
        f"# Seed-Blocked ANOVA on Oracle Gap\n\nF({df_Tr}, {df_E}) = {F:.4f}, p = {p_val:.4f}\n"
    )
