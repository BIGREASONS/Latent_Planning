import matplotlib.pyplot as plt
import numpy as np

# Data
archs = ['Linear', 'MLP', 'Transformer']
cosine = [0.9407, 0.9457, 0.9460]
true_sg = [-0.0337, -0.0183, -0.0252]
oracle_gap = [0.1944, 0.1788, 0.1860]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Cosine
bars1 = ax1.bar(archs, cosine, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_title('A: Raw Cosine (Geometry)', pad=15)
ax1.set_ylim(0, 1.1)  # Fixed spacing to avoid clash
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

# Panel B: True Action SG
bars2 = ax2.bar(archs, true_sg, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax2.set_title('B: True Action Semantic Gain (Task)', pad=15)
ax2.set_ylim(-0.05, 0.01)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval - 0.002, f'{yval:.4f}', ha='center', va='top', fontsize=12)

# Panel C: Oracle Gap
bars3 = ax3.bar(archs, oracle_gap, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax3.set_title('C: Oracle Gap (Lost Semantics)', pad=15)
ax3.set_ylim(0, 0.25)
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('fig1.pdf', bbox_inches='tight')
