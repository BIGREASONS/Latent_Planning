import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot architectures comparison (Probe A vs Depth)")
    parser.add_argument("--linear_csv", required=True, help="Path to Linear coherence CSV")
    parser.add_argument("--mlp_csv", required=True, help="Path to MLP coherence CSV")
    parser.add_argument("--transformer_csv", required=True, help="Path to Transformer coherence CSV")
    parser.add_argument("--oracle_csv", required=True, help="Path to Oracle coherence CSV")
    parser.add_argument("--out", required=True, help="Output image path")
    args = parser.parse_args()

    dfs = {}
    try:
        dfs["Linear"] = pd.read_csv(args.linear_csv)
        dfs["MLP"] = pd.read_csv(args.mlp_csv)
        dfs["Transformer"] = pd.read_csv(args.transformer_csv)
        dfs["Oracle"] = pd.read_csv(args.oracle_csv)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        "Identity": "gray",
        "Linear": "orange",
        "MLP": "blue",
        "Transformer": "purple",
        "Oracle": "green"
    }
    
    line_styles = {
        "Identity": "--",
        "Linear": "-",
        "MLP": "-",
        "Transformer": "-",
        "Oracle": "-."
    }
    
    # ---------------------------
    # Plot 1: Probe A Accuracy
    # ---------------------------
    ax = axes[0]
    
    if "identity_state_probe_accuracy" in dfs["MLP"].columns:
        ax.plot(dfs["MLP"]["depth"], dfs["MLP"]["identity_state_probe_accuracy"], 
                 label="Identity", color=colors["Identity"], linestyle=line_styles["Identity"], linewidth=2)
    
    for arch in ["Linear", "MLP", "Transformer"]:
        df = dfs[arch]
        if "state_probe_accuracy" in df.columns:
            ax.plot(df["depth"], df["state_probe_accuracy"], 
                     label=arch, color=colors[arch], linestyle=line_styles[arch], linewidth=2)
            
    if "state_probe_accuracy" in dfs["Oracle"].columns:
        ax.plot(dfs["Oracle"]["depth"], dfs["Oracle"]["state_probe_accuracy"], 
                 label="Oracle", color=colors["Oracle"], linestyle=line_styles["Oracle"], linewidth=2)
                 
    ax.set_xlabel("Rollout Depth")
    ax.set_ylabel("Probe A Accuracy (State/Action)")
    ax.set_title("Probe A Accuracy vs. Rollout Depth")
    ax.legend()
    
    # ---------------------------
    # Plot 2: Semantic Gain
    # ---------------------------
    ax2 = axes[1]
    
    # Semantic Gain = state_probe_accuracy - identity_state_probe_accuracy
    for arch in ["Linear", "MLP", "Transformer"]:
        df = dfs[arch]
        if "state_probe_accuracy" in df.columns and "identity_state_probe_accuracy" in df.columns:
            gain = df["state_probe_accuracy"] - df["identity_state_probe_accuracy"]
            ax2.plot(df["depth"], gain, 
                     label=arch, color=colors[arch], linestyle=line_styles[arch], linewidth=2)
                     
    if "state_probe_accuracy" in dfs["Oracle"].columns and "identity_state_probe_accuracy" in dfs["Oracle"].columns:
        oracle_gain = dfs["Oracle"]["state_probe_accuracy"] - dfs["Oracle"]["identity_state_probe_accuracy"]
        ax2.plot(dfs["Oracle"]["depth"], oracle_gain, 
                 label="Oracle", color=colors["Oracle"], linestyle=line_styles["Oracle"], linewidth=2)
    
    # Baseline for zero gain
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Rollout Depth")
    ax2.set_ylabel("Semantic Gain (Accuracy - Identity)")
    ax2.set_title("Semantic Gain vs. Rollout Depth")
    ax2.legend()
    
    plt.suptitle("Architecture Comparison: Linear vs MLP vs Transformer")
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=300)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
