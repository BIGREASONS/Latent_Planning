import subprocess
import os
import glob
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Multi-seed runner for transition models")
    parser.add_argument("--domain", type=str, default="countdown")
    parser.add_argument("--archs", nargs="+", default=['linear', 'mlp', 'transformer'])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--traj_dir", type=str, default="reports/trajectories")
    args = parser.parse_args()

    print(f"Running Multi-Seed Analysis for {args.domain} across seeds {args.seeds}...")
    print("This runner explicitly fixes the dataset and trajectories while varying the training seed.")

    results = []

    for arch in args.archs:
        for seed in args.seeds:
            out_dir = f"reports/multiseed_{arch}_seed{seed}"
            print(f"\n--- Running {arch} with training seed {seed} ---")
            
            expected_files = [
                f"{out_dir}/phase_a_report.md",
                f"{out_dir}/transition_model_action.pt",
                f"{out_dir}/transition_model_blind.pt",
                f"{out_dir}/coherence_action_depth.csv"
            ]
            
            if all(os.path.exists(f) for f in expected_files):
                print(f"[{arch} - seed {seed}] All completion artifacts found, skipping execution to resume.")

            else:
                # Run Phase A
                import sys
                cmd_phase_a = [
                    sys.executable, "scripts/run_phase_a.py",
                    "--domain", args.domain,
                    "--transition_arch", arch,
                    "--out_dir", out_dir,
                    "--trajectories_dir", args.traj_dir,
                    "--seed", str(seed)
                ]
                subprocess.run(cmd_phase_a, check=True)

            
            # Collect metrics (Depth 1)
            def _get_metrics(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    d1 = df[df['depth'] == 1].iloc[0]
                    sem_gain = d1['state_probe_accuracy'] - d1['identity_state_probe_accuracy']
                    cosine = d1['cosine_similarity']
                    mse = d1['mse']
                    mcc = d1.get('mean_centered_cosine', np.nan)
                    return sem_gain, cosine, mse, mcc, d1['state_probe_accuracy']
                except Exception as e:
                    return np.nan, np.nan, np.nan, np.nan, np.nan
            
            true_sg, true_cos, true_mse, true_mcc, true_acc = _get_metrics(f"{out_dir}/coherence_action_depth.csv")
            shuf_sg, _, _, _, _ = _get_metrics(f"{out_dir}/coherence_shuffled_depth.csv")
            const_sg, _, _, _, _ = _get_metrics(f"{out_dir}/coherence_constant_depth.csv")
            blind_sg, _, _, _, _ = _get_metrics(f"{out_dir}/coherence_blind_depth.csv")
            oracle_sg, _, _, _, oracle_acc = _get_metrics(f"{out_dir}/coherence_oracle_depth.csv")
            
            oracle_gap = oracle_acc - true_acc if not np.isnan(oracle_acc) and not np.isnan(true_acc) else np.nan

            results.append({
                "Architecture": arch,
                "Seed": seed,
                "True Action SG": true_sg,
                "Shuffled Action SG": shuf_sg,
                "Constant Action SG": const_sg,
                "Blind SG": blind_sg,
                "Oracle SG": oracle_sg,
                "Oracle Gap": oracle_gap,
                "Cosine": true_cos,
                "Mean-Centered Cosine": true_mcc,
                "MSE": true_mse
            })

    new_df = pd.DataFrame(results)
    csv_path = "reports/multiseed_results.csv"
    
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        # Merge, drop duplicates based on Arch and Seed (keep latest), then sort
        df = pd.concat([old_df, new_df]).drop_duplicates(subset=["Architecture", "Seed"], keep="last")
    else:
        df = new_df
        
    # Sort deterministically
    df['Arch_cat'] = pd.Categorical(df['Architecture'], categories=['linear', 'mlp', 'transformer'], ordered=True)
    df = df.sort_values(by=['Arch_cat', 'Seed']).drop('Arch_cat', axis=1)
    
    df.to_csv(csv_path, index=False)

    print("\n--- Final Aggregated Results ---")
    agg = df.groupby("Architecture").agg(["mean", "std"])
    print(agg)

    print("\nMarkdown Table (Action Controls):")
    print("| Architecture | True Action SG | Shuffled Action SG | Constant Action SG | Blind SG | Oracle Gap |")
    print("|---|---|---|---|---|---|")
    for arch in args.archs:
        arch_data = df[df["Architecture"] == arch]
        def _fmt(col):
            m, s = arch_data[col].mean(), arch_data[col].std()
            return f"{m:+.3f} ± {s:.3f}"
        
        t_sg = _fmt("True Action SG")
        s_sg = _fmt("Shuffled Action SG")
        c_sg = _fmt("Constant Action SG")
        b_sg = _fmt("Blind SG")
        o_gap = _fmt("Oracle Gap")
        
        print(f"| {arch.capitalize():<12} | {t_sg:<16} | {s_sg:<18} | {c_sg:<18} | {b_sg:<10} | {o_gap:<12} |")

if __name__ == "__main__":
    main()
