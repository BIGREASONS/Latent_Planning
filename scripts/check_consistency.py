import os
import pandas as pd
import re

ARCHIVE_DIR = "C:/Users/singh/OneDrive/Documents/latent_planning/research_archive/experiments"

def check_consistency():
    experiments = os.listdir(ARCHIVE_DIR)
    all_consistent = True
    
    for exp in experiments:
        exp_dir = os.path.join(ARCHIVE_DIR, exp)
        csv_path = os.path.join(exp_dir, "coherence_action_depth.csv")
        md_a_path = os.path.join(exp_dir, "phase_a_report.md")
        md_b_path = os.path.join(exp_dir, "phase_b_oracle_report.md")
        
        if not os.path.exists(csv_path):
            continue
            
        print(f"\n--- Checking {exp} ---")
        df = pd.read_csv(csv_path)
        
        # Check Phase B Oracle Report
        if os.path.exists(md_b_path):
            with open(md_b_path, "r", encoding="utf-8") as f:
                md_b = f.read()
            
            # Check Depth 1 Identity Cosine
            d1_cos_id = df[df['depth'] == 1]['identity_cosine_similarity'].values[0]
            expected_str_id = f"{d1_cos_id:.3f}"
            if expected_str_id in md_b:
                print(f"  [OK] Depth 1 Identity Cosine {expected_str_id} found in Phase B report.")
            else:
                print(f"  [FAIL] Depth 1 Identity Cosine {expected_str_id} NOT found in Phase B report!")
                all_consistent = False
                
            # Check Depth 1 Action Cosine
            d1_cos_action = df[df['depth'] == 1]['cosine_similarity'].values[0]
            expected_str_action = f"{d1_cos_action:.3f}"
            if expected_str_action in md_b:
                print(f"  [OK] Depth 1 Action Cosine {expected_str_action} found in Phase B report.")
            else:
                print(f"  [FAIL] Depth 1 Action Cosine {expected_str_action} NOT found in Phase B report!")
                all_consistent = False

        # Check Phase A Report
        if os.path.exists(md_a_path):
            with open(md_a_path, "r", encoding="utf-8") as f:
                md_a = f.read()
                
            d1_cos_action = df[df['depth'] == 1]['cosine_similarity'].values[0]
            expected_str_action = f"{d1_cos_action:.3f}"
            if expected_str_action in md_a:
                print(f"  [OK] Depth 1 Action Cosine {expected_str_action} found in Phase A report.")
            else:
                print(f"  [FAIL] Depth 1 Action Cosine {expected_str_action} NOT found in Phase A report!")
                all_consistent = False
                
    if all_consistent:
        print("\nSUCCESS: All reports are strictly consistent with underlying CSV data.")
    else:
        print("\nERROR: Inconsistencies detected.")

if __name__ == "__main__":
    check_consistency()
