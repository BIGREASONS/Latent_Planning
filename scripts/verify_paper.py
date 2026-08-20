import sys
import pandas as pd
import os

def verify():
    if len(sys.argv) < 2:
        print("Usage: python verify_paper.py <tex_file>")
        sys.exit(1)
        
    tex_file = sys.argv[1]
    with open(tex_file, "r") as f:
        content = f.read()

    # Determine paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv_path = os.path.join(script_dir, "..", "analysis", "phase3_master_results.csv")
    
    if not os.path.exists(master_csv_path):
        print(f"ERROR: Cannot find master results at {master_csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(master_csv_path)
    
    errors = 0
    expected_values = []
    
    # Generate expected formatted values from the authoritative CSV
    for _, row in df.iterrows():
        og = row["Oracle Gap"]
        sg = row["True SG"]
        cos = row["Cosine"]
        expected_values.append(f"{og:.4f}")
        expected_values.append(f"{sg:.4f}")
        expected_values.append(f"{cos:.4f}")
        
    # Remove duplicates
    expected_values = list(set(expected_values))
    
    for val in expected_values:
        if val not in content:
            print(f"ERROR: Expected value {val} not found in manuscript!")
            errors += 1
            
    if errors == 0:
        print("Verification Passed: All expected Phase 3 metrics found in manuscript.")
    else:
        print("Verification Failed.")
        sys.exit(1)

if __name__ == "__main__":
    verify()

