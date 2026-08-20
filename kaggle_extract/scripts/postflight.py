import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Latent Planning Postflight Validator")
    parser.add_argument("--out_dir", required=True, help="Output directory to validate")
    args = parser.parse_args()

    print(f"=== LATENT PLANNING POSTFLIGHT VALIDATION ===")
    print(f"Checking artifacts in: {args.out_dir}")
    print("-" * 40)

    if not os.path.exists(args.out_dir):
        print(f"[FAIL] Output directory '{args.out_dir}' does not exist.")
        sys.exit(1)

    expected_files = [
        "probe_results.csv",
        "phase_b_oracle_report.md",
        "coherence_action_depth.csv",
        "coherence_blind_depth.csv",
        "coherence_oracle_depth.csv",
        "transition_model_action.pt",
        "transition_model_blind.pt",
        "trajectories/train.pt",
        "trajectories/test.pt",
        "trajectories/val.pt",
    ]

    missing = []
    for f in expected_files:
        path = os.path.join(args.out_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size == 0:
                print(f"[WARN] File exists but is empty (0 bytes): {f}")
                missing.append(f)
            else:
                print(f"[OK] Found: {f}")
        else:
            print(f"[FAIL] Missing: {f}")
            missing.append(f)

    print("-" * 40)
    if missing:
        print(
            f"❌ POSTFLIGHT FAILED: {len(missing)} expected artifacts are missing or empty."
        )
        print(f"Missing: {missing}")
        sys.exit(1)
    else:
        print("✅ POSTFLIGHT PASSED: All expected artifacts generated successfully.")


if __name__ == "__main__":
    main()
