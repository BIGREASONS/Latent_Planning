import sys


def verify():
    if len(sys.argv) < 2:
        print("Usage: python verify_paper.py <tex_file>")
        sys.exit(1)

    tex_file = sys.argv[1]
    with open(tex_file, "r") as f:
        content = f.read()

    errors = 0

    # Check Phase 3 values
    expected_values = [
        "0.1944",
        "0.1788",
        "0.1860",  # Oracle Gap
        "-0.0337",
        "-0.0183",
        "-0.0252",  # SG
        "0.941",
        "0.946",
        "0.946",  # Cosine
        "1.1748",
        "0.4050",  # ANOVA
    ]

    for val in expected_values:
        if val not in content:
            print(f"ERROR: Expected value {val} not found in manuscript!")
            errors += 1

    if errors == 0:
        print("Verification Passed: All expected Phase 3 metrics found in manuscript.")
    else:
        print("Verification Failed.")


if __name__ == "__main__":
    verify()
