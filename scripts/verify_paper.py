import sys

def verify():
    if len(sys.argv) < 2:
        print('Usage: python verify_paper.py <tex_file>')
        sys.exit(1)
        
    tex_file = sys.argv[1]
    with open(tex_file, 'r') as f:
        content = f.read()

    errors = 0
    
    # Check Phase 3 values
    expected_values = [
        '0.1944', '0.1789', '0.1860', # Oracle Gap
        '-0.0339', '-0.0183', '-0.0254', # SG
        '0.9411', '0.9461', '0.9466', # Cosine
        '1.160', '0.409', # Welch ANOVA
        '0.604', '0.590'  # Blocked ANOVA
    ]
    
    for val in expected_values:
        if val not in content:
            print(f'ERROR: Expected value {val} not found in manuscript!')
            errors += 1
            
    if errors == 0:
        print('Verification Passed: All expected Phase 3 metrics found in manuscript.')
    else:
        print('Verification Failed.')

if __name__ == '__main__':
    verify()
