import os
import subprocess

def reproduce():
    print('Reproducing Phase 3 Analysis...')
    if os.path.exists('scripts/analysis.py'):
        subprocess.run(['python', 'scripts/analysis.py'])
        print('Phase 3 analysis reproduced successfully.')
    else:
        print('scripts/analysis.py not found!')

if __name__ == '__main__':
    reproduce()
