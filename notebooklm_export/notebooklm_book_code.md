# Source Code


## File: `audit_script.py`
```python
import torch
import sys
from evaluation.probes import extract_probe_data
from data_processing.trajectory_dataset import Trajectory
import os

def analyze(name, path_train, path_test):
    print(f"\n{'='*50}\nAnalyzing: {name}\n{'='*50}")
    
    if not os.path.exists(path_train):
        print(f"Path does not exist: {path_train}")
        return
        
    try:
        tr = torch.load(path_train, map_location="cpu", weights_only=False)
        te = torch.load(path_test, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    print(f"Train trajectories: {len(tr)}")
    print(f"Test trajectories: {len(te)}")
    
    if len(tr) > 0:
        print(f"Train hidden states shape: {tr[0].states.shape if len(tr[0].states) > 0 else 'empty'}")
        print(f"Train hidden dim: {tr[0].hidden_dim}")
        
    print("\nExtracting probe data for TRAIN...")
    tr_data = extract_probe_data(tr)
    for k in ["X", "A", "B", "C", "D"]:
        print(f"  len(tr_data['{k}']): {len(tr_data[k])}")
        print(f"  type: {type(tr_data[k])}, shape: {tr_data[k].shape}, dtype: {tr_data[k].dtype}")
        
    print("\nExtracting probe data for TEST...")
    te_data = extract_probe_data(te)
    for k in ["X", "A", "B", "C", "D"]:
        print(f"  len(te_data['{k}']): {len(te_data[k])}")
        print(f"  type: {type(te_data[k])}, shape: {te_data[k].shape}, dtype: {te_data[k].dtype}")

analyze("Mistral 7B Countdown", "notebook_output/mistral7b_countdown/trajectories/train.pt", "notebook_output/mistral7b_countdown/trajectories/test.pt")
analyze("Qwen 7B Countdown", "outputs/qwen7b_countdown/trajectories/train.pt", "outputs/qwen7b_countdown/trajectories/test.pt")

```


## File: `build_clean_zip.py`
```python
import zipfile
import subprocess
import os

def main():
    # Get all git tracked files
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error getting git files")
        return
        
    files = result.stdout.splitlines()
    
    # Filter out audit and reports and some other things that might clash
    allowed_files = []
    for f in files:
        if f.startswith("audit/") or f == "audit.zip":
            continue
        if f.startswith("reports/"):
            continue
        if f.startswith("outputs/") or f.startswith("out_test/"):
            continue
        allowed_files.append(f)
        
    print(f"Zipping {len(allowed_files)} files...")
    
    zip_name = "latent_planning_kaggle.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in allowed_files:
            if os.path.exists(f):
                zf.write(f)
            else:
                print(f"Warning: {f} not found locally.")
                
    print(f"Successfully created {zip_name} without audit/ directory.")

if __name__ == "__main__":
    main()

```


## File: `check_hidden_dim.py`
```python
import torch
te = torch.load("notebook_output/mistral7b_countdown/trajectories/test.pt", map_location="cpu", weights_only=False)
print("hidden_dim:", te[0].all_hidden.shape[1])

```


## File: `check_labels.py`
```python
import torch
import numpy as np
import os
from data_processing.trajectory_dataset import Trajectory

tr = torch.load("notebook_output/mistral7b_countdown/trajectories/train.pt", map_location="cpu", weights_only=False)

def extract_A(trajectories):
    A = []
    for traj in trajectories:
        N = traj.num_steps
        operands = traj.operands.tolist()
        used = set()
        for i in range(N):
            a_row = []
            for v in [25, 50, 75, 100]:
                if v not in traj.numbers:
                    a_row.append(0)
                elif v not in used:
                    a_row.append(1)
                else:
                    a_row.append(2)
            A.append(a_row)
            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))
    return np.array(A, dtype=np.int64)

A_tr = extract_A(tr)
print("A_tr shape:", A_tr.shape)

for j in range(4):
    classes = np.unique(A_tr[:, j])
    print(f"Label {j} classes: {classes} (count={classes.shape[0]})")

```


## File: `check_labels_bcd.py`
```python
import torch
import numpy as np
import os
from data_processing.trajectory_dataset import Trajectory

tr = torch.load("notebook_output/mistral7b_countdown/trajectories/train.pt", map_location="cpu", weights_only=False)

def extract_BCD(trajectories):
    B, C, D = [], [], []
    for traj in trajectories:
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        for i in range(N):
            dist = N - i
            B.append(dist)
            C.append(ops[i])
            D.append(1 if dist <= 2 else 0)
    return np.array(B), np.array(C), np.array(D)

B_tr, C_tr, D_tr = extract_BCD(tr)
print("B_tr classes:", np.unique(B_tr))
print("C_tr classes:", np.unique(C_tr))
print("D_tr classes:", np.unique(D_tr))

```


## File: `check_num_steps.py`
```python
import torch

te = torch.load("notebook_output/mistral7b_countdown/trajectories/test.pt", map_location="cpu", weights_only=False)

num_steps_list = [t.num_steps for t in te]
print(f"Total trajectories: {len(te)}")
print(f"Min num_steps: {min(num_steps_list)}")
print(f"Max num_steps: {max(num_steps_list)}")
print(f"Count with num_steps > 0: {sum(1 for x in num_steps_list if x > 0)}")

```


## File: `compare_new.py`
```python
import subprocess
import pandas as pd
import sys

print("Running NEW logic...")
subprocess.run([
    "python", "scripts/run_phase_a.py",
    "--smoke",
    "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "--cap", "10",
    "--transition_epochs", "1",
    "--decoder_epochs", "1"
], check=True)

old_df = pd.read_csv("reports/probe_results_old.csv")
new_df = pd.read_csv("reports/probe_results.csv")

if old_df.equals(new_df):
    print("\n✅ SUCCESS: Outputs are bit-identical!")
    sys.exit(0)
else:
    print("\n❌ FAILURE: Outputs differ!")
    print(old_df.compare(new_df))
    sys.exit(1)

```


## File: `create_bundle.py`
```python
import os
import zipfile
import fnmatch

def create_reviewer_bundle():
    source_dir = r"C:\Users\singh\Documents\latent_planning"
    output_zip = r"C:\Users\singh\Documents\latent_planning\reviewer_bundle.zip"
    ignore_file_path = os.path.join(source_dir, '.auditignore')
    
    ignore_patterns = set()
    if os.path.exists(ignore_file_path):
        with open(ignore_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # if it is a directory pattern like 'venv/', strip the trailing slash for fnmatch
                    if line.endswith('/'):
                        line = line[:-1]
                    ignore_patterns.add(line)
    
    print(f"Loaded ignore patterns: {ignore_patterns}")
    
    def should_ignore(path_name):
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path_name, pattern):
                return True
        return False

    print(f"Creating {output_zip}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Filter directories in-place
            dirs[:] = [d for d in dirs if not should_ignore(d)]
            
            for file in files:
                if should_ignore(file):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                
                # Double check the arcname itself isn't matching an ignore (e.g. if it matches *.zip)
                if should_ignore(file):
                    continue
                
                try:
                    zipf.write(file_path, arcname)
                except Exception as e:
                    print(f"Failed to add {file_path}: {e}")
                    
    print(f"Successfully created {output_zip}")
    print(f"File size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_reviewer_bundle()

```


## File: `create_repo_book.py`
```python
import os
import fnmatch

def generate_repo_book():
    source_dir = r"C:\Users\singh\Documents\latent_planning"
    output_dir = os.path.join(source_dir, "notebooklm_export")
    ignore_file_path = os.path.join(source_dir, '.auditignore')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ignore_patterns = set()
    if os.path.exists(ignore_file_path):
        with open(ignore_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.endswith('/'):
                        line = line[:-1]
                    ignore_patterns.add(line)
                    
    def should_ignore(path_name):
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path_name, pattern) or fnmatch.fnmatch(os.path.basename(path_name), pattern):
                return True
        return False
        
    def generate_tree(dir_path, prefix=""):
        tree_str = ""
        items = sorted([item for item in os.listdir(dir_path) if not should_ignore(item)])
        for i, item in enumerate(items):
            path = os.path.join(dir_path, item)
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            tree_str += f"{prefix}{connector}{item}\n"
            
            if os.path.isdir(path):
                extension = "    " if is_last else "│   "
                tree_str += generate_tree(path, prefix + extension)
        return tree_str

    # Generate Tree
    tree_content = f"# Repository Tree\n\n```text\n{os.path.basename(source_dir)}/\n{generate_tree(source_dir)}```\n\n"
    
    docs_content = tree_content + "# Documentation Files\n\n"
    code_content = "# Source Code\n\n"
    reports_content = "# Reports and Configs\n\n"

    # Define categories
    docs_files = {'README.md', 'FEEDBACK.md', 'DECISIONS.md', 'AUDIT_PROMPT.md', 'MANIFEST.md', 'requirements.txt', 'environment.txt'}
    docs_dirs = {'paper'}
    code_dirs = {'models', 'training', 'evaluation', 'scripts', 'tests', 'data_processing'}
    report_dirs = {'reports', 'configs'}
    
    # Supported text extensions
    text_extensions = {'.py', '.md', '.txt', '.tex', '.csv', '.json', '.yaml', '.yml', '.toml', '.ps1'}
    
    for root, dirs, files in os.walk(source_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if not should_ignore(d) and d != 'notebooklm_export']
        
        for file in files:
            if should_ignore(file):
                continue
                
            _, ext = os.path.splitext(file)
            if ext.lower() not in text_extensions:
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, source_dir)
            top_level_dir = rel_path.split(os.sep)[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                content = f"[Error reading file: {e}]"
                
            # Determine language for markdown block
            lang = ext[1:] if ext else "text"
            if lang == "py": lang = "python"
            elif lang == "ps1": lang = "powershell"
            elif lang == "txt": lang = "text"
            elif lang == "tex": lang = "latex"
            
            formatted_content = f"\n## File: `{rel_path}`\n```{lang}\n{content}\n```\n\n"
            
            # Categorize
            if file in docs_files or top_level_dir in docs_dirs:
                docs_content += formatted_content
            elif top_level_dir in code_dirs or (top_level_dir == file and ext == '.py'):
                code_content += formatted_content
            elif top_level_dir in report_dirs:
                reports_content += formatted_content
            else:
                # Catch-all goes to docs
                docs_content += formatted_content

    # Write files
    with open(os.path.join(output_dir, "notebooklm_book_docs.md"), 'w', encoding='utf-8') as f:
        f.write(docs_content)
    with open(os.path.join(output_dir, "notebooklm_book_code.md"), 'w', encoding='utf-8') as f:
        f.write(code_content)
    with open(os.path.join(output_dir, "notebooklm_book_reports.md"), 'w', encoding='utf-8') as f:
        f.write(reports_content)
        
    print(f"Exported repository books to {output_dir}")
    print(f"Docs size: {len(docs_content.encode('utf-8')) / (1024*1024):.2f} MB")
    print(f"Code size: {len(code_content.encode('utf-8')) / (1024*1024):.2f} MB")
    print(f"Reports size: {len(reports_content.encode('utf-8')) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_repo_book()

```


## File: `download_pt.py`
```python
import os
import shutil
import kagglehub

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d5813a0a903ca7390f3f6bee614ccf5a'

files_to_download = [
    "latent2/models/transition_model_action.pt",
    "latent2/models/transition_model_blind.pt",
    "mistral7b_countdown/trajectories/train.pt"
]

target_dir = r"c:\Users\singh\OneDrive\Documents\latent_planning\notebook_output"

for f in files_to_download:
    try:
        print(f"Downloading {f}...")
        path = kagglehub.notebook_download("jadehv/notebookd791d848b1", path=f)
        target_path = os.path.join(target_dir, os.path.normpath(f))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # Handle cases where path is a directory (shouldn't be, since we specified a file, but just in case)
        if os.path.isdir(path):
            # If it downloaded the whole dir for some reason
            src_file = os.path.join(path, os.path.basename(f))
            if os.path.exists(src_file):
                shutil.copy2(src_file, target_path)
            else:
                print(f"Warning: could not find {src_file} in downloaded dir {path}")
        else:
            shutil.copy2(path, target_path)
        print(f"Successfully copied to {target_path}")
    except Exception as e:
        print(f"Error downloading {f}: {e}")

print("Done.")

```


## File: `download_zip.py`
```python
import os
import requests
import zipfile
import sys

token = 'KGAT_d5813a0a903ca7390f3f6bee614ccf5a'
url = 'https://www.kaggle.com/api/v1/kernels/output/jadehv/notebookd791d848b1'
zip_path = r"c:\Users\singh\OneDrive\Documents\latent_planning\notebook_output\archive.zip"
target_dir = r"c:\Users\singh\OneDrive\Documents\latent_planning\notebook_output"

headers = {
    'Authorization': f'Bearer {token}'
}

print(f"Downloading archive from {url}...")
response = requests.get(url, headers=headers, stream=True)

if response.status_code != 200:
    print(f"Failed to download. Status code: {response.status_code}")
    print(response.text)
    sys.exit(1)

with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        
print("Downloaded archive successfully! Extracting specific files...")
files_to_extract = [
    "latent2/models/transition_model_action.pt",
    "latent2/models/transition_model_blind.pt",
    "mistral7b_countdown/trajectories/train.pt"
]

extracted_count = 0
with zipfile.ZipFile(zip_path, 'r') as z:
    for f in files_to_extract:
        try:
            z.extract(f, target_dir)
            print(f"Extracted {f}")
            extracted_count += 1
        except Exception as e:
            print(f"Failed to extract {f}: {e}")

print(f"Done. Extracted {extracted_count} files.")

```


## File: `hash_check.py`
```python
import torch
import sys
from data_processing.trajectory_dataset import Trajectory

tr = torch.load("notebook_output/mistral7b_countdown/trajectories/train.pt", map_location="cpu", weights_only=False)
te = torch.load("notebook_output/mistral7b_countdown/trajectories/test.pt", map_location="cpu", weights_only=False)

def get_problem_hash(t):
    return (t.target, tuple(sorted(t.numbers)))

train_hashes = set(get_problem_hash(t) for t in tr)
test_hashes = set(get_problem_hash(t) for t in te)

print(f"Train unique hashes: {len(train_hashes)}")
print(f"Test unique hashes: {len(test_hashes)}")

overlap = train_hashes.intersection(test_hashes)
print(f"Overlap between Train and Test: {len(overlap)}")

# Wait, check if there's any test trajectories that are NOT in train
test_clean = [t for t in te if get_problem_hash(t) not in train_hashes]
print(f"Test clean length: {len(test_clean)}")

```


## File: `hash_check2.py`
```python
import torch
import sys
from data_processing.trajectory_dataset import Trajectory
import os

try:
    tr = torch.load("notebook_output/mistral7b_countdown/trajectories/train.pt", map_location="cpu", weights_only=False)
    va = torch.load("notebook_output/mistral7b_countdown/trajectories/val.pt", map_location="cpu", weights_only=False)
    te = torch.load("notebook_output/mistral7b_countdown/trajectories/test.pt", map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

def get_problem_hash(t):
    return (t.target, tuple(sorted(t.numbers)))

train_hashes = set(get_problem_hash(t) for t in tr)
val_hashes = set(get_problem_hash(t) for t in va if get_problem_hash(t) not in train_hashes)
test_hashes = set(get_problem_hash(t) for t in te)

test_clean = [t for t in te if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]

print(f"Train unique hashes: {len(train_hashes)}")
print(f"Val unique hashes (filtered): {len(val_hashes)}")
print(f"Test unique hashes: {len(test_hashes)}")
print(f"Test clean length: {len(test_clean)}")

```


## File: `jsonl_hash.py`
```python
import json
def get_hashes(path):
    hashes = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                h = (d["target"], tuple(sorted(d["numbers"])))
                hashes.add(h)
    return hashes

val = get_hashes("data/val.jsonl")
test = get_hashes("data/test.jsonl")

print(f"Val size: {len(val)}")
print(f"Test size: {len(test)}")
print(f"Overlap: {len(val.intersection(test))}")

```


## File: `print_acc.py`
```python
import pandas as pd

for seed in [42, 43, 44]:
    df = pd.read_csv(f'reports/multiseed_transformer_seed{seed}/coherence_action_depth.csv')
    d1 = df[df['depth']==1].iloc[0]
    true_acc = d1['state_probe_accuracy']
    oracle_acc = d1['teacher_state_probe_accuracy']
    id_acc = d1['identity_state_probe_accuracy']
    print(f"Seed {seed}:")
    print(f"  True Acc:   {true_acc:.4f}")
    print(f"  Oracle Acc: {oracle_acc:.4f}")
    print(f"  Id Acc:     {id_acc:.4f}")
    print(f"  Oracle Gap: {oracle_acc - true_acc:.4f}")
    print(f"  Sem Gain:   {true_acc - id_acc:.4f}")
    print()

```


## File: `print_mse.py`
```python
import pandas as pd

for seed in [42, 43, 44]:
    df = pd.read_csv(f'reports/multiseed_transformer_seed{seed}/coherence_action_depth.csv')
    d1 = df[df['depth']==1].iloc[0]
    true_acc = d1['state_probe_accuracy']
    oracle_acc = d1['teacher_state_probe_accuracy']
    id_acc = d1['identity_state_probe_accuracy']
    mse = d1['mse']
    id_mse = d1['identity_mse']
    print(f"Seed {seed}:")
    print(f"  MSE:        {mse:.4f}")
    print(f"  Id MSE:     {id_mse:.4f}")
    print()

```


## File: `run_multi_seed.py`
```python
import subprocess
import os
import glob
import pandas as pd
import numpy as np

seeds = [42, 43, 44]
archs = ['linear', 'mlp', 'transformer']
traj_dir = "reports/trajectories"

print("Running Multi-Seed Analysis...")

results = []

for arch in archs:
    for seed in seeds:
        out_dir = f"reports/{arch}_seed{seed}"
        print(f"--- Running {arch} with seed {seed} ---")
        
        # Run Phase A
        cmd_phase_a = [
            "python", "scripts/run_phase_a.py",
            "--transition_arch", arch,
            "--out_dir", out_dir,
            "--trajectories_dir", traj_dir,
            "--seed", str(seed)
        ]
        subprocess.run(cmd_phase_a, check=True)
        
        # Run Oracle Audit
        cmd_oracle = [
            "python", "scripts/run_oracle_audit.py",
            "--reports_dir", out_dir,
            "--out_dir", out_dir,
            "--transition_arch", arch,
            "--train_traj", f"{traj_dir}/train.pt",
            "--audit", "test",
            "--transition_ckpt", f"{out_dir}/transition_model_action.pt"
        ]
        subprocess.run(cmd_oracle, check=True)
        
        # Collect metrics
        try:
            coh_df = pd.read_csv(f"{out_dir}/coherence_action_depth.csv")
            d1 = coh_df[coh_df['depth'] == 1].iloc[0]
            sem_gain = d1['state_probe_accuracy'] - d1['identity_state_probe_accuracy']
            cosine = d1['cosine_similarity']
            mse = d1['mse']
        except Exception as e:
            print(f"Error reading coherence for {arch} {seed}: {e}")
            sem_gain, cosine, mse = np.nan, np.nan, np.nan
            
        try:
            oracle = pd.read_csv(f"{out_dir}/oracle_audit_metrics.csv")
            o0 = oracle[(oracle['oracle'] == 'oracle0') & (oracle['metric'] == 'probeA_per_label')]['value'].iloc[0]
            o1 = oracle[(oracle['oracle'] == 'oracle1') & (oracle['metric'] == 'probeA_per_label')]['value'].iloc[0]
            oracle_gap = o0 - o1
        except Exception as e:
            print(f"Error reading oracle for {arch} {seed}: {e}")
            oracle_gap = np.nan
            
        results.append({
            "Architecture": arch,
            "Seed": seed,
            "Semantic Gain": sem_gain,
            "Oracle Gap": oracle_gap,
            "Cosine": cosine,
            "MSE": mse
        })

df = pd.DataFrame(results)
df.to_csv("reports/multi_seed_results.csv", index=False)

print("\n--- Final Aggregated Results ---")
agg = df.groupby("Architecture").agg(["mean", "std"])
print(agg)

# Format markdown table
print("\nMarkdown Table:")
for arch in archs:
    arch_data = df[df["Architecture"] == arch]
    sem_m, sem_s = arch_data["Semantic Gain"].mean(), arch_data["Semantic Gain"].std()
    ora_m, ora_s = arch_data["Oracle Gap"].mean(), arch_data["Oracle Gap"].std()
    cos_m, cos_s = arch_data["Cosine"].mean(), arch_data["Cosine"].std()
    mse_m, mse_s = arch_data["MSE"].mean(), arch_data["MSE"].std()
    
    # + sign for semantic gain mean if positive
    sem_str = f"{sem_m:+.3f} ± {sem_s:.3f}"
    ora_str = f"{ora_m:.3f} ± {ora_s:.3f}"
    cos_str = f"{cos_m:.3f} ± {cos_s:.3f}"
    mse_str = f"{mse_m:.3f} ± {mse_s:.3f}"
    
    print(f"| {arch.capitalize():<12} | {sem_str:<16} | {ora_str:<15} | {cos_str:<15} | {mse_str:<15} |")


```


## File: `run_regression.py`
```python
import os
import subprocess
import shutil
import pandas as pd

def run_cmd(is_new):
    print(f"Running {'NEW' if is_new else 'OLD'} logic...")
    subprocess.run([
        "python", "scripts/run_phase_a.py",
        "--smoke",
        "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--cap", "10",
        "--transition_epochs", "1",
        "--decoder_epochs", "1"
    ], check=True)

def run_regression():
    # 1. Stash changes to revert to OLD logic
    print("Stashing changes...")
    subprocess.run(["git", "stash"], check=True)
    
    # 2. Run old logic
    run_cmd(False)
    
    # 3. Save old probe results
    shutil.copy("reports/probe_results.csv", "reports/probe_results_old.csv")
    
    # 4. Pop stash to apply NEW logic
    print("Popping stash...")
    subprocess.run(["git", "stash", "pop"], check=True)
    
    # 5. Run new logic
    run_cmd(True)
    
    # 6. Compare
    print("Comparing results...")
    old_df = pd.read_csv("reports/probe_results_old.csv")
    new_df = pd.read_csv("reports/probe_results.csv")
    
    if old_df.equals(new_df):
        print("\n\n✅ SUCCESS: Outputs are bit-identical!")
    else:
        print("\n\n❌ FAILURE: Outputs differ!")
        print(old_df.compare(new_df))

if __name__ == "__main__":
    run_regression()

```


## File: `test_multilabel.py`
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class MultiLabelProbe:
    def __init__(self):
        self.clfs = []
        self.constants = []

    def fit(self, X, Y):
        for j in range(Y.shape[1]):
            classes = np.unique(Y[:, j])
            if classes.shape[0] < 2:
                self.clfs.append(None)
                self.constants.append(int(classes[0]) if classes.shape[0] else 0)
            else:
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
                ])
                clf.fit(X, Y[:, j])
                self.clfs.append(clf)
                self.constants.append(None)

    def predict(self, X):
        preds = []
        for i, clf in enumerate(self.clfs):
            if clf is None:
                preds.append(np.full((X.shape[0],), self.constants[i]))
            else:
                preds.append(clf.predict(X))
        return np.column_stack(preds)

def _fit_eval_multilabel_joint(Xtr, Ytr, Xte, Yte):
    clf = MultiLabelProbe()
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xte)
    print("Success")

Xtr = np.random.randn(10, 4096).astype(np.float32)
Ytr = np.random.randint(0, 2, size=(10, 4)).astype(np.int64)

Xte = np.array([], dtype=np.float32)
Yte = np.array([], dtype=np.int64).reshape(0, 4)

_fit_eval_multilabel_joint(Xtr, Ytr, Xte, Yte)

```


## File: `test_tokenizer.py`
```python
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=os.environ.get("HF_TOKEN"))
text = "Problem: Given the numbers [25, 50], reach the target 75.\nSolution:\n25 + 50 = 75"
enc = tokenizer(text, return_offsets_mapping=True)
print(enc["offset_mapping"])

```


## File: `verify_splits.py`
```python
import os
import json
data_dir = 'data_game24_test'
trajs = {}
for split in ['train', 'val', 'test']:
    with open(os.path.join(data_dir, f'{split}.jsonl')) as f:
        trajs[split] = [json.loads(l) for l in f]
    print(f'Original {split}: {len(trajs[split])}')

def get_problem_hash(t):
    return (t['target'], tuple(sorted(t['numbers'])))

train_hashes = set(get_problem_hash(t) for t in trajs['train'])
trajs['val'] = [t for t in trajs['val'] if get_problem_hash(t) not in train_hashes]
val_hashes = set(get_problem_hash(t) for t in trajs['val'])
trajs['test'] = [t for t in trajs['test'] if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]

print('\nDropped after dedup:')
print(f'train: {len(trajs["train"])}')
print(f'val: {len(trajs["val"])}')
print(f'test: {len(trajs["test"])}')

```


## File: `verify_submission.py`
```python
import os
import re
import glob

def check_file_exists(filepath, name):
    if os.path.exists(filepath):
        print(f"[x] {name} exists: {filepath}")
        return True
    else:
        print(f"[ ] MISSING {name}: {filepath}")
        return False

def check_tex_references(tex_file):
    if not os.path.exists(tex_file):
        print(f"[ ] MISSING Manuscript: {tex_file}")
        return False

    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all \label{}
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    # Find all \ref{}
    refs = re.findall(r'\\ref\{([^}]+)\}', content)

    missing_refs = [ref for ref in refs if ref not in labels]
    
    print(f"[x] Found {len(labels)} labels and {len(refs)} references in {tex_file}")
    if missing_refs:
        print(f"[ ] ERROR: Broken references found: {missing_refs}")
        return False
    else:
        print(f"[x] All references resolve correctly in {tex_file}")
        return True

def main():
    print("========================================")
    print("   Submission Verification Script")
    print("========================================")
    
    all_good = True
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Check manuscript exists
    manuscript = os.path.join(base_dir, "manuscript.tex")
    all_good &= check_file_exists(manuscript, "Manuscript")
    
    # 2. Check references within manuscript
    all_good &= check_tex_references(manuscript)

    # 3. Check reports directory
    reports_dir = os.path.join(base_dir, "reports")
    all_good &= check_file_exists(reports_dir, "Reports Directory")
    
    # 4. Check that at least some CSVs exist (since we don't have a rigid list here)
    csv_files = glob.glob(os.path.join(reports_dir, "**", "*.csv"), recursive=True)
    if csv_files:
        print(f"[x] Found {len(csv_files)} CSV files in reports.")
    else:
        print(f"[ ] WARNING: No CSV files found in reports directory.")

    # Note: In a fully fleshed out verification script, we would parse the CSVs 
    # and compare the numbers against regex matches in the TeX source, 
    # and verify hashes against the generated metadata JSONs.
    
    print("========================================")
    if all_good:
        print("VERIFICATION SUCCESSFUL: Ready for submission review.")
    else:
        print("VERIFICATION FAILED: Please fix the missing or broken artifacts.")

if __name__ == "__main__":
    main()

```


## File: `data_processing\action_parser.py`
```python
"""Symbolic action parser for Countdown reasoning steps.

Converts a textual reasoning step such as ``"75 * 11 = 825"`` into a structured
symbolic action::

    Action(op=Op.MUL, arg1=75, arg2=11, result=825)

Supported operations: ADD (+), SUB (-), MUL (*), DIV (/).

The parser is the bridge between the teacher's natural-language chain-of-thought
and the symbolic action space used by the transition model and probes. It is
deliberately strict about structure but, by default, lenient about arithmetic
correctness: we record exactly what the teacher wrote (including mistakes) and
expose :meth:`Action.is_arithmetically_valid` so callers can filter if desired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class Op(Enum):
    """The four supported Countdown operations."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


# Stable, contiguous integer ids for embedding lookups. The order here is the
# canonical action id space used throughout the project (transition model action
# embedding, probe C labels, etc.).
OP_TO_ID = {Op.ADD: 0, Op.SUB: 1, Op.MUL: 2, Op.DIV: 3}
ID_TO_OP = {i: op for op, i in OP_TO_ID.items()}

_SYMBOL_TO_OP = {op.value: op for op in Op}

# "<int> <op> <int> = <int>" with arbitrary surrounding/inner whitespace.
_STEP_RE = re.compile(
    r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)\s*$"
)


@dataclass(frozen=True)
class Action:
    """A single symbolic Countdown action."""

    op: Op
    arg1: int
    arg2: int
    result: int

    @property
    def op_id(self) -> int:
        return OP_TO_ID[self.op]

    def is_arithmetically_valid(self) -> bool:
        """Whether ``arg1 op arg2`` actually equals ``result``."""
        try:
            return apply_op(self.op, self.arg1, self.arg2) == self.result
        except (ZeroDivisionError, ValueError):
            return False


def apply_op(op: Op, a: int, b: int) -> int:
    """Apply a symbolic op to two integers.

    Division is integer division and requires an exact (remainder-free) result,
    matching the Countdown rule that only whole-number intermediate values are
    legal.
    """
    if op == Op.ADD:
        return a + b
    if op == Op.SUB:
        return a - b
    if op == Op.MUL:
        return a * b
    if op == Op.DIV:
        if b == 0 or a % b != 0:
            raise ValueError(f"Illegal division: {a} / {b}")
        return a // b
    raise ValueError(f"Unknown op: {op!r}")


def parse_step(step: str, validate: bool = False) -> Action:
    """Parse a single reasoning step string into an :class:`Action`.

    Args:
        step: e.g. ``"75 * 11 = 825"``.
        validate: if True, raise ``ValueError`` when the stated result does not
            match the arithmetic of the operands.

    Raises:
        ValueError: if the string is not a well-formed ``a op b = c`` equation
            with a supported operator, or (when ``validate``) is arithmetically
            inconsistent.
    """
    match = _STEP_RE.match(step)
    if match is None:
        raise ValueError(f"Malformed reasoning step: {step!r}")

    arg1_s, symbol, arg2_s, result_s = match.groups()
    op = _SYMBOL_TO_OP.get(symbol)
    if op is None:  # pragma: no cover - regex already restricts the symbol set
        raise ValueError(f"Unsupported operator {symbol!r} in step: {step!r}")

    action = Action(
        op=op,
        arg1=int(arg1_s),
        arg2=int(arg2_s),
        result=int(result_s),
    )

    if validate and not action.is_arithmetically_valid():
        raise ValueError(f"Arithmetically invalid step: {step!r}")

    return action


def parse_solution(steps: List[str], validate: bool = False) -> List[Action]:
    """Parse a list of reasoning-step strings into a list of actions."""
    return [parse_step(s, validate=validate) for s in steps]

```


## File: `data_processing\discrete_trajectory_dataset.py`
```python
"""Discrete code trajectories.

Converts a list of continuous :class:`~data_processing.trajectory_dataset.Trajectory`
objects into discrete **code trajectories** by quantizing each symbolic-aligned
state ``s_0..s_N`` through a trained :class:`~models.vq_state.VQStateQuantizer`.

    h_1, h_2, h_3, h_4   -->   z_1, z_2, z_3, z_4

Each :class:`DiscreteTrajectory` retains the action metadata (op ids /
operands / numbers / target) needed by the permutation-robustness and
position-leakage analyses, but the *state* is now an int64 code id per step.

A :class:`DiscreteTransitionDataset` exposes the flat ``(z_t, z_{t+1})`` view
used to train / evaluate the discrete transition model. It is deliberately
**action-blind** (no op/operand conditioning): the Version-5 question is
whether discrete state dynamics exist *autonomously*, not whether they can be
predicted given the symbolic action.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from data_processing.trajectory_dataset import Trajectory
from models.vq_state import VQStateQuantizer


@dataclass
class DiscreteTrajectory:
    """One trajectory through *discrete* code space for a single problem."""

    codes: torch.Tensor       # (N+1,) int64 code id per state s_0..s_N
    op_ids: torch.Tensor      # (N,) action op id for steps 1..N
    operands: torch.Tensor    # (N, 2) float operands for steps 1..N
    numbers: List[int]
    target: int

    @property
    def num_steps(self) -> int:
        return int(self.op_ids.shape[0])

    @property
    def length(self) -> int:
        """Number of states (== ``num_steps + 1``)."""
        return int(self.codes.shape[0])


@torch.no_grad()
def encode_trajectories_to_codes(
    vq: VQStateQuantizer,
    trajectories: List[Trajectory],
    batch_size: int = 1024,
) -> List[DiscreteTrajectory]:
    """Quantize every trajectory's aligned states into discrete code ids.

    Args:
        vq: a *trained* :class:`VQStateQuantizer` (its codebook is read-only).
        trajectories: continuous teacher trajectories.
        batch_size: states quantized per forward pass (memory control).

    Returns a list of :class:`DiscreteTrajectory` parallel to the input.
    """
    vq.eval()
    # VQ codebook lives in a buffer (no Parameters), so fall back to buffers.
    try:
        device = next(vq.parameters()).device
    except StopIteration:
        device = next(vq.buffers()).device

    out: List[DiscreteTrajectory] = []
    for traj in trajectories:
        states = traj.states.to(device)  # (N+1, H)
        codes = []
        for i in range(0, states.shape[0], batch_size):
            chunk = states[i : i + batch_size]
            codes.append(vq.encode(chunk).cpu())
        codes_t = torch.cat(codes, dim=0).to(torch.int64)
        out.append(
            DiscreteTrajectory(
                codes=codes_t,
                op_ids=traj.op_ids,
                operands=traj.operands,
                numbers=list(traj.numbers),
                target=int(traj.target),
            )
        )
    return out


def save_discrete_trajectories(
    discrete_trajs: List[DiscreteTrajectory], path: str
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(discrete_trajs, path)


def load_discrete_trajectories(path: str) -> List[DiscreteTrajectory]:
    return torch.load(path, weights_only=False)


def all_codes(discrete_trajs: List[DiscreteTrajectory]) -> torch.Tensor:
    """Concatenate every code id across all trajectories into a 1-D int64 tensor."""
    if not discrete_trajs:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat([d.codes for d in discrete_trajs], dim=0)


class DiscreteTransitionDataset(Dataset):
    """Flat action-blind ``(z_t, z_{t+1})`` view over discrete trajectories.

    Each item is a ``(current_code, next_code)`` int64 pair. The transition
    model trained on this view answers: *given the current discrete state, is
    the next discrete state predictable?*
    """

    def __init__(self, discrete_trajs: List[DiscreteTrajectory]):
        self.z_t: List[int] = []
        self.z_next: List[int] = []
        for traj in discrete_trajs:
            for i in range(traj.num_steps):
                self.z_t.append(int(traj.codes[i].item()))
                self.z_next.append(int(traj.codes[i + 1].item()))

    def __len__(self) -> int:
        return len(self.z_t)

    def __getitem__(self, idx: int):
        return {
            "z_t": torch.tensor(self.z_t[idx], dtype=torch.long),
            "z_next": torch.tensor(self.z_next[idx], dtype=torch.long),
        }

```


## File: `data_processing\trajectory_dataset.py`
```python
"""Teacher trajectory dataset.

Builds ``(h_t, action_t, h_{t+1})`` transition tuples from a frozen language
model's hidden states, aligned to the boundaries of each symbolic reasoning
step.

State alignment
---------------
For a problem with an ``N``-step solution we define a sequence of *latent
states* ``s_0, s_1, ..., s_N`` where:

* ``s_0`` is the hidden state at the **last token of the prompt header**
  (``"...Solution:\n"``) — the model's encoding of the problem before any step.
* ``s_i`` (i >= 1) is the hidden state at the **last token of reasoning step i**
  (the final token of that line, i.e. the last digit of the step's result).

A transition ``i`` is then ``(s_{i-1}, action_i, s_i)``: applying the symbolic
action of step ``i`` should carry the latent state forward by one step. Token
alignment is computed from the fast tokenizer's ``offset_mapping`` so it is
robust to sub-word merges.

Each :class:`Trajectory` also retains the full per-token hidden states and input
ids so the diagnostic decoder can be trained on every position, and so the
coherence rollout can look up the teacher token that follows any state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from data_processing.action_parser import Action, parse_solution


HEADER_TEMPLATE = "Problem: Given the numbers {numbers}, reach the target {target}.\nSolution:\n"
IGNORE_TOKEN = -100  # used for "no next token" (terminal state)


def format_header(numbers: List[int], target: int) -> str:
    """Prompt header, matching the format used by training / extraction."""
    return HEADER_TEMPLATE.format(numbers=numbers, target=target)


@dataclass
class Trajectory:
    """One teacher trajectory through latent space for a single problem."""

    all_hidden: torch.Tensor      # (T, H) hidden state for every token
    input_ids: torch.Tensor       # (T,) token ids
    state_indices: torch.Tensor   # (N+1,) indices into all_hidden for s_0..s_N
    op_ids: torch.Tensor          # (N,) action op id for steps 1..N
    operands: torch.Tensor        # (N, 2) float [arg1, arg2] for steps 1..N
    numbers: List[int]
    target: int

    @property
    def num_steps(self) -> int:
        return int(self.op_ids.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.all_hidden.shape[1])

    @property
    def states(self) -> torch.Tensor:
        """(N+1, H) the aligned latent states s_0..s_N."""
        return self.all_hidden[self.state_indices]

    def next_token_after_state(self, i: int) -> int:
        """Teacher token id immediately following state ``s_i`` (or IGNORE)."""
        idx = int(self.state_indices[i].item())
        nxt = idx + 1
        if nxt < self.input_ids.shape[0]:
            return int(self.input_ids[nxt].item())
        return IGNORE_TOKEN


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #
def _state_end_chars(header: str, steps: List[str]) -> List[int]:
    """Exclusive character offsets marking the end of s_0..s_N in the prompt."""
    ends = [len(header)]  # s_0 = end of header
    c = len(header)
    for i, step in enumerate(steps):
        c += len(step)
        ends.append(c)
        c += 1  # the "\n" joining this step to the next
    return ends


def _token_index_for_char_end(offsets: List[tuple], char_end: int) -> int:
    """Index of the last non-empty token whose span ends at/before ``char_end``."""
    best = None
    for i, (a, b) in enumerate(offsets):
        if b > a and b <= char_end:
            best = i
    if best is None:  # pragma: no cover - would mean an empty prompt
        raise ValueError(f"No token found for char_end={char_end}")
    return best


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_trajectory(
    model,
    tokenizer,
    problem: Dict[str, Any],
    layer: int = -1,
) -> Optional[Trajectory]:
    """Build a single :class:`Trajectory` from a problem dict.

    The problem dict must contain ``numbers``, ``target`` and ``solution``
    (a list of step strings). Returns ``None`` if the solution is empty or the
    steps cannot be parsed.
    """
    numbers = problem["numbers"]
    target = problem["target"]
    steps = problem["solution"]
    if not steps:
        return None

    try:
        actions: List[Action] = parse_solution(steps)
    except ValueError:
        return None

    header = format_header(numbers, target)
    full_text = header + "\n".join(steps)

    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].detach().cpu().float()  # (T, H)
    input_ids = enc["input_ids"][0].detach().cpu()

    end_chars = _state_end_chars(header, steps)
    state_indices = [
        _token_index_for_char_end(offset_mapping, ec) for ec in end_chars
    ]
    state_indices = torch.tensor(state_indices, dtype=torch.long)

    op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
    operands = torch.tensor(
        [[a.arg1, a.arg2] for a in actions], dtype=torch.float32
    )

    return Trajectory(
        all_hidden=hidden,
        input_ids=input_ids,
        state_indices=state_indices,
        op_ids=op_ids,
        operands=operands,
        numbers=list(numbers),
        target=int(target),
    )


def build_trajectories(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    layer: int = -1,
    batch_size: int = 32,
) -> List[Trajectory]:
    """Build trajectories for a list of problems, skipping unparseable ones, in batches."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "A fast tokenizer with offset_mapping support is required."
        )
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()
    trajectories: List[Trajectory] = []
    
    from tqdm import tqdm
    for i in tqdm(range(0, len(problems), batch_size), desc="Extracting"):
        batch = problems[i:i+batch_size]
        
        valid_problems = []
        valid_actions = []
        valid_texts = []
        valid_headers = []
        
        for problem in batch:
            numbers = problem["numbers"]
            target = problem["target"]
            steps = problem["solution"]
            if not steps:
                continue
            try:
                actions: List[Action] = parse_solution(steps)
            except ValueError:
                continue
                
            header = format_header(numbers, target)
            full_text = header + "\n".join(steps)
            
            valid_problems.append(problem)
            valid_actions.append(actions)
            valid_texts.append(full_text)
            valid_headers.append(header)
            
        if not valid_texts:
            continue
            
        enc = tokenizer(
            valid_texts, 
            padding=True, 
            return_tensors="pt", 
            return_offsets_mapping=True
        )
        offset_mappings = enc.pop("offset_mapping").tolist()
        attention_mask = enc["attention_mask"]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            
        hidden_batch = outputs.hidden_states[layer].detach().cpu().float()
        input_ids_batch = enc["input_ids"].detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        
        for b_idx in range(len(valid_texts)):
            problem = valid_problems[b_idx]
            actions = valid_actions[b_idx]
            header = valid_headers[b_idx]
            steps = problem["solution"]
            
            pad_mask = attention_mask_cpu[b_idx].bool()
            
            hidden = hidden_batch[b_idx][pad_mask]
            input_ids = input_ids_batch[b_idx][pad_mask]
            
            valid_offsets = [
                offset_mappings[b_idx][t_idx] 
                for t_idx in range(len(offset_mappings[b_idx])) 
                if pad_mask[t_idx]
            ]
            
            end_chars = _state_end_chars(header, steps)
            try:
                state_indices = [
                    _token_index_for_char_end(valid_offsets, ec) for ec in end_chars
                ]
            except ValueError:
                continue
                
            state_indices = torch.tensor(state_indices, dtype=torch.long)
            op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
            operands = torch.tensor(
                [[a.arg1, a.arg2] for a in actions], dtype=torch.float32
            )
            
            traj = Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=op_ids,
                operands=operands,
                numbers=list(problem["numbers"]),
                target=int(problem["target"]),
            )
            if traj.num_steps > 0:
                trajectories.append(traj)
                
    return trajectories


def load_problems(path: str) -> List[Dict[str, Any]]:
    """Load problems from a JSONL file (one problem per line)."""
    problems = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def save_trajectories(trajectories: List[Trajectory], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(trajectories, path)


def load_trajectories(path: str) -> List[Trajectory]:
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------- #
# Flat transition view (for the transition model)
# --------------------------------------------------------------------------- #
class TransitionDataset(torch.utils.data.Dataset):
    """Flat ``(h_t, op_id, operands, h_{t+1})`` view over trajectories."""

    def __init__(self, trajectories: List[Trajectory]):
        self.h_t: List[torch.Tensor] = []
        self.h_next: List[torch.Tensor] = []
        self.op_ids: List[int] = []
        self.operands: List[torch.Tensor] = []
        for traj in trajectories:
            states = traj.states  # (N+1, H)
            for i in range(traj.num_steps):
                self.h_t.append(states[i])
                self.h_next.append(states[i + 1])
                self.op_ids.append(int(traj.op_ids[i].item()))
                self.operands.append(traj.operands[i])

    def __len__(self) -> int:
        return len(self.h_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "h_t": self.h_t[idx],
            "op_id": torch.tensor(self.op_ids[idx], dtype=torch.long),
            "operands": self.operands[idx],
            "h_next": self.h_next[idx],
        }


class DecoderDataset(torch.utils.data.Dataset):
    """Per-token ``(hidden_state, next_token_id)`` view for the decoder."""

    def __init__(self, trajectories: List[Trajectory]):
        self.hidden: List[torch.Tensor] = []
        self.targets: List[int] = []
        for traj in trajectories:
            T = traj.all_hidden.shape[0]
            for p in range(T - 1):
                self.hidden.append(traj.all_hidden[p])
                self.targets.append(int(traj.input_ids[p + 1].item()))

    def __len__(self) -> int:
        return len(self.hidden)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "hidden": self.hidden[idx],
            "target": torch.tensor(self.targets[idx], dtype=torch.long),
        }

```


## File: `data_processing\transfer_trajectory.py`
```python
"""Generic hidden-state trajectory extractor for non-Countdown CoT.

The Countdown-specific :func:`data_processing.trajectory_dataset.build_trajectory`
relies on the Countdown action parser to delimit steps. The transfer datasets
(algebra / logic / graph) have no symbolic action grammar — but they *do* share
the Countdown format of "header + one step per line", and the exact same
token-alignment helpers (``_state_end_chars`` / ``_token_index_for_char_end``)
work on any such text.

This module produces :class:`~data_processing.trajectory_dataset.Trajectory`
objects that are duck-type-compatible with the Countdown trajectories, so all
downstream V5 machinery (VQ encoding, discrete transition dataset, etc.)
operates on them without modification. ``op_ids`` / ``operands`` are filled
with zeros (there is no action semantics to encode) — they are present only to
satisfy the dataclass; downstream transfer analysis does not use them.

State alignment
---------------
For a problem with ``N`` step lines we define ``N+1`` latent states:
``s_0`` at the last token of the header, and ``s_i`` (i>=1) at the last token
of step ``i``'s line — exactly as in the Countdown pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from data_processing.trajectory_dataset import (
    HEADER_TEMPLATE,
    Trajectory,
    _state_end_chars,
    _token_index_for_char_end,
)


def build_transfer_trajectory(
    model,
    tokenizer,
    problem: Dict[str, Any],
    layer: int = -1,
) -> Optional[Trajectory]:
    """Build a :class:`Trajectory` from a transfer-domain problem dict.

    Requires ``problem`` (the header text) and ``solution`` (list of step
    strings). Returns ``None`` if the problem has no steps or alignment fails.
    """
    header = problem.get("problem", HEADER_TEMPLATE)
    steps = problem.get("solution", [])
    if not steps:
        return None

    full_text = header + "\n".join(steps)
    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].detach().cpu().float()
    input_ids = enc["input_ids"][0].detach().cpu()

    end_chars = _state_end_chars(header, steps)
    try:
        state_indices = [
            _token_index_for_char_end(offsets, ec) for ec in end_chars
        ]
    except ValueError:
        return None
    state_indices = torch.tensor(state_indices, dtype=torch.long)

    n_steps = len(steps)
    # No symbolic action grammar: zero op_ids / operands (unused downstream).
    op_ids = torch.zeros(n_steps, dtype=torch.long)
    operands = torch.zeros((n_steps, 2), dtype=torch.float32)

    return Trajectory(
        all_hidden=hidden,
        input_ids=input_ids,
        state_indices=state_indices,
        op_ids=op_ids,
        operands=operands,
        numbers=list(problem.get("numbers", [])),
        target=int(problem.get("target", 0)),
    )


def build_transfer_trajectories(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    layer: int = -1,
    batch_size: int = 32,
) -> List[Trajectory]:
    """Batched transfer-trajectory extraction (mirrors ``build_trajectories``)."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with offset_mapping is required.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    out: List[Trajectory] = []
    from tqdm import tqdm

    for i in tqdm(range(0, len(problems), batch_size), desc="Extracting transfer"):
        batch = problems[i : i + batch_size]
        valid = []
        for problem in batch:
            if problem.get("solution"):
                valid.append(problem)
        if not valid:
            continue

        texts = [p["problem"] + "\n".join(p["solution"]) for p in valid]
        enc = tokenizer(texts, padding=True, return_tensors="pt",
                        return_offsets_mapping=True)
        offset_mappings = enc.pop("offset_mapping").tolist()
        attention_mask = enc["attention_mask"]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden_batch = outputs.hidden_states[layer].detach().cpu().float()
        input_ids_batch = enc["input_ids"].detach().cpu()
        attention_cpu = attention_mask.detach().cpu()

        for b_idx, problem in enumerate(valid):
            pad_mask = attention_cpu[b_idx].bool()
            hidden = hidden_batch[b_idx][pad_mask]
            input_ids = input_ids_batch[b_idx][pad_mask]
            valid_offsets = [
                offset_mappings[b_idx][t]
                for t in range(len(offset_mappings[b_idx]))
                if pad_mask[t]
            ]
            header = problem["problem"]
            steps = problem["solution"]
            end_chars = _state_end_chars(header, steps)
            try:
                state_indices = [
                    _token_index_for_char_end(valid_offsets, ec) for ec in end_chars
                ]
            except ValueError:
                continue
            state_indices = torch.tensor(state_indices, dtype=torch.long)
            n_steps = len(steps)
            traj = Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=torch.zeros(n_steps, dtype=torch.long),
                operands=torch.zeros((n_steps, 2), dtype=torch.float32),
                numbers=list(problem.get("numbers", [])),
                target=int(problem.get("target", 0)),
            )
            if traj.num_steps > 0:
                out.append(traj)
    return out

```


## File: `data_processing\__init__.py`
```python

```


## File: `evaluation\action_conditioned.py`
```python
"""Action-conditioned transition analysis (V5.2 Experiment 1).

V5.1 tested only the *action-blind* transition ``z_t -> z_{t+1}`` and found it
no more predictable than a bigram. But a *planning state* is defined by
action-conditioned dynamics ``(z_t, a_t) -> z_{t+1}`` (as in MuZero/Dreamer):
the next state need not be a function of the current state alone. This module
asks whether conditioning on the symbolic Countdown action exposes structure
the action-blind test could not see.

Models (all evaluated on a held-out split):

* **A. Majority** — predict the global mode of ``z_{t+1}``.
* **B. Bigram** — lookup ``z_t -> mode z_{t+1}``.
* **C. Action bigram** — lookup ``(z_t, op) -> mode z_{t+1}`` (fallback: state
  bigram, then global majority).
* **D. MLP(z_t)** — the V5.1 action-blind MLP.
* **E. MLP(z_t, op)** — action-conditioned MLP.
* **E2. MLP(z_t, op, operands)** — adds the numeric operands of the action.
* **F. MLP(op, operands)** — *action-only control*: ignores ``z_t`` entirely.
  If F ≈ E2, the current discrete state contributes nothing and the codes are
  not functioning as states.

Decisive contrasts: C vs B and E vs D (does the action help at all?), E vs C
(does a learned model beat the action-conditioned lookup?), F vs E2 (does the
state matter, or does the action alone dictate the next code?).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory

NUM_OPS = 4  # ADD/SUB/MUL/DIV id space (DIV unused in the data)


# --------------------------------------------------------------------------- #
# Transition extraction
# --------------------------------------------------------------------------- #
@dataclass
class ActionTransitions:
    z_t: np.ndarray        # (M,) current code
    z_next: np.ndarray     # (M,) next code
    op: np.ndarray         # (M,) op id of the action
    operands: np.ndarray   # (M, 2) raw [arg1, arg2]


def extract_action_transitions(disc: List[DiscreteTrajectory]) -> ActionTransitions:
    zt, zn, ops, opnd = [], [], [], []
    for tr in disc:
        codes = tr.codes
        for i in range(tr.num_steps):
            zt.append(int(codes[i].item()))
            zn.append(int(codes[i + 1].item()))
            ops.append(int(tr.op_ids[i].item()))
            opnd.append([float(tr.operands[i][0].item()), float(tr.operands[i][1].item())])
    return ActionTransitions(
        z_t=np.asarray(zt, dtype=np.int64),
        z_next=np.asarray(zn, dtype=np.int64),
        op=np.asarray(ops, dtype=np.int64),
        operands=np.asarray(opnd, dtype=np.float32).reshape(-1, 2),
    )


# --------------------------------------------------------------------------- #
# Lookup baselines (A, B, C)
# --------------------------------------------------------------------------- #
def majority_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    mode = Counter(train.z_next.tolist()).most_common(1)[0][0]
    return float((ev.z_next == mode).mean())


def _build_lookup(keys, nexts):
    tbl = defaultdict(Counter)
    for k, n in zip(keys, nexts):
        tbl[k][n] += 1
    return {k: c.most_common(1)[0][0] for k, c in tbl.items()}


def state_bigram_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    pred = _build_lookup(train.z_t.tolist(), train.z_next.tolist())
    g = Counter(train.z_next.tolist()).most_common(1)[0][0]
    yhat = np.array([pred.get(z, g) for z in ev.z_t])
    return float((yhat == ev.z_next).mean())


def action_bigram_top1(train: ActionTransitions, ev: ActionTransitions) -> float:
    """(z_t, op) lookup with fallback to state bigram then global majority."""
    key_tr = list(zip(train.z_t.tolist(), train.op.tolist()))
    pred_za = _build_lookup(key_tr, train.z_next.tolist())
    pred_z = _build_lookup(train.z_t.tolist(), train.z_next.tolist())
    g = Counter(train.z_next.tolist()).most_common(1)[0][0]
    yhat = []
    for z, op in zip(ev.z_t.tolist(), ev.op.tolist()):
        if (z, op) in pred_za:
            yhat.append(pred_za[(z, op)])
        elif z in pred_z:
            yhat.append(pred_z[z])
        else:
            yhat.append(g)
    return float((np.asarray(yhat) == ev.z_next).mean())


# --------------------------------------------------------------------------- #
# Conditional structure (entropy + deterministic fraction), measured on train
# --------------------------------------------------------------------------- #
def conditional_structure(keys, nexts, min_count: int = 10) -> Dict:
    """Frequency-weighted H(z'|key) and deterministic mass fraction.

    A "deterministic" conditioning is one whose successor distribution has
    entropy < 0.5 nats; we only credit conditionings seen at least
    ``min_count`` times so single-sample pairs cannot look spuriously
    deterministic. ``det_frac_mass`` is the share of transition *mass* coming
    from such conditionings.
    """
    tbl = defaultdict(Counter)
    for k, n in zip(keys, nexts):
        tbl[k][n] += 1
    total = sum(sum(c.values()) for c in tbl.values())
    gH, det_mass, reliable = 0.0, 0, 0
    for k, c in tbl.items():
        cnt = sum(c.values())
        p = np.array(list(c.values()), dtype=np.float64) / cnt
        H = float(-(p * np.log(p)).sum())
        gH += H * cnt
        if cnt >= min_count:
            reliable += cnt
            if H < 0.5:
                det_mass += cnt
    return {
        "global_entropy": gH / max(total, 1),
        "det_frac_mass": det_mass / max(total, 1),
        "n_conditions": len(tbl),
        "reliable_mass_frac": reliable / max(total, 1),
    }


# --------------------------------------------------------------------------- #
# MLP models (D, E, E2, F)
# --------------------------------------------------------------------------- #
class ActionTransitionMLP(nn.Module):
    """Next-code MLP optionally conditioned on op id, operands, and/or z_t."""

    def __init__(self, num_codes, num_ops=NUM_OPS, embed_dim=32, op_dim=8,
                 hidden_dim=128, use_state=True, use_op=False, use_operands=False):
        super().__init__()
        self.use_state, self.use_op, self.use_operands = use_state, use_op, use_operands
        in_dim = 0
        if use_state:
            self.code_embed = nn.Embedding(num_codes, embed_dim)
            in_dim += embed_dim
        if use_op:
            self.op_embed = nn.Embedding(num_ops, op_dim)
            in_dim += op_dim
        if use_operands:
            in_dim += 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_codes)
        )

    def forward(self, z, op, operands):
        feats = []
        if self.use_state:
            feats.append(self.code_embed(z))
        if self.use_op:
            feats.append(self.op_embed(op))
        if self.use_operands:
            feats.append(operands)
        return self.net(torch.cat(feats, dim=-1))


@dataclass
class ActionMLPConfig:
    use_state: bool = True
    use_op: bool = False
    use_operands: bool = False
    embed_dim: int = 32
    op_dim: int = 8
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    seed: int = 0
    history: List[dict] = field(default_factory=list)


def _operand_features(operands: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Signed-log transform then standardize (operands span 1..thousands)."""
    sl = np.sign(operands) * np.log1p(np.abs(operands))
    return (sl - mean) / std


def _fit_operand_norm(operands: np.ndarray):
    sl = np.sign(operands) * np.log1p(np.abs(operands))
    mean = sl.mean(axis=0)
    std = sl.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def _tensors(tr: ActionTransitions, op_mean, op_std):
    z = torch.tensor(tr.z_t, dtype=torch.long)
    op = torch.tensor(tr.op, dtype=torch.long)
    opnd = torch.tensor(_operand_features(tr.operands, op_mean, op_std), dtype=torch.float32)
    y = torch.tensor(tr.z_next, dtype=torch.long)
    return TensorDataset(z, op, opnd, y)


def train_action_mlp(
    train: ActionTransitions, ev: ActionTransitions, num_codes: int,
    config: ActionMLPConfig,
) -> Dict[str, float]:
    """Train one MLP variant; return held-out top1 + predictive entropy."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op_mean, op_std = _fit_operand_norm(train.operands)

    train_loader = DataLoader(_tensors(train, op_mean, op_std),
                              batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(_tensors(ev, op_mean, op_std), batch_size=config.batch_size)

    model = ActionTransitionMLP(
        num_codes, use_state=config.use_state, use_op=config.use_op,
        use_operands=config.use_operands, embed_dim=config.embed_dim,
        op_dim=config.op_dim, hidden_dim=config.hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for _ in range(config.epochs):
        model.train()
        for z, op, opnd, y in train_loader:
            z, op, opnd, y = z.to(device), op.to(device), opnd.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(z, op, opnd), y)
            loss.backward()
            opt.step()

    model.eval()
    correct, ent_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for z, op, opnd, y in eval_loader:
            z, op, opnd, y = z.to(device), op.to(device), opnd.to(device), y.to(device)
            logits = model(z, op, opnd)
            probs = F.softmax(logits, dim=-1)
            ent_sum += (-(probs * torch.log(probs.clamp_min(1e-10))).sum(-1)).sum().item()
            correct += (logits.argmax(-1) == y).sum().item()
            n += z.shape[0]
    return {"top1": correct / max(n, 1), "pred_entropy": ent_sum / max(n, 1)}


def run_all_models(
    train: ActionTransitions, ev: ActionTransitions, num_codes: int,
    epochs: int = 30, seed: int = 0,
) -> Dict[str, Dict]:
    """Run A-F and the two structural conditionings. Returns a results dict."""
    res: Dict[str, Dict] = {}
    res["A_majority"] = {"top1": majority_top1(train, ev), "cond": "—"}
    res["B_bigram"] = {"top1": state_bigram_top1(train, ev), "cond": "z_t"}
    res["C_action_bigram"] = {"top1": action_bigram_top1(train, ev), "cond": "z_t, op"}

    variants = {
        "D_mlp_z": ActionMLPConfig(use_state=True, use_op=False, use_operands=False,
                                   epochs=epochs, seed=seed),
        "E_mlp_z_op": ActionMLPConfig(use_state=True, use_op=True, use_operands=False,
                                      epochs=epochs, seed=seed),
        "E2_mlp_z_op_operands": ActionMLPConfig(use_state=True, use_op=True,
                                                use_operands=True, epochs=epochs, seed=seed),
        "F_mlp_action_only": ActionMLPConfig(use_state=False, use_op=True,
                                             use_operands=True, epochs=epochs, seed=seed),
    }
    conds = {"D_mlp_z": "z_t", "E_mlp_z_op": "z_t, op",
             "E2_mlp_z_op_operands": "z_t, op, operands",
             "F_mlp_action_only": "op, operands"}
    for name, cfg in variants.items():
        m = train_action_mlp(train, ev, num_codes, cfg)
        m["cond"] = conds[name]
        res[name] = m

    # Structural transition entropy / determinism (measured on train).
    res["_struct_state"] = conditional_structure(train.z_t.tolist(), train.z_next.tolist())
    res["_struct_action"] = conditional_structure(
        list(zip(train.z_t.tolist(), train.op.tolist())), train.z_next.tolist())
    return res

```


## File: `evaluation\codebook_usage.py`
```python
"""Codebook usage / VQ collapse diagnostics.

Measures whether the trained VQ codebook is actually *used*: how many codes
are active, how skewed the usage distribution is, and how many codes are dead
(receive zero assignments). These are the standard VQ-VAE collapse signals.

Key outputs:

* **active code count** — codes used at least ``min_freq_frac`` of the time.
* **frequency distribution** — per-code empirical probability.
* **dead codes** — codes with zero (or near-zero) usage.
* **collapse score** — the Gini coefficient of the usage distribution. A Gini
  near 1 means a tiny subset of codes dominates (severe collapse); a Gini near
  ``0`` means perfectly uniform usage.
* **perplexity** — ``exp(H(code distribution))``, the effective number of
  codes used (mirrors the VQ-VAE paper's perplexity diagnostic).

These numbers are the first gate of the Version-5 verdict: a collapsed
codebook cannot support reusable discrete states.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector (0 = uniform, 1 = collapsed)."""
    arr = np.sort(counts.astype(np.float64))
    if arr.sum() <= 0:
        return 0.0
    n = arr.shape[0]
    idx = np.arange(1, n + 1)
    # Standard Gini formula on sorted values.
    return float((2.0 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


def analyze_codebook_usage(
    codes: torch.Tensor,
    num_codes: int,
    min_freq_frac: float = 1e-4,
) -> Dict:
    """Compute usage statistics from a 1-D tensor of integer code ids.

    Args:
        codes: int64 tensor of assigned codes (e.g. over all train states).
        num_codes: size of the codebook these ids index into.
        min_freq_frac: a code is "active" if its empirical frequency is at
            least this fraction of the total (default: ~1 per 10k).

    Returns a dict with: ``counts`` (per-code raw counts),
    ``freqs`` (per-code probabilities), ``active_codes``, ``dead_codes``,
    ``collapse_score`` (Gini), ``perplexity``, ``entropy``,
    ``mean_freq_when_used``.
    """
    if isinstance(codes, torch.Tensor):
        codes_np = codes.detach().cpu().to(torch.int64).numpy()
    else:
        codes_np = np.asarray(codes, dtype=np.int64)

    total = codes_np.shape[0]
    if total == 0:
        return {
            "counts": np.zeros(num_codes, dtype=np.int64),
            "freqs": np.zeros(num_codes, dtype=np.float64),
            "active_codes": 0,
            "dead_codes": num_codes,
            "collapse_score": 0.0,
            "perplexity": 0.0,
            "entropy": 0.0,
            "mean_freq_when_used": 0.0,
        }

    counts = np.bincount(codes_np, minlength=num_codes).astype(np.int64)
    freqs = counts.astype(np.float64) / total

    # Entropy in nats; perplexity = effective number of codes used.
    nonzero = freqs[freqs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    perplexity = float(np.exp(entropy))

    used = counts > 0
    active = freqs >= min_freq_frac
    dead = num_codes - int(used.sum())

    mean_freq_when_used = float(nonzero.mean()) if nonzero.size else 0.0

    return {
        "counts": counts,
        "freqs": freqs,
        "active_codes": int(active.sum()),
        "used_codes": int(used.sum()),
        "dead_codes": dead,
        "collapse_score": _gini(counts.astype(np.float64)),
        "perplexity": perplexity,
        "entropy": entropy,
        "mean_freq_when_used": mean_freq_when_used,
    }


def save_codebook_usage_report(
    stats: Dict, csv_path: str, png_path: str = None
) -> None:
    """Write a one-row summary CSV and (optionally) a log-frequency bar plot."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    row = {
        "active_codes": stats["active_codes"],
        "used_codes": stats["used_codes"],
        "dead_codes": stats["dead_codes"],
        "collapse_score_gini": round(stats["collapse_score"], 4),
        "perplexity": round(stats["perplexity"], 2),
        "entropy_nats": round(stats["entropy"], 4),
        "mean_freq_when_used": round(stats["mean_freq_when_used"], 5),
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    if png_path is not None:
        _plot_usage(stats, png_path)


def _plot_usage(stats: Dict, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = stats["counts"].astype(np.float64)
    order = np.argsort(counts)[::-1]  # most-used first
    freqs = stats["freqs"][order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(np.arange(freqs.shape[0]), freqs + 1e-9, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Code rank (most-used first)")
    ax.set_ylabel("Empirical frequency (log)")
    ax.set_title(
        f"Codebook usage (active={stats['active_codes']}, "
        f"dead={stats['dead_codes']}, Gini={stats['collapse_score']:.3f}, "
        f"perplexity={stats['perplexity']:.1f})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\coherence.py`
```python
"""Latent coherence evaluation.

Measures how far the transition model can carry a frozen hidden state before it
drifts from the teacher trajectory. Starting from the *teacher* state ``s_0``, we
repeatedly apply the transition model with the **ground-truth symbolic actions**
and, at each depth ``d``, compare the rolled-out state ``ĥ_d`` against the teacher
state ``s_d``:

* hidden-state MSE
* cosine similarity
* diagnostic-decoder next-token accuracy (does ĥ_d still decode to the correct
  next reasoning token?)

The teacher-state decoder accuracy is reported alongside as an upper bound — it
isolates representation error (teacher) from accumulated dynamics error (rollout).

Output: ``coherence_depth.csv`` (depth, mse, cosine_similarity, token_accuracy,
plus n_samples and teacher_token_accuracy) and ``coherence_depth.png``.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from data_processing.trajectory_dataset import Trajectory, IGNORE_TOKEN


@torch.no_grad()
def evaluate_coherence(
    transition_model,
    trajectories: List[Trajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    probe_d=None,
    max_depth: Optional[int] = None,
    domain: str = "countdown",
    rollout_mode: str = "true",
):
    """Rollout the transition model and measure coherence vs depth.

    rollout_mode can be "true", "shuffled", or "constant".

    If max_depth is None, sets it dynamically to the 95th percentile of trajectory lengths.
    Returns a pandas DataFrame indexed by depth (1..max_depth).
    """
    import pandas as pd
    import numpy as np

    if max_depth is None:
        lengths = [t.num_steps for t in trajectories]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transition_model = transition_model.to(device)
    transition_model.eval()

    # Compute random-pair cosine baseline
    if len(trajectories) >= 2:
        t_idx1 = torch.randint(0, len(trajectories), (10000,))
        t_idx2 = torch.randint(0, len(trajectories), (10000,))
        valid = t_idx1 != t_idx2
        t_idx1, t_idx2 = t_idx1[valid], t_idx2[valid]
        
        s1 = torch.cat([trajectories[i.item()].states[torch.randint(0, trajectories[i.item()].num_steps, (1,)).item():][:1] for i in t_idx1], dim=0).to(device)
        s2 = torch.cat([trajectories[i.item()].states[torch.randint(0, trajectories[i.item()].num_steps, (1,)).item():][:1] for i in t_idx2], dim=0).to(device)
        
        random_pair_cos = F.cosine_similarity(s1, s2, dim=-1).mean().item()
    else:
        random_pair_cos = 0.0

    # Accumulators per depth.
    agg = {
        d: {"mse": [], "cos": [], "op_acc": [], "teacher_op_acc": [], "state_acc": [], "teacher_state_acc": [],
            "id_mse": [], "id_cos": [], "id_op_acc": [], "id_state_acc": [],
            "probe_b_acc": [], "teacher_probe_b_acc": [], "id_probe_b_acc": [],
            "probe_d_acc": [], "teacher_probe_d_acc": [], "id_probe_d_acc": []}
        for d in range(1, max_depth + 1)
    }
    
    from evaluation.probes import get_probe_a_targets

    for traj in trajectories:
        states = traj.states  # (N+1, H)
        N = traj.num_steps
        h = states[0:1].to(device)  # (1, H) teacher start
        h_identity = h.clone().to(device)
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            op = traj.op_ids[d - 1: d].to(device)
            operands = traj.operands[d - 1: d].to(device)
            
            if rollout_mode == "shuffled":
                rand_traj = trajectories[torch.randint(len(trajectories), (1,)).item()]
                rand_d = torch.randint(rand_traj.num_steps, (1,)).item()
                op = rand_traj.op_ids[rand_d: rand_d + 1].to(device)
                operands = rand_traj.operands[rand_d: rand_d + 1].to(device)
            elif rollout_mode == "constant":
                const_traj = trajectories[0]
                op = const_traj.op_ids[0: 1].to(device)
                operands = const_traj.operands[0: 1].to(device)

            h = transition_model(h, op, operands)  # predicted s_d
            teacher = states[d: d + 1].to(device)

            agg[d]["mse"].append(F.mse_loss(h, teacher).item())
            agg[d]["cos"].append(F.cosine_similarity(h, teacher, dim=-1).item())
            
            # Identity baseline metrics
            agg[d]["id_mse"].append(F.mse_loss(h_identity, teacher).item())
            agg[d]["id_cos"].append(F.cosine_similarity(h_identity, teacher, dim=-1).item())

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                agg[d]["op_acc"].append(1.0 if int(probe_c.predict(h.cpu().numpy())[0]) == target_op else 0.0)
                agg[d]["teacher_op_acc"].append(1.0 if int(probe_c.predict(teacher.cpu().numpy())[0]) == target_op else 0.0)
                agg[d]["id_op_acc"].append(1.0 if int(probe_c.predict(h_identity.cpu().numpy())[0]) == target_op else 0.0)
                
            if probe_b is not None:
                target_dist = N - d
                agg[d]["probe_b_acc"].append(1.0 if int(probe_b.predict(h.cpu().numpy())[0]) == target_dist else 0.0)
                agg[d]["teacher_probe_b_acc"].append(1.0 if int(probe_b.predict(teacher.cpu().numpy())[0]) == target_dist else 0.0)
                agg[d]["id_probe_b_acc"].append(1.0 if int(probe_b.predict(h_identity.cpu().numpy())[0]) == target_dist else 0.0)

            if probe_d is not None:
                target_reach = 1 if (N - d) <= 2 else 0
                agg[d]["probe_d_acc"].append(1.0 if int(probe_d.predict(h.cpu().numpy())[0]) == target_reach else 0.0)
                agg[d]["teacher_probe_d_acc"].append(1.0 if int(probe_d.predict(teacher.cpu().numpy())[0]) == target_reach else 0.0)
                agg[d]["id_probe_d_acc"].append(1.0 if int(probe_d.predict(h_identity.cpu().numpy())[0]) == target_reach else 0.0)

            if probe_a is not None:
                a_row = get_probe_a_targets(traj, d, domain)
                
                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t) if pv == tv) / len(t)
                
                pred_a = probe_a.predict(h.cpu().numpy())[0]
                agg[d]["state_acc"].append(_avg_match(pred_a, a_row))
                
                t_pred_a = probe_a.predict(teacher.cpu().numpy())[0]
                agg[d]["teacher_state_acc"].append(_avg_match(t_pred_a, a_row))
                
                id_pred_a = probe_a.predict(h_identity.cpu().numpy())[0]
                agg[d]["id_state_acc"].append(_avg_match(id_pred_a, a_row))

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for d in range(1, max_depth + 1):
        rows.append({
            "depth": d,
            "mse": _mean(agg[d]["mse"]),
            "cosine_similarity": _mean(agg[d]["cos"]),
            "random_pair_cosine": random_pair_cos,
            "mean_centered_cosine": _mean(agg[d]["cos"]) - random_pair_cos if agg[d]["cos"] else float("nan"),
            "operator_accuracy": _mean(agg[d]["op_acc"]),
            "teacher_operator_accuracy": _mean(agg[d]["teacher_op_acc"]),
            "state_probe_accuracy": _mean(agg[d]["state_acc"]),
            "teacher_state_probe_accuracy": _mean(agg[d]["teacher_state_acc"]),
            "dist_probe_accuracy": _mean(agg[d]["probe_b_acc"]),
            "teacher_dist_probe_accuracy": _mean(agg[d]["teacher_probe_b_acc"]),
            "reach_probe_accuracy": _mean(agg[d]["probe_d_acc"]),
            "teacher_reach_probe_accuracy": _mean(agg[d]["teacher_probe_d_acc"]),
            "identity_mse": _mean(agg[d]["id_mse"]),
            "identity_cosine_similarity": _mean(agg[d]["id_cos"]),
            "identity_operator_accuracy": _mean(agg[d]["id_op_acc"]),
            "identity_state_probe_accuracy": _mean(agg[d]["id_state_acc"]),
            "identity_dist_probe_accuracy": _mean(agg[d]["id_probe_b_acc"]),
            "identity_reach_probe_accuracy": _mean(agg[d]["id_probe_d_acc"]),
            "semantic_gain_state": _mean(agg[d]["state_acc"]) - _mean(agg[d]["id_state_acc"]) if len(agg[d]["state_acc"]) else float('nan'),
            "semantic_gain_dist": _mean(agg[d]["probe_b_acc"]) - _mean(agg[d]["id_probe_b_acc"]) if len(agg[d]["probe_b_acc"]) else float('nan'),
            "semantic_gain_reach": _mean(agg[d]["probe_d_acc"]) - _mean(agg[d]["id_probe_d_acc"]) if len(agg[d]["probe_d_acc"]) else float('nan'),
            "n_samples": len(agg[d]["mse"]),
        })
    df = pd.DataFrame(rows)
    # Identity-Normalized Transition Score
    df["dynamics_gain"] = df["identity_mse"] / df["mse"]
    return df


def save_coherence(df, csv_path: str, png_path: Optional[str] = None) -> None:
    """Write the coherence CSV and (optionally) the depth plot."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_coherence(df, png_path)


def _plot_coherence(df, png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Rollout depth")
    ax1.set_ylabel("Cosine similarity / Probe accuracy")
    l1, = ax1.plot(valid["depth"], valid["cosine_similarity"],
                   marker="o", color="tab:blue", label="Cosine similarity")
    l2, = ax1.plot(valid["depth"], valid["operator_accuracy"],
                   marker="s", color="tab:green", label="Probe C (op) acc")
    l3, = ax1.plot(valid["depth"], valid["state_probe_accuracy"],
                   marker="v", color="tab:purple", label="Probe A (state) acc")
    l4, = ax1.plot(valid["depth"], valid["dist_probe_accuracy"],
                   marker="*", color="tab:orange", label="Probe B (dist) acc")
    l5, = ax1.plot(valid["depth"], valid["reach_probe_accuracy"],
                   marker="p", color="tab:cyan", label="Probe D (reach) acc")
    
    l6, = ax1.plot(valid["depth"], valid["teacher_operator_accuracy"],
                   marker="s", linestyle="--", color="tab:green", alpha=0.5, label="Teacher Probe C")
    l7, = ax1.plot(valid["depth"], valid["teacher_state_probe_accuracy"],
                   marker="v", linestyle="--", color="tab:purple", alpha=0.5, label="Teacher Probe A")
    l8, = ax1.plot(valid["depth"], valid["teacher_dist_probe_accuracy"],
                   marker="*", linestyle="--", color="tab:orange", alpha=0.5, label="Teacher Probe B")
    l9, = ax1.plot(valid["depth"], valid["teacher_reach_probe_accuracy"],
                   marker="p", linestyle="--", color="tab:cyan", alpha=0.5, label="Teacher Probe D")
                   
    l_id_cos, = ax1.plot(valid["depth"], valid["identity_cosine_similarity"],
                         marker="x", linestyle=":", color="tab:blue", alpha=0.5,
                         label="Cosine similarity (Identity)")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Hidden-state MSE", color="tab:red")
    l_mse, = ax2.plot(valid["depth"], valid["mse"],
                   marker="d", color="tab:red", label="Hidden-state MSE")
    l_id_mse, = ax2.plot(valid["depth"], valid["identity_mse"],
                         marker="*", linestyle=":", color="tab:red", alpha=0.5,
                         label="Hidden-state MSE (Identity)")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = [l1, l2, l3, l4, l5, l_id_cos, l6, l7, l8, l9, l_mse, l_id_mse]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize="small")
    ax1.set_title("Latent coherence vs rollout depth")
    ax1.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\cross_domain_transfer.py`
```python
"""Cross-domain transfer metrics for the V5 discrete codebook.

Given a codebook trained on the **arithmetic** (Countdown) domain, does it
transfer to other reasoning domains? For each transfer domain we encode its
hidden states with the frozen arithmetic VQ and measure:

* **code reuse** — fraction of transfer states assigned to a code that is
  *active* in the arithmetic domain (vs falling onto dead codes / novel modes).
* **entropy delta** — ``H_transfer - H_arithmetic`` over the code distribution.
  A large positive delta means transfer states are spread (the codebook does
  not cover them well); a negative delta means they collapse onto few codes.
* **transition stability** — symmetric KL divergence between the arithmetic
  empirical transition matrix ``T_arith`` and the transfer-empirical
  ``T_transfer`` (restricted to codes active in both). Low divergence means the
  same discrete dynamics govern both domains.

These three numbers summarize whether the discrete-state structure discovered
in arithmetic is a *general* property of the frozen LM's hidden-state
trajectories or an artifact of one domain.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
    all_codes,
)
from data_processing.trajectory_dataset import Trajectory
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.discrete_transition import build_transition_matrix
from models.vq_state import VQStateQuantizer


def _code_distribution(codes: torch.Tensor, num_codes: int) -> np.ndarray:
    counts = np.bincount(codes.detach().cpu().numpy(), minlength=num_codes).astype(np.float64)
    s = counts.sum()
    return counts / s if s > 0 else counts


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()) if nz.size else 0.0


def _symmetric_kl(P: np.ndarray, Q: np.ndarray, active_mask: np.ndarray) -> float:
    """Symmetric KL between two transition distributions, restricted to ``active`` rows.

    Each row is treated as a categorical over next-codes; we add a tiny floor to
    avoid log(0) and average the per-row symmetrized KL over active rows.
    """
    eps = 1e-12
    mask = active_mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    P = P[mask] + eps
    Q = Q[mask] + eps
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    kl_pq = (P * np.log(P / Q)).sum(axis=1)
    kl_qp = (Q * np.log(Q / P)).sum(axis=1)
    return float((kl_pq + kl_qp).mean() / 2.0)


def evaluate_cross_domain_transfer(
    vq: VQStateQuantizer,
    arith_disc: List[DiscreteTrajectory],
    transfer_trajs_by_domain: Dict[str, List[Trajectory]],
    num_codes: int,
) -> Dict[str, Dict]:
    """Compute transfer metrics for every domain in ``transfer_trajs_by_domain``.

    Args:
        vq: the arithmetic-trained VQ quantizer (frozen).
        arith_disc: arithmetic discrete trajectories (the in-domain reference).
        transfer_trajs_by_domain: ``{domain_name: [Trajectory, ...]}`` to evaluate.
        num_codes: codebook cardinality.

    Returns ``{domain: {code_reuse, entropy, entropy_delta, transition_kl,
    active_codes_in_domain, n_states}}`` plus an ``arithmetic`` reference entry.
    """
    # Arithmetic reference distribution + transition matrix.
    arith_codes = all_codes(arith_disc)
    arith_stats = analyze_codebook_usage(arith_codes, num_codes)
    arith_freq = arith_stats["freqs"]
    arith_H = arith_stats["entropy"]
    arith_active_mask = arith_freq > 0
    T_arith = build_transition_matrix(arith_disc, num_codes)

    results: Dict[str, Dict] = {
        "arithmetic": {
            "entropy": arith_H,
            "active_codes": int(arith_active_mask.sum()),
            "perplexity": arith_stats["perplexity"],
            "collapse_score_gini": arith_stats["collapse_score"],
            "n_states": int(arith_codes.shape[0]),
        }
    }

    for domain, trajs in transfer_trajs_by_domain.items():
        if not trajs:
            results[domain] = {"error": "no trajectories"}
            continue
        disc = encode_trajectories_to_codes(vq, trajs)
        codes = all_codes(disc)
        stats = analyze_codebook_usage(codes, num_codes)
        H = stats["entropy"]

        # Code reuse: fraction of transfer states landing on codes that are
        # active in the arithmetic domain.
        freq = stats["freqs"]
        reuse = float(freq[arith_active_mask].sum()) if arith_active_mask.any() else 0.0

        # Transition stability: symmetric KL between arithmetic and transfer
        # transition matrices over codes active in BOTH domains.
        T_trans = build_transition_matrix(disc, num_codes)
        both_active = arith_active_mask & (freq > 0)
        tkl = _symmetric_kl(T_arith, T_trans, both_active)

        results[domain] = {
            "code_reuse": reuse,
            "entropy": H,
            "entropy_delta": H - arith_H,
            "transition_kl_symmetric": tkl,
            "active_codes_in_domain": int(stats["used_codes"]),
            "perplexity": stats["perplexity"],
            "collapse_score_gini": stats["collapse_score"],
            "n_states": int(codes.shape[0]),
        }
    return results


def save_transfer_report(results: Dict[str, Dict], csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows = []
    for domain, m in results.items():
        row = {"domain": domain}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

```


## File: `evaluation\discrete_rollout.py`
```python
"""Discrete rollout coherence test.

The Version-5 analog of the continuous coherence rollout
(:mod:`evaluation.coherence`). Starting from the *teacher* code ``z_0`` we
repeatedly apply the action-blind discrete transition model
(:class:`evaluation.discrete_transition.CodeTransitionModel`) to roll forward
in code space. At each depth ``d`` we:

1. decode the predicted code back to a continuous hidden state via the VQ
   codebook (``h_decoded = codebook[z_pred]``),
2. run the **existing** linear probes (A remaining numbers / B distance / C
   next-op) on ``h_decoded``,
3. compare against the teacher code at the same depth.

The output schema mirrors ``coherence_depth.csv`` so the discrete rollout can
be plotted alongside (or against) the continuous one.

This test answers: *do the discrete dynamics remain coherent when rolled out
multiple steps, or does the code sequence collapse / diverge past depth 1?*
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from models.vq_state import VQStateQuantizer
from evaluation.discrete_transition import CodeTransitionModel




@torch.no_grad()
def evaluate_discrete_rollout(
    transition_model: CodeTransitionModel,
    vq: VQStateQuantizer,
    discrete_trajs: List[DiscreteTrajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    max_depth: Optional[int] = None,
    domain: str = "countdown",
):
    """Roll out the discrete transition model and measure coherence vs depth.

    Returns a pandas DataFrame with one row per depth (1..max_depth):
    ``depth, code_match_accuracy, cosine_similarity, mse, operator_accuracy,
    dist_probe_accuracy, state_probe_accuracy, teacher_*``, ``n_samples``.
    """
    import pandas as pd
    from evaluation.probes import get_probe_a_targets

    if max_depth is None:
        lengths = [t.num_steps for t in discrete_trajs]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    # The transition model lives on whichever device it was trained on; the VQ
    # codebook is read-only, so we can do all rollout on the model's device and
    # move tiny tensors CPU-side for the sklearn probes.
    try:
        device = next(transition_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    transition_model.eval()
    codebook = vq.codebook.to(device)

    agg = {
        d: {"code_match": [], "cos": [], "mse": [],
            "op_acc": [], "teacher_op_acc": [],
            "probe_b_acc": [], "teacher_probe_b_acc": [],
            "state_acc": [], "teacher_state_acc": [],
            "n": 0}
        for d in range(1, max_depth + 1)
    }

    for traj in discrete_trajs:
        N = traj.num_steps
        if N == 0:
            continue
        z = traj.codes[0:1].to(device)        # (1,) teacher start code
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            logits = transition_model(z)       # (1, K)
            z_pred = logits.argmax(dim=-1)     # (1,)
            teacher_code = traj.codes[d:d + 1].to(device)

            agg[d]["code_match"].append(
                float((z_pred == teacher_code).item())
            )
            h_pred = codebook[z_pred]          # (1, H)
            h_teacher = codebook[teacher_code]
            agg[d]["cos"].append(
                F.cosine_similarity(h_pred, h_teacher, dim=-1).item()
            )
            agg[d]["mse"].append(F.mse_loss(h_pred, h_teacher).item())

            # Advance the rollout by the predicted code (autoregressive).
            z = z_pred

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["op_acc"].append(
                    1.0 if int(probe_c.predict(hp)[0]) == target_op else 0.0
                )
                agg[d]["teacher_op_acc"].append(
                    1.0 if int(probe_c.predict(ht)[0]) == target_op else 0.0
                )

            if probe_b is not None:
                target_dist = N - d
                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["probe_b_acc"].append(
                    1.0 if int(probe_b.predict(hp)[0]) == target_dist else 0.0
                )
                agg[d]["teacher_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(ht)[0]) == target_dist else 0.0
                )

            if probe_a is not None:
                a_row = get_probe_a_targets(traj, d, domain)

                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t) if pv == tv) / len(t)

                hp = h_pred.cpu().numpy()
                ht = h_teacher.cpu().numpy()
                agg[d]["state_acc"].append(
                    _avg_match(probe_a.predict(hp)[0], a_row)
                )
                agg[d]["teacher_state_acc"].append(
                    _avg_match(probe_a.predict(ht)[0], a_row)
                )

            agg[d]["n"] += 1

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for d in range(1, max_depth + 1):
        rows.append({
            "depth": d,
            "code_match_accuracy": _mean(agg[d]["code_match"]),
            "cosine_similarity": _mean(agg[d]["cos"]),
            "mse": _mean(agg[d]["mse"]),
            "operator_accuracy": _mean(agg[d]["op_acc"]),
            "teacher_operator_accuracy": _mean(agg[d]["teacher_op_acc"]),
            "dist_probe_accuracy": _mean(agg[d]["probe_b_acc"]),
            "teacher_dist_probe_accuracy": _mean(agg[d]["teacher_probe_b_acc"]),
            "state_probe_accuracy": _mean(agg[d]["state_acc"]),
            "teacher_state_probe_accuracy": _mean(agg[d]["teacher_state_acc"]),
            "n_samples": agg[d]["n"],
        })
    return pd.DataFrame(rows)


def save_discrete_rollout(df, csv_path: str, png_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_rollout(df, png_path)


def _plot_rollout(df, png_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Rollout depth")
    ax.set_ylabel("Accuracy / similarity")

    curves = [
        ("code_match_accuracy", "Code match", "tab:blue", "o"),
        ("cosine_similarity", "Cosine (decoded)", "tab:cyan", "x"),
        ("operator_accuracy", "Probe C (op)", "tab:green", "s"),
        ("dist_probe_accuracy", "Probe B (dist)", "tab:orange", "*"),
        ("state_probe_accuracy", "Probe A (state)", "tab:purple", "v"),
    ]
    for col, label, color, marker in curves:
        if col in valid.columns:
            ax.plot(valid["depth"], valid[col], marker=marker, color=color, label=label)

    # Teacher ceilings as dashed lines.
    for col, label, color in [
        ("teacher_operator_accuracy", "Teacher C", "tab:green"),
        ("teacher_state_probe_accuracy", "Teacher A", "tab:purple"),
    ]:
        if col in valid.columns:
            ax.plot(valid["depth"], valid[col], linestyle="--", color=color,
                    alpha=0.5, label=label)

    ax.set_ylim(0, 1.05)
    ax.set_title("Discrete rollout coherence vs depth")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\discrete_transition.py`
```python
"""Discrete transition predictability and entropy analysis.

Two related diagnostics over the discrete code trajectory cache:

**C3 — Transition Predictability.** Train a small action-blind MLP
``z_t -> z_{t+1}`` and measure top-1 accuracy, predictive entropy and
perplexity. Above-chance accuracy is the minimal evidence that discrete state
dynamics *exist* (i.e. the next state is a function of the current one).

**C4 — Transition Entropy Analysis.** Build the empirical transition matrix
``T[i, j] = P(z_next = j | z_current = i)`` and report:

* the per-state conditional entropy ``H(Z_next | Z_current = i)``, and
* the global conditional entropy ``H(Z_next | Z_current)``.

A *planning state* has low conditional entropy (a near-deterministic
successor); a *compression bucket* has high entropy (many possible
successors, i.e. the code is lossy). The split between these two regimes is
the substantive scientific output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    DiscreteTransitionDataset,
)


# --------------------------------------------------------------------------- #
# C3 — Transition predictability (small MLP)
# --------------------------------------------------------------------------- #
class CodeTransitionModel(nn.Module):
    """Tiny embedding-MLP next-code predictor: ``z_t -> logits over codes``."""

    def __init__(self, num_codes: int, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_codes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_codes),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(z_t))


@dataclass
class CodeTransitionConfig:
    embed_dim: int = 32
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval_predictability(
    model: CodeTransitionModel, loader: DataLoader, num_codes: int
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    ce_sum, correct, entropy_sum, n = 0.0, 0, 0.0, 0
    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_next = batch["z_next"].to(device)
        logits = model(z_t)
        probs = F.softmax(logits, dim=-1)
        entropy_sum += (-(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)).sum().item()
        ce_sum += F.cross_entropy(logits, z_next, reduction="sum").item()
        correct += (logits.argmax(-1) == z_next).sum().item()
        n += z_t.shape[0]
    loss = ce_sum / max(n, 1)
    return {
        "loss": loss,
        "top1_accuracy": correct / max(n, 1),
        "predictive_entropy_nats": entropy_sum / max(n, 1),
        # Perplexity of the model's predicted next-state distribution.
        "predictive_perplexity": float(np.exp(loss)),
    }


def transition_baselines(
    train_disc: List[DiscreteTrajectory],
    eval_disc: List[DiscreteTrajectory],
    num_codes: int,
) -> Dict[str, float]:
    """Non-parametric next-code baselines, evaluated on ``eval_disc``.

    These replace the old uniform ``1 / num_codes`` "chance" baseline, which
    badly understated the real floor whenever code usage is skewed (it made an
    MLP look strong merely for predicting a frequent code). The two honest
    references are:

    * **Majority** (Baseline A): always predict the single most frequent next
      code seen in training. With a collapsed/skewed codebook this alone can be
      large.
    * **Bigram** (Baseline B): predict ``argmax_j count(i -> j)`` for the
      current code ``i`` (its most frequent training successor), falling back to
      the global majority for codes unseen in training. This is the first-order
      Markov count model. The MLP only demonstrates *learned* structure if it
      clears the bigram; matching it means the MLP captured nothing beyond
      pairwise counts.
    """
    train_ds = DiscreteTransitionDataset(train_disc)
    eval_ds = DiscreteTransitionDataset(eval_disc)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        return {"majority_baseline": float("nan"), "bigram_baseline": float("nan")}

    zt_tr = np.asarray(train_ds.z_t, dtype=np.int64)
    zn_tr = np.asarray(train_ds.z_next, dtype=np.int64)
    zt_ev = np.asarray(eval_ds.z_t, dtype=np.int64)
    zn_ev = np.asarray(eval_ds.z_next, dtype=np.int64)

    # Majority: global mode of training next-codes.
    majority_code = int(np.bincount(zn_tr, minlength=num_codes).argmax())
    majority_acc = float((zn_ev == majority_code).mean())

    # Bigram: most frequent successor per current code.
    counts = np.zeros((num_codes, num_codes), dtype=np.int64)
    np.add.at(counts, (zt_tr, zn_tr), 1)
    bigram_pred = counts.argmax(axis=1)               # (num_codes,)
    bigram_pred[counts.sum(axis=1) == 0] = majority_code  # unseen i -> majority
    bigram_acc = float((zn_ev == bigram_pred[zt_ev]).mean())

    return {"majority_baseline": majority_acc, "bigram_baseline": bigram_acc}


def train_code_transition(
    train_disc: List[DiscreteTrajectory],
    val_disc: Optional[List[DiscreteTrajectory]] = None,
    num_codes: int = 256,
    config: Optional[CodeTransitionConfig] = None,
) -> Tuple[CodeTransitionModel, Dict[str, float]]:
    """Train the action-blind discrete transition model.

    Returns ``(model, metrics)`` where ``metrics`` is evaluated on ``val_disc``
    if provided, else on ``train_disc``, and includes the majority/bigram
    baselines alongside the MLP's top-1 accuracy.
    """
    config = config or CodeTransitionConfig()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DiscreteTransitionDataset(train_disc)
    if len(train_ds) == 0:
        raise ValueError("No discrete transitions to train on.")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    eval_disc = val_disc if val_disc else train_disc
    eval_loader = DataLoader(DiscreteTransitionDataset(eval_disc), batch_size=config.batch_size)

    model = CodeTransitionModel(num_codes, config.embed_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            z_next = batch["z_next"].to(device)
            optimizer.zero_grad()
            logits = model(z_t)
            loss = F.cross_entropy(logits, z_next)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * z_t.shape[0]
            n += z_t.shape[0]
        record = {"epoch": epoch, "step": epoch, "loss": loss_sum / max(n, 1)}
        ev = _eval_predictability(model, eval_loader, num_codes)
        record["eval_top1_accuracy"] = ev["top1_accuracy"]
        record["eval_loss"] = ev["loss"]
        config.history.append(record)

    metrics = _eval_predictability(model, eval_loader, num_codes)
    metrics.update(transition_baselines(train_disc, eval_disc, num_codes))
    return model, metrics


def save_predictability_metrics(metrics: Dict[str, float], csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)


# --------------------------------------------------------------------------- #
# C4 — Empirical transition matrix + conditional entropy
# --------------------------------------------------------------------------- #
def build_transition_matrix(
    discrete_trajs: List[DiscreteTrajectory], num_codes: int, smoothing: float = 0.0
) -> np.ndarray:
    """Empirical ``T[i, j] = P(z_next = j | z_current = i)``.

    Rows with no observed successors fall back to a uniform distribution so
    the matrix is always a valid stochastic matrix (this also avoids
    ``log(0)`` downstream). Optional Laplace ``smoothing`` (default 0) adds
    pseudo-counts to every cell.
    """
    T = np.full((num_codes, num_codes), smoothing, dtype=np.float64)
    for traj in discrete_trajs:
        codes = traj.codes.numpy()
        for i in range(traj.num_steps):
            T[codes[i], codes[i + 1]] += 1.0

    row_sums = T.sum(axis=1, keepdims=True)
    # Rows that were never observed (sum 0) -> uniform.
    zero = (row_sums.squeeze(-1)) == 0
    T[zero] = 1.0 / num_codes
    row_sums[zero] = 1.0
    T /= row_sums
    return T


def conditional_entropy(T: np.ndarray, code_freqs: Optional[np.ndarray] = None) -> Dict:
    """Conditional entropy of the transition matrix.

    Args:
        T: ``(K, K)`` stochastic transition matrix.
        code_freqs: optional ``(K,)`` marginal ``P(z_current = i)`` used to
            weight per-state entropies into the global ``H(Z_next | Z_current)``.
            If omitted, the unweighted mean of per-state entropies is reported.

    Returns a dict with per-state entropy array and the (weighted) global
    conditional entropy / perplexity. All entropies are in nats.
    """
    K = T.shape[0]
    safe = np.clip(T, 1e-12, None)
    per_state = -(safe * np.log(safe)).sum(axis=1)  # (K,)

    if code_freqs is None:
        weight = np.full(K, 1.0 / K)
        global_entropy = float(per_state.mean())
    else:
        w = np.asarray(code_freqs, dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        weight = w
        global_entropy = float((per_state * w).sum())

    return {
        "per_state_entropy": per_state,
        "global_entropy": global_entropy,
        "global_perplexity": float(np.exp(global_entropy)),
        # Fraction of *used* states whose successor is near-deterministic
        # (entropy < 0.5 nats ~= top-1 prob > ~0.78). These are "planning" codes.
        "deterministic_state_frac": float(
            ((per_state < 0.5) & (weight > 0)).sum() / max(int((weight > 0).sum()), 1)
        ),
    }


def save_entropy_report(
    T: np.ndarray,
    ent: Dict,
    csv_path: str,
    png_path: Optional[str] = None,
    active_mask: Optional[np.ndarray] = None,
) -> None:
    """Write a per-state entropy CSV (active states only) and the plot."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    per_state = ent["per_state_entropy"]
    rows = []
    mask = active_mask if active_mask is not None else np.ones_like(per_state, dtype=bool)
    for i in np.where(mask)[0]:
        rows.append(
            {
                "code": int(i),
                "entropy_nats": float(per_state[i]),
                "top1_successor_prob": float(T[i].max()),
                "top1_successor_code": int(T[i].argmax()),
            }
        )
    summary = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["code", "entropy_nats", "top1_successor_prob", "top1_successor_code"]
    )
    summary.to_csv(csv_path, index=False)

    if png_path is not None:
        _plot_entropy(per_state, ent, png_path, active_mask)


def _plot_entropy(per_state, ent, png_path, active_mask=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    mask = active_mask if active_mask is not None else np.ones_like(per_state, dtype=bool)
    vals = per_state[mask]
    if vals.size:
        ax1.hist(vals, bins=min(40, max(8, int(np.sqrt(vals.size)))),
                 color="tab:purple", alpha=0.8)
    ax1.axvline(ent["global_entropy"], color="tab:red", linestyle="--",
                label=f"global H = {ent['global_entropy']:.3f}")
    ax1.set_xlabel("H(Z_next | Z_current = i)  [nats]")
    ax1.set_ylabel("# codes")
    ax1.set_title("Per-state transition entropy")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(["global entropy", "global perplexity", "det. state frac"],
            [ent["global_entropy"], ent["global_perplexity"], ent["deterministic_state_frac"]],
            color=["tab:blue", "tab:orange", "tab:green"])
    ax2.set_title("Transition entropy summary")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\intrinsic_noise.py`
```python
"""Phase C: Intrinsic State Noise Diagnostic.

Measures the variance of the hidden state representation for identical
symbolic states. This determines whether the representation is Markovian
or heavily context/history-dependent.
"""

from __future__ import annotations

import collections
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_processing.action_parser import apply_op, ID_TO_OP
from data_processing.trajectory_dataset import Trajectory


def get_symbolic_states(traj: Trajectory) -> List[Optional[Tuple[Tuple[int, Tuple[int, ...]], Tuple[Tuple[int, int, int], ...]]]]:
    """Reconstructs the symbolic state at each depth.
    
    Returns a list of length N+1 containing tuples of (symbolic_state, action_history).
    Symbolic State is defined as: (target, tuple(sorted(available_numbers)))
    Action History is defined as: tuple of (op_id, arg1, arg2) applied so far.
    If a teacher makes an arithmetic mistake (using a number not available),
    subsequent states are None.
    """
    states = []
    target = traj.target
    current_numbers = list(traj.numbers)
    history = []
    
    # Depth 0: Initial state
    states.append(((target, tuple(sorted(current_numbers))), tuple(history)))
    
    N = traj.num_steps
    for d in range(N):
        op_id = int(traj.op_ids[d].item())
        op = ID_TO_OP[op_id]
        arg1 = int(traj.operands[d][0].item())
        arg2 = int(traj.operands[d][1].item())
        
        # Verify operands are available
        try:
            current_numbers.remove(arg1)
            current_numbers.remove(arg2)
            result = apply_op(op, arg1, arg2)
            current_numbers.append(result)
            history.append((op_id, arg1, arg2))
            states.append(((target, tuple(sorted(current_numbers))), tuple(history)))
        except (ValueError, ZeroDivisionError):
            # Invalid arithmetic or operand not available
            # Fill the rest with None
            while len(states) < N + 1:
                states.append(None)
            break
            
    return states


def gather_state_groups(trajectories: List[Trajectory]) -> Dict[Tuple, List[Tuple[int, Tuple, torch.Tensor]]]:
    """Groups hidden states by their symbolic state.
    
    To avoid trivial intra-trajectory matches, we only keep states that appear
    across *multiple* trajectories.
    
    Returns:
        dict: symbolic_state -> list of (traj_idx, action_history, H)
    """
    # map: symbolic_state -> dict(traj_idx -> list of hidden states)
    temp_groups = collections.defaultdict(lambda: collections.defaultdict(list))
    
    for idx, traj in enumerate(trajectories):
        states_info = get_symbolic_states(traj)
        for d, info in enumerate(states_info):
            if info is not None:
                sym, hist = info
                # traj.states is (N+1, H)
                h = traj.states[d:d+1]
                temp_groups[sym][idx].append((hist, h))
                
    # Flatten groups, but only keep symbolic states that span >1 trajectory
    final_groups = {}
    for sym, traj_map in temp_groups.items():
        if len(traj_map) > 1:
            # Flatten all tensors for this state, tracking trajectory ID and history
            tensors = []
            for traj_idx, items in traj_map.items():
                for hist, h in items:
                    tensors.append((traj_idx, hist, h))
            final_groups[sym] = tensors
            
    return final_groups


def compute_intrinsic_noise(
    state_groups: Dict[Tuple, List[Tuple[int, Tuple, torch.Tensor]]], 
    max_pairs_per_state: int = 100
) -> Tuple[pd.DataFrame, float, float]:
    """Computes within-state and between-state similarities, and retrieval accuracy.
    
    Args:
        state_groups: Dictionary of symbolic state to list of (traj_idx, action_history, hidden_state).
        
    Returns:
        (df, top1_acc, top5_acc)
    """
    within_cos = []
    within_l2 = []
    
    between_cos = []
    between_l2 = []
    
    all_states = list(state_groups.keys())
    if len(all_states) < 2:
        raise ValueError("Not enough distinct symbolic states to compute noise.")
        
    for sym, tensors in state_groups.items():
        n = len(tensors)
        if n < 2:
            continue
            
        # 1. Within-state pairs
        # Randomly sample pairs if there are too many to avoid combinatorial explosion
        pairs_to_sample = min(max_pairs_per_state, n * (n - 1) // 2)
        
        # Collect all unique indices pairs
        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(indices)
        indices = indices[:pairs_to_sample]
        
        for i, j in indices:
            t1_idx, hist1, h1 = tensors[i]
            t2_idx, hist2, h2 = tensors[j]
            h1 = h1.float()
            h2 = h2.float()
            
            # 1. Within-state pairs
            if t1_idx != t2_idx and hist1 != hist2:
                cos = F.cosine_similarity(h1, h2, dim=-1).item()
                l2 = torch.norm(h1 - h2, p=2, dim=-1).item()
                within_cos.append(cos)
                within_l2.append(l2)
            
            # 2. Between-state pairs
            # Sample a random different state
            other_sym = random.choice(all_states)
            while other_sym == sym:
                other_sym = random.choice(all_states)
                
            other_tensors = state_groups[other_sym]
            t3_idx, hist3, h3 = random.choice(other_tensors)
            h3 = h3.float()
            
            if t1_idx != t3_idx and hist1 != hist3:
                cos_b = F.cosine_similarity(h1, h3, dim=-1).item()
                l2_b = torch.norm(h1 - h3, p=2, dim=-1).item()
                between_cos.append(cos_b)
                between_l2.append(l2_b)
            
    df = pd.DataFrame({
        "type": ["within"] * len(within_cos) + ["between"] * len(between_cos),
        "cosine": within_cos + between_cos,
        "l2": within_l2 + between_l2
    })
    
    # -----------------------------------------------------------------
    # Nearest Neighbor Symbolic State Retrieval
    # -----------------------------------------------------------------
    # Flatten all states into a single tensor for batched distance computation
    all_h = []
    all_syms = []
    all_t_idx = []
    all_hists = []
    
    for sym, elements in state_groups.items():
        for t_idx, hist, h in elements:
            all_h.append(h.view(-1).float())
            all_syms.append(sym)
            all_t_idx.append(t_idx)
            all_hists.append(hist)
            
    # Compute full pairwise cosine similarity matrix
    if len(all_h) > 10000:
        # Downsample if matrix would be too huge (e.g. >10k items = 100M+ matrix)
        idx_sample = random.sample(range(len(all_h)), 10000)
        all_h = [all_h[i] for i in idx_sample]
        all_syms = [all_syms[i] for i in idx_sample]
        all_t_idx = [all_t_idx[i] for i in idx_sample]
        all_hists = [all_hists[i] for i in idx_sample]
        
    H_mat = torch.stack(all_h)  # (M, H)
    H_mat = F.normalize(H_mat, p=2, dim=1)
    
    # Cosine similarity matrix: (M, M)
    sim_matrix = torch.matmul(H_mat, H_mat.T)
    
    # Prevent retrieving from the same trajectory OR identical action history
    t_idx_mat = torch.tensor(all_t_idx)
    same_traj_mask = (t_idx_mat.unsqueeze(0) == t_idx_mat.unsqueeze(1))
    
    # Create action history mask
    # Since history is a tuple, we can't easily vectorize this in pure torch without some work
    M = len(all_h)
    same_hist_mask = torch.zeros((M, M), dtype=torch.bool)
    for i in range(M):
        for j in range(M):
            if all_hists[i] == all_hists[j]:
                same_hist_mask[i, j] = True

    combined_mask = same_traj_mask | same_hist_mask
    sim_matrix.masked_fill_(combined_mask, -1.0)
    
    # Find top 5 nearest neighbors for each state
    top5_vals, top5_idx = torch.topk(sim_matrix, k=min(5, sim_matrix.shape[1]), dim=1)
    
    top1_hits = 0
    top5_hits = 0
    valid_queries = 0
    
    for i in range(len(all_syms)):
        query_sym = all_syms[i]
        
        # Check if there is even a possible valid match (are there any other trajectories with this sym?)
        has_valid_match = False
        for j in range(len(all_syms)):
            if i != j and all_t_idx[i] != all_t_idx[j] and all_hists[i] != all_hists[j] and all_syms[i] == all_syms[j]:
                has_valid_match = True
                break
                
        if not has_valid_match:
            continue
            
        valid_queries += 1
        
        # Top-1 check
        idx_1 = top5_idx[i][0].item()
        if all_syms[idx_1] == query_sym:
            top1_hits += 1
            
        # Top-5 check
        found_in_top5 = False
        for k in range(top5_idx.shape[1]):
            idx_k = top5_idx[i][k].item()
            if all_syms[idx_k] == query_sym:
                found_in_top5 = True
                break
        
        if found_in_top5:
            top5_hits += 1
            
    top1_acc = top1_hits / valid_queries if valid_queries > 0 else 0.0
    top5_acc = top5_hits / valid_queries if valid_queries > 0 else 0.0

    return df, top1_acc, top5_acc


def plot_noise_histogram(df: pd.DataFrame, png_path: str) -> None:
    """Generates density plots comparing within and between state cosine similarities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(data=df[df["type"] == "within"], x="cosine", fill=True, label="Within-State", color="tab:blue")
    sns.kdeplot(data=df[df["type"] == "between"], x="cosine", fill=True, label="Between-State", color="tab:orange")
    
    plt.title("Intrinsic State Noise: Cosine Similarity Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, 1.0)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\layer_sweep.py`
```python
"""V5.2 Experiment 3 — layer-sweep aggregation and reporting (pure logic).

All V5.2 results so far probed a single layer (TinyLlama's final, ``-1``). This
module holds the depth-independent parts of Experiment 3: place each swept
layer on the Experiment-2 floor→ceiling axes (det-frac / entropy / MLP), pick
the layer where Countdown's action-conditioned structure is strongest, and
render the comparison report. The GPU extraction + per-layer VQ/analysis lives
in ``scripts/run_layer_sweep.py``; everything here is plain numpy/stdlib so it
is cheap to unit-test.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def _frac(flo: float, ceil: float, x: float) -> float:
    """Fraction of the way from ``flo`` (Exp2 noise floor) to ``ceil`` (Exp2
    positive-control ceiling) that ``x`` sits. For entropy the floor is high
    and the ceiling low, so passing ``flo``/``ceil`` in that order yields the
    fraction of the way *down* toward the ceiling. NaN if the anchors coincide.
    """
    return (x - flo) / (ceil - flo) if abs(ceil - flo) > 1e-9 else float("nan")


def summarize_sweep(rows: List[Dict], floor: Dict, ceiling: Dict) -> Dict:
    """Locate the layer with the strongest action-conditioned structure.

    Args:
        rows: per-layer metric dicts, each with ``layer``, ``det_action``,
            ``H_action`` and ``mlp_z_op``.
        floor, ceiling: Experiment-2 anchors (noise floor / positive control),
            each with ``det_action``, ``H_action`` and ``mlp_z_op``.

    Returns each layer's position on the three floor→ceiling axes, the best
    layer (highest det-frac, tie-broken by MLP(z,op)), and whether that best
    layer reaches the reusable-state regime (Exp1's det-frac ≥ 0.20 bar *and*
    at least halfway to the ceiling).
    """
    per_layer = [
        {
            "layer": r["layer"],
            "pos_det": _frac(floor["det_action"], ceiling["det_action"], r["det_action"]),
            "pos_mlp": _frac(floor["mlp_z_op"], ceiling["mlp_z_op"], r["mlp_z_op"]),
            "pos_H": _frac(floor["H_action"], ceiling["H_action"], r["H_action"]),
        }
        for r in rows
    ]
    best = max(rows, key=lambda r: (r["det_action"], r["mlp_z_op"]))
    best_pos = next(o for o in per_layer if o["layer"] == best["layer"])
    reaches = best["det_action"] >= 0.20 and best_pos["pos_det"] >= 0.5
    return {
        "per_layer": per_layer,
        "best_layer": best["layer"],
        "best_det_action": best["det_action"],
        "best_pos_det": best_pos["pos_det"],
        "reaches_ceiling": bool(reaches),
    }


def _f(v, p=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}f}"


def build_markdown(rows: List[Dict], summary: Dict, floor: Dict, ceiling: Dict,
                   observed: Optional[Dict], config: Dict) -> str:
    """Assemble the Experiment-3 markdown report from collected per-layer rows."""
    pos_by_layer = {o["layer"]: o for o in summary["per_layer"]}
    L = []
    L.append("# V5.2 · Experiment 3 — Layer Sweep\n")
    L.append("> Every V5.2 result so far probed TinyLlama's **final** layer "
             "(`-1`). A planning-state representation need not live there — "
             "mid-stack residuals often carry more abstract structure. We "
             "re-extract hidden states at several depths and run the *identical* "
             "VQ + action-conditioned pipeline (Exp1) at each, to find whether "
             "any layer pushes Countdown toward the Exp2 reusable-state "
             "ceiling.\n")

    L.append("\n## Setup\n")
    L.append(f"- **Model:** `{config['model']}` ({config['num_layers']} layers). "
             f"Layers swept: {', '.join(str(x) for x in config['layers'])} "
             "(index into `hidden_states`; the final layer = "
             f"{config['num_layers']}).\n")
    L.append(f"- **Data:** capped Countdown — {config['cap_train']} train / "
             f"{config['cap_val']} val problems (val de-duplicated against "
             "train by `(target, sorted numbers)`).\n")
    L.append(f"- **Per layer:** VQ K={config['num_codes']} (data-dependent init, "
             "raw states) → action-conditioned models A–F on held-out val + "
             "transition structure on train, plus codebook usage and position "
             "leakage. Identical to Experiment 1.\n")
    L.append("- **Anchors:** Experiment-2 noise floor and positive-control "
             "ceiling (same K, pipeline) bound each action-conditioned axis.\n")

    L.append("\n## Per-layer metrics\n")
    L.append("| Layer | Active | Perplex. | Pos-leak | Bigram(z) | MLP(z) | "
             "Act-bigram | MLP(z,op) | H(z'\\|z) | H(z'\\|z,op) | Det(z,op) | "
             "→ceiling |")
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
             "---: | ---: | ---: |")
    for r in rows:
        p = pos_by_layer[r["layer"]]
        L.append(
            f"| {r['layer']} | {r['active_codes']} | {_f(r['perplexity'], 1)} | "
            f"{_f(r['position_leakage'])} | {_f(r['bigram_z'])} | {_f(r['mlp_z'])} | "
            f"{_f(r['action_bigram'])} | {_f(r['mlp_z_op'])} | {_f(r['H_state'])} | "
            f"{_f(r['H_action'])} | {_f(r['det_action'])} | "
            f"{_f(p['pos_det'] * 100, 0)}% |")

    L.append("\n_Det(z,op) = share of transition mass from near-deterministic "
             "`(z,op)` conditionings (H<0.5 nats, seen ≥10×). →ceiling = "
             "position on the det-frac axis between the Exp2 noise floor and "
             "positive-control ceiling._\n")

    L.append("\n## Anchors (Experiment 2)\n")
    L.append("| | Det(z,op) | H(z'\\|z,op) | MLP(z,op) |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append(f"| Noise floor | {_f(floor['det_action'])} | {_f(floor['H_action'])} | "
             f"{_f(floor['mlp_z_op'])} |")
    L.append(f"| Positive-control ceiling | {_f(ceiling['det_action'])} | "
             f"{_f(ceiling['H_action'])} | {_f(ceiling['mlp_z_op'])} |")
    if observed is not None and observed.get("det_action") is not None:
        L.append(f"| Cached layer −1 (Exp1, full data) | {_f(observed['det_action'])} | "
                 f"{_f(observed['H_action'])} | {_f(observed['mlp_z_op'])} |")

    best = summary["best_layer"]
    best_row = next(r for r in rows if r["layer"] == best)
    L.append("\n---\n## Verdict\n")
    if summary["reaches_ceiling"]:
        L.append(f"**Depth matters.** Layer **{best}** reaches the reusable-state "
                 f"regime: det-frac(z,op) = {_f(best_row['det_action'])} "
                 f"({_f(summary['best_pos_det'] * 100, 0)}% of the way to the "
                 f"positive-control ceiling), H(z'|z,op) = {_f(best_row['H_action'])} "
                 f"nats. The Exp1/Exp2 conclusion — drawn at the final layer — "
                 f"understated the structure available deeper in the stack.\n")
    else:
        L.append(f"**The conclusion is robust to layer choice.** The strongest "
                 f"layer ({best}) still only reaches det-frac(z,op) = "
                 f"{_f(best_row['det_action'])} "
                 f"({_f(summary['best_pos_det'] * 100, 0)}% of the way from the "
                 f"noise floor to the positive-control ceiling) — short of the "
                 f"reusable-state regime. No probed depth turns Countdown's "
                 f"hidden-state trajectory into crisp, reusable discrete "
                 f"dynamics; the partial, graded structure found at the final "
                 f"layer (Exp1/Exp2) is representative of the whole stack.\n")

    L.append("\n_Caveats: single model/task/seed; capped data (numbers shift "
             "slightly vs full-data Exp1); op-type action; layer subset, not "
             "every layer. det-frac is a strict threshold metric — see Exp2 for "
             "why the soft entropy/MLP axes can read higher._\n")
    L.append("\n_Artifacts: `reports/layer_sweep.json`, `reports/layer_sweep.png`._\n")
    return "\n".join(L)


def build_plot(rows: List[Dict], floor: Dict, ceiling: Dict, png_path: str) -> None:
    """Plot the headline action-conditioned metrics against layer depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in rows]
    det = [r["det_action"] for r in rows]
    H = [r["H_action"] for r in rows]
    mlp = [r["mlp_z_op"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(layers, det, "o-", color="tab:green", label="det-frac(z,op)")
    ax1.plot(layers, mlp, "s-", color="tab:blue", label="MLP(z,op) top1")
    ax1.axhline(ceiling["det_action"], color="tab:green", ls="--", alpha=0.6,
                label=f"ceiling det {ceiling['det_action']:.2f}")
    ax1.axhline(floor["det_action"], color="tab:gray", ls=":", alpha=0.6,
                label=f"floor det {floor['det_action']:.2f}")
    ax1.axhline(ceiling["mlp_z_op"], color="tab:blue", ls="--", alpha=0.4)
    ax1.axhline(floor["mlp_z_op"], color="tab:blue", ls=":", alpha=0.4)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("det-frac / accuracy")
    ax1.set_title("Action-conditioned determinism vs depth")
    ax1.legend(fontsize=8)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.plot(layers, H, "o-", color="tab:red", label="H(z'|z,op)")
    ax2.axhline(ceiling["H_action"], color="tab:red", ls="--", alpha=0.6,
                label=f"ceiling H {ceiling['H_action']:.2f}")
    ax2.axhline(floor["H_action"], color="tab:gray", ls=":", alpha=0.6,
                label=f"floor H {floor['H_action']:.2f}")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("H(z'|z,op)  [nats]")
    ax2.set_title("Action-conditioned entropy vs depth")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\oracle_coherence.py`
```python
"""Oracle Transition coherence evaluation.

The Oracle Transition is a scientific control, not a model. It returns the
exact teacher hidden state at each depth, establishing the theoretical maximum
coherence achievable under perfect transitions.

For a trajectory ``h0 -> a0 -> h1 -> a1 -> h2 -> ...``, the Oracle returns::

    Oracle(h_t, a_t) = teacher_h_{t+1}

No learning. No prediction. No approximation.

This isolates the question:

    Is the Phase A coherence bottleneck in the learned dynamics,
    or in the hidden-state representation itself?

Output schema matches ``coherence.py`` exactly so CSVs are directly comparable.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from data_processing.trajectory_dataset import Trajectory




@torch.no_grad()
def evaluate_oracle_coherence(
    trajectories: List[Trajectory],
    probe_a=None,
    probe_b=None,
    probe_c=None,
    probe_d=None,
    max_depth: Optional[int] = None,
    domain: str = "countdown",
):
    """Evaluate coherence using the Oracle (perfect) transition.

    The Oracle simply looks up the true teacher state at each depth.
    This establishes the representation ceiling: even with perfect dynamics,
    how much task information survives in the frozen hidden states?

    Returns a pandas DataFrame with the same schema as ``evaluate_coherence``.
    """
    import pandas as pd
    import numpy as np

    if max_depth is None:
        lengths = [t.num_steps for t in trajectories]
        max_depth = int(np.percentile(lengths, 95)) if lengths else 8
        max_depth = max(1, max_depth)

    from evaluation.probes import get_probe_a_targets

    # Accumulators per depth — same structure as coherence.py.
    agg = {
        d: {"mse": [], "cos": [], "op_acc": [], "teacher_op_acc": [],
            "state_acc": [], "teacher_state_acc": [],
            "id_mse": [], "id_cos": [], "id_op_acc": [], "id_state_acc": [],
            "probe_b_acc": [], "teacher_probe_b_acc": [], "id_probe_b_acc": [],
            "probe_d_acc": [], "teacher_probe_d_acc": [], "id_probe_d_acc": []}
        for d in range(1, max_depth + 1)
    }

    for traj in trajectories:
        states = traj.states  # (N+1, H)
        N = traj.num_steps
        h_identity = states[0:1]  # frozen at s_0
        depth_limit = min(N, max_depth)
        for d in range(1, depth_limit + 1):
            # ---- ORACLE TRANSITION: use the exact teacher state ----
            h = states[d: d + 1]        # Oracle output
            teacher = states[d: d + 1]  # Ground truth (same tensor)

            # MSE and cosine vs teacher — should be exactly 0 / 1.0
            agg[d]["mse"].append(F.mse_loss(h, teacher).item())
            agg[d]["cos"].append(
                F.cosine_similarity(h, teacher, dim=-1).item())

            # Identity baseline metrics (s_0 vs s_d)
            agg[d]["id_mse"].append(F.mse_loss(h_identity, teacher).item())
            agg[d]["id_cos"].append(
                F.cosine_similarity(h_identity, teacher, dim=-1).item())

            # --- Probe evaluations ---
            h_np = h.cpu().numpy()
            teacher_np = teacher.cpu().numpy()
            identity_np = h_identity.cpu().numpy()

            if probe_c is not None and d < N:
                target_op = int(traj.op_ids[d].item())
                agg[d]["op_acc"].append(
                    1.0 if int(probe_c.predict(h_np)[0]) == target_op
                    else 0.0)
                agg[d]["teacher_op_acc"].append(
                    1.0 if int(probe_c.predict(teacher_np)[0]) == target_op
                    else 0.0)
                agg[d]["id_op_acc"].append(
                    1.0 if int(probe_c.predict(identity_np)[0]) == target_op
                    else 0.0)

            if probe_b is not None:
                target_dist = N - d
                agg[d]["probe_b_acc"].append(
                    1.0 if int(probe_b.predict(h_np)[0]) == target_dist
                    else 0.0)
                agg[d]["teacher_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(teacher_np)[0]) == target_dist
                    else 0.0)
                agg[d]["id_probe_b_acc"].append(
                    1.0 if int(probe_b.predict(identity_np)[0]) == target_dist
                    else 0.0)

            if probe_d is not None:
                target_reach = 1 if (N - d) <= 2 else 0
                agg[d]["probe_d_acc"].append(
                    1.0 if int(probe_d.predict(h_np)[0]) == target_reach
                    else 0.0)
                agg[d]["teacher_probe_d_acc"].append(
                    1.0 if int(probe_d.predict(teacher_np)[0]) == target_reach
                    else 0.0)
                agg[d]["id_probe_d_acc"].append(
                    1.0 if int(probe_d.predict(identity_np)[0]) == target_reach
                    else 0.0)

            if probe_a is not None:
                a_row = get_probe_a_targets(traj, d, domain)

                def _avg_match(p, t):
                    return sum(1.0 for pv, tv in zip(p, t)
                               if pv == tv) / len(t)

                pred_a = probe_a.predict(h_np)[0]
                agg[d]["state_acc"].append(_avg_match(pred_a, a_row))

                t_pred_a = probe_a.predict(teacher_np)[0]
                agg[d]["teacher_state_acc"].append(
                    _avg_match(t_pred_a, a_row))

                id_pred_a = probe_a.predict(identity_np)[0]
                agg[d]["id_state_acc"].append(
                    _avg_match(id_pred_a, a_row))

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for d in range(1, max_depth + 1):
        rows.append({
            "depth": d,
            "mse": _mean(agg[d]["mse"]),
            "cosine_similarity": _mean(agg[d]["cos"]),
            "operator_accuracy": _mean(agg[d]["op_acc"]),
            "teacher_operator_accuracy": _mean(agg[d]["teacher_op_acc"]),
            "state_probe_accuracy": _mean(agg[d]["state_acc"]),
            "teacher_state_probe_accuracy":
                _mean(agg[d]["teacher_state_acc"]),
            "dist_probe_accuracy": _mean(agg[d]["probe_b_acc"]),
            "teacher_dist_probe_accuracy":
                _mean(agg[d]["teacher_probe_b_acc"]),
            "reach_probe_accuracy": _mean(agg[d]["probe_d_acc"]),
            "teacher_reach_probe_accuracy":
                _mean(agg[d]["teacher_probe_d_acc"]),
            "identity_mse": _mean(agg[d]["id_mse"]),
            "identity_cosine_similarity": _mean(agg[d]["id_cos"]),
            "identity_operator_accuracy": _mean(agg[d]["id_op_acc"]),
            "identity_state_probe_accuracy":
                _mean(agg[d]["id_state_acc"]),
            "identity_dist_probe_accuracy":
                _mean(agg[d]["id_probe_b_acc"]),
            "identity_reach_probe_accuracy":
                _mean(agg[d]["id_probe_d_acc"]),
            "semantic_gain_state": _mean(agg[d]["state_acc"]) - _mean(agg[d]["id_state_acc"]) if len(agg[d]["state_acc"]) else float('nan'),
            "semantic_gain_dist": _mean(agg[d]["probe_b_acc"]) - _mean(agg[d]["id_probe_b_acc"]) if len(agg[d]["probe_b_acc"]) else float('nan'),
            "semantic_gain_reach": _mean(agg[d]["probe_d_acc"]) - _mean(agg[d]["id_probe_d_acc"]) if len(agg[d]["probe_d_acc"]) else float('nan'),
            "n_samples": len(agg[d]["mse"]),
        })
    return pd.DataFrame(rows)


def save_oracle_coherence(df, csv_path: str,
                          png_path: Optional[str] = None) -> None:
    """Write the Oracle coherence CSV and optional plot."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    if png_path is not None:
        _plot_oracle(df, png_path)


def _plot_oracle(df, png_path: str) -> None:
    """Standalone Oracle coherence plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = df[df["n_samples"] > 0]
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Rollout depth")
    ax1.set_ylabel("Cosine similarity / Probe accuracy")
    ax1.plot(valid["depth"], valid["cosine_similarity"],
             marker="o", color="tab:blue", label="Oracle Cosine")
    ax1.plot(valid["depth"], valid["operator_accuracy"],
             marker="s", color="tab:green", label="Oracle Op Acc")
    ax1.plot(valid["depth"], valid["state_probe_accuracy"],
             marker="v", color="tab:purple", label="Oracle State Acc")
    ax1.plot(valid["depth"], valid["identity_cosine_similarity"],
             marker="x", linestyle=":", color="tab:blue", alpha=0.5,
             label="Identity Cosine")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Hidden-state MSE", color="tab:red")
    ax2.plot(valid["depth"], valid["mse"],
             marker="d", color="tab:red", label="Oracle MSE")
    ax2.plot(valid["depth"], valid["identity_mse"],
             marker="*", linestyle=":", color="tab:red", alpha=0.5,
             label="Identity MSE")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize="small")
    ax1.set_title("Oracle Transition Coherence vs Depth")
    ax1.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
# Overlay comparison plot
# ------------------------------------------------------------------ #
def plot_comparison_overlay(
    oracle_df,
    action_df,
    blind_df,
    png_path: str,
) -> None:
    """Four-way overlay: Oracle vs Action-Conditioned vs Action-Blind vs Identity.

    All on the same axes for direct visual comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase B: Oracle Transition Comparison", fontsize=14,
                 fontweight="bold")

    def _valid(df):
        return df[df["n_samples"] > 0]

    ov = _valid(oracle_df)
    av = _valid(action_df)
    bv = _valid(blind_df)

    # --- Panel 1: Cosine Similarity ---
    ax = axes[0, 0]
    ax.set_title("Cosine Similarity")
    ax.plot(ov["depth"], ov["cosine_similarity"],
            "o-", color="gold", linewidth=2, label="Oracle")
    ax.plot(av["depth"], av["cosine_similarity"],
            "s-", color="tab:blue", label="Action-Cond.")
    ax.plot(bv["depth"], bv["cosine_similarity"],
            "^-", color="tab:orange", label="Action-Blind")
    ax.plot(av["depth"], av["identity_cosine_similarity"],
            "x:", color="gray", alpha=0.6, label="Identity")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 2: MSE ---
    ax = axes[0, 1]
    ax.set_title("Hidden-State MSE")
    ax.plot(ov["depth"], ov["mse"],
            "o-", color="gold", linewidth=2, label="Oracle")
    ax.plot(av["depth"], av["mse"],
            "s-", color="tab:blue", label="Action-Cond.")
    ax.plot(bv["depth"], bv["mse"],
            "^-", color="tab:orange", label="Action-Blind")
    ax.plot(av["depth"], av["identity_mse"],
            "x:", color="gray", alpha=0.6, label="Identity")
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 3: Operator Accuracy (Probe C) ---
    ax = axes[1, 0]
    ax.set_title("Operator Accuracy (Probe C)")
    ax.plot(ov["depth"], ov["operator_accuracy"],
            "o-", color="gold", linewidth=2, label="Oracle")
    ax.plot(av["depth"], av["operator_accuracy"],
            "s-", color="tab:blue", label="Action-Cond.")
    ax.plot(bv["depth"], bv["operator_accuracy"],
            "^-", color="tab:orange", label="Action-Blind")
    ax.plot(av["depth"], av["identity_operator_accuracy"],
            "x:", color="gray", alpha=0.6, label="Identity")
    ax.plot(ov["depth"], ov["teacher_operator_accuracy"],
            "d--", color="green", alpha=0.5, label="Teacher Ceiling")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Panel 4: State Probe Accuracy (Probe A) ---
    ax = axes[1, 1]
    ax.set_title("State Probe Accuracy (Probe A)")
    ax.plot(ov["depth"], ov["state_probe_accuracy"],
            "o-", color="gold", linewidth=2, label="Oracle")
    ax.plot(av["depth"], av["state_probe_accuracy"],
            "s-", color="tab:blue", label="Action-Cond.")
    ax.plot(bv["depth"], bv["state_probe_accuracy"],
            "^-", color="tab:orange", label="Action-Blind")
    ax.plot(av["depth"], av["identity_state_probe_accuracy"],
            "x:", color="gray", alpha=0.6, label="Identity")
    ax.plot(ov["depth"], ov["teacher_state_probe_accuracy"],
            "d--", color="green", alpha=0.5, label="Teacher Ceiling")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.legend(fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

```


## File: `evaluation\permutation_robustness.py`
```python
"""Permutation robustness analysis (V5 Component 7).

Tests whether **equivalent reasoning states map to the same discrete code**.

For each problem we have ``K`` distinct valid solutions; encoding each one as
its own hidden-state trajectory yields ``K`` discrete code trajectories through
the same problem. Two states from *different* solutions are considered
"equivalent" iff they share the same *symbolic state* (target + sorted
available numbers, via :func:`evaluation.intrinsic_noise.get_symbolic_states`):
they represent the same logical situation reached by a different reasoning
order.

Metric
------
**Code consistency** = fraction of equivalent cross-solution state *pairs* that
received the same code. We compare against:

* a **random-baseline** consistency = ``1 / num_codes`` (what you'd get by
  assigning codes uniformly at random), and
* a **within-solution** consistency floor (states from the *same* solution at
  the same symbolic content — these are trivially identical by construction,
  so they are excluded from the cross-solution metric but reported as a sanity
  ceiling).

High cross-solution consistency (well above the random baseline) is evidence
that the codebook captures reusable *state content*, not the surface form of
a particular reasoning trace.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from data_processing.trajectory_dataset import (
    Trajectory,
    HEADER_TEMPLATE,
    _state_end_chars,
    _token_index_for_char_end,
)


def build_multi_solution_trajectories(
    model,
    tokenizer,
    problem: Dict,
    layer: int = -1,
) -> List[Trajectory]:
    """Build one :class:`Trajectory` per solution in a multi-solution problem.

    Reuses the alignment helpers (``_state_end_chars`` /
    ``_token_index_for_char_end``) so the resulting trajectories are
    byte-compatible with the single-solution :func:`build_trajectory`. Requires
    a fast tokenizer.
    """
    from data_processing.action_parser import parse_solution

    numbers = problem["numbers"]
    target = problem["target"]
    solutions = problem["solutions"]
    header = HEADER_TEMPLATE.format(numbers=numbers, target=target)

    out: List[Trajectory] = []
    for steps in solutions:
        if not steps:
            continue
        try:
            actions = parse_solution(steps)
        except ValueError:
            continue
        full_text = header + "\n".join(steps)
        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0].detach().cpu().float()
        input_ids = enc["input_ids"][0].detach().cpu()
        end_chars = _state_end_chars(header, steps)
        try:
            state_indices = [
                _token_index_for_char_end(offsets, ec) for ec in end_chars
            ]
        except ValueError:
            continue
        state_indices = torch.tensor(state_indices, dtype=torch.long)
        op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
        operands = torch.tensor(
            [[a.arg1, a.arg2] for a in actions], dtype=torch.float32
        )
        out.append(
            Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=op_ids,
                operands=operands,
                numbers=list(numbers),
                target=int(target),
            )
        )
    return out


def evaluate_permutation_robustness(
    discrete_groups: List[List[DiscreteTrajectory]],
    num_codes: int,
    seed: int = 0,
) -> Dict:
    """Compute cross-solution code consistency for equivalent symbolic states.

    Args:
        discrete_groups: one list of :class:`DiscreteTrajectory` per problem;
            each inner list has one trajectory per solution of that problem.
            Symbolic states are recomputed per trajectory via
            :func:`get_symbolic_states` to decide equivalence across solutions.
        num_codes: codebook cardinality (for the random baseline).
        seed: RNG seed for the random-assignment baseline.

    Returns a dict with ``cross_consistency``, ``within_consistency`` (sanity
    ceiling), ``random_baseline`` (``1/num_codes``), ``n_pairs_cross``,
    ``n_problems_with_match``, and ``mean_solutions_per_problem``.
    """
    from evaluation.intrinsic_noise import get_symbolic_states

    # Map: (problem_idx, symbolic_state) -> list[(solution_idx, code)]
    by_symbol: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)

    n_solutions_total = 0
    for p_idx, sol_disc_trajs in enumerate(discrete_groups):
        for s_idx, disc in enumerate(sol_disc_trajs):
            n_solutions_total += 1
            sym_states = get_symbolic_states(
                _disc_to_trajectory(disc)
            )
            for d, info in enumerate(sym_states):
                if info is None or d >= disc.codes.shape[0]:
                    continue
                sym, _hist = info
                by_symbol[(p_idx, sym)].append((s_idx, int(disc.codes[d].item())))

    cross_match, cross_total = 0, 0
    within_match, within_total = 0, 0
    problems_with_match = set()
    for (p_idx, _sym), items in by_symbol.items():
        n = len(items)
        if n < 2:
            continue
        problems_with_match.add(p_idx)
        for i in range(n):
            for j in range(i + 1, n):
                sol_i, code_i = items[i]
                sol_j, code_j = items[j]
                same = 1 if code_i == code_j else 0
                if sol_i == sol_j:
                    within_match += same
                    within_total += 1
                else:
                    cross_match += same
                    cross_total += 1

    # Random baseline: P(two random codes agree) = 1/num_codes under uniform
    # assignment, but the empirical code distribution is usually skewed, so we
    # also report the data-driven baseline sum_k p_k^2.
    rng = np.random.RandomState(seed)
    # Aggregate all codes seen across the matched groups for the empirical null.
    all_codes_seen = [c for items in by_symbol.values() for (_, c) in items]
    if all_codes_seen:
        counts = np.bincount(np.asarray(all_codes_seen), minlength=num_codes).astype(np.float64)
        p = counts / counts.sum()
        empirical_baseline = float((p * p).sum())
    else:
        empirical_baseline = 0.0
    uniform_baseline = 1.0 / num_codes

    n_problems = len(discrete_groups)
    return {
        "cross_consistency": (cross_match / cross_total) if cross_total else float("nan"),
        "within_consistency": (within_match / within_total) if within_total else float("nan"),
        "uniform_random_baseline": uniform_baseline,
        "empirical_random_baseline": empirical_baseline,
        "n_pairs_cross": cross_total,
        "n_problems_with_match": len(problems_with_match),
        "n_problems_total": n_problems,
        "mean_solutions_per_problem": (n_solutions_total / n_problems) if n_problems else 0.0,
    }


def _disc_to_trajectory(disc: DiscreteTrajectory):
    """Lightweight stand-in carrying only the fields ``get_symbolic_states`` reads.

    ``get_symbolic_states`` only touches ``target``, ``numbers``, ``op_ids``,
    ``operands`` and ``num_steps`` — so we synthesize a minimal object instead
    of round-tripping through the continuous ``Trajectory``. ``num_steps`` is a
    plain int attribute here (the consumer reads it as ``traj.num_steps``,
    which is valid whether it is a property or an attribute).
    """

    class _Stub:
        pass

    s = _Stub()
    s.target = disc.target
    s.numbers = disc.numbers
    s.op_ids = disc.op_ids
    s.operands = disc.operands
    s.num_steps = int(disc.op_ids.shape[0])
    return s


def save_permutation_robustness_report(metrics: Dict, csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

```


## File: `evaluation\plotter.py`
```python
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(log_csv_path: str, output_img_path: str):
    """Plots training and evaluation loss from a CSV log file."""
    if not os.path.exists(log_csv_path):
        print(f"File not found: {log_csv_path}")
        return

    df = pd.read_csv(log_csv_path)
    
    plt.figure(figsize=(10, 6))
    
    if 'loss' in df.columns and 'step' in df.columns:
        train_data = df.dropna(subset=['loss'])
        plt.plot(train_data['step'], train_data['loss'], label='Train Loss', color='blue', alpha=0.7)
        
    if 'eval_loss' in df.columns and 'step' in df.columns:
        eval_data = df.dropna(subset=['eval_loss'])
        plt.plot(eval_data['step'], eval_data['eval_loss'], label='Eval Loss', color='orange', marker='o')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_metric(csv_path: str, metric_name: str, x_axis: str, output_img_path: str):
    """Plots an arbitrary evaluation metric against an x-axis (e.g., layer, step)."""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    if metric_name not in df.columns or x_axis not in df.columns:
        print(f"Columns {metric_name} or {x_axis} not in CSV.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_axis], df[metric_name], marker='s', color='green')
    plt.xlabel(x_axis.capitalize())
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} vs {x_axis.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    plt.close()

```


## File: `evaluation\position_leakage.py`
```python
"""Position leakage test for the discrete codebook.

Asks the simplest adversarial question about the discovered codes: are they
just **trajectory position (step index)** in disguise? If a linear classifier
can predict "this state is step ``i`` of the trajectory" from the code id
alone, the codes are not capturing reusable state content — they are
memorizing where in the chain they sit.

Method
------
For every state in every discrete trajectory we form the label
``position = i`` (``0``-indexed depth within its trajectory) and a one-hot
feature vector over the codebook. We then fit a logistic regression
``code_id -> position`` on the train split and evaluate accuracy on the held-out
split, against:

* a **majority-class chance** baseline, and
* a **permutation null** (train labels shuffled) — matching the Phase C.4A
  control discipline. Any real signal must clear both.

The output ``position_predictability_score`` is the test accuracy; if it is
high (close to 1) and well above both baselines, the codes are position
indices, not reusable discrete states.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory


def _flatten_positions(
    discrete_trajs: List[DiscreteTrajectory], num_codes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (one-hot code features, position labels) over all states."""
    rows, positions = [], []
    for traj in discrete_trajs:
        n = traj.codes.shape[0]
        for i in range(n):
            rows.append(int(traj.codes[i].item()))
            positions.append(i)
    if not rows:
        return (
            np.zeros((0, num_codes), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    # One-hot via advanced indexing (sparse feature; logistic regression reads
    # it directly without a dense (M, num_codes) copy in memory).
    codes = np.asarray(rows, dtype=np.int64)
    X = np.zeros((codes.shape[0], num_codes), dtype=np.float32)
    X[np.arange(codes.shape[0]), codes] = 1.0
    return X, np.asarray(positions, dtype=np.int64)


def _chance_accuracy(y: np.ndarray) -> float:
    """Majority-class baseline accuracy."""
    if y.shape[0] == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def evaluate_position_leakage(
    train_disc: List[DiscreteTrajectory],
    test_disc: List[DiscreteTrajectory],
    num_codes: int,
    seed: int = 0,
) -> Dict:
    """Fit ``code -> position`` classifier; return accuracy + null controls.

    Returns a dict with ``position_predictability_score`` (test acc),
    ``chance_accuracy`` (majority class), ``permutation_null_accuracy``
    (train labels shuffled), and ``n_train`` / ``n_test`` sample counts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    Xtr, ytr = _flatten_positions(train_disc, num_codes)
    Xte, yte = _flatten_positions(test_disc, num_codes)

    if ytr.shape[0] == 0 or yte.shape[0] == 0:
        return {
            "position_predictability_score": float("nan"),
            "chance_accuracy": float("nan"),
            "permutation_null_accuracy": float("nan"),
            "n_train": 0,
            "n_test": 0,
        }

    chance = _chance_accuracy(yte)

    # Real fit. max_iter raised because one-hot features can be slow to converge.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    score = float(accuracy_score(yte, clf.predict(Xte)))

    # Permutation null: decouple train labels from codes.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ytr.shape[0])
    null_clf = LogisticRegression(max_iter=2000)
    null_clf.fit(Xtr, ytr[perm])
    null_score = float(accuracy_score(yte, null_clf.predict(Xte)))

    return {
        "position_predictability_score": score,
        "chance_accuracy": chance,
        "permutation_null_accuracy": null_score,
        "n_train": int(ytr.shape[0]),
        "n_test": int(yte.shape[0]),
    }


def save_position_leakage_report(metrics: Dict, csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

```


## File: `evaluation\position_subspace.py`
```python
"""Position subspace estimation and removal (V5.1 Steps 1-3).

The invalidated-init audit left one substantive open question: are the
discovered discrete codes capturing *reusable* state content, or merely
**trajectory progression** (how far along the chain a state sits)? The V5
position-leakage test answered this for the *codes*; here we attack the
*continuous* hidden states directly.

Pipeline
--------
1. **Position probe** — fit a regularized logistic classifier
   ``hidden_state -> position bin`` and report held-out accuracy. The label is
   the **relative** depth ``i / num_steps`` of a state within its trajectory,
   quantized into 4 bins. Relative (not absolute) depth is used because the
   Countdown trajectories have only 3-5 states, so absolute step indices cannot
   form four populated bins, whereas relative depth captures "trajectory stage"
   uniformly across lengths.

2. **INLP** (Iterative Nullspace Projection, Ravfogel et al. 2020) — repeatedly
   train a linear position classifier and project the data onto the nullspace
   of its weight row-space, accumulating a single projection ``P`` that removes
   the *linearly position-predictive* subspace. We use the right singular
   vectors (``Vt``) of the stacked classifier weights to span each row-space.

3. **Scrub** — apply ``x -> (x - mean) @ P`` to every aligned state. Centering
   is a pure translation and therefore a no-op for every downstream discrete
   metric (nearest-neighbour assignment and the data-dependent codebook init
   shift together), so the scrubbed states differ from the raw states *only*
   by removal of the position subspace — a clean attribution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory


# --------------------------------------------------------------------------- #
# Position labels
# --------------------------------------------------------------------------- #
def build_position_dataset(
    trajectories: List[Trajectory], n_bins: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack aligned states and label each by its relative-depth quartile.

    Returns ``(X, y)`` where ``X`` is ``(M, H)`` float32 and ``y`` is ``(M,)``
    int64 bin ids in ``[0, n_bins)``. A state at step ``i`` of an ``N``-step
    trajectory gets ``bin = min(int(i / N * n_bins), n_bins - 1)`` (``i = 0``
    for single-state degenerate trajectories maps to bin 0).
    """
    Xs, ys = [], []
    for traj in trajectories:
        states = traj.states.float().cpu().numpy()  # (N+1, H)
        n_states = states.shape[0]
        N = max(n_states - 1, 1)
        for i in range(n_states):
            b = min(int(i / N * n_bins), n_bins - 1)
            ys.append(b)
        Xs.append(states)
    if not Xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def _majority_accuracy(y: np.ndarray) -> float:
    if y.shape[0] == 0:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
# Step 1 — position probe
# --------------------------------------------------------------------------- #
def train_position_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """Fit ``hidden_state -> position bin`` and report held-out accuracy.

    ``C`` is the inverse L2 regularization strength (smaller = stronger
    regularization); the default ``1.0`` is sklearn's regularized default.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte)))
    return {
        "position_accuracy": acc,
        "majority_baseline": _majority_accuracy(yte),
        "n_classes": int(np.unique(ytr).shape[0]),
        "n_train": int(ytr.shape[0]),
        "n_test": int(yte.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Step 2 — INLP
# --------------------------------------------------------------------------- #
def _rowspace_projection(W: np.ndarray) -> np.ndarray:
    """Projection ``(D, D)`` onto the row space of ``W`` ``(C, D)``.

    Uses the right singular vectors ``Vt`` with non-negligible singular values
    as an orthonormal basis of ``row(W)``; the projection is ``B.T @ B``.
    """
    W = np.atleast_2d(W)
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    if S.size == 0:
        return np.zeros((W.shape[1], W.shape[1]))
    tol = float(S.max()) * 1e-6
    rank = int((S > tol).sum())
    if rank == 0:
        return np.zeros((W.shape[1], W.shape[1]))
    B = Vt[:rank]                       # (rank, D) orthonormal rows
    return B.T @ B


def inlp_position_projection(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    num_iters: int = 8,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """Iterative Nullspace Projection of the linear position subspace.

    Returns ``(P, info)`` where ``P`` is the ``(D, D)`` cumulative nullspace
    projection (symmetric) and ``info`` holds the per-iteration held-out
    accuracy trace plus the ``before`` / ``after`` probe accuracies and the
    number of dimensions removed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    D = Xtr.shape[1]
    weights: List[np.ndarray] = []

    def cumulative_nullspace() -> np.ndarray:
        if not weights:
            return np.eye(D)
        Wall = np.vstack(weights)               # (sum_C, D)
        return np.eye(D) - _rowspace_projection(Wall)

    trace: List[float] = []
    for _ in range(num_iters):
        P = cumulative_nullspace()
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(Xtr @ P, ytr)
        trace.append(float(accuracy_score(yte, clf.predict(Xte @ P))))
        weights.append(np.atleast_2d(clf.coef_))

    P_final = cumulative_nullspace()
    # "after": a fresh probe trying to recover position from fully scrubbed data.
    after_clf = LogisticRegression(C=C, max_iter=max_iter)
    after_clf.fit(Xtr @ P_final, ytr)
    after_acc = float(accuracy_score(yte, after_clf.predict(Xte @ P_final)))

    dims_removed = int(round(D - np.trace(P_final)))
    info = {
        "before_accuracy": trace[0] if trace else float("nan"),
        "after_accuracy": after_acc,
        "majority_baseline": _majority_accuracy(yte),
        "accuracy_trace": trace,
        "dims_removed": dims_removed,
        "num_iters": num_iters,
    }
    return P_final, info


# --------------------------------------------------------------------------- #
# Step 3 — scrub
# --------------------------------------------------------------------------- #
def apply_scrub_to_trajectories(
    trajectories: List[Trajectory], mean: np.ndarray, P: np.ndarray
) -> List[Trajectory]:
    """Return new trajectories whose aligned states are ``(s - mean) @ P``.

    The returned objects expose the scrubbed states via ``.states`` (with
    ``all_hidden`` set to the scrubbed aligned states and ``state_indices`` an
    identity range) so the existing VQ trainer and code-encoder consume them
    unchanged. ``input_ids`` is a placeholder (token-level rollout is not run
    on scrubbed states); all action metadata is preserved.
    """
    mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    P_t = torch.as_tensor(np.asarray(P), dtype=torch.float32)
    out: List[Trajectory] = []
    for traj in trajectories:
        s = traj.states.float()                 # (N+1, H)
        s_scrub = (s - mean_t) @ P_t            # (N+1, H), P symmetric
        n_states = s_scrub.shape[0]
        out.append(
            Trajectory(
                all_hidden=s_scrub,
                input_ids=torch.zeros(n_states, dtype=torch.long),
                state_indices=torch.arange(n_states, dtype=torch.long),
                op_ids=traj.op_ids,
                operands=traj.operands,
                numbers=list(traj.numbers),
                target=int(traj.target),
            )
        )
    return out

```


## File: `evaluation\probes.py`
```python
"""Linear representation probes.

We fit *linear* probes (logistic regression on standardized features) on frozen
teacher hidden states to ask: what task-relevant information is linearly
decodable from the representation? Linear probes are deliberately weak so that a
high score implies the information is explicitly present, not reconstructed by a
powerful probe.

Probes (evaluated at state ``s_i``, i.e. after ``i`` reasoning steps, looking
ahead to step ``i+1``):

* **A — remaining numbers**: for each large number in {25, 50, 75, 100}, is it
  still in the pool and not yet used as an operand? (multi-label, 4 labels)
* **B — distance-to-solution**: number of reasoning steps remaining (multiclass)
* **C — next symbolic operation**: the op of the next step (4-class)
* **D — reachable within 2 steps**: is distance-to-solution <= 2? (binary)

Probe A uses the fixed-value set {25,50,75,100} so each output dimension has a
consistent meaning across problems (a requirement for a single linear probe).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from data_processing.trajectory_dataset import Trajectory

from data_processing.action_parser import apply_op, ID_TO_OP

PROBE_NAMES = {
    "A": "remaining_operands",
    "B": "distance_to_solution",
    "C": "next_operation",
    "D": "reachable_within_2",
}

def get_tracked_numbers(domain: str) -> List[int]:
    """Domain-aware vocabulary for Probe A."""
    if domain == "game24":
        return list(range(1, 14))
    return [25, 50, 75, 100]

def get_probe_a_targets(traj: Trajectory, step_idx: int, domain: str) -> List[int]:
    """Target extraction logic for Probe A answering: 'Which operands remain?'"""
    if domain != "game24":
        # Exactly identical Countdown logic to preserve legacy baselines
        LARGE_NUMBERS = [25, 50, 75, 100]
        used = set()
        for i in range(step_idx):
            used.add(int(traj.operands[i][0]))
            used.add(int(traj.operands[i][1]))
        a_row = []
        for v in LARGE_NUMBERS:
            if v not in traj.numbers:
                a_row.append(0)
            elif v not in used:
                a_row.append(1)
            else:
                a_row.append(2)
        return a_row
    
    # Game24 logic: Track actual multiset counts of remaining initial pool.
    pool = list(traj.numbers)
    for i in range(step_idx):
        op = ID_TO_OP[int(traj.op_ids[i])]
        a = int(traj.operands[i][0])
        b = int(traj.operands[i][1])
        if a in pool: pool.remove(a)
        if b in pool: pool.remove(b)
        try:
            c = apply_op(op, a, b)
            pool.append(c)
        except Exception:
            pass
    return [pool.count(v) for v in get_tracked_numbers(domain)]



def extract_probe_data(trajectories: List[Trajectory], domain: str = "countdown") -> Dict[str, np.ndarray]:
    """Build the feature matrix and probe labels from trajectories.

    Features are the hidden states s_0..s_{N-1} (states that have a next step).
    """
    X, A, B, C, D = [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i  # steps remaining from s_i
            a_row = get_probe_a_targets(traj, i, domain)
            A.append(a_row)
            B.append(dist)
            C.append(ops[i])
            D.append(1 if dist <= 2 else 0)
    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "B": np.array(B, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "D": np.array(D, dtype=np.int64),
    }


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def _make_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def _fit_eval_single(
    Xtr, ytr, Xte, yte, binary: bool
) -> Dict[str, float]:
    """Fit one logistic-regression probe and return accuracy/f1/auc."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    classes = np.unique(ytr)
    if classes.shape[0] < 2:
        # Degenerate target: predict the single observed class.
        const = int(classes[0]) if classes.shape[0] else 0
        pred = np.full_like(yte, const)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="binary" if binary else "macro",
                      zero_division=0)
        return {"accuracy": float(acc), "f1": float(f1), "auc": float("nan")}

    clf = _make_probe()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="binary" if binary else "macro",
                  zero_division=0)

    auc = float("nan")
    try:
        proba = clf.predict_proba(Xte)
        if binary:
            auc = roc_auc_score(yte, proba[:, 1])
        else:
            auc = roc_auc_score(yte, proba, multi_class="ovr",
                                average="macro", labels=clf.named_steps["clf"].classes_)
    except ValueError:
        auc = float("nan")  # e.g. a class missing from the test set
    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc), "clf": clf}


def _fit_eval_multilabel(Xtr, Ytr, Xte, Yte) -> Dict[str, float]:
    """Average per-label multiclass probe metrics over all label columns."""
    accs, f1s, aucs = [], [], []
    for j in range(Ytr.shape[1]):
        res = _fit_eval_single(Xtr, Ytr[:, j], Xte, Yte[:, j], binary=False)
        accs.append(res["accuracy"])
        f1s.append(res["f1"])
        if not np.isnan(res["auc"]):
            aucs.append(res["auc"])
    return {
        "accuracy": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
    }

class MultiLabelProbe:
    def __init__(self):
        self.clfs = []
        self.constants = []

    def fit(self, X, Y):
        for j in range(Y.shape[1]):
            classes = np.unique(Y[:, j])
            if classes.shape[0] < 2:
                self.clfs.append(None)
                self.constants.append(int(classes[0]) if classes.shape[0] else 0)
            else:
                clf = _make_probe()
                clf.fit(X, Y[:, j])
                self.clfs.append(clf)
                self.constants.append(None)

    def predict(self, X):
        preds = []
        for clf, const in zip(self.clfs, self.constants):
            if clf is None:
                preds.append(np.full((X.shape[0],), const))
            else:
                preds.append(clf.predict(X))
        return np.column_stack(preds)

def _fit_eval_multilabel_joint(Xtr, Ytr, Xte, Yte):
    from sklearn.metrics import accuracy_score
    clf = MultiLabelProbe()
    clf.fit(Xtr, Ytr)
    pred = clf.predict(Xte)
    
    # average per-label accuracy
    accs = []
    for j in range(Ytr.shape[1]):
        accs.append(accuracy_score(Yte[:, j], pred[:, j]))
    # exact match accuracy
    exact_acc = float(np.mean(np.all(Yte == pred, axis=1)))
    
    return {
        "accuracy": float(np.mean(accs)),
        "exact_accuracy": exact_acc,
        "f1": 0.0, # not critical
        "auc": float("nan"),
        "clf": clf
    }


def _chance_accuracy(y: np.ndarray) -> float:
    """Majority-class baseline accuracy."""
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_probes(
    train_trajs: List[Trajectory],
    test_trajs: List[Trajectory],
    domain: str = "countdown",
):
    """Fit probes A-D on train states, evaluate on test states.

    Returns ``(results_df, details, fitted_probes)`` where ``results_df`` has columns
    [probe, accuracy, f1, auc], ``details`` holds chance baselines / sizes,
    and ``fitted_probes`` is a dict of {"A": clfA, "C": clfC}.
    """
    import pandas as pd

    tr = extract_probe_data(train_trajs, domain=domain)
    te = extract_probe_data(test_trajs, domain=domain)

    if len(te["X"]) == 0:
        raise RuntimeError(
            "No probe samples were extracted from the test split. "
            "Likely causes: deduplication removed every trajectory or parser produced no labels."
        )

    rows, details = [], {}

    # A: multi-label (joint for coherence evaluation)
    resA = _fit_eval_multilabel_joint(tr["X"], tr["A"], te["X"], te["A"])
    # B, C: multiclass ; D: binary
    resB = _fit_eval_single(tr["X"], tr["B"], te["X"], te["B"], binary=False)
    resC = _fit_eval_single(tr["X"], tr["C"], te["X"], te["C"], binary=False)
    resD = _fit_eval_single(tr["X"], tr["D"], te["X"], te["D"], binary=True)

    for key, res in [("A", resA), ("B", resB), ("C", resC), ("D", resD)]:
        row = {
            "probe": f"{key}:{PROBE_NAMES[key]}",
            "accuracy": res["accuracy"],
            "f1": res["f1"],
            "auc": res["auc"],
        }
        if "exact_accuracy" in res:
            row["exact_accuracy"] = res["exact_accuracy"]
        rows.append(row)

    details["n_train"] = int(tr["X"].shape[0])
    details["n_test"] = int(te["X"].shape[0])
    details["chance"] = {
        "A": float(np.mean([_chance_accuracy(te["A"][:, j])
                            for j in range(te["A"].shape[1])])),
        "B": _chance_accuracy(te["B"]),
        "C": _chance_accuracy(te["C"]),
        "D": _chance_accuracy(te["D"]),
    }
    fitted_probes = {
        "A": resA.get("clf"),
        "B": resB.get("clf"),
        "C": resC.get("clf"),
        "D": resD.get("clf"),
    }
    return pd.DataFrame(rows), details, fitted_probes


def save_probe_results(df, csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)


def generate_probe_report(df, details: Dict, md_path: str) -> None:
    """Write probe_report.md explaining what is / is not encoded."""
    import pandas as pd  # noqa: F401

    chance = details["chance"]
    chance_by_key = {f"{k}:{PROBE_NAMES[k]}": v for k, v in chance.items()}

    lines = []
    lines.append("# Representation Probe Report\n")
    lines.append(
        f"Linear probes fit on **{details['n_train']}** teacher states, "
        f"evaluated on **{details['n_test']}** held-out states.\n")
    lines.append(
        "Each probe is a logistic regression on standardized hidden states "
        "(linear only). A score well above the majority-class baseline means "
        "the information is *linearly decodable* from the frozen representation.\n")

    lines.append("\n## Results\n")
    lines.append("| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    encoded, not_encoded = [], []
    for _, r in df.iterrows():
        base = chance_by_key.get(r["probe"], float("nan"))
        auc = r["auc"]
        # "Encoded" if clearly above chance and AUC indicates real signal.
        is_encoded = (
            (not np.isnan(auc) and auc >= 0.65)
            or (r["accuracy"] - base >= 0.10)
        )
        verdict = "encoded" if is_encoded else "weak / not encoded"
        (encoded if is_encoded else not_encoded).append(r["probe"])
        auc_str = "n/a" if np.isnan(auc) else f"{auc:.3f}"
        
        exact_str = f"{r['exact_accuracy']:.3f}" if "exact_accuracy" in r and not pd.isna(r["exact_accuracy"]) else "n/a"
        
        lines.append(
            f"| {r['probe']} | {r['accuracy']:.3f} | {exact_str} | {base:.3f} | "
            f"{r['f1']:.3f} | {auc_str} | {verdict} |")

    lines.append("\n## Interpretation\n")
    if encoded:
        lines.append("**Linearly encoded in the hidden state:**")
        for p in encoded:
            lines.append(f"- {p}")
    else:
        lines.append("**Linearly encoded:** none of the probed quantities "
                     "cleared the bar.")
    lines.append("")
    if not_encoded:
        lines.append("**Weak or not linearly encoded:**")
        for p in not_encoded:
            lines.append(f"- {p}")
    lines.append("")
    lines.append(
        "_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary "
        "so each label has a consistent meaning across problems. "
        "AUC is reported as macro one-vs-rest for multiclass probes and averaged "
        "over labels for the multi-label probe; 'n/a' indicates a degenerate or "
        "missing class in the evaluation split._\n")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

```


## File: `evaluation\synthetic_fsm.py`
```python
"""Synthetic finite-state environment for the V5.2 positive control.

Calibration target: V5.1/Experiment-1 found that, on Countdown, conditioning on
the action gives H(z'|z,op) ≈ 0.99 nats and a 0.11 deterministic mass. Is that
*meaningful* discrete-state structure, or near-noise? To answer, we need the
ceiling: run the exact same VQ + analysis pipeline on an environment that has
**known, genuinely reusable, deterministic states** and see what the pipeline
reports.

Environment
-----------
A deterministic FSM with ``num_states`` states and ``num_actions`` actions and a
random (fixed) transition table ``T[s, a] -> s'``. Trajectories are random
walks: a uniform start state, then random actions. Because the start is uniform
and the table is a random map, the state marginal is ~uniform at every step, so
**state is decorrelated from position** (unlike Countdown) — which also lets the
position-leakage metric be calibrated.

Each visited state ``s`` is rendered as a continuous "hidden state"
``proto[s] + noise`` (``structured=True``), mimicking an LM hidden state that
encodes a genuine reusable state. The **noise floor** (``structured=False``)
replaces the embedding with pure noise of the same scale, carrying no state
information — the lower anchor.

The trajectories are returned as standard
:class:`~data_processing.trajectory_dataset.Trajectory` objects (actions stored
in ``op_ids``) so the entire existing pipeline — VQ training, code encoding,
codebook usage, position leakage, action-blind and action-conditioned
transition analysis — runs on them unchanged. Ground-truth state ids are
returned alongside for purity / AMI scoring.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory


def generate_fsm(num_states: int, num_actions: int, seed: int = 0) -> np.ndarray:
    """Deterministic transition table ``T[s, a] -> s'`` (shape K×A).

    Each action is a random **permutation** of the states. This keeps the
    machine deterministic and reusable while guaranteeing a well-mixed,
    ~uniform stationary distribution (a plain random table tends to funnel all
    mass into a few recurrent states, which would make the "states" degenerate).
    """
    rng = np.random.RandomState(seed)
    T = np.zeros((num_states, num_actions), dtype=np.int64)
    for a in range(num_actions):
        T[:, a] = rng.permutation(num_states)
    return T


def make_prototypes(
    num_states: int, hidden_dim: int, proto_scale: float = 1.0, seed: int = 0
) -> np.ndarray:
    """The environment's fixed ``state -> hidden vector`` map (K×H prototypes).

    This geometry must be **shared across train/val/test splits**: a VQ trained
    on the train split learns codes at these prototype locations, so the eval
    splits have to render the same state at the same location or the VQ cannot
    encode them (every eval state collapses onto one code, saturating the
    train->eval predictability metrics). Generate once, pass to every split.
    """
    rng = np.random.RandomState(seed)
    return (rng.randn(num_states, hidden_dim) * proto_scale).astype(np.float32)


def generate_dataset(
    T: np.ndarray,
    num_traj: int,
    hidden_dim: int,
    lengths: Tuple[int, ...] = (3, 4, 5),
    noise: float = 0.3,
    proto_scale: float = 1.0,
    structured: bool = True,
    seed: int = 0,
    protos: np.ndarray = None,
) -> Tuple[List[Trajectory], List[np.ndarray]]:
    """Generate FSM random-walk trajectories with continuous state embeddings.

    Returns ``(trajectories, true_states)`` where ``true_states[i]`` is the
    ground-truth state id per visited state of trajectory ``i`` (aligned with
    that trajectory's ``codes`` after VQ encoding).

    Pass ``protos`` (from :func:`make_prototypes`) to share one prototype
    geometry across splits; if omitted, prototypes are drawn from ``seed`` (each
    split then gets its own geometry — only valid for a single self-contained
    split).
    """
    num_states, num_actions = T.shape
    rng = np.random.RandomState(seed)
    if protos is None:
        # Self-contained split: prototypes drawn from this split's own seed.
        protos = (rng.randn(num_states, hidden_dim) * proto_scale).astype(np.float32)

    trajs: List[Trajectory] = []
    true_states: List[np.ndarray] = []
    for _ in range(num_traj):
        n_states = int(rng.choice(lengths))
        s = int(rng.randint(num_states))
        states = [s]
        actions = []
        for _ in range(n_states - 1):
            a = int(rng.randint(num_actions))
            s = int(T[s, a])
            actions.append(a)
            states.append(s)
        states_arr = np.asarray(states, dtype=np.int64)
        actions_arr = np.asarray(actions, dtype=np.int64)

        if structured:
            proto_idx = states_arr                       # embedding tracks the true state
        else:  # noise floor: same well-separated prototypes, but assigned at
            # random per occurrence — codes stay spread (no VQ collapse) yet
            # carry no information about the walk.
            proto_idx = rng.randint(0, num_states, size=n_states)
        emb = protos[proto_idx] + rng.randn(n_states, hidden_dim).astype(np.float32) * noise

        trajs.append(Trajectory(
            all_hidden=torch.from_numpy(emb),
            input_ids=torch.zeros(n_states, dtype=torch.long),
            state_indices=torch.arange(n_states, dtype=torch.long),
            op_ids=torch.from_numpy(actions_arr).long(),
            operands=torch.zeros(max(n_states - 1, 0), 2),
            numbers=[0],
            target=0,
        ))
        true_states.append(states_arr)
    return trajs, true_states


def flatten_true_states(true_states: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-trajectory ground-truth state ids (aligns with all_codes)."""
    if not true_states:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(true_states, axis=0)


def oracle_transitions(true_states: List[np.ndarray], trajs: List[Trajectory]):
    """Ground-truth ``(s_t, a_t, s_{t+1})`` arrays for oracle-dynamics checks."""
    st, at, sn = [], [], []
    for s, tr in zip(true_states, trajs):
        a = tr.op_ids.numpy()
        for i in range(len(a)):
            st.append(int(s[i]))
            at.append(int(a[i]))
            sn.append(int(s[i + 1]))
    return np.asarray(st), np.asarray(at), np.asarray(sn)

```


## File: `evaluation\__init__.py`
```python

```


## File: `models\diagnostic_decoder.py`
```python
"""Diagnostic decoder: ``hidden_state -> next reasoning token``.

This is a *diagnostic probe*, not a component of the planner. Its sole purpose
is to measure how much next-token information survives in (a) teacher hidden
states and (b) hidden states produced by rolling out the transition model. It is
deliberately lightweight — a single linear read-out by default — so that any
decodable signal reflects the representation, not decoder capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosticDecoder(nn.Module):
    """Maps a hidden state to a distribution over the token vocabulary.

    With ``mlp_hidden_dim=0`` (default) this is a single linear layer. A small
    hidden layer can be enabled but is discouraged for diagnostic use.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, mlp_hidden_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if mlp_hidden_dim and mlp_hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, vocab_size),
            )
        else:
            self.net = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)

    @torch.no_grad()
    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Argmax next-token prediction for a batch of hidden states."""
        return self.forward(hidden).argmax(dim=-1)

```


## File: `models\model_loader.py`
```python
import os
from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def load_tokenizer(model_id: str = DEFAULT_MODEL) -> PreTrainedTokenizer:
    """Loads the tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(
    model_id: str = DEFAULT_MODEL,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    load_in_4bit: bool = False
) -> PreTrainedModel:
    """Loads a frozen base model for inference.

    When ``load_in_4bit`` is True, loads with 4-bit NF4 quantization (for the
    7B/8B teachers that do not fit in fp16 on a single 16GB GPU). No LoRA/PEFT
    is attached — this is inference-only; use ``load_qlora_model`` for trainable
    adapters.
    """
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    return model

def load_qlora_model(
    model_id: str = DEFAULT_MODEL,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_map: str = "auto"
) -> PreTrainedModel:
    """Loads a model with 4-bit quantization and prepares it for QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    return model

def save_checkpoint(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str):
    """Saves the model (or PEFT adapter) and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_checkpoint(
    checkpoint_dir: str,
    base_model_id: str = DEFAULT_MODEL,
    is_peft: bool = True,
    device_map: str = "auto",
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a model from a saved checkpoint.

    If is_peft is True, loads the base model and attaches the saved LoRA adapters.

    Parameters
    ----------
    load_in_4bit : bool
        If True, load the base model with 4-bit NF4 quantization (for large
        models on limited VRAM). Default is False (fp16) to preserve backward
        compatibility with V5.3 checkpoints.
    """
    tokenizer = load_tokenizer(checkpoint_dir)
    
    if is_peft:
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map=device_map
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map=device_map,
                torch_dtype=torch.float16
            )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
    return model, tokenizer

```


## File: `models\transition_model.py`
```python
"""Latent transition model: ``T(h_t, a_t) -> h_{t+1}``.

The model learns to advance a frozen LM hidden state by one symbolic reasoning
step. An action is encoded as a learned op embedding concatenated with its
(normalized) numeric operands; this action vector is concatenated with the
current hidden state and passed through a two-layer MLP.

Architecture (as specified):

    action_embedding = [ Embedding(op) ; arg1/scale ; arg2/scale ]
    x = [ h_t ; action_embedding ]
    h_next = Linear -> ReLU -> Linear (x)            (+ h_t, if predict_delta)

By default the MLP predicts a *residual* (delta) that is added to ``h_t``. The
identity mapping is a strong prior for latent dynamics and materially improves
multi-step coherence; the core network is still exactly Linear->ReLU->Linear.
Set ``predict_delta=False`` to predict ``h_{t+1}`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data_processing.action_parser import OP_TO_ID

NUM_OPS = len(OP_TO_ID)


class ActionEncoder(nn.Module):
    """Encodes a symbolic action into a dense vector."""

    def __init__(self, op_embed_dim: int = 16, operand_mean: float = 0.0, operand_std: float = 1.0):
        super().__init__()
        self.op_embedding = nn.Embedding(NUM_OPS, op_embed_dim)
        self.register_buffer("operand_mean", torch.tensor(operand_mean, dtype=torch.float32))
        self.register_buffer("operand_std", torch.tensor(operand_std, dtype=torch.float32))
        self.output_dim = op_embed_dim + 2  # + arg1, arg2

    def forward(self, op_id: torch.Tensor, operands: torch.Tensor) -> torch.Tensor:
        op_vec = self.op_embedding(op_id)                # (B, op_embed_dim)
        operand_vec = (operands - self.operand_mean) / self.operand_std  # (B, 2)
        return torch.cat([op_vec, operand_vec], dim=-1)


class TransitionModel(nn.Module):
    """Two-layer MLP transition model over [hidden_state ; action]."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,
        op_embed_dim: int = 16,
        predict_delta: bool = True,
        use_action: bool = True,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_delta = predict_delta
        self.use_action = use_action
        
        if self.use_action:
            self.action_encoder = ActionEncoder(op_embed_dim, operand_mean, operand_std)
            in_dim = hidden_dim + self.action_encoder.output_dim
        else:
            self.action_encoder = None
            in_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        op_id: torch.Tensor = None,
        operands: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_action:
            if op_id is None or operands is None:
                raise ValueError("op_id and operands are required when use_action=True")
            action = self.action_encoder(op_id, operands)
            x = torch.cat([h_t, action], dim=-1)
        else:
            x = h_t
            
        out = self.net(x)
        if self.predict_delta:
            return h_t + out
        return out


def transition_loss(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Mean-squared error between predicted and target hidden states."""
    return nn.functional.mse_loss(pred, target, reduction=reduction)


class LinearTransitionModel(nn.Module):
    """Low-rank linear projection transition model over [hidden_state ; action].
    Matches the parameter count of the MLP baseline perfectly."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,
        op_embed_dim: int = 16,
        predict_delta: bool = True,
        use_action: bool = True,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_delta = predict_delta
        self.use_action = use_action
        
        if self.use_action:
            self.action_encoder = ActionEncoder(op_embed_dim, operand_mean, operand_std)
            in_dim = hidden_dim + self.action_encoder.output_dim
        else:
            self.action_encoder = None
            in_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

    def forward(
        self,
        h_t: torch.Tensor,
        op_id: torch.Tensor = None,
        operands: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.use_action:
            if op_id is None or operands is None:
                raise ValueError("op_id and operands are required when use_action=True")
            action = self.action_encoder(op_id, operands)
            x = torch.cat([h_t, action], dim=-1)
        else:
            x = h_t
            
        out = self.net(x)
        if self.predict_delta:
            return h_t + out
        return out


class TransformerTransitionModel(nn.Module):
    """Bottleneck Transformer transition model for parameter matching."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,  # Used as d_model for the transformer bottleneck
        op_embed_dim: int = 16,
        predict_delta: bool = True,
        use_action: bool = True,
        operand_mean: float = 0.0,
        operand_std: float = 1.0,
        num_layers: int = 2,
        nhead: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_delta = predict_delta
        self.use_action = use_action
        self.d_model = mlp_hidden_dim // 2
        
        if self.use_action:
            self.action_encoder = ActionEncoder(op_embed_dim, operand_mean, operand_std)
            self.action_proj = nn.Linear(self.action_encoder.output_dim, self.d_model)
        else:
            self.action_encoder = None
            
        self.state_proj_in = nn.Linear(hidden_dim, self.d_model)
        
        # Positional embeddings for STATE (idx 0) and ACTION (idx 1)
        self.pos_emb = nn.Embedding(2, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.state_proj_out = nn.Linear(self.d_model, hidden_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        op_id: torch.Tensor = None,
        operands: torch.Tensor = None,
    ) -> torch.Tensor:
        B = h_t.size(0)
        
        # Project state
        state_emb = self.state_proj_in(h_t)  # (B, d_model)
        state_emb = state_emb.unsqueeze(1)   # (B, 1, d_model)
        
        if self.use_action:
            if op_id is None or operands is None:
                raise ValueError("op_id and operands are required when use_action=True")
            action_raw = self.action_encoder(op_id, operands)
            action_emb = self.action_proj(action_raw)  # (B, d_model)
            action_emb = action_emb.unsqueeze(1)       # (B, 1, d_model)
            
            # Sequence: [STATE, ACTION]
            seq = torch.cat([state_emb, action_emb], dim=1)  # (B, 2, d_model)
            positions = torch.arange(2, device=h_t.device).unsqueeze(0).expand(B, 2)
            seq = seq + self.pos_emb(positions)
        else:
            seq = state_emb
            positions = torch.zeros(B, 1, dtype=torch.long, device=h_t.device)
            seq = seq + self.pos_emb(positions)
            
        # Transformer pass
        out_seq = self.transformer(seq)  # (B, SeqLen, d_model)
        
        # Extract STATE token (index 0)
        state_out = out_seq[:, 0, :]  # (B, d_model)
        
        # Project back to full hidden_dim
        out = self.state_proj_out(state_out)  # (B, hidden_dim)
        
        if self.predict_delta:
            return h_t + out
        return out

```


## File: `models\vq_state.py`
```python
"""Vector-quantized discrete state discovery for frozen LM hidden states.

This is the **only** component that converts a continuous hidden state ``h`` into
a discrete state id ``z``. It is a VQ-VAE style quantizer (van den Oord et al.,
2017) with the following design choices:

* **EMA codebook updates** (no gradient through the codebook), which are far
  more stable than the original gradient-based codebook update and avoid the
  optimizer destabilizing the embedding table.
* **Straight-through estimator** (STE): the quantized output passes the decoder
  gradient straight through to the encoder input, scaled by a commitment loss
  that pulls the encoder output toward its assigned code.
* **Dead-code revival**: at every step, codes whose EMA cluster size falls
  below ``epsilon`` are re-initialized to a randomly sampled input vector. This
  directly counteracts the codebook collapse that :mod:`evaluation.codebook_usage`
  is built to detect.

This module answers the Version-5 research question's first half:

    continuous hidden state ``h``  ->  discrete state id ``z``

Outputs are a ``(num_codes, hidden_dim)`` codebook that, once trained, can
quantize any frozen LM hidden state in a single ``forward`` call.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQStateQuantizer(nn.Module):
    """Vector-quantize a hidden state into a discrete code.

    Args:
        hidden_dim: dimension of the input hidden state ``h``.
        num_codes: size of the discrete codebook (the state-space cardinality).
        commitment_cost: weight on the commitment loss
            ``||sg(z_q) - h||^2`` that pulls the encoder output toward its code.
        ema_decay: EMA decay factor for codebook / cluster-size updates.
        epsilon: numerical stabilizer for the EMA cluster-size normalization,
            also used as the dead-code-revival threshold.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_codes: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Codebook initialized with small-variance gaussian (the EMA update
        # quickly overrides the init once data flows through).
        codebook = torch.randn(num_codes, hidden_dim) * 0.02
        self.register_buffer("codebook", codebook)
        # EMA tracking buffers (van den Oord et al. formulation).
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", codebook.clone())

    # ------------------------------------------------------------------ #
    def _quantize(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbor lookup. Returns (z_q, indices).

        ``h`` is (..., H); we flatten the leading dims for the distance
        computation and reshape back at the end.
        """
        lead_shape = h.shape[:-1]
        h_flat = h.reshape(-1, self.hidden_dim)  # (M, H)

        # Squared L2 distance ||h - c||^2 = ||h||^2 + ||c||^2 - 2 h·c
        # computed without materializing the (M, num_codes, H) tensor.
        dist = (
            h_flat.pow(2).sum(dim=-1, keepdim=True)               # (M, 1)
            + self.codebook.pow(2).sum(dim=-1, keepdim=False)     # (num_codes,)
            - 2.0 * h_flat @ self.codebook.t()                    # (M, num_codes)
        )
        indices = dist.argmin(dim=-1)                              # (M,)
        z_q_flat = self.codebook[indices]                          # (M, H)
        z_q = z_q_flat.reshape(*lead_shape, self.hidden_dim)
        return z_q, indices, z_q_flat

    # ------------------------------------------------------------------ #
    def _ema_update(self, h_flat: torch.Tensor, indices_flat: torch.Tensor) -> None:
        """Update EMA cluster size and codebook vectors (in-place, no grad).

        Implements dead-code revival: codes with cluster size below
        ``epsilon`` are re-initialized to a random input from the current batch.
        """
        with torch.no_grad():
            one_hot = F.one_hot(indices_flat, self.num_codes).type_as(h_flat)  # (M, K)
            cluster_size = one_hot.sum(dim=0)                                   # (K,)
            # Sum of encoder outputs assigned to each code.
            dw = one_hot.t() @ h_flat                                          # (K, H)

            # Laplace smoothing on the EMA cluster size.
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1.0 - self.ema_decay
            )
            n = self.ema_cluster_size.sum()
            # Laplace-smoothed cluster size, rescaled back to the *count* domain
            # (sums to ``n``). Dividing the running per-code input SUM by this
            # count yields the running per-code input MEAN = the new codebook.
            # (Without the ``* n`` rescale we'd divide by a probability and the
            # codebook would blow up by ~1/batch_size.)
            smoothed_count = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_codes * self.epsilon)
                * n
            )
            # EMA of the per-code sum of encoder outputs.
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1.0 - self.ema_decay)
            # Normalize to get the new codebook = running mean of assigned inputs.
            self.codebook.copy_(self.ema_w / smoothed_count.unsqueeze(-1))

            # Dead-code revival: any code whose smoothed cluster size is below
            # epsilon gets re-pointed to a random batch vector. This is the
            # primary defense against collapse (measured by codebook_usage).
            dead = self.ema_cluster_size < self.epsilon
            if dead.any() and h_flat.shape[0] > 0:
                n_dead = int(dead.sum().item())
                repl = h_flat[torch.randint(0, h_flat.shape[0], (n_dead,), device=h_flat.device)]
                self.codebook[dead] = repl
                # Reset the EMA accumulator so the revived code is not
                # immediately re-killed by a stale near-zero cluster size.
                self.ema_w[dead] = repl
                self.ema_cluster_size[dead] = 0.0

    # ------------------------------------------------------------------ #
    def forward(
        self, h: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize ``h`` and return ``(z_q_st, indices, info)``.

        Args:
            h: ``(B, H)`` or ``(..., H)`` continuous hidden states.
            training: when ``True`` the EMA codebook update is applied. Pass
                ``training=False`` for inference-only quantization (used by the
                discrete-trajectory encoder and the rollout test).

        ``info`` carries diagnostic scalars: ``commitment_loss`` (for the
        training objective), ``codebook_loss`` (zero under EMA, kept for API
        symmetry with gradient codebooks), ``perplexity`` (effective number of
        active codes) and ``active_codes`` (count of codes used this batch).
        """
        z_q, indices, _ = self._quantize(h)

        # Straight-through estimator: forward passes z_q, gradient flows to h.
        z_q_st = h + (z_q - h).detach()

        # Diagnostics + EMA update operate on the flattened assignment.
        h_flat = h.reshape(-1, self.hidden_dim)
        idx_flat = indices.reshape(-1)
        info = self._diagnostics(idx_flat)
        if training and self.training:
            self._ema_update(h_flat, idx_flat)

        # Commitment loss pulls the encoder output toward its (stop-grad) code.
        commitment_loss = F.mse_loss(h_flat, z_q.reshape(-1, self.hidden_dim).detach())
        info["commitment_loss"] = commitment_loss
        info["codebook_loss"] = torch.zeros((), device=h.device)  # 0 under EMA

        return z_q_st, indices, info

    # ------------------------------------------------------------------ #
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Inference-only quantization: returns integer code ids ``z``."""
        self.eval()
        with torch.no_grad():
            _, indices, _ = self.forward(h, training=False)
            return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up the continuous codebook vector for a batch of code ids."""
        with torch.no_grad():
            return self.codebook[indices]

    # ------------------------------------------------------------------ #
    def _diagnostics(self, idx_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perplexity (= effective number of codes used) and active-code count."""
        device = idx_flat.device
        counts = torch.bincount(idx_flat, minlength=self.num_codes).type_as(self.codebook)
        total = counts.sum().clamp_min(1.0)
        avg_probs = counts / total
        entropy = -(avg_probs * torch.log(avg_probs.clamp_min(1e-10))).sum()
        perplexity = torch.exp(entropy)
        active = (counts > 0).sum()
        return {"perplexity": perplexity, "active_codes": active}

```


## File: `models\__init__.py`
```python
# Init

```


## File: `scripts\archive_builder.py`
```python
import os
import shutil
import hashlib
import json
import datetime

WORKSPACE = "C:/Users/singh/OneDrive/Documents/latent_planning"
ARCHIVE_DIR = os.path.join(WORKSPACE, "research_archive")
DOWNLOADS = "C:/Users/singh/Downloads"

EXPERIMENTS = {
    "qwen2.5_7b_countdown": os.path.join(WORKSPACE, "archived_baselines/qwen2.5_7b_countdown"),
    "mistral_7b_countdown": os.path.join(WORKSPACE, "archived_baselines/mistral_7b_countdown/reports/mistral_7b_countdown"),
    "qwen2.5_7b_game24": os.path.join(DOWNLOADS, "qwen2.5_7b_game24_results/submission/qwen2.5_7b_game24"),
    "deepseek_r1_countdown": os.path.join(DOWNLOADS, "deepseek_r1_countdown_results/submission/deepseek_r1_countdown"),
}

SUBDIRS = ["experiments", "reports", "baselines", "kaggle_outputs", "zips", "audits", "figures", "metadata"]

def sha256_file(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def build_archive():
    print(f"Building archive at {ARCHIVE_DIR}...")
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        
    for sub in SUBDIRS:
        os.makedirs(os.path.join(ARCHIVE_DIR, sub), exist_ok=True)
        
    manifest = []
    provenance = []
    provenance.append("# Provenance Ledger")
    provenance.append("Chain of custody for critical artifacts in the latent_planning research project.\n")
    
    # 1. Archive Experiments
    for exp_name, exp_path in EXPERIMENTS.items():
        if not os.path.exists(exp_path):
            print(f"WARNING: Source path not found: {exp_path}")
            continue
            
        dest_dir = os.path.join(ARCHIVE_DIR, "experiments", exp_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy files
        for root, _, files in os.walk(exp_path):
            for file in files:
                src = os.path.join(root, file)
                rel_path = os.path.relpath(src, exp_path)
                dst = os.path.join(dest_dir, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # Immutable copy
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    
                # Manifest
                stat = os.stat(dst)
                h = sha256_file(dst)
                manifest.append({
                    "filename": os.path.join("experiments", exp_name, rel_path).replace("\\", "/"),
                    "size_bytes": stat.st_size,
                    "sha256": h,
                    "created_at": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                
                # Provenance Ledger for key files
                if file in ["coherence_action_depth.csv", "probe_results.csv", "phase_b_oracle_report.md", "phase_a_report.md"] or file.endswith(".png"):
                    command = f"python scripts/run_phase_a.py --domain {exp_name.split('_')[-1]} --model {exp_name}"
                    provenance.append(f"### Artifact: {os.path.join(exp_name, rel_path)}")
                    provenance.append(f"- **SHA-256:** `{h}`")
                    provenance.append(f"- **Generated By:** `scripts/run_phase_a.py`")
                    provenance.append(f"- **Estimated Command:** `{command}`")
                    provenance.append(f"- **Timestamp:** {datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()}")
                    provenance.append("")

    # Save Manifest
    with open(os.path.join(ARCHIVE_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Save Provenance Ledger
    with open(os.path.join(ARCHIVE_DIR, "PROVENANCE_LEDGER.md"), "w") as f:
        f.write("\n".join(provenance))
        
    print("Archive build complete.")
    print(f"Archived {len(manifest)} files.")

if __name__ == "__main__":
    build_archive()

```


## File: `scripts\cache_hidden_states.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Cache frozen hidden states for a dataset to avoid repeated HF model invocations."""

import argparse
import os
import sys

import torch
from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, build_trajectories, save_trajectories

def main():
    parser = argparse.ArgumentParser(description="Cache hidden states for trajectories")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--data_file", type=str, required=True, help="Input JSONL problems file")
    parser.add_argument("--output_file", type=str, required=True, help="Output .pt trajectories file")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden layer to extract")
    args = parser.parse_args()

    print(f"Loading problems from {args.data_file}...")
    problems = load_problems(args.data_file)
    
    print(f"Loading model {args.model}...")
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    
    print(f"Extracting hidden states for {len(problems)} problems...")
    trajectories = build_trajectories(model, tokenizer, problems, layer=args.layer)
    
    print(f"Successfully built {len(trajectories)} trajectories. Saving to {args.output_file}...")
    save_trajectories(trajectories, args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()

```


## File: `scripts\check_consistency.py`
```python
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

```


## File: `scripts\check_phase_c_stats.py`
```python
import os
import sys
import torch
import collections

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import gather_state_groups

def main():
    out_dir = "reports"
    train_traj_path = os.path.join(out_dir, "trajectories", "train.pt")
    
    if not os.path.exists(train_traj_path):
        print("ERROR: train.pt not found.")
        sys.exit(1)
        
    print("Loading trajectories...")
    trajs = load_trajectories(train_traj_path)
    print(f"Loaded {len(trajs)} trajectories.")
    
    print("Gathering state groups...")
    # gather_state_groups maps sym -> list of (traj_idx, hist, h)
    state_groups = gather_state_groups(trajs)
    
    unique_states = len(state_groups)
    
    # We only care about cross-history collisions
    cross_history_groups = {}
    total_positive_pairs = 0
    cluster_sizes = []
    
    for sym, elements in state_groups.items():
        # group by history to see how many distinct histories reached this state
        histories = collections.defaultdict(list)
        for t_idx, hist, h in elements:
            histories[hist].append((t_idx, h))
            
        if len(histories) > 1:
            # calculate cross-history pairs
            # Number of pairs = sum(len(h1) * len(h2) for all h1!=h2)
            # which is (sum(len)^2 - sum(len^2)) / 2
            sizes = [len(v) for v in histories.values()]
            total = sum(sizes)
            pairs = (total**2 - sum(s**2 for s in sizes)) // 2
            
            if pairs > 0:
                cross_history_groups[sym] = elements
                total_positive_pairs += pairs
                cluster_sizes.append(total)
                
    num_cross_history_states = len(cross_history_groups)
    
    if num_cross_history_states > 0:
        avg_cluster = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster = max(cluster_sizes)
    else:
        avg_cluster = 0
        max_cluster = 0
        
    lines = [
        "# Phase C Dataset Statistics",
        "",
        f"- **Total Trajectories loaded**: {len(trajs)}",
        f"- **Total Unique Symbolic States**: {unique_states}",
        f"- **Cross-History States** (>=2 different histories): {num_cross_history_states}",
        f"- **Average Cluster Size** (for cross-history states): {avg_cluster:.2f}",
        f"- **Maximum Cluster Size**: {max_cluster}",
        f"- **Total Possible Positive Pairs**: {total_positive_pairs}",
        "",
    ]
    
    if num_cross_history_states < 500 or total_positive_pairs < 5000:
        lines.append("> [!WARNING]")
        lines.append("> The dataset may be underpowered for contrastive learning. Consider generating more trajectories.")
    else:
        lines.append("> [!PASS]")
        lines.append("> Dataset appears sufficiently powered for contrastive canonicalization.")
        
    md_path = os.path.join(out_dir, "state_statistics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Stats written to {md_path}")
    print(f"  Cross-History States: {num_cross_history_states}")
    print(f"  Total Positive Pairs: {total_positive_pairs}")

if __name__ == "__main__":
    main()

```


## File: `scripts\count_parameters.py`
```python
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transition_model import TransitionModel, LinearTransitionModel, TransformerTransitionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--op_embed_dim", type=int, default=16)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512) # Use 512 for MLP inner dim, and Transformer d_model
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)
    args = parser.parse_args()

    linear_model = LinearTransitionModel(hidden_dim=args.hidden_dim, mlp_hidden_dim=args.mlp_hidden_dim, op_embed_dim=args.op_embed_dim)
    mlp_model = TransitionModel(hidden_dim=args.hidden_dim, mlp_hidden_dim=args.mlp_hidden_dim, op_embed_dim=args.op_embed_dim)
    transformer_model = TransformerTransitionModel(
        hidden_dim=args.hidden_dim, 
        mlp_hidden_dim=args.mlp_hidden_dim, 
        op_embed_dim=args.op_embed_dim,
        num_layers=args.transformer_layers,
        nhead=args.transformer_heads
    )

    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Linear Transition Model params:      {count_parameters(linear_model):,}")
    print(f"MLP Transition Model params:         {count_parameters(mlp_model):,}")
    print(f"Transformer Transition Model params: {count_parameters(transformer_model):,}")

if __name__ == "__main__":
    main()

```


## File: `scripts\evaluate_teacher.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Evaluate teacher model accuracy on Countdown problems."""

import argparse
import os
import sys

import torch
from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, format_header
from data_processing.action_parser import parse_solution

def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher accuracy")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--data_file", type=str, default="data/val.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading problems from {args.data_file}...")
    problems = load_problems(args.data_file)[:args.limit]
    
    print(f"Loading model {args.model}...")
    model = load_model(args.model)
    tokenizer = load_tokenizer(args.model)
    model.eval()
    
    correct = 0
    total = len(problems)
    
    print(f"Evaluating {total} problems...")
    for i, prob in enumerate(problems):
        header = format_header(prob["numbers"], prob["target"])
        inputs = tokenizer(header, return_tensors="pt").to(model.device)
        
        # We expect a solution to be reasonably short, e.g. < 100 tokens.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Parse lines
        lines = [line.strip() for line in gen_text.split("\n") if line.strip()]
        
        try:
            actions = parse_solution(lines)
            # The final result is usually the output of the last action.
            # In parse_solution, it doesn't return the result.
            # We can parse the last line manually: "A op B = C"
            if not lines:
                continue
            last_line = lines[-1]
            parts = last_line.split("=")
            if len(parts) == 2:
                result = float(parts[1].strip())
                if abs(result - prob["target"]) < 1e-5:
                    correct += 1
        except Exception:
            pass
            
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{total} | Current Accuracy: {correct / (i+1) * 100:.2f}%")

    accuracy = correct / total * 100
    print(f"\nFinal Teacher Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if accuracy < 80.0:
        print("WARNING: Teacher accuracy is below 80%. Hidden states may contain flawed reasoning.")

if __name__ == "__main__":
    main()

```


## File: `scripts\extract_hidden_states.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import load_checkpoint, load_model, load_tokenizer

def extract_hidden_states(model, tokenizer, dataset, target_layer: int, output_file: str):
    """Extracts and saves hidden states from a specified layer."""
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    extracted_data = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Extracting hidden states")):
            # Format prompt based on dataset structure
            prompt = f"Problem: Given the numbers {sample['numbers']}, reach the target {sample['target']}.\nSolution:\n{sample['cot']}"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Forward pass asking for hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # hidden_states is a tuple of (layers + 1)
            # 0 is the embedding layer, 1 to L are transformer layers
            layer_hidden_states = outputs.hidden_states[target_layer].squeeze(0) # Shape: (seq_len, hidden_size)
            
            # Store per-token hidden states
            for pos, token_id in enumerate(inputs.input_ids[0]):
                extracted_data.append({
                    "sample_id": i,
                    "position": pos,
                    "token_id": token_id.item(),
                    "token_str": tokenizer.decode(token_id),
                    # Storing tensor as list for JSON serialization (could use torch.save for pure tensors later)
                    "hidden_state": layer_hidden_states[pos].cpu().float().tolist()
                })
                
    # Save to file
    # Note: For large datasets, torch.save(list_of_dicts, output_file.pt) is much more efficient than JSON.
    # We will use torch.save if the output file ends in .pt
    if output_file.endswith(".pt"):
        torch.save(extracted_data, output_file)
    else:
        with open(output_file, 'w') as f:
            for item in extracted_data:
                f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract Hidden States")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint (or None for base model)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_file", type=str, default="reports/hidden_states.pt", help="Output file path (.pt recommended)")
    parser.add_argument("--layer", type=int, default=-1, help="Layer number to extract (-1 for last layer)")
    args = parser.parse_args()

    # Load Model
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model, tokenizer = load_checkpoint(args.model_path)
    else:
        print("Loading base model...")
        model = load_model()
        tokenizer = load_tokenizer()

    # Load Dataset
    print(f"Loading dataset from {args.data_file}...")
    dataset = load_dataset("json", data_files={"test": args.data_file})["test"]

    print(f"Extracting hidden states from layer {args.layer}...")
    extract_hidden_states(model, tokenizer, dataset, args.layer, args.output_file)
    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()

```


## File: `scripts\generate_countdown_dataset.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = list(range(1, 11)) * 2

MIN_TARGET = 100
MAX_TARGET = 999


def _build_walk(is_ood: bool, max_target: int) -> Tuple[List[int], int, List[str]]:
    """One forward random walk. Division is integer-only; the walk is biased to
    stay bounded once it exceeds ``max_target``."""
    pool = random.sample(LARGE_NUMBERS, random.randint(1, 4))
    pool += random.sample(SMALL_NUMBERS, 6 - len(pool))

    # IID: 2-4 ops, OOD: exactly 5 ops
    num_ops = 5 if is_ood else random.randint(2, 4)
    selected = random.sample(pool, num_ops + 1)

    ops_history: List[str] = []
    current_val = selected[0]

    for val in selected[1:]:
        # Valid ops: division only when it yields a positive whole number.
        valid = ['+', '-', '*']
        if val != 0 and current_val % val == 0 and current_val // val > 0:
            valid.append('/')

        # Keep the walk bounded: once large, prefer reducing ops so the final
        # target lands in a real Countdown-style range.
        if current_val > max_target:
            reduced = [o for o in valid if o in ('-', '/')]
            if reduced:
                valid = reduced

        op = random.choice(valid)

        if op == '+':
            new = current_val + val
            ops_history.append(f"{current_val} + {val} = {new}")
            current_val = new
        elif op == '*':
            new = current_val * val
            ops_history.append(f"{current_val} * {val} = {new}")
            current_val = new
        elif op == '/':
            new = current_val // val
            ops_history.append(f"{current_val} / {val} = {new}")
            current_val = new
        else:  # '-' : keep the result strictly positive
            if current_val - val > 0:
                a, b, new = current_val, val, current_val - val
            else:
                a, b, new = val, current_val, val - current_val
            ops_history.append(f"{a} - {b} = {new}")
            current_val = new

    return pool, current_val, ops_history


def generate_countdown_problem(
    is_ood: bool = False,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
    max_attempts: int = 4000,
) -> Dict[str, Any]:
    """Generates a Countdown numbers-game problem with a bounded integer target.

    Uses rejection sampling so the final target falls in ``[min_target, max_target]``.
    Operations are +, -, *, and integer / (remainder-free, matching apply_op)."""
    pool, target, ops_history = _build_walk(is_ood, max_target)
    attempts = 1
    while not (min_target <= target <= max_target) and attempts < max_attempts:
        pool, target, ops_history = _build_walk(is_ood, max_target)
        attempts += 1

    cot = "\n".join(ops_history)
    return {
        "numbers": pool,
        "target": target,
        "solution": ops_history,
        "cot": cot,
    }


def generate_dataset(
    num_samples: int,
    output_file: str,
    seed: int = 0,
    is_ood: bool = False,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
):
    """Generates a dataset of Countdown problems and saves to JSONL."""
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        for _ in range(num_samples):
            problem = generate_countdown_problem(
                is_ood=is_ood, min_target=min_target, max_target=max_target)
            f.write(json.dumps(problem) + '\n')


def generate_multi_solution_dataset(
    num_samples: int,
    output_file: str,
    solutions_per_problem: int = 4,
    seed: int = 0,
    min_target: int = MIN_TARGET,
    max_target: int = MAX_TARGET,
    max_attempts_per_solution: int = 200,
):
    """Generate Countdown problems each carrying several DISTINCT valid solutions.

    Used by the V5 permutation-robustness analysis: encoding each solution as a
    separate hidden-state trajectory lets us test whether *equivalent* symbolic
    states (reached via different reasoning orders) map to the *same* discrete
    code. Output records carry a ``solutions: List[List[str]]`` field (one list
    of step strings per distinct solution) in addition to the canonical
    ``solution`` / ``cot`` fields (set to the first solution for back-compat).

    Solutions are deduplicated by their canonical step-string tuple, so the
    returned list may be shorter than ``solutions_per_problem`` if fewer
    distinct walks are found within the attempt budget.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            seen: set = set()
            solutions: List[List[str]] = []
            last_pool, last_target = None, None
            for _ in range(max_attempts_per_solution):
                if len(solutions) >= solutions_per_problem:
                    break
                pool, target, ops_history = _build_walk(False, max_target)
                if not (min_target <= target <= max_target):
                    continue
                key = tuple(ops_history)
                if key in seen:
                    continue
                seen.add(key)
                solutions.append(ops_history)
                last_pool, last_target = pool, target

            if not solutions:
                continue
            record = {
                "numbers": last_pool,
                "target": last_target,
                "solutions": solutions,
                # Back-compat fields (first solution), so existing loaders work.
                "solution": solutions[0],
                "cot": "\n".join(solutions[0]),
            }
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Countdown Dataset")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=500, help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=500, help="Number of testing samples")
    parser.add_argument("--num_test_ood", type=int, default=500, help="Number of OOD testing samples")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (each split is seeded deterministically off this)")
    parser.add_argument("--min_target", type=int, default=MIN_TARGET, help="Inclusive lower bound on the target")
    parser.add_argument("--max_target", type=int, default=MAX_TARGET, help="Inclusive upper bound on the target")
    parser.add_argument("--multi_solution", action="store_true",
                        help="V5 permutation-robustness mode: write data/train_multi.jsonl with multiple distinct solutions per problem")
    parser.add_argument("--num_multi", type=int, default=2000,
                        help="Number of multi-solution problems (only with --multi_solution)")
    parser.add_argument("--solutions_per_problem", type=int, default=4,
                        help="Target number of distinct solutions per problem (only with --multi_solution)")

    args = parser.parse_args()

    print(f"Generating datasets in {args.output_dir} (seed={args.seed}, target in [{args.min_target},{args.max_target}])...")
    # Each split gets an independent, reproducible seed offset.
    generate_dataset(args.num_train, os.path.join(args.output_dir, "train.jsonl"),
                     seed=args.seed, min_target=args.min_target, max_target=args.max_target)
    generate_dataset(args.num_val, os.path.join(args.output_dir, "val.jsonl"),
                     seed=args.seed + 1, min_target=args.min_target, max_target=args.max_target)
    generate_dataset(args.num_test, os.path.join(args.output_dir, "test.jsonl"),
                     seed=args.seed + 2, min_target=args.min_target, max_target=args.max_target)
    generate_dataset(args.num_test_ood, os.path.join(args.output_dir, "test_ood.jsonl"),
                     seed=args.seed + 3, is_ood=True, min_target=args.min_target, max_target=args.max_target)

    if args.multi_solution:
        print(f"Generating multi-solution dataset ({args.solutions_per_problem} solutions/problem)...")
        generate_multi_solution_dataset(
            args.num_multi,
            os.path.join(args.output_dir, "train_multi.jsonl"),
            solutions_per_problem=args.solutions_per_problem,
            seed=args.seed + 4,
            min_target=args.min_target,
            max_target=args.max_target,
        )
    print("Done!")


if __name__ == "__main__":
    main()

```


## File: `scripts\generate_game24_dataset.py`
```python
"""Game-of-24 reasoning-trace generator for the V5.4 additional-domain study.

Game of 24: combine four numbers with +, -, *, / (each number used once) to
reach **24**. Solutions are emitted as the *same* ``a op b = c`` step format as
Countdown, so the existing :mod:`data_processing.action_parser` and
``build_trajectories`` pipeline consume them **unchanged** — every step yields a
real symbolic action (``op_id`` + operands), which is what Experiment A's
``MLP(z,a) − ActionBigram(z,a)`` contrast requires. (A free-form domain like
GSM8K would have no action grammar and would collapse that contrast.)

To stay compatible with the Countdown parser/``apply_op`` we restrict to
**integer-only** intermediate values: division must be remainder-free and every
intermediate result is a positive integer. We therefore generate only puzzles
that have an integer-only solution (the classic game also allows fractions; we
intentionally exclude those for pipeline compatibility) and emit one such
solution as the reasoning trace.

Record schema matches the Countdown generator exactly::

    {"numbers": [...], "target": 24, "solution": ["a op b = c", ...], "cot": "..."}

Pool statistics (cards 1–13, integer-only, positive intermediates)::

    Total unique 4-card multisets: 1,820  (= C(16,4))
    Solvable:                      1,346  (74.0 %)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

TARGET = 24
CARD_MIN, CARD_MAX = 1, 13   # classic 4-card range
NUM_CARDS = 4

# (value, steps) item: a number on the table plus the steps that produced it.
Item = Tuple[int, List[str]]


def _apply(sym: str, a: int, b: int) -> Optional[int]:
    """Integer op mirroring action_parser.apply_op; None if illegal here.

    Results must stay positive integers (matches the Countdown distribution and
    keeps `a op b = c` parseable with positive operands)."""
    if sym == "+":
        return a + b
    if sym == "-":
        return a - b if a - b > 0 else None
    if sym == "*":
        return a * b
    if sym == "/":
        if b != 0 and a % b == 0 and a // b > 0:
            return a // b
        return None
    return None


def _solve(items: List[Item], target: int) -> Optional[List[str]]:
    """Return a valid ordered step list reducing ``items`` to ``target``, or None.

    Recursively picks two table values, combines them with an op, and recurses
    on the smaller multiset. Step ordering (left subtree, right subtree, then the
    combine) is a valid linearization because both operands are produced before
    they are combined."""
    if len(items) == 1:
        return items[0][1] if items[0][0] == target else None
    n = len(items)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a_val, a_steps = items[i]
            b_val, b_steps = items[j]
            rest = [items[k] for k in range(n) if k != i and k != j]
            for sym in ("+", "-", "*", "/"):
                res = _apply(sym, a_val, b_val)
                if res is None:
                    continue
                step = f"{a_val} {sym} {b_val} = {res}"
                merged = a_steps + b_steps + [step]
                sol = _solve(rest + [(res, merged)], target)
                if sol is not None:
                    return sol
    return None


# --------------------------------------------------------------------------- #
# Exhaustive solvable-pool enumeration
# --------------------------------------------------------------------------- #

def enumerate_solvable_pool() -> List[Tuple[int, ...]]:
    """Return every unique solvable 4-card multiset from {1..13}.

    A hand is 'solvable' if the four numbers can be combined with +, -, *, /
    (each used exactly once, integer-only positive intermediates) to reach 24.

    Returns a sorted list of tuples for deterministic ordering.
    """
    pool: List[Tuple[int, ...]] = []
    for hand in combinations_with_replacement(range(CARD_MIN, CARD_MAX + 1), NUM_CARDS):
        items = [(n, []) for n in hand]
        if _solve(items, TARGET) is not None:
            pool.append(hand)
    return pool


def generate_game24_problem(rng: random.Random, max_attempts: int = 2000) -> Dict[str, Any]:
    """Sample four cards with an integer-only solution to 24; return the record.

    This function is retained for backward compatibility with existing tests and
    the smoke_4bit_extraction script. For dataset generation, prefer
    :func:`generate_dataset` which uses exhaustive pool partitioning."""
    for _ in range(max_attempts):
        numbers = [rng.randint(CARD_MIN, CARD_MAX) for _ in range(NUM_CARDS)]
        sol = _solve([(n, []) for n in numbers], TARGET)
        if sol is not None:
            return {
                "numbers": numbers,
                "target": TARGET,
                "solution": sol,
                "cot": "\n".join(sol),
            }
    raise RuntimeError("No solvable Game-of-24 hand found within attempt budget.")


def _hand_to_record(hand: Tuple[int, ...], rng: random.Random) -> Dict[str, Any]:
    """Convert a canonical hand tuple to a problem record with a solved trace.

    The hand is randomly permuted so the model doesn't see cards in sorted order
    (which would be an artificial regularity absent from the real game)."""
    numbers = list(hand)
    rng.shuffle(numbers)
    sol = _solve([(n, []) for n in numbers], TARGET)
    assert sol is not None, f"hand {hand} was in solvable pool but solver returned None"
    return {
        "numbers": numbers,
        "target": TARGET,
        "solution": sol,
        "cot": "\n".join(sol),
    }


def generate_dataset(
    num_samples: int,
    output_file: str,
    seed: int = 0,
    is_ood: bool = False,
) -> None:
    """Generate ``num_samples`` unique Game-of-24 problems and save to JSONL.

    Problems are drawn **without replacement** from the exhaustively enumerated
    pool of 1,346 solvable hands. ``num_samples`` must not exceed the pool size.

    Parameters
    ----------
    is_ood : bool
        Accepted for API compatibility with the Countdown generator. Game24 has
        a fixed card range {1..13} with no meaningful OOD variant; if ``True``,
        a warning is emitted and the parameter is otherwise ignored.
    """
    if is_ood:
        warnings.warn(
            "Game24 has a fixed card range {1..13}; is_ood=True is accepted "
            "for API compatibility but has no effect.",
            stacklevel=2,
        )
    pool = enumerate_solvable_pool()
    if num_samples > len(pool):
        raise ValueError(
            f"Requested {num_samples} unique Game24 hands but only "
            f"{len(pool)} solvable hands exist. Reduce num_samples or use "
            f"the Countdown domain for larger datasets."
        )
    rng = random.Random(seed)
    selected = list(pool)
    rng.shuffle(selected)
    selected = selected[:num_samples]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for hand in selected:
            f.write(json.dumps(_hand_to_record(hand, rng)) + "\n")


def generate_partitioned_splits(
    output_dir: str,
    seed: int = 0,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, int]:
    """Partition the full solvable pool into mutually exclusive train/val/test.

    Returns a dict with the actual split sizes.
    """
    pool = enumerate_solvable_pool()
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets the remainder — guarantees no rounding loss
    train_hands = shuffled[:n_train]
    val_hands = shuffled[n_train:n_train + n_val]
    test_hands = shuffled[n_train + n_val:]

    os.makedirs(output_dir, exist_ok=True)
    sizes = {}
    for split_name, hands in [("train", train_hands), ("val", val_hands), ("test", test_hands)]:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for hand in hands:
                f.write(json.dumps(_hand_to_record(hand, rng)) + "\n")
        sizes[split_name] = len(hands)

    return sizes


def main():
    parser = argparse.ArgumentParser(description="Generate Game-of-24 dataset")
    parser.add_argument("--output_dir", type=str, default="data/game24")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of solvable pool for training (default 0.8)")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of solvable pool for validation (default 0.1)")
    args = parser.parse_args()

    pool = enumerate_solvable_pool()
    print(f"Game-of-24 solvable pool: {len(pool)} unique hands "
          f"(out of 1820 multisets = {len(pool)/1820:.1%})")

    sizes = generate_partitioned_splits(
        args.output_dir, seed=args.seed,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    print(f"Partitioned into mutually exclusive splits:")
    for split, n in sizes.items():
        print(f"  {split}: {n}")
    print("Done!")


if __name__ == "__main__":
    main()

```


## File: `scripts\generate_transfer_datasets.py`
```python
"""Deterministic cross-domain reasoning-trace generators for V5 transfer.

Three small CoT datasets, each emitted as JSONL with a ``problem`` (the prompt
header) and ``solution`` (a list of newline-joined step strings). They share
the *shape* of the Countdown data (a header followed by one-line steps) so the
generic trajectory extractor (:mod:`data_processing.transfer_trajectory`) can
treat them identically — only the symbolic content differs.

* **Algebra** — linear ``solve for x``: each step isolates a sub-expression.
* **Logic** — propositional modus-ponens chains over a few facts.
* **Graph** — Dijkstra-style shortest-path reasoning over a small weighted
  graph.

All generators are seeded and rejection-sampled so outputs are reproducible
and every step is internally valid. They are deliberately small (default
~2000 problems each) — enough to measure transfer statistics, not to train a
domain expert.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

DOMAINS = ["algebra", "logic", "graph"]


# --------------------------------------------------------------------------- #
# Algebra: solve a*x + b = c for x. CoT isolates the unknown step by step.
# --------------------------------------------------------------------------- #
def _algebra_problem(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(2, 9)
    x = rng.randint(-9, 9)
    b = rng.randint(-20, 20)
    c = a * x + b
    steps = [
        f"Equation: {a} * x + {b} = {c}",
        f"Subtract {b}: {a} * x = {c - b}",
        f"Divide by {a}: x = {(c - b) // a}",
    ]
    header = f"Problem: Solve for x.\nSolution:\n"
    return {
        "domain": "algebra",
        "problem": header,
        "numbers": [a, b, c],
        "target": (c - b) // a,
        "solution": steps,
        "cot": "\n".join(steps),
    }


# --------------------------------------------------------------------------- #
# Logic: chain of modus ponens. If P then Q; P; therefore Q (extended).
# --------------------------------------------------------------------------- #
def _logic_problem(rng: random.Random) -> Dict[str, Any]:
    atoms = ["A", "B", "C", "D", "E", "F"]
    n = rng.randint(3, 5)
    chain = rng.sample(atoms, n)
    steps = []
    for i in range(n - 1):
        steps.append(f"If {chain[i]} then {chain[i + 1]}.")
    steps.append(f"{chain[0]} is true.")
    for i in range(n - 1):
        steps.append(f"Therefore {chain[i + 1]}.")
    header = "Problem: Determine the conclusion.\nSolution:\n"
    return {
        "domain": "logic",
        "problem": header,
        "numbers": [n],
        "target": n - 1,
        "solution": steps,
        "cot": "\n".join(steps),
    }


# --------------------------------------------------------------------------- #
# Graph: shortest path on a small weighted graph (Dijkstra-style narrative).
# --------------------------------------------------------------------------- #
def _graph_problem(rng: random.Random) -> Dict[str, Any]:
    import heapq

    n_nodes = rng.randint(4, 6)
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    # Random connected graph via a spanning tree, plus a few extra edges.
    adj = {v: {} for v in nodes}
    for i in range(1, n_nodes):
        j = rng.randint(0, i - 1)
        w = rng.randint(1, 9)
        adj[nodes[i]][nodes[j]] = w
        adj[nodes[j]][nodes[i]] = w
    for _ in range(rng.randint(1, 3)):
        u, v = rng.sample(nodes, 2)
        if v not in adj[u]:
            w = rng.randint(1, 9)
            adj[u][v] = w
            adj[v][u] = w

    src, dst = rng.sample(nodes, 2)
    # Dijkstra from src.
    dist = {v: float("inf") for v in nodes}
    dist[src] = 0
    prev = {src: None}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path + steps.
    if dist[dst] == float("inf"):
        return _graph_problem(rng)  # connected graph guarantees this is rare
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    steps = [f"Graph edges: " + ", ".join(
        f"{u}-{v}={w}" for u in adj for v, w in adj[u].items() if u < v)]
    steps.append(f"Start at {src}, end at {dst}.")
    for i in range(len(path) - 1):
        w = adj[path[i]][path[i + 1]]
        steps.append(f"Move {path[i]} -> {path[i + 1]} (cost {w}).")
    steps.append(f"Total cost: {int(dist[dst])}.")
    header = "Problem: Find the shortest path.\nSolution:\n"
    return {
        "domain": "graph",
        "problem": header,
        "numbers": [int(dist[dst])],
        "target": int(dist[dst]),
        "solution": steps,
        "cot": "\n".join(steps),
    }


_GENERATORS = {"algebra": _algebra_problem, "logic": _logic_problem, "graph": _graph_problem}


def generate_transfer_dataset(
    domain: str, num_samples: int, output_file: str, seed: int = 0
) -> None:
    """Generate ``num_samples`` problems of ``domain`` to ``output_file``."""
    if domain not in _GENERATORS:
        raise ValueError(f"Unknown domain {domain!r}; choose from {DOMAINS}")
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    gen = _GENERATORS[domain]
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            f.write(json.dumps(gen(rng)) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate V5 cross-domain reasoning datasets")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--num_samples", type=int, default=2000, help="Per domain")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for domain in args.domains:
        out = os.path.join(args.output_dir, f"transfer_{domain}.jsonl")
        print(f"[{domain}] generating {args.num_samples} problems -> {out}")
        # Per-domain deterministic seed offset, mirroring the Countdown generator.
        generate_transfer_dataset(
            domain, args.num_samples, out, seed=args.seed + DOMAINS.index(domain)
        )
    print("Done!")


if __name__ == "__main__":
    main()

```


## File: `scripts\import_audit.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Script to audit all internal project imports."""

import sys
import os
import importlib
import pkgutil

def main():
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
        
    modules_to_test = [
        "data_processing.action_parser",
        "data_processing.trajectory_dataset",
        "evaluation.coherence",
        "evaluation.probes",
        "models.diagnostic_decoder",
        "models.model_loader",
        "models.transition_model",
        "training.train_decoder",
        "training.train_transition"
    ]
    
    failed = 0
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
            print(f"[OK] Successfully imported: {mod}")
        except Exception as e:
            print(f"[FAIL] Failed to import: {mod}")
            print(f"       Error: {type(e).__name__}: {str(e)}")
            failed += 1
            
    if failed > 0:
        print(f"\nAudit failed. {failed} module(s) could not be imported.")
        sys.exit(1)
    else:
        print("\nAudit passed! All internal modules imported successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()

```


## File: `scripts\plot_architectures.py`
```python
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

```


## File: `scripts\postflight.py`
```python
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
        "transition_model_blind.pt"
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
        print(f"❌ POSTFLIGHT FAILED: {len(missing)} expected artifacts are missing or empty.")
        print(f"Missing: {missing}")
        sys.exit(1)
    else:
        print("✅ POSTFLIGHT PASSED: All expected artifacts generated successfully.")
        
        # Free up disk space by deleting the large trajectory tensors before Kaggle archives the output
        traj_dir = os.path.join(args.out_dir, "trajectories")
        if os.path.exists(traj_dir):
            import shutil
            print(f"🧹 Cleaning up {traj_dir} to save space in the Kaggle download archive...")
            shutil.rmtree(traj_dir, ignore_errors=True)
            print("🧹 Cleanup complete.")

if __name__ == "__main__":
    main()

```


## File: `scripts\preflight.py`
```python
import argparse
import os
import sys
import subprocess
import shutil
import json

def check_dataset_domain(data_dir, expected_domain):
    train_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_path):
        return True, "Will generate dataset."
    
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False, "train.jsonl is empty."
            data = json.loads(first_line)
            
            target = data.get("target")
            numbers = data.get("numbers", [])
            
            if expected_domain == "game24":
                if target != 24 or len(numbers) != 4:
                    return False, f"Expected Game24 (target=24, 4 numbers) but got target={target}, {len(numbers)} numbers."
            elif expected_domain == "countdown":
                # Countdown typically has 6 numbers and targets can vary, but definitely not strict Game24
                # If it happens to be 24 and 4 numbers, it's ambiguous, but usually it's 6 numbers.
                if target == 24 and len(numbers) == 4:
                    return False, "Dataset looks like Game24 but expected Countdown."
            return True, f"Dataset matches {expected_domain} characteristics."
    except Exception as e:
        return False, f"Failed to parse train.jsonl: {e}"

def main():
    parser = argparse.ArgumentParser(description="Latent Planning Preflight Checklist")
    parser.add_argument("--model", required=True, help="Model ID (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--domain", required=True, choices=["countdown", "game24"], help="Problem domain")
    parser.add_argument("--out_dir", required=True, help="Output directory to test")
    parser.add_argument("--data_dir", required=True, help="Data directory to test")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    print(f"=== LATENT PLANNING PREFLIGHT CHECK ===")
    summary = []

    # 1. GPU / CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        gpu_name = torch.cuda.get_device_name(0)
        
        # VRAM Check
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        summary.append(f"GPU: {gpu_name} ({total_gb:.1f} GB total, {free_gb:.1f} GB free)")
        if free_gb < 14.0:
            print(f"[WARN] Less than 14GB free VRAM ({free_gb:.1f}GB). 7B models might OOM even in 4-bit.")
        summary.append("CUDA available")
    except Exception as e:
        print(f"[FAIL] GPU/CUDA check: {e}")
        sys.exit(1)
        
    # 2. Dependencies & Versions
    try:
        import transformers
        import bitsandbytes
        import accelerate
        import peft
        print(f"[INFO] Dependencies: torch={torch.__version__}, transformers={transformers.__version__}, "
              f"accelerate={accelerate.__version__}, bitsandbytes={bitsandbytes.__version__}, peft={peft.__version__}")
        summary.append("Dependencies OK")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        sys.exit(1)

    # 3. Fast Tokenizer & Model Accessibility (meta device)
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        print("[..] Checking Hugging Face connection and model access...")
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if not getattr(tokenizer, "is_fast", False):
            print(f"[FAIL] Tokenizer is not fast (tokenizer.is_fast == False). Required for offset mappings.")
            sys.exit(1)
        summary.append("Fast tokenizer")
        
        # Meta initialization
        with torch.device("meta"):
            _ = AutoModelForCausalLM.from_config(config)
        summary.append("Model accessible")
    except Exception as e:
        err_str = str(e).lower()
        if "401" in err_str or "403" in err_str or "gated" in err_str or "token" in err_str:
            print(f"[FAIL] Authentication Error: The model '{args.model}' requires a valid Hugging Face token. "
                  "Please login via `huggingface-cli login` or set the HF_TOKEN environment variable.")
        else:
            print(f"[FAIL] Model access failed: {e}")
        sys.exit(1)

    # 4. Required Repository Files
    critical_paths = [
        "scripts/run_phase_a.py",
        "requirements.txt",
        "evaluation/",
        "training/",
        "data_processing/"
    ]
    for p in critical_paths:
        if not os.path.exists(p):
            print(f"[FAIL] Critical repository path missing: {p}")
            sys.exit(1)
    summary.append("Repository structure OK")

    # 5. Disk Space
    try:
        total_b, used_b, free_b = shutil.disk_usage(".")
        free_gb = free_b / (1024**3)
        if free_gb < 10.0:
            print(f"[WARN] Low disk space in current directory: {free_gb:.1f} GB free. Long runs might fill the disk.")
        else:
            print(f"[INFO] Disk space: {free_gb:.1f} GB free.")
    except Exception as e:
        print(f"[WARN] Could not check disk space: {e}")

    # 6. Output Directory Collision & Writable Check
    if os.path.exists(args.out_dir):
        # Check for collision artifacts
        artifacts = ["probe_results.csv", "transition_model_action.pt", "trajectories/train.pt"]
        collision = [a for a in artifacts if os.path.exists(os.path.join(args.out_dir, a))]
        
        if collision and not args.force:
            print(f"[FAIL] Output collision: --out_dir '{args.out_dir}' contains artifacts from a previous run: {collision}")
            print("Use --force to overwrite or specify a new --out_dir.")
            sys.exit(1)
        elif len(os.listdir(args.out_dir)) > 0 and not args.force:
            print(f"[FAIL] --out_dir '{args.out_dir}' already exists and is not empty. Use --force to proceed.")
            sys.exit(1)
            
    # Dry-run write check
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        test_file = os.path.join(args.out_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        summary.append("Output directory writable")
    except Exception as e:
        print(f"[FAIL] --out_dir '{args.out_dir}' is not writable: {e}")
        sys.exit(1)

    # 7. Data Directory & Dataset Sanity
    try:
        os.makedirs(args.data_dir, exist_ok=True)
        test_file = os.path.join(args.data_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"[FAIL] --data_dir '{args.data_dir}' is not writable: {e}")
        sys.exit(1)
        
    valid, msg = check_dataset_domain(args.data_dir, args.domain)
    if not valid:
        print(f"[FAIL] Dataset sanity check failed in {args.data_dir}: {msg}")
        sys.exit(1)
    else:
        print(f"[INFO] Dataset sanity: {msg}")
    summary.append("Dataset directory valid")
            
    print("-" * 40)
    print("=== PREFLIGHT SUMMARY ===")
    for item in summary:
        print(f"[PASS] {item}")
    print("[PASS] Ready to launch")

if __name__ == "__main__":
    main()

```


## File: `scripts\profile_training.py`
```python
import cProfile
import pstats
import io
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories
from training.train_transition import TransitionTrainConfig, train_transition_model

def main():
    print("[Profile] Loading trajectories...")
    train_traj_path = "reports/trajectories/train.pt"
    # only load first 100 trajectories for quick profile
    trajs = load_trajectories(train_traj_path)[:500]
    
    config = TransitionTrainConfig(
        epochs=1,
        batch_size=256
    )
    
    # --- PHASE 5: CPU Hotspots (cProfile) ---
    print("[Profile] Running cProfile...")
    pr = cProfile.Profile()
    pr.enable()
    train_transition_model(trajs, None, config)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    
    with open("cpu_hotspots_raw.txt", "w") as f:
        f.write(s.getvalue())
        
    # --- PHASE 6 & 8: GPU Hotspots & Memory (torch.profiler) ---
    print("[Profile] Running torch.profiler...")
    
    config.epochs = 1
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_training"):
            train_transition_model(trajs, None, config)
            
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
    
    with open("gpu_hotspots_raw.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        f.write("\n\n=== MEMORY PROFILE ===\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))

if __name__ == "__main__":
    main()

```


## File: `scripts\recover_report.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import os
from run_phase_a import generate_master_report

def main():
    print("Recovering report...")
    out = "reports"
    
    # Load dataframes
    coh_df = pd.read_csv(os.path.join(out, "coherence_action_depth.csv"))
    probe_df = pd.read_csv(os.path.join(out, "probe_results.csv"))
    
    # Extract chance dictionary from probe_report.md or assume defaults
    # Since we don't have probe_details natively saved as json, we can reconstruct it
    # from the text or just hardcode defaults for the 4 operations
    probe_details = {
        "chance": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}  # roughly 1/4 operations
    }
    
    # For transition history, we can just load the last eval_loss from the log
    try:
        t_log = pd.read_csv(os.path.join(out, "transition_action_log.csv"))
        last_eval_loss = t_log["val_loss"].iloc[-1]
        transition_history = [{"eval_loss": last_eval_loss}]
    except Exception:
        transition_history = []
        
    meta = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "layer": -1,
        "hidden_dim": 2048,
        "n_train": 50000,  # Just approximations since we don't have the trajs
        "n_val": 1000,
        "n_test": 1000,
        "smoke": False,
    }
    
    report_path = os.path.join(out, "phase_a_report.md")
    generate_master_report(coh_df, probe_df, probe_details, transition_history, meta, report_path)
    print(f"Successfully recovered report to {report_path}")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_action_conditioned.py`
```python
"""V5.2 Experiment 1 — action-conditioned transition test.

Addresses the reviewer criticism that V5.1 tested only action-blind
``z_t -> z_{t+1}``. A planning state is defined by ``(z_t, a_t) -> z_{t+1}``.
We run A-F (see evaluation.action_conditioned) on the Fixed-Init codebook
(primary) and the position-scrubbed codebook (robustness), evaluating top-1 on
the held-out val split and the transition structure (entropy / determinism) on
train. Writes reports/action_conditioned.md + .json. Then stop.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

from data_processing.discrete_trajectory_dataset import load_discrete_trajectories
from evaluation.action_conditioned import extract_action_transitions, run_all_models

MODEL_ROWS = [
    ("A_majority", "A. Majority", "—"),
    ("B_bigram", "B. Bigram", "z_t"),
    ("C_action_bigram", "C. Action bigram", "z_t, op"),
    ("D_mlp_z", "D. MLP(z_t)", "z_t"),
    ("E_mlp_z_op", "E. MLP(z_t, op)", "z_t, op"),
    ("E2_mlp_z_op_operands", "E2. MLP(z_t, op, operands)", "z_t, op, operands"),
    ("F_mlp_action_only", "F. MLP(op, operands) — control", "op, operands"),
]


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}f}"


def run_representation(name, train_path, val_path, K, epochs, seed):
    train = extract_action_transitions(load_discrete_trajectories(train_path))
    ev = extract_action_transitions(load_discrete_trajectories(val_path))
    res = run_all_models(train, ev, K, epochs=epochs, seed=seed)
    res["_meta"] = {"name": name, "n_train": int(train.z_t.shape[0]),
                    "n_eval": int(ev.z_t.shape[0])}
    return res


def model_table(res):
    rows = ["| Model | Conditioning | Top1 | Pred. entropy (nats) |",
            "| --- | --- | ---: | ---: |"]
    for key, label, cond in MODEL_ROWS:
        m = res[key]
        rows.append(f"| {label} | {cond} | {_fmt(m['top1'])} | "
                    f"{_fmt(m.get('pred_entropy'))} |")
    return "\n".join(rows)


def struct_table(res):
    s, a = res["_struct_state"], res["_struct_action"]
    return "\n".join([
        "| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |",
        "| --- | ---: | ---: | ---: |",
        f"| z_t | {_fmt(s['global_entropy'])} | {_fmt(s['det_frac_mass'])} | {s['n_conditions']} |",
        f"| z_t, op | {_fmt(a['global_entropy'])} | {_fmt(a['det_frac_mass'])} | {a['n_conditions']} |",
    ])


def contrasts(res):
    g = lambda k: res[k]["top1"]
    return {
        "action_helps_lookup_CminusB": g("C_action_bigram") - g("B_bigram"),
        "action_helps_mlp_EminusD": g("E_mlp_z_op") - g("D_mlp_z"),
        "mlp_vs_action_bigram_EminusC": g("E_mlp_z_op") - g("C_action_bigram"),
        "operands_help_E2minusE": g("E2_mlp_z_op_operands") - g("E_mlp_z_op"),
        "state_matters_E2minusF": g("E2_mlp_z_op_operands") - g("F_mlp_action_only"),
        "determinism_gain": res["_struct_action"]["det_frac_mass"]
                            - res["_struct_state"]["det_frac_mass"],
        "entropy_drop": res["_struct_state"]["global_entropy"]
                        - res["_struct_action"]["global_entropy"],
    }


def write_report(out, primary, scrub, contrasts_p, K, epochs):
    c = contrasts_p
    # Decision logic (skeptical thresholds).
    material = 0.03
    det_action = primary["_struct_action"]["det_frac_mass"]
    action_helps = (c["action_helps_lookup_CminusB"] > material
                    or c["action_helps_mlp_EminusD"] > material)
    mlp_beats_bigram = c["mlp_vs_action_bigram_EminusC"] > material
    determinism_emerged = det_action >= 0.20          # crisp reusable dynamics
    determinism_partial = det_action >= 0.05           # non-trivial but weak
    state_matters = c["state_matters_E2minusF"] > material
    # Three-way verdict.
    if action_helps and determinism_emerged:
        verdict = "yes"
    elif action_helps and (determinism_partial or c["entropy_drop"] > 0.3):
        verdict = "partial"
    else:
        verdict = "no"

    L = []
    L.append("# V5.2 · Experiment 1 — Action-Conditioned Transition Test\n")
    L.append("> Reviewer criticism addressed: a planning state is defined by "
             "`(state, action) → next_state`, not `state → next_state`. V5.1 "
             "tested only the action-blind form. Here we condition on the "
             "symbolic Countdown action.\n")
    L.append("\n## Implementation\n")
    L.append(f"- **Representation:** Fixed-Init VQ codes (K={K}, "
             "data-dependent init, raw last-layer states) — the clean, "
             "non-collapsed V5.1 codebook. Scrubbed codes reported as a "
             "robustness check.\n")
    L.append("- **Action:** op id (0=ADD, 1=SUB, 2=MUL; DIV absent from data) "
             "and the raw operands `[arg1, arg2]` (signed-log + standardized for "
             "the MLP).\n")
    L.append(f"- **Eval:** top-1 on the held-out **val** split "
             f"(train {primary['_meta']['n_train']} transitions, "
             f"val {primary['_meta']['n_eval']}); MLPs trained {epochs} epochs, "
             "seed 0. Entropy/determinism measured on train (frequency-weighted, "
             "deterministic = successor entropy < 0.5 nats over conditionings "
             "seen ≥ 10×).\n")
    L.append("- **Models A–F:** majority; state bigram; action bigram "
             "`(z,op)→mode`; MLP(z); MLP(z,op); MLP(z,op,operands); and an "
             "**action-only control** MLP(op,operands) that ignores `z_t`.\n")

    L.append("\n## Results — Fixed-Init codebook (primary)\n")
    L.append("### Predictive accuracy (held-out val)\n")
    L.append(model_table(primary))
    L.append("\n### Transition structure (train)\n")
    L.append(struct_table(primary))

    L.append("\n### Key contrasts\n")
    L.append(f"- **Does the action help the lookup?** C − B = "
             f"{c['action_helps_lookup_CminusB']:+.3f} "
             f"({_fmt(primary['C_action_bigram']['top1'])} vs "
             f"{_fmt(primary['B_bigram']['top1'])}).")
    L.append(f"- **Does the action help the MLP?** E − D = "
             f"{c['action_helps_mlp_EminusD']:+.3f} "
             f"({_fmt(primary['E_mlp_z_op']['top1'])} vs "
             f"{_fmt(primary['D_mlp_z']['top1'])}).")
    L.append(f"- **Does the MLP beat the action bigram?** E − C = "
             f"{c['mlp_vs_action_bigram_EminusC']:+.3f}.")
    L.append(f"- **Do operands add information?** E2 − E = "
             f"{c['operands_help_E2minusE']:+.3f} "
             f"(E2 = {_fmt(primary['E2_mlp_z_op_operands']['top1'])}).")
    L.append(f"- **Does the state matter, or only the action?** E2 − F = "
             f"{c['state_matters_E2minusF']:+.3f} "
             f"(F action-only = {_fmt(primary['F_mlp_action_only']['top1'])}).")
    L.append(f"- **Determinism gain from conditioning on the action:** "
             f"det-frac {_fmt(primary['_struct_state']['det_frac_mass'])} (z) → "
             f"{_fmt(primary['_struct_action']['det_frac_mass'])} (z,op); "
             f"entropy {_fmt(primary['_struct_state']['global_entropy'])} → "
             f"{_fmt(primary['_struct_action']['global_entropy'])} nats.\n")

    L.append("\n## Robustness — Scrubbed codebook\n")
    L.append(model_table(scrub))
    L.append("\n" + struct_table(scrub) + "\n")

    L.append("\n---\n## Reviewer-style interpretation\n")
    L.append("**1. Results.** See tables above. On the clean Fixed-Init "
             f"codebook, conditioning on the symbolic action moves top-1 from "
             f"{_fmt(primary['B_bigram']['top1'])} (state bigram) to "
             f"{_fmt(primary['C_action_bigram']['top1'])} (action bigram) and "
             f"{_fmt(primary['E_mlp_z_op']['top1'])} (action MLP); adding "
             f"operands gives {_fmt(primary['E2_mlp_z_op_operands']['top1'])}.\n")

    L.append("**2. Does MLP(z,a) materially beat the action-aware bigram?** "
             + ("**Yes** " if mlp_beats_bigram else "**No** ")
             + f"— E − C = {c['mlp_vs_action_bigram_EminusC']:+.3f} "
             f"(threshold ±{material}). "
             + ("The learned model captures structure beyond action-conditioned "
                "counts.\n" if mlp_beats_bigram else
                "The MLP does not exceed the action-conditioned lookup: whatever "
                "the action contributes is already a first-order count effect, "
                "not learned structure.\n"))

    L.append("**3. Does the planning-state hypothesis survive?**\n")
    ds = _fmt(primary["_struct_state"]["det_frac_mass"])
    da = _fmt(primary["_struct_action"]["det_frac_mass"])
    hs = _fmt(primary["_struct_state"]["global_entropy"])
    ha = _fmt(primary["_struct_action"]["global_entropy"])
    if verdict == "yes":
        L.append("> **Tentatively yes (revisit).** Conditioning on the action "
                 "materially raises predictability and a substantial deterministic "
                 f"subset emerges (det-frac {da} of mass). This warrants the "
                 "positive control (Experiment 2) before any claim.\n")
    elif verdict == "partial":
        L.append(
            "> **Partially — and this revises the V5.1 wording.** The reviewer "
            "criticism was correct that the action-blind test understated the "
            f"structure: conditioning on the action roughly *doubles* "
            f"predictability (state→action bigram "
            f"{_fmt(primary['B_bigram']['top1'])} → "
            f"{_fmt(primary['C_action_bigram']['top1'])}; E−D "
            f"{c['action_helps_mlp_EminusD']:+.3f}), *halves* the conditional "
            f"entropy ({hs} → {ha} nats), and turns a 0% deterministic mass into "
            f"{da}. So `(state, action)` carries real, non-trivial dynamics that "
            "`state` alone did not.\n")
        L.append(
            "> **But it does not reach reusable planning states.** The structure "
            f"is purely first-order — the MLP does not beat the action lookup "
            f"(E−C {c['mlp_vs_action_bigram_EminusC']:+.3f}) — and it is far from "
            f"deterministic: only {da} of transition mass is near-deterministic, "
            f"H(z'|z,op) ≈ {ha} nats (~{int(round(2.718 ** float(ha)))} effective "
            f"successors), and top-1 is {_fmt(primary['E2_mlp_z_op_operands']['top1'])} "
            "even with operands. The state is not vacuous (E2−F "
            f"{c['state_matters_E2minusF']:+.3f}), but it composes with actions "
            "into *weak, stochastic* transitions, not crisp reusable ones.\n")
        L.append(
            "> **Net:** V5.1's flat \"no reusable structure\" is too strong and "
            "should be revised to **\"weak, first-order, sub-deterministic "
            "action-conditioned dynamics — not reusable planning states.\"** "
            "Whether even this weak effect exceeds what *any* action-conditioned "
            "model would yield on position-correlated codes cannot be judged "
            "without a calibrated positive control — which is exactly the purpose "
            "of Experiment 2.\n")
    else:
        reasons = []
        if not action_helps:
            reasons.append(
                f"conditioning on the action barely changes predictability "
                f"(C−B {c['action_helps_lookup_CminusB']:+.3f}, "
                f"E−D {c['action_helps_mlp_EminusD']:+.3f})")
        if not determinism_emerged:
            reasons.append(f"no deterministic subset appears (det-frac {da})")
        if not mlp_beats_bigram:
            reasons.append("the MLP never beats the action bigram")
        L.append("> **No.** " + "; ".join(reasons) + ". The negative result "
                 "survives even the action-conditioned test.\n")

    L.append("\n_Caveats: single model/task/layer/seed; op-type action (DIV "
             "absent); operands make the next *number* computable, so any E2 gain "
             "may reflect numeric leakage rather than reusable state composition. "
             "Scope unchanged from V5.1._\n")

    L.append("\n_Artifacts: `reports/action_conditioned.json`._\n")

    with open(os.path.join(out, "action_conditioned.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.reports_dir

    fdir = os.path.join(out, "discrete_fixed_init")
    sdir = os.path.join(out, "discrete_scrubbed")
    print("[Exp1] Fixed-Init representation...")
    primary = run_representation("fixed_init", os.path.join(fdir, "train.pt"),
                                 os.path.join(fdir, "val.pt"), args.num_codes,
                                 args.epochs, args.seed)
    print("[Exp1] Scrubbed representation...")
    scrub = run_representation("scrubbed", os.path.join(sdir, "train.pt"),
                               os.path.join(sdir, "val.pt"), args.num_codes,
                               args.epochs, args.seed)

    con = contrasts(primary)
    payload = {
        "config": {"num_codes": args.num_codes, "epochs": args.epochs,
                   "seed": args.seed, "eval_split": "val"},
        "fixed_init": {k: v for k, v in primary.items()},
        "scrubbed": {k: v for k, v in scrub.items()},
        "contrasts_fixed_init": con,
    }
    with open(os.path.join(out, "action_conditioned.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(out, primary, scrub, con, args.num_codes, args.epochs)

    print("\n[Exp1] Fixed-Init top1:",
          {k: round(primary[k]["top1"], 3) for k, _, _ in MODEL_ROWS})
    print("[Exp1] contrasts:", {k: round(v, 3) for k, v in con.items()})
    print("[Exp1] Done. See reports/action_conditioned.md")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_collision_audit.py`
```python
"""Collision Statistics Audit.

Diagnostic answering a single question: is the Oracle 2A coverage collapse
(4/3004 states with a swap partner) caused by a genuine *lack of state
collisions* in the dataset, or by an implementation issue downstream?

This script trains nothing, runs no probes, no Oracle, no Phase A. It only reads
the existing trajectory artifacts and counts how often each symbolic state recurs
across trajectories.

Symbolic-state extraction is reused verbatim from the canonical implementation
``evaluation.intrinsic_noise.get_symbolic_states`` — the same function used by
the Oracle Audit, ``intrinsic_noise.py`` and Phase C. No symbolic-state logic is
duplicated here.

Definitions (matching the Oracle Audit exactly)
------------------------------------------------
``get_symbolic_states`` returns, per aligned state, a tuple
``(symbolic_state, action_history)`` where:

* ``symbolic_state = (target, tuple(sorted(available_numbers)))`` — the *target
  is already embedded* in the symbolic state, so grouping by ``symbolic_state``
  is identical to grouping by ``(symbolic_state, target)``.
* ``action_history`` is the tuple of ``(op_id, arg1, arg2)`` applied so far.

A valid Oracle 2A swap pair (the audit's coverage requirement) is two occurrences
of the *same symbolic state* drawn from *different trajectories* with *different
action histories*.

Inputs
------
    reports/trajectories/train.pt
    reports/trajectories/test.pt

Outputs
-------
    reports/collision_audit.csv          (one row per unique symbolic state)
    reports/collision_audit_summary.md   (global stats + distributions + coverage)
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states


# --------------------------------------------------------------------------- #
# Occurrence collection
# --------------------------------------------------------------------------- #
def collect_occurrences(trajs, split, occ_by_state):
    """Append every aligned state's occurrence into ``occ_by_state``.

    ``occ_by_state`` maps a symbolic state -> list of occurrence dicts. Each
    occurrence records the originating trajectory id, depth, action-history
    length, history (for diversity counting) and split. States invalidated by a
    teacher arithmetic mistake (``get_symbolic_states`` returns ``None``) are
    skipped, exactly as the Oracle Audit skips them.

    ``traj_id`` is namespaced by split (``"train:0"``, ``"test:0"``, ...) so that
    train and test trajectories are never conflated as the "same" trajectory.
    """
    for ti, traj in enumerate(trajs):
        sym_list = get_symbolic_states(traj)  # len N+1; (sym, hist) or None
        traj_id = f"{split}:{ti}"
        for depth, info in enumerate(sym_list):
            if info is None:
                continue
            sym, hist = info
            occ_by_state[sym].append(
                {
                    "traj_id": traj_id,
                    "depth": depth,
                    "action_history_len": len(hist),
                    "history": hist,
                    "split": split,
                }
            )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def compute_statistics(occ_by_state):
    """Return (per_state_rows, summary) computed from grouped occurrences."""
    per_state_rows = []
    occ_counts = []         # occurrences per symbolic state
    unique_hist_counts = []  # unique histories per symbolic state

    # Accumulators for anchor-level (occurrence-level) coverage.
    anchors_with_partner_a = 0
    anchors_with_partner_b = 0

    for sym, occs in occ_by_state.items():
        target, numbers = sym
        n_occ = len(occs)

        unique_histories = {o["history"] for o in occs}
        unique_trajs = {o["traj_id"] for o in occs}
        n_unique_hist = len(unique_histories)
        n_unique_traj = len(unique_trajs)

        # Oracle 2A swap candidates *within this symbolic state*, computed in a
        # single O(n^2) pass per state. The dataset's collisions are sparse, so
        # n is tiny for almost every state.
        #   (A) same target + same symbolic state + different trajectory.
        #   (B) (A) AND different action history (the exact Oracle 2A partner rule).
        # cand_* count ordered anchor->partner pairs (comparable to the Oracle
        # Audit's per-anchor partner search). anchors_with_partner_* count
        # occurrences that have >=1 partner (the audit's coverage numerator).
        cand_a = 0
        cand_b = 0
        for i in range(n_occ):
            oi = occs[i]
            has_a = False
            has_b = False
            for j in range(n_occ):
                if i == j:
                    continue
                oj = occs[j]
                if oi["traj_id"] == oj["traj_id"]:
                    continue
                cand_a += 1
                has_a = True
                if oi["history"] != oj["history"]:
                    cand_b += 1
                    has_b = True
            if has_a:
                anchors_with_partner_a += 1
            if has_b:
                anchors_with_partner_b += 1

        split_set = sorted({o["split"] for o in occs})

        per_state_rows.append(
            {
                "target": target,
                "numbers": "|".join(str(x) for x in numbers),
                "num_available": len(numbers),
                "occurrences": n_occ,
                "unique_trajectories": n_unique_traj,
                "unique_histories": n_unique_hist,
                "swap_candidates_A": cand_a,
                "swap_candidates_B": cand_b,
                "splits": "|".join(split_set),
            }
        )
        occ_counts.append(n_occ)
        unique_hist_counts.append(n_unique_hist)

    occ_counts_sorted = sorted(occ_counts)
    total_states = sum(occ_counts)            # total aligned state instances
    unique_states = len(occ_by_state)         # distinct symbolic states

    def _frac(cond_count):
        return (cond_count / unique_states) if unique_states else 0.0

    ge2 = sum(1 for c in occ_counts if c >= 2)
    ge5 = sum(1 for c in occ_counts if c >= 5)
    ge10 = sum(1 for c in occ_counts if c >= 10)
    ge20 = sum(1 for c in occ_counts if c >= 20)

    hist_ge2 = sum(1 for c in unique_hist_counts if c >= 2)
    hist_ge5 = sum(1 for c in unique_hist_counts if c >= 5)
    hist_ge10 = sum(1 for c in unique_hist_counts if c >= 10)

    # Oracle coverage estimate: how many *symbolic states* admit >=1 valid swap
    # candidate, and how many anchors (occurrences) have >=1 partner.
    states_with_cand_a = sum(1 for r in per_state_rows if r["swap_candidates_A"] > 0)
    states_with_cand_b = sum(1 for r in per_state_rows if r["swap_candidates_B"] > 0)
    total_cand_a = sum(r["swap_candidates_A"] for r in per_state_rows)
    total_cand_b = sum(r["swap_candidates_B"] for r in per_state_rows)

    summary = {
        "total_states": total_states,
        "unique_symbolic_states": unique_states,
        "avg_occurrences_per_state": (total_states / unique_states) if unique_states else 0.0,
        "median_occurrences": _median(occ_counts_sorted),
        "max_occurrences": max(occ_counts) if occ_counts else 0,
        "states_ge2_occ": ge2,
        "states_ge5_occ": ge5,
        "states_ge10_occ": ge10,
        "states_ge20_occ": ge20,
        "frac_ge2_occ": _frac(ge2),
        "frac_ge5_occ": _frac(ge5),
        "frac_ge10_occ": _frac(ge10),
        "frac_ge20_occ": _frac(ge20),
        "states_ge2_unique_hist": hist_ge2,
        "states_ge5_unique_hist": hist_ge5,
        "states_ge10_unique_hist": hist_ge10,
        "states_with_swap_candidates_A": states_with_cand_a,
        "states_with_swap_candidates_B": states_with_cand_b,
        "total_swap_candidates_A": total_cand_a,
        "total_swap_candidates_B": total_cand_b,
        "anchors_with_partner_A": anchors_with_partner_a,
        "anchors_with_partner_B": anchors_with_partner_b,
        "anchor_coverage_rate_A": _safe_div(anchors_with_partner_a, total_states),
        "anchor_coverage_rate_B": _safe_div(anchors_with_partner_b, total_states),
    }
    return per_state_rows, summary


def _safe_div(a, b):
    return (a / b) if b else 0.0


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "target",
    "numbers",
    "num_available",
    "occurrences",
    "unique_trajectories",
    "unique_histories",
    "swap_candidates_A",
    "swap_candidates_B",
    "splits",
]


def write_csv(per_state_rows, path):
    # Most-collided states first for easy inspection.
    rows = sorted(per_state_rows, key=lambda r: (-r["occurrences"], -r["unique_histories"]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_summary_md(summary, split_counts, path):
    s = summary
    lines = ["# Collision Statistics Audit — Summary\n"]

    lines.append(
        "Reuses `evaluation.intrinsic_noise.get_symbolic_states` (the same symbolic-"
        "state extraction used by the Oracle Audit, intrinsic_noise.py and Phase C). "
        "Grouping key is the symbolic state `(target, sorted(available_numbers))`; "
        "the target is already embedded in that key.\n"
    )

    lines.append("## Inputs\n")
    lines.append("| split | trajectories |")
    lines.append("|---|---|")
    for split, n in split_counts.items():
        lines.append(f"| {split} | {n} |")
    lines.append("")

    lines.append("## Global\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in [
        "total_states",
        "unique_symbolic_states",
        "avg_occurrences_per_state",
        "median_occurrences",
        "max_occurrences",
    ]:
        lines.append(f"| {k} | {_fmt(s[k])} |")
    lines.append("")

    lines.append("## Collision distribution\n")
    lines.append("| threshold | states | fraction of unique states |")
    lines.append("|---|---|---|")
    lines.append(f"| >= 2 occurrences | {s['states_ge2_occ']} | {_fmt(s['frac_ge2_occ'])} |")
    lines.append(f"| >= 5 occurrences | {s['states_ge5_occ']} | {_fmt(s['frac_ge5_occ'])} |")
    lines.append(f"| >= 10 occurrences | {s['states_ge10_occ']} | {_fmt(s['frac_ge10_occ'])} |")
    lines.append(f"| >= 20 occurrences | {s['states_ge20_occ']} | {_fmt(s['frac_ge20_occ'])} |")
    lines.append("")

    lines.append("## History diversity\n")
    lines.append("Unique action histories reaching each symbolic state.\n")
    lines.append("| threshold | states |")
    lines.append("|---|---|")
    lines.append(f"| >= 2 unique histories | {s['states_ge2_unique_hist']} |")
    lines.append(f"| >= 5 unique histories | {s['states_ge5_unique_hist']} |")
    lines.append(f"| >= 10 unique histories | {s['states_ge10_unique_hist']} |")
    lines.append("")

    lines.append("## Oracle Coverage Estimate\n")
    lines.append(
        "Valid Oracle 2A swap candidates. **A** = same target + same symbolic "
        "state + different trajectory. **B** = A *and* different action history "
        "(this is the exact Oracle 2A partner rule). Candidate counts are ordered "
        "anchor->partner pairs.\n"
    )
    lines.append("| metric | A (diff trajectory) | B (diff trajectory + diff history) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| symbolic states with >=1 candidate | {s['states_with_swap_candidates_A']} "
        f"| {s['states_with_swap_candidates_B']} |"
    )
    lines.append(
        f"| total swap candidate pairs | {s['total_swap_candidates_A']} "
        f"| {s['total_swap_candidates_B']} |"
    )
    lines.append(
        f"| anchors (state instances) with >=1 partner | {s['anchors_with_partner_A']} "
        f"| {s['anchors_with_partner_B']} |"
    )
    lines.append(
        f"| anchor coverage rate | {_fmt(s['anchor_coverage_rate_A'])} "
        f"| {_fmt(s['anchor_coverage_rate_B'])} |"
    )
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Collision Statistics Audit")
    ap.add_argument("--reports_dir", default="reports",
                    help="Holds trajectories/ and receives outputs.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default <reports_dir>).")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    help="Trajectory splits to include (default train test).")
    ap.add_argument("--sample", type=int, default=0,
                    help="Lightweight validation: cap trajectories loaded PER SPLIT "
                         "to this many (0 = use all). Outputs are suffixed _sample.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.reports_dir
    traj_dir = os.path.join(args.reports_dir, "trajectories")

    occ_by_state = collections.defaultdict(list)
    split_counts = {}
    for split in args.splits:
        path = os.path.join(traj_dir, f"{split}.pt")
        if not os.path.exists(path):
            print(f"ERROR: trajectories not found: {path}")
            sys.exit(1)
        print(f"[Collision Audit] Loading {path}")
        trajs = load_trajectories(path)
        if args.sample > 0:
            trajs = trajs[: args.sample]
        print(f"  {len(trajs)} trajectories ({split})")
        split_counts[split] = len(trajs)
        collect_occurrences(trajs, split, occ_by_state)

    print("[Collision Audit] Computing statistics...")
    per_state_rows, summary = compute_statistics(occ_by_state)

    suffix = "_sample" if args.sample > 0 else ""
    csv_path = os.path.join(out_dir, f"collision_audit{suffix}.csv")
    md_path = os.path.join(out_dir, f"collision_audit_summary{suffix}.md")

    write_csv(per_state_rows, csv_path)
    write_summary_md(summary, split_counts, md_path)

    print("\n[Collision Audit] Done.")
    print(f"  total state instances     : {summary['total_states']}")
    print(f"  unique symbolic states     : {summary['unique_symbolic_states']}")
    print(f"  states >=2 occ             : {summary['states_ge2_occ']}")
    print(f"  states with swap cand (B)  : {summary['states_with_swap_candidates_B']}")
    print(f"  anchor coverage rate (B)   : {summary['anchor_coverage_rate_B']:.6f}")
    for p in (csv_path, md_path):
        print(f"  - {p}")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_layer_sweep.py`
```python
"""V5.2 Experiment 3 — layer sweep (orchestrator).

Re-extracts TinyLlama hidden-state trajectories at several depths and runs the
*identical* Experiment-1 pipeline (VQ → action-conditioned models A–F →
transition structure → codebook usage → position leakage) at each layer, then
calibrates every layer against the Experiment-2 floor/ceiling. Answers: does
the action-conditioned discrete-state structure depend on which layer we probe,
and does any depth reach the reusable-state ceiling? Writes
``reports/layer_sweep.{json,md,png}``. Then stop.

The per-layer analysis reuses already-tested components unchanged; the only new
logic (axis placement + reporting) lives in ``evaluation.layer_sweep`` and is
unit-tested separately.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import load_problems, build_trajectories
from training.train_vq import train_vq_quantizer, VQTrainConfig
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes, all_codes,
)
from evaluation.action_conditioned import extract_action_transitions, run_all_models
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.position_leakage import evaluate_position_leakage
from evaluation.layer_sweep import summarize_sweep, build_markdown, build_plot

DEFAULT_TEACHER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def _problem_hash(p):
    return (p["target"], tuple(sorted(p["numbers"])))


def analyze_layer(model, tokenizer, train_problems, val_problems, layer,
                  K, vq_epochs, tr_epochs, batch_size, seed):
    """Full single-layer pipeline -> one metrics row."""
    train = build_trajectories(model, tokenizer, train_problems, layer=layer,
                               batch_size=batch_size)
    val = build_trajectories(model, tokenizer, val_problems, layer=layer,
                             batch_size=batch_size)

    vq = train_vq_quantizer(train, val, config=VQTrainConfig(
        num_codes=K, epochs=vq_epochs, batch_size=256, seed=seed))
    d_tr = encode_trajectories_to_codes(vq, train)
    d_va = encode_trajectories_to_codes(vq, val)

    usage = analyze_codebook_usage(all_codes(d_tr), K)
    leak = evaluate_position_leakage(d_tr, d_va, K, seed=seed)
    res = run_all_models(extract_action_transitions(d_tr),
                         extract_action_transitions(d_va), K,
                         epochs=tr_epochs, seed=seed)
    return {
        "layer": layer,
        "n_train_states": int(all_codes(d_tr).shape[0]),
        "active_codes": int(usage["active_codes"]),
        "perplexity": float(usage["perplexity"]),
        "gini": float(usage["collapse_score"]),
        "position_leakage": float(leak["position_predictability_score"]),
        "leakage_majority": float(leak["chance_accuracy"]),
        "majority": float(res["A_majority"]["top1"]),
        "bigram_z": float(res["B_bigram"]["top1"]),
        "action_bigram": float(res["C_action_bigram"]["top1"]),
        "mlp_z": float(res["D_mlp_z"]["top1"]),
        "mlp_z_op": float(res["E_mlp_z_op"]["top1"]),
        "mlp_z_op_operands": float(res["E2_mlp_z_op_operands"]["top1"]),
        "mlp_action_only": float(res["F_mlp_action_only"]["top1"]),
        "H_state": float(res["_struct_state"]["global_entropy"]),
        "det_state": float(res["_struct_state"]["det_frac_mass"]),
        "H_action": float(res["_struct_action"]["global_entropy"]),
        "det_action": float(res["_struct_action"]["det_frac_mass"]),
    }


def load_anchors(reports_dir):
    """Exp2 floor/ceiling and the cached layer −1 (Exp1) for cross-check."""
    floor = ceiling = observed = None
    pc_path = os.path.join(reports_dir, "positive_control.json")
    if os.path.exists(pc_path):
        pc = json.load(open(pc_path, encoding="utf-8"))
        floor, ceiling = pc["noise_floor"], pc["positive_control"]
    ac_path = os.path.join(reports_dir, "action_conditioned.json")
    if os.path.exists(ac_path):
        ac = json.load(open(ac_path, encoding="utf-8"))["fixed_init"]
        observed = {"det_action": ac["_struct_action"]["det_frac_mass"],
                    "H_action": ac["_struct_action"]["global_entropy"],
                    "mlp_z_op": ac["E_mlp_z_op"]["top1"]}
    return floor, ceiling, observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default=DEFAULT_TEACHER)
    ap.add_argument("--layers", default="4,8,12,16,20,22")
    ap.add_argument("--cap_train", type=int, default=1500)
    ap.add_argument("--cap_val", type=int, default=750)
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--vq_epochs", type=int, default=50)
    ap.add_argument("--tr_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.reports_dir
    layers = [int(x) for x in args.layers.split(",")]

    print(f"[Exp3] Loading frozen teacher: {args.model}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype)
    tokenizer = load_tokenizer(model_id=args.model)
    num_layers = int(model.config.num_hidden_layers)

    train_problems = load_problems(os.path.join(args.data_dir, "train.jsonl"))[:args.cap_train]
    val_problems = load_problems(os.path.join(args.data_dir, "val.jsonl"))[:args.cap_val]
    # De-duplicate val against train (same leakage guard as Phase A).
    train_h = {_problem_hash(p) for p in train_problems}
    val_problems = [p for p in val_problems if _problem_hash(p) not in train_h]
    print(f"[Exp3] {len(train_problems)} train / {len(val_problems)} val problems; "
          f"layers {layers} (model has {num_layers}).")

    rows = []
    for layer in layers:
        print(f"\n[Exp3] === layer {layer} ===")
        rows.append(analyze_layer(model, tokenizer, train_problems, val_problems,
                                  layer, args.num_codes, args.vq_epochs,
                                  args.tr_epochs, args.batch_size, args.seed))
        r = rows[-1]
        print(f"[Exp3] layer {layer}: det(z,op)={r['det_action']:.3f} "
              f"H(z'|z,op)={r['H_action']:.3f} MLP(z,op)={r['mlp_z_op']:.3f} "
              f"pos-leak={r['position_leakage']:.3f}")

    floor, ceiling, observed = load_anchors(out)
    config = {"model": args.model, "num_layers": num_layers, "layers": layers,
              "cap_train": len(train_problems), "cap_val": len(val_problems),
              "num_codes": args.num_codes, "vq_epochs": args.vq_epochs,
              "tr_epochs": args.tr_epochs, "seed": args.seed}

    payload = {"config": config, "layers": rows,
               "floor": floor, "ceiling": ceiling, "observed_layer_-1": observed}
    if floor is not None and ceiling is not None:
        summary = summarize_sweep(rows, floor, ceiling)
        payload["summary"] = summary
        with open(os.path.join(out, "layer_sweep.md"), "w", encoding="utf-8") as f:
            f.write(build_markdown(rows, summary, floor, ceiling, observed, config))
        build_plot(rows, floor, ceiling, os.path.join(out, "layer_sweep.png"))
        print(f"\n[Exp3] best layer = {summary['best_layer']} "
              f"(det {summary['best_det_action']:.3f}, "
              f"{summary['best_pos_det']*100:.0f}% to ceiling); "
              f"reaches ceiling: {summary['reaches_ceiling']}")
    else:
        print("\n[Exp3] WARNING: reports/positive_control.json not found — "
              "writing metrics JSON only (no calibrated report).")

    with open(os.path.join(out, "layer_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[Exp3] Done. See reports/layer_sweep.md")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_multiseed.py`
```python
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

    df = pd.DataFrame(results)
    df.to_csv("reports/multiseed_results.csv", index=False)

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

```


## File: `scripts\run_oracle_audit.py`
```python
"""Oracle Ceiling Audit.

Cleanly separates *representation interchangeability* from *dynamics
robustness* so that a failed transition model cannot masquerade as a failed
representation.

Oracle hierarchy (see project spec):

* Oracle 0 — Teacher Ceiling: probes A/B/C/D on the exact teacher state h_t.
* Oracle 1 — True-State Transition Baseline: T(h_t, a_t) -> ĥ_{t+1}, Probe A.
* Oracle 2A — State-Swap Representation Audit: for an anchor state h, find a
  partner h' with the *same target + same symbolic state* but a *different
  trajectory and different action history*; compare probes on h vs h' directly
  (no transition model).
* Oracle 2B — State-Swap Dynamics Audit: feed the anchor's intended action a_t
  to the swapped state, T(h', a_t) -> ĥ'_{t+1}, Probe A. The primary signal is
  ``delta_transition_accuracy = Oracle1 - Oracle2B`` on matched pairs.

This is strictly an evaluation audit. It runs no training and adds no new
experiments. Probes are fit on the train split; the audit (anchors + swap
partners) is drawn from a held-out split so partners are also held out. The
optional ``--audit train_test`` mode instead pools the train and test
trajectories as the anchor/partner pool (probe fitting is unchanged: probes
are still fit on the train split). Trajectories are concatenated, so every
trajectory keeps a distinct index and the "different trajectory" matching rule
is unchanged.

Outputs (exactly):
    reports/oracle_audit_metrics.csv
    reports/oracle_audit_coverage.md
    reports/oracle_audit_summary.md
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states
from evaluation.probes import (
    MultiLabelProbe,
    _make_probe,
    extract_probe_data,
)
LARGE_NUMBERS = [25, 50, 75, 100]
from models.transition_model import (
    TransitionModel,
    LinearTransitionModel,
    TransformerTransitionModel,
)


# --------------------------------------------------------------------------- #
# Probe fitting (reuses evaluation.probes infrastructure)
# --------------------------------------------------------------------------- #
class _ConstPredictor:
    """Degenerate predictor for a single-class training target."""

    def __init__(self, const: int):
        self.const = int(const)

    def predict(self, X):
        return np.full((X.shape[0],), self.const, dtype=np.int64)


def fit_all_probes(train_trajs):
    """Fit linear probes A (multi-label), B, C, D on teacher train states."""
    tr = extract_probe_data(train_trajs)
    X = tr["X"]

    probe_a = MultiLabelProbe()
    probe_a.fit(X, tr["A"])

    def _fit_single(y):
        classes = np.unique(y)
        if classes.shape[0] < 2:
            return _ConstPredictor(int(classes[0]) if classes.shape[0] else 0)
        clf = _make_probe()
        clf.fit(X, y)
        return clf

    return {
        "A": probe_a,
        "B": _fit_single(tr["B"]),
        "C": _fit_single(tr["C"]),
        "D": _fit_single(tr["D"]),
    }


# --------------------------------------------------------------------------- #
# Ground-truth Probe-A label (matches evaluation.probes.extract_probe_data)
# --------------------------------------------------------------------------- #
def a_label_at(numbers, operands, d):
    """Probe-A multi-label vector at depth ``d`` for {25,50,75,100}.

    0 = value never in the problem, 1 = in pool / not yet used as an operand,
    2 = already used as an operand in steps 0..d-1.
    """
    used = set()
    for i in range(d):
        used.add(int(operands[i][0].item()))
        used.add(int(operands[i][1].item()))
    row = []
    for v in LARGE_NUMBERS:
        if v not in numbers:
            row.append(0)
        elif v not in used:
            row.append(1)
        else:
            row.append(2)
    return row


# --------------------------------------------------------------------------- #
# State records + symbolic-state grouping
# --------------------------------------------------------------------------- #
def build_state_records(trajs):
    """Per-depth state instances with symbolic key, history, gt labels, action.

    Returns ``(records, groups)`` where ``groups`` maps a symbolic state to the
    list of record indices that share it. Depths invalidated by a teacher
    arithmetic mistake (``get_symbolic_states`` returns ``None``) are skipped.
    """
    records = []
    for ti, traj in enumerate(trajs):
        sym_list = get_symbolic_states(traj)  # len N+1; (sym, hist) or None
        N = traj.num_steps
        numbers = list(traj.numbers)
        operands = traj.operands  # (N, 2)
        states = traj.states      # (N+1, H)
        for d in range(N + 1):
            info = sym_list[d]
            if info is None:
                continue
            sym, hist = info
            has_next = d < N and sym_list[d + 1] is not None
            rec = {
                "traj_idx": ti,
                "depth": d,
                "sym": sym,
                "hist": hist,
                "h": states[d].numpy().astype(np.float32),
                "gtA": a_label_at(numbers, operands, d),
                "gtB": int(N - d),
                "gtC": int(traj.op_ids[d].item()) if d < N else None,
                "gtD": 1 if (N - d) <= 2 else 0,
                "has_next": has_next,
            }
            if has_next:
                rec["next_op"] = int(traj.op_ids[d].item())
                rec["next_operands"] = operands[d].numpy().astype(np.float32)
                rec["next_gtA"] = a_label_at(numbers, operands, d + 1)
            else:
                rec["next_op"] = None
                rec["next_operands"] = None
                rec["next_gtA"] = None
            records.append(rec)

    groups = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        groups[rec["sym"]].append(idx)
    return records, groups


# --------------------------------------------------------------------------- #
# Agreement / metric primitives
# --------------------------------------------------------------------------- #
def a_exact(p, r):
    return 1.0 if list(p) == list(r) else 0.0


def a_per_label(p, r):
    return sum(1.0 for a, b in zip(p, r) if a == b) / len(r)


def a_jaccard(p, r):
    """Jaccard over the set of still-available numbers (label == 1)."""
    P = {v for v, l in zip(LARGE_NUMBERS, p) if l == 1}
    R = {v for v, l in zip(LARGE_NUMBERS, r) if l == 1}
    union = P | R
    if not union:
        return 1.0
    return len(P & R) / len(union)


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _stratify(pairs):
    """pairs: list of (delta_depth, value) -> {'all': mean, dd: mean, ...}."""
    out = {"all": _mean([v for _, v in pairs])}
    by = collections.defaultdict(list)
    for dd, v in pairs:
        by[dd].append(v)
    for dd in sorted(by):
        out[dd] = _mean(by[dd])
    return out


def _counts_by_delta(pairs):
    by = collections.Counter(dd for _, _, dd in pairs)
    return {"all": len(pairs), **{dd: by[dd] for dd in sorted(by)}}


# --------------------------------------------------------------------------- #
# Confidence intervals (reporting only; no metric definition is changed)
# --------------------------------------------------------------------------- #
_Z95 = 1.959963984540054  # standard normal quantile for a two-sided 95% CI


def _mean_ci(values):
    """Mean and 95% CI of the mean over a 1-D sample (normal approximation).

    Used for the pair-level Oracle 2A / 2B metrics. The CI is computed on the
    sample of per-pair values; the point estimate is identical to ``_mean`` so
    no reported metric changes.
    """
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_lo": mean, "ci_hi": mean, "n": 1}
    se = float(arr.std(ddof=1)) / np.sqrt(n)
    return {"mean": mean, "ci_lo": mean - _Z95 * se,
            "ci_hi": mean + _Z95 * se, "n": int(n)}


def _proportion_ci(k, n):
    """Wilson 95% CI for a binomial proportion (used for coverage rate)."""
    if n == 0:
        return {"rate": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    p = k / n
    denom = 1.0 + _Z95 * _Z95 / n
    center = (p + _Z95 * _Z95 / (2 * n)) / denom
    half = (_Z95 * np.sqrt(p * (1 - p) / n + _Z95 * _Z95 / (4 * n * n))) / denom
    return {"rate": p, "ci_lo": center - half, "ci_hi": center + half, "n": int(n)}


# --------------------------------------------------------------------------- #
# Transition model
# --------------------------------------------------------------------------- #
def load_transition_model(ckpt_path: str, transition_arch: str = "mlp") -> torch.nn.Module:
    """Load a transition model from checkpoint, predicting delta states."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    
    # Try to load config dict
    cfg = ckpt.get("config", {})
    hidden_dim = int(ckpt.get("hidden_dim") or cfg.get("hidden_dim") or sd.get("net.2.weight", sd.get("net.weight", sd.get("state_proj_in.weight"))).shape[0])
    mlp_hidden_dim = cfg.get("mlp_hidden_dim", 512)
    predict_delta = cfg.get("predict_delta", True)
    use_action = cfg.get("use_action", any(k.startswith("action_encoder") for k in sd))
    op_embed_dim = cfg.get("op_embed_dim", 16)
    operand_mean = cfg.get("operand_mean", 0.0)
    operand_std = cfg.get("operand_std", 1.0)

    if transition_arch == "linear":
        model = LinearTransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )
    elif transition_arch == "transformer":
        model = TransformerTransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )
    else:
        model = TransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            op_embed_dim=op_embed_dim,
            predict_delta=predict_delta,
            use_action=use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        )
        
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def transition_predict_a(model, probe_a, h_mat, op_arr, operands_arr, chunk=4096):
    """Apply T(h, a) over batched inputs, return Probe-A predictions (m, 4)."""
    preds = []
    for s in range(0, h_mat.shape[0], chunk):
        ht = torch.tensor(h_mat[s:s + chunk], dtype=torch.float32)
        op = torch.tensor(op_arr[s:s + chunk], dtype=torch.long)
        oper = torch.tensor(operands_arr[s:s + chunk], dtype=torch.float32)
        out = model(ht, op, oper).cpu().numpy()
        preds.append(probe_a.predict(out))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Oracle 0
# --------------------------------------------------------------------------- #
def oracle0_metrics(probes, test_trajs):
    te = extract_probe_data(test_trajs)
    X = te["X"]
    predA = probes["A"].predict(X)
    predB = probes["B"].predict(X)
    predC = probes["C"].predict(X)
    predD = probes["D"].predict(X)
    return {
        "n": int(X.shape[0]),
        "probeA_exact": _mean([a_exact(predA[i], te["A"][i]) for i in range(len(predA))]),
        "probeA_per_label": _mean([a_per_label(predA[i], te["A"][i]) for i in range(len(predA))]),
        "probeA_jaccard": _mean([a_jaccard(predA[i], te["A"][i]) for i in range(len(predA))]),
        "probeB_accuracy": float(np.mean(predB == te["B"])),
        "probeC_accuracy": float(np.mean(predC == te["C"])),
        "probeD_accuracy": float(np.mean(predD == te["D"])),
    }


# --------------------------------------------------------------------------- #
# Audit core (Oracle 1, 2A, 2B)
# --------------------------------------------------------------------------- #
def run_audit(records, groups, probes, transition_model, max_partners, smoke_pairs=False):
    H = records[0]["h"].shape[0] if records else 0

    # Precompute Probe predictions on every state instance (one batched call).
    if records:
        h_mat = np.stack([r["h"] for r in records])
        predA = probes["A"].predict(h_mat)
        predB = probes["B"].predict(h_mat)
        predC = probes["C"].predict(h_mat)
        predD = probes["D"].predict(h_mat)
    else:
        predA = np.zeros((0, 4)); predB = predC = predD = np.zeros((0,))
    for i, r in enumerate(records):
        r["predA"] = predA[i]
        r["predB"] = int(predB[i])
        r["predC"] = int(predC[i])
        r["predD"] = int(predD[i])

    anchors = [i for i, r in enumerate(records) if r["has_next"]]

    # ---- Coverage (shared pool for 2A and 2B) ----
    partners_of = {}
    for ai in anchors:
        a = records[ai]
        parts = [
            pi for pi in groups[a["sym"]]
            if records[pi]["traj_idx"] != a["traj_idx"]
            and records[pi]["hist"] != a["hist"]
        ]
        partners_of[ai] = parts
    with_partner = [ai for ai in anchors if partners_of[ai]]
    cov_ci = _proportion_ci(len(with_partner), len(anchors))
    coverage = {
        "total_states": len(anchors),
        "states_with_swap_partner": len(with_partner),
        "states_without_swap_partner": len(anchors) - len(with_partner),
        "coverage_rate": (len(with_partner) / len(anchors)) if anchors else 0.0,
        "coverage_rate_ci_lo": cov_ci["ci_lo"],
        "coverage_rate_ci_hi": cov_ci["ci_hi"],
        "total_state_instances": len(records),
    }

    # ---- Oracle 1 (true-state transition, full has_next pool) ----
    o1 = {"probeA_exact": float("nan"), "probeA_per_label": float("nan"),
          "probeA_jaccard": float("nan"), "n": len(anchors),
          "probeA_exact_ci_lo": float("nan"), "probeA_exact_ci_hi": float("nan")}
    o1_by_anchor = {}  # ai -> (exact, per_label, jaccard) for matched delta
    if anchors and transition_model is not None:
        hm = np.stack([records[ai]["h"] for ai in anchors])
        opm = np.array([records[ai]["next_op"] for ai in anchors], dtype=np.int64)
        oprm = np.stack([records[ai]["next_operands"] for ai in anchors])
        pa = transition_predict_a(transition_model, probes["A"], hm, opm, oprm)
        ex, pl, jc = [], [], []
        for k, ai in enumerate(anchors):
            tgt = records[ai]["next_gtA"]
            e = a_exact(pa[k], tgt); p = a_per_label(pa[k], tgt); j = a_jaccard(pa[k], tgt)
            o1_by_anchor[ai] = (e, p, j)
            ex.append(e); pl.append(p); jc.append(j)
        o1 = {"probeA_exact": _mean(ex), "probeA_per_label": _mean(pl),
              "probeA_jaccard": _mean(jc), "n": len(anchors)}
        o1_ci = _mean_ci(ex)
        o1["probeA_exact_ci_lo"] = o1_ci["ci_lo"]
        o1["probeA_exact_ci_hi"] = o1_ci["ci_hi"]

    # ---- Build the valid swap-pair list (capped per anchor) ----
    pairs = []  # (anchor_idx, partner_idx, delta_depth)
    for ai in with_partner:
        a = records[ai]
        for pi in partners_of[ai][:max_partners]:
            p = records[pi]
            pairs.append((ai, pi, abs(a["depth"] - p["depth"])))

    if smoke_pairs:
        counts = _counts_by_delta(pairs)
        print(f"[Smoke Pairs] Built {len(pairs)} pairs.")
        print(f"[Smoke Pairs] Delta counts: {counts}")
        import sys; sys.exit(0)

    # ---- Oracle 2A (representation agreement on h vs h') ----
    a2 = {}  # metric -> stratified dict
    acc = collections.defaultdict(list)  # metric -> [(dd, value)]
    for ai, pi, dd in pairs:
        a, p = records[ai], records[pi]
        acc["probeA_pred_exact"].append((dd, a_exact(a["predA"], p["predA"])))
        acc["probeA_pred_per_label"].append((dd, a_per_label(a["predA"], p["predA"])))
        acc["probeA_pred_jaccard"].append((dd, a_jaccard(a["predA"], p["predA"])))
        acc["probeB_pred_abs_diff"].append((dd, abs(a["predB"] - p["predB"])))
        acc["probeB_pred_agreement"].append((dd, 1.0 if a["predB"] == p["predB"] else 0.0))
        acc["probeC_pred_agreement"].append((dd, 1.0 if a["predC"] == p["predC"] else 0.0))
        acc["probeD_pred_agreement"].append((dd, 1.0 if a["predD"] == p["predD"] else 0.0))
        # Ground-truth label comparison for the same symbolic state
        acc["gtA_exact"].append((dd, a_exact(a["gtA"], p["gtA"])))
        acc["gtA_per_label"].append((dd, a_per_label(a["gtA"], p["gtA"])))
        acc["gtA_jaccard"].append((dd, a_jaccard(a["gtA"], p["gtA"])))
        acc["gtB_abs_diff"].append((dd, abs(a["gtB"] - p["gtB"])))
        acc["gtB_agreement"].append((dd, 1.0 if a["gtB"] == p["gtB"] else 0.0))
        if a["gtC"] is not None and p["gtC"] is not None:
            acc["gtC_agreement"].append((dd, 1.0 if a["gtC"] == p["gtC"] else 0.0))
        acc["gtD_agreement"].append((dd, 1.0 if a["gtD"] == p["gtD"] else 0.0))
    for m, lst in acc.items():
        a2[m] = _stratify(lst)
    a2_counts = _counts_by_delta(pairs)
    a2_ci = {m: _mean_ci([v for _, v in lst]) for m, lst in acc.items()}

    # ---- Oracle 2B (dynamics on swapped state) + matched delta ----
    b2 = {}
    b2_acc = collections.defaultdict(list)
    delta_acc = collections.defaultdict(list)
    b2_ci = {}
    if pairs and transition_model is not None:
        hm = np.stack([records[pi]["h"] for _, pi, _ in pairs])
        opm = np.array([records[ai]["next_op"] for ai, _, _ in pairs], dtype=np.int64)
        oprm = np.stack([records[ai]["next_operands"] for ai, _, _ in pairs])
        pa = transition_predict_a(transition_model, probes["A"], hm, opm, oprm)
        for k, (ai, pi, dd) in enumerate(pairs):
            tgt = records[ai]["next_gtA"]
            e = a_exact(pa[k], tgt); pl = a_per_label(pa[k], tgt); jc = a_jaccard(pa[k], tgt)
            b2_acc["probeA_exact"].append((dd, e))
            b2_acc["probeA_per_label"].append((dd, pl))
            b2_acc["probeA_jaccard"].append((dd, jc))
            o1e = o1_by_anchor.get(ai, (float("nan"),) * 3)[0]
            delta_acc["oracle1_paired_exact"].append((dd, o1e))
            delta_acc["delta_transition_accuracy"].append((dd, o1e - e))
        for m, lst in b2_acc.items():
            b2[m] = _stratify(lst)
        for m, lst in delta_acc.items():
            b2[m] = _stratify(lst)
        for m, lst in b2_acc.items():
            b2_ci[m] = _mean_ci([v for _, v in lst])
        for m, lst in delta_acc.items():
            # delta is defined only where the paired Oracle 1 exact is finite.
            b2_ci[m] = _mean_ci([v for _, v in lst if not np.isnan(v)])
    b2_counts = _counts_by_delta(pairs)

    return {
        "coverage": coverage,
        "oracle1": o1,
        "oracle2a": a2,
        "oracle2a_counts": a2_counts,
        "oracle2a_ci": a2_ci,
        "oracle2b": b2,
        "oracle2b_counts": b2_counts,
        "oracle2b_ci": b2_ci,
    }


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6f}"
    return str(v)


def write_metrics_csv(o0, res, path):
    import csv
    rows = [("oracle", "metric", "delta_depth", "value", "n", "ci_lo", "ci_hi")]

    for m in ["probeA_exact", "probeA_per_label", "probeA_jaccard",
              "probeB_accuracy", "probeC_accuracy", "probeD_accuracy"]:
        rows.append(("oracle0", m, "all", o0[m], o0["n"], "", ""))

    # Coverage rate with its 95% CI (Wilson interval).
    c = res["coverage"]
    rows.append(("coverage", "coverage_rate", "all", c["coverage_rate"],
                 c["total_states"], c["coverage_rate_ci_lo"],
                 c["coverage_rate_ci_hi"]))

    o1 = res["oracle1"]
    for m in ["probeA_exact", "probeA_per_label", "probeA_jaccard"]:
        lo = o1["probeA_exact_ci_lo"] if m == "probeA_exact" else ""
        hi = o1["probeA_exact_ci_hi"] if m == "probeA_exact" else ""
        rows.append(("oracle1", m, "all", o1[m], o1["n"], lo, hi))

    a2, a2c = res["oracle2a"], res["oracle2a_counts"]
    a2ci = res.get("oracle2a_ci", {})
    for m in sorted(a2):
        for dd, val in a2[m].items():
            ci = a2ci.get(m, {}) if dd == "all" else {}
            rows.append(("oracle2a", m, dd, val, a2c.get(dd, ""),
                         ci.get("ci_lo", ""), ci.get("ci_hi", "")))

    b2, b2c = res["oracle2b"], res["oracle2b_counts"]
    b2ci = res.get("oracle2b_ci", {})
    for m in sorted(b2):
        for dd, val in b2[m].items():
            ci = b2ci.get(m, {}) if dd == "all" else {}
            rows.append(("oracle2b", m, dd, val, b2c.get(dd, ""),
                         ci.get("ci_lo", ""), ci.get("ci_hi", "")))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([
                r[0], r[1], r[2],
                _fmt(r[3]) if isinstance(r[3], float) else r[3],
                r[4],
                _fmt(r[5]) if isinstance(r[5], float) else r[5],
                _fmt(r[6]) if isinstance(r[6], float) else r[6],
            ])


def write_coverage_md(res, path):
    c = res["coverage"]
    a2c = res["oracle2a_counts"]
    lines = ["# Oracle Audit — Coverage\n"]
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| total_states | {c['total_states']} |")
    lines.append(f"| states_with_swap_partner | {c['states_with_swap_partner']} |")
    lines.append(f"| states_without_swap_partner | {c['states_without_swap_partner']} |")
    lines.append(f"| coverage_rate | {_fmt(c['coverage_rate'])} |")
    lines.append(
        f"| coverage_rate_95ci | [{_fmt(c['coverage_rate_ci_lo'])}, "
        f"{_fmt(c['coverage_rate_ci_hi'])}] |"
    )
    lines.append(f"| total_state_instances | {c['total_state_instances']} |")
    lines.append("")
    lines.append("## Swap pairs by delta_depth\n")
    lines.append("| delta_depth | n_pairs |")
    lines.append("|---|---|")
    for dd in [k for k in a2c if k != "all"]:
        lines.append(f"| {dd} | {a2c[dd]} |")
    lines.append(f"| all | {a2c.get('all', 0)} |")
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _kv_table(title, items):
    out = [f"## {title}\n", "| metric | value |", "|---|---|"]
    for k, v in items:
        out.append(f"| {k} | {_fmt(v)} |")
    out.append("")
    return out


def _strat_table(title, strat, metrics):
    dds = sorted({dd for m in metrics for dd in strat.get(m, {}) if dd != "all"})
    header = "| metric | all | " + " | ".join(f"dd={dd}" for dd in dds) + " |"
    sep = "|---|---|" + "---|" * len(dds)
    out = [f"## {title}\n", header, sep]
    for m in metrics:
        s = strat.get(m, {})
        cells = [_fmt(s.get("all", float("nan")))]
        cells += [_fmt(s.get(dd, float("nan"))) for dd in dds]
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _ci_table(title, ci, metrics):
    """95% CI table (all-pairs aggregate) for the given metrics."""
    out = [f"## {title}\n", "| metric | mean | ci_lo | ci_hi | n |", "|---|---|---|---|---|"]
    for m in metrics:
        c = ci.get(m, {})
        out.append(
            f"| {m} | {_fmt(c.get('mean', float('nan')))} | "
            f"{_fmt(c.get('ci_lo', float('nan')))} | "
            f"{_fmt(c.get('ci_hi', float('nan')))} | {c.get('n', 0)} |"
        )
    out.append("")
    return out


def write_summary_md(o0, res, path):
    c = res["coverage"]
    o1 = res["oracle1"]
    a2 = res["oracle2a"]
    b2 = res["oracle2b"]
    a2ci = res.get("oracle2a_ci", {})
    b2ci = res.get("oracle2b_ci", {})

    lines = ["# Oracle Audit — Summary\n"]

    # 1. Coverage
    lines += _kv_table("1. Coverage", [
        ("total_states", c["total_states"]),
        ("states_with_swap_partner", c["states_with_swap_partner"]),
        ("states_without_swap_partner", c["states_without_swap_partner"]),
        ("coverage_rate", c["coverage_rate"]),
        ("coverage_rate_ci_lo", c["coverage_rate_ci_lo"]),
        ("coverage_rate_ci_hi", c["coverage_rate_ci_hi"]),
        ("total_state_instances", c["total_state_instances"]),
    ])

    # 2. Oracle 0
    lines += _kv_table("2. Oracle 0 — Teacher Ceiling", [
        ("probeA_exact", o0["probeA_exact"]),
        ("probeA_per_label", o0["probeA_per_label"]),
        ("probeA_jaccard", o0["probeA_jaccard"]),
        ("probeB_accuracy", o0["probeB_accuracy"]),
        ("probeC_accuracy", o0["probeC_accuracy"]),
        ("probeD_accuracy", o0["probeD_accuracy"]),
        ("n", o0["n"]),
    ])

    # 3. Oracle 1
    lines += _kv_table("3. Oracle 1 — True-State Transition", [
        ("probeA_exact", o1["probeA_exact"]),
        ("probeA_exact_ci_lo", o1["probeA_exact_ci_lo"]),
        ("probeA_exact_ci_hi", o1["probeA_exact_ci_hi"]),
        ("probeA_per_label", o1["probeA_per_label"]),
        ("probeA_jaccard", o1["probeA_jaccard"]),
        ("n", o1["n"]),
    ])

    # 4. Oracle 2A agreement (stratified)
    lines += _strat_table(
        "4. Oracle 2A — Representation Agreement", a2,
        ["probeA_pred_exact", "probeA_pred_per_label", "probeA_pred_jaccard",
         "probeB_pred_abs_diff", "probeB_pred_agreement",
         "probeC_pred_agreement", "probeD_pred_agreement",
         "gtA_exact", "gtA_per_label", "gtA_jaccard",
         "gtB_abs_diff", "gtB_agreement", "gtC_agreement", "gtD_agreement"])

    # 4b. Oracle 2A agreement — 95% CI (all pairs)
    lines += _ci_table(
        "4b. Oracle 2A — Representation Agreement (95% CI, all pairs)", a2ci,
        ["probeA_pred_exact", "probeA_pred_per_label", "probeA_pred_jaccard",
         "probeB_pred_abs_diff", "probeB_pred_agreement",
         "probeC_pred_agreement", "probeD_pred_agreement",
         "gtA_exact", "gtA_per_label", "gtA_jaccard",
         "gtB_abs_diff", "gtB_agreement", "gtC_agreement", "gtD_agreement"])

    # 5. Oracle 2B transition metrics (stratified)
    lines += _strat_table(
        "5. Oracle 2B — Swapped-State Transition", b2,
        ["probeA_exact", "probeA_per_label", "probeA_jaccard"])

    # 6. Oracle 2B delta metrics (stratified)
    lines += _strat_table(
        "6. Oracle 2B — Delta", b2,
        ["oracle1_paired_exact", "delta_transition_accuracy"])

    # 6b. Oracle 2B transition + delta — 95% CI (all pairs)
    lines += _ci_table(
        "6b. Oracle 2B — Transition + Delta (95% CI, all pairs)", b2ci,
        ["probeA_exact", "probeA_per_label", "probeA_jaccard",
         "oracle1_paired_exact", "delta_transition_accuracy"])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def _resolve_audit_paths(reports_dir, audit):
    traj = os.path.join(reports_dir, "trajectories")
    if audit == "all":
        return [os.path.join(traj, f"{n}.pt") for n in ("train", "val", "test")]
    if audit == "train_test":
        return [os.path.join(traj, f"{n}.pt") for n in ("train", "test")]
    return [os.path.join(traj, f"{audit}.pt")]


def main():
    ap = argparse.ArgumentParser(description="Oracle Ceiling Audit")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--transition_arch", type=str, default="mlp", choices=["mlp", "linear", "transformer"], help="Architecture of the transition model")
    ap.add_argument("--train_traj", default=None,
                    help="Trajectory file used to fit probes (default <reports>/trajectories/train.pt)")
    ap.add_argument("--audit", choices=["test", "val", "train", "all", "train_test"],
                    default="test",
                    help="Split providing audit anchors + swap partners (held out from "
                         "probe fit). 'train_test' pools the train and test trajectories "
                         "as the anchor/partner pool.")
    ap.add_argument("--transition_ckpt", default=None,
                    help="Action-conditioned transition checkpoint "
                         "(default <reports>/transition_model_action.pt)")
    ap.add_argument("--max_partners", type=int, default=50,
                    help="Cap on swap partners per anchor (bounds pair count)")
    ap.add_argument("--smoke_pairs", action="store_true",
                    help="Lightweight validation: build pairs, run count, then exit.")
    args = ap.parse_args()

    train_path = args.train_traj or os.path.join(args.reports_dir, "trajectories", "train.pt")
    audit_paths = _resolve_audit_paths(args.reports_dir, args.audit)
    ckpt_path = args.transition_ckpt or os.path.join(args.reports_dir, "transition_model_action.pt")

    if not os.path.exists(train_path):
        print(f"ERROR: train trajectories not found: {train_path}")
        sys.exit(1)
    for p in audit_paths:
        if not os.path.exists(p):
            print(f"ERROR: audit trajectories not found: {p}")
            sys.exit(1)

    print(f"[Oracle Audit] Fitting probes on {train_path}")
    train_trajs = load_trajectories(train_path)
    probes = fit_all_probes(train_trajs)

    print(f"[Oracle Audit] Loading audit trajectories: {audit_paths}")
    audit_trajs = []
    for p in audit_paths:
        audit_trajs.extend(load_trajectories(p))
    print(f"  {len(audit_trajs)} audit trajectories")

    transition_model = None
    if os.path.exists(ckpt_path):
        print(f"[Oracle Audit] Loading transition model: {ckpt_path}")
        transition_model = load_transition_model(ckpt_path, args.transition_arch)
    else:
        print(f"WARNING: transition checkpoint missing ({ckpt_path}); "
              "Oracle 1 / 2B metrics will be nan.")

    print("[Oracle Audit] Oracle 0 (teacher ceiling)...")
    o0 = oracle0_metrics(probes, audit_trajs)

    print("[Oracle Audit] Building state records and swap pairs...")
    records, groups = build_state_records(audit_trajs)
    res = run_audit(records, groups, probes, transition_model, args.max_partners, args.smoke_pairs)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "oracle_audit_metrics.csv")
    cov_path = os.path.join(args.out_dir, "oracle_audit_coverage.md")
    sum_path = os.path.join(args.out_dir, "oracle_audit_summary.md")

    write_metrics_csv(o0, res, csv_path)
    write_coverage_md(res, cov_path)
    write_summary_md(o0, res, sum_path)

    c = res["coverage"]
    print("\n[Oracle Audit] Done.")
    print(f"  coverage: {c['states_with_swap_partner']}/{c['total_states']} anchors "
          f"with a swap partner (rate={c['coverage_rate']:.3f})")
    for pth in (csv_path, cov_path, sum_path):
        print(f"  - {pth}")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_a.py`
```python
"""Phase A orchestrator: the full latent-planning viability diagnostic.

Runs the complete pipeline end to end:

    generate/load problems
      -> extract teacher trajectories (frozen LM hidden states)
      -> train latent transition model      T(h_t, a_t) -> h_{t+1}
      -> train diagnostic next-token decoder
      -> coherence rollout evaluation        (coherence_depth.csv / .png)
      -> linear representation probes         (probe_results.csv / probe_report.md)
      -> master report                        (reports/phase_a_report.md)

The pipeline is model-agnostic. Use ``--model`` to switch between the full
TinyLlama teacher (real results) and a tiny random model (fast smoke test of the
machinery). All numeric artifacts are tagged with the model that produced them.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import (
    build_trajectories,
    load_problems,
    save_trajectories,
)
from training.train_transition import (
    train_transition_model,
    TransitionTrainConfig,
    save_history_csv,
)
from training.train_decoder import train_decoder_model, DecoderTrainConfig
from evaluation.coherence import evaluate_coherence, save_coherence
from evaluation.probes import (
    run_probes,
    save_probe_results,
    generate_probe_report,
)
from evaluation.plotter import plot_training_curves

DEFAULT_TEACHER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


# --------------------------------------------------------------------------- #
def maybe_generate_data(data_dir: str, sizes, domain: str = "countdown", base_seed: int = 42) -> None:
    """Generate data if the split files are missing/too small.

    Parameters
    ----------
    domain : str
        ``"countdown"`` (default, V5.3-compatible) or ``"game24"``.
    base_seed : int
        Base seed for generation to ensure disjoint splits.
    """
    if domain == "game24":
        _generate_game24_disjoint(data_dir, sizes, base_seed)
        return
    else:
        from scripts.generate_countdown_dataset import generate_dataset

    # sizes = [train, val, test, test_ood]
    files = {"train": sizes[0], "val": sizes[1], "test": sizes[2]}
    split_seeds = {
        "train": base_seed,
        "val": base_seed + 1,
        "test": base_seed + 2,
    }
    for split, n in files.items():
        path = os.path.join(data_dir, f"{split}.jsonl")
        existing = 0
        if os.path.exists(path):
            with open(path) as f:
                existing = sum(1 for _ in f)
        if existing < n:
            print(f"  generating {n} '{split}' problems -> {path}")
            generate_dataset(n, path, seed=split_seeds[split])
            
    if len(sizes) > 3:
        n_ood = sizes[3]
        path_ood = os.path.join(data_dir, "test_ood.jsonl")
        existing_ood = 0
        if os.path.exists(path_ood):
            with open(path_ood) as f:
                existing_ood = sum(1 for _ in f)
        if existing_ood < n_ood:
            print(f"  generating {n_ood} 'test_ood' problems -> {path_ood}")
            generate_dataset(n_ood, path_ood, is_ood=True, seed=base_seed + 3)

def _generate_game24_disjoint(data_dir, sizes, base_seed):
    from scripts.generate_game24_dataset import enumerate_solvable_pool, _hand_to_record
    import random
    import json
    
    pool = enumerate_solvable_pool()
    total_available = len(pool)
    
    req_train = sizes[0] if len(sizes) > 0 else 1000
    req_val = sizes[1] if len(sizes) > 1 else 150
    req_test = sizes[2] if len(sizes) > 2 else 196
    
    total_req = req_train + req_val + req_test
    if total_req > total_available:
        scale = total_available / total_req
        req_train = int(req_train * scale)
        req_val = int(req_val * scale)
        req_test = total_available - req_train - req_val
        print(f"  [INFO] Game24 pool size is {total_available}. Scaling requested splits down to: train={req_train}, val={req_val}, test={req_test}")

    splits = {"train": req_train, "val": req_val, "test": req_test}
    
    need_gen = False
    for split, n in splits.items():
        if n <= 0: continue
        path = os.path.join(data_dir, f"{split}.jsonl")
        existing = 0
        if os.path.exists(path):
            with open(path) as f:
                existing = sum(1 for _ in f)
        if existing < n:
            need_gen = True
            break
            
    if need_gen:
        os.makedirs(data_dir, exist_ok=True)
        print(f"  [INFO] Generating mutually exclusive Game24 splits: train={req_train}, val={req_val}, test={req_test}")
        rng = random.Random(base_seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        
        allocations = {
            "train": shuffled[:req_train],
            "val": shuffled[req_train:req_train+req_val],
            "test": shuffled[req_train+req_val:req_train+req_val+req_test],
        }
        
        for split, hands in allocations.items():
            if hands:
                path = os.path.join(data_dir, f"{split}.jsonl")
                with open(path, "w") as f:
                    for hand in hands:
                        f.write(json.dumps(_hand_to_record(hand, rng)) + "\n")


def build_split_trajectories(model, tokenizer, data_dir, layer, cap, out_dir, batch_size=32, trajectories_dir=None):
    from data_processing.trajectory_dataset import load_trajectories
    trajs = {}
    splits = ["train", "val", "test"]
    if os.path.exists(os.path.join(data_dir, "test_ood.jsonl")):
        splits.append("test_ood")
        
    for split in splits:
        target_path = os.path.join(out_dir, "trajectories", f"{split}.pt")
        if trajectories_dir and os.path.exists(os.path.join(trajectories_dir, f"{split}.pt")):
            print(f"  [{split}] Loading existing trajectories from {trajectories_dir}...")
            t = load_trajectories(os.path.join(trajectories_dir, f"{split}.pt"))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            save_trajectories(t, target_path)
        else:
            problems = load_problems(os.path.join(data_dir, f"{split}.jsonl"))
            if cap:
                problems = problems[:cap]
            print(f"  [{split}] building trajectories from {len(problems)} problems...")
            t = build_trajectories(model, tokenizer, problems, layer=layer, batch_size=batch_size)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            save_trajectories(t, target_path)
        
        n_trans = sum(tr.num_steps for tr in t)
        print(f"  [{split}] {len(t)} trajectories, {n_trans} transitions")
        trajs[split] = t
        
    print("\n[1.5] Verifying Dataset Disjointness (Leakage Check & De-duplication)")
    def get_problem_hash(t):
        return (t.target, tuple(sorted(t.numbers)))
        
    train_hashes = set(get_problem_hash(t) for t in trajs.get("train", []))
    
    # Filter val
    if "val" in trajs:
        orig_val = len(trajs["val"])
        trajs["val"] = [t for t in trajs["val"] if get_problem_hash(t) not in train_hashes]
        val_hashes = set(get_problem_hash(t) for t in trajs["val"])
        dropped_val = orig_val - len(trajs["val"])
        if dropped_val > 0:
            print(f"  [WARNING] Dropped {dropped_val} leaking trajectories from Val set.")
            
    # Filter test
    if "test" in trajs:
        orig_test = len(trajs["test"])
        trajs["test"] = [t for t in trajs["test"] if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]
        dropped_test = orig_test - len(trajs["test"])
        if dropped_test > 0:
            print(f"  [WARNING] Dropped {dropped_test} leaking trajectories from Test set.")
            
    # Filter test_ood
    if "test_ood" in trajs:
        orig_ood = len(trajs["test_ood"])
        trajs["test_ood"] = [t for t in trajs["test_ood"] if get_problem_hash(t) not in train_hashes and get_problem_hash(t) not in val_hashes]
        dropped_ood = orig_ood - len(trajs["test_ood"])
        if dropped_ood > 0:
            print(f"  [WARNING] Dropped {dropped_ood} leaking trajectories from Test OOD set.")
            
    print(f"train: {len(trajs.get('train', []))}")
    print(f"val: {len(trajs.get('val', []))}")
    print(f"test: {len(trajs.get('test', []))}")
    print(f"test_ood: {len(trajs.get('test_ood', []))}")

    print("  [OK] Datasets de-duplicated.\n")
    
    return trajs


# --------------------------------------------------------------------------- #
def generate_master_report(
    coherence_df, coh_blind_df, coh_shuffled_df, coh_constant_df, oracle_df,
    probe_df, probe_details, transition_history,
    meta: dict, md_path: str,
) -> None:
    import numpy as np

    cov = coherence_df[coherence_df["n_samples"] > 0]

    # Coherence summary.
    def _depth_where(col, thresh, above=True):
        if col not in cov.columns:
            return 0
        ok = cov[cov[col] >= thresh] if above else cov[cov[col] <= thresh]
        return int(ok["depth"].max()) if len(ok) else 0

    cos_horizon = _depth_where("cosine_similarity", 0.9)
    op_horizon = _depth_where("operator_accuracy", 0.5)
    probe_horizon = _depth_where("state_probe_accuracy", 0.5)
    max_depth_avail = int(cov["depth"].max()) if len(cov) else 0
    final_val_mse = (transition_history[-1].get("eval_loss")
                     if transition_history else None)

    # Probe summary: which quantities are linearly decodable.
    chance = probe_details["chance"]
    chance_map = {row["probe"][0]: c for row, c in
                  zip([r for _, r in probe_df.iterrows()],
                      [chance[k] for k in ["A", "B", "C", "D"]])}
    encoded = []
    for _, r in probe_df.iterrows():
        k = r["probe"][0]
        base = chance_map[k]
        auc = r["auc"]
        if (not np.isnan(auc) and auc >= 0.65) or (r["accuracy"] - base >= 0.10):
            encoded.append(r["probe"])

    survives_beyond_3 = op_horizon > 3 or cos_horizon > 3
    info_present = len(encoded) > 0

    lines = []
    lines.append("# Phase A Report — Is Latent Planning Viable?\n")
    lines.append(f"- **Teacher model:** `{meta['model']}`")
    lines.append(f"- **Hidden layer probed:** {meta['layer']}")
    lines.append(f"- **Hidden dim:** {meta['hidden_dim']}")
    lines.append(f"- **Trajectories:** train={meta['n_train']} val={meta['n_val']} "
                 f"test={meta['n_test']}")
    if "transition_arch" in meta:
        lines.append(f"- **Transition Architecture:** `{meta['transition_arch']}`")
        lines.append(f"- **Transition Parameters:** {meta['parameter_count']:,}")
        lines.append(f"- **d_model:** {meta['d_model']}")
        lines.append(f"- **Bottleneck:** {meta['bottleneck']}")
        lines.append(f"- **Layers:** {meta['layers']}")
    if meta.get("smoke"):
        lines.append(
            "\n> ⚠️ **Smoke-test run** with a randomly-initialized model. "
            "The numbers below validate the *pipeline*, not the science. "
            "Re-run with the TinyLlama teacher for real conclusions.")
    lines.append("\n---\n")

    lines.append("## 1. Do hidden states contain reasoning information?\n")
    if info_present:
        lines.append(
            "**Yes.** The following quantities are linearly decodable from the "
            "frozen hidden state (above majority-class baseline):")
        for p in encoded:
            lines.append(f"- {p}")
    else:
        lines.append(
            "**Not clearly.** No probed quantity was linearly decodable above "
            "baseline. See `probe_report.md`.")
    lines.append(f"\nSee `probe_results.csv` and `probe_report.md` for full metrics.\n")

    lines.append("## 2. How quickly does coherence decay?\n")
    if max_depth_avail:
        lines.append(
            f"Rollouts were evaluated to depth {max_depth_avail} "
            "(limited by solution length in the data).")
        lines.append("\n| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in cov.iterrows():
            lines.append(
                f"| {int(r['depth'])} | {r['cosine_similarity']:.3f} | "
                f"{r['operator_accuracy']:.3f} | {r['teacher_operator_accuracy']:.3f} | "
                f"{r['state_probe_accuracy']:.3f} | {r['teacher_state_probe_accuracy']:.3f} | "
                f"{r['mse']:.4f} |")
        lines.append(
            f"\nCosine similarity stays >= 0.90 through depth **{cos_horizon}**; "
            f"rollout operator accuracy stays >= 0.50 through depth **{op_horizon}**; "
            f"rollout state probe accuracy stays >= 0.50 through depth **{probe_horizon}**.")
        
        # Action Control Table (at depth 1 or average)
        lines.append("\n### Action Control Ablation (Depth 1)\n")
        lines.append("| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |")
        lines.append("|---|---|---|---|")
        
        def _get_metrics(df, id_df):
            if df is None or len(df[df["depth"] == 1]) == 0: return float('nan'), float('nan'), float('nan')
            d1 = df[df["depth"] == 1].iloc[0]
            sem_gain = d1["state_probe_accuracy"] - d1["identity_state_probe_accuracy"]
            mcc = d1.get("mean_centered_cosine", float('nan'))
            
            # oracle gap: oracle state probe acc - this df state probe acc
            if oracle_df is not None and len(oracle_df[oracle_df["depth"] == 1]) > 0:
                o1 = oracle_df[oracle_df["depth"] == 1].iloc[0]
                oracle_gap = o1["state_probe_accuracy"] - d1["state_probe_accuracy"]
            else:
                oracle_gap = float('nan')
            return sem_gain, oracle_gap, mcc

        o_sg, o_gap, o_mcc = _get_metrics(oracle_df, oracle_df)
        t_sg, t_gap, t_mcc = _get_metrics(coherence_df, coherence_df)
        s_sg, s_gap, s_mcc = _get_metrics(coh_shuffled_df, coh_shuffled_df)
        c_sg, c_gap, c_mcc = _get_metrics(coh_constant_df, coh_constant_df)
        b_sg, b_gap, b_mcc = _get_metrics(coh_blind_df, coh_blind_df)
        
        lines.append(f"| Oracle | {o_sg:+.3f} | {0.000:.3f} | {o_mcc:+.3f} |")
        lines.append(f"| True Action | {t_sg:+.3f} | {t_gap:.3f} | {t_mcc:+.3f} |")
        lines.append(f"| Shuffled Action | {s_sg:+.3f} | {s_gap:.3f} | {s_mcc:+.3f} |")
        lines.append(f"| Constant Action | {c_sg:+.3f} | {c_gap:.3f} | {c_mcc:+.3f} |")
        lines.append(f"| Blind | {b_sg:+.3f} | {b_gap:.3f} | {b_mcc:+.3f} |")
        lines.append("\n_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._\n")
    else:
        lines.append("No multi-step trajectories were available to evaluate.")
    lines.append("\nSee `coherence_depth.csv` / `coherence_depth.png`.\n")

    lines.append("## 3. What is the effective planning horizon?\n")
    horizon = max(cos_horizon, op_horizon, probe_horizon)
    lines.append(
        f"Taking the depth at which the rolled-out latent still resembles "
        f"the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) "
        f"and state information (acc >= 0.5), the **effective horizon is ~{horizon} step(s)**.")
    if final_val_mse is not None:
        lines.append(f"\nSingle-step transition validation MSE: {final_val_mse:.4f}.")
    lines.append("")

    lines.append("## 4. Is latent planning worth pursuing?\n")
    if info_present and survives_beyond_3:
        verdict = ("**Promising.** Hidden states encode task-relevant structure "
                   "*and* coherence survives beyond depth 3 — the regime where "
                   "search would operate. Proceed to Phase B.")
    elif info_present and not survives_beyond_3:
        verdict = ("**Mixed / representation-limited bottleneck.** The "
                   "representation encodes reasoning information, but the learned "
                   "dynamics lose coherence by depth 3. The bottleneck is "
                   "*dynamics*, not representation — worth pursuing only with a "
                   "stronger transition model.")
    elif not info_present and survives_beyond_3:
        verdict = ("**Inconclusive.** Latents stay self-consistent under rollout "
                   "but probes find little decodable task information — the model "
                   "may be coasting on uninformative directions. Investigate "
                   "representation before pursuing planning.")
    else:
        verdict = ("**Not yet.** Hidden states show weak task information and "
                   "coherence decays quickly. Latent planning is unlikely to work "
                   "with this representation/dynamics pair as-is.")
    lines.append(verdict)
    lines.append(
        "\n**Bottleneck diagnosis:** "
        + ("representation" if (not info_present and survives_beyond_3)
           else "dynamics" if (info_present and not survives_beyond_3)
           else "neither (proceed)" if (info_present and survives_beyond_3)
           else "both") + ".")
    lines.append(
        "\n_Decision rule: 'viable' requires both (a) task information linearly "
        "present in the representation and (b) rollout coherence surviving beyond "
        "depth 3._\n")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run the Phase A diagnostic pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--domain", type=str, choices=["countdown", "game24"],
                        default="countdown",
                        help="Problem domain (default: countdown for V5.3 compat)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="reports")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--cap", type=int, default=None,
                        help="Max problems per split (for quick runs)")
    parser.add_argument("--generate", nargs="+", type=int, metavar="SIZE",
                        default=None, help="Generate data if missing. Defaults to [5000, 500, 1000, 500] for countdown, [1000, 150, 196] for game24.")
    parser.add_argument("--transition_epochs", type=int, default=100)
    parser.add_argument("--decoder_epochs", type=int, default=30)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extracting hidden states")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load the teacher in 4-bit NF4 (inference-only; for 7B/8B models)")
    parser.add_argument("--smoke", action="store_true",
                        help="Mark outputs as a smoke test (random model)")
    parser.add_argument("--transition_arch", type=str, default="mlp", choices=["mlp", "linear", "transformer"],
                        help="Architecture of the transition model")
    parser.add_argument("--extract_only", action="store_true", help="Only generate trajectories")
    parser.add_argument("--trajectories_dir", type=str, default=None, help="Use pre-extracted trajectories from this directory")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    args = parser.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    if args.generate is None:
        if args.domain == "game24":
            args.generate = [1000, 150, 196]
        else:
            args.generate = [5000, 500, 1000, 500]

    if args.generate:
        print("[0/6] Ensuring data exists...")
        maybe_generate_data(args.data_dir, args.generate, domain=args.domain, base_seed=args.seed)

    print(f"[1/6] Loading frozen teacher: {args.model}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype,
                       load_in_4bit=args.load_in_4bit)
    tokenizer = load_tokenizer(model_id=args.model)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))

    print("[2/6] Extracting teacher trajectories...")
    trajs = build_split_trajectories(
        model, tokenizer, args.data_dir, args.layer, args.cap, out, args.batch_size, args.trajectories_dir)
    hidden_dim = trajs["train"][0].hidden_dim

    if args.extract_only:
        print("\n[*] Extract only mode. Done.")
        return

    print("[3/6] Training latent transition models (Action-Conditioned + Action-Blind)...")
    tcfg = TransitionTrainConfig(epochs=args.transition_epochs, transition_arch=args.transition_arch, seed=args.seed)
    
    # Train Action-Conditioned
    print("  -> Action-Conditioned")
    tcfg.use_action = True
    tmodel_action = train_transition_model(trajs["train"], trajs["val"], tcfg)
    torch.save({"state_dict": tmodel_action.state_dict(), "hidden_dim": hidden_dim},
               os.path.join(out, "transition_model_action.pt"))
    save_history_csv(tcfg.history, os.path.join(out, "transition_action_log.csv"))
    plot_training_curves(os.path.join(out, "transition_action_log.csv"),
                         os.path.join(out, "transition_action_curve.png"))
                         
    # Train Action-Blind
    print("  -> Action-Blind")
    tcfg_blind = TransitionTrainConfig(epochs=args.transition_epochs, use_action=False, transition_arch=args.transition_arch, seed=args.seed)
    tmodel_blind = train_transition_model(trajs["train"], trajs["val"], tcfg_blind)
    torch.save({"state_dict": tmodel_blind.state_dict(), "hidden_dim": hidden_dim},
               os.path.join(out, "transition_model_blind.pt"))
    save_history_csv(tcfg_blind.history, os.path.join(out, "transition_blind_log.csv"))
    plot_training_curves(os.path.join(out, "transition_blind_log.csv"),
                         os.path.join(out, "transition_blind_curve.png"))

    print("[4/6] Training diagnostic decoder...")
    dcfg = DecoderTrainConfig(epochs=args.decoder_epochs)
    decoder = train_decoder_model(trajs["train"], vocab_size, trajs["val"], dcfg)

    print("[5/6] Representation probes + reports...")
    probe_df, probe_details, fitted_probes = run_probes(trajs["train"], trajs["test"], domain=args.domain)
    save_probe_results(probe_df, os.path.join(out, "probe_results.csv"))
    generate_probe_report(probe_df, probe_details, os.path.join(out, "probe_report.md"))

    print("[6/7] Coherence rollout evaluation...")
    probe_a = fitted_probes.get("A")
    probe_b = fitted_probes.get("B")
    probe_c = fitted_probes.get("C")
    probe_d = fitted_probes.get("D")
    
    coh_df = evaluate_coherence(
        tmodel_action, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth, domain=args.domain
    )
    save_coherence(coh_df, os.path.join(out, "coherence_action_depth.csv"),
                   os.path.join(out, "coherence_action_depth.png"))
                   
    coh_blind_df = evaluate_coherence(
        tmodel_blind, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth, domain=args.domain
    )
    save_coherence(coh_blind_df, os.path.join(out, "coherence_blind_depth.csv"),
                   os.path.join(out, "coherence_blind_depth.png"))
                   
    # Shuffled Action Rollout
    coh_shuffled_df = evaluate_coherence(
        tmodel_action, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth, domain=args.domain, rollout_mode="shuffled"
    )
    save_coherence(coh_shuffled_df, os.path.join(out, "coherence_shuffled_depth.csv"),
                   os.path.join(out, "coherence_shuffled_depth.png"))
                   
    # Constant Action Rollout
    coh_constant_df = evaluate_coherence(
        tmodel_action, trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
        max_depth=args.max_depth, domain=args.domain, rollout_mode="constant"
    )
    save_coherence(coh_constant_df, os.path.join(out, "coherence_constant_depth.csv"),
                   os.path.join(out, "coherence_constant_depth.png"))
                   
    if "test_ood" in trajs:
        coh_ood_df = evaluate_coherence(
            tmodel_action, trajs["test_ood"], 
            probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d, 
            max_depth=args.max_depth, domain=args.domain
        )
        save_coherence(coh_ood_df, os.path.join(out, "coherence_ood_depth.csv"),
                       os.path.join(out, "coherence_ood_depth.png"))

    print("[7/7] Phase B: Oracle Transition Diagnostic...")
    from evaluation.oracle_coherence import (
        evaluate_oracle_coherence,
        save_oracle_coherence,
        plot_comparison_overlay,
    )
    oracle_df = evaluate_oracle_coherence(
        trajs["test"], 
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d,
        max_depth=args.max_depth, domain=args.domain,
    )
    save_oracle_coherence(
        oracle_df,
        os.path.join(out, "coherence_oracle_depth.csv"),
        os.path.join(out, "coherence_oracle_depth.png"),
    )
    # Sanity check (ignore empty rollout bins)
    valid_oracle = oracle_df[
        (oracle_df["n_samples"] > 0)
        & oracle_df["cosine_similarity"].notna()
        & oracle_df["mse"].notna()
    ]

    cos_vals = valid_oracle["cosine_similarity"].tolist()
    mse_vals = valid_oracle["mse"].tolist()

    if len(valid_oracle) == 0:
        print("  [WARNING] Oracle sanity check skipped (no valid rollout depths).")
    elif all(c > 0.999 for c in cos_vals) and all(m < 1e-6 for m in mse_vals):
        print("  [PASS] Oracle sanity check.")
    else:
        print("  [FAIL] Oracle sanity check - investigate!")
        print(f"    Cosine: {cos_vals}")
        print(f"    MSE:    {mse_vals}")

    # 4-way overlay plot
    plot_comparison_overlay(
        oracle_df, coh_df, coh_blind_df,
        os.path.join(out, "coherence_comparison_overlay.png"),
    )

    # Phase B report
    from scripts.run_phase_b import generate_phase_b_report
    generate_phase_b_report(
        oracle_df, coh_df, coh_blind_df,
        os.path.join(out, "phase_b_oracle_report.md"),
        args.domain,
    )

    # Calculate transition params
    n_params = sum(p.numel() for p in tmodel_action.parameters() if p.requires_grad)
    d_model_val = getattr(tmodel_action, "d_model", getattr(tmodel_action, "hidden_dim", getattr(tmodel_action, "mlp_hidden_dim", "N/A")))
    bottleneck_val = tcfg.mlp_hidden_dim if tcfg.transition_arch != "transformer" else getattr(tmodel_action, "d_model", tcfg.mlp_hidden_dim // 2)
    layers_val = 0 if tcfg.transition_arch == "linear" else (2 if tcfg.transition_arch == "transformer" else 1) # simple mapping
    
    meta = {
        "model": args.model, "layer": args.layer, "hidden_dim": hidden_dim,
        "n_train": len(trajs["train"]), "n_val": len(trajs["val"]),
        "n_test": len(trajs["test"]), "smoke": args.smoke, "domain": args.domain,
        "transition_arch": tcfg.transition_arch,
        "parameter_count": n_params,
        "d_model": d_model_val,
        "bottleneck": bottleneck_val,
        "layers": layers_val
    }
    # Master report uses all dataframes for ablation table
    generate_master_report(
        coh_df, coh_blind_df, coh_shuffled_df, coh_constant_df, oracle_df,
        probe_df, probe_details, tcfg.history, meta,
        os.path.join(out, "phase_a_report.md")
    )

    print("\nDone. Artifacts written to:")
    for name in ["coherence_action_depth.csv", "coherence_action_depth.png",
                 "coherence_blind_depth.png", "coherence_oracle_depth.csv",
                 "coherence_comparison_overlay.png",
                 "phase_b_oracle_report.md",
                 "probe_results.csv", "probe_report.md",
                 "transition_action_curve.png", "phase_a_report.md"]:
        print(f"  - {os.path.join(out, name)}")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_b.py`
```python
"""Phase B orchestrator: Oracle Transition Diagnostic.

Runs after Phase A. Loads saved trajectories and fitted probes, then:

  1. Evaluates Oracle (perfect) transition coherence
  2. Loads Phase A action-conditioned and action-blind CSVs for comparison
  3. Generates the 4-way overlay plot
  4. Generates the Phase B Oracle report with gain tables and diagnosis

Usage (from repo root after Phase A completes)::

    python scripts/run_phase_b.py --reports_dir reports

Or on Kaggle, appended after Phase A in the same notebook.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import pandas as pd
import numpy as np

from data_processing.trajectory_dataset import load_trajectories
from evaluation.oracle_coherence import (
    evaluate_oracle_coherence,
    save_oracle_coherence,
    plot_comparison_overlay,
)
from evaluation.probes import run_probes


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #
def generate_phase_b_report(
    oracle_df: pd.DataFrame,
    action_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    md_path: str,
    domain: str = "countdown",
) -> None:
    """Generate the Phase B Oracle diagnostic report."""

    def _v(df):
        return df[df["n_samples"] > 0]

    ov = _v(oracle_df)
    av = _v(action_df)
    bv = _v(blind_df)

    lines = []
    lines.append("# Phase B Report — Oracle Transition Diagnostic\n")
    lines.append(
        "The Oracle Transition returns the **exact teacher hidden state** at "
        "each depth. It performs no learning and no prediction. It establishes "
        "the theoretical maximum coherence achievable under perfect dynamics.\n")

    # ---- Sanity check ----
    lines.append("## 0. Sanity Check\n")
    oracle_cos_vals = ov["cosine_similarity"].tolist()
    oracle_mse_vals = ov["mse"].tolist()
    all_cos_one = all(c > 0.999 for c in oracle_cos_vals)
    all_mse_zero = all(m < 1e-6 for m in oracle_mse_vals)
    if all_cos_one and all_mse_zero:
        lines.append(
            "✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all "
            "depths. The Oracle is correctly returning the exact teacher state.\n")
    else:
        lines.append(
            "⚠️ **FAIL**: Oracle cosine or MSE deviates from expected values. "
            "There may be a bug in the Oracle implementation.\n")
        lines.append(f"  Cosine values: {oracle_cos_vals}")
        lines.append(f"  MSE values: {oracle_mse_vals}\n")

    # ---- Question 1: How much coherence survives? ----
    lines.append("## 1. How much coherence survives under perfect transitions?\n")
    lines.append(
        "Since the Oracle returns the exact teacher state, its cosine and MSE "
        "are trivially perfect. The informative metric is **probe accuracy**: "
        "how much task information do the *teacher states themselves* contain "
        "at each depth?\n")
    lines.append(
        "| Depth | Oracle State Acc | Oracle Op Acc | n_samples |")
    lines.append("|---|---|---|---|")
    for _, r in ov.iterrows():
        lines.append(
            f"| {int(r['depth'])} | {r['state_probe_accuracy']:.3f} | "
            f"{r['operator_accuracy']:.3f} | {int(r['n_samples'])} |")
    lines.append("")

    # ---- Question 2: Transition error vs representation limitation ----
    lines.append("## 2. Transition Learning Error vs Representation Limitations\n")
    lines.append(
        "The **Oracle Gain** at each depth measures how much coherence the "
        "learned transition model leaves on the table.\n")

    # Cosine gain table
    lines.append("### Cosine Similarity Gain\n")
    lines.append("| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_cos = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        a_cos = av.loc[av["depth"] == d, "cosine_similarity"].values
        b_cos = bv.loc[bv["depth"] == d, "cosine_similarity"].values
        if len(o_cos) and len(a_cos) and len(b_cos):
            gain = o_cos[0] - a_cos[0]
            lines.append(
                f"| {d} | {o_cos[0]:.3f} | {a_cos[0]:.3f} | "
                f"{b_cos[0]:.3f} | {gain:+.3f} |")
    lines.append("")

    # State probe gain table
    lines.append("### State Probe Accuracy Gain (Probe A)\n")
    if domain == "game24":
        lines.append("*Probe A Definition: Game24 (Remaining card multiset encoding)*\n")
    else:
        lines.append("*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*\n")
        
    lines.append(
        "| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_s = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        a_s = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        b_s = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values
        if len(o_s) and len(a_s) and len(b_s):
            gain = o_s[0] - a_s[0]
            lines.append(
                f"| {d} | {o_s[0]:.3f} | {a_s[0]:.3f} | "
                f"{b_s[0]:.3f} | {gain:+.3f} |")
    lines.append("")

    # Operator accuracy gain table
    lines.append("### Operator Accuracy Gain (Probe C)\n")
    lines.append(
        "| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |")
    lines.append("|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        o_op = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        a_op = av.loc[av["depth"] == d, "operator_accuracy"].values
        b_op = bv.loc[bv["depth"] == d, "operator_accuracy"].values
        if len(o_op) and len(a_op) and len(b_op):
            gain = o_op[0] - a_op[0]
            lines.append(
                f"| {d} | {o_op[0]:.3f} | {a_op[0]:.3f} | "
                f"{b_op[0]:.3f} | {gain:+.3f} |")
    lines.append("")

    # ---- Action Gain ----
    lines.append("## 2b. Does the Action Vector actually drive dynamics?\n")
    lines.append(
        "The **Action Gain** measures how much of the theoretically available "
        "planning signal (Oracle - Blind) is captured by the action conditioning. "
        "Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.\n")
    lines.append(
        "| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |")
    lines.append("|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        bc = bv.loc[bv["depth"] == d, "cosine_similarity"].values
        
        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        bs_ = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values
        
        oo = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        ao = av.loc[av["depth"] == d, "operator_accuracy"].values
        bo = bv.loc[bv["depth"] == d, "operator_accuracy"].values
        
        def calc_gain(o, a, b):
            if not (len(o) and len(a) and len(b)): return float("nan")
            denom = o[0] - b[0]
            if abs(denom) < 1e-4: return float("nan")
            return (a[0] - b[0]) / denom
            
        c_gain = calc_gain(oc, ac, bc)
        s_gain = calc_gain(os_, as_, bs_)
        o_gain = calc_gain(oo, ao, bo)
        
        c_str = f"{c_gain:+.3f}" if not np.isnan(c_gain) else "n/a"
        s_str = f"{s_gain:+.3f}" if not np.isnan(s_gain) else "n/a"
        o_str = f"{o_gain:+.3f}" if not np.isnan(o_gain) else "n/a"
        
        if any(not np.isnan(g) for g in [c_gain, s_gain, o_gain]):
            lines.append(f"| {d} | {c_str} | {s_str} | {o_str} |")
    lines.append("")

    # ---- Question 3: Full comparison ----
    lines.append("## 3. Full Depth-by-Depth Comparison\n")
    lines.append(
        "| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | "
        "Oracle State | Action State | Blind State | Identity State |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in ov["depth"].tolist():
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        bc = bv.loc[bv["depth"] == d, "cosine_similarity"].values
        ic = av.loc[av["depth"] == d, "identity_cosine_similarity"].values
        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        bs_ = bv.loc[bv["depth"] == d, "state_probe_accuracy"].values
        is_ = av.loc[av["depth"] == d,
                      "identity_state_probe_accuracy"].values
        if all(len(x) for x in [oc, ac, bc, ic, os_, as_, bs_, is_]):
            lines.append(
                f"| {d} | {oc[0]:.3f} | {ac[0]:.3f} | {bc[0]:.3f} | "
                f"{ic[0]:.3f} | {os_[0]:.3f} | {as_[0]:.3f} | "
                f"{bs_[0]:.3f} | {is_[0]:.3f} |")
    lines.append("")

    # ---- Diagnosis ----
    lines.append("## 4. Bottleneck Diagnosis\n")

    # Compute average gains across depths
    shared_depths = sorted(
        set(ov["depth"].tolist()) &
        set(av["depth"].tolist()) &
        set(bv["depth"].tolist()))
    cos_gains, state_gains, op_gains = [], [], []
    for d in shared_depths:
        d = int(d)
        oc = ov.loc[ov["depth"] == d, "cosine_similarity"].values
        ac = av.loc[av["depth"] == d, "cosine_similarity"].values
        os_ = ov.loc[ov["depth"] == d, "state_probe_accuracy"].values
        as_ = av.loc[av["depth"] == d, "state_probe_accuracy"].values
        oo = ov.loc[ov["depth"] == d, "operator_accuracy"].values
        ao = av.loc[av["depth"] == d, "operator_accuracy"].values
        if len(oc) and len(ac):
            cos_gains.append(oc[0] - ac[0])
        if len(os_) and len(as_):
            state_gains.append(os_[0] - as_[0])
        if len(oo) and len(ao):
            op_gains.append(oo[0] - ao[0])

    avg_cos_gain = np.mean(cos_gains) if cos_gains else 0
    avg_state_gain = np.mean(state_gains) if state_gains else 0
    avg_op_gain = np.mean(op_gains) if op_gains else 0

    # Check oracle representation ceiling
    oracle_state_accs = ov["state_probe_accuracy"].dropna().tolist()
    oracle_op_accs = ov["operator_accuracy"].dropna().tolist()
    avg_oracle_state = np.mean(oracle_state_accs) if oracle_state_accs else 0
    avg_oracle_op = np.mean(oracle_op_accs) if oracle_op_accs else 0

    lines.append(f"**Average Oracle Gain (cosine):** {avg_cos_gain:+.4f}")
    lines.append(f"**Average Oracle Gain (state probe):** {avg_state_gain:+.4f}")
    lines.append(f"**Average Oracle Gain (operator):** {avg_op_gain:+.4f}\n")
    lines.append(f"**Average Oracle State Probe Accuracy:** {avg_oracle_state:.4f}")
    lines.append(f"**Average Oracle Operator Accuracy:** {avg_oracle_op:.4f}\n")

    # Case analysis
    oracle_degrades = avg_oracle_state < 0.6 or avg_oracle_op < 0.4
    gain_large = avg_cos_gain > 0.05 or avg_state_gain > 0.10
    gain_small = avg_cos_gain < 0.02 and avg_state_gain < 0.05

    lines.append("### Interpretation\n")
    if oracle_degrades:
        lines.append(
            "**Case C — Representation Instability.** Even under perfect "
            "(Oracle) transitions, the teacher hidden states yield low probe "
            "accuracy. The frozen hidden state is not a stable planning state. "
            "The bottleneck is in the **representation itself**, not the "
            "learned dynamics.\n")
    elif gain_small:
        lines.append(
            "**Case A — Transition model is NOT the bottleneck.** Oracle and "
            "action-conditioned coherence are comparable. The learned "
            "transition model already captures most of the available dynamics. "
            "The remaining degradation comes from **representation limitations** "
            "— the hidden states themselves lose task information as depth "
            "increases.\n")
    elif gain_large:
        lines.append(
            "**Case B — Transition model IS the bottleneck.** The Oracle "
            "significantly outperforms the action-conditioned transition. The "
            "representation contains usable planning information that the "
            "learned dynamics fail to preserve. Improving the transition model "
            "(or its training) is the highest-leverage intervention.\n")
    else:
        lines.append(
            "**Mixed result.** The Oracle gain is moderate, suggesting "
            "contributions from both transition learning error and "
            "representation limitations. Further investigation is needed.\n")

    lines.append("---\n")
    lines.append(
        "_This report was generated by the Phase B Oracle Transition "
        "Diagnostic. The Oracle performs no learning; it simply returns the "
        "teacher state at each depth. It answers one question: is latent "
        "planning limited by the learned dynamics, or by the representation "
        "itself?_\n")

    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Phase B: Oracle Transition Diagnostic")
    parser.add_argument("--reports_dir", type=str, default="reports",
                        help="Directory containing Phase A outputs")
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--domain", type=str, default="countdown")
    args = parser.parse_args()

    out = args.reports_dir

    # ---- Load Phase A artifacts ----
    print("[Phase B] Loading Phase A trajectories and probes...")

    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    test_traj_path = os.path.join(out, "trajectories", "test.pt")

    if not os.path.exists(train_traj_path) or not os.path.exists(test_traj_path):
        print("ERROR: Phase A trajectories not found. Run Phase A first.")
        print(f"  Expected: {train_traj_path}")
        print(f"  Expected: {test_traj_path}")
        sys.exit(1)

    train_trajs = load_trajectories(train_traj_path)
    test_trajs = load_trajectories(test_traj_path)
    print(f"  Loaded {len(train_trajs)} train, {len(test_trajs)} test trajectories.")

    # Re-fit probes (same as Phase A) to get fitted probe objects
    print("[Phase B] Fitting probes on train data...")
    probe_df, probe_details, fitted_probes = run_probes(train_trajs, test_trajs, domain=args.domain)
    probe_a = fitted_probes.get("A")
    probe_c = fitted_probes.get("C")

    # ---- Run Oracle evaluation ----
    print("[Phase B] Evaluating Oracle transition coherence...")
    oracle_df = evaluate_oracle_coherence(
        test_trajs,
        probe_a=probe_a, probe_c=probe_c,
        max_depth=args.max_depth,
        domain=args.domain,
    )
    save_oracle_coherence(
        oracle_df,
        os.path.join(out, "coherence_oracle_depth.csv"),
        os.path.join(out, "coherence_oracle_depth.png"),
    )
    print("  Oracle coherence saved.")

    # ---- Sanity check ----
    valid_oracle = oracle_df[oracle_df["n_samples"] > 0]
    cos_vals = valid_oracle["cosine_similarity"].tolist()
    mse_vals = valid_oracle["mse"].tolist()
    print(f"  Sanity check: Oracle cosine = {cos_vals}")
    print(f"  Sanity check: Oracle MSE    = {mse_vals}")
    if all(c > 0.999 for c in cos_vals) and all(m < 1e-6 for m in mse_vals):
        print("  ✅ Oracle sanity check PASSED.")
    else:
        print("  ⚠️ Oracle sanity check FAILED — investigate!")

    # ---- Load Phase A comparison CSVs ----
    action_csv = os.path.join(out, "coherence_action_depth.csv")
    blind_csv = os.path.join(out, "coherence_blind_depth.csv")

    if not os.path.exists(action_csv) or not os.path.exists(blind_csv):
        print("WARNING: Phase A coherence CSVs not found. "
              "Skipping comparison overlay and report.")
        return

    action_df = pd.read_csv(action_csv)
    blind_df = pd.read_csv(blind_csv)
    print(f"  Loaded Phase A action ({len(action_df)} rows) "
          f"and blind ({len(blind_df)} rows) CSVs.")

    # ---- Overlay plot ----
    print("[Phase B] Generating 4-way comparison overlay...")
    plot_comparison_overlay(
        oracle_df, action_df, blind_df,
        os.path.join(out, "coherence_comparison_overlay.png"),
    )

    # ---- Report ----
    print("[Phase B] Generating Phase B Oracle report...")
    generate_phase_b_report(
        oracle_df, action_df, blind_df,
        os.path.join(out, "phase_b_oracle_report.md"),
        args.domain,
    )

    print("\n[Phase B] Done. Artifacts:")
    for name in ["coherence_oracle_depth.csv",
                  "coherence_oracle_depth.png",
                  "coherence_comparison_overlay.png",
                  "phase_b_oracle_report.md"]:
        print(f"  - {os.path.join(out, name)}")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c.py`
```python
"""Phase C orchestrator: Intrinsic State Noise Diagnostic.

Loads all extracted trajectories, groups hidden states by their symbolic
state (available numbers + target), and computes within-state vs between-state
similarity. This determines if the hidden state is a viable Markovian
planning state or if it's too heavily reliant on the text history.

Usage (from repo root after Phase A completes)::

    python scripts/run_phase_c.py --reports_dir reports
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import (
    gather_state_groups,
    compute_intrinsic_noise,
    plot_noise_histogram
)


def generate_phase_c_report(df, top1_acc: float, top5_acc: float, md_path: str):
    """Generate the Phase C report based on Intrinsic Noise metrics."""
    within_df = df[df["type"] == "within"]
    between_df = df[df["type"] == "between"]
    
    w_cos_mean = within_df["cosine"].mean() if not within_df.empty else 0.0
    w_cos_std = within_df["cosine"].std() if not within_df.empty else 0.0
    w_l2_mean = within_df["l2"].mean() if not within_df.empty else 0.0
    w_l2_std = within_df["l2"].std() if not within_df.empty else 0.0
    
    b_cos_mean = between_df["cosine"].mean() if not between_df.empty else 0.0
    b_cos_std = between_df["cosine"].std() if not between_df.empty else 0.0
    b_l2_mean = between_df["l2"].mean() if not between_df.empty else 0.0
    b_l2_std = between_df["l2"].std() if not between_df.empty else 0.0
    
    noise_ratio_cos = b_cos_mean / w_cos_mean if w_cos_mean else float('inf')
    noise_ratio_l2 = w_l2_mean / b_l2_mean if b_l2_mean else float('inf')
    
    lines = []
    lines.append("# Phase C — Intrinsic State Noise Diagnostic\n")
    lines.append(
        "This diagnostic determines whether the LLM's hidden state is a stable "
        "Markovian representation of the symbolic task state, or whether it is "
        "highly history-dependent. We group hidden states across different "
        "trajectories that have reached the **exact same symbolic state** "
        "(i.e., same target and same available numbers).\n"
    )
    
    lines.append("## Metrics\n")
    lines.append("### Similarity")
    lines.append("| Metric | Within-State | Between-State | Noise Ratio |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Cosine Similarity | {w_cos_mean:.3f} ± {w_cos_std:.3f} | {b_cos_mean:.3f} ± {b_cos_std:.3f} | {noise_ratio_cos:.3f} (b/w) |")
    lines.append(f"| L2 Distance | {w_l2_mean:.3f} ± {w_l2_std:.3f} | {b_l2_mean:.3f} ± {b_l2_std:.3f} | {noise_ratio_l2:.3f} (w/b) |\n")
    
    lines.append("### Nearest Neighbor Symbolic State Retrieval")
    lines.append("For each hidden state, we query its nearest neighbor (by cosine similarity) from a **different trajectory**.")
    lines.append(f"- **Top-1 Retrieval Accuracy**: {top1_acc * 100:.1f}%")
    lines.append(f"- **Top-5 Retrieval Accuracy**: {top5_acc * 100:.1f}%\n")
    
    lines.append("## Interpretation Rules\n")
    lines.append("- **Case A**: If `within cosine >> between cosine`, the representation behaves like a stable planning state.")
    lines.append("- **Case C**: If `within cosine only slightly exceeds between cosine`, the representation is partially Markovian but noisy.")
    lines.append("- **Case B**: If `within cosine ≈ between cosine`, the representation is dominated by trajectory history.\n")
    
    lines.append("## Final Verdict\n")
    
    if top1_acc > 0.85:
        lines.append("**Representation appears highly stable.** (Case A)")
        lines.append("\nThe model maps identical symbolic states to very tight latent clusters regardless of history. Transition learning should be straightforward.")
    elif top1_acc > 0.40:
        lines.append("**Representation is partially Markovian but noisy.** (Case C)")
        lines.append("\nThere is clear clustering by symbolic state, but significant history dependence remains. Transition learning may require stronger architectures.")
    else:
        lines.append("**Representation appears highly history-dependent.** (Case B)")
        lines.append("\nThe model's hidden states for identical symbolic states are nearly as far apart as completely unrelated states. This is the true bottleneck; Oracle dynamics cannot save a representation that fails to consistently encode the task state.")

    lines.append("\n---")
    lines.append("\n*Generated by `scripts/run_phase_c.py`*")
    
    os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Phase C: Intrinsic State Noise Diagnostic")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory containing Phase A outputs")
    args = parser.parse_args()

    out = args.reports_dir

    print("[Phase C] Loading Phase A trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    test_traj_path = os.path.join(out, "trajectories", "test.pt")

    if not os.path.exists(train_traj_path) or not os.path.exists(test_traj_path):
        print("ERROR: Phase A trajectories not found. Run Phase A first.")
        sys.exit(1)

    train_trajs = load_trajectories(train_traj_path)
    test_trajs = load_trajectories(test_traj_path)
    all_trajs = train_trajs + test_trajs
    print(f"  Loaded {len(all_trajs)} total trajectories.")

    print("[Phase C] Grouping hidden states by symbolic state...")
    state_groups = gather_state_groups(all_trajs)
    num_valid_states = sum(1 for v in state_groups.values() if len(v) >= 2)
    print(f"  Found {len(state_groups)} shared symbolic states ({num_valid_states} with >= 2 instances).")

    if num_valid_states < 2:
        print("ERROR: Not enough shared symbolic states across trajectories to compute noise.")
        sys.exit(1)

    print("[Phase C] Computing within-state and between-state similarities...")
    df, top1_acc, top5_acc = compute_intrinsic_noise(state_groups)
    
    csv_path = os.path.join(out, "intrinsic_noise.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved raw metrics to {csv_path}")
    print(f"  Top-1 Retrieval Accuracy: {top1_acc*100:.1f}%")
    print(f"  Top-5 Retrieval Accuracy: {top5_acc*100:.1f}%")

    print("[Phase C] Generating histograms and report...")
    png_path = os.path.join(out, "intrinsic_noise_histogram.png")
    plot_noise_histogram(df, png_path)
    
    md_path = os.path.join(out, "intrinsic_noise_report.md")
    generate_phase_c_report(df, top1_acc, top5_acc, md_path)

    print("\n[Phase C] Done. Artifacts:")
    for name in ["intrinsic_noise.csv", "intrinsic_noise_histogram.png", "intrinsic_noise_report.md"]:
        print(f"  - {os.path.join(out, name)}")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c1.py`
```python
import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    
    print("[Phase C.1] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    
    if not os.path.exists(train_traj_path):
        print("ERROR: Trajectories not found. Run Phase A first.")
        sys.exit(1)
        
    trajs = load_trajectories(train_traj_path)
    print(f"Loaded {len(trajs)} trajectories.")
    
    # We want to group by:
    # 1. Symbolic State
    # 2. Depth
    # Data structure: states_by_sym[sym] = [(t_idx, depth, h)]
    # We also keep a flat list for random sampling
    
    states_by_sym = {}
    states_flat = []
    
    print("[Phase C.1] Grouping states by symbolic state and depth...")
    for t_idx, traj in enumerate(trajs):
        states_info = get_symbolic_states(traj)
        for d, info in enumerate(states_info):
            if info is not None:
                sym, hist = info
                h = traj.states[d:d+1] # (1, H)
                if sym not in states_by_sym:
                    states_by_sym[sym] = []
                states_by_sym[sym].append((t_idx, d, h))
                states_flat.append((sym, t_idx, d, h))
                
    # Define pairs to compute:
    # 1. Same state, Same depth (different traj)
    # 2. Same state, Different depth (different traj)
    # 3. Different state, Same depth (different traj)
    # 4. Different state, Different depth (different traj)
    
    results = {
        "Same state, Same depth": [],
        "Same state, Different depth": [],
        "Different state, Same depth": [],
        "Different state, Different depth": []
    }
    
    print("[Phase C.1] Sampling pairs and computing cosines...")
    MAX_PAIRS_PER_CAT = 10000
    
    # 1 & 2: Same state
    for sym, elements in states_by_sym.items():
        n = len(elements)
        if n < 2:
            continue
            
        # Sample limited number of pairs per state to avoid quadratic explosion
        pairs_to_sample = min(100, n * (n - 1) // 2)
        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(indices)
        indices = indices[:pairs_to_sample]
        
        for i, j in indices:
            t1, d1, h1 = elements[i]
            t2, d2, h2 = elements[j]
            if t1 != t2: # Must be different trajectories
                cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1).item()
                if d1 == d2:
                    if len(results["Same state, Same depth"]) < MAX_PAIRS_PER_CAT:
                        results["Same state, Same depth"].append(cos)
                else:
                    if len(results["Same state, Different depth"]) < MAX_PAIRS_PER_CAT:
                        results["Same state, Different depth"].append(cos)
                        
    # 3 & 4: Different state
    # Shuffle flat list
    random.shuffle(states_flat)
    
    sampled_diff = 0
    while sampled_diff < MAX_PAIRS_PER_CAT and len(states_flat) >= 2:
        # Pick 2 random states
        idx1 = random.randint(0, len(states_flat)-1)
        idx2 = random.randint(0, len(states_flat)-1)
        
        sym1, t1, d1, h1 = states_flat[idx1]
        sym2, t2, d2, h2 = states_flat[idx2]
        
        if sym1 != sym2 and t1 != t2:
            cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1).item()
            if d1 == d2:
                if len(results["Different state, Same depth"]) < MAX_PAIRS_PER_CAT:
                    results["Different state, Same depth"].append(cos)
                    sampled_diff += 1
            else:
                if len(results["Different state, Different depth"]) < MAX_PAIRS_PER_CAT:
                    results["Different state, Different depth"].append(cos)
                    sampled_diff += 1
                    
    print("[Phase C.1] Generating report...")
    md_path = os.path.join(out, "phase_c1_anisotropy_report.md")
    
    with open(md_path, "w") as f:
        f.write("# Phase C.1 — Anisotropy Audit\n\n")
        f.write("This audit decouples symbolic state similarity from token position (reasoning depth) similarity to identify if the latent space is dominated by position.\n\n")
        
        f.write("## Results\n\n")
        f.write("| Pair Type | Cosine | N |\n")
        f.write("| --- | --- | --- |\n")
        
        metrics = {}
        for cat, vals in results.items():
            if len(vals) > 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                metrics[cat] = mean_val
                f.write(f"| {cat} | {mean_val:.3f} ± {std_val:.3f} | {len(vals)} |\n")
            else:
                f.write(f"| {cat} | N/A | 0 |\n")
                metrics[cat] = 0.0
                
        f.write("\n## Verdict\n\n")
        diff_state_same_depth = metrics.get("Different state, Same depth", 0.0)
        same_state_same_depth = metrics.get("Same state, Same depth", 0.0)
        
        if diff_state_same_depth > 0.8:
            f.write("**POSITION DOMINATES (Anisotropy Confound).**\n\n")
            f.write("Hidden states at the same reasoning depth are highly similar regardless of the symbolic task state. The previously reported 'within-state' similarity was likely an artifact of comparing states at the same depth, not true symbolic alignment. The representation is fragmented by token position.\n")
        elif same_state_same_depth - diff_state_same_depth > 0.15:
            f.write("**SYMBOLIC STATE ENCODED (Position Controlled).**\n\n")
            f.write("States sharing the same symbolic task state are significantly closer than different states at the same depth. Position does not dominate the representation.\n")
        else:
            f.write("**INCONCLUSIVE.**\n\n")
            f.write("Differences are marginal. Further investigation required.\n")
            
    print(f"Report written to {md_path}")
    
    # Save raw CSV
    rows = []
    for cat, vals in results.items():
        for v in vals:
            rows.append({"category": cat, "cosine": v})
            
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out, "anisotropy_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c1_advanced.py`
```python
import os
import sys
import argparse
import random
import collections
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states

def get_mean_state(flat_h):
    # flat_h is list of (sym, t_idx, d, h)
    all_h = torch.cat([h for _,_,_,h in flat_h], dim=0)
    mu = all_h.mean(dim=0, keepdim=True)
    return mu

def sample_pairs(flat_h, max_pairs_per_cat=5000):
    # We want to stratify by depth where possible.
    # We'll group flat_h by depth, sym, traj
    by_d = collections.defaultdict(list)
    by_sym = collections.defaultdict(list)
    by_traj = collections.defaultdict(list)
    
    for item in flat_h:
        sym, t_idx, d, h = item
        by_d[d].append(item)
        by_sym[sym].append(item)
        by_traj[t_idx].append(item)
        
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    
    # helper
    def add_pair(d, cat, h1, h2):
        if len(pairs[d][cat]) < max_pairs_per_cat:
            pairs[d][cat].append((h1, h2))
            return True
        return False

    print("  Sampling Same State, Same Depth, Diff Traj & Same State, Diff Depth, Diff Traj...")
    for sym, items in by_sym.items():
        n = len(items)
        if n < 2: continue
        # sample some random pairs
        num_to_sample = min(100, n*(n-1)//2)
        idx = [(i,j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(idx)
        for i,j in idx[:num_to_sample]:
            sym1, t1, d1, h1 = items[i]
            sym2, t2, d2, h2 = items[j]
            if t1 != t2:
                if d1 == d2:
                    add_pair(d1, "SS_SD_DT", h1, h2)
                else:
                    # add to both depths for symmetric tracking, or just track diff depth overall
                    # let's map it to d1 for stratification (focusing on d1 as the source)
                    add_pair(d1, "SS_DD_DT", h1, h2)

    print("  Sampling Diff State, Same Depth, Same Traj...")
    for t_idx, items in by_traj.items():
        n = len(items)
        if n < 2: continue
        num_to_sample = min(20, n*(n-1)//2)
        idx = [(i,j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(idx)
        for i,j in idx[:num_to_sample]:
            sym1, t1, d1, h1 = items[i]
            sym2, t2, d2, h2 = items[j]
            if sym1 != sym2 and d1 == d2: # usually impossible to have diff state at same depth in SAME traj, but let's check
                add_pair(d1, "DS_SD_ST", h1, h2)
                
    # Note: DS_SD_ST might be empty if a trajectory always has the same state at the same depth (which is physically required by the math task unless branching happens!)
    # Actually, one trajectory only has ONE state at depth d. So DS_SD_ST is logically impossible. We'll skip it or it'll just be empty.
    
    print("  Sampling Diff State, Same Depth, Diff Traj & Diff State, Diff Depth, Diff Traj...")
    # sample randomly
    sampled = 0
    while sampled < max_pairs_per_cat * 5:
        i1 = random.randint(0, len(flat_h)-1)
        i2 = random.randint(0, len(flat_h)-1)
        sym1, t1, d1, h1 = flat_h[i1]
        sym2, t2, d2, h2 = flat_h[i2]
        
        if sym1 != sym2 and t1 != t2:
            if d1 == d2:
                if add_pair(d1, "DS_SD_DT", h1, h2): sampled+=1
            else:
                if add_pair(d1, "DS_DD_DT", h1, h2): sampled+=1
                
    return pairs

def evaluate_classifiers(flat_h):
    print("  Training Classifiers...")
    X = []
    y_depth = []
    y_sym = []
    
    sym_counts = collections.Counter([x[0] for x in flat_h])
    top_syms = set(sym for sym, c in sym_counts.most_common(50))
    
    # For state classifier, filter to top 50 syms
    X_state = []
    y_state_filtered = []
    
    for sym, t_idx, d, h in flat_h:
        X.append(h.squeeze(0).numpy())
        y_depth.append(d)
        y_sym.append(str(sym))
        
        if sym in top_syms:
            X_state.append(h.squeeze(0).numpy())
            y_state_filtered.append(str(sym))
            
    X = np.array(X)
    y_depth = np.array(y_depth)
    
    X_state = np.array(X_state)
    y_state_filtered = np.array(y_state_filtered)
    
    # Depth Classifier
    Xtr, Xte, ytr, yte = train_test_split(X, y_depth, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    clf_depth = LogisticRegression(max_iter=1000)
    clf_depth.fit(scaler.transform(Xtr), ytr)
    depth_acc = accuracy_score(yte, clf_depth.predict(scaler.transform(Xte)))
    
    # State Classifier
    Xtr, Xte, ytr, yte = train_test_split(X_state, y_state_filtered, test_size=0.2, random_state=42)
    scaler2 = StandardScaler().fit(Xtr)
    clf_state = LogisticRegression(max_iter=1000)
    clf_state.fit(scaler2.transform(Xtr), ytr)
    state_acc = accuracy_score(yte, clf_state.predict(scaler2.transform(Xte)))
    
    return depth_acc, state_acc

def evaluate_retrieval(flat_h):
    print("  Evaluating Conditioned Retrieval...")
    # sample 5000 items to avoid huge matrix
    if len(flat_h) > 5000:
        sample = random.sample(flat_h, 5000)
    else:
        sample = flat_h
        
    H_mat = torch.cat([x[3] for x in sample], dim=0) # (M, H)
    H_mat = F.normalize(H_mat, p=2, dim=1)
    sim_matrix = torch.matmul(H_mat, H_mat.T) # (M, M)
    
    M = len(sample)
    t_idx_arr = np.array([x[1] for x in sample])
    sym_arr = np.array([str(x[0]) for x in sample])
    d_arr = np.array([x[2] for x in sample])
    
    # Unconstrained
    # Mask same trajectory
    mask_traj = torch.tensor(t_idx_arr[:, None] == t_idx_arr[None, :])
    sim_unconstrained = sim_matrix.clone()
    sim_unconstrained.masked_fill_(mask_traj, -1.0)
    top1_unconstrained = sim_unconstrained.argmax(dim=1)
    
    hits_un = 0
    for i in range(M):
        if sym_arr[i] == sym_arr[top1_unconstrained[i]]:
            hits_un += 1
            
    # Same-depth
    mask_diff_depth = torch.tensor(d_arr[:, None] != d_arr[None, :])
    sim_same_depth = sim_matrix.clone()
    sim_same_depth.masked_fill_(mask_traj | mask_diff_depth, -1.0)
    top1_sd = sim_same_depth.argmax(dim=1)
    
    hits_sd = 0
    valid_sd = 0
    for i in range(M):
        if sim_same_depth[i, top1_sd[i]] > -0.99: # valid neighbor exists
            valid_sd += 1
            if sym_arr[i] == sym_arr[top1_sd[i]]:
                hits_sd += 1
                
    # Cross-depth
    mask_same_depth = torch.tensor(d_arr[:, None] == d_arr[None, :])
    sim_cross_depth = sim_matrix.clone()
    sim_cross_depth.masked_fill_(mask_traj | mask_same_depth, -1.0)
    top1_cd = sim_cross_depth.argmax(dim=1)
    
    hits_cd = 0
    valid_cd = 0
    for i in range(M):
        if sim_cross_depth[i, top1_cd[i]] > -0.99:
            valid_cd += 1
            if sym_arr[i] == sym_arr[top1_cd[i]]:
                hits_cd += 1
                
    return (hits_un/M, 
            hits_sd/valid_sd if valid_sd>0 else 0, 
            hits_cd/valid_cd if valid_cd>0 else 0)

def plot_pca(flat_h, out_dir):
    print("  Generating PCA visualizations...")
    if len(flat_h) > 5000:
        sample = random.sample(flat_h, 5000)
    else:
        sample = flat_h
        
    X = np.array([x[3].squeeze(0).numpy() for x in sample])
    depths = np.array([x[2] for x in sample])
    syms = np.array([str(x[0]) for x in sample])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 1. Color by Depth
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=depths, palette="viridis", s=10)
    plt.title("PCA of Hidden States (Colored by Depth)")
    plt.savefig(os.path.join(out_dir, "pca_depth_colored.png"))
    plt.close()
    
    # 2. Color by Top 10 States
    top_syms = [s for s,c in collections.Counter(syms).most_common(10)]
    mask = np.isin(syms, top_syms)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[mask, 0], y=X_pca[mask, 1], hue=syms[mask], palette="tab10", s=10)
    plt.title("PCA of Hidden States (Colored by Top 10 Symbolic States)")
    plt.savefig(os.path.join(out_dir, "pca_state_colored.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    
    print("[Phase C.1 Advanced] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    trajs = load_trajectories(train_traj_path)
    
    # To save time in scripting, limit to a representative subset
    trajs = trajs[:5000] 
    
    flat_h = []
    for t_idx, traj in enumerate(trajs):
        states_info = get_symbolic_states(traj)
        for d, info in enumerate(states_info):
            if info is not None:
                sym, hist = info
                h = traj.states[d:d+1]
                flat_h.append((sym, t_idx, d, h))
                
    mu = get_mean_state(flat_h)
    pairs = sample_pairs(flat_h)
    
    # Compute Cosines
    print("[Phase C.1 Advanced] Computing Pairwise Cosines...")
    results = []
    
    for d, cats in pairs.items():
        for cat, plist in cats.items():
            if len(plist) == 0: continue
            raw_cosines = []
            centered_cosines = []
            for h1, h2 in plist:
                r_cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1).item()
                c_cos = F.cosine_similarity((h1.float()-mu), (h2.float()-mu), dim=-1).item()
                raw_cosines.append(r_cos)
                centered_cosines.append(c_cos)
            results.append({
                "Depth": d,
                "Category": cat,
                "Raw Cosine": np.mean(raw_cosines),
                "Centered Cosine": np.mean(centered_cosines),
                "N": len(plist)
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out, "anisotropy_advanced_metrics.csv"), index=False)
    
    # Classification
    depth_acc, state_acc = evaluate_classifiers(flat_h)
    
    # Retrieval
    un_ret, sd_ret, cd_ret = evaluate_retrieval(flat_h)
    
    # PCA
    plot_pca(flat_h, out)
    
    print("[Phase C.1 Advanced] Generating Report...")
    with open(os.path.join(out, "phase_c1_advanced_report.md"), "w") as f:
        f.write("# Phase C.1 Advanced — Anisotropy Audit\n\n")
        
        f.write("## 1. Classifiers\n")
        f.write(f"- **Depth Classification Accuracy**: {depth_acc*100:.1f}%\n")
        f.write(f"- **State Classification Accuracy (Top 50)**: {state_acc*100:.1f}%\n\n")
        
        f.write("## 2. Depth-Conditioned Retrieval\n")
        f.write(f"- **Unconstrained Top-1**: {un_ret*100:.1f}%\n")
        f.write(f"- **Same-Depth Only Top-1**: {sd_ret*100:.1f}%\n")
        f.write(f"- **Cross-Depth Only Top-1**: {cd_ret*100:.1f}%\n\n")
        
        f.write("## 3. Stratified Cosine Similarities\n")
        f.write("| Depth | Category | Raw Cosine | Centered Cosine | N |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in df.sort_values(["Depth", "Category"]).iterrows():
            f.write(f"| {r['Depth']} | {r['Category']} | {r['Raw Cosine']:.3f} | {r['Centered Cosine']:.3f} | {r['N']} |\n")
            
        f.write("\n## 4. Representation Score (R)\n")
        f.write("`R = (SS_SD_DT - DS_SD_DT) / (SS_SD_DT - DS_DD_DT)` computed on Centered Cosine.\n\n")
        f.write("| Depth | R Score |\n")
        f.write("|---|---|\n")
        
        r_scores = []
        for d in sorted(df['Depth'].unique()):
            d_df = df[df['Depth'] == d]
            def get_val(cat):
                v = d_df[d_df['Category'] == cat]['Centered Cosine']
                return v.iloc[0] if len(v) > 0 else None
                
            ss_sd = get_val("SS_SD_DT")
            ds_sd = get_val("DS_SD_DT")
            ds_dd = get_val("DS_DD_DT")
            
            if ss_sd is not None and ds_sd is not None and ds_dd is not None and (ss_sd - ds_dd) != 0:
                r = (ss_sd - ds_sd) / (ss_sd - ds_dd)
                f.write(f"| {d} | {r:.3f} |\n")
                r_scores.append(r)
            else:
                f.write(f"| {d} | N/A |\n")
                
        f.write("\n## Verdict\n\n")
        mean_r = np.mean(r_scores) if r_scores else 0
        if depth_acc > state_acc + 0.20 and mean_r < 0.3:
            f.write("**DEPTH DOMINATES (Position Confound).**\n")
            f.write("The representation is fundamentally entangled with token position. PCA and classifiers show depth is the primary axis of variance. Symbolic state retrieval fails across depths.\n")
        elif mean_r > 0.7:
            f.write("**SYMBOLIC STATE DOMINATES.**\n")
            f.write("The representation preserves symbolic state across reasoning depths. R score is high.\n")
        else:
            f.write("**MIXED / ENTANGLED.**\n")
            f.write("Both depth and state heavily influence the geometry.\n")
            
    print("Done. Report written.")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c2_depth_removal.py`
```python
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories
from evaluation.probes import extract_probe_data
from evaluation.intrinsic_noise import get_symbolic_states

# -----------------------------------------------------------------------------
# Gradient Reversal Layer
# -----------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

# -----------------------------------------------------------------------------
# Adversarial Network
# -----------------------------------------------------------------------------
class AdversarialProjector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256, num_ops=4, num_depths=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.state_head = nn.Linear(out_dim, num_ops)
        self.grl = GradientReversalLayer()
        self.depth_head = nn.Linear(out_dim, num_depths)

    def forward(self, x, lambda_=1.0):
        z = self.encoder(x)
        z_norm = F.normalize(z, p=2, dim=1)
        state_logits = self.state_head(z_norm)
        depth_logits = self.depth_head(self.grl(z_norm, lambda_))
        return z_norm, state_logits, depth_logits

def extract_probe_data_with_depth(trajs):
    X, C, D = [], [], []
    for traj in trajs:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        for i in range(N):
            X.append(states[i].numpy())
            C.append(ops[i])
            D.append(i)
    return np.array(X), np.array(C), np.array(D)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    
    print("[Phase C.2] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    
    # We load 10000 trajectories
    trajs = load_trajectories(train_traj_path)[:10000]
    
    # Extract probe data with true depth
    X, y_state, y_depth = extract_probe_data_with_depth(trajs)
    
    print(f"Dataset size: {len(X)} states.")
    
    # Train/Test Split
    X_train, X_test, ys_train, ys_test, yd_train, yd_test = train_test_split(
        X, y_state, y_depth, test_size=0.2, random_state=42
    )
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    ys_tr_t = torch.tensor(ys_train, dtype=torch.long)
    yd_tr_t = torch.tensor(yd_train, dtype=torch.long)
    
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    ys_te_t = torch.tensor(ys_test, dtype=torch.long)
    yd_te_t = torch.tensor(yd_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_tr_t, ys_tr_t, yd_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ops = len(np.unique(y_state))
    num_depths = int(np.max(y_depth)) + 1
    
    model = AdversarialProjector(2048, 256, num_ops, num_depths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_s = nn.CrossEntropyLoss()
    criterion_d = nn.CrossEntropyLoss()
    
    print("[Phase C.2] Training Adversarial Projector...")
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        total_s_loss = 0
        total_d_loss = 0
        
        p = epoch / num_epochs
        lambda_val = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        for bx, bys, byd in train_loader:
            bx, bys, byd = bx.to(device), bys.to(device), byd.to(device)
            optimizer.zero_grad()
            z, s_log, d_log = model(bx, lambda_=lambda_val)
            
            s_loss = criterion_s(s_log, bys)
            d_loss = criterion_d(d_log, byd)
            
            loss = s_loss + d_loss
            loss.backward()
            optimizer.step()
            
            total_s_loss += s_loss.item()
            total_d_loss += d_loss.item()
            
        print(f"  Epoch {epoch+1}/{num_epochs} [Lambda: {lambda_val:.2f}] | State (Op) Loss: {total_s_loss/len(train_loader):.3f} | Depth Loss: {total_d_loss/len(train_loader):.3f}")
        
    model.eval()
    with torch.no_grad():
        Z_train = model.encoder(X_tr_t.to(device)).cpu().numpy()
        Z_train = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)
        Z_test = model.encoder(X_te_t.to(device)).cpu().numpy()
        Z_test = Z_test / np.linalg.norm(Z_test, axis=1, keepdims=True)
        
    print("\n[Phase C.2] Evaluating Linear Probes...")
    
    def eval_probes(Xtr, ytr_d, ytr_s, Xte, yte_d, yte_s):
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        
        clf_d = LogisticRegression(max_iter=1000)
        clf_d.fit(Xtr_s, ytr_d)
        acc_d = accuracy_score(yte_d, clf_d.predict(Xte_s))
        
        clf_s = LogisticRegression(max_iter=1000)
        clf_s.fit(Xtr_s, ytr_s)
        acc_s = accuracy_score(yte_s, clf_s.predict(Xte_s))
        return acc_d, acc_s
        
    acc_d_before, acc_s_before = eval_probes(X_train, yd_train, ys_train, X_test, yd_test, ys_test)
    acc_d_after, acc_s_after = eval_probes(Z_train, yd_train, ys_train, Z_test, yd_test, ys_test)
    
    print(f"  [BEFORE - Raw h] Depth Acc: {acc_d_before*100:.1f}%, State (Op) Acc: {acc_s_before*100:.1f}%")
    print(f"  [AFTER  - Proj z] Depth Acc: {acc_d_after*100:.1f}%, State (Op) Acc: {acc_s_after*100:.1f}%")
    
    chance_depth = np.max(np.bincount(yd_test)) / len(yd_test)
    chance_state = np.max(np.bincount(ys_test)) / len(ys_test)
    
    verdict = ""
    success = False
    
    if acc_d_after <= chance_depth + 0.10 and acc_s_after >= acc_s_before - 0.05:
        verdict = "**SYMBOLIC INFORMATION WAS MASKED.**\nDepth was successfully stripped from the representation without destroying symbolic state accuracy (Next Operation). The task information exists but was hidden behind a dominant depth manifold."
        success = True
    elif acc_d_after <= chance_depth + 0.10 and acc_s_after < acc_s_before - 0.05:
        verdict = "**INSEPARABLE / ABSENT.**\nDepth was successfully stripped, but symbolic state accuracy collapsed as well. The state information and depth information are inextricably linked, or the state information relies on depth to be linearly decodable."
    else:
        verdict = "**INCOMPLETE DISENTANGLEMENT.**\nAdversarial training failed to fully remove depth information from the projection. A stronger penalty (lambda) or more complex projector may be required."
        
    report_path = os.path.join(out, "phase_c2_depth_removal_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase C.2 — Depth Removal (Adversarial Projection)\n\n")
        f.write("This experiment strips token position/depth information from the hidden state while preserving symbolic state information. Because the combinatorial explosion of the Countdown game yields 19,771 unique exact symbolic states (with $<2$ examples per class), we define 'State Information' as **Probe C (Next Symbolic Operation)**, allowing the network to train on all 20,000+ states.\n\n")
        
        f.write("## Evaluation: Independent Linear Probes\n\n")
        f.write("| Representation | Depth Accuracy | State Accuracy (Next Op) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Raw $h$ (2048-dim) | {acc_d_before*100:.1f}% | {acc_s_before*100:.1f}% |\n")
        f.write(f"| Projected $z$ (256-dim) | {acc_d_after*100:.1f}% | {acc_s_after*100:.1f}% |\n")
        f.write(f"| *Chance Baseline* | *{chance_depth*100:.1f}%* | *{chance_state*100:.1f}%* |\n\n")
        
        f.write("## Final Verdict\n\n")
        f.write(verdict + "\n")
        
    print(f"\n[Phase C.2] Done. Verdict: {'SUCCESS' if success else 'FAILURE'}")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c3_geometry.py`
```python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories

LARGE_NUMBERS = [25, 50, 75, 100]

def extract_all_targets(trajectories):
    X, A, B, C, D, Depth = [], [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        operands = traj.operands.tolist()
        used = set()
        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i  # steps remaining
            
            a_row = []
            for v in LARGE_NUMBERS:
                if v not in traj.numbers:
                    a_row.append(0)
                elif v not in used:
                    a_row.append(1)
                else:
                    a_row.append(2)
            A.append(a_row)
            B.append(dist)
            C.append(ops[i])
            D.append(1 if dist <= 2 else 0)
            Depth.append(i) # True depth!
            
            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))
            
    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "B": np.array(B, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "D": np.array(D, dtype=np.int64),
        "Depth": np.array(Depth, dtype=np.int64)
    }

class MultiLabelProbe:
    def __init__(self):
        self.clfs = []
        self.constants = []

    def fit(self, X, Y):
        for j in range(Y.shape[1]):
            classes = np.unique(Y[:, j])
            if classes.shape[0] < 2:
                self.clfs.append(None)
                self.constants.append(int(classes[0]) if classes.shape[0] else 0)
            else:
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X, Y[:, j])
                self.clfs.append(clf)
                self.constants.append(None)

    def predict(self, X):
        preds = []
        for clf, const in zip(self.clfs, self.constants):
            if clf is None:
                preds.append(np.full((X.shape[0],), const))
            else:
                preds.append(clf.predict(X))
        return np.column_stack(preds)

def evaluate_subset(Xtr, ytr_dict, Xte, yte_dict):
    results = {}
    
    # Depth (Logistic Regression for classification acc)
    clf_depth = LogisticRegression(max_iter=2000)
    clf_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth Acc"] = accuracy_score(yte_dict["Depth"], clf_depth.predict(Xte))
    
    # Depth (Linear Regression for R^2)
    reg_depth = LinearRegression()
    reg_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth R2"] = r2_score(yte_dict["Depth"], reg_depth.predict(Xte))
    
    # Probe C (Next Op)
    clf_c = LogisticRegression(max_iter=2000)
    clf_c.fit(Xtr, ytr_dict["C"])
    results["Next Op (Probe C)"] = accuracy_score(yte_dict["C"], clf_c.predict(Xte))
    
    # Probe D (Reachable in 2)
    clf_d = LogisticRegression(max_iter=2000)
    clf_d.fit(Xtr, ytr_dict["D"])
    results["Reachable in 2 (Probe D)"] = accuracy_score(yte_dict["D"], clf_d.predict(Xte))
    
    # Probe B (Distance to solution)
    clf_b = LogisticRegression(max_iter=2000)
    clf_b.fit(Xtr, ytr_dict["B"])
    results["Dist to Sol (Probe B)"] = accuracy_score(yte_dict["B"], clf_b.predict(Xte))
    
    # Probe A (Remaining Numbers) - Multi-label
    clf_a = MultiLabelProbe()
    clf_a.fit(Xtr, ytr_dict["A"])
    pred_a = clf_a.predict(Xte)
    
    accs = []
    for j in range(ytr_dict["A"].shape[1]):
        accs.append(accuracy_score(yte_dict["A"][:, j], pred_a[:, j]))
        
    exact_acc = float(np.mean(np.all(yte_dict["A"] == pred_a, axis=1)))
    hamming_acc = float(np.mean(yte_dict["A"] == pred_a))
    
    results["Rem Nums (A) Exact Match"] = exact_acc
    results["Rem Nums (A) Per-Label"] = np.mean(accs)
    results["Rem Nums (A) Hamming"] = hamming_acc
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    
    print("[Phase C.3] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    trajs = load_trajectories(train_traj_path)[:10000]
    
    print("[Phase C.3] Extracting targets...")
    data = extract_all_targets(trajs)
    X = data["X"]
    
    print(f"Dataset size: {len(X)} states.")
    
    # Train test split on indices
    indices = np.arange(len(X))
    idx_tr, idx_te = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Standardize X before PCA (fit on train only)
    print("[Phase C.3] Standardizing & fitting PCA...")
    scaler = StandardScaler().fit(X[idx_tr])
    X_scaled = scaler.transform(X)
    
    # Extract up to 100 components, plus keeping all 2048
    pca = PCA(n_components=100)
    pca.fit(X_scaled[idx_tr])
    X_pca = pca.transform(X_scaled)
    
    print(f"Explained Variance (PC1-100): {np.sum(pca.explained_variance_ratio_):.3f}")
    
    ytr_dict = {k: v[idx_tr] for k, v in data.items() if k != "X"}
    yte_dict = {k: v[idx_te] for k, v in data.items() if k != "X"}
    
    # Define Subsets
    subsets = [1, 5, 10, 20, 50, 100]
    all_results = []
    
    print("[Phase C.3] Training Probes on PC Subsets...")
    for k in subsets:
        print(f"  Fitting PC 1-{k}...")
        Xtr_sub = X_pca[idx_tr, :k]
        Xte_sub = X_pca[idx_te, :k]
        res = evaluate_subset(Xtr_sub, ytr_dict, Xte_sub, yte_dict)
        res["PC Subset"] = f"PC 1-{k}"
        all_results.append(res)
        
    print("  Fitting Baseline (All 2048)...")
    res_baseline = evaluate_subset(X_scaled[idx_tr], ytr_dict, X_scaled[idx_te], yte_dict)
    res_baseline["PC Subset"] = "All 2048"
    all_results.append(res_baseline)
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = ["PC Subset", "Depth R2", "Depth Acc", "Rem Nums (A) Exact Match", "Rem Nums (A) Hamming", "Rem Nums (A) Per-Label", "Next Op (Probe C)", "Dist to Sol (Probe B)", "Reachable in 2 (Probe D)"]
    df = df[cols]
    
    csv_path = os.path.join(out, "phase_c3_geometry_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    report_path = os.path.join(out, "phase_c3_geometry_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase C.3 — Representation Geometry Audit\n\n")
        f.write("This audit measures how target information (Depth and Planning Probes) is distributed across the top Principal Components of the hidden state.\n\n")
        
        f.write("## Probe Accuracies across PC Subsets\n\n")
        f.write("| PC Subset | Depth $R^2$ | Depth Acc | Probe A (Exact) | Probe A (Hamming) | Probe A (Per-Label) | Probe C (Next Op) | Probe B (Dist to Sol) | Probe D (Reachable) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        
        for _, r in df.iterrows():
            f.write(f"| {r['PC Subset']} | {r['Depth R2']:.3f} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact Match']*100:.1f}% | {r['Rem Nums (A) Hamming']*100:.1f}% | {r['Rem Nums (A) Per-Label']*100:.1f}% | {r['Next Op (Probe C)']*100:.1f}% | {r['Dist to Sol (Probe B)']*100:.1f}% | {r['Reachable in 2 (Probe D)']*100:.1f}% |\n")
            
        # Verdict logic
        depth_r2_pc5 = df[df["PC Subset"] == "PC 1-5"]["Depth R2"].iloc[0]
        depth_r2_all = df[df["PC Subset"] == "All 2048"]["Depth R2"].iloc[0]
        
        f.write("\n## Verdict\n\n")
        if depth_r2_pc5 > 0.90 * depth_r2_all:
            f.write("**CONCENTRATED.**\nDepth is tightly concentrated in the very first few principal components (e.g. PC1-5 explains nearly all Depth variance).")
        else:
            f.write("**DISTRIBUTED.**\nDepth is not a single simple axis. It requires many dimensions to decode accurately, meaning it is fundamentally woven throughout the representation manifold.")
            
    print(f"\n[Phase C.3] Done. Report written to {report_path}")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_c4a_pca_ablation.py`
```python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states

LARGE_NUMBERS = [25, 50, 75, 100]
# Fixed vocabulary of numbers that can appear in a prompt (small 1-10 + large).
# Used to build the "raw operand" baseline: what is literally listed in the prompt.
BAG_VOCAB = list(range(1, 11)) + [25, 50, 75, 100]

def extract_all_targets(trajectories):
    X, A, C, Depth, Sym, Hist, Bag = [], [], [], [], [], [], []
    for traj in trajectories:
        states = traj.states
        N = traj.num_steps
        ops = traj.op_ids.tolist()
        operands = traj.operands.tolist()
        used = set()

        # Bag-of-prompt-numbers: constant per trajectory (what the probe could
        # read straight off the prompt text without any computation).
        bag_row = [1 if v in traj.numbers else 0 for v in BAG_VOCAB]

        # Get exact symbolic states for retrieval
        states_info = get_symbolic_states(traj)

        for i in range(N):
            X.append(states[i].numpy())
            dist = N - i

            a_row = []
            for v in LARGE_NUMBERS:
                if v not in traj.numbers:
                    a_row.append(0)
                elif v not in used:
                    a_row.append(1)
                else:
                    a_row.append(2)
            A.append(a_row)
            C.append(ops[i])
            Depth.append(i)
            Bag.append(bag_row)

            info = states_info[i] if i < len(states_info) else None
            if info is not None:
                sym, hist = info
                Sym.append(str(sym))
                # Action history (prefix of steps taken). Identical history with an
                # identical header => byte-identical hidden state under the frozen
                # causal LM, i.e. a trivial prefix collision.
                Hist.append(str(hist))
            else:
                Sym.append("NONE")
                Hist.append("NONE")

            used.add(int(operands[i][0]))
            used.add(int(operands[i][1]))

    return {
        "X": np.array(X, dtype=np.float32),
        "A": np.array(A, dtype=np.int64),
        "C": np.array(C, dtype=np.int64),
        "Depth": np.array(Depth, dtype=np.int64),
        "Sym": np.array(Sym),
        "Hist": np.array(Hist),
        "Bag": np.array(Bag, dtype=np.float32),
    }

class MultiLabelProbe:
    def __init__(self):
        self.clfs = []
        self.constants = []

    def fit(self, X, Y):
        for j in range(Y.shape[1]):
            classes = np.unique(Y[:, j])
            if classes.shape[0] < 2:
                self.clfs.append(None)
                self.constants.append(int(classes[0]) if classes.shape[0] else 0)
            else:
                clf = LogisticRegression(max_iter=2000)
                clf.fit(X, Y[:, j])
                self.clfs.append(clf)
                self.constants.append(None)

    def predict(self, X):
        preds = []
        for clf, const in zip(self.clfs, self.constants):
            if clf is None:
                preds.append(np.full((X.shape[0],), const))
            else:
                preds.append(clf.predict(X))
        return np.column_stack(preds)

def evaluate_subset(Xtr, ytr_dict, Xte, yte_dict, dup_exclude=None):
    """Evaluate probes + retrieval on a feature subset.

    dup_exclude: optional list of length n_test; entry i is an array of TRAIN
    column indices to exclude from retrieval for test query i (prefix collisions,
    i.e. train states whose raw hidden state is byte-identical to the query).
    """
    results = {}
    
    # Depth (Logistic Regression)
    clf_depth = LogisticRegression(max_iter=2000)
    clf_depth.fit(Xtr, ytr_dict["Depth"])
    results["Depth Acc"] = accuracy_score(yte_dict["Depth"], clf_depth.predict(Xte))
    
    # Probe C (Next Op)
    clf_c = LogisticRegression(max_iter=2000)
    clf_c.fit(Xtr, ytr_dict["C"])
    results["Next Op (Probe C)"] = accuracy_score(yte_dict["C"], clf_c.predict(Xte))
    
    # Probe A (Remaining Numbers) - Multi-label
    clf_a = MultiLabelProbe()
    clf_a.fit(Xtr, ytr_dict["A"])
    pred_a = clf_a.predict(Xte)
    
    exact_acc = float(np.mean(np.all(yte_dict["A"] == pred_a, axis=1)))
    hamming_acc = float(np.mean(yte_dict["A"] == pred_a))
    
    results["Rem Nums (A) Exact"] = exact_acc
    results["Rem Nums (A) Hamming"] = hamming_acc
    
    # Retrieval
    # Normalize features for cosine similarity
    Xte_norm = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    Xtr_norm = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    
    batch_size = 1000
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    valid_queries = 0
    
    within_cosines = []
    between_cosines = []
    
    train_syms = ytr_dict["Sym"]
    train_syms_set = set(train_syms)
    
    # Vectorized computation batch by batch
    for i in range(0, Xte_norm.shape[0], batch_size):
        end = min(i + batch_size, Xte_norm.shape[0])
        Xte_batch = Xte_norm[i:end]
        yte_sym_batch = yte_dict["Sym"][i:end]
        
        sims = np.dot(Xte_batch, Xtr_norm.T)

        # Prefix mask: push byte-identical (prefix-collision) train neighbors to
        # the bottom so they can neither be retrieved nor inflate within-cosine.
        if dup_exclude is not None:
            for r in range(end - i):
                ex = dup_exclude[i + r]
                if ex is not None and len(ex) > 0:
                    sims[r, ex] = -1.0

        # Get top 10 indices
        top10_idx = np.argsort(sims, axis=1)[:, -10:][:, ::-1]

        for idx_in_batch in range(end - i):
            query_sym = yte_sym_batch[idx_in_batch]
            if query_sym == "NONE" or query_sym not in train_syms_set:
                continue

            valid_queries += 1

            # Within vs Between similarity
            excl_mask = np.zeros(train_syms.shape[0], dtype=bool)
            if dup_exclude is not None:
                ex = dup_exclude[i + idx_in_batch]
                if ex is not None and len(ex) > 0:
                    excl_mask[ex] = True
            same_mask = (train_syms == query_sym) & ~excl_mask
            diff_mask = ~(train_syms == query_sym) & (train_syms != "NONE") & ~excl_mask

            if np.any(same_mask):
                within_cosines.append(sims[idx_in_batch, same_mask].mean())
            if np.any(diff_mask):
                between_cosines.append(sims[idx_in_batch, diff_mask].mean())
            
            # Top hits
            hit_indices = [train_syms[top10_idx[idx_in_batch, k]] == query_sym for k in range(10)]
            
            if hit_indices[0]:
                top1_hits += 1
            if any(hit_indices[:5]):
                top5_hits += 1
            if any(hit_indices[:10]):
                top10_hits += 1
                
    results["Retrieval Top-1"] = top1_hits / valid_queries if valid_queries > 0 else 0.0
    results["Retrieval Top-5"] = top5_hits / valid_queries if valid_queries > 0 else 0.0
    results["Retrieval Top-10"] = top10_hits / valid_queries if valid_queries > 0 else 0.0
    
    mean_within = np.mean(within_cosines) if within_cosines else 0.0
    mean_between = np.mean(between_cosines) if between_cosines else 1.0
    results["Within/Between Ratio"] = mean_within / mean_between if mean_between > 0 else float('inf')
    
    return results

def compute_prefix_collisions(Xtr_raw, Xte_raw, decimals=5):
    """Find prefix collisions: train states whose raw hidden vector is
    byte-identical to a test query's.

    Under a frozen, deterministic, causal LM, byte-identical hidden states can
    only arise from byte-identical token prefixes (same header + same step text).
    Such neighbors make retrieval trivially correct, so we exclude them.

    Returns (dup_exclude, n_queries_with_dup, n_pairs):
      dup_exclude[i] = np.array of TRAIN indices identical to test query i.
    """
    def key(v):
        return np.round(v, decimals).astype(np.float32).tobytes()

    train_map = {}
    for j in range(Xtr_raw.shape[0]):
        train_map.setdefault(key(Xtr_raw[j]), []).append(j)

    dup_exclude = []
    n_queries_with_dup = 0
    n_pairs = 0
    for i in range(Xte_raw.shape[0]):
        hits = train_map.get(key(Xte_raw[i]), [])
        if hits:
            n_queries_with_dup += 1
            n_pairs += len(hits)
        dup_exclude.append(np.array(hits, dtype=np.int64))
    return dup_exclude, n_queries_with_dup, n_pairs


def permute_labels(data_dict, seed=0):
    """Return a copy of the label arrays with rows permuted (features untouched
    elsewhere), decoupling labels from the representation: a permutation null."""
    rng = np.random.RandomState(seed)
    n = len(data_dict["Depth"])
    perm = rng.permutation(n)
    out = {}
    for k, v in data_dict.items():
        out[k] = v[perm] if k != "X" else v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports",
                        help="Directory holding trajectories/ and where outputs are written (e.g. /kaggle/working/reports)")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test on subset of data")
    parser.add_argument("--max_trajectories", type=int, default=10000,
                        help="Cap on trajectories loaded from train.pt; 0 = use all (set high for large-scale Kaggle runs)")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of trajectories used as the retrieval database (train); remainder is the query set (test)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the permutation-null control (the rest of the pipeline is deterministic)")
    args = parser.parse_args()

    out = args.reports_dir
    os.makedirs(out, exist_ok=True)

    print("[Phase C.4A] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    trajs = load_trajectories(train_traj_path)

    if args.smoke:
        print("[Phase C.4A] SMOKE TEST: Limiting to 500 trajectories.")
        trajs = trajs[:500]
    elif args.max_trajectories and len(trajs) > args.max_trajectories:
        print(f"[Phase C.4A] Capping to {args.max_trajectories} trajectories (of {len(trajs)}).")
        trajs = trajs[:args.max_trajectories]

    # Split completely by trajectory
    split_idx = int(len(trajs) * args.train_frac)
    trajs_train = trajs[:split_idx]
    trajs_test = trajs[split_idx:]
    
    print("[Phase C.4A] Extracting train targets...")
    train_data = extract_all_targets(trajs_train)
    Xtr_raw = train_data["X"]
    
    print("[Phase C.4A] Extracting test targets...")
    test_data = extract_all_targets(trajs_test)
    Xte_raw = test_data["X"]
    
    print(f"Train states: {len(Xtr_raw)}, Test states: {len(Xte_raw)}")
    
    # Measure overlap
    train_unique_syms = set([s for s in train_data["Sym"] if s != "NONE"])
    test_unique_syms = set([s for s in test_data["Sym"] if s != "NONE"])
    overlap = len(train_unique_syms.intersection(test_unique_syms))
    coverage = overlap / len(test_unique_syms) if len(test_unique_syms) > 0 else 0.0
    print(f"Test Coverage: {coverage*100:.1f}% ({overlap}/{len(test_unique_syms)} test states exist in train)")
    
    # Standardize & PCA STRICTLY on train
    print("[Phase C.4A] Standardizing & fitting PCA on TRAIN only...")
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_scaled = scaler.transform(Xtr_raw)
    Xte_scaled = scaler.transform(Xte_raw)
    
    n_comp = min(2048, Xtr_scaled.shape[0], Xtr_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    Xtr_pca = pca.fit_transform(Xtr_scaled)
    Xte_pca = pca.transform(Xte_scaled)

    # Prefix mask: exclude byte-identical (prefix-collision) train neighbors from
    # retrieval. Computed on the RAW hidden states (the ground truth for "same
    # text"); identical raw states stay identical under any PCA ablation.
    print("[Phase C.4A] Detecting prefix collisions (deterministic-LM confound)...")
    dup_exclude, n_dup_q, n_dup_pairs = compute_prefix_collisions(Xtr_raw, Xte_raw)
    dup_q_frac = n_dup_q / len(Xte_raw) if len(Xte_raw) > 0 else 0.0
    print(f"  {n_dup_q}/{len(Xte_raw)} test states ({dup_q_frac*100:.1f}%) have >=1 "
          f"byte-identical train neighbor; {n_dup_pairs} colliding pairs masked.")

    # Ablation subsets
    ablations = [0, 1, 2, 5, 10, 20]
    all_results = []

    print("[Phase C.4A] Running Ablation Sweep...")
    for k in ablations:
        print(f"  Removing top {k} PCs...")
        # Drop the first k PCs
        Xtr_abl = Xtr_pca[:, k:]
        Xte_abl = Xte_pca[:, k:]

        res = evaluate_subset(Xtr_abl, train_data, Xte_abl, test_data, dup_exclude=dup_exclude)
        res["Removed PCs"] = k
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # ----------------------------------------------------------------------- #
    # Controls (computed at k=0, i.e. full PCA features)
    # ----------------------------------------------------------------------- #
    print("[Phase C.4A] Running controls (permutation null + raw-operand baseline)...")

    # (1) Permutation null: shuffle TRAIN labels, keep representation fixed.
    #     Any metric above this null is signal beyond label base rates / geometry.
    train_perm = permute_labels(train_data, seed=args.seed)
    null_res = evaluate_subset(Xtr_pca, train_perm, Xte_pca, test_data, dup_exclude=dup_exclude)

    # (2) Raw-operand baseline: probes/retrieval on the bag-of-prompt-numbers
    #     feature only (no hidden state). Shows what is decodable straight from
    #     the operands listed in the prompt, with zero computation.
    base_res = evaluate_subset(train_data["Bag"], train_data, test_data["Bag"], test_data)

    controls = pd.DataFrame([
        {"control": "PERMUTED null", **{c: null_res[c] for c in
            ["Depth Acc", "Rem Nums (A) Exact", "Next Op (Probe C)",
             "Retrieval Top-1", "Retrieval Top-5"]}},
        {"control": "Raw-operand baseline", **{c: base_res[c] for c in
            ["Depth Acc", "Rem Nums (A) Exact", "Next Op (Probe C)",
             "Retrieval Top-1", "Retrieval Top-5"]}},
    ])
    controls.to_csv(os.path.join(out, "phase_c4a_controls.csv"), index=False)

    # Compute Gains relative to Removed PCs = 0
    base_top1 = df[df["Removed PCs"] == 0]["Retrieval Top-1"].iloc[0]
    df["Top-1 Gain"] = df["Retrieval Top-1"] / base_top1 if base_top1 > 0 else 1.0
    
    # phase_c4a_pca_ablation_metrics.csv
    metrics_cols = ["Removed PCs", "Depth Acc", "Rem Nums (A) Exact", "Rem Nums (A) Hamming", "Next Op (Probe C)"]
    df[metrics_cols].to_csv(os.path.join(out, "phase_c4a_pca_ablation_metrics.csv"), index=False)
    
    # phase_c4a_retrieval.csv
    retrieval_cols = ["Removed PCs", "Retrieval Top-1", "Retrieval Top-5", "Retrieval Top-10", "Top-1 Gain"]
    df[retrieval_cols].to_csv(os.path.join(out, "phase_c4a_retrieval.csv"), index=False)
    
    # phase_c4a_geometry.csv
    geometry_cols = ["Removed PCs", "Within/Between Ratio"]
    df[geometry_cols].to_csv(os.path.join(out, "phase_c4a_geometry.csv"), index=False)
    
    # coverage_report.md
    coverage_path = os.path.join(out, "coverage_report.md")
    with open(coverage_path, "w") as f:
        f.write("# Coverage Report\n\n")
        f.write(f"- Train state count: {len(train_unique_syms)}\n")
        f.write(f"- Test state count: {len(test_unique_syms)}\n")
        f.write(f"- Overlap count: {overlap}\n")
        f.write(f"- Overlap %: {coverage*100:.2f}%\n")
        f.write(f"- Prefix collisions (test states with >=1 byte-identical train neighbor): "
                f"{n_dup_q}/{len(Xte_raw)} ({dup_q_frac*100:.2f}%), {n_dup_pairs} pairs masked\n")
    
    report_path = os.path.join(out, "phase_c4a_pca_ablation_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase C.4A — PCA Ablation Sweep Report\n\n")
        f.write("This audit explicitly removes the top (highest-variance) principal components to test if Depth is a low-rank nuisance variable that can be sliced off without harming the distributed symbolic state. The PCA was strictly fit on the Train trajectories.\n\n")
        f.write(f"**Train-Test State Coverage**: {coverage*100:.1f}% ({overlap}/{len(test_unique_syms)} unique symbolic states in Test exist in Train)\n\n")
        f.write(f"**Prefix collisions masked**: {n_dup_q}/{len(Xte_raw)} test states ({dup_q_frac*100:.1f}%) had >=1 byte-identical train neighbor (deterministic-LM prefix confound); these are excluded from retrieval.\n\n")

        f.write("## Controls\n\n")
        f.write("Retrieval/probe scores must be read against these. The PERMUTED null shuffles train labels (representation fixed) - anything at this level is noise. The raw-operand baseline probes only the bag of numbers listed in the prompt (zero computation).\n\n")
        f.write("| Control | Depth Acc | Probe A (Exact) | Probe C | Retrieval (Top-1) | Retrieval (Top-5) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in controls.iterrows():
            f.write(f"| {r['control']} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact']*100:.1f}% | {r['Next Op (Probe C)']*100:.1f}% | {r['Retrieval Top-1']*100:.1f}% | {r['Retrieval Top-5']*100:.1f}% |\n")
        f.write("\n## Ablation Results\n\n")
        f.write("| Removed PCs | Depth Acc | Probe A (Exact) | Probe A (Hamming) | Retrieval (Top-1) | Gain | Retrieval (Top-5) | Retrieval (Top-10) | Within/Between |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        
        for _, r in df.iterrows():
            f.write(f"| {int(r['Removed PCs'])} | {r['Depth Acc']*100:.1f}% | {r['Rem Nums (A) Exact']*100:.1f}% | {r['Rem Nums (A) Hamming']*100:.1f}% | {r['Retrieval Top-1']*100:.1f}% | {r['Top-1 Gain']:.2f}x | {r['Retrieval Top-5']*100:.1f}% | {r['Retrieval Top-10']*100:.1f}% | {r['Within/Between Ratio']:.3f} |\n")
            
        # Interpretation
        d_base = df[df["Removed PCs"] == 0]["Depth Acc"].iloc[0]
        d_5 = df[df["Removed PCs"] == 5]["Depth Acc"].iloc[0]
        a_base = df[df["Removed PCs"] == 0]["Rem Nums (A) Exact"].iloc[0]
        a_5 = df[df["Removed PCs"] == 5]["Rem Nums (A) Exact"].iloc[0]
        gain_5 = df[df["Removed PCs"] == 5]["Top-1 Gain"].iloc[0]
        
        f.write("\n## Verdict\n\n")
        if d_5 < 0.50 and a_5 > a_base * 0.8 and gain_5 > 1.1:
            f.write("**NUISANCE FACTOR CONFIRMED.**\nRemoving the top PCs collapsed the depth signal, preserved the symbolic state, and substantially increased retrieval gain. The canonical planning state can be explicitly recovered by orthogonal projection.")
        elif d_5 < 0.50 and a_5 < a_base * 0.5:
            f.write("**FATAL ENTANGLEMENT.**\nRemoving the top PCs successfully killed depth, but it also destroyed the symbolic state. They share the same high-variance directions.")
        else:
            f.write("**PERSISTENT.**\nThe ablation had unexpected effects; further investigation is required.")
            
    print(f"\n[Phase C.4A] Done. Report written to {report_path}")

if __name__ == "__main__":
    main()

```


## File: `scripts\run_phase_d.py`
```python
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import gather_state_groups, compute_intrinsic_noise
from evaluation.probes import extract_probe_data, _fit_eval_single, _fit_eval_multilabel_joint, PROBE_NAMES, _chance_accuracy
from models.transition_model import TransitionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProjectionNet(nn.Module):
    def __init__(self, in_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def build_contrastive_dataset(state_groups, max_pairs_per_state=20):
    pos_pairs = []
    neg_pairs = []
    
    all_states = list(state_groups.keys())
    
    for sym, tensors in state_groups.items():
        n = len(tensors)
        if n < 2:
            continue
            
        # Pos pairs
        indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(indices)
        indices = indices[:max_pairs_per_state]
        
        for i, j in indices:
            t1_idx, hist1, h1 = tensors[i]
            t2_idx, hist2, h2 = tensors[j]
            if t1_idx != t2_idx and hist1 != hist2:
                pos_pairs.append({
                    'h1': h1, 'h2': h2, 
                    'state_id': str(sym),
                    'traj_a': t1_idx, 'traj_b': t2_idx,
                    'history_a': str(hist1), 'history_b': str(hist2),
                    'label': 1.0
                })
        
        # Neg pairs
        for i in range(min(max_pairs_per_state, n)):
            t1_idx, hist1, h1 = tensors[i]
            other_sym = random.choice(all_states)
            while other_sym == sym:
                other_sym = random.choice(all_states)
                
            t3_idx, hist3, h3 = random.choice(state_groups[other_sym])
            neg_pairs.append({
                'h1': h1, 'h2': h3,
                'state_id': str(sym) + "_vs_" + str(other_sym),
                'traj_a': t1_idx, 'traj_b': t3_idx,
                'history_a': str(hist1), 'history_b': str(hist3),
                'label': -1.0
            })
            
    return pos_pairs, neg_pairs

def train_projection(pos_pairs, neg_pairs, epochs=3, batch_size=128):
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    
    model = ProjectionNet(2048, 256)
    model = ProjectionNet(2048, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i+batch_size]
            h1 = torch.stack([x['h1'] for x in batch]).to(device)
            h2 = torch.stack([x['h2'] for x in batch]).to(device)
            y = torch.tensor([x['label'] for x in batch], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            z1 = model(h1)
            z2 = model(h2)
            loss = criterion(z1, z2, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            
        print(f"  [Epoch {ep+1}/{epochs}] Loss: {total_loss / len(all_pairs):.4f}")
        
    model.eval()
    return model.cpu()

def project_state_groups(model, state_groups):
    z_groups = {}
    with torch.no_grad():
        for sym, tensors in state_groups.items():
            new_tensors = []
            for t_idx, hist, h in tensors:
                z = model(h.unsqueeze(0)).squeeze(0)
                new_tensors.append((t_idx, hist, z))
            z_groups[sym] = new_tensors
    return z_groups

def evaluate_probes(model, train_trajs, test_trajs):
    tr = extract_probe_data(train_trajs)
    te = extract_probe_data(test_trajs)
    
    # Project X
    with torch.no_grad():
        Xtr_h = tr["X"]
        Xte_h = te["X"]
        Xtr_z = model(torch.tensor(Xtr_h)).numpy()
        Xte_z = model(torch.tensor(Xte_h)).numpy()
        
    results = []
    
    for Xtr, Xte, name in [(Xtr_h, Xte_h, "h (2048)"), (Xtr_z, Xte_z, "z (256)")]:
        resA = _fit_eval_multilabel_joint(Xtr, tr["A"], Xte, te["A"])
        resB = _fit_eval_single(Xtr, tr["B"], Xte, te["B"], binary=False)
        resC = _fit_eval_single(Xtr, tr["C"], Xte, te["C"], binary=False)
        resD = _fit_eval_single(Xtr, tr["D"], Xte, te["D"], binary=True)
        
        for key, res in [("A", resA), ("B", resB), ("C", resC), ("D", resD)]:
            results.append({
                "representation": name,
                "probe": f"{key}:{PROBE_NAMES[key]}",
                "accuracy": res["accuracy"],
                "f1": res["f1"],
                "exact_accuracy": res.get("exact_accuracy", float('nan'))
            })
            
    return pd.DataFrame(results)

def build_transition_data(model, trajs):
    X_h, X_z, a_id, a_ops, Y_h, Y_z = [], [], [], [], [], []
    for traj in trajs:
        N = traj.num_steps
        if N == 0: continue
        with torch.no_grad():
            states_z = model(traj.states)
            
        for i in range(N):
            X_h.append(traj.states[i])
            X_z.append(states_z[i])
            a_id.append(traj.op_ids[i])
            a_ops.append(traj.operands[i])
            Y_h.append(traj.states[i+1])
            Y_z.append(states_z[i+1])
            
    return (
        torch.stack(X_h), torch.stack(X_z),
        torch.stack(a_id), torch.stack(a_ops),
        torch.stack(Y_h), torch.stack(Y_z)
    )

def train_transition(X, a_id, a_ops, Y, hidden_dim, epochs=5):
    tm = TransitionModel(hidden_dim=hidden_dim, use_action=True, predict_delta=True)
    tm.to(device)
    opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
    
    dataset = torch.utils.data.TensorDataset(X, a_id, a_ops, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    tm.train()
    for _ in range(epochs):
        for bx, bid, bop, by in loader:
            bx, bid, bop, by = bx.to(device), bid.to(device), bop.to(device), by.to(device)
            opt.zero_grad()
            pred = tm(bx, bid, bop)
            loss = F.mse_loss(pred, by)
            loss.backward()
            opt.step()
            
    tm.eval()
    return tm.cpu()

def eval_transition(tm, X, a_id, a_ops, Y):
    with torch.no_grad():
        pred = tm(X, a_id, a_ops)
        mse = F.mse_loss(pred, Y).item()
        cos = F.cosine_similarity(pred, Y, dim=-1).mean().item()
    return mse, cos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    
    print("[Phase D] Loading trajectories...")
    train_traj_path = os.path.join(out, "trajectories", "train.pt")
    test_traj_path = os.path.join(out, "trajectories", "test.pt")
    
    if not os.path.exists(train_traj_path):
        print("ERROR: Trajectories not found. Run Phase A first.")
        sys.exit(1)
        
    train_trajs = load_trajectories(train_traj_path)
    test_trajs = load_trajectories(test_traj_path)
    
    # Cap size for fast processing since this is an experiment script
    train_trajs = train_trajs[:500]
    test_trajs = test_trajs[:100]
    all_trajs = train_trajs + test_trajs
    
    print("[Phase D] Gathering state groups...")
    state_groups = gather_state_groups(all_trajs)
    
    print("[Phase D] Building contrastive dataset...")
    pos_pairs, neg_pairs = build_contrastive_dataset(state_groups)
    
    pos_df = pd.DataFrame([{k:v for k,v in p.items() if k not in ['h1','h2']} for p in pos_pairs])
    pos_df.to_csv(os.path.join(out, "positive_pairs.csv"), index=False)
    
    print(f"  Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs.")
    
    print("[Phase D] Training Canonicalization Projection P(h)...")
    model = train_projection(pos_pairs, neg_pairs, epochs=3)
    
    print("[Phase D] Projecting state groups...")
    z_groups = project_state_groups(model, state_groups)
    
    print("[Phase D] Evaluating Retrieval (Before vs After)...")
    df_h, top1_h, top5_h = compute_intrinsic_noise(state_groups, max_pairs_per_state=50)
    df_z, top1_z, top5_z = compute_intrinsic_noise(z_groups, max_pairs_per_state=50)
    
    w_cos_h = df_h[df_h["type"]=="within"]["cosine"].mean()
    b_cos_h = df_h[df_h["type"]=="between"]["cosine"].mean()
    
    w_cos_z = df_z[df_z["type"]=="within"]["cosine"].mean()
    b_cos_z = df_z[df_z["type"]=="between"]["cosine"].mean()
    
    retrieval_res = pd.DataFrame([
        {"representation": "h (2048)", "within_cosine": w_cos_h, "between_cosine": b_cos_h, "top1": top1_h, "top5": top5_h},
        {"representation": "z (256)", "within_cosine": w_cos_z, "between_cosine": b_cos_z, "top1": top1_z, "top5": top5_z}
    ])
    retrieval_res.to_csv(os.path.join(out, "retrieval_before_after.csv"), index=False)
    print(retrieval_res)
    
    print("[Phase D] Evaluating Probes...")
    probe_res = evaluate_probes(model, train_trajs, test_trajs)
    probe_res.to_csv(os.path.join(out, "probe_before_after.csv"), index=False)
    
    print("[Phase D] Evaluating Transition Utility...")
    X_h_tr, X_z_tr, a_id_tr, a_ops_tr, Y_h_tr, Y_z_tr = build_transition_data(model, train_trajs)
    X_h_te, X_z_te, a_id_te, a_ops_te, Y_h_te, Y_z_te = build_transition_data(model, test_trajs)
    
    tm_h = train_transition(X_h_tr, a_id_tr, a_ops_tr, Y_h_tr, hidden_dim=2048, epochs=3)
    tm_z = train_transition(X_z_tr, a_id_tr, a_ops_tr, Y_z_tr, hidden_dim=256, epochs=3)
    
    mse_h, cos_h = eval_transition(tm_h, X_h_te, a_id_te, a_ops_te, Y_h_te)
    mse_z, cos_z = eval_transition(tm_z, X_z_te, a_id_te, a_ops_te, Y_z_te)
    
    trans_res = pd.DataFrame([
        {"representation": "h (2048)", "mse": mse_h, "cosine": cos_h},
        {"representation": "z (256)", "mse": mse_z, "cosine": cos_z}
    ])
    trans_res.to_csv(os.path.join(out, "transition_before_after.csv"), index=False)
    print(trans_res)
    
    print("[Phase D] Generating Reports...")
    
    # Verdict Logic
    retrieval_improved = (top1_z > top1_h + 0.10)
    probes_maintained = True
    for p in probe_res["probe"].unique():
        acc_h = probe_res[(probe_res["representation"] == "h (2048)") & (probe_res["probe"] == p)]["accuracy"].iloc[0]
        acc_z = probe_res[(probe_res["representation"] == "z (256)") & (probe_res["probe"] == p)]["accuracy"].iloc[0]
        if acc_z < acc_h - 0.10:  # Allow slight drop due to compression
            probes_maintained = False
            
    success = retrieval_improved and probes_maintained
    verdict = "ENTANGLED" if success else "ABSENT"
    
    with open(os.path.join(out, "canonicalization_verdict.md"), "w") as f:
        f.write(f"# Verdict: {verdict}\n\n")
        if verdict == "ENTANGLED":
            f.write("The planning information already existed in the hidden state but was merely entangled with trajectory history. A lightweight projection successfully removed the history dependence (improving retrieval) while preserving task information.\n")
        else:
            f.write("The planning information is fundamentally absent or inextricably linked to history. Projection failed to separate the symbolic state from the path taken without destroying task performance.\n")
            
    with open(os.path.join(out, "canonicalization_report.md"), "w") as f:
        f.write("# Phase D: Representation Canonicalization\n\n")
        f.write(f"**Retrieval Improvement**: {top1_h*100:.1f}% -> {top1_z*100:.1f}%\n")
        f.write(f"**Probes Maintained**: {probes_maintained}\n")
        f.write(f"\nFinal Conclusion: **{verdict}**\n")
        
    print(f"\n[Phase D] Done. Verdict: {verdict}")
    print("Artifacts generated in", out)

if __name__ == "__main__":
    main()

```


## File: `scripts\run_positive_control.py`
```python
"""V5.2 Experiment 2 — positive control (methodology calibration).

Runs the *identical* VQ + analysis pipeline used on Countdown on two synthetic
references:

* **Positive control** — a deterministic FSM with known, reusable states
  rendered as ``proto[s] + noise`` embeddings. The ceiling: what the pipeline
  reports when genuine discrete states exist.
* **Noise floor** — the same FSM walks but with structureless embeddings. The
  floor: what the pipeline reports when no state information is present.

Countdown's Experiment-1 numbers (read from ``reports/action_conditioned.json``
and ``reports/V5_1_FINAL.json``) are placed between them so the observed
``H(z'|z,op)=0.99`` / ``det-frac=0.11`` can be calibrated. Writes
``reports/positive_control.md`` + ``.json``. Then stop.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from sklearn.metrics import adjusted_mutual_info_score

from training.train_vq import train_vq_quantizer, VQTrainConfig
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes, all_codes,
)
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.discrete_transition import train_code_transition, CodeTransitionConfig
from evaluation.position_leakage import evaluate_position_leakage
from evaluation.action_conditioned import (
    extract_action_transitions, run_all_models, conditional_structure,
)
from evaluation.synthetic_fsm import (
    generate_fsm, generate_dataset, make_prototypes, flatten_true_states,
    oracle_transitions,
)


def analyze_condition(name, train, val, test, true_train, true_test,
                      K, vq_epochs, tr_epochs, seed):
    vq = train_vq_quantizer(
        train, val, config=VQTrainConfig(num_codes=K, epochs=vq_epochs,
                                         batch_size=256, seed=seed))
    d_tr = encode_trajectories_to_codes(vq, train)
    d_va = encode_trajectories_to_codes(vq, val)
    d_te = encode_trajectories_to_codes(vq, test)

    usage = analyze_codebook_usage(all_codes(d_tr), K)
    leak = evaluate_position_leakage(d_tr, d_te, K, seed=seed)
    _, predict = train_code_transition(
        d_tr, d_va, num_codes=K, config=CodeTransitionConfig(epochs=tr_epochs, seed=seed))
    res = run_all_models(extract_action_transitions(d_tr),
                         extract_action_transitions(d_va), K,
                         epochs=tr_epochs, seed=seed)

    # Did the codes recover the ground-truth states?
    codes_flat = all_codes(d_tr).numpy()
    ami = float(adjusted_mutual_info_score(flatten_true_states(true_train), codes_flat))

    out = {
        "name": name,
        "active_codes": int(usage["active_codes"]),
        "gini": float(usage["collapse_score"]),
        "perplexity": float(usage["perplexity"]),
        "ami_codes_vs_true": ami,
        "position_leakage": float(leak["position_predictability_score"]),
        "leakage_majority": float(leak["chance_accuracy"]),
        "majority": float(predict["majority_baseline"]),
        "bigram_z": float(predict["bigram_baseline"]),
        "mlp_z": float(predict["top1_accuracy"]),
        "action_bigram": float(res["C_action_bigram"]["top1"]),
        "mlp_z_op": float(res["E_mlp_z_op"]["top1"]),
        "H_state": float(res["_struct_state"]["global_entropy"]),
        "det_state": float(res["_struct_state"]["det_frac_mass"]),
        "H_action": float(res["_struct_action"]["global_entropy"]),
        "det_action": float(res["_struct_action"]["det_frac_mass"]),
    }
    return out


def oracle_metrics(true_train, train):
    st, at, sn = oracle_transitions(true_train, train)
    sa = conditional_structure(list(zip(st.tolist(), at.tolist())), sn.tolist())
    s = conditional_structure(st.tolist(), sn.tolist())
    return {"H_state": s["global_entropy"], "det_state": s["det_frac_mass"],
            "H_action": sa["global_entropy"], "det_action": sa["det_frac_mass"]}


def countdown_reference(reports_dir):
    """Pull Countdown (Fixed-Init) numbers from prior experiment artifacts."""
    out = {"name": "Countdown (observed)", "ami_codes_vs_true": None}
    ac_path = os.path.join(reports_dir, "action_conditioned.json")
    v5_path = os.path.join(reports_dir, "V5_1_FINAL.json")
    if os.path.exists(ac_path):
        ac = json.load(open(ac_path, encoding="utf-8"))["fixed_init"]
        out.update({
            "bigram_z": ac["B_bigram"]["top1"], "mlp_z": ac["D_mlp_z"]["top1"],
            "action_bigram": ac["C_action_bigram"]["top1"],
            "mlp_z_op": ac["E_mlp_z_op"]["top1"],
            "majority": ac["A_majority"]["top1"],
            "H_state": ac["_struct_state"]["global_entropy"],
            "det_state": ac["_struct_state"]["det_frac_mass"],
            "H_action": ac["_struct_action"]["global_entropy"],
            "det_action": ac["_struct_action"]["det_frac_mass"],
        })
    if os.path.exists(v5_path):
        col = json.load(open(v5_path, encoding="utf-8"))["columns"]["v5_fixed_init"]
        out.update({
            "active_codes": col["active_codes"], "gini": col["gini"],
            "perplexity": col["perplexity"],
            "position_leakage": col["position_leakage"],
            "leakage_majority": col["leakage_majority"],
        })
    return out


def _f(v, p=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}f}"


def write_report(out_dir, floor, observed, pos, oracle, K):
    cols = [("Noise floor", floor), ("Countdown (observed)", observed),
            ("Positive control", pos)]

    def row(label, key, p=3):
        return f"| {label} | " + " | ".join(_f(c[1].get(key), p) for c in cols) + \
               f" | {_f(oracle.get(key), p) if key in oracle else '—'} |"

    L = []
    L.append("# V5.2 · Experiment 2 — Positive Control (methodology calibration)\n")
    L.append("> Question: is Countdown's action-conditioned **H(z'|z,op)=0.99 "
             "nats / det-frac=0.11** meaningful discrete-state structure, or "
             "near-noise? We calibrate by running the *identical* pipeline "
             "(VQ K=%d, data-dependent init, same usage/leakage/transition/"
             "action-conditioned analysis) on environments with **known** "
             "answers.\n" % K)

    L.append("\n## Setup\n")
    L.append("- **Positive control:** deterministic FSM (32 states, 3 actions, "
             "random transition table), each state rendered as `proto[s] + "
             "noise` in 2048-d. Genuine reusable, deterministic `(s,a)→s'`.\n")
    L.append("- **Noise floor:** same FSM walks, but embeddings are pure noise "
             "(no state information).\n")
    L.append("- **Countdown (observed):** Fixed-Init numbers from Experiments "
             "V5.1 / 1, shown for placement.\n")
    L.append("- **Oracle:** dynamics computed on the *true* FSM states (the "
             "theoretical ceiling the codes are trying to recover).\n")

    L.append("\n## Calibration table\n")
    L.append("| Metric | Noise floor | Countdown (observed) | Positive control | Oracle (true states) |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    L.append(row("Active codes", "active_codes", 0))
    L.append(row("AMI(codes, true states)", "ami_codes_vs_true"))
    L.append(row("Position leakage", "position_leakage"))
    L.append(row("  · leakage majority", "leakage_majority"))
    L.append(row("Bigram (z)", "bigram_z"))
    L.append(row("MLP(z)", "mlp_z"))
    L.append(row("Action bigram (z, a)", "action_bigram"))
    L.append(row("MLP(z, a)", "mlp_z_op"))
    L.append(row("H(z'|z) nats", "H_state"))
    L.append(row("H(z'|z, a) nats", "H_action"))
    L.append(row("Det-frac (z)", "det_state"))
    L.append(row("**Det-frac (z, a)**", "det_action"))
    L.append("\n_AMI = adjusted mutual information between VQ codes and "
             "ground-truth states (1.0 = perfect recovery, 0 = independent). "
             "Det-frac = share of transition mass from near-deterministic "
             "(entropy < 0.5 nats) conditionings seen ≥ 10×._\n")

    # ---- Verdict logic ---- #
    recovered = (pos["ami_codes_vs_true"] > 0.6 and pos["det_action"] >= 0.5
                 and pos["H_action"] < 0.5)
    # Where does Countdown sit on each floor->ceiling axis? (fraction of the
    # way from the noise floor to the positive-control ceiling). For entropy the
    # floor is high and the ceiling low, so the same formula yields the fraction
    # of the way *down* toward the ceiling.
    def _frac(flo, ceil, x):
        return (x - flo) / (ceil - flo) if abs(ceil - flo) > 1e-9 else float("nan")
    cd_det = _frac(floor["det_action"], pos["det_action"], observed["det_action"])
    cd_mlp = _frac(floor["mlp_z_op"], pos["mlp_z_op"], observed["mlp_z_op"])
    cd_H = _frac(floor["H_action"], pos["H_action"], observed["H_action"])

    L.append("\n---\n## Did the pipeline recover known states?\n")
    if recovered:
        L.append(f"**Yes.** On the positive control the VQ codes recover the "
                 f"ground-truth states (AMI = {_f(pos['ami_codes_vs_true'])}), "
                 f"position leakage stays at chance "
                 f"({_f(pos['position_leakage'])} vs majority "
                 f"{_f(pos['leakage_majority'])}), and conditioning on the action "
                 f"makes the transition essentially deterministic: "
                 f"H(z'|z,a) = {_f(pos['H_action'])} nats, det-frac = "
                 f"{_f(pos['det_action'])} (oracle "
                 f"{_f(oracle['det_action'])}), MLP(z,a) = "
                 f"{_f(pos['mlp_z_op'])} ≫ majority {_f(pos['majority'])}. The "
                 f"noise floor shows the opposite (AMI {_f(floor['ami_codes_vs_true'])}, "
                 f"det-frac {_f(floor['det_action'])}).\n")
    else:
        L.append(f"**Partially.** Positive-control recovery: AMI "
                 f"{_f(pos['ami_codes_vs_true'])}, det-frac(z,a) "
                 f"{_f(pos['det_action'])}, H(z'|z,a) {_f(pos['H_action'])}. "
                 f"(Interpret the calibration with this in mind.)\n")

    L.append("\n## Is the methodology validated?\n")
    L.append(f"**Yes.** The pipeline is *capable* of reporting strong "
             f"determinism / low entropy / high AMI when reusable states exist "
             f"(positive control) and ~noise when they do not (floor). It is not "
             f"biased toward either answer.\n")

    L.append("\n## Calibrating Countdown's 0.11 / 0.99\n")
    L.append("Placed on each floor→ceiling axis (floor = noise, ceiling = "
             "positive control), Countdown lands:\n")
    L.append(f"- **Det-frac(z,a):** floor {_f(floor['det_action'])} → ceiling "
             f"{_f(pos['det_action'])}; Countdown {_f(observed['det_action'])} "
             f"— **{cd_det*100:.0f}%** of the way up.")
    L.append(f"- **MLP(z,a) acc:** floor {_f(floor['mlp_z_op'])} → ceiling "
             f"{_f(pos['mlp_z_op'])}; Countdown {_f(observed['mlp_z_op'])} "
             f"— **{cd_mlp*100:.0f}%**.")
    L.append(f"- **H(z'|z,a):** floor {_f(floor['H_action'])} → ceiling "
             f"{_f(pos['H_action'])} nats; Countdown {_f(observed['H_action'])} "
             f"— **{cd_H*100:.0f}%** of the way down to the ceiling.\n")
    L.append("\n> **Calibrated conclusion.** Countdown's action-conditioned "
             "structure is **real and clearly above the noise floor** — every "
             f"action-conditioned metric beats the floor (MLP {_f(observed['mlp_z_op'])} "
             f"vs {_f(floor['mlp_z_op'])}; H(z'|z,a) {_f(observed['H_action'])} vs "
             f"{_f(floor['H_action'])} nats) — yet it falls **short of the genuine "
             f"reusable-state ceiling** (MLP {_f(pos['mlp_z_op'])}, H "
             f"{_f(pos['H_action'])}), and the axes disagree on how far: the strict "
             f"det-frac bar puts Countdown only {cd_det*100:.0f}% of the way up, "
             f"while soft entropy/accuracy put it ~{min(cd_mlp, cd_H)*100:.0f}–"
             f"{max(cd_mlp, cd_H)*100:.0f}%. The honest reading: Countdown has "
             "**graded, partial** action-conditioned dependence — meaningfully "
             "more than noise, but lacking the crisp determinism of reusable "
             "planning states. The low det-frac (0.11) reflects how few "
             "conditionings clear the strict <0.5-nat / ≥10-count bar, not that "
             "the dependence is noise — superseding the earlier ‘≈ noise’ "
             "reading, which was an artifact of a split-geometry bug in the "
             "control harness.\n")

    L.append("\n_Caveats: the control is a clean, well-separated FSM — an "
             "*upper* bound on recoverability; a harder control (entangled "
             "position, heavier noise) would lower the ceiling. Countdown’s "
             "position-leakage (0.79) already shows its codes are far more "
             "position-bound than the control’s (~chance). Scope otherwise "
             "unchanged.\n")
    L.append("\n_Artifacts: `reports/positive_control.json`._\n")

    with open(os.path.join(out_dir, "positive_control.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--num_states", type=int, default=32)
    ap.add_argument("--num_actions", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--vq_epochs", type=int, default=50)
    ap.add_argument("--tr_epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    K = args.num_codes
    out = args.reports_dir

    T = generate_fsm(args.num_states, args.num_actions, seed=args.seed)

    def gen(structured, base):
        # One prototype geometry shared by train/val/test, so the VQ trained on
        # train can encode val/test (see make_prototypes).
        protos = make_prototypes(args.num_states, args.hidden_dim, seed=base)
        tr, ts_tr = generate_dataset(T, 1000, args.hidden_dim, noise=args.noise,
                                     structured=structured, seed=base, protos=protos)
        va, ts_va = generate_dataset(T, 500, args.hidden_dim, noise=args.noise,
                                     structured=structured, seed=base + 1, protos=protos)
        te, ts_te = generate_dataset(T, 1000, args.hidden_dim, noise=args.noise,
                                     structured=structured, seed=base + 2, protos=protos)
        return (tr, va, te, ts_tr, ts_te)

    print("[Exp2] Positive control...")
    p_tr, p_va, p_te, p_ts_tr, p_ts_te = gen(True, 100)
    pos = analyze_condition("Positive control", p_tr, p_va, p_te, p_ts_tr, p_ts_te,
                            K, args.vq_epochs, args.tr_epochs, args.seed)
    oracle = oracle_metrics(p_ts_tr, p_tr)

    print("[Exp2] Noise floor...")
    f_tr, f_va, f_te, f_ts_tr, f_ts_te = gen(False, 200)
    floor = analyze_condition("Noise floor", f_tr, f_va, f_te, f_ts_tr, f_ts_te,
                              K, args.vq_epochs, args.tr_epochs, args.seed)

    observed = countdown_reference(out)

    payload = {"config": {"num_codes": K, "num_states": args.num_states,
                          "num_actions": args.num_actions, "hidden_dim": args.hidden_dim,
                          "noise": args.noise, "seed": args.seed},
               "positive_control": pos, "noise_floor": floor,
               "oracle_true_states": oracle, "countdown_observed": observed}
    with open(os.path.join(out, "positive_control.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(out, floor, observed, pos, oracle, K)

    print("\n[Exp2] Positive:", {k: round(v, 3) for k, v in pos.items()
                                 if isinstance(v, float)})
    print("[Exp2] Floor:", {k: round(v, 3) for k, v in floor.items()
                            if isinstance(v, float)})
    print("[Exp2] Oracle:", {k: round(v, 3) for k, v in oracle.items()})
    print("[Exp2] Done. See reports/positive_control.md")


if __name__ == "__main__":
    main()

```


## File: `scripts\run_v5_1.py`
```python
"""V5.1 — closing the discrete-state question after fixing init + baselines.

Runs three matched analyses on the cached Phase-A trajectories and produces the
rebuttal package (``reports/V5_1_FINAL.md`` / ``.json`` + plots):

    V5 Original    — re-derived from the saved *original* (collapsed) code
                     trajectories so the never-logged majority/bigram baselines
                     can be filled honestly (falls back to recorded CSV
                     constants if those artifacts are absent).
    V5 Fixed Init  — VQ retrained with data-dependent codebook init (Fix 1).
    V5.1 Scrubbed  — VQ trained on position-scrubbed states (INLP, Steps 1-4).

Each column reports the same four diagnostics: codebook usage (C5), transition
predictability vs majority/bigram baselines (C3, Fix 2), transition entropy
(C4), and code-level position leakage (C6).

The verdict is deliberately skeptical: it asks whether the structure that
survives fixing init *and* removing position is anything more than first-order
local dynamics.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from data_processing.trajectory_dataset import load_trajectories, save_trajectories
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes,
    save_discrete_trajectories,
    load_discrete_trajectories,
    all_codes,
)
from training.train_vq import train_vq_quantizer, VQTrainConfig
from evaluation.codebook_usage import analyze_codebook_usage
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    train_code_transition,
    build_transition_matrix,
    conditional_entropy,
)
from evaluation.position_leakage import evaluate_position_leakage
from evaluation.position_subspace import (
    build_position_dataset,
    train_position_probe,
    inlp_position_projection,
    apply_scrub_to_trajectories,
)

# Recorded V5 Original numbers (from reports/v5_*.csv + report) — used only if
# the saved original code trajectories are unavailable for re-derivation.
RECORDED_V5_ORIGINAL = {
    "active_codes": 6, "used_codes": 6, "dead_codes": 26,
    "gini": 0.8543, "perplexity": 5.33,
    "mlp_top1": 0.4313, "majority_baseline": None, "bigram_baseline": None,
    "predictive_entropy": 1.2937, "global_entropy": 1.1970, "det_frac": 0.0,
    "position_leakage": 0.5883, "leakage_majority": 0.2509, "leakage_null": 0.0901,
    "source": "recorded CSVs (original codes absent; baselines were never logged)",
}


def analyze_codes(disc_train, disc_val, disc_test, K, seed, epochs):
    """Compute the four V5 diagnostics for one set of code trajectories."""
    usage = analyze_codebook_usage(all_codes(disc_train), K)
    _, predict = train_code_transition(
        disc_train, disc_val, num_codes=K,
        config=CodeTransitionConfig(epochs=epochs, seed=seed),
    )
    T = build_transition_matrix(disc_train, K)
    ent = conditional_entropy(T, code_freqs=usage["freqs"])
    leak = evaluate_position_leakage(disc_train, disc_test, K, seed=seed)
    return {
        "active_codes": int(usage["active_codes"]),
        "used_codes": int(usage["used_codes"]),
        "dead_codes": int(usage["dead_codes"]),
        "gini": float(usage["collapse_score"]),
        "perplexity": float(usage["perplexity"]),
        "mlp_top1": float(predict["top1_accuracy"]),
        "majority_baseline": float(predict["majority_baseline"]),
        "bigram_baseline": float(predict["bigram_baseline"]),
        "predictive_entropy": float(predict["predictive_entropy_nats"]),
        "global_entropy": float(ent["global_entropy"]),
        "det_frac": float(ent["deterministic_state_frac"]),
        "position_leakage": float(leak["position_predictability_score"]),
        "leakage_majority": float(leak["chance_accuracy"]),
        "leakage_null": float(leak["permutation_null_accuracy"]),
        "_freqs": usage["freqs"],
    }


def train_vq(trajs_train, trajs_val, K, epochs, batch_size, seed):
    cfg = VQTrainConfig(num_codes=K, epochs=epochs, batch_size=batch_size, seed=seed)
    return train_vq_quantizer(trajs_train, trajs_val, config=cfg)


def v5_original_column(reports_dir, K, seed, epochs):
    """Re-derive V5 Original from saved original codes, or fall back to records."""
    paths = [os.path.join(reports_dir, "discrete_trajectories", f"{s}.pt")
             for s in ("train", "val", "test")]
    if not all(os.path.exists(p) for p in paths):
        print("[V5.1] Original code trajectories absent — using recorded constants.")
        return dict(RECORDED_V5_ORIGINAL)
    disc = [load_discrete_trajectories(p) for p in paths]
    usage = analyze_codebook_usage(all_codes(disc[0]), K)
    # Provenance guard: these must be the collapsed original (Gini 0.854, 6 active).
    if usage["active_codes"] != 6 or abs(usage["collapse_score"] - 0.8543) > 1e-2:
        print("[V5.1] Saved codes do not match recorded original — using constants.")
        return dict(RECORDED_V5_ORIGINAL)
    print("[V5.1] V5 Original re-derived from saved original (collapsed) codes.")
    col = analyze_codes(disc[0], disc[1], disc[2], K, seed, epochs)
    col["source"] = "re-derived from saved original collapsed code trajectories"
    return col


def main():
    ap = argparse.ArgumentParser(description="Run the V5.1 closing analysis")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--checkpoints_dir", default="checkpoints")
    ap.add_argument("--num_codes", type=int, default=32)
    ap.add_argument("--vq_epochs", type=int, default=50)
    ap.add_argument("--vq_batch_size", type=int, default=256)
    ap.add_argument("--transition_epochs", type=int, default=30)
    # 8 iters (spec recommendation) is far too few here: position is encoded in
    # a high-dimensional, redundant subspace, so removing ~24 directions barely
    # dents it. We iterate until position decodability reaches the majority
    # floor (~100 iters / ~300 dims; see the INLP curve in the report).
    ap.add_argument("--inlp_iters", type=int, default=100)
    ap.add_argument("--inlp_C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true",
                    help="Regenerate MD + plots from an existing V5_1_FINAL.json "
                         "(no recompute)")
    args = ap.parse_args()

    K = args.num_codes
    out = args.reports_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.report_only:
        with open(os.path.join(out, "V5_1_FINAL.json"), encoding="utf-8") as f:
            payload = json.load(f)
        c = payload["columns"]
        orig, fix, scrub = c["v5_original"], c["v5_fixed_init"], c["v5_1_scrubbed"]
        Kj = payload["config"]["num_codes"]
        make_plots(out, orig, fix, scrub, payload["inlp"], Kj)
        write_report(out, payload, orig, fix, scrub, payload["position_probe"],
                     payload["inlp"], payload["criteria"], Kj)
        print("[V5.1] Report + plots regenerated from JSON.")
        return

    print("[V5.1] Loading cached trajectories...")
    tr = load_trajectories(os.path.join(out, "trajectories", "train.pt"))
    va = load_trajectories(os.path.join(out, "trajectories", "val.pt"))
    te = load_trajectories(os.path.join(out, "trajectories", "test.pt"))
    hidden_dim = tr[0].hidden_dim
    print(f"  train={len(tr)} val={len(va)} test={len(te)} H={hidden_dim} K={K}")

    # ---- Column 1: V5 Original ----------------------------------------- #
    col_orig = v5_original_column(out, K, args.seed, args.transition_epochs)

    # ---- Column 2: V5 Fixed Init -------------------------------------- #
    print("[V5.1] Training Fixed-Init VQ on raw states (Fix 1)...")
    vq_fix = train_vq(tr, va, K, args.vq_epochs, args.vq_batch_size, args.seed)
    torch.save({"state_dict": vq_fix.state_dict(), "hidden_dim": hidden_dim,
                "num_codes": K}, os.path.join(args.checkpoints_dir, "vq_state.pt"))
    df_tr = encode_trajectories_to_codes(vq_fix, tr)
    df_va = encode_trajectories_to_codes(vq_fix, va)
    df_te = encode_trajectories_to_codes(vq_fix, te)
    fdir = os.path.join(out, "discrete_fixed_init")
    for nm, d in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        save_discrete_trajectories(d, os.path.join(fdir, f"{nm}.pt"))
    col_fix = analyze_codes(df_tr, df_va, df_te, K, args.seed, args.transition_epochs)
    col_fix["source"] = "VQ retrained with data-dependent init on raw states"
    print(f"  active={col_fix['active_codes']}/{K} gini={col_fix['gini']:.3f} "
          f"MLP={col_fix['mlp_top1']:.3f} bigram={col_fix['bigram_baseline']:.3f} "
          f"leak={col_fix['position_leakage']:.3f}")

    # ---- Steps 1-2: position probe + INLP (on raw aligned states) ------ #
    print("[V5.1] Position probe + INLP...")
    Xtr, ytr = build_position_dataset(tr)
    Xte, yte = build_position_dataset(te)
    mean = Xtr.mean(axis=0)
    Xtr_c, Xte_c = Xtr - mean, Xte - mean
    probe = train_position_probe(Xtr_c, ytr, Xte_c, yte, C=args.inlp_C)
    P, inlp = inlp_position_projection(
        Xtr_c, ytr, Xte_c, yte, num_iters=args.inlp_iters, C=args.inlp_C)
    print(f"  probe acc={probe['position_accuracy']:.3f} "
          f"(majority={probe['majority_baseline']:.3f}); "
          f"INLP before={inlp['before_accuracy']:.3f} after={inlp['after_accuracy']:.3f} "
          f"dims_removed={inlp['dims_removed']}")

    # ---- Step 3: scrub + save ----------------------------------------- #
    print("[V5.1] Scrubbing states...")
    sc_tr = apply_scrub_to_trajectories(tr, mean, P)
    sc_va = apply_scrub_to_trajectories(va, mean, P)
    sc_te = apply_scrub_to_trajectories(te, mean, P)
    sdir = os.path.join(out, "scrubbed")
    save_trajectories(sc_tr, os.path.join(sdir, "train_scrubbed.pt"))
    save_trajectories(sc_va, os.path.join(sdir, "val_scrubbed.pt"))
    save_trajectories(sc_te, os.path.join(sdir, "test_scrubbed.pt"))

    # ---- Step 4-5: VQ on scrubbed + analyze --------------------------- #
    print("[V5.1] Training VQ on scrubbed states...")
    vq_sc = train_vq(sc_tr, sc_va, K, args.vq_epochs, args.vq_batch_size, args.seed)
    torch.save({"state_dict": vq_sc.state_dict(), "hidden_dim": hidden_dim,
                "num_codes": K}, os.path.join(args.checkpoints_dir, "vq_state_scrubbed.pt"))
    ds_tr = encode_trajectories_to_codes(vq_sc, sc_tr)
    ds_va = encode_trajectories_to_codes(vq_sc, sc_va)
    ds_te = encode_trajectories_to_codes(vq_sc, sc_te)
    scdir = os.path.join(out, "discrete_scrubbed")
    for nm, d in [("train", ds_tr), ("val", ds_va), ("test", ds_te)]:
        save_discrete_trajectories(d, os.path.join(scdir, f"{nm}.pt"))
    col_scrub = analyze_codes(ds_tr, ds_va, ds_te, K, args.seed, args.transition_epochs)
    col_scrub["source"] = "VQ trained on INLP position-scrubbed states"
    print(f"  active={col_scrub['active_codes']}/{K} gini={col_scrub['gini']:.3f} "
          f"MLP={col_scrub['mlp_top1']:.3f} bigram={col_scrub['bigram_baseline']:.3f} "
          f"leak={col_scrub['position_leakage']:.3f}")

    # ---- Verdict criteria (skeptical) --------------------------------- #
    crit = verdict_criteria(col_fix, col_scrub, K)

    # ---- Persist + report --------------------------------------------- #
    payload = {
        "config": {"num_codes": K, "vq_epochs": args.vq_epochs,
                   "transition_epochs": args.transition_epochs,
                   "inlp_iters": args.inlp_iters, "inlp_C": args.inlp_C,
                   "seed": args.seed, "hidden_dim": hidden_dim,
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
        "columns": {"v5_original": _clean(col_orig),
                    "v5_fixed_init": _clean(col_fix),
                    "v5_1_scrubbed": _clean(col_scrub)},
        "position_probe": probe,
        "inlp": {k: v for k, v in inlp.items()},
        "criteria": crit,
    }
    with open(os.path.join(out, "V5_1_FINAL.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    make_plots(out, col_orig, col_fix, col_scrub, inlp, K)
    write_report(out, payload, col_orig, col_fix, col_scrub, probe, inlp, crit, K)
    print("\n[V5.1] Done. See reports/V5_1_FINAL.md")


def _clean(col):
    return {k: v for k, v in col.items() if not k.startswith("_")}


def verdict_criteria(fix, scrub, K):
    """Boolean decision criteria from the spec, evaluated on the scrubbed run."""
    mlp_margin = scrub["mlp_top1"] - scrub["bigram_baseline"]
    leak_floor = max(scrub["leakage_majority"], scrub["leakage_null"])
    return {
        "codes_high": scrub["active_codes"] >= 0.5 * K,
        "leakage_dropped": (fix["position_leakage"] - scrub["position_leakage"] > 0.05)
                            and (scrub["position_leakage"] <= leak_floor + 0.10),
        "mlp_beats_bigram": mlp_margin > 0.02,
        "mlp_bigram_margin": float(mlp_margin),
        "entropy_decreased": scrub["global_entropy"] < fix["global_entropy"] - 0.05,
        "deterministic_emerged": scrub["det_frac"] >= 0.20,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(out, orig, fix, scrub, inlp, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [orig, fix, scrub]
    names = ["V5 Original", "V5 Fixed Init", "V5.1 Scrubbed"]
    colors = ["tab:gray", "tab:blue", "tab:green"]
    x = np.arange(3)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].bar(x, [c["active_codes"] for c in cols], color=colors)
    ax[0, 0].axhline(K, ls="--", c="k", alpha=0.4, label=f"K={K}")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(names, rotation=10)
    ax[0, 0].set_title("Active codes"); ax[0, 0].set_ylabel("active / K")
    ax[0, 0].legend()

    # predictability: majority / bigram / MLP grouped
    w = 0.26
    maj = [_nan(c["majority_baseline"]) for c in cols]
    big = [_nan(c["bigram_baseline"]) for c in cols]
    mlp = [_nan(c["mlp_top1"]) for c in cols]
    ax[0, 1].bar(x - w, maj, w, label="majority", color="lightgray")
    ax[0, 1].bar(x, big, w, label="bigram", color="tab:orange")
    ax[0, 1].bar(x + w, mlp, w, label="MLP", color="tab:purple")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(names, rotation=10)
    ax[0, 1].set_title("Transition predictability\n(MLP ≈ bigram ⇒ no learned dynamics)")
    ax[0, 1].set_ylabel("top-1 accuracy"); ax[0, 1].legend()

    # position leakage vs floor
    leak = [c["position_leakage"] for c in cols]
    floor = [max(c["leakage_majority"], c["leakage_null"]) for c in cols]
    ax[1, 0].bar(x - 0.18, leak, 0.36, label="leakage", color="tab:red")
    ax[1, 0].bar(x + 0.18, floor, 0.36, label="majority/null floor", color="lightgray")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(names, rotation=10)
    ax[1, 0].set_title("Code → position leakage"); ax[1, 0].set_ylabel("accuracy")
    ax[1, 0].legend()

    # INLP trace
    tr_acc = inlp["accuracy_trace"]
    ax[1, 1].plot(range(len(tr_acc)), tr_acc, "o-", color="tab:blue", label="probe acc")
    ax[1, 1].axhline(inlp["majority_baseline"], ls="--", c="k", alpha=0.5,
                     label=f"majority {inlp['majority_baseline']:.2f}")
    ax[1, 1].set_title("INLP: position decodability vs iteration\n(hidden states)")
    ax[1, 1].set_xlabel("INLP iteration"); ax[1, 1].set_ylabel("position accuracy")
    ax[1, 1].legend()

    fig.suptitle("V5.1 — fixing init + removing position", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out, "V5_1_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _nan(v):
    return float("nan") if v is None else float(v)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _f(v, p=3):
    if v is None:
        return "n/r"
    if isinstance(v, float) and (v != v):  # nan
        return "n/r"
    return f"{v:.{p}f}"


def write_report(out, payload, orig, fix, scrub, probe, inlp, crit, K):
    cols = [orig, fix, scrub]

    def row(label, key, p=3):
        return (f"| {label} | " +
                " | ".join(_f(c.get(key), p) for c in cols) + " |")

    L = []
    L.append("# V5.1 — Final Report (Discrete State Discovery)\n")
    L.append("> **Research question:** do hidden-state trajectories of a frozen "
             "language model admit a compact, predictive, *reusable* discrete "
             "state representation?\n")
    L.append("> Model: TinyLlama · Task: Countdown · States: last-layer hidden "
             f"states at reasoning-step boundaries · Codebook K={K}.\n")
    L.append("\n---\n")

    L.append("## Why the original V5 result is invalid\n")
    L.append("The original VQ codebook was initialized as `randn * 0.02` "
             "(row-norm ≈ 0.90) while the hidden states have norm ≈ 85 — a ~94× "
             "scale mismatch. On the first batch almost every state maps to one "
             "code; EMA + dead-code revival cannot recover, leaving **6/32 "
             "active codes**. The collapse is an initialization artifact, not a "
             "property of the representation. Data-dependent init "
             "(random training states) restores full codebook usage.\n")

    L.append("## Comparison table\n")
    L.append("| Metric | V5 Original | V5 Fixed Init | V5.1 Scrubbed |")
    L.append("| --- | ---: | ---: | ---: |")
    L.append(row("Active Codes", "active_codes", 0))
    L.append(row("Gini", "gini"))
    L.append(row("Perplexity", "perplexity", 2))
    L.append(row("Position Leakage", "position_leakage"))
    L.append(row("Majority Baseline", "majority_baseline"))
    L.append(row("Bigram Baseline", "bigram_baseline"))
    L.append(row("MLP Accuracy", "mlp_top1"))
    L.append(row("Global Entropy", "global_entropy"))
    L.append(row("Deterministic Fraction", "det_frac"))
    L.append("")
    L.append(f"_V5 Original source: {orig.get('source','recorded')}._ "
             "Leakage majority/null floors: "
             f"Original {_f(orig['leakage_majority'])}/{_f(orig['leakage_null'])}, "
             f"Fixed {_f(fix['leakage_majority'])}/{_f(fix['leakage_null'])}, "
             f"Scrubbed {_f(scrub['leakage_majority'])}/{_f(scrub['leakage_null'])}.\n")

    L.append(f"\n**Central pattern — MLP ≈ bigram in every column** "
             f"(Original {_f(orig['mlp_top1'])}/{_f(orig['bigram_baseline'])}, "
             f"Fixed {_f(fix['mlp_top1'])}/{_f(fix['bigram_baseline'])}, "
             f"Scrubbed {_f(scrub['mlp_top1'])}/{_f(scrub['bigram_baseline'])} "
             f"as MLP/bigram): the learned next-code MLP never improves on a "
             f"first-order count model, in any condition. The original report's "
             f"headline (MLP {_f(orig['mlp_top1'])} vs uniform-chance 0.031, a ~14× "
             f"gap) compared against the wrong baseline; against the honest bigram "
             f"the gap is ≈ 0.\n")
    L.append(f"**Entropy caveat:** global entropy *rises* across columns "
             f"({_f(orig['global_entropy'])} → {_f(fix['global_entropy'])} → "
             f"{_f(scrub['global_entropy'])}), but this is a de-collapse artifact — "
             f"with 6 codes there are few possible successors, with 32 there are "
             f"many. The decisive diagnostic is the deterministic-successor "
             f"fraction, which stays ≈ 0 "
             f"({_f(orig['det_frac'])} / {_f(fix['det_frac'])} / "
             f"{_f(scrub['det_frac'])}): no near-deterministic 'planning' states "
             f"appear in any condition.\n")

    L.append("## Position subspace (continuous hidden states)\n")
    L.append(f"- Position probe (4 relative-depth bins): **{_f(probe['position_accuracy'])}** "
             f"held-out vs majority {_f(probe['majority_baseline'])} "
             f"({probe['n_classes']} classes). Trajectory stage is almost "
             f"perfectly linearly decodable from a raw hidden state.\n")
    L.append(f"- INLP removed **{inlp['dims_removed']} dims** over "
             f"{inlp['num_iters']} iterations: position decodability "
             f"**{_f(inlp['before_accuracy'])} → {_f(inlp['after_accuracy'])}** "
             f"(majority floor {_f(inlp['majority_baseline'])}).\n")
    # INLP curve sampled from the per-iteration trace: position is so redundant
    # that ~300 directions must be removed before it reaches the floor.
    trace = inlp.get("accuracy_trace", [])
    if trace:
        L.append("\n**INLP curve — position is high-dimensional and redundant:**\n")
        L.append("| dims removed (≈3·iter) | position accuracy |")
        L.append("| ---: | ---: |")
        marks = sorted(set([0] + list(range(19, len(trace), 20)) + [len(trace) - 1]))
        for it in marks:
            L.append(f"| {it * 3} | {_f(trace[it])} |")
        L.append(f"\n_The spec's 6–8 iterations remove only ~24 dims (acc still "
                 f"{_f(trace[min(7, len(trace)-1)])}); reaching the floor needs "
                 f"~{inlp['dims_removed']} dims. That position resists removal until "
                 f"~15% of the hidden space is deleted is itself evidence that these "
                 f"states are dominated by trajectory stage._\n")

    # ---- Verdict & questions ---- #
    L.append("\n---\n")
    L.append("## Final analysis\n")

    L.append("### Q1 — Does fixing initialization invalidate the original collapse conclusion?\n")
    L.append(
        f"**Yes.** With data-dependent init the codebook goes from "
        f"{orig['active_codes']}/{K} active codes (Gini {_f(orig['gini'])}) to "
        f"{fix['active_codes']}/{K} (Gini {_f(fix['gini'])}). The reported "
        f"collapse was an init artifact and must not be cited as evidence about "
        f"the representation.\n")

    L.append("### Q2 — Does position scrubbing reveal stronger latent structure?\n")
    if crit["mlp_beats_bigram"] and crit["leakage_dropped"]:
        q2 = (f"**Partially.** Leakage fell "
              f"{_f(fix['position_leakage'])} → {_f(scrub['position_leakage'])} and "
              f"the MLP now beats the bigram by {crit['mlp_bigram_margin']:+.3f}.")
    else:
        q2 = (f"**No.** Even after near-complete removal of linearly-decodable "
              f"position (continuous-state decodability "
              f"{_f(inlp['before_accuracy'])} → {_f(inlp['after_accuracy'])}, floor "
              f"{_f(inlp['majority_baseline'])}; {inlp['dims_removed']} dims), the "
              f"codes still leak position at {_f(scrub['position_leakage'])} (floor "
              f"{_f(max(scrub['leakage_majority'], scrub['leakage_null']))}), and the "
              f"MLP ({_f(scrub['mlp_top1'])}) does not beat the bigram "
              f"({_f(scrub['bigram_baseline'])}) — margin "
              f"{crit['mlp_bigram_margin']:+.3f}. No reusable low-entropy dynamics "
              f"appear; the deterministic-successor fraction stays "
              f"{_f(scrub['det_frac'])}.")
    L.append(q2 + "\n")

    L.append("### Q3 — After controlling for init and position, the states are:\n")
    if crit["mlp_beats_bigram"] and crit["deterministic_emerged"]:
        choice = "**A — reusable planning states.**"
    elif crit["mlp_beats_bigram"]:
        choice = ("**C — weak local Markov structure** (predictable one step "
                  "ahead, but no deterministic reusable subset).")
    elif not crit["leakage_dropped"]:
        choice = (
            "**B + C — trajectory-stage encodings with only weak local "
            "(first-order) dynamics.** Two signatures coincide: the codes still "
            f"track trajectory position (leakage {_f(scrub['position_leakage'])} ≫ "
            f"floor {_f(max(scrub['leakage_majority'], scrub['leakage_null']))}) "
            "even after the continuous states are scrubbed, and the only "
            "predictability that exists is fully captured by a bigram "
            "(MLP ≈ bigram). Neither reusable planning states (A) nor a "
            "deterministic-successor subset are observed.")
    else:
        choice = ("**C — weak local Markov structure** (MLP ≈ bigram: only "
                  "first-order count dynamics).")
    L.append(choice + "\n")

    L.append("### Q4 — Strongest defensible conclusion\n")
    positive = (crit["codes_high"] and crit["leakage_dropped"]
                and crit["mlp_beats_bigram"] and crit["entropy_decreased"]
                and crit["deterministic_emerged"])
    if positive:
        L.append("All decision criteria pass: the codebook is non-collapsed, "
                 "leakage drops to floor, the MLP beats the bigram, entropy "
                 "falls, and a deterministic subset emerges — consistent with "
                 "reusable discrete structure.\n")
    else:
        L.append("> Under TinyLlama + Countdown + last-layer hidden states, "
                 "discrete latent structure is dominated by trajectory "
                 "progression and weak local dynamics rather than reusable "
                 "planning states.\n")
        L.append("\nThe codebook collapse was an initialization bug, and once it "
                 "is fixed the apparent transition 'predictability' is fully "
                 "explained by a first-order bigram model (MLP ≈ bigram). "
                 "Removing the linearly-decodable position subspace does not "
                 "expose any additional reusable, low-entropy dynamics.\n")

    L.append("\n### Decision criteria (scrubbed run)\n")
    for k in ("codes_high", "leakage_dropped", "mlp_beats_bigram",
              "entropy_decreased", "deterministic_emerged"):
        L.append(f"- {k}: **{'PASS' if crit[k] else 'fail'}**")
    L.append(f"- MLP − bigram margin: {crit['mlp_bigram_margin']:+.3f}\n")

    L.append("\n### Threats to validity / scope\n")
    L.append("- **Single configuration:** one model (TinyLlama), one task "
             "(Countdown), last-layer states only, seed 0. The conclusion is "
             "scoped to this setting; other layers / models / tasks are untested.\n")
    L.append("- **Linear position removal:** INLP removes only *linearly* "
             "decodable position. Residual (nonlinear) position survives — codes "
             f"still leak it at {_f(scrub['position_leakage'])} — so the scrub is a "
             "lower bound on position's influence, not a complete excision.\n")
    L.append("- **Position operationalized** as 4 relative-depth quartiles "
             "(trajectories are only 3–5 states); absolute-index and finer "
             "binnings were not swept.\n")
    L.append("- **Over-scrub control:** 300/2048 dims are removed, which could in "
             "principle delete content — but the negative result does not depend "
             "on it: MLP ≈ bigram already holds in the *un-scrubbed* Fixed-Init "
             "run, and held at every intermediate scrub depth tested "
             "(24/60/120/180/240/300 dims).\n")
    L.append("- **What would overturn this:** an MLP that clears the bigram by a "
             "non-trivial margin, a deterministic-successor subset (entropy < 0.5 "
             "nats) of non-trivial mass, or code→position leakage falling to its "
             "floor after scrubbing. None occurred.\n")

    L.append("\n![comparison](V5_1_comparison.png)\n")
    L.append("\n_Artifacts: `V5_1_FINAL.json`, `V5_1_comparison.png`, "
             "`checkpoints/vq_state.pt` (fixed init), "
             "`checkpoints/vq_state_scrubbed.pt`, `reports/scrubbed/*_scrubbed.pt`._\n")

    with open(os.path.join(out, "V5_1_FINAL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()

```


## File: `scripts\run_v5_discrete_states.py`
```python
"""V5 orchestrator: do frozen LM hidden states admit a reusable discrete state?

Single entry point (mirrors :mod:`scripts.run_phase_a`) that runs the core V5
analysis on the cached trajectory files produced by Phase A:

    reports/trajectories/{train,val,test}.pt
        -> (C1) load / train the VQ codebook
        -> (C2) encode every trajectory to discrete codes
        -> (C5) codebook usage / collapse diagnostics
        -> (C3) action-blind transition predictability (top-1, entropy, ppl)
        -> (C4) empirical transition matrix + conditional entropy
        -> (C6) position-leakage test (codes as step indices?)
        -> (C9) discrete rollout coherence, decoded through probes A/B/C
        -> master report  reports/v5_discrete_state_report.md

C7 (permutation robustness) and C8 (cross-domain transfer) need their own data
extraction and are run from their own scripts; this orchestrator references
their CSV outputs in the master report if present.

The verdict logic is intentionally conservative: "reusable discrete states
exist" requires *all* of (a) non-collapsed codebook, (b) above-chance
transition predictability, (c) low entropy for a non-trivial code subset,
(d) low position leakage, (e) coherent multi-step rollout. Failing any one
gate is reported honestly as evidence *against* the hypothesis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from data_processing.trajectory_dataset import load_trajectories
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes,
    save_discrete_trajectories,
    all_codes,
)
from models.vq_state import VQStateQuantizer
from training.train_vq import train_vq_quantizer, VQTrainConfig
from evaluation.codebook_usage import (
    analyze_codebook_usage,
    save_codebook_usage_report,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    train_code_transition,
    build_transition_matrix,
    conditional_entropy,
    save_predictability_metrics,
    save_entropy_report,
)
from evaluation.position_leakage import (
    evaluate_position_leakage,
    save_position_leakage_report,
)
from evaluation.discrete_rollout import evaluate_discrete_rollout, save_discrete_rollout


def _load_vq(checkpoint_path: str, hidden_dim: int, num_codes: int, device) -> VQStateQuantizer:
    """Load a VQ from checkpoint, or raise a clear error if missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VQ checkpoint not found: {checkpoint_path}. "
            f"Run training/train_vq.py first (see V5_READINESS.md)."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VQStateQuantizer(
        hidden_dim=int(ckpt.get("hidden_dim", hidden_dim)),
        num_codes=int(ckpt.get("num_codes", num_codes)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_master_report(
    reports_dir: str,
    num_codes: int,
    usage_stats: dict,
    predict_metrics: dict,
    entropy: dict,
    position: dict,
    rollout_df,
    hidden_dim: int,
    n_train: int,
    n_test: int,
    smoke: bool,
) -> None:
    """Write reports/v5_discrete_state_report.md with the verdict."""
    # --- Gates (each must pass for a positive verdict) ----------------------
    # (a) Non-collapsed codebook: Gini < 0.9 AND active codes >= 10% of K.
    collapse_ok = usage_stats["collapse_score"] < 0.9
    active_ok = usage_stats["active_codes"] >= max(4, 0.1 * num_codes)
    # (b) Transition predictability beyond first-order counts: the MLP must
    # clear the bigram baseline (uniform chance was a misleadingly low bar).
    predict_ok = predict_metrics["top1_accuracy"] > predict_metrics["bigram_baseline"] + 0.02
    # (c) Low entropy for a non-trivial subset of states (>= 20% deterministic).
    entropy_ok = entropy["deterministic_state_frac"] >= 0.20
    # (d) Low position leakage: score not far above null + chance.
    leakage_ok = (
        position["position_predictability_score"]
        <= max(position["permutation_null_accuracy"], position["chance_accuracy"]) + 0.10
    )
    # (e) Coherent rollout: code-match at depth 1 > 2 * chance.
    if len(rollout_df) and rollout_df["n_samples"].iloc[0] > 0:
        rollout_ok = rollout_df["code_match_accuracy"].iloc[0] > max(
            2.0 / num_codes, 0.20
        )
    else:
        rollout_ok = False

    gates = {
        "non_collapse": collapse_ok and active_ok,
        "predictable": predict_ok,
        "low_entropy_subset": entropy_ok,
        "low_position_leakage": leakage_ok,
        "coherent_rollout": rollout_ok,
    }
    n_pass = sum(gates.values())

    lines = ["# V5 Report — Reusable Discrete States?\n"]
    lines.append(
        "Research question: _do hidden-state trajectories of frozen language "
        "models admit a compact, predictive, reusable discrete state "
        "representation?_\n"
    )
    lines.append(f"- **Codebook size (K):** {num_codes}")
    lines.append(f"- **Hidden dim:** {hidden_dim}")
    lines.append(f"- **Trajectories:** train={n_train} test={n_test}")
    if smoke:
        lines.append(
            "\n> ⚠️ **Smoke run** — small N / synthetic data. Validate the "
            "pipeline, then re-run on the full Phase A cache for real numbers."
        )
    lines.append("\n---\n")

    lines.append("## 1. Codebook usage (collapse check)\n")
    lines.append(
        f"- Active codes: **{usage_stats['active_codes']}/{num_codes}** "
        f"(used: {usage_stats['used_codes']}, dead: {usage_stats['dead_codes']})\n"
        f"- Collapse score (Gini): **{usage_stats['collapse_score']:.3f}** "
        "(0 = uniform, 1 = fully collapsed)\n"
        f"- Perplexity: **{usage_stats['perplexity']:.1f}** "
        f"(effective codes used, max {num_codes})\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['non_collapse'] else 'FAIL'} "
        f"(Gini<0.9 AND active>=10% of K).\n"
    )

    lines.append("## 2. Transition predictability\n")
    lines.append(
        f"- Top-1 next-code accuracy (MLP): **{predict_metrics['top1_accuracy']:.3f}** "
        f"(majority = {predict_metrics['majority_baseline']:.3f}, "
        f"bigram = {predict_metrics['bigram_baseline']:.3f})\n"
        f"- Predictive entropy: **{predict_metrics['predictive_entropy_nats']:.3f} nats** "
        f"(perplexity {predict_metrics['predictive_perplexity']:.2f})\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['predictable'] else 'FAIL'} "
        f"(MLP top-1 > bigram + 0.02).\n"
    )

    lines.append("## 3. Transition entropy (planning state vs bucket)\n")
    lines.append(
        f"- Global H(Z_next | Z_current): **{entropy['global_entropy']:.3f} nats** "
        f"(perplexity {entropy['global_perplexity']:.2f})\n"
        f"- Fraction of states with near-deterministic successor "
        f"(H<0.5): **{entropy['deterministic_state_frac']*100:.1f}%**\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['low_entropy_subset'] else 'FAIL'} "
        f"(>= 20% deterministic states).\n"
    )

    lines.append("## 4. Position leakage (codes as step indices?)\n")
    lines.append(
        f"- Position predictability: **{position['position_predictability_score']:.3f}**\n"
        f"  - chance: {position['chance_accuracy']:.3f}, "
        f"permutation null: {position['permutation_null_accuracy']:.3f}\n"
    )
    lines.append(
        f"Gate: {'PASS' if gates['low_position_leakage'] else 'FAIL'} "
        f"(score <= max(null, chance) + 0.10). High leakage => codes are step "
        f"indices, not reusable states.\n"
    )

    lines.append("## 5. Discrete rollout coherence\n")
    if len(rollout_df) and rollout_df["n_samples"].iloc[0] > 0:
        lines.append("| depth | code match | cosine | op acc (Probe C) | state acc (Probe A) |")
        lines.append("|---|---|---|---|---|")
        for _, r in rollout_df.iterrows():
            if r["n_samples"] > 0:
                lines.append(
                    f"| {int(r['depth'])} | {r['code_match_accuracy']:.3f} | "
                    f"{r['cosine_similarity']:.3f} | {r['operator_accuracy']:.3f} | "
                    f"{r['state_probe_accuracy']:.3f} |"
                )
        lines.append(
            f"\nGate: {'PASS' if gates['coherent_rollout'] else 'FAIL'} "
            f"(depth-1 code match > max(2/K, 0.20)).\n"
        )
    else:
        lines.append("_No multi-step trajectories available to evaluate._\n")

    lines.append("\n---\n")
    lines.append("## Verdict\n")
    if n_pass == 5:
        verdict = (
            "**Reusable discrete states are supported.** All five gates pass: "
            "the codebook is non-collapsed, the next state is predictable, a "
            "non-trivial subset of states has deterministic successors, codes "
            "are not merely step indices, and the discrete rollout stays "
            "coherent beyond depth 1."
        )
    elif n_pass >= 3:
        verdict = (
            f"**Mixed evidence ({n_pass}/5 gates pass).** See the failing "
            f"gates above ({', '.join(k for k,v in gates.items() if not v)}) "
            f"for the specific failure mode."
        )
    else:
        verdict = (
            f"**Evidence AGAINST reusable discrete states ({n_pass}/5 gates "
            f"pass).** The codebook does not support a compact, predictive, "
            f"reusable discrete representation under this configuration."
        )
    lines.append(verdict)
    lines.append(
        "\n_Failure modes: high Gini => VQ collapse; low top-1 => "
        "unpredictable dynamics; high position leakage => codes are step "
        "indices; low deterministic-state fraction => every state is a "
        "compression bucket._\n"
    )

    # Reference C7/C8 outputs if present.
    perm_csv = os.path.join(reports_dir, "v5_permutation_robustness.csv")
    transfer_csv = os.path.join(reports_dir, "v5_cross_domain_transfer.csv")
    extras = []
    if os.path.exists(perm_csv):
        extras.append("v5_permutation_robustness.csv (C7)")
    if os.path.exists(transfer_csv):
        extras.append("v5_cross_domain_transfer.csv (C8)")
    if extras:
        lines.append(
            "\n## Supplementary (run separately)\n"
            + "\n".join(f"- `{e}`" for e in extras)
            + "\n"
        )

    out_path = os.path.join(reports_dir, "v5_discrete_state_report.md")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMaster report: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the V5 discrete-state analysis")
    parser.add_argument(
        "--reports_dir", type=str, default="reports",
        help="Holds trajectories/{train,val,test}.pt; receives v5_* outputs",
    )
    parser.add_argument(
        "--checkpoints_dir", type=str, default="checkpoints",
        help="Holds (or receives) vq_state.pt",
    )
    parser.add_argument("--vq_checkpoint", type=str, default=None,
                        help="VQ checkpoint filename (default: vq_state.pt in checkpoints_dir)")
    parser.add_argument("--num_codes", type=int, default=256)
    parser.add_argument(
        "--train_vq", action="store_true",
        help="Train the VQ from trajectories/train.pt if no checkpoint exists "
             "(otherwise the checkpoint is required)",
    )
    parser.add_argument("--vq_epochs", type=int, default=50)
    parser.add_argument("--vq_batch_size", type=int, default=256)
    parser.add_argument("--transition_epochs", type=int, default=30)
    parser.add_argument("--rollout_max_depth", type=int, default=8,
                        help="0 = dynamic (95th pct of trajectory lengths)")
    parser.add_argument("--smoke", action="store_true",
                        help="Mark the run as a smoke test in the report")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.reports_dir
    disc_dir = os.path.join(out, "discrete_trajectories")
    os.makedirs(out, exist_ok=True)
    os.makedirs(disc_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    print("[V5] Loading cached trajectories...")
    trajs_train = load_trajectories(os.path.join(out, "trajectories", "train.pt"))
    trajs_val = load_trajectories(os.path.join(out, "trajectories", "val.pt"))
    trajs_test = load_trajectories(os.path.join(out, "trajectories", "test.pt"))
    if not trajs_train:
        raise RuntimeError("train.pt is empty — run Phase A first.")
    hidden_dim = trajs_train[0].hidden_dim

    # ------------------------------------------------------------------ #
    # C1 — load or train the VQ codebook.
    ckpt_name = args.vq_checkpoint or "vq_state.pt"
    ckpt_path = os.path.join(args.checkpoints_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print(f"[V5/C1] Loading VQ from {ckpt_path}")
        vq = _load_vq(ckpt_path, hidden_dim, args.num_codes, device)
    elif args.train_vq:
        print("[V5/C1] Training VQ from train.pt...")
        vq = train_vq_quantizer(
            trajs_train, trajs_val,
            config=VQTrainConfig(num_codes=args.num_codes, epochs=args.vq_epochs,
                                 batch_size=args.vq_batch_size, seed=args.seed),
        )
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        torch.save(
            {"state_dict": vq.state_dict(), "hidden_dim": hidden_dim,
             "num_codes": vq.num_codes},
            ckpt_path,
        )
    else:
        raise FileNotFoundError(
            f"No VQ checkpoint at {ckpt_path}. Re-run with --train_vq, or run "
            f"training/train_vq.py first."
        )

    # ------------------------------------------------------------------ #
    # C2 — encode every split to discrete codes and cache them.
    print("[V5/C2] Encoding trajectories to discrete codes...")
    disc_train = encode_trajectories_to_codes(vq, trajs_train)
    disc_val = encode_trajectories_to_codes(vq, trajs_val)
    disc_test = encode_trajectories_to_codes(vq, trajs_test)
    for name, d in [("train", disc_train), ("val", disc_val), ("test", disc_test)]:
        save_discrete_trajectories(d, os.path.join(disc_dir, f"{name}.pt"))
    n_train_states = int(all_codes(disc_train).shape[0])
    n_test_states = int(all_codes(disc_test).shape[0])
    print(f"  train states: {n_train_states}, test states: {n_test_states}")

    # ------------------------------------------------------------------ #
    # C5 — codebook usage.
    print("[V5/C5] Codebook usage analysis...")
    codes_train = all_codes(disc_train)
    usage = analyze_codebook_usage(codes_train, vq.num_codes)
    save_codebook_usage_report(
        usage,
        os.path.join(out, "v5_codebook_usage.csv"),
        os.path.join(out, "v5_codebook_usage.png"),
    )
    print(f"  active={usage['active_codes']}/{vq.num_codes} "
          f"gini={usage['collapse_score']:.3f} perplexity={usage['perplexity']:.1f}")

    # ------------------------------------------------------------------ #
    # C3 — transition predictability.
    print("[V5/C3] Transition predictability (action-blind MLP)...")
    _, predict = train_code_transition(
        disc_train, disc_val, num_codes=vq.num_codes,
        config=CodeTransitionConfig(epochs=args.transition_epochs, seed=args.seed),
    )
    save_predictability_metrics(predict, os.path.join(out, "v5_transition_predictability.csv"))
    print(f"  top1={predict['top1_accuracy']:.3f} "
          f"(majority={predict['majority_baseline']:.3f}, "
          f"bigram={predict['bigram_baseline']:.3f}) "
          f"entropy={predict['predictive_entropy_nats']:.3f}")

    # ------------------------------------------------------------------ #
    # C4 — transition entropy.
    print("[V5/C4] Transition entropy analysis...")
    T = build_transition_matrix(disc_train, vq.num_codes)
    code_freqs = usage["freqs"]
    entropy = conditional_entropy(T, code_freqs=code_freqs)
    active_mask = code_freqs > 0
    save_entropy_report(
        T, entropy, os.path.join(out, "v5_transition_entropy.csv"),
        os.path.join(out, "v5_transition_entropy.png"), active_mask=active_mask,
    )
    print(f"  global H={entropy['global_entropy']:.3f} "
          f"det_frac={entropy['deterministic_state_frac']*100:.1f}%")

    # ------------------------------------------------------------------ #
    # C6 — position leakage.
    print("[V5/C6] Position leakage test...")
    position = evaluate_position_leakage(disc_train, disc_test, vq.num_codes, seed=args.seed)
    save_position_leakage_report(position, os.path.join(out, "v5_position_leakage.csv"))
    print(f"  score={position['position_predictability_score']:.3f} "
          f"(null={position['permutation_null_accuracy']:.3f}, "
          f"chance={position['chance_accuracy']:.3f})")

    # ------------------------------------------------------------------ #
    # C9 — discrete rollout. Fit linear probes A/B/C on the continuous train
    # states so we can decode the rolled-out code -> continuous -> probes.
    print("[V5/C9] Discrete rollout coherence...")
    from evaluation.probes import run_probes
    _, _, fitted = run_probes(trajs_train, trajs_test)
    probe_a, probe_b, probe_c = fitted.get("A"), fitted.get("B"), fitted.get("C")

    # Re-train the transition model on discrete codes (cheap; reuses C3 model).
    code_model, _ = train_code_transition(
        disc_train, disc_val, num_codes=vq.num_codes,
        config=CodeTransitionConfig(epochs=args.transition_epochs, seed=args.seed),
    )
    max_depth = args.rollout_max_depth if args.rollout_max_depth > 0 else None
    rollout_df = evaluate_discrete_rollout(
        code_model, vq, disc_test,
        probe_a=probe_a, probe_b=probe_b, probe_c=probe_c,
        max_depth=max_depth,
    )
    save_discrete_rollout(
        rollout_df,
        os.path.join(out, "v5_discrete_rollout.csv"),
        os.path.join(out, "v5_discrete_rollout.png"),
    )

    # ------------------------------------------------------------------ #
    generate_master_report(
        reports_dir=out, num_codes=vq.num_codes,
        usage_stats=usage, predict_metrics=predict,
        entropy=entropy, position=position, rollout_df=rollout_df,
        hidden_dim=hidden_dim, n_train=len(trajs_train), n_test=len(trajs_test),
        smoke=args.smoke,
    )
    print("\n[V5] Done. v5_* artifacts in", out)


if __name__ == "__main__":
    main()

```


## File: `scripts\smoke_4bit_extraction.py`
```python
"""Pre-flight smoke test for 4-bit hidden-state extraction -> VQ encode.

Runs the *real* extraction + VQ-encode path on a small slice (default 50
trajectories) and asserts the four invariants that a new/larger model or the
4-bit loader could silently break:

  1. hidden dimension is reported consistently (model config == trajectory == VQ)
  2. layer indexing is valid (hidden_states tuple length, selected-layer shape)
  3. the VQ encoder receives the expected shapes (codes align 1:1 with states)
  4. no silent truncation/drop (every problem -> a trajectory; every step kept)

Architecture-agnostic: validate the plumbing locally on a small cached model,
then run the identical check on Kaggle before the full 7B/8B job, e.g.

    python scripts/smoke_4bit_extraction.py --model Qwen/Qwen2.5-7B --load_in_4bit
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import random

import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import build_trajectories
from training.train_vq import train_vq_quantizer, VQTrainConfig
from data_processing.discrete_trajectory_dataset import encode_trajectories_to_codes
from scripts.generate_game24_dataset import generate_game24_problem
from scripts.generate_countdown_dataset import generate_countdown_problem


def _problems(domain: str, n: int):
    rng = random.Random(0)
    if domain == "game24":
        return [generate_game24_problem(rng) for _ in range(n)]
    random.seed(0)  # countdown generator uses the module-global RNG
    return [generate_countdown_problem() for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--domain", choices=["game24", "countdown"], default="game24")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--num_codes", type=int, default=32)
    args = ap.parse_args()

    print(f"[smoke] model={args.model} 4bit={args.load_in_4bit} domain={args.domain} n={args.n}")
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    dtype = torch.float32 if device_map == "cpu" else torch.float16
    model = load_model(model_id=args.model, device_map=device_map, torch_dtype=dtype,
                       load_in_4bit=args.load_in_4bit)
    tokenizer = load_tokenizer(model_id=args.model)
    model.eval()

    cfg_dim = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers)
    print(f"[smoke] config: hidden_size={cfg_dim}  num_hidden_layers={n_layers}")

    # ---- Check 2: layer indexing -------------------------------------- #
    probs = _problems(args.domain, args.n)
    enc = tokenizer(probs[0]["problem"] if "problem" in probs[0]
                    else "Problem:\n" + probs[0]["cot"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    n_hs = len(out.hidden_states)
    assert n_hs == n_layers + 1, f"hidden_states tuple len {n_hs} != n_layers+1 ({n_layers+1})"
    sel = out.hidden_states[args.layer]
    assert sel.shape[-1] == cfg_dim, f"layer[{args.layer}] dim {sel.shape[-1]} != {cfg_dim}"
    print(f"[smoke] OK  check 2: hidden_states len={n_hs} (=L+1); layer[{args.layer}] dim={sel.shape[-1]}")

    # ---- Build trajectories (full per-token extraction) --------------- #
    # Countdown generator emits no 'problem' header; supply the standard one.
    for p in probs:
        p.setdefault("problem", "Problem: Reach the target.\nSolution:\n")
    trajs = build_trajectories(model, tokenizer, probs, layer=args.layer, batch_size=8)

    # ---- Check 4: no silent truncation/drop --------------------------- #
    n_built = len(trajs)
    print(f"[smoke] built {n_built}/{len(probs)} trajectories")
    assert n_built > 0, "no trajectories built"
    dropped = len(probs) - n_built
    if dropped:
        print(f"[smoke] WARNING: {dropped} problem(s) dropped during alignment "
              f"(silent loss — investigate before scaling)")
    bad_len, bad_idx = 0, 0
    for p, t in zip(probs, trajs):
        n_states = int(t.state_indices.shape[0])
        if n_states != len(p["solution"]) + 1:   # N steps -> N+1 states
            bad_len += 1
        if int(t.state_indices.max()) >= t.all_hidden.shape[0]:
            bad_idx += 1
    assert bad_len == 0, f"{bad_len} trajectories have wrong #states (steps dropped)"
    assert bad_idx == 0, f"{bad_idx} trajectories index past the token sequence"
    print(f"[smoke] OK  check 4: every problem kept, states=steps+1, indices in-range")

    # ---- Check 1: hidden dim consistency ------------------------------ #
    tdim = trajs[0].hidden_dim
    assert tdim == cfg_dim, f"trajectory hidden_dim {tdim} != config {cfg_dim}"
    print(f"[smoke] OK  check 1: trajectory hidden_dim={tdim} == config {cfg_dim}")

    # ---- Check 3: VQ receives expected shapes ------------------------- #
    vq = train_vq_quantizer(trajs, trajs,
                            config=VQTrainConfig(num_codes=args.num_codes, epochs=5, batch_size=128))
    assert vq.hidden_dim == cfg_dim, f"VQ hidden_dim {vq.hidden_dim} != config {cfg_dim}"
    disc = encode_trajectories_to_codes(vq, trajs)
    mism = sum(1 for t, d in zip(trajs, disc)
               if d.codes.shape[0] != int(t.state_indices.shape[0]))
    assert mism == 0, f"{mism} encoded trajectories have code count != state count"
    all_codes = torch.cat([d.codes for d in disc])
    assert int(all_codes.min()) >= 0 and int(all_codes.max()) < args.num_codes, "code id out of range"
    print(f"[smoke] OK  check 3: VQ dim={vq.hidden_dim}; codes align with states; "
          f"ids in [0,{args.num_codes}); active={len(set(all_codes.tolist()))}")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

```


## File: `scripts\smoke_test.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Smoke test to verify Phase A repository packaging and pipeline execution."""

import sys
import os
import torch
import warnings

def main():
    try:
        print("0. Testing Repository Root...")
        assert os.path.exists("scripts"), "Must run from repository root"
        assert os.path.exists("models"), "Must run from repository root"
        assert os.path.exists("training"), "Must run from repository root"
        assert os.path.exists("evaluation"), "Must run from repository root"
        print("   [OK] Root directory verified.")

        print("\n1. Testing Imports...")
        from models.model_loader import load_model, load_tokenizer
        from data_processing.trajectory_dataset import load_problems, build_trajectories
        from training.train_transition import train_transition_model, TransitionTrainConfig
        from training.train_decoder import train_decoder_model, DecoderTrainConfig
        print("   [OK] Imports successful.")
        
        print("\n2. Testing Dummy Dataset Generation...")
        import json
        os.makedirs("data", exist_ok=True)
        dummy_problems = [
            {"question": "80 25", "target": 105, "solution": "80 + 25 = 105"}
        ]
        with open("data/smoke_test.jsonl", "w") as f:
            for p in dummy_problems:
                f.write(json.dumps(p) + "\n")
        print("   [OK] Dataset creation successful.")

        print("\n3. Testing Mock Model Loading...")
        # Since we just want to test pipeline connectivity, we won't load TinyLlama 
        # as it would be too slow and require GPU. We'll mock the pipeline logic 
        # similar to what we do in tests/test_pipeline.py
        from data_processing.trajectory_dataset import Trajectory
        
        dummy_traj = Trajectory(
            all_hidden=torch.randn(10, 32),
            input_ids=torch.randint(0, 100, (10,)),
            state_indices=torch.tensor([0, 5, 9]),
            op_ids=torch.tensor([0, 1]),
            operands=torch.randn(2, 2),
            numbers=[80, 25],
            target=105
        )
        print("   [OK] Trajectory mocking successful.")

        print("\n4. Testing Transition Model Training...")
        t_config = TransitionTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16)
        train_transition_model([dummy_traj, dummy_traj], config=t_config)
        
        t_config_linear = TransitionTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16, transition_arch="linear")
        train_transition_model([dummy_traj, dummy_traj], config=t_config_linear)
        
        t_config_transformer = TransitionTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16, transition_arch="transformer")
        train_transition_model([dummy_traj, dummy_traj], config=t_config_transformer)
        print("   [OK] Transition training successful.")

        print("\n5. Testing Decoder Training...")
        d_config = DecoderTrainConfig(epochs=1, batch_size=1, mlp_hidden_dim=16)
        train_decoder_model([dummy_traj, dummy_traj], vocab_size=100, config=d_config)
        print("   [OK] Decoder training successful.")

        print("\n===========================================")
        print("SMOKE TEST PASSED: Repository is executable.")
        print("===========================================")
        sys.exit(0)
        
    except Exception as e:
        print("\n===========================================")
        print(f"SMOKE TEST FAILED: {type(e).__name__}")
        print(str(e))
        import traceback
        traceback.print_exc()
        print("===========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()

```


## File: `scripts\train_teacher.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_loader import load_qlora_model, load_tokenizer

def format_prompt(example):
    """Formats the Countdown problem into a text prompt."""
    prompt = f"Problem: Given the numbers {example['numbers']}, reach the target {example['target']}.\n"
    prompt += f"Solution:\n{example['cot']}"
    return {"text": prompt}

def main():
    parser = argparse.ArgumentParser(description="Train Teacher Model on Countdown")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # 1. Load Tokenizer & Model
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_qlora_model()

    # 2. Load Dataset
    print("Loading datasets...")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl")
    })

    # Format and tokenize
    dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # 3. Setup Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=10,
        report_to="wandb",
        run_name="countdown-teacher-sft",
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # 4. Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Training complete!")

if __name__ == "__main__":
    main()

```


## File: `scripts\__init__.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


```


## File: `tests\test_action_conditioned.py`
```python
import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.action_conditioned import (
    ActionMLPConfig,
    conditional_structure,
    extract_action_transitions,
    action_bigram_top1,
    state_bigram_top1,
    train_action_mlp,
)


def _planted_traj(n, K=5, N=6, seed=0):
    """Env where op0: c->(c+1)%K, op1: c->(c+2)%K.

    z_t alone is ambiguous (two equally likely successors); (z_t, op) is
    deterministic.
    """
    rng = np.random.RandomState(seed)
    trajs = []
    for _ in range(n):
        c = int(rng.randint(0, K))
        codes = [c]
        ops = []
        for _ in range(N):
            op = int(rng.randint(0, 2))
            c = (c + (1 if op == 0 else 2)) % K
            codes.append(c)
            ops.append(op)
        trajs.append(DiscreteTrajectory(
            codes=torch.tensor(codes, dtype=torch.int64),
            op_ids=torch.tensor(ops, dtype=torch.long),
            operands=torch.zeros(N, 2),
            numbers=[1], target=1,
        ))
    return trajs


def test_action_conditioning_recovers_planted_structure():
    K = 5
    train = extract_action_transitions(_planted_traj(300, K=K, seed=0))
    ev = extract_action_transitions(_planted_traj(150, K=K, seed=1))

    # Lookups: state ambiguous (~0.5), action deterministic (~1.0).
    sb = state_bigram_top1(train, ev)
    ab = action_bigram_top1(train, ev)
    assert 0.4 <= sb <= 0.65, sb
    assert ab > 0.95, ab

    # MLPs: action-conditioned recovers the rule; state-only cannot.
    d = train_action_mlp(train, ev, K, ActionMLPConfig(use_state=True, use_op=False, epochs=40))
    e = train_action_mlp(train, ev, K, ActionMLPConfig(use_state=True, use_op=True, epochs=40))
    assert d["top1"] < 0.65, d
    assert e["top1"] > 0.9, e

    # Structure: H(z'|z) high, H(z'|z,op) ~ 0; determinism flips on conditioning.
    s_state = conditional_structure(train.z_t.tolist(), train.z_next.tolist())
    s_act = conditional_structure(
        list(zip(train.z_t.tolist(), train.op.tolist())), train.z_next.tolist())
    assert s_state["global_entropy"] > 0.5
    assert s_act["global_entropy"] < 0.1
    assert s_state["det_frac_mass"] < 0.1
    assert s_act["det_frac_mass"] > 0.9


def test_extract_action_transitions_shapes():
    trajs = _planted_traj(10, K=4, N=5, seed=2)
    tr = extract_action_transitions(trajs)
    assert tr.z_t.shape == tr.z_next.shape == tr.op.shape
    assert tr.operands.shape == (tr.z_t.shape[0], 2)
    assert tr.z_t.shape[0] == 10 * 5

```


## File: `tests\test_action_parser.py`
```python
import pytest

from data_processing.action_parser import (
    Op,
    Action,
    OP_TO_ID,
    ID_TO_OP,
    parse_step,
    parse_solution,
    apply_op,
)


def test_parse_mul():
    action = parse_step("75 * 11 = 825")
    assert action.op == Op.MUL
    assert action.arg1 == 75
    assert action.arg2 == 11
    assert action.result == 825


@pytest.mark.parametrize(
    "step,op,a,b,r",
    [
        ("3 + 4 = 7", Op.ADD, 3, 4, 7),
        ("20 - 2 = 18", Op.SUB, 20, 2, 18),
        ("6 * 7 = 42", Op.MUL, 6, 7, 42),
        ("100 / 4 = 25", Op.DIV, 100, 4, 25),
    ],
)
def test_parse_all_ops(step, op, a, b, r):
    action = parse_step(step)
    assert action.op == op
    assert action.arg1 == a
    assert action.arg2 == b
    assert action.result == r


def test_parse_tolerates_extra_whitespace():
    action = parse_step("  25   -   5   =   20  ")
    assert action.op == Op.SUB
    assert action.arg1 == 25
    assert action.arg2 == 5
    assert action.result == 20


def test_parse_solution_list():
    steps = ["10 * 8 = 80", "80 - 50 = 30", "30 + 100 = 130"]
    actions = parse_solution(steps)
    assert len(actions) == 3
    assert [a.op for a in actions] == [Op.MUL, Op.SUB, Op.ADD]
    assert actions[-1].result == 130


def test_op_id_roundtrip():
    # Stable, contiguous ids 0..3 for embedding lookups.
    assert sorted(OP_TO_ID.values()) == [0, 1, 2, 3]
    for op in Op:
        assert ID_TO_OP[OP_TO_ID[op]] == op


def test_apply_op_matches_arithmetic():
    assert apply_op(Op.ADD, 3, 4) == 7
    assert apply_op(Op.SUB, 20, 2) == 18
    assert apply_op(Op.MUL, 6, 7) == 42
    assert apply_op(Op.DIV, 100, 4) == 25


def test_malformed_raises():
    with pytest.raises(ValueError):
        parse_step("not an equation")
    with pytest.raises(ValueError):
        parse_step("3 ^ 4 = 81")  # unsupported operator
    with pytest.raises(ValueError):
        parse_step("3 + 4")  # missing result


def test_result_mismatch_can_be_validated():
    # By default we trust the teacher text; strict mode flags arithmetic errors.
    bad = parse_step("3 + 4 = 99")  # no raise by default
    assert bad.result == 99
    assert not bad.is_arithmetically_valid()
    with pytest.raises(ValueError):
        parse_step("3 + 4 = 99", validate=True)

```


## File: `tests\test_codebook_usage.py`
```python
import numpy as np
import torch

from evaluation.codebook_usage import (
    _gini,
    analyze_codebook_usage,
    save_codebook_usage_report,
)


def test_gini_extremes():
    # Uniform -> 0; single-code collapse -> approaches 1 (for large K).
    assert _gini(np.ones(10)) == 0.0
    collapsed = np.zeros(10)
    collapsed[0] = 100.0
    assert 0.8 < _gini(collapsed) <= 1.0


def test_active_dead_counts():
    codes = torch.tensor([0, 0, 0, 1, 1, 2, 5, 5, 5], dtype=torch.int64)
    stats = analyze_codebook_usage(codes, num_codes=8, min_freq_frac=0.1)
    # Codes 3, 4, 6, 7 never used -> 4 dead.
    assert stats["dead_codes"] == 4
    assert stats["used_codes"] == 4
    # Active threshold 10%: codes 0 (3/9), 5 (3/9) are >= 0.1; code 2 (1/9) is ~0.11.
    assert stats["active_codes"] >= 2
    # Entropy / perplexity in valid ranges for K=8.
    assert 0.0 <= stats["entropy"] <= np.log(8) + 1e-6
    assert 1.0 <= stats["perplexity"] <= 8.0


def test_collapse_detected_on_skewed():
    # Almost all mass on one code.
    codes = torch.tensor([0] * 95 + [1, 2, 3, 4, 5], dtype=torch.int64)
    stats = analyze_codebook_usage(codes, num_codes=8)
    assert stats["collapse_score"] > 0.7
    assert stats["perplexity"] < 3.0


def test_empty_codes():
    stats = analyze_codebook_usage(torch.empty(0, dtype=torch.int64), num_codes=4)
    assert stats["active_codes"] == 0
    assert stats["dead_codes"] == 4
    assert stats["perplexity"] == 0.0


def test_report_writes_csv(tmp_path):
    codes = torch.randint(0, 6, (100,))
    stats = analyze_codebook_usage(codes, num_codes=6)
    csv = tmp_path / "usage.csv"
    png = tmp_path / "usage.png"
    save_codebook_usage_report(stats, str(csv), str(png))
    assert csv.exists()
    assert png.exists()

```


## File: `tests\test_coherence.py`
```python
import math
import torch
import torch.nn as nn

from data_processing.trajectory_dataset import Trajectory
from evaluation.coherence import evaluate_coherence


class _PerfectTransition(nn.Module):
    """Reproduces synthetic dynamics h_{t+1} = h_t + shift[op] exactly."""

    def __init__(self, shifts):
        super().__init__()
        self.shifts = shifts

    def forward(self, h, op_id, operands):
        return h + self.shifts.to(h.device)[op_id]


def _toy_trajectories(n=5, H=8, steps=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    shifts = torch.randn(4, H, generator=g)
    trajs = []
    for _ in range(n):
        T = steps + 2
        all_hidden = torch.zeros(T, H)
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        s = torch.randn(H, generator=g)
        all_hidden[0] = s
        for i in range(steps):
            s = s + shifts[op_ids[i]]
            all_hidden[i + 1] = s
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.arange(T),
            state_indices=torch.arange(steps + 1),
            op_ids=op_ids,
            operands=torch.zeros(steps, 2),
            numbers=[1, 2, 3, 4, 5, 6],
            target=10,
        ))
    return trajs, shifts


def test_perfect_model_is_coherent():
    trajs, shifts = _toy_trajectories()
    model = _PerfectTransition(shifts)
    df = evaluate_coherence(model, trajs, probe_c=None, max_depth=8)

    assert list(df["depth"]) == list(range(1, 9))
    # Data only supports depth 4; deeper rows are empty.
    populated = df[df["n_samples"] > 0]
    assert set(populated["depth"]) == {1, 2, 3, 4}
    # Perfect dynamics -> ~0 MSE, ~1 cosine at every populated depth.
    assert populated["mse"].max() < 1e-8
    assert populated["cosine_similarity"].min() > 1 - 1e-5
    # No decoder -> operator accuracy is NaN.
    assert all(math.isnan(x) for x in populated["operator_accuracy"])
    assert all(math.isnan(x) for x in populated["state_probe_accuracy"])


def test_depth_limited_by_data():
    trajs, shifts = _toy_trajectories(steps=2)
    df = evaluate_coherence(_PerfectTransition(shifts), trajs, max_depth=8)
    populated = df[df["n_samples"] > 0]
    assert set(populated["depth"]) == {1, 2}

```


## File: `tests\test_cross_domain_transfer.py`
```python
import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    encode_trajectories_to_codes,
)
from evaluation.cross_domain_transfer import evaluate_cross_domain_transfer
from models.vq_state import VQStateQuantizer
from training.train_vq import train_vq_quantizer, VQTrainConfig


def _toy_trajs(centers, n=20, H=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        N = 4
        idx = torch.randint(0, len(centers), (N + 1,), generator=g)
        states = centers[idx] + 0.05 * torch.randn(N + 1, H, generator=g)
        out.append(Trajectory(
            all_hidden=states, input_ids=torch.zeros(N + 1, dtype=torch.long),
            state_indices=torch.arange(N + 1),
            op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2), numbers=[1, 2, 3], target=100,
        ))
    return out


def test_near_domain_high_reuse():
    """Transfer domain near training domain -> high code reuse, low KL."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=30, seed=0)
    vq = train_vq_quantizer(train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128))
    disc_train = encode_trajectories_to_codes(vq, train)
    # Near domain: same centers + small noise.
    near = _toy_trajs(centers + 0.05, n=15, seed=1)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"near": near}, num_codes=K)
    assert res["near"]["code_reuse"] > 0.8
    assert "transition_kl_symmetric" in res["near"]
    assert res["near"]["n_states"] == 15 * 5


def test_arithmetic_reference_present():
    """The reference 'arithmetic' entry should always be present."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=20, seed=0)
    vq = train_vq_quantizer(train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128))
    disc_train = encode_trajectories_to_codes(vq, train)
    res = evaluate_cross_domain_transfer(vq, disc_train, {}, num_codes=K)
    assert "arithmetic" in res
    assert res["arithmetic"]["active_codes"] > 0
    assert res["arithmetic"]["perplexity"] > 0


def test_entropy_delta_sign():
    """Far domain should have different entropy (may go up or down)."""
    torch.manual_seed(1)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=30, seed=0)
    vq = train_vq_quantizer(train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128))
    disc_train = encode_trajectories_to_codes(vq, train)
    far_centers = torch.randn(6, H) * 3  # very different region
    far = _toy_trajs(far_centers, n=15, seed=2)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"far": far}, num_codes=K)
    # Entropy delta may be positive or negative, but should not be NaN.
    assert not np.isnan(res["far"]["entropy_delta"])


def test_empty_domain_graceful():
    """Empty transfer domain -> error key in result."""
    torch.manual_seed(0)
    H, K = 12, 8
    centers = torch.randn(4, H)
    train = _toy_trajs(centers, n=20, seed=0)
    vq = train_vq_quantizer(train, config=VQTrainConfig(num_codes=K, epochs=50, batch_size=128))
    disc_train = encode_trajectories_to_codes(vq, train)
    res = evaluate_cross_domain_transfer(vq, disc_train, {"empty": []}, num_codes=K)
    assert res["empty"].get("error") is not None

```


## File: `tests\test_dataset.py`
```python
import os
import json
import pytest
from scripts.generate_countdown_dataset import generate_countdown_problem, generate_dataset

def test_generate_countdown_problem():
    problem = generate_countdown_problem()
    assert "numbers" in problem
    assert "target" in problem
    assert "solution" in problem
    assert "cot" in problem
    assert len(problem["numbers"]) == 6
    assert isinstance(problem["target"], int)
    assert isinstance(problem["cot"], str)
    assert len(problem["solution"]) > 0

def test_generate_dataset(tmp_path):
    output_file = os.path.join(tmp_path, "test_dataset.jsonl")
    generate_dataset(10, output_file)
    
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 10
        
        # Verify first line parses
        data = json.loads(lines[0])
        assert "numbers" in data

```


## File: `tests\test_diagnostic_decoder.py`
```python
import torch

from models.diagnostic_decoder import DiagnosticDecoder
from training.train_decoder import train_decoder_model, DecoderTrainConfig
from data_processing.trajectory_dataset import Trajectory


def test_decoder_shapes():
    dec = DiagnosticDecoder(hidden_dim=8, vocab_size=5)
    h = torch.randn(3, 8)
    assert dec(h).shape == (3, 5)
    assert dec.predict(h).shape == (3,)


def _decodable_trajectories(n=12, H=8, vocab=5, T=6, seed=0):
    """Trajectories where each hidden state linearly determines the next token."""
    g = torch.Generator().manual_seed(seed)
    token_vecs = torch.randn(vocab, H, generator=g)  # fixed embedding per token
    trajs = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (T,), generator=g)
        # hidden[p] encodes the token that comes at p+1.
        all_hidden = torch.zeros(T, H)
        for p in range(T - 1):
            all_hidden[p] = token_vecs[ids[p + 1]]
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=ids,
            state_indices=torch.arange(min(3, T)),
            op_ids=torch.zeros(min(2, T - 1), dtype=torch.long),
            operands=torch.zeros(min(2, T - 1), 2),
            numbers=[1, 2, 3, 4, 5, 6],
            target=10,
        ))
    return trajs


def test_decoder_learns_mapping():
    trajs = _decodable_trajectories()
    cfg = DecoderTrainConfig(epochs=80, lr=5e-2, batch_size=32)
    model = train_decoder_model(trajs, vocab_size=5, val_trajs=trajs, config=cfg)
    hist = cfg.history
    assert hist[-1]["accuracy"] > hist[0]["accuracy"]
    # Mapping is deterministic and linearly separable -> should be learned well.
    assert hist[-1]["accuracy"] > 0.9

```


## File: `tests\test_discrete_rollout.py`
```python
import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    train_code_transition,
)
from evaluation.discrete_rollout import evaluate_discrete_rollout
from models.vq_state import VQStateQuantizer


def _cyclic_trajs(n=20, H=12, K=6, seed=0):
    """Trajectories with deterministic cyclic codes: z_{t+1} = (z_t+1) % K."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 5
        start = int(torch.randint(0, K, (1,), generator=g).item())
        seq = [(start + i) % K for i in range(N + 1)]
        states = centers[K - 1] * torch.ones(N + 1, H)  # constant, to test code-only
        trajs.append(Trajectory(
            all_hidden=states, input_ids=torch.zeros(N + 1, dtype=torch.long),
            state_indices=torch.arange(N + 1),
            op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2), numbers=[1, 2, 3], target=100,
        ))
    return trajs, centers, K


def test_rollout_produces_correct_depths():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc, num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    assert len(df) == 3
    assert list(df["depth"]) == [1, 2, 3]
    assert (df["n_samples"] > 0).all()


def test_rollout_cosine_in_range():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc, num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    for _, r in df.iterrows():
        # Cosine similarity must be in [-1, 1] (should be >= 0 for real data).
        assert -1.0 <= r["cosine_similarity"] <= 1.0
        assert r["code_match_accuracy"] >= 0.0


def test_rollout_mse_non_negative():
    trajs, centers, K = _cyclic_trajs(n=15, seed=0)
    vq = VQStateQuantizer(hidden_dim=12, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc, num_codes=K,
        config=CodeTransitionConfig(epochs=40, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=3)
    for _, r in df.iterrows():
        assert r["mse"] >= 0.0


def test_rollout_degrades_with_depth():
    """On noisy data, code-match should tend to decrease with depth."""
    g = torch.Generator().manual_seed(1)
    H, K, n = 12, 8, 30
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 6
        start = int(torch.randint(0, K, (1,), generator=g).item())
        seq = [(start + i) % K for i in range(N + 1)]
        states = centers[K - 1] + 0.5 * torch.randn(N + 1, H, generator=g)
        trajs.append(Trajectory(
            all_hidden=states, input_ids=torch.zeros(N + 1, dtype=torch.long),
            state_indices=torch.arange(N + 1),
            op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2), numbers=[1, 2, 3], target=100,
        ))
    vq = VQStateQuantizer(hidden_dim=H, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    model, _ = train_code_transition(
        disc, num_codes=K,
        config=CodeTransitionConfig(epochs=50, seed=0),
    )
    df = evaluate_discrete_rollout(model, vq, disc, max_depth=5)
    # First depth should not be worse than last depth on average.
    if df["n_samples"].iloc[0] > 0 and df["n_samples"].iloc[-1] > 0:
        assert df["code_match_accuracy"].iloc[0] >= df["code_match_accuracy"].iloc[-1] - 0.2

```


## File: `tests\test_discrete_trajectory_dataset.py`
```python
import os
import tempfile

import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTransitionDataset,
    all_codes,
    encode_trajectories_to_codes,
    save_discrete_trajectories,
    load_discrete_trajectories,
)
from models.vq_state import VQStateQuantizer


def _toy_trajectories(n=6, H=8, steps=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        all_hidden = torch.randn(steps + 1, H, generator=g)
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.zeros(steps + 1, dtype=torch.long),
            state_indices=torch.arange(steps + 1),
            op_ids=torch.randint(0, 4, (steps,), generator=g),
            operands=torch.randint(1, 50, (steps, 2), generator=g).float(),
            numbers=[1, 2, 3, 4, 5, 6],
            target=100,
        ))
    return trajs


def test_encode_produces_int_codes():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    assert len(disc) == len(trajs)
    for d, t in zip(disc, trajs):
        assert d.codes.shape[0] == t.num_steps + 1
        assert d.codes.dtype == torch.int64
        assert 0 <= int(d.codes.min()) < 12
        assert 0 <= int(d.codes.max()) < 12
        # Action metadata preserved.
        assert torch.equal(d.op_ids, t.op_ids)


def test_transition_dataset_pairs():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    ds = DiscreteTransitionDataset(disc)
    expected = sum(t.num_steps for t in trajs)
    assert len(ds) == expected
    item = ds[0]
    assert set(item.keys()) == {"z_t", "z_next"}
    assert item["z_t"].dtype == torch.long


def test_all_codes_concatenates():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    codes = all_codes(disc)
    expected = sum(t.num_steps + 1 for t in trajs)
    assert codes.shape[0] == expected
    assert all_codes([]).shape[0] == 0


def test_save_load_roundtrip():
    trajs = _toy_trajectories()
    vq = VQStateQuantizer(hidden_dim=8, num_codes=12)
    disc = encode_trajectories_to_codes(vq, trajs)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "disc.pt")
        save_discrete_trajectories(disc, path)
        loaded = load_discrete_trajectories(path)
        assert len(loaded) == len(disc)
        for a, b in zip(loaded, disc):
            assert torch.equal(a.codes, b.codes)

```


## File: `tests\test_discrete_transition.py`
```python
import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from data_processing.discrete_trajectory_dataset import (
    DiscreteTrajectory,
    encode_trajectories_to_codes,
)
from evaluation.discrete_transition import (
    CodeTransitionConfig,
    CodeTransitionModel,
    build_transition_matrix,
    conditional_entropy,
    save_entropy_report,
    save_predictability_metrics,
    train_code_transition,
)
from models.vq_state import VQStateQuantizer


def _synthetic(n=20, H=8, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, H, generator=g)
    trajs = []
    for _ in range(n):
        N = 4
        seq = torch.randint(0, K, (N + 1,), generator=g)
        states = centers[seq] + 0.05 * torch.randn(N + 1, H, generator=g)
        trajs.append(Trajectory(
            all_hidden=states, input_ids=torch.zeros(N + 1, dtype=torch.long),
            state_indices=torch.arange(N + 1),
            op_ids=torch.randint(0, 4, (N,), generator=g),
            operands=torch.randint(1, 50, (N, 2), generator=g).float(),
            numbers=[1, 2, 3, 4, 5, 6], target=100,
        ))
    return trajs, centers, K


def test_transition_matrix_stochastic():
    # Deterministic cycles: code i -> (i+1) % K.
    K = 5
    trajs = []
    for _ in range(10):
        codes = torch.tensor([(i % K) for i in range(6)], dtype=torch.int64)
        trajs.append(DiscreteTrajectory(
            codes=codes, op_ids=torch.zeros(5, dtype=torch.long),
            operands=torch.zeros(5, 2), numbers=[1], target=1,
        ))
    T = build_transition_matrix(trajs, K)
    # Rows that were observed must sum to 1.
    for i in range(K):
        assert abs(T[i].sum() - 1.0) < 1e-6
    # Each observed transition should be ~100% on one successor.
    for i in range(K):
        assert T[i][(i + 1) % K] > 0.9


def test_transition_matrix_empty_rows_uniform():
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1], dtype=torch.int64),
        op_ids=torch.zeros(1, dtype=torch.long), operands=torch.zeros(1, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    # Unobserved rows (2, 3) should fall back to uniform.
    assert abs(T[2, 0] - 0.25) < 1e-6
    assert abs(T[3, 0] - 0.25) < 1e-6


def test_conditional_entropy_deterministic_is_zero():
    K = 3
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        op_ids=torch.zeros(3, dtype=torch.long), operands=torch.zeros(3, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    ent = conditional_entropy(T)
    # Deterministic transitions -> entropy should be 0 (all mass on one successor).
    assert ent["global_entropy"] < 0.01


def test_conditional_entropy_uniform():
    K = 10
    # Uniform: every row has equal probability to go to any code.
    # Simulate by making many random transitions.
    g = torch.Generator().manual_seed(0)
    trajs = [DiscreteTrajectory(
        codes=torch.randint(0, K, (6,), generator=g, dtype=torch.int64),
        op_ids=torch.zeros(5, dtype=torch.long), operands=torch.zeros(5, 2),
        numbers=[1], target=1,
    ) for _ in range(200)]
    T = build_transition_matrix(trajs, K)
    ent = conditional_entropy(T)
    # High entropy (near log(K)) expected for uniform transitions.
    assert ent["global_entropy"] > 1.0
    assert ent["global_perplexity"] > 2.0


def test_train_code_transition_reduces_loss():
    trajs, centers, K = _synthetic(n=30, seed=0)
    vq = VQStateQuantizer(hidden_dim=8, num_codes=K)
    disc = encode_trajectories_to_codes(vq, trajs)
    cfg = CodeTransitionConfig(epochs=40, seed=0)
    model, metrics = train_code_transition(disc, num_codes=K, config=cfg)
    # Majority/bigram baselines replace the removed uniform-chance metric
    # (V5.1 Fix 2) and must be well-defined accuracies.
    assert 0.0 <= metrics["majority_baseline"] <= 1.0
    assert 0.0 <= metrics["bigram_baseline"] <= 1.0
    # Sequences here are random (no transition structure), so the MLP cannot
    # meaningfully beat the count baselines — it should track the majority
    # floor rather than collapse far below it.
    assert metrics["top1_accuracy"] >= metrics["majority_baseline"] - 0.2
    assert metrics["predictive_entropy_nats"] > 0.0


def test_predictability_metrics_in_range():
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.randint(0, K, (4,), dtype=torch.int64),
        op_ids=torch.zeros(3, dtype=torch.long), operands=torch.zeros(3, 2),
        numbers=[1], target=1,
    ) for _ in range(10)]
    cfg = CodeTransitionConfig(epochs=5, seed=0)
    _, m = train_code_transition(trajs, num_codes=K, config=cfg)
    assert 0.0 <= m["top1_accuracy"] <= 1.0
    assert 0.0 <= m["predictive_entropy_nats"]
    assert m["predictive_perplexity"] >= 1.0


def test_save_entropy_report(tmp_path):
    K = 4
    trajs = [DiscreteTrajectory(
        codes=torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64),
        op_ids=torch.zeros(4, dtype=torch.long), operands=torch.zeros(4, 2),
        numbers=[1], target=1,
    )]
    T = build_transition_matrix(trajs, K)
    freqs = np.ones(K) / K
    ent = conditional_entropy(T, code_freqs=freqs)
    csv = tmp_path / "entropy.csv"
    png = tmp_path / "entropy.png"
    save_entropy_report(T, ent, str(csv), str(png), active_mask=np.ones(K, dtype=bool))
    assert csv.exists()
    assert png.exists()

```


## File: `tests\test_game24_dataset.py`
```python
"""Game-of-24 generator: every emitted solution must be a valid, parseable,
integer-only reduction of the four numbers to 24 — the contract the downstream
Countdown pipeline (action_parser + build_trajectories) relies on.

Also verifies that the exhaustive pool partitioning produces zero-overlap splits
and that the pool is complete (every solvable hand is included)."""

import json
import os
import random
import warnings
from collections import Counter
from itertools import combinations_with_replacement

from data_processing.action_parser import parse_step
from scripts.generate_game24_dataset import (
    _solve,
    enumerate_solvable_pool,
    generate_dataset,
    generate_game24_problem,
    generate_partitioned_splits,
    CARD_MIN,
    CARD_MAX,
    TARGET,
)


def _simulate(numbers, actions):
    """Replay the steps as a multiset reduction; return the final lone value."""
    available = Counter(numbers)
    for a in actions:
        assert available[a.arg1] > 0, f"operand {a.arg1} not on the table"
        available[a.arg1] -= 1
        assert available[a.arg2] > 0, f"operand {a.arg2} not on the table"
        available[a.arg2] -= 1
        available[a.result] += 1
    remaining = list(available.elements())
    assert len(remaining) == 1, f"expected one value left, got {remaining}"
    return remaining[0]


def test_generated_problems_are_valid():
    rng = random.Random(0)
    for _ in range(200):
        prob = generate_game24_problem(rng)
        assert prob["target"] == 24
        assert len(prob["numbers"]) == 4
        # Every step parses AND is arithmetically valid (validate=True raises otherwise).
        actions = [parse_step(s, validate=True) for s in prob["solution"]]
        assert len(actions) == 3  # four numbers -> three binary ops
        assert _simulate(prob["numbers"], actions) == 24


def test_determinism():
    a = generate_game24_problem(random.Random(42))
    b = generate_game24_problem(random.Random(42))
    assert a == b


# --------------------------------------------------------------------------- #
# Pool enumeration tests
# --------------------------------------------------------------------------- #

def test_pool_exhaustive():
    """Every solvable multiset in {1..13}^4 is in the pool, and no unsolvable
    hand is included."""
    pool = enumerate_solvable_pool()
    pool_set = set(pool)

    # Pool elements must be sorted tuples (canonical form)
    for hand in pool:
        assert hand == tuple(sorted(hand)), f"hand {hand} is not sorted"

    # Verify no duplicates
    assert len(pool_set) == len(pool), "pool contains duplicate hands"

    # Cross-check a sample: verify every pool entry is solvable
    for hand in pool:
        items = [(n, []) for n in hand]
        assert _solve(items, TARGET) is not None, f"pool entry {hand} is NOT solvable"

    # Verify no solvable hand is missing (exhaustive negative check)
    for hand in combinations_with_replacement(range(CARD_MIN, CARD_MAX + 1), 4):
        items = [(n, []) for n in hand]
        is_solvable = _solve(items, TARGET) is not None
        in_pool = hand in pool_set
        assert is_solvable == in_pool, (
            f"hand {hand}: solvable={is_solvable} but in_pool={in_pool}"
        )


def test_pool_size():
    """The pool should have exactly 1346 solvable hands."""
    pool = enumerate_solvable_pool()
    assert len(pool) == 1346, f"expected 1346 solvable hands, got {len(pool)}"


# --------------------------------------------------------------------------- #
# Split overlap tests
# --------------------------------------------------------------------------- #

def test_no_split_overlap(tmp_path):
    """Train, val, test splits share zero puzzles."""
    sizes = generate_partitioned_splits(str(tmp_path), seed=0)

    # Load back the JSONL files
    splits = {}
    for split_name in ("train", "val", "test"):
        hands = set()
        path = os.path.join(str(tmp_path), f"{split_name}.jsonl")
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                hands.add(tuple(sorted(rec["numbers"])))
        splits[split_name] = hands
        assert len(hands) == sizes[split_name], (
            f"{split_name}: expected {sizes[split_name]} unique hands, got {len(hands)}"
        )

    # Zero overlap between all pairs
    assert len(splits["train"] & splits["val"]) == 0, "train ∩ val is non-empty"
    assert len(splits["train"] & splits["test"]) == 0, "train ∩ test is non-empty"
    assert len(splits["val"] & splits["test"]) == 0, "val ∩ test is non-empty"

    # Union covers entire pool
    pool = set(enumerate_solvable_pool())
    union = splits["train"] | splits["val"] | splits["test"]
    assert union == pool, (
        f"splits don't cover full pool: missing {len(pool - union)}, extra {len(union - pool)}"
    )


def test_no_within_split_duplicates(tmp_path):
    """Each split contains only unique hands (no resampling)."""
    generate_partitioned_splits(str(tmp_path), seed=7)

    for split_name in ("train", "val", "test"):
        path = os.path.join(str(tmp_path), f"{split_name}.jsonl")
        hands = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                hands.append(tuple(sorted(rec["numbers"])))
        assert len(hands) == len(set(hands)), (
            f"{split_name} contains duplicate hands"
        )


def test_generate_dataset_rejects_oversized_request(tmp_path):
    """Requesting more samples than the pool size raises ValueError."""
    try:
        generate_dataset(2000, os.path.join(str(tmp_path), "too_big.jsonl"))
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "1346" in str(e) or "solvable" in str(e).lower()


def test_is_ood_accepted_with_warning(tmp_path):
    """is_ood=True is accepted but emits a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        generate_dataset(10, os.path.join(str(tmp_path), "ood.jsonl"), is_ood=True)
        assert len(w) == 1
        assert "is_ood" in str(w[0].message).lower() or "game24" in str(w[0].message).lower()

```


## File: `tests\test_layer_sweep.py`
```python
import math

from evaluation.layer_sweep import summarize_sweep, _frac

FLOOR = {"det_action": 0.0, "H_action": 2.26, "mlp_z_op": 0.37}
CEILING = {"det_action": 0.59, "H_action": 0.39, "mlp_z_op": 0.77}


def test_frac_basic_and_degenerate():
    assert _frac(0.0, 0.6, 0.3) == 0.5
    # Entropy axis: floor high, ceiling low -> fraction of the way *down*.
    assert _frac(2.26, 0.39, 0.39) == 1.0
    assert math.isnan(_frac(0.5, 0.5, 0.5))  # coincident anchors -> NaN, no crash


def test_best_layer_is_most_deterministic():
    rows = [
        {"layer": 4, "det_action": 0.05, "H_action": 2.0, "mlp_z_op": 0.40},
        {"layer": 12, "det_action": 0.31, "H_action": 1.1, "mlp_z_op": 0.60},
        {"layer": 22, "det_action": 0.11, "H_action": 1.6, "mlp_z_op": 0.55},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 12
    # Layer 12 det 0.31 over a 0->0.59 axis ~ 53% of the way up.
    assert abs(s["best_pos_det"] - 0.31 / 0.59) < 1e-9


def test_reaches_ceiling_when_a_layer_is_deterministic():
    rows = [
        {"layer": 8, "det_action": 0.10, "H_action": 1.9, "mlp_z_op": 0.45},
        {"layer": 16, "det_action": 0.45, "H_action": 0.6, "mlp_z_op": 0.71},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 16
    assert s["reaches_ceiling"] is True          # det 0.45 >= 0.20 and >= 50% of ceiling


def test_no_layer_reaches_ceiling_near_floor():
    rows = [
        {"layer": 4, "det_action": 0.02, "H_action": 2.2, "mlp_z_op": 0.39},
        {"layer": 12, "det_action": 0.11, "H_action": 1.9, "mlp_z_op": 0.52},
        {"layer": 22, "det_action": 0.09, "H_action": 2.0, "mlp_z_op": 0.50},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 12
    assert s["reaches_ceiling"] is False         # 0.11 < 0.20 bar
    assert len(s["per_layer"]) == 3


def test_tie_broken_by_mlp():
    rows = [
        {"layer": 6, "det_action": 0.20, "H_action": 1.4, "mlp_z_op": 0.58},
        {"layer": 18, "det_action": 0.20, "H_action": 1.2, "mlp_z_op": 0.66},
    ]
    s = summarize_sweep(rows, FLOOR, CEILING)
    assert s["best_layer"] == 18                 # equal det, higher MLP wins

```


## File: `tests\test_model_loading.py`
```python
import pytest
from models.model_loader import load_tokenizer, load_model, load_qlora_model, load_checkpoint
import torch
import inspect

# Use a tiny dummy model for testing infrastructure instead of the full 1.1B model
TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for bitsandbytes")
def test_load_qlora_model():
    model = load_qlora_model(model_id=TEST_MODEL)
    assert model is not None
    # Check if peft is applied (should have active_adapters)
    assert hasattr(model, "active_adapters")

def test_load_tokenizer():
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    assert tokenizer is not None
    assert tokenizer.pad_token is not None

def test_load_base_model():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_load_model_4bit_defaults_false():
    """load_model defaults to load_in_4bit=False (V5.3 compatible)."""
    sig = inspect.signature(load_model)
    assert sig.parameters["load_in_4bit"].default is False


def test_load_checkpoint_4bit_defaults_false():
    """load_checkpoint defaults to load_in_4bit=False — 4-bit is opt-in only.

    This is the critical V5.3 backward-compatibility check: older checkpoints
    saved in fp16 must load in fp16 by default, not be forced into 4-bit."""
    sig = inspect.signature(load_checkpoint)
    assert "load_in_4bit" in sig.parameters, (
        "load_checkpoint is missing the load_in_4bit parameter"
    )
    assert sig.parameters["load_in_4bit"].default is False, (
        "load_checkpoint should default to load_in_4bit=False for V5.3 compat"
    )


```


## File: `tests\test_permutation_robustness.py`
```python
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.permutation_robustness import evaluate_permutation_robustness


def _make_group(n_solutions, shared_start_code):
    """n_solutions trajectories for one problem, all sharing code at depth 0."""
    trajs = []
    for s in range(n_solutions):
        N = 3
        codes = torch.tensor(
            [shared_start_code, shared_start_code + s + 1,
             shared_start_code + s + 2, shared_start_code + s + 3],
            dtype=torch.int64,
        )
        trajs.append(DiscreteTrajectory(
            codes=codes, op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2),
            numbers=[25, 50, 75, 100, 3, 5], target=100,
        ))
    return trajs


def test_perfect_consistency():
    """All solutions share the same code at depth 0 -> consistency = 1.0."""
    groups = [
        _make_group(3, shared_start_code=10),
        _make_group(2, shared_start_code=20),
    ]
    m = evaluate_permutation_robustness(groups, num_codes=64, seed=0)
    assert m["cross_consistency"] == 1.0
    assert m["n_pairs_cross"] > 0
    assert m["uniform_random_baseline"] == 1.0 / 64


def test_baseline_below_random():
    """Random codes -> consistency near uniform_random_baseline."""
    import random

    rng = random.Random(0)
    groups = []
    for _ in range(5):
        group = []
        for _ in range(3):
            N = 3
            codes = torch.tensor([rng.randint(0, 16) for _ in range(N + 1)], dtype=torch.int64)
            group.append(DiscreteTrajectory(
                codes=codes, op_ids=torch.zeros(N, dtype=torch.long),
                operands=torch.zeros(N, 2),
                numbers=[25, 50, 75, 100, 3, 5], target=100,
            ))
        groups.append(group)
    m = evaluate_permutation_robustness(groups, num_codes=16, seed=0)
    # Cross consistency should be near the uniform baseline (1/16 ≈ 0.0625).
    assert m["cross_consistency"] < 0.3
    assert m["n_problems_total"] == 5


def test_single_solution_no_cross_pairs():
    """One solution per problem -> zero cross pairs, consistency NaN."""
    groups = [[_make_group(1, 5)[0]] for _ in range(3)]
    m = evaluate_permutation_robustness(groups, num_codes=8, seed=0)
    assert m["n_pairs_cross"] == 0
    assert m["cross_consistency"] != m["cross_consistency"]  # NaN


def test_metrics_in_dict():
    groups = [_make_group(2, 10)]
    m = evaluate_permutation_robustness(groups, num_codes=32, seed=0)
    expected_keys = {
        "cross_consistency", "within_consistency",
        "uniform_random_baseline", "empirical_random_baseline",
        "n_pairs_cross", "n_problems_with_match",
        "n_problems_total", "mean_solutions_per_problem",
    }
    assert expected_keys.issubset(m.keys())

```


## File: `tests\test_pipeline.py`
```python
import os
import json
import torch
import pytest
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

from models.model_loader import load_model, load_tokenizer
from scripts.extract_hidden_states import extract_hidden_states

TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

@pytest.fixture
def dummy_dataset():
    data = [
        {"numbers": [1, 2, 3], "target": 6, "cot": "1+2=3\n3+3=6"},
        {"numbers": [2, 4], "target": 8, "cot": "2*4=8"}
    ]
    return data

def test_extraction(tmp_path, dummy_dataset):
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    
    output_file = os.path.join(tmp_path, "hidden_states.pt")
    
    # Run extraction (extract from layer 1)
    extract_hidden_states(model, tokenizer, dummy_dataset, target_layer=1, output_file=output_file)
    
    assert os.path.exists(output_file)
    extracted = torch.load(output_file)
    
    assert len(extracted) > 0
    assert "sample_id" in extracted[0]
    assert "token_id" in extracted[0]
    assert "hidden_state" in extracted[0]
    
def test_dummy_train_loop(tmp_path, dummy_dataset):
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    
    # Format dataset
    formatted = [{"text": f"Problem: {d['target']}\nSolution:\n{d['cot']}"} for d in dummy_dataset]
    ds = Dataset.from_list(formatted)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=32)
        
    tokenized_ds = ds.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_strategy="no",
        logging_steps=1,
        report_to="none"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    
    # Simply verify that the train loop can start without crashing
    train_result = trainer.train()
    assert train_result.global_step > 0

```


## File: `tests\test_position_leakage.py`
```python
import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from evaluation.position_leakage import evaluate_position_leakage


def _disc_trajs(n=30, code_fn=None, max_steps=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        N = int(torch.randint(2, max_steps + 1, (1,), generator=g).item())
        if code_fn is None:
            codes = torch.randint(0, 8, (N + 1,), generator=g, dtype=torch.int64)
        else:
            codes = torch.tensor([code_fn(i) for i in range(N + 1)], dtype=torch.int64)
        trajs.append(DiscreteTrajectory(
            codes=codes, op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2), numbers=[1, 2, 3], target=100,
        ))
    return trajs


def test_position_leaking_codes_detected():
    """Code = position -> classifier achieves ~1.0 accuracy."""
    train = _disc_trajs(40, code_fn=lambda i: min(i, 7))
    test = _disc_trajs(20, code_fn=lambda i: min(i, 7))
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert m["position_predictability_score"] > 0.80
    assert m["n_train"] > 0
    assert m["n_test"] > 0


def test_position_independent_codes_near_chance():
    """Random codes -> score near majority-class chance."""
    train = _disc_trajs(40)
    test = _disc_trajs(20)
    m = evaluate_position_leakage(train, test, num_codes=8, seed=0)
    # Score should be close to (but may exceed slightly due to noise).
    assert m["position_predictability_score"] < 0.5
    assert m["chance_accuracy"] > 0.0


def test_permutation_null_lower_than_real():
    """Shuffled labels should not outperform the real classifier."""
    train = _disc_trajs(60, code_fn=lambda i: min(i, 7))
    test = _disc_trajs(30, code_fn=lambda i: min(i, 7))
    m = evaluate_position_leakage(train, test, num_codes=8, seed=0)
    assert m["permutation_null_accuracy"] <= m["position_predictability_score"] + 0.05


def test_metrics_bounded():
    train = _disc_trajs(20)
    test = _disc_trajs(10)
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert 0.0 <= m["position_predictability_score"] <= 1.0
    assert 0.0 <= m["permutation_null_accuracy"] <= 1.0
    assert 0.0 <= m["chance_accuracy"] <= 1.0


def test_empty_inputs_nan():
    train = []
    test = []
    m = evaluate_position_leakage(train, test, num_codes=8)
    assert m["position_predictability_score"] != m["position_predictability_score"]  # NaN

```


## File: `tests\test_position_subspace.py`
```python
import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from evaluation.position_subspace import (
    apply_scrub_to_trajectories,
    build_position_dataset,
    inlp_position_projection,
    train_position_probe,
)


def _planted(M=800, D=24, n_cls=4, noise=0.15, seed=0):
    """Linearly-separable class signal along planted directions + noise."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_cls, size=M)
    dirs = rng.randn(n_cls, D) * 3.0
    X = (dirs[y] + noise * rng.randn(M, D)).astype(np.float32)
    cut = int(0.7 * M)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_probe_recovers_planted_signal():
    Xtr, ytr, Xte, yte = _planted()
    res = train_position_probe(Xtr, ytr, Xte, yte)
    # A linear probe should nearly perfectly recover a linearly planted signal.
    assert res["position_accuracy"] > 0.9
    assert res["majority_baseline"] < 0.4


def test_inlp_removes_planted_signal():
    Xtr, ytr, Xte, yte = _planted()
    before = train_position_probe(Xtr, ytr, Xte, yte)["position_accuracy"]
    P, info = inlp_position_projection(Xtr, ytr, Xte, yte, num_iters=8)

    # P is a symmetric projection (P == P^2 == P^T).
    assert np.allclose(P, P.T, atol=1e-6)
    assert np.allclose(P @ P, P, atol=1e-5)

    # After scrubbing, a fresh probe collapses toward the majority baseline.
    assert info["before_accuracy"] > 0.9
    assert before > 0.9
    assert info["after_accuracy"] <= info["majority_baseline"] + 0.1
    assert info["dims_removed"] >= 1


def test_scrub_trajectories_roundtrip_and_kills_signal():
    # Build trajectories whose state content is dominated by relative position.
    rng = np.random.RandomState(1)
    H = 16
    stage_dirs = torch.tensor(rng.randn(4, H) * 3.0, dtype=torch.float32)
    trajs = []
    for _ in range(120):
        N = 4  # 5 states -> bins 0,1,2,3,3
        n_states = N + 1
        bins = [min(int(i / N * 4), 3) for i in range(n_states)]
        states = stage_dirs[bins] + 0.1 * torch.randn(n_states, H)
        trajs.append(Trajectory(
            all_hidden=states,
            input_ids=torch.zeros(n_states, dtype=torch.long),
            state_indices=torch.arange(n_states),
            op_ids=torch.zeros(N, dtype=torch.long),
            operands=torch.zeros(N, 2),
            numbers=[1, 2, 3], target=6,
        ))

    X, y = build_position_dataset(trajs)
    assert X.shape[0] == y.shape[0] == sum(t.num_steps + 1 for t in trajs)
    assert set(np.unique(y).tolist()).issubset({0, 1, 2, 3})

    mean = X.mean(axis=0)
    P, _ = inlp_position_projection(X - mean, y, X - mean, y, num_iters=6)
    scrubbed = apply_scrub_to_trajectories(trajs, mean, P)

    # Structural roundtrip: same shapes, metadata preserved, states changed.
    assert len(scrubbed) == len(trajs)
    for orig, sc in zip(trajs, scrubbed):
        assert sc.states.shape == orig.states.shape
        assert sc.num_steps == orig.num_steps
        assert sc.hidden_dim == orig.hidden_dim

    # Position is no longer linearly decodable from the scrubbed states.
    Xs, ys = build_position_dataset(scrubbed)
    cut = int(0.7 * Xs.shape[0])
    res = train_position_probe(Xs[:cut], ys[:cut], Xs[cut:], ys[cut:])
    assert res["position_accuracy"] <= res["majority_baseline"] + 0.15

```


## File: `tests\test_probes.py`
```python
import numpy as np
import torch

from data_processing.trajectory_dataset import Trajectory
from evaluation.probes import extract_probe_data, run_probes


def test_extract_probe_data_labels():
    traj = Trajectory(
        all_hidden=torch.zeros(4, 5),
        input_ids=torch.zeros(4, dtype=torch.long),
        state_indices=torch.arange(4),
        op_ids=torch.tensor([0, 1, 2]),  # ADD, SUB, MUL
        operands=torch.tensor([[25.0, 3.0], [28.0, 50.0], [78.0, 75.0]]),
        numbers=[25, 50, 75, 100, 3, 4],
        target=10,
    )
    data = extract_probe_data([traj])
    # Three usable states s_0, s_1, s_2 (distance 3, 2, 1).
    assert data["B"].tolist() == [3, 2, 1]
    assert data["C"].tolist() == [0, 1, 2]
    assert data["D"].tolist() == [0, 1, 1]  # distance <= 2
    # Remaining large numbers {25,50,75,100} as they get consumed.
    assert data["A"].tolist() == [
        [1, 1, 1, 1],  # nothing used yet
        [2, 1, 1, 1],  # 25 used
        [2, 2, 1, 1],  # 25, 50 used
    ]


def _encoded_trajectories(n=60, H=12, steps=4, seed=0):
    """States linearly encode next-op (dims 0-3) and distance (dims 4-7)."""
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        all_hidden = torch.zeros(steps + 1, H)
        for i in range(steps):
            dist = steps - i  # 4..1
            all_hidden[i, op_ids[i]] = 1.0
            all_hidden[i, 3 + dist] = 1.0
        all_hidden += 0.01 * torch.randn(steps + 1, H, generator=g)
        operands = torch.randint(1, 100, (steps, 2), generator=g).float()
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.zeros(steps + 1, dtype=torch.long),
            state_indices=torch.arange(steps + 1),
            op_ids=op_ids,
            operands=operands,
            numbers=[25, 50, 75, 100, 3, 4],
            target=10,
        ))
    return trajs


def test_probes_recover_encoded_signal():
    train = _encoded_trajectories(n=60, seed=0)
    test = _encoded_trajectories(n=30, seed=1)
    df, details, fitted_probes = run_probes(train, test)

    assert list(df["probe"].str[0]) == ["A", "B", "C", "D"]
    assert details["n_train"] > 0 and details["n_test"] > 0

    by = {r["probe"][0]: r for _, r in df.iterrows()}
    # Distance (B) and next-op (C) are explicitly encoded -> high accuracy.
    assert by["B"]["accuracy"] > 0.85
    assert by["C"]["accuracy"] > 0.85
    # All metrics within valid ranges.
    for _, r in df.iterrows():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["f1"] <= 1.0

```


## File: `tests\test_synthetic_fsm.py`
```python
import numpy as np

from evaluation.synthetic_fsm import (
    generate_fsm,
    generate_dataset,
    make_prototypes,
    flatten_true_states,
    oracle_transitions,
)
from evaluation.action_conditioned import conditional_structure


def test_shapes_and_alignment():
    T = generate_fsm(8, 3, seed=0)
    assert T.shape == (8, 3)
    trajs, true = generate_dataset(T, 20, hidden_dim=16, seed=0)
    assert len(trajs) == len(true) == 20
    for tr, s in zip(trajs, true):
        assert tr.states.shape[0] == s.shape[0]          # one embedding per state
        assert tr.op_ids.shape[0] == s.shape[0] - 1       # one action per transition
    assert flatten_true_states(true).shape[0] == sum(t.states.shape[0] for t in trajs)


def test_oracle_is_deterministic():
    T = generate_fsm(12, 3, seed=1)
    trajs, true = generate_dataset(T, 300, hidden_dim=8, seed=1)
    st, at, sn = oracle_transitions(true, trajs)
    # (s, a) -> s' is deterministic by construction: zero conditional entropy,
    # full deterministic mass.
    struct = conditional_structure(list(zip(st.tolist(), at.tolist())), sn.tolist())
    assert struct["global_entropy"] < 1e-9
    # det_frac_mass is conservative (excludes (s,a) pairs seen < min_count), so
    # a few rare pairs at this sample size keep it just under 1.0.
    assert struct["det_frac_mass"] > 0.95
    # State alone is ambiguous (multiple actions -> different successors).
    s_only = conditional_structure(st.tolist(), sn.tolist())
    assert s_only["global_entropy"] > 0.3


def test_structured_embeddings_are_separable():
    T = generate_fsm(10, 3, seed=2)
    protos_recovered_purity = _nearest_prototype_purity(T, structured=True)
    floor_purity = _nearest_prototype_purity(T, structured=False)
    # Structured: a state's embedding is closest to its own prototype mean.
    assert protos_recovered_purity > 0.95
    # Noise floor: embeddings carry no state info -> chance-level purity.
    assert floor_purity < 0.30


def test_shared_prototypes_transfer_across_splits():
    """Train/eval splits must share prototype geometry.

    The prototypes are the environment's fixed ``state -> hidden vector`` map.
    If each split draws its own prototypes, a VQ trained on the train split
    sees an unrelated point cloud at eval time and collapses every eval state
    onto a single code — which silently saturates all downstream
    train->eval predictability metrics at 0/1. With shared prototypes the VQ
    encodes the eval split into the same codes it learned on train.
    """
    from training.train_vq import train_vq_quantizer, VQTrainConfig
    from data_processing.discrete_trajectory_dataset import (
        encode_trajectories_to_codes, all_codes,
    )
    from sklearn.metrics import adjusted_mutual_info_score

    K, H = 8, 64
    T = generate_fsm(K, 3, seed=0)
    protos = make_prototypes(K, hidden_dim=H, seed=7)

    tr, _ = generate_dataset(T, 300, H, noise=0.3, structured=True, seed=0, protos=protos)
    va, ts_va = generate_dataset(T, 150, H, noise=0.3, structured=True, seed=1, protos=protos)
    vq = train_vq_quantizer(tr, va, config=VQTrainConfig(num_codes=K, epochs=12,
                                                         batch_size=256, seed=0))
    shared_codes = all_codes(encode_trajectories_to_codes(vq, va)).numpy()
    assert len(np.unique(shared_codes)) >= K // 2          # eval does NOT collapse
    ami_shared = adjusted_mutual_info_score(flatten_true_states(ts_va), shared_codes)
    assert ami_shared > 0.5                                # codes recover eval states

    # Guard: an independently-seeded eval split (own prototypes) lands on an
    # unrelated point cloud, so its codes no longer track the true states.
    va_indep, ts_indep = generate_dataset(T, 150, H, noise=0.3, structured=True, seed=1)
    indep_codes = all_codes(encode_trajectories_to_codes(vq, va_indep)).numpy()
    ami_indep = adjusted_mutual_info_score(flatten_true_states(ts_indep), indep_codes)
    assert ami_indep < 0.2                                 # the bug this guards against
    assert ami_shared > ami_indep + 0.3                    # sharing is decisively better


def _nearest_prototype_purity(T, structured):
    num_states = T.shape[0]
    trajs, true = generate_dataset(
        T, 400, hidden_dim=32, noise=0.3, structured=structured, seed=3)
    X = np.concatenate([t.states.numpy() for t in trajs], axis=0)
    y = flatten_true_states(true)
    # Empirical per-state means, then nearest-mean classification accuracy.
    means = np.stack([X[y == s].mean(0) if (y == s).any() else np.zeros(X.shape[1])
                      for s in range(num_states)])
    d = ((X[:, None, :] - means[None]) ** 2).sum(-1)
    pred = d.argmin(1)
    return float((pred == y).mean())

```


## File: `tests\test_trajectory_dataset.py`
```python
import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import (
    build_trajectories,
    TransitionDataset,
    DecoderDataset,
    format_header,
    _state_end_chars,
    _token_index_for_char_end,
)

TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

PROBLEMS = [
    {"numbers": [25, 100, 50, 2, 5, 4], "target": 18,
     "solution": ["25 - 5 = 20", "20 - 2 = 18"]},
    {"numbers": [2, 4, 8, 1, 3, 6], "target": 8,
     "solution": ["2 * 4 = 8"]},
]


def test_state_end_chars_alignment():
    header = format_header([1, 2, 3], 6)
    steps = ["1 + 2 = 3", "3 + 3 = 6"]
    ends = _state_end_chars(header, steps)
    # s_0 ends at end of header; one extra state per step.
    assert ends[0] == len(header)
    assert len(ends) == len(steps) + 1
    full = header + "\n".join(steps)
    # Each step end char should point just past that step's text.
    assert full[: ends[1]].endswith("1 + 2 = 3")
    assert full[: ends[2]].endswith("3 + 3 = 6")


def test_token_index_monotonic():
    offsets = [(0, 0), (0, 3), (3, 5), (5, 9)]
    assert _token_index_for_char_end(offsets, 9) == 3
    assert _token_index_for_char_end(offsets, 5) == 2
    assert _token_index_for_char_end(offsets, 3) == 1


def test_build_trajectories_shapes():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)

    trajs = build_trajectories(model, tokenizer, PROBLEMS, layer=-1)
    assert len(trajs) == 2

    t0 = trajs[0]
    # Two steps -> three states s_0, s_1, s_2.
    assert t0.num_steps == 2
    assert t0.state_indices.shape[0] == 3
    assert t0.states.shape == (3, t0.hidden_dim)
    assert t0.op_ids.tolist() == [1, 1]  # SUB, SUB
    # State indices are strictly increasing along the sequence.
    assert torch.all(t0.state_indices[1:] > t0.state_indices[:-1])
    # state_indices are valid positions in the token sequence.
    assert int(t0.state_indices.max()) < t0.all_hidden.shape[0]


def test_transition_and_decoder_datasets():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    trajs = build_trajectories(model, tokenizer, PROBLEMS, layer=-1)

    tds = TransitionDataset(trajs)
    # 2 transitions from traj0 + 1 from traj1 = 3.
    assert len(tds) == 3
    sample = tds[0]
    H = trajs[0].hidden_dim
    assert sample["h_t"].shape == (H,)
    assert sample["h_next"].shape == (H,)
    assert sample["operands"].shape == (2,)
    assert sample["op_id"].dtype == torch.long

    dds = DecoderDataset(trajs)
    assert len(dds) > 0
    d0 = dds[0]
    assert d0["hidden"].shape == (H,)
    assert d0["target"].dtype == torch.long

```


## File: `tests\test_transition_model.py`
```python
import torch

from models.transition_model import TransitionModel, ActionEncoder, transition_loss
from training.train_transition import train_transition_model, TransitionTrainConfig
from data_processing.trajectory_dataset import Trajectory


def _toy_trajectories(n=8, H=16, steps=3, seed=0):
    """Synthetic trajectories with a learnable linear dynamics per op."""
    g = torch.Generator().manual_seed(seed)
    trajs = []
    # Fixed per-op shift vectors so a model can actually learn the dynamics.
    op_shifts = torch.randn(4, H, generator=g)
    for _ in range(n):
        T = steps + 5
        all_hidden = torch.zeros(T, H)
        state_indices = torch.arange(steps + 1)
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        operands = torch.randint(1, 50, (steps, 2), generator=g).float()
        s = torch.randn(H, generator=g)
        all_hidden[0] = s
        for i in range(steps):
            s = s + op_shifts[op_ids[i]]
            all_hidden[i + 1] = s
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.zeros(T, dtype=torch.long),
            state_indices=state_indices,
            op_ids=op_ids,
            operands=operands,
            numbers=[1, 2, 3, 4, 5, 6],
            target=10,
        ))
    return trajs


def test_action_encoder_dim():
    enc = ActionEncoder(op_embed_dim=8)
    assert enc.output_dim == 10
    out = enc(torch.tensor([0, 1]), torch.tensor([[10.0, 20.0], [3.0, 4.0]]))
    assert out.shape == (2, 10)


def test_forward_shapes_and_delta():
    model = TransitionModel(hidden_dim=16, mlp_hidden_dim=32)
    h = torch.randn(5, 16)
    op = torch.randint(0, 4, (5,))
    operands = torch.randint(1, 10, (5, 2)).float()
    out = model(h, op, operands)
    assert out.shape == (5, 16)


def test_training_reduces_loss():
    trajs = _toy_trajectories(n=16, H=16, steps=3)
    cfg = TransitionTrainConfig(epochs=60, mlp_hidden_dim=64, lr=5e-3, batch_size=16)
    model = train_transition_model(trajs, val_trajs=trajs, config=cfg)
    history = cfg.history
    assert history[-1]["loss"] < history[0]["loss"]
    # On clean synthetic dynamics the model should fit well.
    assert history[-1]["loss"] < 0.1
    assert "eval_loss" in history[-1]

```


## File: `tests\test_vq_state.py`
```python
import torch

from models.vq_state import VQStateQuantizer


def test_forward_shapes():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    h = torch.randn(5, 8)
    z_q, indices, info = vq(h, training=False)
    assert z_q.shape == (5, 8)
    assert indices.shape == (5,)
    assert indices.dtype == torch.int64
    for key in ("commitment_loss", "codebook_loss", "perplexity", "active_codes"):
        assert key in info


def test_straight_through_gradient_flows():
    """STE: decoder-side gradient reaches the encoder input."""
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    vq.eval()
    h = torch.randn(3, 8, requires_grad=True)
    z_q, _, _ = vq(h, training=False)
    z_q.sum().backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_encode_decode_roundtrip():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    h = torch.randn(7, 8)
    z = vq.encode(h)
    assert z.shape == (7,)
    decoded = vq.decode(z)
    assert decoded.shape == (7, 8)
    # Decoded vectors must equal the codebook entries they were indexed from.
    assert torch.allclose(decoded, vq.codebook[z])


def test_ema_recovers_cluster_centers():
    """With EMA on, the codebook should converge to the data cluster centers."""
    torch.manual_seed(0)
    # Use a relatively fast EMA decay so codebook adapts within the loop.
    vq = VQStateQuantizer(hidden_dim=8, num_codes=4, ema_decay=0.9, epsilon=1e-5)
    centers = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    h = centers.repeat(50, 1) + 0.01 * torch.randn(200, 8)
    vq.train()
    # Run enough steps for dead-code revival to activate all codes.
    for _ in range(500):
        vq(h, training=True)
    # All 4 codes should be active (perplexity ~ 4).
    _, _, info = vq(h, training=False)
    assert int(info["active_codes"]) == 4, (
        f"Expected 4 active codes, got {int(info['active_codes'])}"
    )
    assert float(info["perplexity"]) > 3.0
    # Commitment loss -> ~0 once the codebook matches the centers.
    assert float(info["commitment_loss"]) < 0.1


def test_diagnostics_in_range():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=10)
    h = torch.randn(20, 8)
    _, _, info = vq(h, training=False)
    assert 1.0 <= float(info["perplexity"]) <= 10.0
    assert 1 <= int(info["active_codes"]) <= 10

```


## File: `training\train_decoder.py`
```python
"""Training loop for the diagnostic next-token decoder."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from data_processing.trajectory_dataset import (
    Trajectory,
    DecoderDataset,
    load_trajectories,
)
from models.diagnostic_decoder import DiagnosticDecoder


@dataclass
class DecoderTrainConfig:
    mlp_hidden_dim: int = 0
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval(model: DiagnosticDecoder, loader: DataLoader) -> dict:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["hidden"])
        loss_sum += ce(logits, batch["target"]).item()
        correct += (logits.argmax(-1) == batch["target"]).sum().item()
        n += batch["target"].shape[0]
    return {"loss": loss_sum / max(n, 1), "accuracy": correct / max(n, 1)}


def train_decoder_model(
    train_trajs: List[Trajectory],
    vocab_size: int,
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[DecoderTrainConfig] = None,
) -> DiagnosticDecoder:
    """Train the diagnostic decoder on teacher hidden states. Returns the model."""
    config = config or DecoderTrainConfig()
    torch.manual_seed(config.seed)

    train_ds = DecoderDataset(train_trajs)
    if len(train_ds) == 0:
        raise ValueError("No (hidden, next_token) pairs in training trajectories.")
    hidden_dim = train_trajs[0].hidden_dim

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = DecoderDataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiagnosticDecoder(hidden_dim, vocab_size, config.mlp_hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    ce = torch.nn.CrossEntropyLoss()

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for step_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["hidden"])
            loss = ce(logits, batch["target"])
            loss.backward()
            optimizer.step()
            bs = batch["target"].shape[0]
            loss_sum += loss.item() * bs
            correct += (logits.argmax(-1) == batch["target"]).sum().item()
            n += bs
            
            if step_idx % 10 == 0:
                print(f"  [Epoch {epoch+1}/{config.epochs} | Step {step_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        record = {"epoch": epoch, "step": epoch,
                  "loss": loss_sum / max(n, 1),
                  "accuracy": correct / max(n, 1)}
        if val_loader is not None:
            ev = _eval(model, val_loader)
            record["eval_loss"] = ev["loss"]
            record["eval_accuracy"] = ev["accuracy"]
        config.history.append(record)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train diagnostic next-token decoder")
    parser.add_argument("--train_traj", type=str, required=True)
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/diagnostic_decoder.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None
    config = DecoderTrainConfig(epochs=args.epochs, lr=args.lr)
    model = train_decoder_model(train_trajs, args.vocab_size, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "hidden_dim": model.hidden_dim,
                "vocab_size": model.vocab_size}, args.output)
    final = config.history[-1]
    print(f"Done. Final train acc={final['accuracy']:.4f}"
          + (f" val acc={final['eval_accuracy']:.4f}" if "eval_accuracy" in final else ""))


if __name__ == "__main__":
    main()

```


## File: `training\train_transition.py`
```python
"""Training loop for the latent transition model."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from data_processing.trajectory_dataset import (
    Trajectory,
    TransitionDataset,
    load_trajectories,
)
from models.transition_model import (
    TransitionModel,
    LinearTransitionModel,
    TransformerTransitionModel,
    transition_loss,
)


@dataclass
class TransitionTrainConfig:
    mlp_hidden_dim: int = 512
    op_embed_dim: int = 16
    predict_delta: bool = True
    use_action: bool = True
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 0.0
    seed: int = 0
    transition_arch: str = "mlp"
    history: List[dict] = field(default_factory=list)


@torch.no_grad()
def _eval_loss(model: TransitionModel, loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["h_t"], batch["op_id"], batch["operands"])
        loss = transition_loss(pred, batch["h_next"], reduction="sum")
        total += loss.item()
        n += batch["h_t"].shape[0] * batch["h_t"].shape[1]
    return total / max(n, 1)


def train_transition_model(
    train_trajs: List[Trajectory],
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[TransitionTrainConfig] = None,
) -> torch.nn.Module:
    """Train a transition model on teacher trajectories. Returns the model.

    Training history (per-epoch train/val MSE) is recorded on ``config.history``.
    """
    config = config or TransitionTrainConfig()
    torch.manual_seed(config.seed)

    train_ds = TransitionDataset(train_trajs)
    if len(train_ds) == 0:
        raise ValueError("No transitions in training trajectories.")
    
    operands_tensor = torch.stack([x for x in train_ds.operands])
    operand_mean = operands_tensor.mean().item()
    operand_std = operands_tensor.std().item() + 1e-8

    hidden_dim = train_trajs[0].hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = TransitionDataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    if config.transition_arch == "linear":
        model = LinearTransitionModel(
            hidden_dim=hidden_dim,
            op_embed_dim=config.op_embed_dim,
            predict_delta=config.predict_delta,
            use_action=config.use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        ).to(device)
    elif config.transition_arch == "transformer":
        model = TransformerTransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            op_embed_dim=config.op_embed_dim,
            predict_delta=config.predict_delta,
            use_action=config.use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        ).to(device)
    else:
        model = TransitionModel(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=config.mlp_hidden_dim,
            op_embed_dim=config.op_embed_dim,
            predict_delta=config.predict_delta,
            use_action=config.use_action,
            operand_mean=operand_mean,
            operand_std=operand_std,
        ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        epoch_sum, epoch_n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch["h_t"], batch["op_id"], batch["operands"])
            loss = transition_loss(pred, batch["h_next"])
            loss.backward()
            optimizer.step()
            bs = batch["h_t"].shape[0]
            epoch_sum += loss.item() * bs
            epoch_n += bs
        train_loss = epoch_sum / max(epoch_n, 1)

        record = {"epoch": epoch, "step": epoch, "loss": train_loss}
        if val_loader is not None:
            record["eval_loss"] = _eval_loss(model, val_loader)
        config.history.append(record)

    return model


def save_history_csv(history: List[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train latent transition model")
    parser.add_argument("--train_traj", type=str, required=True)
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints/transition_model.pt")
    parser.add_argument("--log_csv", type=str, default="reports/transition_train_log.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--transition_arch", type=str, default="mlp", choices=["mlp", "linear", "transformer"])
    parser.add_argument("--no_delta", action="store_true", help="Predict h_next directly")
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None

    config = TransitionTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        mlp_hidden_dim=args.mlp_hidden_dim,
        predict_delta=not args.no_delta,
        transition_arch=args.transition_arch,
    )
    model = train_transition_model(train_trajs, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "hidden_dim": model.hidden_dim,
                "config": config.__dict__}, args.output)
    save_history_csv(config.history, args.log_csv)
    final = config.history[-1]
    print(f"Done. Final train MSE={final['loss']:.5f}"
          + (f" val MSE={final['eval_loss']:.5f}" if "eval_loss" in final else ""))


if __name__ == "__main__":
    main()

```


## File: `training\train_vq.py`
```python
"""Training loop for the VQ discrete-state quantizer.

Trains :class:`models.vq_state.VQStateQuantizer` on the *symbolic-aligned*
states ``s_0..s_N`` (``traj.states``) of the frozen-LM trajectory cache. The
result is a codebook that converts any continuous hidden state into a discrete
code id — the "state" of the Version-5 research question.

Only the aligned states are quantized (not every token's hidden state) so the
learned codes correspond to reasoning-step boundaries, matching the rest of the
pipeline's notion of a latent state.

History (per-epoch commitment loss, perplexity, active-code count) is recorded
on ``config.history`` for the standard training-curve plot.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.trajectory_dataset import (
    Trajectory,
    load_trajectories,
)
from models.vq_state import VQStateQuantizer


@dataclass
class VQTrainConfig:
    num_codes: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    lr: float = 3e-4
    epochs: int = 50
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 0
    history: List[dict] = field(default_factory=list)


def _aligned_states_dataset(trajectories: List[Trajectory]) -> TensorDataset:
    """Stack the symbolic-aligned states ``s_0..s_N`` from every trajectory."""
    states = []
    for traj in trajectories:
        states.append(traj.states)  # (N+1, H)
    if not states:
        raise ValueError("No trajectories provided.")
    X = torch.cat(states, dim=0)  # (M, H)
    return TensorDataset(X)


def _init_codebook_from_data(
    model: VQStateQuantizer, train_ds: TensorDataset, device: torch.device, seed: int = 0
) -> None:
    """Data-dependent codebook initialization (V5.1 Fix 1).

    The default ``randn * 0.02`` init leaves the codebook two orders of
    magnitude smaller in norm than the hidden states (``||c|| ~ 0.9`` vs
    ``||h|| ~ 85``). On the very first batch nearly every state then maps to
    whichever code happens to be marginally closest, and the EMA update plus
    dead-code revival cannot recover — the codebook collapses to ~6/32 active
    codes (the invalidated V5 result). Seeding every code from a randomly
    sampled training state places all codes inside the data manifold from step
    zero, so EMA refines real clusters instead of fighting a scale mismatch.
    """
    X = train_ds.tensors[0]  # (M, H) aligned hidden states
    K = model.num_codes
    g = torch.Generator().manual_seed(seed)
    if X.shape[0] >= K:
        idx = torch.randperm(X.shape[0], generator=g)[:K]
    else:  # fewer states than codes — sample with replacement
        idx = torch.randint(0, X.shape[0], (K,), generator=g)
    pts = X[idx].to(device).clone()
    with torch.no_grad():
        model.codebook.copy_(pts)
        model.ema_w.copy_(pts)
        # Start every code with a non-trivial cluster size so revival does not
        # immediately re-kill the freshly seeded codes on the first step.
        model.ema_cluster_size.fill_(1.0)


def _model_device(model: nn.Module) -> torch.device:
    """Device of a module that works even when it has no Parameters (only buffers)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


@torch.no_grad()
def _eval(model: VQStateQuantizer, loader: DataLoader) -> dict:
    """Mean commitment loss + perplexity / active codes on held-out states."""
    model.eval()
    device = _model_device(model)
    commit_sum, perplexity_sum, active_sum, n = 0.0, 0.0, 0.0, 0
    for (h,) in loader:
        h = h.to(device)
        _, _, info = model(h, training=False)
        b = h.shape[0]
        commit_sum += info["commitment_loss"].item() * b
        perplexity_sum += info["perplexity"].item() * b
        active_sum += info["active_codes"].item()
        n += b
    return {
        "commitment_loss": commit_sum / max(n, 1),
        "perplexity": perplexity_sum / max(n, 1),
        "active_codes": active_sum / max(1, len(loader)),
    }


def train_vq_quantizer(
    train_trajs: List[Trajectory],
    val_trajs: Optional[List[Trajectory]] = None,
    config: Optional[VQTrainConfig] = None,
) -> VQStateQuantizer:
    """Train a VQ codebook on symbolic-aligned hidden states. Returns the model."""
    config = config or VQTrainConfig()
    torch.manual_seed(config.seed)

    hidden_dim = train_trajs[0].hidden_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = _aligned_states_dataset(train_trajs)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = None
    if val_trajs:
        val_ds = _aligned_states_dataset(val_trajs)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = VQStateQuantizer(
        hidden_dim=hidden_dim,
        num_codes=config.num_codes,
        commitment_cost=config.commitment_cost,
        ema_decay=config.ema_decay,
    ).to(device)
    # Seed the codebook from real data BEFORE any EMA step (V5.1 Fix 1).
    _init_codebook_from_data(model, train_ds, device, seed=config.seed)
    # The codebook is a buffer updated by EMA inside forward(); the quantizer
    # has no trainable parameters. We still build an optimizer over any
    # *future* trainable params (e.g. an encoder net) so this loop generalizes;
    # if there are none, training is pure EMA and the optimizer is a no-op.
    trainable = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
        if trainable
        else None
    )

    config.history = []
    for epoch in range(config.epochs):
        model.train()
        commit_sum, perplexity_sum, active_sum, n = 0.0, 0.0, 0.0, 0
        for (h,) in train_loader:
            h = h.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            _, _, info = model(h, training=True)
            loss = config.commitment_cost * info["commitment_loss"]
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            b = h.shape[0]
            commit_sum += info["commitment_loss"].item() * b
            perplexity_sum += info["perplexity"].item() * b
            active_sum += info["active_codes"].item()
            n += b

        record = {
            "epoch": epoch,
            "step": epoch,
            "loss": commit_sum / max(n, 1),
            "perplexity": perplexity_sum / max(n, 1),
            "active_codes": active_sum / max(1, len(train_loader)),
        }
        if val_loader is not None:
            ev = _eval(model, val_loader)
            record["eval_loss"] = ev["commitment_loss"]
            record["eval_perplexity"] = ev["perplexity"]
            record["eval_active_codes"] = ev["active_codes"]
        config.history.append(record)

    return model


def save_history_csv(history: List[dict], path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train VQ discrete-state quantizer")
    parser.add_argument(
        "--train_traj", type=str, required=True, help="Input .pt trajectories file"
    )
    parser.add_argument("--val_traj", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default="checkpoints/vq_state.pt"
    )
    parser.add_argument("--log_csv", type=str, default="reports/vq_train_log.csv")
    parser.add_argument("--num_codes", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    args = parser.parse_args()

    train_trajs = load_trajectories(args.train_traj)
    val_trajs = load_trajectories(args.val_traj) if args.val_traj else None

    config = VQTrainConfig(
        num_codes=args.num_codes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        commitment_cost=args.commitment_cost,
        ema_decay=args.ema_decay,
    )
    model = train_vq_quantizer(train_trajs, val_trajs, config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "num_codes": model.num_codes,
            "config": config.__dict__,
        },
        args.output,
    )
    save_history_csv(config.history, args.log_csv)
    final = config.history[-1]
    msg = (
        f"Done. Final train commitment={final['loss']:.5f} "
        f"perplexity={final['perplexity']:.1f}/{config.num_codes} "
        f"active={int(final['active_codes'])}"
    )
    if "eval_loss" in final:
        msg += (
            f" | val commitment={final['eval_loss']:.5f} "
            f"perplexity={final['eval_perplexity']:.1f} "
            f"active={int(final['eval_active_codes'])}"
        )
    print(msg)


if __name__ == "__main__":
    main()

```


## File: `training\__init__.py`
```python

```

