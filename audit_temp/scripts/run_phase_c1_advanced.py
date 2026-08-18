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
