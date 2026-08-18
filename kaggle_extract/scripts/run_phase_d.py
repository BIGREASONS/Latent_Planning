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
