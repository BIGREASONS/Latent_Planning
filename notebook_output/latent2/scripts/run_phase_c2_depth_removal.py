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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
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

        print(
            f"  Epoch {epoch+1}/{num_epochs} [Lambda: {lambda_val:.2f}] | State (Op) Loss: {total_s_loss/len(train_loader):.3f} | Depth Loss: {total_d_loss/len(train_loader):.3f}"
        )

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

    acc_d_before, acc_s_before = eval_probes(
        X_train, yd_train, ys_train, X_test, yd_test, ys_test
    )
    acc_d_after, acc_s_after = eval_probes(
        Z_train, yd_train, ys_train, Z_test, yd_test, ys_test
    )

    print(
        f"  [BEFORE - Raw h] Depth Acc: {acc_d_before*100:.1f}%, State (Op) Acc: {acc_s_before*100:.1f}%"
    )
    print(
        f"  [AFTER  - Proj z] Depth Acc: {acc_d_after*100:.1f}%, State (Op) Acc: {acc_s_after*100:.1f}%"
    )

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
        f.write(
            "This experiment strips token position/depth information from the hidden state while preserving symbolic state information. Because the combinatorial explosion of the Countdown game yields 19,771 unique exact symbolic states (with $<2$ examples per class), we define 'State Information' as **Probe C (Next Symbolic Operation)**, allowing the network to train on all 20,000+ states.\n\n"
        )

        f.write("## Evaluation: Independent Linear Probes\n\n")
        f.write("| Representation | Depth Accuracy | State Accuracy (Next Op) |\n")
        f.write("|---|---|---|\n")
        f.write(
            f"| Raw $h$ (2048-dim) | {acc_d_before*100:.1f}% | {acc_s_before*100:.1f}% |\n"
        )
        f.write(
            f"| Projected $z$ (256-dim) | {acc_d_after*100:.1f}% | {acc_s_after*100:.1f}% |\n"
        )
        f.write(
            f"| *Chance Baseline* | *{chance_depth*100:.1f}%* | *{chance_state*100:.1f}%* |\n\n"
        )

        f.write("## Final Verdict\n\n")
        f.write(verdict + "\n")

    print(f"\n[Phase C.2] Done. Verdict: {'SUCCESS' if success else 'FAILURE'}")


if __name__ == "__main__":
    main()
