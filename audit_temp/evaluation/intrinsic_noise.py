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
