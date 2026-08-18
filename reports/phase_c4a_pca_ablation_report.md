# Phase C.4A — PCA Ablation Sweep Report

This audit explicitly removes the top (highest-variance) principal components to test if Depth is a low-rank nuisance variable that can be sliced off without harming the distributed symbolic state. The PCA was strictly fit on the Train trajectories.

**Train-Test State Coverage**: 0.0% (0/12 unique symbolic states in Test exist in Train)

**Prefix collisions masked**: 0/12 test states (0.0%) had >=1 byte-identical train neighbor (deterministic-LM prefix confound); these are excluded from retrieval.

## Controls

Retrieval/probe scores must be read against these. The PERMUTED null shuffles train labels (representation fixed) — anything at this level is noise. The raw-operand baseline probes only the bag of numbers listed in the prompt (zero computation).

| Control | Depth Acc | Probe A (Exact) | Probe C | Retrieval (Top-1) | Retrieval (Top-5) |
|---|---|---|---|---|---|
| PERMUTED null | 8.3% | 8.3% | 33.3% | 0.0% | 0.0% |
| Raw-operand baseline | 33.3% | 41.7% | 41.7% | 0.0% | 0.0% |

## Ablation Results

| Removed PCs | Depth Acc | Probe A (Exact) | Probe A (Hamming) | Retrieval (Top-1) | Gain | Retrieval (Top-5) | Retrieval (Top-10) | Within/Between |
|---|---|---|---|---|---|---|---|---|
| 0 | 75.0% | 16.7% | 50.0% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |
| 1 | 75.0% | 16.7% | 45.8% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |
| 2 | 75.0% | 16.7% | 50.0% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |
| 5 | 50.0% | 16.7% | 45.8% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |
| 10 | 50.0% | 16.7% | 56.2% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |
| 20 | 25.0% | 16.7% | 54.2% | 0.0% | 1.00x | 0.0% | 0.0% | 0.000 |

## Verdict

**PERSISTENT.**
The ablation had unexpected effects; further investigation is required.