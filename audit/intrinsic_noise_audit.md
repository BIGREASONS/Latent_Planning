# Intrinsic Noise Audit (HIGHEST PRIORITY)

Source: `extracted_results/reports/intrinsic_noise_report.md`, `intrinsic_noise.csv`, `state_statistics.md`, and the `train.pt` trajectory tensors (5000 trajectories, 20,022 symbolic-state hidden vectors, TinyLlama 2048-dim). All numbers below independently recomputed from the raw tensors unless marked "(report)".

## Reported claims

| Metric | Report value |
|---|---|
| Within-state cosine | 0.927 ± 0.066 |
| Between-state cosine | 0.687 ± 0.244 |
| Within-state L2 | 28.819 ± 12.504 |
| Between-state L2 | 61.935 ± 28.186 |
| Top-1 retrieval | 24.6% |
| Top-5 retrieval | 39.3% |
| Verdict | "highly history-dependent (Case B)" |

## Independent reproduction

| Metric (my recompute) | Value | Matches report? |
|---|---|---|
| Within-state cross-traj cosine | **0.929** (n≈200 pairs) | ✓ (0.927) |
| Between-state random cosine | **0.656–0.662** | ✓ (0.687) |
| **Global random-pair cosine (anisotropy floor)** | **0.664 ± 0.242** | new |
| **Same-step, different-state cosine (position confound)** | **0.889** (n=2499) | new |

### The decisive confound

The report frames within (0.93) vs between (0.69) as a 0.24 gap and reads it as "barely Markovian." But:

- **Anisotropy floor = 0.664.** Two *completely unrelated* TinyLlama hidden states already cosine at 0.66. So "between-state 0.69" is essentially the anisotropy floor — between-state similarity carries almost no information; it just reflects that this layer's hidden space is highly anisotropic (cone-shaped).
- **Same-step similarity = 0.889.** Hidden states from the **same reasoning-step position but different symbolic states** already cosine at 0.889. The within-state value (0.929) is only **+0.04 above the same-step confound.** → The apparent within-vs-between gap is **dominated by token position**, not symbolic-state identity.

Mean-centering test (remove the global mean to kill anisotropy):

| | within | between | gap |
|---|---|---|---|
| Raw cosine | 0.929 | 0.656 | 0.273 |
| **Mean-centered** | 0.722 | 0.025 | **0.697** |

After centering there *is* a residual within>between gap (0.70) — so some symbolic-state signal survives. **But** it is entangled with step-position (same-step=0.89 raw) and rests on a tiny sample. The honest statement is: *"there is weak, position-confounded state signal,"* not the report's clean dichotomy.

## Retrieval audit

- Report: Top-1 24.6%, Top-5 39.3%.
- **Not reproducible.** Under a defensible symbolic-state key `(target, numbers_multiset, step)`, only **379 of 20,022** states have any cross-trajectory positive; sampling 300 queries yields **n=7** eligible. I cannot reconstruct 24.6% — the report's positive-set definition and labeling come from `run_phase_c.py`, which **is not provided.** The metric is therefore **unauditable**, and the eligible-pair scarcity suggests the report used a *looser* key (e.g., target only), which would inflate apparent retrieval.
- **Position contamination confirmed:** Top-1 nearest neighbor shares the same reasoning-**step index** 85.7% of the time. The NN is matching *trajectory depth*, not symbolic state. Any retrieval number is contaminated by:
  - same step/position (85.7% same-step) ✓ confirmed
  - same trajectory — excluded in my recompute, status of report's exclusion unknown
  - same prefix / same action history — uncontrolled (steps with the same op-history will have near-identical context)

## Statistical power (from their own `state_statistics.md`)

- 240 unique symbolic states; **181 cross-history**; **average cluster size 2.05; max cluster size 3**; **198 positive pairs total.**
- Their own file emits: `[!WARNING] dataset may be underpowered for contrastive learning.`
- Within-state cosine 0.927 ± 0.066 is computed over ~200 pairs → standard error ≈ 0.066/√200 ≈ 0.005, but the *clusters* are only ~2–3 members, so pairs are non-independent and the effective n is closer to 181, not 200. CIs are wider than reported (no CI is reported at all).

## Answers to required questions

- **Do identical symbolic states converge?** Weakly, and mostly because they occur at the same step position. After removing position/anisotropy the residual signal is small and underpowered.
- **within vs between cosine/L2?** Reproduced: within 0.93 / between 0.66 cosine; within 28.8 / between 61.9 L2. **But the between value ≈ anisotropy floor, so the contrast is inflated.**
- **Top-1/5/10 retrieval?** Report 24.6/39.3%; **not reproducible** from supplied tensors+code; **85.7% same-step contamination** confirmed.
- **Contamination from same trajectory / prefix / action history / text?** Same-step (position) contamination is severe and confirmed; same-prefix/action-history is uncontrolled; same-trajectory exclusion in the report is unverifiable.

## Verdict

The intrinsic-noise diagnostic is the project's only real experiment, and it **points against clean representation**. But its specific numbers **overstate the within/between contrast** (between ≈ anisotropy floor), its retrieval metric is **not reproducible and position-contaminated (85.7%)**, and the whole thing is computed on an **admittedly underpowered** sample (max cluster 3, ~200 pairs, no CIs). Net: the *direction* of the conclusion (representation is noisy/history-dependent) is plausibly correct, but the *evidence* is confounded and underpowered, and the retrieval claim is **inflated**.
