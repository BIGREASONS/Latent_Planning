# Representation Audit — Is the hidden state Markovian?

Goal: can symbolic state / action history / trajectory identity be recovered from the hidden state? If trajectory ID is easier than symbolic state → severe history dependence.

## What the data + missing code allow

The classifiers the audit requests (symbolic-state, action-history, trajectory-ID decoders) are **not in the repo**, and the labels needed to train them (parsed symbolic state per token) come from the missing `data_processing/trajectory_dataset.py` + action parser. So I run the diagnostic *proxies* that the raw tensors permit.

## Reproduced evidence (from train.pt)

1. **Step position is trivially recoverable.** Nearest-neighbor on raw hidden states retrieves a same-step vector **85.7%** of the time (chance ≈ 1/avg_traj_len ≈ 25%). The hidden state strongly encodes "how far into the trajectory I am."

2. **Anisotropy dominates raw geometry.** Random state-pair cosine = **0.664**; mean-centering drops between-state cosine to **0.025**. The representation lives in a narrow cone; most raw cosine "signal" is the cone, not content.

3. **Symbolic-state signal is weak and position-entangled.** Within-state cosine exceeds same-step cosine by only **+0.04** (0.929 vs 0.889). After centering, within>between gap is 0.70 but cannot be disentangled from step position with the available labels.

4. **Trajectory identity is implicitly easy.** Because (a) the full `cot` is in the prompt during extraction and (b) every trajectory has a unique target/number combination in 99% of cases, the hidden state carries trajectory-specific content. The report's own conclusion ("highly history-dependent, Case B") is the qualitative version of "trajectory ID is easier to recover than symbolic state."

## The Markov test, answered as far as possible

- **Can symbolic state be recovered?** Weakly — within-state clustering exists but is underpowered (max cluster 3) and confounded by position. Not demonstrated to be cleanly linearly decodable.
- **Can action history be recovered?** Not directly tested (no code), but step-position recoverability (85.7%) and full-solution-in-prompt strongly imply yes.
- **Can trajectory ID be recovered?** Almost certainly yes (unique problems + full cot in context). The report effectively concedes this.
- **Is trajectory ID easier than symbolic state?** Evidence points **yes** → **flag severe history dependence**, consistent with the report's Case B, but established by confounds (position, prompt content) rather than a clean controlled classifier.

## Verdict

The hidden states are **not demonstrated to be Markovian** and show strong signatures of **history/position dependence**. However, this conclusion is reached through proxies and the project's own admission, not through the required controlled classifiers (symbolic-state vs action-history vs trajectory-ID decoders trained on disjoint problems with position regressed out). The representation appears **not clean enough for latent planning**, but the *quality of evidence* for even this negative result is mediocre because the controlled experiment was never built.
