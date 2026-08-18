# Paper Outline: Do LMs Learn Reusable Planning States?

## 1. Introduction
- The illusion of planning in Chain-of-Thought.
- Core research question: Do hidden states encode an abstract, Markovian transition model?
- Summary of findings (negative result across scales).

## 2. Related Work
- Latent planning & MuZero
- Next-token prediction limits
- Mechanistic interpretability (probes)
- State abstraction

## 3. Methodology
- **Extraction:** Aligning text steps to latent state trajectories $h_t$.
- **Tasks:** Countdown, Game of 24 (strict symbolic grammar).
- **Transition Modeling:** $T(h_t, a_t) \rightarrow h_{t+1}$ vs. baselines.
- **Controls:** Synthetic FSM (positive control), INLP (positional scrubbing).

## 4. Results
- **Failure of Transition Dynamics:** MLP $\approx$ Action-Bigram.
- **Coherence Decay:** Rollouts collapse at depth 2-3.
- **Path Dependence:** Permutation robustness fails.
- **Positional Entanglement:** INLP results.
- **Scaling:** TinyLlama $\rightarrow$ Qwen1.5B $\rightarrow$ Qwen7B $\rightarrow$ Mistral7B.

## 5. Discussion & Limitations
- Why next-token prediction fails to build state spaces.
- Implications for future architectures (e.g., Coconut).
- Limitations (frozen models, decoder bottleneck, scale ceiling).

## 6. Conclusion
- Summary and final verdict on the latent reasoning debate.

## Appendix
- A. Experimental Details & Hyperparameters.
- B. Game24 Dataset Construction (Leak-free 1346 pool).
- C. Diagnostic Decoder specifics.
