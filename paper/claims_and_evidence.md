# Claims and Evidence

This document maps our core claims to the empirical evidence gathered.

| Claim | Evidence | Status / Notes |
| :--- | :--- | :--- |
| **Our methodology can detect latent state transitions if they exist.** | **Positive control works.** A synthetic FSM encoded into trajectories yields a perfectly coherent transition model (high AMI, near-zero MSE, exact sequence reproduction). | Confirmed. Validates the VQ + MLP methodology. |
| **Transformers encode structural/positional metadata in representations.** | **Position is decodable.** Linear probes and INLP identify a low-dimensional positional subspace. Removing it (INLP scrubbing) drastically damages standard generation. | Confirmed. |
| **Next-token predictors do not learn abstract, reusable "planning states".** | **MLP ≈ ActionBigram.** In the symbolic domains, the trained non-linear transition model `MLP(z_t, a_t)` performs no better at predicting `z_{t+1}` than a simple `ActionBigram(z, a)` or `MLP(z_t)` baseline. | Confirmed for TinyLlama 1.1B and Qwen2.5 1.5B. **[Qwen2.5 7B Pending]**. |
| **This failure to learn planning states is consistent across models.** | **Cross-model replication.** Both TinyLlama-1.1B and Qwen2.5-1.5B show the same failure mode: high initial coherence that rapidly decays at depth 2-3, driven by representation collapse, not action-conditioned state transition. | Confirmed. **[Mistral-7B Pending]**. |
| **The failure is not domain-specific or due to dataset leakage.** | **Game24 leak-free replication.** Both the Countdown domain and the strictly partitioned Game24 domain (sampled from 1,346 unique solvable hands) show the same dynamics. | Confirmed Game24 baseline. |
| **Latent paths for equivalent symbolic states are not identical.** | **Permutation robustness fails.** Different paths to the exact same intermediate arithmetic state do not map to the same VQ code or vector neighborhood. | Confirmed. The representation is heavily path-dependent. |
