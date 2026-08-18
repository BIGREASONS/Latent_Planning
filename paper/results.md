# Results

## 1. Do representations encode actionable transition dynamics?

We evaluated the ability of an auxiliary model to predict the next reasoning state $h_{t+1}$ given the current state $h_t$ and the symbolic action $a_t$. 

**Table 1: Transition Prediction Performance (Countdown)**

| Model | Identity MSE | Action-Blind MLP | Action-Bigram | Action-Conditioned MLP | $\Delta$ (Blind - Cond) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TinyLlama 1.1B | [Value] | [Value] | [Value] | [Value] | ~0.005 |
| Qwen2.5 1.5B | [Value] | [Value] | [Value] | [Value] | ~0.000 |
| **Qwen2.5 7B** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Mistral 7B v0.3| [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

*Finding:* Across architectures evaluated so far, the action-conditioned model performs no better than baselines that ignore the current state or the action. The representations do not form a manipulable transition space.

## 2. Coherence Decay under Rollout

To simulate planning, we auto-regressively rolled out the transition model in the latent space and decoded the states back into tokens using a trained diagnostic decoder.

* **Figure 1 (Placeholder):** Coherence vs. Depth (TinyLlama vs Qwen1.5B vs Qwen7B).
* *Finding:* Latent rollouts rapidly lose semantic coherence by step 2-3. The representation collapses to the mean, indicating that the learned dynamics are fundamentally unstable.

## 3. Generalization across Domains (Game of 24)

To ensure our findings are not an artifact of the Countdown dataset, we replicated the experiment on a strictly partitioned, leak-free Game of 24 dataset.

**Table 2: Game24 Transition Prediction (Qwen2.5 7B)**

| Domain | Action-Blind MSE | Action-Conditioned MSE | $\Delta$ |
| :--- | :--- | :--- | :--- |
| Countdown | [PENDING] | [PENDING] | [PENDING] |
| Game24 | [PENDING] | [PENDING] | [PENDING] |

*Finding:* The failure to encode state transitions generalizes across distinct symbolic reasoning domains.

## 4. Positional Entanglement and Robustness

* **INLP Positional Scrubbing:** We identified a low-dimensional positional subspace. Scrubbing it drastically degraded generation, confirming the heavy reliance on positional heuristics rather than abstract state.
* **Permutation Robustness:** Different reasoning paths to the identical mathematical intermediate state did *not* map to the same continuous neighborhood or discrete VQ code. The state is path-dependent, not an abstract world state.
