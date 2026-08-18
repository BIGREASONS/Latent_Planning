# Methods

## 1. Extracting Teacher Trajectories

We frame language models solving symbolic reasoning tasks as agents moving through a latent state space. Given a problem instance (e.g., a Countdown or Game of 24 puzzle) and a sequence of reasoning steps, we align the text generation to discrete reasoning steps.

Let the prompt be $c$ and the generated steps be $x_1, x_2, \dots, x_N$. For a frozen autoregressive LM, we define the latent state $h_t$ as the hidden state vector at the final token of step $t$ in a specific intermediate layer $L$. The initial state $h_0$ is the representation at the end of the prompt header. The "action" $a_t$ is the symbolic operation parsed from the step text (e.g., an operator ID and numeric operands).

This gives us a dataset of transition tuples: $(h_t, a_t, h_{t+1})$.

## 2. Discretization and Codebook Learning

To analyze the effective size of the state space, we train Vector Quantized (VQ) autoencoders on the continuous trajectories. The VQ bottleneck forces the continuous $h_t$ into a discrete codebook, allowing us to measure state reuse and path equivalence using information-theoretic metrics like Adjusted Mutual Information (AMI).

## 3. Transition Modeling

We test whether the representation supports abstract planning by training an auxiliary transition model $T_\theta$ to predict $h_{t+1}$ given $(h_t, a_t)$. 
If the LM inherently models an environment state transition, $T_\theta$ should successfully learn the dynamics.

We compare a non-linear Multi-Layer Perceptron $T_\text{MLP}(h_t, a_t)$ against three crucial baselines:
1. **Action-Blind:** $T(h_t)$ — predicts the next state ignoring the action.
2. **Action Bigram:** predicts $h_{t+1}$ by falling back to the average next-state given only the action $a_t$.
3. **Identity:** assumes $h_{t+1} \approx h_t$.

The critical metric is the difference in loss: $\Delta = \mathcal{L}(T_\text{MLP}(h_t)) - \mathcal{L}(T_\text{MLP}(h_t, a_t))$. If $\Delta \approx 0$, the representation does not encode actionable transition dynamics.

## 4. Methodological Controls

### Positive Control: Synthetic FSM
To guarantee our extraction and transition modeling pipeline can detect latent transitions if they exist, we inject a synthetic Finite State Machine into trajectories. We successfully recover the exact FSM transition matrix, yielding high AMI and near-perfect transition prediction.

### Position Scrubbing (INLP)
Autoregressive hidden states entangle semantic content with absolute positional encoding. We use Iterative Nullspace Projection (INLP) to identify the positional subspace and scrub it from the representations, verifying its causal role in generation and ensuring our transition models are not merely predicting "position $t+1$".
