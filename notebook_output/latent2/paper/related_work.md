# Related Work

## 1. Latent Planning and State Space Models

Traditional planning algorithms operate over explicit, symbolic states. Recent reinforcement learning approaches, notably MuZero (Schrittwieser et al., 2020), demonstrate that planning can be performed entirely within a learned latent space. These models explicitly train a transition function $s_{t+1} = T(s_t, a_t)$ and optimize it for value prediction and reward. 

In natural language processing, similar ideas are emerging. Coconut (Training Large Language Models to Reason in a Continuous Latent Space) explores whether LMs can learn to reason over multiple steps directly in their hidden continuous space, bypassing discrete token generation.

## 2. Next-Token Prediction vs. Abstract Reasoning

A core debate in modern AI is whether autoregressive next-token prediction on massive corpora naturally induces abstract models of the world. While models can emit text that *looks* like planning (Chain of Thought), it is unclear if the internal representations correspond to manipulable world states or merely sequential heuristics.

Our work investigates this explicitly by testing if the internal states of standard LMs, when isolated, obey the Markovian transition dynamics expected of a true world model.

## 3. Mechanistic Interpretability and Hidden-State Probing

Mechanistic interpretability aims to reverse-engineer the representations learned by neural networks. Probing (Alain & Bengio, 2016) has been widely used to decode syntactic and semantic features from hidden states. Recent work has identified linear representations for concepts like truthfulness, spatial layout, and board game states (e.g., Othello-GPT).

We employ linear probing to decode symbolic operands and operations from reasoning traces, but extend this by analyzing the *dynamics* of these representations over time, rather than their static decodability.

## 4. State Abstraction and Positional Entanglement

A key challenge in analyzing LM representations is the heavy entanglement of semantic information with positional encodings (Rotary Position Embeddings, absolute indices). Work on Iterative Nullspace Projection (Ravfogel et al., 2020) provides tools for removing specific linear features. We utilize INLP to control for the artifact of positional leakage, ensuring our transition models evaluate logical state rather than sequence depth.
