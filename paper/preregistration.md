# Pre-registered Interpretation Rules

This document outlines the explicit decision rules established before executing the final multi-seed and action-control validation experiments. These rules strictly govern how the resulting empirical data will be interpreted for the manuscript.

## H1: Transition Capacity Expressiveness
- **Supporting Evidence:** Transformer consistently achieves higher Semantic Gain than both Linear and MLP across initialization seeds, and consistently reduces the Oracle Gap.
- **Refuting Evidence:** Transformer performs similarly to MLP within observed seed-to-seed variability.

## H2: Geometry vs Semantics
- **Supporting Evidence:** Geometric reconstruction metrics (Cosine, MSE) remain relatively stable while Semantic Gain differs meaningfully across transition architectures or action controls.
- **Refuting Evidence:** Improvements in Semantic Gain closely track improvements in geometric reconstruction metrics.

## H3: Representation vs Transition
- **Supporting Evidence:** Oracle evaluations (ceiling) significantly outperform all learned transition models across all evaluated settings.
- **Refuting Evidence:** A learned transition model closes the Oracle Gap to a negligible level.

## Action Conditioning Effectiveness
Expected hierarchy of Semantic Gain:
`Oracle > True Action > Shuffled Action ≈ Constant Action ≈ Blind`

- **If `True Action ≈ Shuffled Action`:** We will conclude that under the evaluated training objective and architecture, the transition model does not effectively exploit the action input. We will explicitly avoid claiming that "action conditioning never works".

## Multi-seed Stability Interpretation
- If the observed differences between architectures (or action controls) are substantially larger than the seed-to-seed variation, the ranking will be treated and reported as robust.
- If the differences are comparable to the variability, the comparison will be presented as inconclusive and we will refrain from overinterpreting it.

---
**Core Philosophy:**
> This paper is about understanding latent transition models, not proving that latent planning works.
We evaluate methodology and failure modes, irrespective of whether the current transition models are sufficient to solve the tasks.
