# Oracle Gap Definition Audit

**Definition in Manuscript:**
Oracle Gap = $P^*(h_{t+1}) - P(\hat{h}_{t+1})$
(Teacher Oracle Probe Accuracy - Predicted Latent Probe Accuracy)

## Why did earlier audits show a negative Oracle Gap?
The previous audit (which found a negative Oracle Gap for the Transformer) accidentally parsed the `train=20` debug directories. In those directories, the linear probe was trained and tested on merely 20 trajectories.

Because the probe was so weak and the test set so small, the linear decision boundary was noisy. The predicted latent state ($\hat{h}$) occasionally fell on the correct side of this noisy boundary even when the true teacher state ($h$) fell on the wrong side. This resulted in Predicted Accuracy > Oracle Accuracy for that tiny batch, yielding a negative Oracle Gap.

## Does the mathematical definition hold?
Yes. The Oracle Gap is an empirical measure of *relative* semantic decodability. The Oracle (frozen teacher state) establishes the ceiling of what the probe *could* recover if the transition were perfect. 

When evaluated on the canonical Phase 3 dataset (5000 trajectories, test set = 970), the probe reaches ~93% accuracy on the Oracle states. The predicted states never exceed this ceiling, and the Oracle Gap strictly remains a large positive value (approx 0.17 - 0.19) for all 9 architectural conditions.

The metric definition is mathematically sound, and the manuscript's reported values are derived from the correct, large-sample test set.

