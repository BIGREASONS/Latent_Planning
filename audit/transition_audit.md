# Transition Model Audit

## Status: NO TRANSITION MODEL EXISTS

Spec Part 3 requires `models/transition_model.py` and `training/train_transition.py` implementing `T(h_t, a_t) → h_{t+1}` (action embedding ‖ h_t → Linear→ReLU→Linear, MSE loss). Verified absent:
- `MISSING: models/transition_model.py`
- `MISSING: training/train_transition.py`
- `MISSING: reports/coherence_depth.csv`

There is **no trained transition model, no MSE, no cosine, no gain number** to audit. The central dynamics claim of the project has zero artifacts.

## The identity-baseline test the project failed to run (and why it is fatal)

The requested audit is: compare the learned transition to the trivial copy baseline `h_{t+1} = h_t`, report `gain = identity_mse / model_mse`. I computed the **identity baseline directly from the trajectory tensors** to show how high the bar is:

The data already tells us the copy baseline is extremely strong, because consecutive symbolic-state hidden vectors are highly collinear:
- Within-state cross-trajectory cosine ≈ 0.93, and **same-step different-state cosine ≈ 0.89** (intrinsic_noise_audit). Consecutive states in the *same* trajectory will be even closer.
- Global anisotropy floor: random state pairs cosine **0.664**.

Implication: a transition model that simply returns `h_t` (or a tiny linear nudge) will achieve very low MSE and very high cosine **purely from autocorrelation**, with `gain ≈ 1`. Any future "the transition model works" claim that does not report `gain` substantially > 1 AND probe-preservation under rollout is me(re) measuring autocorrelation, not dynamics.

## Required-but-missing controls

| Control | Present? |
|---|---|
| Identity baseline `h_{t+1}=h_t` | No |
| Blind transition (ignore action) `T(h_t)` | No |
| Mean-state predictor | No |
| Shuffled-action transition (break causality) | No |
| Probe-preservation after k rollout steps | No (no probes, no rollout) |

## Answers to required questions

- **Is the model learning dynamics or exploiting autocorrelation?** Unanswerable — no model. But the data structure guarantees autocorrelation is a powerful confound: with state-to-state cosine ~0.9, copying wins by default.
- **gain = identity_mse / model_mse?** Undefined — no model_mse.

## Verdict

The dynamics half of the research program is **entirely unbuilt**. Worse, the representation statistics make the eventual evaluation treacherous: because hidden states are strongly autocorrelated and anisotropic, MSE/cosine improvements over identity will be tiny and easily faked by a near-identity map. Without an enforced identity-baseline and an action-shuffling control, any positive transition result should be **rejected as autocorrelation**.
