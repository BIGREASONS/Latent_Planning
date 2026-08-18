# Phase 6 — Metric Inflation Audit

**Raw artifacts inspected:** `reports/probe_report.md`, `reports/probe_results.csv`, `reports/coherence_action_depth.csv`. Conclusions traced to those files, not to the prose reports.

## M0 — The shipped results are a smoke run, reported as if scientific — CRITICAL

`reports/probe_report.md` (raw):
> Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

`reports/coherence_action_depth.csv` (raw): `n_samples` per depth = **20, 20, 17, …**; `cosine_similarity` at depth 1 = **0.435**.

These are tiny-N values, yet the generated reports emit categorical verdicts — Probe report: "remaining_numbers … **encoded**", "distance_to_solution … **encoded**", and `phase_a_report.md` computes an "**effective horizon**." With 52 train / 63 test states and **zero** confidence intervals, significance tests, or seed repeats anywhere in the codebase, none of these verdicts are statistically defensible. **Any conclusion currently in `reports/` is UNVERIFIED at best and noise at worst.** A real run is required before any number is quoted.

## M1 — Probe "encoded" rule fires on noise and ignores its own F1 — HIGH

`evaluation/probes.py:281-284`:
```python
is_encoded = (not np.isnan(auc) and auc >= 0.65) or (accuracy - chance >= 0.10)
```
Evidence it misfires — from `probe_report.md`:
| Probe | Acc | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|
| A:remaining_numbers | 0.607 | 0.456 | **0.000** | n/a | **encoded** |

Probe A is declared "encoded" purely because `0.607 − 0.456 ≥ 0.10`, while its **F1 is 0.000** — i.e. the per-label classifiers are predicting the majority class and the "accuracy" is just base-rate. The rule never looks at F1, so a degenerate majority-class predictor on an imbalanced multilabel target is labelled "encoded." With n=63 the 0.15 accuracy margin is within sampling noise. The rule is an inflation generator.

## M2 — Probe A averaging hides degeneracy — HIGH

`_fit_eval_multilabel_joint` (`probes.py:169`) reports **mean per-label accuracy** as the headline "accuracy." Each of the 4 large-number labels is dominated by class 0 ("not in pool") or 2 ("used"), so per-label accuracy is high at base rate. The honest metric — **exact match** — is reported as **0.190** in `probe_report.md` but is *not* what the "encoded" verdict keys on. The high number is advertised; the low honest number is a side column.

## M3 — Probe B & D are position detectors, not reasoning detectors — HIGH

- Probe B label = `N − i` (`probes.py:57`); Probe D = `1 if N−i ≤ 2` (`:66`). Both are deterministic functions of **token position within the trajectory**.
- The frozen LM encodes absolute/relative position strongly (rotary embeddings). So "distance-to-solution is linearly decodable" largely restates "position is decodable."
- The repo **itself suspects this** — Phase C.1/C.2/C.3 are entire experiments built to test the depth/position confound (`run_phase_c2_depth_removal.py` adversarially strips depth). Yet Phase A's master report still counts B/D as evidence that "hidden states contain reasoning information" (`run_phase_a.py:191-202`). Inflation by confound.

## M4 — The learned transition barely beats the do-nothing baseline — HIGH (deflation of the core claim)

`coherence_action_depth.csv`, `dynamics_gain = identity_mse / mse`:
- depth 1: 1.147, depth 2: 1.112, **depth 3: 0.960**.

`dynamics_gain < 1` at depth 3 means the trained transition model is **worse than the identity map** (copy `h_0` unchanged) by depth 3. The "latent transition model advances the state" claim is not supported even on the smoke data; the residual/`predict_delta=True` prior (`transition_model.py:92`) means the model is mostly learning to do nothing, and still loses to literally doing nothing past 2 steps. The headline "latent planning" framing is contradicted by the repo's own CSV.

## M5 — Oracle "sanity check" is a tautology dressed as validation — MEDIUM

`oracle_coherence.py:77-83` sets `h = states[d]` and `teacher = states[d]` (the **same tensor**), then reports cosine≈1.0 / MSE≈0 and a `[PASS]` (`run_phase_a.py:392`, `run_phase_b.py:68`). This proves only that a tensor equals itself. It is presented in `phase_b_oracle_report.md` §0 as a passing sanity check, lending false credibility. The Oracle's *informative* output is its probe accuracy — but those probes were fit on train states (`run_phase_b.py:332`) and evaluated on teacher states that overlap by symbolic sub-state (leakage_audit L3), so the "representation ceiling" is itself optimistic.

## M6 — Coherence probe accuracy uses single-sample sklearn predictions — MEDIUM

`coherence.py:92-132` calls `probe.predict(h.cpu().numpy())[0]` per trajectory/depth. Correct in principle, but: probes were trained on standardized **teacher** states; at rollout depth the input is a *predicted* state with different norm/scale. StandardScaler inside the probe applies **train** statistics, so out-of-distribution rolled-out states get mapped arbitrarily — accuracy here is not calibrated and shouldn't be compared 1:1 with teacher accuracy without noting the distribution shift. Reported side-by-side without that caveat.

## M7 — No negative controls anywhere — MEDIUM

There is **no** shuffled-label probe, no random-Gaussian-feature baseline, no "probe the raw input-token embedding" baseline. Chance is only majority-class (`_chance_accuracy`, `:191`). Without a permutation control, "above chance" on n=63 is uninterpretable.

## Verdict

Inflation is systemic, not incidental: a verdict rule that fires on majority-class noise and ignores F1 (M1/M2), position-confounded probes counted as reasoning (M3), a self-referential Oracle "pass" (M5), and — most damning — the project's **own CSV shows the transition model losing to the identity baseline by depth 3** (M4), all computed on a **52/63-sample smoke run** (M0). The reported metrics overstate the evidence in every direction that flatters the latent-planning hypothesis.
