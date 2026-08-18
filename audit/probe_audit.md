# Probe Audit

## Status: NO PROBES EXIST

The spec (`prompts/phase_a_implementation.md`, Part 6) requires four linear probes:
- Probe A: predict remaining numbers
- Probe B: predict distance-to-solution
- Probe C: predict next symbolic operation
- Probe D: predict reachability within 2 steps

and `evaluation/probes.py` + `reports/probe_results.csv` + `probe_report.md`.

**None of these files exist** (verified: `MISSING: evaluation/probes.py`, `MISSING: reports/probe_results.csv`, `MISSING: reports/phase_a_report.md`). There is no probe code, no probe metrics, no `accuracy/f1/auc` table anywhere in the repo or the extracted results.

Therefore there is **nothing to validate** and, more importantly, **no probe evidence supports any latent-planning claim.**

## Baselines that would be mandatory (and are absent)

Because no probe results exist, I record the baselines any future probe MUST beat, so that the absence is not silently "fixed" with an inflated number later:

| Probe | Majority-class / trivial baseline that must be reported |
|---|---|
| C: next operation (3–4 classes, `+ - *`, no `/`) | Op frequencies are near-uniform (measured: + 2383, − 2376, * 2304 over 7063 steps) → majority baseline ≈ **33.7%**. A probe must exceed this with a CI that excludes it. |
| D: reachable-in-2-steps (binary) | Base rate unknown; balanced-accuracy and the positive rate MUST be reported, else accuracy is meaningless. |
| B: distance-to-solution (ordinal) | Trajectory length is 3–6 ops; predicting the mean depth gives a strong MAE baseline that must be reported. |
| A: remaining numbers (multi-label) | Exact-match vs per-element F1 diverge wildly; exact-match baseline of "predict the most common remaining set" must be reported. |

## Confounds that would invalidate any future probe (pre-registered objections)

1. **Position leakage.** Step index is recoverable from the hidden state with high accuracy (same-step cosine 0.889; NN same-step 85.7% — see intrinsic_noise_audit). "Distance-to-solution" (Probe B) is **trivially** decodable from position, so a high B score is not evidence of planning — it is evidence of a token counter.
2. **Full solution in prompt.** Hidden states are extracted with the entire `cot` in context (see extraction_audit). "Remaining numbers" / "next op" may be readable because the answer tokens are present, not because the model planned.
3. **No held-out by problem.** With 24 leaked (target,numbers) problems train∩test, a probe trained/tested across the standard split can memorize problem-specific patterns.

## Verdict

**Probe success is currently 0% supported — because there are no probes.** Any later probe numbers must be treated as guilty until they (a) report majority/random/exact-match baselines with CIs, (b) control for step-position decodability, (c) use problem-disjoint splits, and (d) extract hidden states from prefixes that do **not** contain the step's own result. None of these controls are present in the codebase today.
