# Oracle Audit

## Status: NO ORACLE EXISTS — but the report invokes one rhetorically

The spec Part "DO NOT IMPLEMENT" explicitly forbids Oracle transitions in Phase A, and indeed no Oracle code exists. However, `intrinsic_noise_report.md` makes an **Oracle argument without an Oracle experiment**:

> "Oracle dynamics cannot save a representation that fails to consistently encode the task state."

This is a claim about a hypothetical perfect transition model (the "Oracle") being unable to rescue the pipeline. It is asserted, not measured.

## Classification of the (implicit) Oracle claim

- **Informative?** No. No Oracle transition was run; no comparison of model-vs-Oracle coherence exists.
- **Weakly informative?** Marginally — the underlying premise (if the representation can't separate states, perfect dynamics on top can't help) is a *reasonable* logical argument. But it is leveraged off the contested within/between metrics (see intrinsic_noise_audit), which are themselves confounded by anisotropy and position.
- **Tautological?** **Partly.** "A perfect dynamics model cannot fix a non-Markovian representation" is close to true by definition (if state isn't recoverable, no function of it recovers state). Using it as a *finding* adds no empirical evidence; it restates the representation result.

## The hostile reading

The Oracle sentence functions as a **rhetorical amplifier**: it converts a noisy, underpowered, confounded representation metric into a sweeping claim ("Oracle cannot save it") that sounds like a stronger, dynamics-level result. No Oracle was tested. The claim should be deleted or backed by an actual Oracle-transition coherence experiment (which the project forbids in this phase and has not run).

## Verdict

The Oracle contributes **no empirical evidence**. Its single appearance is a tautological/ rhetorical flourish attached to a contested metric. Classify: **Tautological / non-informative.** Any future Oracle-transition experiment must be run and reported with the identity and blind baselines before such a claim is permissible.
