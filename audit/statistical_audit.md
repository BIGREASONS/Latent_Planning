# Statistical Audit

For every reported number: mean, std, SE, bootstrap CI, and whether conclusions survive uncertainty.

## Numbers actually reported by the project

Only `intrinsic_noise_report.md` and `state_statistics.md` report quantities. They report **means ± std** but **no standard errors, no confidence intervals, and no significance tests.** No p-values, no bootstrap, anywhere.

## Reproduced statistics with uncertainty (train.pt)

| Quantity | Mean | Std | n (pairs) | SE | Notes |
|---|---|---|---|---|---|
| Within-state cosine | 0.929 | ~0.066 | ~200 | ~0.005 nominal | pairs non-independent (clusters of 2–3) → effective n ≈ 181, SE understated |
| Between-state cosine | ~0.66 | ~0.24 | 2000 | ~0.005 | ≈ anisotropy floor |
| Global random cosine (floor) | 0.664 | 0.242 | 2000 | 0.005 | the real baseline |
| Same-step diff-state cosine | 0.889 | — | 2499 | small | position confound |
| **Within − between gap** | **0.273** | — | — | — | **95% bootstrap CI [0.258, 0.288]** |

The within−between gap CI [0.258, 0.288] **excludes 0**, so the *raw* contrast is statistically "significant." **But significance ≠ validity**: the relevant null is not "0," it is **the same-step/anisotropy confound**. Against the correct baseline (same-step cosine 0.889), the within-state advantage is only **+0.04**, which on ~200 non-independent pairs is fragile and was never tested.

## Power analysis

From their own `state_statistics.md`: 181 cross-history states, 198 positive pairs, max cluster size **3**. This is far below what is needed for stable contrastive/retrieval estimates. Their file literally warns it is underpowered. A retrieval accuracy of 24.6% computed on ~7–380 eligible queries (depending on key) has a binomial SE of order ±5–15 percentage points — **the Top-1 (24.6%) and Top-5 (39.3%) numbers are not distinguishable from each other or from a confounded baseline at this n.**

## Multiplicity / researcher-degrees-of-freedom

- Symbolic-state key definition (target only vs target+numbers vs +step) swings eligible-pair count from ~7 to ~380 and was chosen in missing code → uncontrolled experimenter degree of freedom that directly drives the retrieval number.
- Layer choice for extraction is unrecorded.
- No seed anywhere → every rerun gives different splits and different numbers.

## Answers to required questions

- **Would conclusions survive uncertainty?** The *qualitative* conclusion (noisy representation) is robust to its own (raw) statistics but **not robust to confound-correction** (the within/between contrast mostly evaporates against same-step baseline). The *quantitative* retrieval claims do **not** survive — underpowered and key-dependent.
- **Are conclusions stable?** No — no seeds, unrecorded layer, key-dependent metrics.

## Verdict

Statistically **substandard**: means ± std only, no SE/CI/tests in the deliverables, non-independent pairs treated as independent, severe underpowering acknowledged but not remediated, and the one contrast that is "significant vs 0" is **not significant vs the correct confounded baseline.** Conclusions are **unstable** under proper uncertainty and confound accounting.
