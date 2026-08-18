# V5.2 · Experiment 2 — Positive Control (methodology calibration)

> Question: is Countdown's action-conditioned **H(z'|z,op)=0.99 nats / det-frac=0.11** meaningful discrete-state structure, or near-noise? We calibrate by running the *identical* pipeline (VQ K=32, data-dependent init, same usage/leakage/transition/action-conditioned analysis) on environments with **known** answers.


## Setup

- **Positive control:** deterministic FSM (32 states, 3 actions, random transition table), each state rendered as `proto[s] + noise` in 2048-d. Genuine reusable, deterministic `(s,a)→s'`.

- **Noise floor:** same FSM walks, but embeddings are pure noise (no state information).

- **Countdown (observed):** Fixed-Init numbers from Experiments V5.1 / 1, shown for placement.

- **Oracle:** dynamics computed on the *true* FSM states (the theoretical ceiling the codes are trying to recover).


## Calibration table

| Metric | Noise floor | Countdown (observed) | Positive control | Oracle (true states) |
| --- | ---: | ---: | ---: | ---: |
| Active codes | 32 | 32 | 32 | — |
| AMI(codes, true states) | -0.001 | — | 0.935 | — |
| Position leakage | 0.263 | 0.790 | 0.247 | — |
|   · leakage majority | 0.248 | 0.251 | 0.249 | — |
| Bigram (z) | 0.370 | 0.263 | 0.308 | — |
| MLP(z) | 0.370 | 0.266 | 0.307 | — |
| Action bigram (z, a) | 0.362 | 0.553 | 0.770 | — |
| MLP(z, a) | 0.370 | 0.558 | 0.772 | — |
| H(z'|z) nats | 2.394 | 1.977 | 1.341 | 1.044 |
| H(z'|z, a) nats | 2.259 | 0.993 | 0.386 | 0.000 |
| Det-frac (z) | 0.000 | 0.000 | 0.000 | 0.000 |
| **Det-frac (z, a)** | 0.000 | 0.110 | 0.593 | 1.000 |

_AMI = adjusted mutual information between VQ codes and ground-truth states (1.0 = perfect recovery, 0 = independent). Det-frac = share of transition mass from near-deterministic (entropy < 0.5 nats) conditionings seen ≥ 10×._


---
## Did the pipeline recover known states?

**Yes.** On the positive control the VQ codes recover the ground-truth states (AMI = 0.935), position leakage stays at chance (0.247 vs majority 0.249), and conditioning on the action makes the transition essentially deterministic: H(z'|z,a) = 0.386 nats, det-frac = 0.593 (oracle 1.000), MLP(z,a) = 0.772 ≫ majority 0.098. The noise floor shows the opposite (AMI -0.001, det-frac 0.000).


## Is the methodology validated?

**Yes.** The pipeline is *capable* of reporting strong determinism / low entropy / high AMI when reusable states exist (positive control) and ~noise when they do not (floor). It is not biased toward either answer.


## Calibrating Countdown's 0.11 / 0.99

Placed on each floor→ceiling axis (floor = noise, ceiling = positive control), Countdown lands:

- **Det-frac(z,a):** floor 0.000 → ceiling 0.593; Countdown 0.110 — **19%** of the way up.
- **MLP(z,a) acc:** floor 0.370 → ceiling 0.772; Countdown 0.558 — **47%**.
- **H(z'|z,a):** floor 2.259 → ceiling 0.386 nats; Countdown 0.993 — **68%** of the way down to the ceiling.


> **Calibrated conclusion.** Countdown's action-conditioned structure is **real and clearly above the noise floor** — every action-conditioned metric beats the floor (MLP 0.558 vs 0.370; H(z'|z,a) 0.993 vs 2.259 nats) — yet it falls **short of the genuine reusable-state ceiling** (MLP 0.772, H 0.386), and the axes disagree on how far: the strict det-frac bar puts Countdown only 19% of the way up, while soft entropy/accuracy put it ~47–68%. The honest reading: Countdown has **graded, partial** action-conditioned dependence — meaningfully more than noise, but lacking the crisp determinism of reusable planning states. The low det-frac (0.11) reflects how few conditionings clear the strict <0.5-nat / ≥10-count bar, not that the dependence is noise — superseding the earlier ‘≈ noise’ reading, which was an artifact of a split-geometry bug in the control harness.


_Caveats: the control is a clean, well-separated FSM — an *upper* bound on recoverability; a harder control (entangled position, heavier noise) would lower the ceiling. Countdown’s position-leakage (0.79) already shows its codes are far more position-bound than the control’s (~chance). Scope otherwise unchanged.


_Artifacts: `reports/positive_control.json`._
