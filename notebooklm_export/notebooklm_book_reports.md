# Reports and Configs


## File: `configs\kaggle.yaml`
```yaml
# Kaggle Experiment Configuration
experiment_name: "latent_planning_phase_a"

# Dataset settings
train_data: "data/train.jsonl"
val_data: "data/val.jsonl"
test_data: "data/test.jsonl"

# Caching
hidden_states_dir: "data/hidden_states"

# Transition Model Training
transition_training:
  mlp_hidden_dim: 1024
  lr: 0.001
  epochs: 50
  batch_size: 64
  weight_decay: 0.0001
  mixed_precision: "bf16"
  use_action: true
  save_dir: "checkpoints"

# Artifacts and Logging
artifact_dir: "artifacts"
runs_dir: "runs"

# Resume Support
resume_from_checkpoint: false
checkpoint_path: ""

```


## File: `configs\v5.yaml`
```yaml
# Version 5 — Discrete State Discovery configuration.
#
# Components C1-C6 and C9 consume the cached trajectory files produced by the
# (already-built) Phase A pipeline:
#   <reports_dir>/trajectories/{train,val,test}.pt
# C7 (permutation robustness) and C8 (cross-domain transfer) need their own
# data extraction — see V5_READINESS.md for the full Kaggle execution plan.
#
# Everything is repo-relative so the same config runs locally and on Kaggle
# (point --reports_dir / --out_dir at /kaggle/working/reports on Kaggle).

experiment_name: "v5_discrete_states"

paths:
  reports_dir: "reports"          # holds trajectories/ and receives v5_* outputs
  checkpoints_dir: "checkpoints"  # vq_state.pt etc.
  discrete_dir: "reports/discrete_trajectories"  # encoded code trajectories

# C1 — VQ State Quantizer
vq:
  num_codes: 256              # codebook cardinality (state-space size)
  commitment_cost: 0.25       # weight on ||sg(z_q) - h||^2
  ema_decay: 0.99             # EMA codebook update decay
  lr: 0.0003                  # (no-op for pure-EMA VQ; reserved for encoder nets)
  epochs: 50
  batch_size: 256

# C3 — Discrete transition predictability (action-blind MLP)
discrete_transition:
  embed_dim: 32
  hidden_dim: 128
  lr: 0.001
  epochs: 30
  batch_size: 256

# C9 — Discrete rollout coherence
rollout:
  max_depth: 8                # 0 => dynamic (95th percentile of trajectory lengths)

# C7 — Permutation robustness (multi-solution Countdown)
multi_solution:
  num_problems: 2000
  solutions_per_problem: 4

# C8 — Cross-domain transfer
transfer:
  num_algebra: 2000
  num_logic: 2000
  num_graph: 2000

seed: 0

```


## File: `reports\action_conditioned.json`
```json
{
  "config": {
    "num_codes": 32,
    "epochs": 30,
    "seed": 0,
    "eval_split": "val"
  },
  "fixed_init": {
    "A_majority": {
      "top1": 0.11612475116124751,
      "cond": "\u2014"
    },
    "B_bigram": {
      "top1": 0.26343729263437293,
      "cond": "z_t"
    },
    "C_action_bigram": {
      "top1": 0.5534173855341739,
      "cond": "z_t, op"
    },
    "D_mlp_z": {
      "top1": 0.2660915726609157,
      "pred_entropy": 2.0493449190237225,
      "cond": "z_t"
    },
    "E_mlp_z_op": {
      "top1": 0.5580623755806238,
      "pred_entropy": 1.107272284190704,
      "cond": "z_t, op"
    },
    "E2_mlp_z_op_operands": {
      "top1": 0.6224286662242866,
      "pred_entropy": 0.9749013575482385,
      "cond": "z_t, op, operands"
    },
    "F_mlp_action_only": {
      "top1": 0.4054412740544127,
      "pred_entropy": 1.470632491083278,
      "cond": "op, operands"
    },
    "_struct_state": {
      "global_entropy": 1.9768534756904257,
      "det_frac_mass": 0.0,
      "n_conditions": 32,
      "reliable_mass_frac": 0.9943389943389943
    },
    "_struct_action": {
      "global_entropy": 0.9932319690453785,
      "det_frac_mass": 0.10989010989010989,
      "n_conditions": 91,
      "reliable_mass_frac": 0.9773559773559773
    },
    "_meta": {
      "name": "fixed_init",
      "n_train": 3003,
      "n_eval": 1507
    }
  },
  "scrubbed": {
    "A_majority": {
      "top1": 0.18911745189117452,
      "cond": "\u2014"
    },
    "B_bigram": {
      "top1": 0.21765096217650962,
      "cond": "z_t"
    },
    "C_action_bigram": {
      "top1": 0.43397478433974784,
      "cond": "z_t, op"
    },
    "D_mlp_z": {
      "top1": 0.21566025215660253,
      "pred_entropy": 2.586186743763479,
      "cond": "z_t"
    },
    "E_mlp_z_op": {
      "top1": 0.43928334439283345,
      "pred_entropy": 1.5952312264445607,
      "cond": "z_t, op"
    },
    "E2_mlp_z_op_operands": {
      "top1": 0.5613802256138023,
      "pred_entropy": 1.3291857440343502,
      "cond": "z_t, op, operands"
    },
    "F_mlp_action_only": {
      "top1": 0.53085600530856,
      "pred_entropy": 1.4440424057847234,
      "cond": "op, operands"
    },
    "_struct_state": {
      "global_entropy": 2.494212601731941,
      "det_frac_mass": 0.0,
      "n_conditions": 32,
      "reliable_mass_frac": 0.9973359973359973
    },
    "_struct_action": {
      "global_entropy": 1.456162008304367,
      "det_frac_mass": 0.022977022977022976,
      "n_conditions": 92,
      "reliable_mass_frac": 0.9663669663669664
    },
    "_meta": {
      "name": "scrubbed",
      "n_train": 3003,
      "n_eval": 1507
    }
  },
  "contrasts_fixed_init": {
    "action_helps_lookup_CminusB": 0.2899800928998009,
    "action_helps_mlp_EminusD": 0.29197080291970806,
    "mlp_vs_action_bigram_EminusC": 0.004644990046449915,
    "operands_help_E2minusE": 0.06436629064366284,
    "state_matters_E2minusF": 0.2169873921698739,
    "determinism_gain": 0.10989010989010989,
    "entropy_drop": 0.9836215066450472
  }
}
```


## File: `reports\action_conditioned.md`
```md
# V5.2 · Experiment 1 — Action-Conditioned Transition Test

> Reviewer criticism addressed: a planning state is defined by `(state, action) → next_state`, not `state → next_state`. V5.1 tested only the action-blind form. Here we condition on the symbolic Countdown action.


## Implementation

- **Representation:** Fixed-Init VQ codes (K=32, data-dependent init, raw last-layer states) — the clean, non-collapsed V5.1 codebook. Scrubbed codes reported as a robustness check.

- **Action:** op id (0=ADD, 1=SUB, 2=MUL; DIV absent from data) and the raw operands `[arg1, arg2]` (signed-log + standardized for the MLP).

- **Eval:** top-1 on the held-out **val** split (train 3003 transitions, val 1507); MLPs trained 30 epochs, seed 0. Entropy/determinism measured on train (frequency-weighted, deterministic = successor entropy < 0.5 nats over conditionings seen ≥ 10×).

- **Models A–F:** majority; state bigram; action bigram `(z,op)→mode`; MLP(z); MLP(z,op); MLP(z,op,operands); and an **action-only control** MLP(op,operands) that ignores `z_t`.


## Results — Fixed-Init codebook (primary)

### Predictive accuracy (held-out val)

| Model | Conditioning | Top1 | Pred. entropy (nats) |
| --- | --- | ---: | ---: |
| A. Majority | — | 0.116 | — |
| B. Bigram | z_t | 0.263 | — |
| C. Action bigram | z_t, op | 0.553 | — |
| D. MLP(z_t) | z_t | 0.266 | 2.049 |
| E. MLP(z_t, op) | z_t, op | 0.558 | 1.107 |
| E2. MLP(z_t, op, operands) | z_t, op, operands | 0.622 | 0.975 |
| F. MLP(op, operands) — control | op, operands | 0.405 | 1.471 |

### Transition structure (train)

| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |
| --- | ---: | ---: | ---: |
| z_t | 1.977 | 0.000 | 32 |
| z_t, op | 0.993 | 0.110 | 91 |

### Key contrasts

- **Does the action help the lookup?** C − B = +0.290 (0.553 vs 0.263).
- **Does the action help the MLP?** E − D = +0.292 (0.558 vs 0.266).
- **Does the MLP beat the action bigram?** E − C = +0.005.
- **Do operands add information?** E2 − E = +0.064 (E2 = 0.622).
- **Does the state matter, or only the action?** E2 − F = +0.217 (F action-only = 0.405).
- **Determinism gain from conditioning on the action:** det-frac 0.000 (z) → 0.110 (z,op); entropy 1.977 → 0.993 nats.


## Robustness — Scrubbed codebook

| Model | Conditioning | Top1 | Pred. entropy (nats) |
| --- | --- | ---: | ---: |
| A. Majority | — | 0.189 | — |
| B. Bigram | z_t | 0.218 | — |
| C. Action bigram | z_t, op | 0.434 | — |
| D. MLP(z_t) | z_t | 0.216 | 2.586 |
| E. MLP(z_t, op) | z_t, op | 0.439 | 1.595 |
| E2. MLP(z_t, op, operands) | z_t, op, operands | 0.561 | 1.329 |
| F. MLP(op, operands) — control | op, operands | 0.531 | 1.444 |

| Conditioning | H(z'|·) nats | Det. fraction (mass, H<0.5) | # conditionings |
| --- | ---: | ---: | ---: |
| z_t | 2.494 | 0.000 | 32 |
| z_t, op | 1.456 | 0.023 | 92 |


---
## Reviewer-style interpretation

**1. Results.** See tables above. On the clean Fixed-Init codebook, conditioning on the symbolic action moves top-1 from 0.263 (state bigram) to 0.553 (action bigram) and 0.558 (action MLP); adding operands gives 0.622.

**2. Does MLP(z,a) materially beat the action-aware bigram?** **No** — E − C = +0.005 (threshold ±0.03). The MLP does not exceed the action-conditioned lookup: whatever the action contributes is already a first-order count effect, not learned structure.

**3. Does the planning-state hypothesis survive?**

> **Partially — and this revises the V5.1 wording.** The reviewer criticism was correct that the action-blind test understated the structure: conditioning on the action roughly *doubles* predictability (state→action bigram 0.263 → 0.553; E−D +0.292), *halves* the conditional entropy (1.977 → 0.993 nats), and turns a 0% deterministic mass into 0.110. So `(state, action)` carries real, non-trivial dynamics that `state` alone did not.

> **But it does not reach reusable planning states.** The structure is purely first-order — the MLP does not beat the action lookup (E−C +0.005) — and it is far from deterministic: only 0.110 of transition mass is near-deterministic, H(z'|z,op) ≈ 0.993 nats (~3 effective successors), and top-1 is 0.622 even with operands. The state is not vacuous (E2−F +0.217), but it composes with actions into *weak, stochastic* transitions, not crisp reusable ones.

> **Net:** V5.1's flat "no reusable structure" is too strong and should be revised to **"weak, first-order, sub-deterministic action-conditioned dynamics — not reusable planning states."** Whether even this weak effect exceeds what *any* action-conditioned model would yield on position-correlated codes cannot be judged without a calibrated positive control — which is exactly the purpose of Experiment 2.


_Caveats: single model/task/layer/seed; op-type action (DIV absent); operands make the next *number* computable, so any E2 gain may reflect numeric leakage rather than reusable state composition. Scope unchanged from V5.1._


_Artifacts: `reports/action_conditioned.json`._

```


## File: `reports\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,n_samples,dynamics_gain
1,4.760268330574036,0.35856705605983735,0.4,0.15,0.7153846153846154,0.7576923076923077,0.35,0.45,,,5.446681714057922,0.28723630383610727,0.5,0.7115384615384616,0.35,,20,1.144196363695555
2,5.167408108711243,0.34885752350091936,0.4117647058823529,0.5882352941176471,0.7423076923076923,0.8230769230769232,0.25,0.65,,,5.61424069404602,0.24624466970562936,0.4117647058823529,0.7384615384615385,0.25,,20,1.0864713171350846
3,6.089606761932373,0.35194505663479075,0.16666666666666666,0.6666666666666666,0.751131221719457,0.8325791855203619,0.058823529411764705,0.058823529411764705,,,5.673072871039896,0.24148888798320994,0.3333333333333333,0.7149321266968326,0.0,,17,0.9315992136805396
4,7.515286286671956,0.3487296899159749,,,0.7692307692307693,0.8461538461538461,0.0,0.0,,,5.735038121541341,0.23732860138018927,,0.8076923076923078,0.0,,6,0.7631163874238815
5,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,0,

```


## File: `reports\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,n_samples,dynamics_gain
1,4.8010594844818115,0.3535108707845211,0.4,0.15,0.7346153846153847,0.7576923076923077,0.2,0.45,,,5.446681714057922,0.28723630383610727,0.5,0.7115384615384616,0.35,,20,1.1344749490529993
2,5.116099548339844,0.3472119942307472,0.47058823529411764,0.5882352941176471,0.7807692307692309,0.8230769230769232,0.05,0.65,,,5.61424069404602,0.24624466970562936,0.4117647058823529,0.7384615384615385,0.25,,20,1.097367367659572
3,5.980908534106086,0.34831679217955647,0.16666666666666666,0.6666666666666666,0.8009049773755655,0.8325791855203619,0.0,0.058823529411764705,,,5.673072871039896,0.24148888798320994,0.3333333333333333,0.7149321266968326,0.0,,17,0.9485302841013602
4,7.3302788734436035,0.34645629425843555,,,0.8076923076923078,0.8461538461538461,0.0,0.0,,,5.735038121541341,0.23732860138018927,,0.8076923076923078,0.0,,6,0.7823765262626559
5,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,0,

```


## File: `reports\coherence_depth.csv`
```csv
depth,mse,cosine_similarity,token_accuracy,teacher_token_accuracy,n_samples
1,0.8075717452913522,0.4036983195692301,0.0,0.0,40
2,0.8277104247361422,0.41020950344391166,0.03571428571428571,0.07142857142857142,40
3,7.901257994983878,0.26588797562622596,0.0,0.047619047619047616,28
4,707.290652692318,0.2244227289089135,0.1,0.1,21
5,1.3026339814066887,0.4739085525274277,,,10
6,,,,,0
7,,,,,0
8,,,,,0

```


## File: `reports\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,n_samples,dynamics_gain
1,4.890261840820313,0.34168516397476195,0.35,0.45,0.7269230769230769,0.7576923076923077,0.05,0.0,,,5.568423962593078,0.2720899984240532,0.3,0.7076923076923076,0.05,,20,1.1386760349132978
2,5.448688006401062,0.31247783452272415,0.45,0.45,0.7384615384615385,0.7923076923076924,0.05,0.05,,,5.839201188087463,0.2155156686902046,0.4,0.7269230769230769,0.05,,20,1.0716710483748804
3,6.1951096534729,0.3399553939700127,0.4,0.3,0.7346153846153847,0.8346153846153846,0.9,0.15,,,5.700663423538208,0.23710524737834932,0.3,0.7307692307692307,0.9,,20,0.9201876548452194
4,7.859960460662842,0.3125404119491577,0.4,0.55,0.7384615384615385,0.8615384615384615,0.6,0.9,,,5.803190970420838,0.2201945848762989,0.35,0.75,0.0,,20,0.7383231759834382
5,9.713274383544922,0.3213836759328842,,,0.7346153846153846,0.8461538461538461,0.0,0.0,,,5.7545857429504395,0.23032009825110436,,0.7692307692307693,0.0,,20,0.5924455045457355
6,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,0,

```


## File: `reports\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.7576923076923077,0.7576923076923077,0.45,0.45,,,5.446681618690491,0.28723626732826235,0.5,0.7115384615384616,0.35,,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.8230769230769232,0.8230769230769232,0.65,0.65,,,5.614240741729736,0.24624463841319083,0.4117647058823529,0.7384615384615385,0.25,,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.8325791855203619,0.8325791855203619,0.058823529411764705,0.058823529411764705,,,5.673072814941406,0.24148884766242085,0.3333333333333333,0.7149321266968326,0.0,,17
4,0.0,0.9999998609224955,,,0.8461538461538461,0.8461538461538461,0.0,0.0,,,5.7350380420684814,0.23732855916023254,,0.8076923076923078,0.0,,6
5,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,0

```


## File: `reports\collision_audit.csv`
```csv
target,numbers,num_available,occurrences,unique_trajectories,unique_histories,swap_candidates_A,swap_candidates_B,splits
5003,3|4|25|50|75|100,6,1,1,1,0,0,train
5003,3|4|25|75|5000,5,1,1,1,0,0,train
5003,4|25|75|5003,4,1,1,1,0,0,train
13,1|3|6|7|9|100,6,1,1,1,0,0,train
13,1|3|6|16|100,5,1,1,1,0,0,train
13,1|6|13|100,4,1,1,1,0,0,train
84,1|5|6|10|75|100,6,1,1,1,0,0,train
84,1|5|10|75|94,5,1,1,1,0,0,train
84,5|10|75|94,4,1,1,1,0,0,train
84,5|75|84,3,1,1,1,0,0,train
74,1|2|6|25|50|100,6,1,1,1,0,0,train
74,2|6|26|50|100,5,1,1,1,0,0,train
74,2|6|50|74,4,1,1,1,0,0,train
3,1|4|5|5|9|50,6,1,1,1,0,0,train
3,1|1|5|9|50,5,1,1,1,0,0,train
3,1|6|9|50,4,1,1,1,0,0,train
3,1|3|50,3,1,1,1,0,0,train
7506,5|6|25|50|75|100,6,1,1,1,0,0,train
7506,5|6|25|50|7500,5,1,1,1,0,0,train
7506,5|25|50|7506,4,1,1,1,0,0,train
52,1|3|4|5|6|50,6,1,1,1,0,0,train
52,3|4|5|6|49,5,1,1,1,0,0,train
52,4|5|6|52,4,1,1,1,0,0,train
27,8|10|25|50|75|100,6,1,1,1,0,0,train
27,10|17|50|75|100,5,1,1,1,0,0,train
27,27|50|75|100,4,1,1,1,0,0,train
250,1|4|6|10|50|100,6,1,1,1,0,0,train
250,5|6|10|50|100,5,1,1,1,0,0,train
250,6|10|100|250,4,1,1,1,0,0,train
42,1|3|4|5|7|50,6,1,1,1,0,0,train
42,1|5|7|7|50,5,1,1,1,0,0,train
42,5|6|7|50,4,1,1,1,0,0,train
42,5|42|50,3,1,1,1,0,0,train
12500,1|2|4|10|25|50,6,1,1,1,0,0,train
12500,1|2|4|10|1250,5,1,1,1,0,0,train
12500,1|2|4|12500,4,1,1,1,0,0,train
1249,1|10|25|50|75|100,6,1,1,1,0,0,train
1249,1|10|75|100|1250,5,1,1,1,0,0,train
1249,10|75|100|1249,4,1,1,1,0,0,train
108,1|2|6|9|10|100,6,1,1,1,0,0,train
108,2|6|9|10|100,5,1,1,1,0,0,train
108,6|9|10|102,4,1,1,1,0,0,train
108,9|10|108,3,1,1,1,0,0,train
81,2|3|25|50|75|100,6,1,1,1,0,0,train
81,6|25|50|75|100,5,1,1,1,0,0,train
81,25|50|81|100,4,1,1,1,0,0,train
630000,6|9|25|50|75|100,6,1,1,1,0,0,train
630000,6|34|50|75|100,5,1,1,1,0,0,train
630000,6|75|84|100,4,1,1,1,0,0,train
630000,6|75|8400,3,1,1,1,0,0,train
630000,6|630000,2,1,1,1,0,0,train
135,1|3|6|25|50|100,6,1,1,1,0,0,train
135,4|6|25|50|100,5,1,1,1,0,0,train
135,6|29|50|100,4,1,1,1,0,0,train
135,6|50|129,3,1,1,1,0,0,train
135,50|135,2,1,1,1,0,0,train
131,6|8|25|50|75|100,6,1,1,1,0,0,train
131,6|8|50|75|125,5,1,1,1,0,0,train
131,8|50|75|131,4,1,1,1,0,0,train
233,3|4|5|6|7|50,6,1,1,1,0,0,train
233,3|5|6|7|46,5,1,1,1,0,0,train
233,3|6|7|230,4,1,1,1,0,0,train
233,6|7|233,3,1,1,1,0,0,train
374971,4|7|25|50|75|100,6,1,1,1,0,0,train
374971,4|7|25|50|7500,5,1,1,1,0,0,train
374971,4|7|25|375000,4,1,1,1,0,0,train
374971,4|7|374975,3,1,1,1,0,0,train
374971,7|374971,2,1,1,1,0,0,train
33,6|7|10|25|75|100,6,1,1,1,0,0,train
33,6|17|25|75|100,5,1,1,1,0,0,train
33,6|25|58|100,4,1,1,1,0,0,train
33,6|33|100,3,1,1,1,0,0,train
9600,4|7|25|50|75|100,6,1,1,1,0,0,test
9600,4|32|50|75|100,5,1,1,1,0,0,test
9600,4|50|100|2400,4,1,1,1,0,0,test
9600,50|100|9600,3,1,1,1,0,0,test
104,1|4|6|7|10|100,6,1,1,1,0,0,test
104,1|3|4|6|100,5,1,1,1,0,0,test
104,1|4|6|103,4,1,1,1,0,0,test
104,4|6|104,3,1,1,1,0,0,test
550,2|7|25|50|75|100,6,1,1,1,0,0,test
550,2|50|75|100|175,5,1,1,1,0,0,test
550,2|50|75|275,4,1,1,1,0,0,test
550,50|75|550,3,1,1,1,0,0,test
296,3|4|4|6|7|75,6,1,1,1,0,0,test
296,3|4|6|7|300,5,1,1,1,0,0,test
296,3|6|7|296,4,1,1,1,0,0,test
1891,3|8|9|9|10|100,6,1,1,1,0,0,test
1891,3|8|9|19|100,5,1,1,1,0,0,test
1891,3|8|9|1900,4,1,1,1,0,0,test
1891,3|8|1891,3,1,1,1,0,0,test
30,1|1|2|3|10|25,6,1,1,1,0,0,test
30,1|2|3|10|25,5,1,1,1,0,0,test
30,1|3|10|27,4,1,1,1,0,0,test
30,1|10|30,3,1,1,1,0,0,test
27,2|6|10|25|50|75,6,1,1,1,0,0,test
27,6|10|23|50|75,5,1,1,1,0,0,test
27,6|10|27|75,4,1,1,1,0,0,test
1650,5|7|9|9|50|75,6,1,1,1,0,0,test
1650,4|7|9|50|75,5,1,1,1,0,0,test
1650,9|28|50|75,4,1,1,1,0,0,test
1650,9|22|75,3,1,1,1,0,0,test
1650,9|1650,2,1,1,1,0,0,test
39993,4|7|8|25|50|100,6,1,1,1,0,0,test
39993,7|8|50|100|100,5,1,1,1,0,0,test
39993,7|50|100|800,4,1,1,1,0,0,test
39993,7|100|40000,3,1,1,1,0,0,test
39993,100|39993,2,1,1,1,0,0,test
187675,2|7|25|50|75|100,6,1,1,1,0,0,test
187675,7|25|50|100|150,5,1,1,1,0,0,test
187675,7|25|100|7500,4,1,1,1,0,0,test
187675,25|100|7507,3,1,1,1,0,0,test
187675,100|187675,2,1,1,1,0,0,test
8,1|4|5|8|75|100,6,1,1,1,0,0,test
8,1|1|8|75|100,5,1,1,1,0,0,test
8,1|8|75|100,4,1,1,1,0,0,test
8,8|75|100,3,1,1,1,0,0,test
290,5|10|10|25|75|100,6,1,1,1,0,0,test
290,10|10|30|75|100,5,1,1,1,0,0,test
290,10|75|100|300,4,1,1,1,0,0,test
290,75|100|290,3,1,1,1,0,0,test
696,4|7|25|50|75|100,6,1,1,1,0,0,test
696,4|25|50|75|700,5,1,1,1,0,0,test
696,25|50|75|696,4,1,1,1,0,0,test
11175,1|6|9|25|50|75,6,1,1,1,0,0,test
11175,6|9|25|50|75,5,1,1,1,0,0,test
11175,6|9|75|1250,4,1,1,1,0,0,test
11175,6|75|11250,3,1,1,1,0,0,test
11175,6|11175,2,1,1,1,0,0,test
900,1|8|25|50|75|100,6,1,1,1,0,0,test
900,1|25|50|75|800,5,1,1,1,0,0,test
900,25|50|75|800,4,1,1,1,0,0,test
900,25|50|875,3,1,1,1,0,0,test
900,50|900,2,1,1,1,0,0,test
165,2|10|25|50|75|100,6,1,1,1,0,0,test
165,2|10|75|75|100,5,1,1,1,0,0,test
165,2|10|75|175,4,1,1,1,0,0,test
165,2|75|165,3,1,1,1,0,0,test
56,1|5|6|7|9|50,6,1,1,1,0,0,test
56,5|7|7|9|50,5,1,1,1,0,0,test
56,5|7|50|63,4,1,1,1,0,0,test
56,5|50|56,3,1,1,1,0,0,test
120,2|5|5|7|8|100,6,1,1,1,0,0,test
120,2|5|5|15|100,5,1,1,1,0,0,test
120,2|5|20|100,4,1,1,1,0,0,test
120,2|5|120,3,1,1,1,0,0,test
100,2|6|25|50|75|100,6,1,1,1,0,0,test
100,2|6|50|75|75,5,1,1,1,0,0,test
100,0|2|6|50,4,1,1,1,0,0,test
100,2|6|50,3,1,1,1,0,0,test
100,6|100,2,1,1,1,0,0,test
93,5|5|8|8|10|25,6,1,1,1,0,0,test
93,5|5|8|10|17,5,1,1,1,0,0,test
93,5|8|10|85,4,1,1,1,0,0,test
93,5|10|93,3,1,1,1,0,0,test

```


## File: `reports\collision_audit_summary.md`
```md
# Collision Statistics Audit — Summary

Reuses `evaluation.intrinsic_noise.get_symbolic_states` (the same symbolic-state extraction used by the Oracle Audit, intrinsic_noise.py and Phase C). Grouping key is the symbolic state `(target, sorted(available_numbers))`; the target is already embedded in that key.

## Inputs

| split | trajectories |
|---|---|
| train | 20 |
| test | 20 |

## Global

| metric | value |
|---|---|
| total_states | 155 |
| unique_symbolic_states | 155 |
| avg_occurrences_per_state | 1.000000 |
| median_occurrences | 1.000000 |
| max_occurrences | 1 |

## Collision distribution

| threshold | states | fraction of unique states |
|---|---|---|
| >= 2 occurrences | 0 | 0.000000 |
| >= 5 occurrences | 0 | 0.000000 |
| >= 10 occurrences | 0 | 0.000000 |
| >= 20 occurrences | 0 | 0.000000 |

## History diversity

Unique action histories reaching each symbolic state.

| threshold | states |
|---|---|
| >= 2 unique histories | 0 |
| >= 5 unique histories | 0 |
| >= 10 unique histories | 0 |

## Oracle Coverage Estimate

Valid Oracle 2A swap candidates. **A** = same target + same symbolic state + different trajectory. **B** = A *and* different action history (this is the exact Oracle 2A partner rule). Candidate counts are ordered anchor->partner pairs.

| metric | A (diff trajectory) | B (diff trajectory + diff history) |
|---|---|---|
| symbolic states with >=1 candidate | 0 | 0 |
| total swap candidate pairs | 0 | 0 |
| anchors (state instances) with >=1 partner | 0 | 0 |
| anchor coverage rate | 0.000000 | 0.000000 |

```


## File: `reports\coverage_report.md`
```md
# Coverage Report

- Train state count: 40
- Test state count: 12
- Overlap count: 0
- Overlap %: 0.00%
- Prefix collisions (test states with >=1 byte-identical train neighbor): 0/12 (0.00%), 0 pairs masked

```


## File: `reports\layer_sweep.json`
```json
{
  "config": {
    "model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "num_layers": 22,
    "layers": [
      12,
      22
    ],
    "cap_train": 1500,
    "cap_val": 497,
    "num_codes": 32,
    "vq_epochs": 50,
    "tr_epochs": 30,
    "seed": 0
  },
  "layers": [
    {
      "layer": 12,
      "n_train_states": 5994,
      "active_codes": 32,
      "perplexity": 29.041063182267685,
      "gini": 0.24317025358692027,
      "position_leakage": 0.6813627254509018,
      "leakage_majority": 0.24899799599198397,
      "majority": 0.08205470313542361,
      "bigram_z": 0.21480987324883255,
      "action_bigram": 0.3669112741827885,
      "mlp_z": 0.21080720480320214,
      "mlp_z_op": 0.3609072715143429,
      "mlp_z_op_operands": 0.5063375583722481,
      "mlp_action_only": 0.4202801867911941,
      "H_state": 2.3303432519356875,
      "det_state": 0.0,
      "H_action": 1.6891784508641488,
      "det_action": 0.0
    },
    {
      "layer": 22,
      "n_train_states": 5994,
      "active_codes": 32,
      "perplexity": 27.40731652358554,
      "gini": 0.31212462462462465,
      "position_leakage": 0.7780561122244489,
      "leakage_majority": 0.24899799599198397,
      "majority": 0.1094062708472315,
      "bigram_z": 0.2555036691127418,
      "action_bigram": 0.5436957971981321,
      "mlp_z": 0.25483655770513675,
      "mlp_z_op": 0.5416944629753169,
      "mlp_z_op_operands": 0.6357571714476318,
      "mlp_action_only": 0.400266844563042,
      "H_state": 2.1512385981091313,
      "det_state": 0.0,
      "H_action": 1.137102459865045,
      "det_action": 0.0
    }
  ],
  "floor": {
    "name": "Noise floor",
    "active_codes": 32,
    "gini": 0.6470388229475766,
    "perplexity": 11.562789904132892,
    "ami_codes_vs_true": -0.0012084979640879772,
    "position_leakage": 0.26285714285714284,
    "leakage_majority": 0.2484472049689441,
    "majority": 0.3696682464454976,
    "bigram_z": 0.3696682464454976,
    "mlp_z": 0.3696682464454976,
    "action_bigram": 0.36222071767095465,
    "mlp_z_op": 0.3696682464454976,
    "H_state": 2.3943378660955945,
    "det_state": 0.0,
    "H_action": 2.2593454832652253,
    "det_action": 0.0
  },
  "ceiling": {
    "name": "Positive control",
    "active_codes": 32,
    "gini": 0.42178486330574366,
    "perplexity": 21.678548764914346,
    "ami_codes_vs_true": 0.9354105793593506,
    "position_leakage": 0.24738154613466334,
    "leakage_majority": 0.24937655860349128,
    "majority": 0.0975130890052356,
    "bigram_z": 0.30824607329842935,
    "mlp_z": 0.3069371727748691,
    "action_bigram": 0.7696335078534031,
    "mlp_z_op": 0.7722513089005235,
    "H_state": 1.3410367249993937,
    "det_state": 0.0,
    "H_action": 0.38638598818770814,
    "det_action": 0.5932373619015735
  },
  "observed_layer_-1": {
    "det_action": 0.10989010989010989,
    "H_action": 0.9932319690453785,
    "mlp_z_op": 0.5580623755806238
  },
  "summary": {
    "per_layer": [
      {
        "layer": 12,
        "pos_det": 0.0,
        "pos_mlp": -0.021761906419332937,
        "pos_H": 0.3044203752935291
      },
      {
        "layer": 22,
        "pos_det": 0.0,
        "pos_mlp": 0.42730614517354903,
        "pos_H": 0.5991816835065797
      }
    ],
    "best_layer": 22,
    "best_det_action": 0.0,
    "best_pos_det": 0.0,
    "reaches_ceiling": false
  }
}
```


## File: `reports\layer_sweep.md`
```md
# V5.2 · Experiment 3 — Heavy Layer Validation (Layer 12 vs Layer 22)

> **Curated final report.** Headline = the heavy run (serious settings) on the two decisive
> depths. A breadth screen across six depths (smoke settings) is folded in below as supporting
> evidence. Raw machine artifacts: `reports/layer_sweep.json` / `.png` (heavy, 12 vs 22) and
> `reports/layer_sweep_smoke6.json` / `.png` (breadth, 6 depths). Note: re-running
> `scripts/run_layer_sweep.py` overwrites `layer_sweep.{md,json,png}` with a generic templated
> report — keep this curated copy if you hand-edit it.


## Why this experiment

A reviewer raised the standard objection: every V5.2 result probes TinyLlama's **final** layer
(index 22 / `-1`), yet planning-relevant structure often lives in **middle** layers. If so, our
"no reusable discrete dynamics" conclusion could be an artifact of *where* we probe rather than a
property of the model.

We test the strongest form of that objection directly. We re-extract hidden states at **layer 12**
— the strongest mid-stack candidate — and **layer 22** — our baseline — and run the *identical*
Experiment-1 pipeline at each, at the **same serious settings as every other V5.2 experiment** (no
smoke shortcuts). This is a targeted two-layer validation, not a full-stack sweep; the breadth
screen below covers the rest of the stack at smoke fidelity.


## Setup

- **Model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (22 layers). Index into
  `hidden_states`; the final layer = 22.
- **Data:** Countdown — **1500 train / 497 val** problems (val de-duplicated against train by
  `(target, sorted numbers)`).
- **Settings (identical to Exp1):** VQ **K=32** (data-dependent init, raw states), **50 VQ
  epochs**, **30 transition epochs**, seed 0. Per layer: VQ → action-conditioned models A–F on
  held-out val + transition structure on train + codebook usage + position leakage.
- **Anchors:** Experiment-2 noise floor and positive-control ceiling (same K, same pipeline) bound
  each action-conditioned axis.


## Per-layer metrics (heavy run)

| Layer | Active | Perplex. | Pos-leak | Bigram(z) | MLP(z) | Act-bigram | MLP(z,op) | H(z'\|z) | H(z'\|z,op) | Det(z,op) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 (middle) | 32 | 29.0 | 0.681 | 0.215 | 0.211 | 0.367 | 0.361 | 2.330 | 1.689 | 0.000 |
| 22 (final)  | 32 | 27.4 | 0.778 | 0.256 | 0.255 | 0.544 | 0.542 | 2.151 | 1.137 | 0.000 |

**Position of each layer between the Exp2 noise floor and the reusable-state ceiling** (0% = at
noise floor, 100% = at positive-control ceiling):

| Axis | Layer 12 | Layer 22 |
| --- | ---: | ---: |
| Det(z,op) — strict reusable-state metric | **0%** (0.000) | **0%** (0.000) |
| MLP(z,op) — soft predictability | **−2%** (0.361, *at the floor*) | **43%** (0.542) |
| H(z'\|z,op) — soft determinism (lower = more structure) | **30%** (1.689) | **60%** (1.137) |

_Det(z,op) = share of transition mass from near-deterministic `(z,op)` conditionings (H<0.5 nats,
seen ≥10×). It is a strict threshold metric; on this run's codebook neither layer crosses it, so
the comparative signal lives on the soft axes (MLP, H) — both of which favour the **final** layer._


## Anchors (Experiment 2)

| | Det(z,op) | H(z'\|z,op) | MLP(z,op) |
| --- | ---: | ---: | ---: |
| Noise floor | 0.000 | 2.259 | 0.370 |
| Positive-control ceiling | 0.593 | 0.386 | 0.772 |
| Cached layer −1 (Exp1, full data) | 0.110 | 0.993 | 0.558 |

---
## Verdict — Case 1: the middle-layer objection is refuted

**Layer 12 is not richer than layer 22 — it is poorer, and neither reaches the reusable-state
regime.**

- On the strict **Det(z,op)** axis both layers sit at **0.000 (0% to ceiling)**: neither produces
  crisp, reusable discrete dynamics.
- On both *soft* axes the **final** layer leads. Layer 12's **MLP(z,op) = 0.361 sits *at the noise
  floor* (0.370)** — no action-conditioned predictability above chance — while layer 22 retains the
  partial structure seen in Exp1 (MLP 0.542 ≈ cached 0.558; 43% to ceiling, vs 30%→60% on the
  entropy axis).

So the strongest mid-stack candidate is, if anything, *closer to noise* than the layer we already
report. There is **no middle-layer peak** (no "Case 2"). The layer-choice criticism is resolved:
our conclusion does not depend on probing the final layer, and probing a middle layer would have
made the structure *weaker*, not stronger.


## Breadth screen (6 depths, smoke settings) — supporting

A wider but lighter screen across `4, 8, 12, 16, 20, 22` (60 train / 28 val, 2 VQ / 1 transition
epoch) returns **Det(z,op) = 0.000 at every depth** with no mid-stack peak. At smoke fidelity only
the strict-axis result is trustworthy; its sole takeaway is the absence of any layer where reusable
dynamics emerge. The heavy run above is what licenses the *comparative* (12 vs 22) claim.

| Layer | 4 | 8 | 12 | 16 | 20 | 22 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Det(z,op) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

_Source: `reports/layer_sweep_smoke6.json`._

---
_Caveats: single model / task / seed; op-type action; two depths at heavy fidelity (rest at smoke).
Det(z,op) is a strict threshold metric — see Exp2 for why the soft entropy/MLP axes can read higher.
The heavy run re-trains VQ on this data, so layer-22 Det reads 0.000 here vs the 0.110 cached in
Exp1's full-data codebook; the comparison that matters (12 vs 22, same codebook regime) is unaffected._

_Artifacts: `reports/layer_sweep.json`, `reports/layer_sweep.png` (heavy);
`reports/layer_sweep_smoke6.json`, `reports/layer_sweep_smoke6.png` (breadth)._

```


## File: `reports\layer_sweep_smoke6.json`
```json
{
  "config": {
    "model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "num_layers": 22,
    "layers": [
      4,
      8,
      12,
      16,
      20,
      22
    ],
    "cap_train": 60,
    "cap_val": 28,
    "num_codes": 32,
    "vq_epochs": 2,
    "tr_epochs": 1,
    "seed": 0
  },
  "layers": [
    {
      "layer": 4,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 26.488196961083666,
      "gini": 0.3338381057268722,
      "position_leakage": 0.4700854700854701,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.12359550561797752,
      "bigram_z": 0.11235955056179775,
      "action_bigram": 0.1348314606741573,
      "mlp_z": 0.011235955056179775,
      "mlp_z_op": 0.07865168539325842,
      "mlp_z_op_operands": 0.011235955056179775,
      "mlp_action_only": 0.07865168539325842,
      "H_state": 1.4787399373288876,
      "det_state": 0.0,
      "H_action": 0.8246554754396366,
      "det_action": 0.0
    },
    {
      "layer": 8,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 22.652298534072358,
      "gini": 0.45966409691629956,
      "position_leakage": 0.5470085470085471,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.1348314606741573,
      "bigram_z": 0.14606741573033707,
      "action_bigram": 0.1348314606741573,
      "mlp_z": 0.011235955056179775,
      "mlp_z_op": 0.0449438202247191,
      "mlp_z_op_operands": 0.07865168539325842,
      "mlp_action_only": 0.0898876404494382,
      "H_state": 1.476024860294629,
      "det_state": 0.0,
      "H_action": 0.8838621257271394,
      "det_action": 0.0
    },
    {
      "layer": 12,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 26.810312730717854,
      "gini": 0.3283314977973568,
      "position_leakage": 0.6495726495726496,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.0449438202247191,
      "bigram_z": 0.1348314606741573,
      "action_bigram": 0.1797752808988764,
      "mlp_z": 0.011235955056179775,
      "mlp_z_op": 0.056179775280898875,
      "mlp_z_op_operands": 0.033707865168539325,
      "mlp_action_only": 0.06741573033707865,
      "H_state": 1.6557223500336047,
      "det_state": 0.0,
      "H_action": 0.9203158722086672,
      "det_action": 0.0
    },
    {
      "layer": 16,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 26.605320121408813,
      "gini": 0.3371420704845815,
      "position_leakage": 0.6239316239316239,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.0898876404494382,
      "bigram_z": 0.0898876404494382,
      "action_bigram": 0.20224719101123595,
      "mlp_z": 0.011235955056179775,
      "mlp_z_op": 0.0449438202247191,
      "mlp_z_op_operands": 0.0449438202247191,
      "mlp_action_only": 0.07865168539325842,
      "H_state": 1.638921460874948,
      "det_state": 0.0,
      "H_action": 0.7970218722775848,
      "det_action": 0.0
    },
    {
      "layer": 20,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 25.741275247284698,
      "gini": 0.3674284140969163,
      "position_leakage": 0.7264957264957265,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.07865168539325842,
      "bigram_z": 0.16853932584269662,
      "action_bigram": 0.39325842696629215,
      "mlp_z": 0.02247191011235955,
      "mlp_z_op": 0.033707865168539325,
      "mlp_z_op_operands": 0.011235955056179775,
      "mlp_action_only": 0.06741573033707865,
      "H_state": 1.668794217908028,
      "det_state": 0.0,
      "H_action": 0.7848076555029203,
      "det_action": 0.0
    },
    {
      "layer": 22,
      "n_train_states": 227,
      "active_codes": 32,
      "perplexity": 24.551186164833137,
      "gini": 0.40597466960352424,
      "position_leakage": 0.7264957264957265,
      "leakage_majority": 0.23931623931623933,
      "majority": 0.0898876404494382,
      "bigram_z": 0.15730337078651685,
      "action_bigram": 0.42696629213483145,
      "mlp_z": 0.02247191011235955,
      "mlp_z_op": 0.056179775280898875,
      "mlp_z_op_operands": 0.0,
      "mlp_action_only": 0.07865168539325842,
      "H_state": 1.5723234468159677,
      "det_state": 0.0,
      "H_action": 0.7167781256016749,
      "det_action": 0.0
    }
  ],
  "floor": {
    "name": "Noise floor",
    "active_codes": 32,
    "gini": 0.6470388229475766,
    "perplexity": 11.562789904132892,
    "ami_codes_vs_true": -0.0012084979640879772,
    "position_leakage": 0.26285714285714284,
    "leakage_majority": 0.2484472049689441,
    "majority": 0.3696682464454976,
    "bigram_z": 0.3696682464454976,
    "mlp_z": 0.3696682464454976,
    "action_bigram": 0.36222071767095465,
    "mlp_z_op": 0.3696682464454976,
    "H_state": 2.3943378660955945,
    "det_state": 0.0,
    "H_action": 2.2593454832652253,
    "det_action": 0.0
  },
  "ceiling": {
    "name": "Positive control",
    "active_codes": 32,
    "gini": 0.42178486330574366,
    "perplexity": 21.678548764914346,
    "ami_codes_vs_true": 0.9354105793593506,
    "position_leakage": 0.24738154613466334,
    "leakage_majority": 0.24937655860349128,
    "majority": 0.0975130890052356,
    "bigram_z": 0.30824607329842935,
    "mlp_z": 0.3069371727748691,
    "action_bigram": 0.7696335078534031,
    "mlp_z_op": 0.7722513089005235,
    "H_state": 1.3410367249993937,
    "det_state": 0.0,
    "H_action": 0.38638598818770814,
    "det_action": 0.5932373619015735
  },
  "observed_layer_-1": {
    "det_action": 0.10989010989010989,
    "H_action": 0.9932319690453785,
    "mlp_z_op": 0.5580623755806238
  },
  "summary": {
    "per_layer": [
      {
        "layer": 4,
        "pos_det": 0.0,
        "pos_mlp": -0.7228733352008562,
        "pos_H": 0.7660016202145421
      },
      {
        "layer": 8,
        "pos_det": 0.0,
        "pos_mlp": -0.806602305225036,
        "pos_H": 0.7343903384740085
      },
      {
        "layer": 12,
        "pos_det": 0.0,
        "pos_mlp": -0.7786926485503095,
        "pos_H": 0.7149271591701661
      },
      {
        "layer": 16,
        "pos_det": 0.0,
        "pos_mlp": -0.806602305225036,
        "pos_H": 0.7807555982021482
      },
      {
        "layer": 20,
        "pos_det": 0.0,
        "pos_mlp": -0.8345119618997625,
        "pos_H": 0.7872769441291508
      },
      {
        "layer": 22,
        "pos_det": 0.0,
        "pos_mlp": -0.7786926485503095,
        "pos_H": 0.8235988881327662
      }
    ],
    "best_layer": 4,
    "best_det_action": 0.0,
    "best_pos_det": 0.0,
    "reaches_ceiling": false
  }
}
```


## File: `reports\multiseed_results.csv`
```csv
Architecture,Seed,True Action SG,Shuffled Action SG,Constant Action SG,Blind SG,Oracle SG,Oracle Gap,Cosine,Mean-Centered Cosine,MSE
linear,42,-0.07499999999999996,-0.07499999999999996,-0.04999999999999993,-0.08750000000000002,-0.03749999999999998,0.03749999999999998,0.8916218936443329,0.3248457968235015,0.7587254464626312
linear,43,-0.08750000000000002,-0.07499999999999996,-0.03749999999999998,-0.04999999999999993,-0.03749999999999998,0.050000000000000044,0.8925475746393203,0.325771477818489,0.751734958589077
linear,44,-0.09999999999999998,-0.07499999999999996,-0.03749999999999998,-0.0625,-0.03749999999999998,0.0625,0.8923281311988831,0.3255520343780518,0.7554293960332871
mlp,42,-0.0625,-0.07499999999999996,-0.025000000000000022,-0.04999999999999993,-0.03749999999999998,0.025000000000000022,0.893628278374672,0.3268521815538406,0.7347443625330925
mlp,43,-0.03749999999999998,-0.03749999999999998,-0.012499999999999956,-0.03749999999999998,-0.03749999999999998,0.0,0.8923725098371506,0.3255964130163192,0.7475928485393524
mlp,44,-0.07499999999999996,-0.07499999999999996,-0.04999999999999993,-0.03749999999999998,-0.03749999999999998,0.03749999999999998,0.892347127199173,0.3255710303783417,0.7441744327545166
transformer,42,0.0,-0.025000000000000022,-0.025000000000000022,-0.03749999999999998,-0.03749999999999998,-0.03749999999999998,0.8945990145206452,0.3278229176998138,0.7374177813529968
transformer,43,0.012500000000000067,-0.012499999999999956,-0.025000000000000022,-0.04999999999999993,-0.03749999999999998,-0.050000000000000044,0.8945491135120391,0.3277730166912078,0.7363465994596481
transformer,44,0.0,0.0,-0.025000000000000022,-0.025000000000000022,-0.03749999999999998,-0.03749999999999998,0.8950793504714966,0.3283032536506652,0.7326967597007752

```


## File: `reports\multi_seed_results.csv`
```csv
Architecture,Seed,Semantic Gain,Oracle Gap,Cosine,MSE
linear,42,-0.07499999999999996,0.11508000000000002,0.8916219413280487,0.75872553139925
linear,43,-0.08750000000000002,0.11111100000000002,0.8925475031137466,0.7517350748181343
linear,44,-0.09999999999999998,0.11904799999999999,0.8923284560441971,0.7554270699620247
mlp,42,-0.0625,0.09920600000000002,0.8936283409595489,0.7347442373633385
mlp,43,-0.08750000000000002,0.10714299999999999,0.8920401424169541,0.7458958730101586
mlp,44,-0.07499999999999996,0.09920600000000002,0.8923472166061401,0.7441744193434715
transformer,42,0.0,0.09126999999999996,0.8908125817775726,0.7585051104426384
transformer,43,0.0,0.05952400000000002,0.8946233868598938,0.7357936844229698
transformer,44,-0.025000000000000022,0.08730199999999999,0.893699124455452,0.7380437552928925

```


## File: `reports\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 19874 |
| states_with_swap_partner | 225 |
| states_without_swap_partner | 19649 |
| coverage_rate | 0.011321 |
| coverage_rate_95ci | [0.009942, 0.012890] |
| total_state_instances | 26374 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| 0 | 235 |
| all | 235 |

```


## File: `reports\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.957130,19874,,
oracle0,probeA_per_label,all,0.986502,19874,,
oracle0,probeA_jaccard,all,0.974011,19874,,
oracle0,probeB_accuracy,all,0.669065,19874,,
oracle0,probeC_accuracy,all,0.653014,19874,,
oracle0,probeD_accuracy,all,0.838080,19874,,
coverage,coverage_rate,all,0.011321,19874,0.009942,0.012890
oracle1,probeA_exact,all,0.417379,19874,0.410523,0.424236
oracle1,probeA_per_label,all,0.814984,19874,,
oracle1,probeA_jaccard,all,0.653127,19874,,
oracle2a,gtA_exact,all,0.404255,235,0.341377,0.467133
oracle2a,gtA_exact,0,0.404255,235,,
oracle2a,gtA_jaccard,all,0.846809,235,0.808407,0.885210
oracle2a,gtA_jaccard,0,0.846809,235,,
oracle2a,gtA_per_label,all,0.743617,235,0.709292,0.777942
oracle2a,gtA_per_label,0,0.743617,235,,
oracle2a,gtB_abs_diff,all,0.255319,235,0.187068,0.323570
oracle2a,gtB_abs_diff,0,0.255319,235,,
oracle2a,gtB_agreement,all,0.791489,235,0.739439,0.843540
oracle2a,gtB_agreement,0,0.791489,235,,
oracle2a,gtC_agreement,all,0.857143,235,0.811215,0.903070
oracle2a,gtC_agreement,0,0.857143,235,,
oracle2a,gtD_agreement,all,0.906383,235,0.869060,0.943706
oracle2a,gtD_agreement,0,0.906383,235,,
oracle2a,probeA_pred_exact,all,0.336170,235,0.275643,0.396697
oracle2a,probeA_pred_exact,0,0.336170,235,,
oracle2a,probeA_pred_jaccard,all,0.755319,235,0.708327,0.802312
oracle2a,probeA_pred_jaccard,0,0.755319,235,,
oracle2a,probeA_pred_per_label,all,0.698936,235,0.662492,0.735380
oracle2a,probeA_pred_per_label,0,0.698936,235,,
oracle2a,probeB_pred_abs_diff,all,0.255319,235,0.188100,0.322538
oracle2a,probeB_pred_abs_diff,0,0.255319,235,,
oracle2a,probeB_pred_agreement,all,0.787234,235,0.734796,0.839672
oracle2a,probeB_pred_agreement,0,0.787234,235,,
oracle2a,probeC_pred_agreement,all,0.753191,235,0.697949,0.808434
oracle2a,probeC_pred_agreement,0,0.753191,235,,
oracle2a,probeD_pred_agreement,all,0.957447,235,0.931585,0.983309
oracle2a,probeD_pred_agreement,0,0.957447,235,,
oracle2b,delta_transition_accuracy,all,0.182979,235,0.114498,0.251460
oracle2b,delta_transition_accuracy,0,0.182979,235,,
oracle2b,oracle1_paired_exact,all,0.306383,235,0.247318,0.365448
oracle2b,oracle1_paired_exact,0,0.306383,235,,
oracle2b,probeA_exact,all,0.123404,235,0.081263,0.165545
oracle2b,probeA_exact,0,0.123404,235,,
oracle2b,probeA_jaccard,all,0.527660,235,0.474205,0.581114
oracle2b,probeA_jaccard,0,0.527660,235,,
oracle2b,probeA_per_label,all,0.561702,235,0.527772,0.595633
oracle2b,probeA_per_label,0,0.561702,235,,

```


## File: `reports\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 19874 |
| states_with_swap_partner | 225 |
| states_without_swap_partner | 19649 |
| coverage_rate | 0.011321 |
| coverage_rate_ci_lo | 0.009942 |
| coverage_rate_ci_hi | 0.012890 |
| total_state_instances | 26374 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.957130 |
| probeA_per_label | 0.986502 |
| probeA_jaccard | 0.974011 |
| probeB_accuracy | 0.669065 |
| probeC_accuracy | 0.653014 |
| probeD_accuracy | 0.838080 |
| n | 19874 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.417379 |
| probeA_exact_ci_lo | 0.410523 |
| probeA_exact_ci_hi | 0.424236 |
| probeA_per_label | 0.814984 |
| probeA_jaccard | 0.653127 |
| n | 19874 |

## 4. Oracle 2A — Representation Agreement

| metric | all | dd=0 |
|---|---|---|
| probeA_pred_exact | 0.336170 | 0.336170 |
| probeA_pred_per_label | 0.698936 | 0.698936 |
| probeA_pred_jaccard | 0.755319 | 0.755319 |
| probeB_pred_abs_diff | 0.255319 | 0.255319 |
| probeB_pred_agreement | 0.787234 | 0.787234 |
| probeC_pred_agreement | 0.753191 | 0.753191 |
| probeD_pred_agreement | 0.957447 | 0.957447 |
| gtA_exact | 0.404255 | 0.404255 |
| gtA_per_label | 0.743617 | 0.743617 |
| gtA_jaccard | 0.846809 | 0.846809 |
| gtB_abs_diff | 0.255319 | 0.255319 |
| gtB_agreement | 0.791489 | 0.791489 |
| gtC_agreement | 0.857143 | 0.857143 |
| gtD_agreement | 0.906383 | 0.906383 |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | 0.336170 | 0.275643 | 0.396697 | 235 |
| probeA_pred_per_label | 0.698936 | 0.662492 | 0.735380 | 235 |
| probeA_pred_jaccard | 0.755319 | 0.708327 | 0.802312 | 235 |
| probeB_pred_abs_diff | 0.255319 | 0.188100 | 0.322538 | 235 |
| probeB_pred_agreement | 0.787234 | 0.734796 | 0.839672 | 235 |
| probeC_pred_agreement | 0.753191 | 0.697949 | 0.808434 | 235 |
| probeD_pred_agreement | 0.957447 | 0.931585 | 0.983309 | 235 |
| gtA_exact | 0.404255 | 0.341377 | 0.467133 | 235 |
| gtA_per_label | 0.743617 | 0.709292 | 0.777942 | 235 |
| gtA_jaccard | 0.846809 | 0.808407 | 0.885210 | 235 |
| gtB_abs_diff | 0.255319 | 0.187068 | 0.323570 | 235 |
| gtB_agreement | 0.791489 | 0.739439 | 0.843540 | 235 |
| gtC_agreement | 0.857143 | 0.811215 | 0.903070 | 224 |
| gtD_agreement | 0.906383 | 0.869060 | 0.943706 | 235 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all | dd=0 |
|---|---|---|
| probeA_exact | 0.123404 | 0.123404 |
| probeA_per_label | 0.561702 | 0.561702 |
| probeA_jaccard | 0.527660 | 0.527660 |

## 6. Oracle 2B — Delta

| metric | all | dd=0 |
|---|---|---|
| oracle1_paired_exact | 0.306383 | 0.306383 |
| delta_transition_accuracy | 0.182979 | 0.182979 |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | 0.123404 | 0.081263 | 0.165545 | 235 |
| probeA_per_label | 0.561702 | 0.527772 | 0.595633 | 235 |
| probeA_jaccard | 0.527660 | 0.474205 | 0.581114 | 235 |
| oracle1_paired_exact | 0.306383 | 0.247318 | 0.365448 | 235 |
| delta_transition_accuracy | 0.182979 | 0.114498 | 0.251460 | 235 |

```


## File: `reports\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20

> ⚠️ **Smoke-test run** with a randomly-initialized model. The numbers below validate the *pipeline*, not the science. Re-run with the TinyLlama teacher for real conclusions.

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.359 | 0.400 | 0.150 | 0.715 | 0.758 | 4.7603 |
| 2 | 0.349 | 0.412 | 0.588 | 0.742 | 0.823 | 5.1674 |
| 3 | 0.352 | 0.167 | 0.667 | 0.751 | 0.833 | 6.0896 |
| 4 | 0.349 | nan | nan | 0.769 | 0.846 | 7.5153 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **4**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~4 step(s)**.

Single-step transition validation MSE: 2.8268.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.758 | 0.150 | 20 |
| 2 | 0.823 | 0.588 | 20 |
| 3 | 0.833 | 0.667 | 17 |
| 4 | 0.846 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.359 | 0.354 | +0.641 |
| 2 | 1.000 | 0.349 | 0.347 | +0.651 |
| 3 | 1.000 | 0.352 | 0.348 | +0.648 |
| 4 | 1.000 | 0.349 | 0.346 | +0.651 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Game24 (Remaining card multiset encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.758 | 0.715 | 0.735 | +0.042 |
| 2 | 0.823 | 0.742 | 0.781 | +0.081 |
| 3 | 0.833 | 0.751 | 0.801 | +0.081 |
| 4 | 0.846 | 0.769 | 0.808 | +0.077 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.400 | -0.250 |
| 2 | 0.588 | 0.412 | 0.471 | +0.176 |
| 3 | 0.667 | 0.167 | 0.167 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.008 | -0.833 | -0.000 |
| 2 | +0.003 | -0.909 | -0.500 |
| 3 | +0.006 | -1.571 | +0.000 |
| 4 | +0.003 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.359 | 0.354 | 0.287 | 0.758 | 0.715 | 0.735 | 0.712 |
| 2 | 1.000 | 0.349 | 0.347 | 0.246 | 0.823 | 0.742 | 0.781 | 0.738 |
| 3 | 1.000 | 0.352 | 0.348 | 0.241 | 0.833 | 0.751 | 0.801 | 0.715 |
| 4 | 1.000 | 0.349 | 0.346 | 0.237 | 0.846 | 0.769 | 0.808 | 0.808 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.6480
**Average Oracle Gain (state probe):** +0.0704
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.8149
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case B — Transition model IS the bottleneck.** The Oracle significantly outperforms the action-conditioned transition. The representation contains usable planning information that the learned dynamics fail to preserve. Improving the transition model (or its training) is the highest-leverage intervention.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\phase_c4a_controls.csv`
```csv
control,Depth Acc,Rem Nums (A) Exact,Next Op (Probe C),Retrieval Top-1,Retrieval Top-5
PERMUTED null,0.08333333333333333,0.08333333333333333,0.3333333333333333,0.0,0.0
Raw-operand baseline,0.3333333333333333,0.4166666666666667,0.4166666666666667,0.0,0.0

```


## File: `reports\phase_c4a_geometry.csv`
```csv
Removed PCs,Within/Between Ratio
0,0.0
1,0.0
2,0.0
5,0.0
10,0.0
20,0.0

```


## File: `reports\phase_c4a_pca_ablation_metrics.csv`
```csv
Removed PCs,Depth Acc,Rem Nums (A) Exact,Rem Nums (A) Hamming,Next Op (Probe C)
0,0.75,0.16666666666666666,0.5,0.4166666666666667
1,0.75,0.16666666666666666,0.4583333333333333,0.4166666666666667
2,0.75,0.16666666666666666,0.5,0.4166666666666667
5,0.5,0.16666666666666666,0.4583333333333333,0.4166666666666667
10,0.5,0.16666666666666666,0.5625,0.4166666666666667
20,0.25,0.16666666666666666,0.5416666666666666,0.25

```


## File: `reports\phase_c4a_pca_ablation_report.md`
```md
[Error reading file: 'utf-8' codec can't decode byte 0x97 in position 13: invalid start byte]
```


## File: `reports\phase_c4a_retrieval.csv`
```csv
Removed PCs,Retrieval Top-1,Retrieval Top-5,Retrieval Top-10,Top-1 Gain
0,0.0,0.0,0.0,1.0
1,0.0,0.0,0.0,1.0
2,0.0,0.0,0.0,1.0
5,0.0,0.0,0.0,1.0
10,0.0,0.0,0.0,1.0
20,0.0,0.0,0.0,1.0

```


## File: `reports\positive_control.json`
```json
{
  "config": {
    "num_codes": 32,
    "num_states": 32,
    "num_actions": 3,
    "hidden_dim": 2048,
    "noise": 0.3,
    "seed": 0
  },
  "positive_control": {
    "name": "Positive control",
    "active_codes": 32,
    "gini": 0.42178486330574366,
    "perplexity": 21.678548764914346,
    "ami_codes_vs_true": 0.9354105793593506,
    "position_leakage": 0.24738154613466334,
    "leakage_majority": 0.24937655860349128,
    "majority": 0.0975130890052356,
    "bigram_z": 0.30824607329842935,
    "mlp_z": 0.3069371727748691,
    "action_bigram": 0.7696335078534031,
    "mlp_z_op": 0.7722513089005235,
    "H_state": 1.3410367249993937,
    "det_state": 0.0,
    "H_action": 0.38638598818770814,
    "det_action": 0.5932373619015735
  },
  "noise_floor": {
    "name": "Noise floor",
    "active_codes": 32,
    "gini": 0.6470388229475766,
    "perplexity": 11.562789904132892,
    "ami_codes_vs_true": -0.0012084979640879772,
    "position_leakage": 0.26285714285714284,
    "leakage_majority": 0.2484472049689441,
    "majority": 0.3696682464454976,
    "bigram_z": 0.3696682464454976,
    "mlp_z": 0.3696682464454976,
    "action_bigram": 0.36222071767095465,
    "mlp_z_op": 0.3696682464454976,
    "H_state": 2.3943378660955945,
    "det_state": 0.0,
    "H_action": 2.2593454832652253,
    "det_action": 0.0
  },
  "oracle_true_states": {
    "H_state": 1.0441182367066004,
    "det_state": 0.0,
    "H_action": 0.0,
    "det_action": 1.0
  },
  "countdown_observed": {
    "name": "Countdown (observed)",
    "ami_codes_vs_true": null,
    "bigram_z": 0.26343729263437293,
    "mlp_z": 0.2660915726609157,
    "action_bigram": 0.5534173855341739,
    "mlp_z_op": 0.5580623755806238,
    "majority": 0.11612475116124751,
    "H_state": 1.9768534756904257,
    "det_state": 0.0,
    "H_action": 0.9932319690453785,
    "det_action": 0.10989010989010989,
    "active_codes": 32,
    "gini": 0.37094835123657255,
    "perplexity": 25.684334586708843,
    "position_leakage": 0.7900150526843954,
    "leakage_majority": 0.2508780732563974
  }
}
```


## File: `reports\positive_control.md`
```md
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

```


## File: `reports\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.767 | 0.032 | 0.829 | 0.000 | n/a | weak / not encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- A:remaining_operands
- C:next_operation

_Note: probe A (remaining numbers) is restricted to the fixed set {25,50,75,100} so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.7667887667887668,0.0,,0.031746031746031744
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\probe_results_old.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_numbers,0.5403225806451613,0.0,,0.12903225806451613
B:distance_to_solution,0.45161290322580644,0.29722222222222217,,
C:next_operation,0.5161290322580645,0.4807692307692308,0.5788647342995169,
D:reachable_within_2,0.6451612903225806,0.7843137254901961,0.9363636363636363,

```


## File: `reports\qwen_action_conditioned.md`
```md
# V5.3 · Experiment A — Action-Conditioned Validation (Qwen2.5-1.5B)

> **Replication question:** Are the V5.2 action-conditioned conclusions specific
> to TinyLlama, or do they hold on a stronger, more modern model?
> This runs the **exact** V5.2 Experiment-1 pipeline
> (`scripts/run_action_conditioned.py` → `evaluation/action_conditioned.py`)
> with the frozen teacher swapped from
> `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` to `Qwen/Qwen2.5-1.5B`.
> No metrics, baselines, or thresholds were changed.

## Setup & adjustments (documented)

- **Model:** `Qwen/Qwen2.5-1.5B` (28 layers, hidden 1536, fp16), frozen. Last
  layer (`-1`), exactly as V5.2.
- **Data:** identical Countdown splits. The hidden-state extraction, VQ
  (K=32, data-dependent init, EMA), INLP position-scrub, and A–F evaluation are
  the unchanged V5.2 code paths.
- **One adjustment — train scale-matching.** The cached TinyLlama trajectories
  were extracted with `cap=1000` train problems (1000 trajectories / **3003**
  transitions). The fresh Qwen extraction defaulted to all 5000. To isolate the
  *model* as the only variable, Qwen train was capped to the **same first 1000
  problems**; this reproduces TinyLlama's transition count to the unit
  (**3003 train / 1507 val** transitions), confirming the two runs see the
  identical Countdown problems. The full 5000-problem Qwen extraction is retained
  (`reports_qwen/trajectories/train_full.pt`) for an optional data-scale
  sensitivity check.
- **Codebook health (no collapse):** Qwen fixed-init VQ uses **32/32** codes
  (Gini 0.356, perplexity 25.8); scrubbed **32/32** (Gini 0.271). Comparable to
  TinyLlama — the comparison is not confounded by collapse.

## Results — Fixed-Init codebook (primary), held-out val

| Model | Conditioning | TinyLlama top1 | **Qwen top1** | Qwen pred-entropy |
| --- | --- | ---: | ---: | ---: |
| A. Majority | — | 0.116 | **0.106** | — |
| B. Bigram | z | 0.263 | **0.316** | — |
| C. Action bigram | z, op | 0.553 | **0.464** | — |
| D. MLP(z) | z | 0.266 | **0.317** | 1.896 |
| E. MLP(z, op) | z, op | 0.558 | **0.464** | 1.493 |
| E2. MLP(z, op, operands) | z, op, operands | 0.622 | **0.509** | 1.380 |
| F. MLP(op, operands) — control | op, operands | 0.405 | **0.277** | 2.006 |

### Transition structure (train, frequency-weighted)

| Conditioning | TinyLlama H (nats) | **Qwen H** | TinyLlama det-frac | **Qwen det-frac** |
| --- | ---: | ---: | ---: | ---: |
| z | 1.977 | **1.820** | 0.000 | **0.000** |
| z, op | 0.993 | **1.341** | 0.110 | **0.006** |

### Decisive contrasts (Qwen)

| Contrast | Meaning | TinyLlama | **Qwen** |
| --- | --- | ---: | ---: |
| C − B | does the action help the lookup? | +0.290 | **+0.149** |
| E − D | does the action help the MLP? | +0.292 | **+0.148** |
| **E − C** | **does MLP(z,a) beat the action bigram?** | **+0.005** | **+0.000** |
| E2 − E | do operands add information? | +0.064 | **+0.044** |
| E2 − F | does the state matter (not just the action)? | +0.217 | **+0.232** |
| det-frac gain (z→z,op) | reusable dynamics from the action | +0.110 | **+0.006** |

### Robustness — Scrubbed codebook (Qwen)

After INLP removal of the linearly-decodable position subspace (300/1536 dims,
position decodability 0.905 → 0.528), the action-conditioned MLP again **fails to
beat** the action bigram: **E − C = −0.005**, det-frac(z,op) = **0.000**. The
negative result is not an artifact of position leakage.

## Interpretation (skeptical)

**The two decisive V5.2 signatures replicate — and on Qwen they are if anything
*cleaner*:**

1. **MLP(z,a) ≈ action-aware bigram.** TinyLlama E−C = +0.005; Qwen E−C =
   **+0.000** (scrubbed −0.005). The learned next-code model captures **nothing**
   beyond an action-conditioned count lookup, in both models. This was the
   central V5.2 finding ("no learned dynamics beyond first order"). It holds.

2. **No reusable / deterministic planning states.** The near-deterministic
   transition mass is det-frac(z,op) = **0.006** on Qwen — below even the 0.05
   "partial" bar, and lower than TinyLlama's already-sub-threshold 0.110.
   H(z'|z,op) = **1.341 nats** (~3.8 effective successors) is *higher* (more
   stochastic) than TinyLlama's 0.993. The stronger model produces **less**
   deterministic action-conditioned structure, not more.

**The shared first-order effect is present in both.** Conditioning on the action
raises predictability (C−B, E−D both positive) and the state is not vacuous
(E2−F = +0.232 > 0): codes carry information beyond the action. But on Qwen the
action effect is *smaller* (C−B +0.149 vs +0.290), and it composes into weaker,
more stochastic transitions.

**Verdict (Experiment A):** the pipeline's own three-way logic labels Qwen
"partial" — but for the same reason it did for TinyLlama: the first-order action
effect plus an entropy drop, **not** reusable states. Read against the
reusable-state criteria (MLP > bigram; a deterministic low-entropy subset), Qwen
is a clean **negative**, matching TinyLlama. The bigger, more modern model does
not rescue the planning-state hypothesis at the final layer.

_Caveats: single seed (0), last layer only, op-type action (DIV absent). The
final-layer-only scope is exactly what Experiment B (layer sweep) would address._

_Artifacts: `reports_qwen/action_conditioned.{md,json}`,
`reports_qwen/V5_1_FINAL.json`, `reports_qwen/discrete_fixed_init/`,
`reports_qwen/discrete_scrubbed/`._

```


## File: `reports\qwen_layer_validation.md`
```md
# V5.3 · Experiment B — Heavy Layer Validation (Qwen2.5-1.5B)

> **Question:** Experiment A probed only the final layer. Does a *middle*-layer
> peak rescue the planning-state hypothesis in a stronger model — i.e. is
> mid-stack discrete structure more deterministic / more predictable than the
> final layer?

## Setup

- **Model:** `Qwen/Qwen2.5-1.5B` (28 layers). Swept **layer 14** (middle, 50%
  depth — the closest equivalent to TinyLlama's layer 12 of 22) and **layer 28**
  (final, = TinyLlama's layer 22). Two layers only, per the Experiment-B spec.
- **Pipeline:** the identical V5.2 layer-sweep (`scripts/run_layer_sweep.py`) —
  fresh hidden-state extraction at each depth → VQ K=32 (50 epochs, data-dependent
  init) → action-conditioned models A–F (30 epochs) + codebook usage + position
  leakage. `cap_train=1500`, full val split (497 after de-dup). Seed 0.
- **Calibration:** a Qwen-matched positive control (1536-d noise floor + FSM
  ceiling, `reports_qwen/positive_control.json`) was subsequently generated, so the
  layer rows are placed on the floor→ceiling axes (`→ceiling` columns below).

## Per-layer metrics

| Layer | Active | Perplx | Pos-leak | Act-bigram | **MLP(z,op)** | **H(z'\|z,op)** | **Det(z,op)** | **E−C** | →ceiling det / MLP / H |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 (middle) | 32 | 29.2 | 0.604 | 0.226 | **0.233** | **2.12** | **0.000** | **+0.007** | 0% / +9% / +23% |
| 28 (final) | 32 | 28.3 | 0.799 | 0.441 | **0.434** | **1.50** | **0.000** | **−0.007** | 0% / +44% / +59% |

_(Final-layer row reproduces Experiment A's final-layer result up to the
cap_train 1500-vs-1000 / VQ-reinit difference: MLP(z,op) 0.434 vs 0.464, E−C
−0.007 vs +0.000, det ≈ 0 vs 0.006 — internally consistent.)_

## Verdict — **No middle-layer advantage. Conclusion #6 replicates.**

On every axis the middle layer is the **weaker**, not the stronger, planning-state
candidate:

- **Determinism:** det(z,op) = **0.000 at both depths.** No near-deterministic
  reusable subset appears anywhere in the stack.
- **Predictability:** MLP(z,op) is **0.233 (middle) vs 0.434 (final)** — the
  middle layer is *worse* by −0.201.
- **Entropy:** H(z'|z,op) is **2.12 (middle) vs 1.50 (final)** nats — the middle
  layer is *more* stochastic (~8.3 vs ~4.5 effective successors).
- **MLP(z,a) ≈ action-bigram at both depths:** E−C = +0.007 (middle), −0.007
  (final). The learned model captures nothing beyond the action-conditioned count
  at *any* probed depth.

The final layer is the strongest discrete representation of the two, and even it
carries no reusable dynamics. On the Qwen-matched floor→ceiling axes, the middle
layer reaches only +9% (MLP) / +23% (entropy) of the way to the FSM ceiling and
the final layer +44% / +59% — **0% on the strict det-frac axis at both depths**,
and `reaches_ceiling = False`. The TinyLlama finding — *"middle layers were not
stronger than the final layer"* — holds on Qwen.

_Caveats: two layers only (14, 28), single seed, op-type action. Calibration uses
a clean synthetic FSM (upper-bound ceiling); see `qwen_positive_control.md`._

_Artifacts: `reports_qwen/layer_sweep.json`, `reports_qwen/layer_sweep.png`,
`scratch_logs/qwen_layer_sweep.log`._

```


## File: `reports\qwen_positive_control.md`
```md
# V5.3 · Qwen Positive Control — Methodology Calibration (Qwen-matched anchors)

> **Why this exists.** Experiments A and B compared Qwen's Countdown numbers
> qualitatively against the *TinyLlama* floor/ceiling. A reviewer could object:
> *"you carried the TinyLlama calibration over to Qwen."* This closes that nit by
> rebuilding the noise floor and reusable-state ceiling at **Qwen's** hidden
> dimensionality (1536-d), then placing Qwen's observed Countdown structure
> between them. It is a calibration of the *pipeline*, not a new LM experiment —
> the floor/ceiling come from a synthetic FSM, not from any model.

## Setup

Identical to V5.2 Experiment 2, only `hidden_dim` changed 2048 → **1536** to match
Qwen. Deterministic FSM (32 states, 3 actions), each state rendered as
`proto[s] + noise(0.3)`. **Ceiling** = structured embeddings (genuine reusable
`(s,a)→s'`); **floor** = same walks, structureless embeddings. K=32, VQ 50 ep,
transition 30 ep, seed 0. Qwen's observed Countdown (final layer) is read from
`reports_qwen/action_conditioned.json` + `reports_qwen/V5_1_FINAL.json`.

## Methodology validated for Qwen's dimensionality

The pipeline recovers known structure when it exists and reads ~noise when it
does not — it is not biased toward either answer at 1536-d:

| | Noise floor | **Countdown (Qwen, final)** | FSM ceiling | Oracle (true states) |
| --- | ---: | ---: | ---: | ---: |
| AMI(codes, true states) | 0.001 | — | **0.815** | 1.000 |
| MLP(z, a) top1 | 0.178 | 0.464 | 0.765 | — |
| Action bigram (z, a) | 0.174 | 0.464 | 0.765 | — |
| H(z'\|z, a) nats | 2.526 | 1.341 | 0.782 | 0.000 |
| **Det-frac (z, a)** | **0.000** | **0.006** | **0.586** | **1.000** |
| Position leakage | 0.238 | 0.779 | 0.236 | — |

AMI 0.815 (≫ 0; floor 0.001) confirms the VQ codes recover the ground-truth
states on the ceiling; det-frac jumps 0.000 → 0.586 and entropy falls 2.53 → 0.78
nats. The methodology is sound at Qwen's dimensionality.

## Where Qwen's Countdown lands (Qwen-matched anchors)

Placed on each floor→ceiling axis:

| Axis | floor → ceiling | Qwen Countdown | % of the way to ceiling | TinyLlama (its own anchors) |
| --- | ---: | ---: | ---: | ---: |
| **Det-frac(z,a)** | 0.000 → 0.586 | 0.006 | **1%** | 19% |
| MLP(z,a) acc | 0.178 → 0.765 | 0.464 | **49%** | 47% |
| H(z'\|z,a) | 2.526 → 0.782 | 1.341 | **68%** (down) | 68% |

## Calibrated conclusion (unchanged from TinyLlama, now on Qwen's own anchors)

Qwen's action-conditioned structure is **real and clearly above the noise floor**
(every metric beats the floor: MLP 0.464 vs 0.178; H 1.341 vs 2.526 nats) but
falls **well short of the reusable-state ceiling** (MLP 0.765, H 0.782, det 0.586).
The axes disagree exactly as for TinyLlama: the strict det-frac bar puts Countdown
~1% of the way up, the soft accuracy/entropy axes ~49–68%. The honest reading is
**graded, partial action-conditioned dependence — not reusable planning states.**

Relative to TinyLlama on its own anchors, Qwen is **indistinguishable on the soft
axes** (MLP 49% vs 47%, H 68% vs 68%) and **lower on the strict determinism axis**
(1% vs 19%) — i.e. the larger model is, if anything, slightly *further* from the
reusable-state ceiling. The calibration carries over; the conclusion does not change.

_Note: the auto-generated `reports_qwen/positive_control.md` reuses the V5.2 report
template, whose title prose hardcodes TinyLlama's example "0.99 / 0.11" — the data
rows and percentages in that file are Qwen's; this curated report states Qwen's
actual observed numbers (1.341 / 0.006)._

_Caveats: clean well-separated FSM = an upper-bound ceiling; a harder control
would lower it. Qwen's position leakage (0.78) shows its codes remain far more
position-bound than the control's (~chance 0.24). Single seed._

_Artifacts: `reports_qwen/positive_control.{json,md}`,
`reports_qwen/layer_sweep.{json,md,png}` (now calibrated)._

```


## File: `reports\qwen_vs_tinyllama.md`
```md
# V5.3 · TinyLlama vs Qwen2.5-1.5B — Comparison

Scale-matched replication (identical Countdown problems, 3003 train / 1507 val
transitions, K=32, last layer, seed 0). Pipeline unchanged between runs.

## Headline comparison (requested metrics)

| Metric | TinyLlama-1.1B | Qwen2.5-1.5B | Replicates? |
| --- | ---: | ---: | :---: |
| **MLP(z,a) top1** — E. MLP(z,op) | 0.558 | 0.464 | ✓ (Qwen weaker) |
| **Action-aware bigram** — C. (z,op) | 0.553 | 0.464 | ✓ (Qwen weaker) |
| **MLP(z,a) − action-bigram** (E − C) | **+0.005** | **+0.000** | ✓ MLP never beats lookup |
| **Entropy** H(z'\|z,op) [nats] | 0.993 | 1.341 | ✓ (Qwen *more* stochastic) |
| **Deterministic fraction** det(z,op) | 0.110 | **0.006** | ✓ both ≪ 0.20; Qwen lower |
| **Position leakage** (fixed-init) | 0.790 | 0.779 | ✓ near-identical, ≫ floor 0.25 |
| det-frac(z) action-blind | 0.000 | 0.000 | ✓ |
| state matters (E2 − F) | +0.217 | +0.232 | ✓ codes non-vacuous |
| action helps lookup (C − B) | +0.290 | +0.149 | ✓ same sign, smaller |

## Per-model A–F (held-out val top1)

| Model | TinyLlama | Qwen |
| --- | ---: | ---: |
| A. Majority | 0.116 | 0.106 |
| B. Bigram(z) | 0.263 | 0.316 |
| C. Action-bigram(z,op) | 0.553 | 0.464 |
| D. MLP(z) | 0.266 | 0.317 |
| E. MLP(z,op) | 0.558 | 0.464 |
| E2. MLP(z,op,operands) | 0.622 | 0.509 |
| F. MLP(op,operands) control | 0.405 | 0.277 |

## Reading

Every signature that defined the V5.2 conclusion reproduces on Qwen:

- **`MLP(z,a) ≈ action-aware bigram`** — the load-bearing finding — reproduces
  almost exactly (E−C: +0.005 → +0.000; scrubbed −0.005). No learned dynamics
  beyond a first-order action-conditioned count, in either model.
- **Deterministic fraction ≈ 0** — reproduces and is *stronger* on Qwen
  (0.110 → 0.006). No reusable low-entropy planning states emerge.
- **Position leakage** — reproduces (0.79 in both): discrete codes track
  trajectory stage, far above the majority/null floor, in both models.

Where the two models differ, **Qwen is the weaker planner-state candidate**, not
the stronger one: lower action-conditioned accuracy, higher conditional entropy,
near-zero determinism. A 36% larger, substantially stronger, more recent model
does **not** convert Countdown hidden states into crisper, reusable discrete
dynamics at the final layer.

## Experiment B — layer sweep (middle vs final)

Identical sweep config in both runs (`cap_train=1500`, full val, K=32, VQ 50 ep,
transition 30 ep, seed 0). TinyLlama swept layers 12/22 (of 22); Qwen 14/28 (of 28).

| | TinyLlama L12 (mid) | TinyLlama L22 (final) | Qwen L14 (mid) | Qwen L28 (final) |
| --- | ---: | ---: | ---: | ---: |
| MLP(z,op) top1 | 0.361 | 0.542 | 0.233 | 0.434 |
| E − C | −0.006 | −0.002 | +0.007 | −0.007 |
| H(z'\|z,op) nats | 1.69 | 1.14 | 2.12 | 1.50 |
| **det(z,op)** | **0.000** | **0.000** | **0.000** | **0.000** |
| position leakage | 0.681 | 0.778 | 0.604 | 0.799 |
| best layer / reaches ceiling | final (22) / **No** | | final (28) / **No** | |

**Same pattern in both models:** the middle layer is *weaker* than the final
(lower MLP, higher entropy); det(z,op) = 0 at every depth; MLP ≈ action-bigram
(E−C ≈ 0) everywhere; the best layer is the final one and it does **not** reach
the reusable-state regime. TinyLlama's conclusion *"middle layers were not
stronger than the final layer"* replicates on Qwen.

## Calibration — each model on its OWN floor/ceiling

Closing the "carried-over calibration" nit: a Qwen-matched synthetic-FSM control
(1536-d vs TinyLlama's 2048-d) gives Qwen its own noise floor and reusable-state
ceiling. Final-layer Countdown placement (% of the way from floor to ceiling):

| Axis | TinyLlama (2048-d anchors) | Qwen (1536-d anchors) |
| --- | ---: | ---: |
| Det-frac(z,a) | 19% | **1%** |
| MLP(z,a) acc | 47% | **49%** |
| H(z'\|z,a) | 68% | **68%** |
| control AMI (codes vs true states) | 0.935 | 0.815 |

Both models: **above the noise floor, well short of the FSM ceiling**, with the
same axis-disagreement (strict det-frac low, soft axes ~half). Qwen is
indistinguishable on the soft axes and *lower* on strict determinism — the larger
model is, if anything, slightly further from reusable states. The calibration
carries over; the conclusion does not change. See `qwen_positive_control.md`.

_All requested rows are now populated; Experiment A (final layer), Experiment B
(middle vs final), and the Qwen-matched calibration are complete._

```


## File: `reports\state_statistics.md`
```md
# Phase C Dataset Statistics

- **Total Trajectories loaded**: 20
- **Total Unique Symbolic States**: 0
- **Cross-History States** (>=2 different histories): 0
- **Average Cluster Size** (for cross-history states): 0.00
- **Maximum Cluster Size**: 0
- **Total Possible Positive Pairs**: 0

> [!WARNING]
> The dataset may be underpowered for contrastive learning. Consider generating more trajectories.
```


## File: `reports\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2860894203186035,2.8268252513447747

```


## File: `reports\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.298072338104248,2.820525622758709

```


## File: `reports\transition_train_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,25.77799967692105,2.0909771256976657
1,1,52.159096879758096,1.3752207623587713
2,2,76.55100516720175,1.3353729910320706
3,3,38.99565217422378,1.2027051316367254
4,4,42.79594469210024,1.0489825540118747
5,5,22.37420850950326,0.9735886918173896
6,6,17.695606075349403,0.9565543201234605
7,7,11.677112185703788,0.9592268533176846
8,8,31.976166325374844,0.848485012849172
9,9,50.37089825346543,1.15346446302202
10,10,26.453902213997807,1.5303655068079631
11,11,91.34916928403551,0.9819845888349745
12,12,49.60181234033661,1.0907020171483357
13,13,14.702802171193465,0.8387697140375773
14,14,15.896075391657737,0.7877360317442152
15,15,5.507012825101544,0.8507650097211202
16,16,5.338554467194532,0.7637359764840868
17,17,4.516644178425121,0.7835371825430129
18,18,3.7357912260419592,0.7646944721539816
19,19,5.68391793003127,0.7485060493151346
20,20,3.131982737179383,0.7500331666734483
21,21,1.936820789717958,0.7478373911645677
22,22,1.271382331150198,0.7653108768992953
23,23,2.0816928653583036,0.7525539464420743
24,24,1.5634799669442188,0.7504046758015951
25,25,1.5796608193417623,0.7636770009994507
26,26,2.3228244535816915,0.7448226544592116
27,27,2.1846786329841166,0.7511241833368937
28,28,1.5208069855212047,0.7451951106389364
29,29,1.7213012604021076,0.744917651017507
30,30,2.3578414380969153,0.7420761717690362
31,31,3.9460892890878805,0.7588564621077644
32,32,2.2257841659373923,0.7585197885831197
33,33,2.780805822557811,0.7556471162372165
34,34,4.951839517932865,0.7489362756411234
35,35,6.184529101262327,0.7599635521570841
36,36,2.2707069281671868,0.7827138238483005
37,37,2.7413602919433377,0.8381811777750651
38,38,8.08894054378782,1.009585354063246
39,39,24.197919396661764,1.123499141799079

```


## File: `reports\V5_1_FINAL.json`
```json
{
  "config": {
    "num_codes": 32,
    "vq_epochs": 50,
    "transition_epochs": 30,
    "inlp_iters": 100,
    "inlp_C": 1.0,
    "seed": 0,
    "hidden_dim": 2048,
    "n_train": 1000,
    "n_val": 500,
    "n_test": 1000
  },
  "columns": {
    "v5_original": {
      "active_codes": 6,
      "used_codes": 6,
      "dead_codes": 26,
      "gini": 0.8542733574818886,
      "perplexity": 5.33343103694012,
      "mlp_top1": 0.43132050431320507,
      "majority_baseline": 0.31320504313205044,
      "bigram_baseline": 0.43132050431320507,
      "predictive_entropy": 1.2936736311276542,
      "global_entropy": 1.1970436180211035,
      "det_frac": 0.0,
      "position_leakage": 0.5883090817862519,
      "leakage_majority": 0.2508780732563974,
      "leakage_null": 0.09006522829904666,
      "source": "re-derived from saved original collapsed code trajectories"
    },
    "v5_fixed_init": {
      "active_codes": 32,
      "used_codes": 32,
      "dead_codes": 0,
      "gini": 0.37094835123657255,
      "perplexity": 25.684334586708843,
      "mlp_top1": 0.2647644326476443,
      "majority_baseline": 0.11612475116124751,
      "bigram_baseline": 0.26542800265428,
      "predictive_entropy": 2.0382032691839127,
      "global_entropy": 1.771690074035009,
      "det_frac": 0.0,
      "position_leakage": 0.7900150526843954,
      "leakage_majority": 0.2508780732563974,
      "leakage_null": 0.1241846462619167,
      "source": "VQ retrained with data-dependent init on raw states"
    },
    "v5_1_scrubbed": {
      "active_codes": 32,
      "used_codes": 32,
      "dead_codes": 0,
      "gini": 0.48373875843117664,
      "perplexity": 21.2213147072519,
      "mlp_top1": 0.2083609820836098,
      "majority_baseline": 0.18911745189117452,
      "bigram_baseline": 0.2183145321831453,
      "predictive_entropy": 2.5760769404238553,
      "global_entropy": 2.416621449360173,
      "det_frac": 0.03125,
      "position_leakage": 0.5667335674862017,
      "leakage_majority": 0.2508780732563974,
      "leakage_null": 0.11189162067235324,
      "source": "VQ trained on INLP position-scrubbed states"
    }
  },
  "position_probe": {
    "position_accuracy": 0.8956347215253387,
    "majority_baseline": 0.33492222779729053,
    "n_classes": 4,
    "n_train": 4003,
    "n_test": 3986
  },
  "inlp": {
    "before_accuracy": 0.8956347215253387,
    "after_accuracy": 0.38961364776718516,
    "majority_baseline": 0.33492222779729053,
    "accuracy_trace": [
      0.8956347215253387,
      0.8848469643753136,
      0.8840943301555444,
      0.8891118916206724,
      0.8793276467636728,
      0.875062719518314,
      0.8672854992473658,
      0.8640240842950326,
      0.8582538886101355,
      0.8549924736578023,
      0.8524836929252383,
      0.8419468138484696,
      0.84470647265429,
      0.8416959357752133,
      0.8391871550426493,
      0.8294029101856498,
      0.8251379829402911,
      0.8146011038635224,
      0.8083291520321124,
      0.8048168590065228,
      0.8015554440541897,
      0.8000501756146513,
      0.7937782237832414,
      0.7930255895634721,
      0.7895132965378826,
      0.7807325639739087,
      0.7880080280983442,
      0.7827395885599598,
      0.775213246362268,
      0.7767185148018063,
      0.7762167586552935,
      0.7759658805820371,
      0.7684395383843452,
      0.7649272453587557,
      0.7571500250878074,
      0.7541394882087306,
      0.748118414450577,
      0.7498745609633718,
      0.7415955845459107,
      0.743100852985449,
      0.7390868038133467,
      0.7288008028098344,
      0.7227797290516809,
      0.7047165077772203,
      0.7034621174109383,
      0.6939287506271952,
      0.6781234320120422,
      0.6648268941294531,
      0.6562970396387355,
      0.6404917210235825,
      0.624937280481686,
      0.6191670847967887,
      0.6151530356246864,
      0.6126442548921224,
      0.594079277471149,
      0.5915704967385851,
      0.5933266432513798,
      0.5840441545408931,
      0.5787757150025088,
      0.5790265930757652,
      0.5762669342699448,
      0.5725037631710989,
      0.5797792272955343,
      0.5772704465629704,
      0.5707476166583041,
      0.5740090316106372,
      0.5614651279478173,
      0.5622177621675866,
      0.5559458103361766,
      0.551179126944305,
      0.5569493226292022,
      0.5524335173105871,
      0.5476668339187155,
      0.5403913697942799,
      0.5331159056698445,
      0.5168088309081786,
      0.5097842448569995,
      0.5040140491721024,
      0.49247365780230806,
      0.4839438033115906,
      0.46914199698946313,
      0.46362267937782237,
      0.4518314099347717,
      0.4417962870045158,
      0.4340190667335675,
      0.4239839438033116,
      0.4212242849974912,
      0.4136979427997993,
      0.41269443050677374,
      0.4099347717009533,
      0.4074259909683894,
      0.4026593075765178,
      0.39663823381836427,
      0.4019066733567486,
      0.3981435022579027,
      0.3941294530858003,
      0.3953838434520823,
      0.3971399899648771,
      0.39739086803813345,
      0.3958855995985951
    ],
    "dims_removed": 300,
    "num_iters": 100
  },
  "criteria": {
    "codes_high": true,
    "leakage_dropped": false,
    "mlp_beats_bigram": false,
    "mlp_bigram_margin": -0.009953550099535496,
    "entropy_decreased": false,
    "deterministic_emerged": false
  }
}
```


## File: `reports\V5_1_FINAL.md`
```md
# V5.1 — Final Report (Discrete State Discovery)

> **Research question:** do hidden-state trajectories of a frozen language model admit a compact, predictive, *reusable* discrete state representation?

> Model: TinyLlama · Task: Countdown · States: last-layer hidden states at reasoning-step boundaries · Codebook K=32.


---

## Why the original V5 result is invalid

The original VQ codebook was initialized as `randn * 0.02` (row-norm ≈ 0.90) while the hidden states have norm ≈ 85 — a ~94× scale mismatch. On the first batch almost every state maps to one code; EMA + dead-code revival cannot recover, leaving **6/32 active codes**. The collapse is an initialization artifact, not a property of the representation. Data-dependent init (random training states) restores full codebook usage.

## Comparison table

| Metric | V5 Original | V5 Fixed Init | V5.1 Scrubbed |
| --- | ---: | ---: | ---: |
| Active Codes | 6 | 32 | 32 |
| Gini | 0.854 | 0.371 | 0.484 |
| Perplexity | 5.33 | 25.68 | 21.22 |
| Position Leakage | 0.588 | 0.790 | 0.567 |
| Majority Baseline | 0.313 | 0.116 | 0.189 |
| Bigram Baseline | 0.431 | 0.265 | 0.218 |
| MLP Accuracy | 0.431 | 0.265 | 0.208 |
| Global Entropy | 1.197 | 1.772 | 2.417 |
| Deterministic Fraction | 0.000 | 0.000 | 0.031 |

_V5 Original source: re-derived from saved original collapsed code trajectories._ Leakage majority/null floors: Original 0.251/0.090, Fixed 0.251/0.124, Scrubbed 0.251/0.112.


**Central pattern — MLP ≈ bigram in every column** (Original 0.431/0.431, Fixed 0.265/0.265, Scrubbed 0.208/0.218 as MLP/bigram): the learned next-code MLP never improves on a first-order count model, in any condition. The original report's headline (MLP 0.431 vs uniform-chance 0.031, a ~14× gap) compared against the wrong baseline; against the honest bigram the gap is ≈ 0.

**Entropy caveat:** global entropy *rises* across columns (1.197 → 1.772 → 2.417), but this is a de-collapse artifact — with 6 codes there are few possible successors, with 32 there are many. The decisive diagnostic is the deterministic-successor fraction, which stays ≈ 0 (0.000 / 0.000 / 0.031): no near-deterministic 'planning' states appear in any condition.

## Position subspace (continuous hidden states)

- Position probe (4 relative-depth bins): **0.896** held-out vs majority 0.335 (4 classes). Trajectory stage is almost perfectly linearly decodable from a raw hidden state.

- INLP removed **300 dims** over 100 iterations: position decodability **0.896 → 0.390** (majority floor 0.335).


**INLP curve — position is high-dimensional and redundant:**

| dims removed (≈3·iter) | position accuracy |
| ---: | ---: |
| 0 | 0.896 |
| 57 | 0.805 |
| 117 | 0.743 |
| 177 | 0.579 |
| 237 | 0.484 |
| 297 | 0.396 |

_The spec's 6–8 iterations remove only ~24 dims (acc still 0.864); reaching the floor needs ~300 dims. That position resists removal until ~15% of the hidden space is deleted is itself evidence that these states are dominated by trajectory stage._


---

## Final analysis

### Q1 — Does fixing initialization invalidate the original collapse conclusion?

**Yes.** With data-dependent init the codebook goes from 6/32 active codes (Gini 0.854) to 32/32 (Gini 0.371). The reported collapse was an init artifact and must not be cited as evidence about the representation.

### Q2 — Does position scrubbing reveal stronger latent structure?

**No.** Even after near-complete removal of linearly-decodable position (continuous-state decodability 0.896 → 0.390, floor 0.335; 300 dims), the codes still leak position at 0.567 (floor 0.251), and the MLP (0.208) does not beat the bigram (0.218) — margin -0.010. No reusable low-entropy dynamics appear; the deterministic-successor fraction stays 0.031.

### Q3 — After controlling for init and position, the states are:

**B + C — trajectory-stage encodings with only weak local (first-order) dynamics.** Two signatures coincide: the codes still track trajectory position (leakage 0.567 ≫ floor 0.251) even after the continuous states are scrubbed, and the only predictability that exists is fully captured by a bigram (MLP ≈ bigram). Neither reusable planning states (A) nor a deterministic-successor subset are observed.

### Q4 — Strongest defensible conclusion

> Under TinyLlama + Countdown + last-layer hidden states, discrete latent structure is dominated by trajectory progression and weak local dynamics rather than reusable planning states.


The codebook collapse was an initialization bug, and once it is fixed the apparent transition 'predictability' is fully explained by a first-order bigram model (MLP ≈ bigram). Removing the linearly-decodable position subspace does not expose any additional reusable, low-entropy dynamics.


### Decision criteria (scrubbed run)

- codes_high: **PASS**
- leakage_dropped: **fail**
- mlp_beats_bigram: **fail**
- entropy_decreased: **fail**
- deterministic_emerged: **fail**
- MLP − bigram margin: -0.010


### Threats to validity / scope

- **Single configuration:** one model (TinyLlama), one task (Countdown), last-layer states only, seed 0. The conclusion is scoped to this setting; other layers / models / tasks are untested.

- **Linear position removal:** INLP removes only *linearly* decodable position. Residual (nonlinear) position survives — codes still leak it at 0.567 — so the scrub is a lower bound on position's influence, not a complete excision.

- **Position operationalized** as 4 relative-depth quartiles (trajectories are only 3–5 states); absolute-index and finer binnings were not swept.

- **Over-scrub control:** 300/2048 dims are removed, which could in principle delete content — but the negative result does not depend on it: MLP ≈ bigram already holds in the *un-scrubbed* Fixed-Init run, and held at every intermediate scrub depth tested (24/60/120/180/240/300 dims).

- **What would overturn this:** an MLP that clears the bigram by a non-trivial margin, a deterministic-successor subset (entropy < 0.5 nats) of non-trivial mass, or code→position leakage falling to its floor after scrubbing. None occurred.


![comparison](V5_1_comparison.png)


_Artifacts: `V5_1_FINAL.json`, `V5_1_comparison.png`, `checkpoints/vq_state.pt` (fixed init), `checkpoints/vq_state_scrubbed.pt`, `reports/scrubbed/*_scrubbed.pt`._

```


## File: `reports\V5_3_VERDICT.md`
```md
# V5.3 — Cross-Model Replication Verdict (Qwen2.5-1.5B)

**Status: Experiments A, B, and the Qwen positive-control calibration complete.**

## What was asked

Does the V5.2 conclusion replicate on a stronger, more modern model, or is it a
TinyLlama artifact? The V5.2 findings were:

1. Action-conditioned structure exists.
2. Structure is above the calibrated noise floor.
3. Structure is weaker than a known finite-state machine.
4. MLP(z,a) ≈ action-aware bigram.
5. No reusable planning states emerged.
6. Middle layers were not stronger than the final layer.

## Verdict → **Outcome 1: the negative result replicates.**

Both experiments used the unchanged V5.2 pipeline with only the frozen teacher
swapped (`TinyLlama-1.1B` → `Qwen2.5-1.5B`), scale-matched to the identical
Countdown problems (Exp A: 3003/1507 transitions; Exp B: cap_train=1500, full val).

| V5.2 finding | Qwen replication | Evidence |
| --- | :---: | --- |
| 1. Action-conditioned structure exists | ✓ | C−B +0.149, E−D +0.148 (action raises predictability) |
| 2. Above the noise floor | ✓ (calibrated) | Qwen 1536-d anchors: MLP 49%, H 68% of floor→ceiling; all metrics beat the floor |
| 3. Weaker than a known FSM | ✓ (calibrated) | short of the FSM ceiling on every axis; det 0.006 vs ceiling 0.586 (1% up) |
| **4. MLP(z,a) ≈ action-aware bigram** | **✓** | **E−C = +0.000 (final), −0.007 / +0.007 across layers** |
| **5. No reusable planning states** | **✓ (stronger)** | **det(z,op) = 0.006 (Exp A), 0.000 at both swept layers** |
| **6. No middle-layer advantage** | **✓** | **middle weaker than final in both models; best = final, no ceiling** |

**All six replicate.** #2 and #3 are now calibrated against a **Qwen-matched
(1536-d) noise floor and FSM ceiling**, not carried over from TinyLlama: Qwen's
Countdown sits 1% (det-frac) / 49% (MLP) / 68% (entropy) of the way from floor to
ceiling — above noise, well short of reusable states — essentially identical to
TinyLlama on its own anchors (19% / 47% / 68%), and *lower* on the strict
determinism axis. The control itself recovers known FSM states (AMI 0.815),
validating the pipeline at Qwen's dimensionality. **No finding flips.**

### Why this is Outcome 1, not Outcome 2 or 3

- **Outcome 3 (hypothesis survives)** is rejected: no deterministic subset
  (det ≈ 0), MLP never beats the action bigram (E−C ≈ 0), no middle-layer peak.
- **Outcome 2 (scaling effect, still sub-deterministic)** is *not* supported:
  Qwen does not exceed TinyLlama. On every axis where the two differ, the larger
  model is the **weaker** planning-state candidate — lower action-conditioned
  accuracy (E 0.464 vs 0.558), higher conditional entropy (1.34 vs 0.99 nats),
  near-zero determinism (0.006 vs 0.110).
- **Outcome 1** is satisfied on its own terms: `MLP(z,a) ≈ action-aware bigram`,
  `deterministic fraction ≈ 0`, and `no middle-layer advantage` — all three.

## Bottom line for the paper

**The V5.2 negative is not a small-model artifact.** A 36%-larger, materially
stronger, more recent language model (Qwen2.5-1.5B) produces *no more* — and on
the determinism axis *less* — reusable discrete planning structure in its
Countdown hidden states than TinyLlama-1.1B. Under both models the learned
next-code dynamics collapse to a first-order action-conditioned count, no
deterministic reusable subset appears, and no probed depth changes this. This
cross-model replication **substantially strengthens** the manuscript's claim.

## Scope / honest limits

- Two models, one task (Countdown), one seed (0), op-type action (DIV absent).
- Two layers in Exp B (middle 14, final 28), not the full stack.
- Calibration is anchored on a clean, well-separated synthetic FSM (an *upper*
  bound on the ceiling); a harder control would lower it. The Qwen control uses
  hidden_dim 1536 / noise 0.3 / 32 states / 3 actions to match the Countdown
  setting; other control geometries were not swept.
- The retained full 5000-problem Qwen extraction
  (`reports_qwen/trajectories/train_full.pt`) allows a data-scale sensitivity
  check if desired; Experiment A was scale-matched to TinyLlama deliberately.

## Deliverables

- `reports/qwen_action_conditioned.md` — ✅ Experiment A.
- `reports/qwen_layer_validation.md` — ✅ Experiment B (calibrated).
- `reports/qwen_positive_control.md` — ✅ Qwen-matched floor/ceiling calibration.
- `reports/qwen_vs_tinyllama.md` — ✅ full comparison (A + B + calibration).
- `reports/V5_3_VERDICT.md` — ✅ this file.

```


## File: `reports\v5_codebook_usage.csv`
```csv
active_codes,used_codes,dead_codes,collapse_score_gini,perplexity,entropy_nats,mean_freq_when_used
6,6,26,0.8543,5.33,1.674,0.16667

```


## File: `reports\v5_discrete_rollout.csv`
```csv
depth,code_match_accuracy,cosine_similarity,mse,operator_accuracy,teacher_operator_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,n_samples
1,0.353,0.9728982648849487,0.1546208958774805,0.328,0.337,0.349,0.349,0.3825,0.434,1000
2,0.304,0.9465346341133117,0.3146874246895313,0.3533026113671275,0.3563748079877112,0.316,0.316,0.33225,0.32825,1000
3,0.48540706605222733,0.9608634781727593,0.23030224932320473,0.31044776119402984,0.3164179104477612,0.5145929339477726,0.5145929339477726,0.35560675883256526,0.3049155145929339,651
4,0.9552238805970149,0.9963125463741929,0.02163870067738775,,,0.0,0.0,0.38283582089552237,0.3753731343283582,335
5,,,,,,,,,,0
6,,,,,,,,,,0
7,,,,,,,,,,0
8,,,,,,,,,,0

```


## File: `reports\v5_discrete_state_report.md`
```md
# V5 Report — Reusable Discrete States?

Research question: _do hidden-state trajectories of frozen language models admit a compact, predictive, reusable discrete state representation?_

- **Codebook size (K):** 32
- **Hidden dim:** 2048
- **Trajectories:** train=1000 test=1000

---

## 1. Codebook usage (collapse check)

- Active codes: **6/32** (used: 6, dead: 26)
- Collapse score (Gini): **0.854** (0 = uniform, 1 = fully collapsed)
- Perplexity: **5.3** (effective codes used, max 32)

Gate: PASS (Gini<0.9 AND active>=10% of K).

## 2. Transition predictability

- Top-1 next-code accuracy: **0.431** (chance = 0.031)
- Predictive entropy: **1.294 nats** (perplexity 3.61)

Gate: PASS (top-1 > max(2*chance, chance+0.02)).

## 3. Transition entropy (planning state vs bucket)

- Global H(Z_next | Z_current): **1.197 nats** (perplexity 3.31)
- Fraction of states with near-deterministic successor (H<0.5): **0.0%**

Gate: FAIL (>= 20% deterministic states).

## 4. Position leakage (codes as step indices?)

- Position predictability: **0.588**
  - chance: 0.251, permutation null: 0.090

Gate: FAIL (score <= max(null, chance) + 0.10). High leakage => codes are step indices, not reusable states.

## 5. Discrete rollout coherence

| depth | code match | cosine | op acc (Probe C) | state acc (Probe A) |
|---|---|---|---|---|
| 1 | 0.353 | 0.973 | 0.328 | 0.383 |
| 2 | 0.304 | 0.947 | 0.353 | 0.332 |
| 3 | 0.485 | 0.961 | 0.310 | 0.356 |
| 4 | 0.955 | 0.996 | nan | 0.383 |

Gate: PASS (depth-1 code match > max(2/K, 0.20)).


---

## Verdict

**Mixed evidence (3/5 gates pass).** See the failing gates above (low_entropy_subset, low_position_leakage) for the specific failure mode.

_Failure modes: high Gini => VQ collapse; low top-1 => unpredictable dynamics; high position leakage => codes are step indices; low deterministic-state fraction => every state is a compression bucket._

```


## File: `reports\v5_position_leakage.csv`
```csv
position_predictability_score,chance_accuracy,permutation_null_accuracy,n_train,n_test
0.5883090817862519,0.2508780732563974,0.09006522829904666,4003,3986

```


## File: `reports\v5_transition_entropy.csv`
```csv
code,entropy_nats,top1_successor_prob,top1_successor_code
12,1.3038152623588608,0.442225392296719,28
15,1.2423939890507498,0.5,28
19,1.2999716280996783,0.365,12
24,1.3062976311878975,0.4431818181818182,28
25,1.41553405325212,0.3669724770642202,24
28,0.8645415915325201,0.6551724137931034,28

```


## File: `reports\v5_transition_predictability.csv`
```csv
loss,top1_accuracy,predictive_entropy_nats,predictive_perplexity,chance_accuracy
1.282549963775817,0.43132050431320507,1.2936736311276542,3.6058227297722847,0.03125

```


## File: `reports\linear\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7469808042049408,0.8931060165166855,0.45,0.15,0.5,0.5625,0.2,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.09999999999999998,-0.14999999999999997,0.25,20,7.291595290531156
2,0.8330953538417816,0.8721769124269485,0.5882352941176471,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.739013329214001
3,1.2065738281782936,0.8291785576764275,0.0,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.7018033530573105
4,2.098354478677114,0.7744909127553304,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,2.7331121504108005
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8462274238467217,0.8769205898046494,0.5,0.15,0.4875,0.5625,0.2,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.11249999999999999,-0.14999999999999997,0.25,20,6.436427797741151
2,0.8855497181415558,0.8641697973012924,0.5294117647058824,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.339836803097011
3,1.2845268705311943,0.8214430107789881,0.5,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.416468819133299
4,2.190103073914846,0.7719624042510986,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,2.618615621268393
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8945804014801979,0.8728564590215683,0.25,0.45,0.475,0.475,0.0,0.0,0.05,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.15000000000000002,20,6.224621010452953
2,1.2432714581489563,0.8110212802886962,0.4,0.45,0.4,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.050000000000000044,-0.05,-0.2,20,4.696642193315652
3,1.6621245294809341,0.7668489068746567,0.35,0.3,0.45,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.21250000000000002,-0.9,0.19999999999999996,20,3.4297450777159706
4,3.300358164310455,0.6349915832281112,0.25,0.55,0.3375,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.16250000000000003,1.0,0.19999999999999996,20,1.7583518762222887
5,3.7012776494026185,0.7420937120914459,,,0.2625,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.15000000000000002,0.0,0.19999999999999996,20,1.5547565700398678
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\linear\metadata.txt`
```text
Base model:
DeepSeek-R1-Distill-Qwen-7B

Dataset:
Countdown

Transition:
linear

Trajectory source:
shared_teacher_hidden_states

Seed:
42

```


## File: `reports\linear\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\linear\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.095238,63,0.022171,0.168306
oracle1,probeA_per_label,all,0.503968,63,,
oracle1,probeA_jaccard,all,0.379630,63,,

```


## File: `reports\linear\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.095238 |
| probeA_exact_ci_lo | 0.022171 |
| probeA_exact_ci_hi | 0.168306 |
| probeA_per_label | 0.503968 |
| probeA_jaccard | 0.379630 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\linear\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.893 | 0.450 | 0.150 | 0.500 | 0.562 | 0.7470 |
| 2 | 0.872 | 0.588 | 0.588 | 0.438 | 0.512 | 0.8331 |
| 3 | 0.829 | 0.000 | 0.667 | 0.500 | 0.471 | 1.2066 |
| 4 | 0.774 | nan | nan | 0.167 | 0.417 | 2.0984 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **2**; rollout state probe accuracy stays >= 0.50 through depth **3**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.3474.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\linear\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.877 | +0.107 |
| 2 | 1.000 | 0.872 | 0.864 | +0.128 |
| 3 | 1.000 | 0.829 | 0.821 | +0.171 |
| 4 | 1.000 | 0.774 | 0.772 | +0.226 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.500 | 0.487 | +0.062 |
| 2 | 0.512 | 0.438 | 0.463 | +0.075 |
| 3 | 0.471 | 0.500 | 0.500 | -0.029 |
| 4 | 0.417 | 0.167 | 0.292 | +0.250 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.450 | 0.500 | -0.300 |
| 2 | 0.588 | 0.588 | 0.529 | +0.000 |
| 3 | 0.667 | 0.000 | 0.500 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.132 | +0.167 | +0.143 |
| 2 | +0.059 | -0.500 | +1.000 |
| 3 | +0.043 | -0.000 | -3.000 |
| 4 | +0.011 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.877 | 0.287 | 0.562 | 0.500 | 0.487 | 0.600 |
| 2 | 1.000 | 0.872 | 0.864 | 0.246 | 0.512 | 0.438 | 0.463 | 0.487 |
| 3 | 1.000 | 0.829 | 0.821 | 0.241 | 0.471 | 0.500 | 0.500 | 0.456 |
| 4 | 1.000 | 0.774 | 0.772 | 0.237 | 0.417 | 0.167 | 0.292 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1578
**Average Oracle Gain (state probe):** +0.0895
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\linear\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\linear\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\linear\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.5337090492248535,3.120876124647797
1,1,3.105609178543091,2.438449546939037
2,2,2.2873589992523193,2.0504565629802767
3,3,1.8091875314712524,1.8243828445184427
4,4,1.5261942148208618,1.6986654312884222
5,5,1.3677406311035156,1.6456582741659196
6,6,1.2927922010421753,1.6062354416143698
7,7,1.2283397912979126,1.5605386202452614
8,8,1.1590793132781982,1.5171999071465163
9,9,1.0982705354690552,1.4733917986760374
10,10,1.0388047695159912,1.440034709992956
11,11,0.9890342354774475,1.4147667806656634
12,12,0.9467793703079224,1.3970411957287399
13,13,0.9085137844085693,1.3847392347992444
14,14,0.8730375170707703,1.3727163095943262
15,15,0.8403750658035278,1.3589687660092213
16,16,0.8084198236465454,1.3483534015592982
17,17,0.7788730263710022,1.3412898329437757
18,18,0.7527260780334473,1.3339534822057506
19,19,0.7259371876716614,1.3280789734887295
20,20,0.7001040577888489,1.319274402055584
21,21,0.6746063232421875,1.308666667000192
22,22,0.6506435871124268,1.3010873012855404
23,23,0.628309428691864,1.2950304375320185
24,24,0.6072245836257935,1.2915489321849385
25,25,0.5884162187576294,1.2867694291912142
26,26,0.57072514295578,1.2806913031906377
27,27,0.5543239712715149,1.2780988099145107
28,28,0.5382649898529053,1.2759784635950306
29,29,0.5227152705192566,1.274091751849065
30,30,0.5074270367622375,1.2697301145459785
31,31,0.49222156405448914,1.2678140108702614
32,32,0.47795039415359497,1.265672152159644
33,33,0.46405771374702454,1.2650414138543802
34,34,0.45120444893836975,1.2618022981237194
35,35,0.4387236535549164,1.2620304295274078
36,36,0.427532434463501,1.261036231869557
37,37,0.41924577951431274,1.2751541137695312
38,38,0.418537437915802,1.2865217865490524
39,39,0.4293277859687805,1.305908953557249
40,40,0.4304525852203369,1.2692130667264345
41,41,0.39079463481903076,1.2566584602731172
42,42,0.36269623041152954,1.2837606961609886
43,43,0.3778074085712433,1.2706639024077868
44,44,0.3655227720737457,1.2561367847880378
45,45,0.3343905508518219,1.2748315529745133
46,46,0.34210339188575745,1.2664892478067367
47,47,0.3340895175933838,1.255400860895876
48,48,0.3102967441082001,1.272385268914895
49,49,0.31640735268592834,1.2642439545178024
50,50,0.30556246638298035,1.2585784411821208
51,51,0.2900031507015228,1.2736421178598873
52,52,0.2943190038204193,1.2632715193951716
53,53,0.28075408935546875,1.2628711637903431
54,54,0.2727867364883423,1.2758502647524974
55,55,0.27341026067733765,1.2661575567526895
56,56,0.2601929306983948,1.2680799140304815
57,57,0.2575795650482178,1.2774150410636527
58,58,0.25408118963241577,1.2714070804783555
59,59,0.24385470151901245,1.2727508544921875
60,60,0.243441641330719,1.2784492617747822
61,61,0.23700641095638275,1.2774162917840677
62,62,0.23064108192920685,1.2778255275038422
63,63,0.2297419309616089,1.2824012881419697
64,64,0.22265173494815826,1.2845924252369365
65,65,0.21916523575782776,1.2839475537909837
66,66,0.21684272587299347,1.287900830878586
67,67,0.2107408195734024,1.291657995005123
68,68,0.20865239202976227,1.2901421218621927
69,69,0.20538760721683502,1.2921287661693135
70,70,0.200570210814476,1.298172122142354
71,71,0.19883841276168823,1.2950216824891136
72,72,0.19531415402889252,1.2981527359759222
73,73,0.19155187904834747,1.3046644867443649
74,74,0.18986167013645172,1.3012585249103483
75,75,0.18651904165744781,1.3057324769066982
76,76,0.18346160650253296,1.3099100081647028
77,77,0.18176838755607605,1.309828836409772
78,78,0.1788662225008011,1.3097604220030739
79,79,0.17637832462787628,1.3211632400262552
80,80,0.1751931756734848,1.3114383885117828
81,81,0.17412813007831573,1.3319614598008453
82,82,0.17534834146499634,1.3233817678983095
83,83,0.18163616955280304,1.3596243936507428
84,84,0.18994247913360596,1.3302742379610655
85,85,0.1878725290298462,1.3456055688076332
86,86,0.16994352638721466,1.326966832895748
87,87,0.15963661670684814,1.3274457337426357
88,88,0.1675586700439453,1.35516357421875
89,89,0.1684568226337433,1.327497263423732
90,90,0.15567326545715332,1.3302684846471569
91,91,0.1547551155090332,1.3536004238441341
92,92,0.15961652994155884,1.3326628638095543
93,93,0.15144190192222595,1.335753644099001
94,94,0.1480177938938141,1.3534199448882556
95,95,0.15185034275054932,1.3388286653112194
96,96,0.14604534208774567,1.3413746317879098
97,97,0.1429811716079712,1.3560694710153047
98,98,0.1454998403787613,1.3462626973136527
99,99,0.14074623584747314,1.347444127817623

```


## File: `reports\linear\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.5114023685455322,3.1401907498719264
1,1,3.1191658973693848,2.4654075747630637
2,2,2.3108527660369873,2.07811161729156
3,3,1.8311959505081177,1.8420682813300462
4,4,1.5342940092086792,1.7076973836930072
5,5,1.3660695552825928,1.6534725251745006
6,6,1.2926548719406128,1.6148834228515625
7,7,1.23108971118927,1.567549783675397
8,8,1.15988028049469,1.525174500512295
9,9,1.0993657112121582,1.483245099177126
10,10,1.0424686670303345,1.4513444744172643
11,11,0.9957171678543091,1.4306598100505892
12,12,0.9589639902114868,1.4145600365810707
13,13,0.9222220182418823,1.4022697073514345
14,14,0.8864278197288513,1.3889312744140625
15,15,0.851875364780426,1.3752841636782787
16,16,0.8198323249816895,1.3648869248687243
17,17,0.7911608815193176,1.3582008236744365
18,18,0.7658111453056335,1.3523479524205944
19,19,0.7395648956298828,1.3498735271516393
20,20,0.7152162194252014,1.3453786881243597
21,21,0.6906940937042236,1.3416678006531761
22,22,0.6680395007133484,1.3393454629866803
23,23,0.6465815901756287,1.3382981097111937
24,24,0.6267397403717041,1.3386177938492572
25,25,0.6089057922363281,1.3348953997502562
26,26,0.5916983485221863,1.3321617001392803
27,27,0.5756795406341553,1.3311512431160348
28,28,0.5600945353507996,1.3328039450723617
29,29,0.545407235622406,1.332289273621606
30,30,0.5309279561042786,1.3302678592869492
31,31,0.5167996287345886,1.3293528322313652
32,32,0.5034576654434204,1.3296173595991292
33,33,0.49027517437934875,1.3294405077324538
34,34,0.477859765291214,1.3278000628361937
35,35,0.46544119715690613,1.3268182473104508
36,36,0.4536914527416229,1.3283143590708248
37,37,0.4421786069869995,1.3294391319399974
38,38,0.43137747049331665,1.3329734176885886
39,39,0.421815425157547,1.3334371848184554
40,40,0.41723814606666565,1.3703699580958633
41,41,0.4336046278476715,1.4155014538374104
42,42,0.49531546235084534,1.442702871854188
43,43,0.484269380569458,1.3364229045930456
44,44,0.37661832571029663,1.3840402071593239
45,45,0.4277089238166809,1.384854676293545
46,46,0.3983716368675232,1.3570494104604252
47,47,0.3626127243041992,1.3789765404873207
48,48,0.39378297328948975,1.3452913878393955
49,49,0.3377125859260559,1.392240180343878
50,50,0.3708086609840393,1.3478881335649333
51,51,0.3270072340965271,1.3674889236200052
52,52,0.34647905826568604,1.3596492829870006
53,53,0.31724467873573303,1.3786158327196465
54,54,0.3266746997833252,1.3549947269627305
55,55,0.3069072365760803,1.3634288350089652
56,56,0.3106902241706848,1.3666421858990778
57,57,0.29581281542778015,1.3747629884813652
58,58,0.2976336181163788,1.35967767434042
59,59,0.28513240814208984,1.3670484198898565
60,60,0.2867741584777832,1.3689375080046107
61,61,0.2749290466308594,1.3761088887199027
62,62,0.27696728706359863,1.3652433801869877
63,63,0.2658837139606476,1.3712814831342854
64,64,0.2678905725479126,1.3722773067286758
65,65,0.25753480195999146,1.3798107710040983
66,66,0.2595115602016449,1.3708048335841445
67,67,0.25011181831359863,1.3760130835361168
68,68,0.2514358162879944,1.3779344402375768
69,69,0.2435046285390854,1.3823117115458503
70,70,0.24391824007034302,1.3781703261078382
71,71,0.2373644858598709,1.3823069588082735
72,72,0.23690035939216614,1.383482886142418
73,73,0.23173674941062927,1.3868938508580944
74,74,0.23040258884429932,1.3865876745005123
75,75,0.22639015316963196,1.389694338939229
76,76,0.22436535358428955,1.3914031982421875
77,77,0.22136729955673218,1.3928657906954405
78,78,0.2188684493303299,1.397550989369877
79,79,0.21650327742099762,1.3982920412157402
80,80,0.21375246345996857,1.399137653288294
81,81,0.2118167281150818,1.4021131171554815
82,82,0.20897801220417023,1.4070469590484118
83,83,0.20734480023384094,1.4070787273469518
84,84,0.20454125106334686,1.4076673163742315
85,85,0.20305421948432922,1.4128175328989498
86,86,0.200412318110466,1.4161309414222591
87,87,0.1989935040473938,1.417083239946209
88,88,0.1966724842786789,1.4167678082575563
89,89,0.1956399530172348,1.4287801023389473
90,90,0.1948176622390747,1.4255293549084274
91,91,0.19816821813583374,1.4560787013319672
92,92,0.20994392037391663,1.455139660444416
93,93,0.2333715260028839,1.4963241327004355
94,94,0.2365565001964569,1.4358665591380635
95,95,0.1989280730485916,1.4346978859823258
96,96,0.18686705827713013,1.4751944620101178
97,97,0.20995980501174927,1.4406618212090163
98,98,0.19494374096393585,1.4401047503361937
99,99,0.18096263706684113,1.4693223296618851

```


## File: `reports\linear_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.75872553139925,0.8916219413280487,0.45,0.15,0.525,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.19999999999999996,20,7.178724701689017
2,0.8277764320373535,0.8731833040714264,0.47058823529411764,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.782315220340409
3,1.2058072090148926,0.8284963299246395,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.704792630717992
4,2.12328694264094,0.7684605121612549,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,2.701018880852774
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8363207250833511,0.8786536991596222,0.6,0.15,0.5125,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.08750000000000002,-0.09999999999999998,0.25,20,6.512670977411308
2,0.8319128721952438,0.8723057359457016,0.6470588235294118,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.748592168349572
3,1.1613570977659786,0.8332820219152114,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.88486519947464
4,1.9572845101356506,0.7839512725671133,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,2.930099375866348
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9052082613110543,0.8721218466758728,0.25,0.45,0.5,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.07500000000000001,-0.05,-0.2,20,6.151539044206332
2,1.2418083608150483,0.8122651308774949,0.45,0.45,0.4375,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.08750000000000002,-0.05,-0.2,20,4.7021757723188164
3,1.5967508792877196,0.7742615073919297,0.4,0.3,0.475,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.2375,-0.9,0.19999999999999996,20,3.57016457450217
4,3.167185032367706,0.6390328884124756,0.35,0.55,0.3625,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.1875,1.0,0.19999999999999996,20,1.8322866871098218
5,3.5649133563041686,0.7311458975076676,,,0.2875,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.175,0.0,0.19999999999999996,20,1.6142287814019571
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\linear_seed42\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\linear_seed42\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.095238,63,0.022171,0.168306
oracle1,probeA_per_label,all,0.492063,63,,
oracle1,probeA_jaccard,all,0.387566,63,,

```


## File: `reports\linear_seed42\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.095238 |
| probeA_exact_ci_lo | 0.022171 |
| probeA_exact_ci_hi | 0.168306 |
| probeA_per_label | 0.492063 |
| probeA_jaccard | 0.387566 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\linear_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.450 | 0.150 | 0.525 | 0.562 | 0.7587 |
| 2 | 0.873 | 0.471 | 0.588 | 0.438 | 0.512 | 0.8278 |
| 3 | 0.828 | 0.167 | 0.667 | 0.471 | 0.471 | 1.2058 |
| 4 | 0.768 | nan | nan | 0.167 | 0.417 | 2.1233 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.3586.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\linear_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.879 | +0.108 |
| 2 | 1.000 | 0.873 | 0.872 | +0.127 |
| 3 | 1.000 | 0.828 | 0.833 | +0.172 |
| 4 | 1.000 | 0.768 | 0.784 | +0.232 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.525 | 0.512 | +0.037 |
| 2 | 0.512 | 0.438 | 0.512 | +0.075 |
| 3 | 0.471 | 0.471 | 0.500 | +0.000 |
| 4 | 0.417 | 0.167 | 0.292 | +0.250 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.450 | 0.600 | -0.300 |
| 2 | 0.588 | 0.471 | 0.647 | +0.118 |
| 3 | 0.667 | 0.167 | 0.333 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.107 | +0.250 | +0.333 |
| 2 | +0.007 | n/a | +3.000 |
| 3 | -0.029 | +1.000 | -0.500 |
| 4 | -0.072 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.879 | 0.287 | 0.562 | 0.525 | 0.512 | 0.600 |
| 2 | 1.000 | 0.873 | 0.872 | 0.246 | 0.512 | 0.438 | 0.512 | 0.487 |
| 3 | 1.000 | 0.828 | 0.833 | 0.241 | 0.471 | 0.471 | 0.500 | 0.456 |
| 4 | 1.000 | 0.768 | 0.784 | 0.237 | 0.417 | 0.167 | 0.292 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1596
**Average Oracle Gain (state probe):** +0.0906
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\linear_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\linear_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\linear_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.5232813358306885,3.0784426829853997
1,1,3.061244010925293,2.3878604075947747
2,2,2.2403244972229004,2.029964259413422
3,3,1.793331503868103,1.8156504396532402
4,4,1.5140321254730225,1.7165387263063525
5,5,1.3788528442382812,1.6749196287061348
6,6,1.3168097734451294,1.6206437407946976
7,7,1.2374018430709839,1.567597936411373
8,8,1.159865379333496,1.5260500048027663
9,9,1.0993807315826416,1.4842356697457735
10,10,1.0412166118621826,1.4537546126568903
11,11,0.9957540035247803,1.4266500004002305
12,12,0.9542944431304932,1.4085473232581966
13,13,0.918175220489502,1.3954060038582223
14,14,0.8842597007751465,1.381606430303855
15,15,0.8511654138565063,1.367311071176998
16,16,0.8182157874107361,1.3544940635806224
17,17,0.7854737639427185,1.3438072829950052
18,18,0.7560672760009766,1.3318641537525615
19,19,0.7265509366989136,1.3233202324538935
20,20,0.6993833184242249,1.3155884039206582
21,21,0.6730312705039978,1.3073567875096055
22,22,0.6492670774459839,1.2988300010806224
23,23,0.6268007159233093,1.292359399013832
24,24,0.6061885952949524,1.2882296452756787
25,25,0.5871606469154358,1.2824040397268828
26,26,0.5691555738449097,1.2748278008132685
27,27,0.5516729354858398,1.2711516833696208
28,28,0.5351634621620178,1.2701318459432633
29,29,0.5199246406555176,1.2674638091540726
30,30,0.5049114227294922,1.2636970144803408
31,31,0.4902104437351227,1.2610631223584785
32,32,0.4756786823272705,1.2584272290839524
33,33,0.4621107280254364,1.255615484519083
34,34,0.44944846630096436,1.2540703445184427
35,35,0.43698203563690186,1.2524386546650872
36,36,0.42517146468162537,1.2507559354188011
37,37,0.4137410819530487,1.2509683077452614
38,38,0.4037213921546936,1.2567531398085297
39,39,0.3978976607322693,1.267413655265433
40,40,0.4024382531642914,1.2961763475762038
41,41,0.42065200209617615,1.2965190449699027
42,42,0.4150490462779999,1.2597872624631787
43,43,0.36139780282974243,1.2580693979732325
44,44,0.3516247868537903,1.2796440749871927
45,45,0.3700992166996002,1.2610097165967598
46,46,0.3352530002593994,1.2570625680391905
47,47,0.323688805103302,1.270326958327997
48,48,0.3347915709018707,1.2551622234407018
49,49,0.3053224980831146,1.263193974729444
50,50,0.30493831634521484,1.2654824178726947
51,51,0.30388960242271423,1.2545681312435963
52,52,0.2818894386291504,1.2702581687051742
53,53,0.2884025275707245,1.262331102715164
54,54,0.275463342666626,1.2605425725217725
55,55,0.2666817903518677,1.27254035824635
56,56,0.26852670311927795,1.263793194880251
57,57,0.25341159105300903,1.269014702468622
58,58,0.2542926073074341,1.2723531254002305
59,59,0.24717190861701965,1.271497507564357
60,60,0.23948031663894653,1.2746210567286758
61,61,0.2397928684949875,1.2736743864465931
62,62,0.22980232536792755,1.2808595250864498
63,63,0.22896364331245422,1.2792918721183402
64,64,0.2239297479391098,1.2806218882076075
65,65,0.21811462938785553,1.2880900648773694
66,66,0.21734708547592163,1.2851532482710042
67,67,0.2104702740907669,1.2891102775198515
68,68,0.20879080891609192,1.2934851724593366
69,69,0.20503965020179749,1.2932127655529586
70,70,0.2005508691072464,1.2955067118660348
71,71,0.19916898012161255,1.2984105094534453
72,72,0.19442874193191528,1.301788580222208
73,73,0.19228774309158325,1.3012956713066726
74,74,0.18956071138381958,1.3049686619492828
75,75,0.18587909638881683,1.3100524652199668
76,76,0.1842636913061142,1.3084310312740137
77,77,0.1809358149766922,1.312695987889024
78,78,0.17849409580230713,1.317352795210041
79,79,0.17649687826633453,1.3165936079181608
80,80,0.1734810471534729,1.3192101150262552
81,81,0.17159822583198547,1.3237908785460426
82,82,0.16930103302001953,1.323987741939357
83,83,0.16688545048236847,1.3256595799180328
84,84,0.16505752503871918,1.330614308841893
85,85,0.16279320418834686,1.3309029751136654
86,86,0.16081275045871735,1.3335994032562757
87,87,0.1589294970035553,1.3365338434938525
88,88,0.15687048435211182,1.3391644837426357
89,89,0.15512827038764954,1.3393502157242572
90,90,0.15329909324645996,1.3450251094630508
91,91,0.1514773666858673,1.343138147573002
92,92,0.14997804164886475,1.3511209956935195
93,93,0.14860546588897705,1.3463244829021517
94,94,0.14772260189056396,1.3633265260790215
95,95,0.14818719029426575,1.3500226130251025
96,96,0.15129248797893524,1.3898711908059043
97,97,0.15794412791728973,1.360491017826268
98,98,0.16426314413547516,1.398404480981045
99,99,0.15833468735218048,1.3586328225057633

```


## File: `reports\linear_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.4580087661743164,3.065345388944032
1,1,3.0447001457214355,2.346475319784196
2,2,2.1839418411254883,1.9878092281153945
3,3,1.734474539756775,1.7977502541463883
4,4,1.489260196685791,1.6987449771068135
5,5,1.3653204441070557,1.6524376791031634
6,6,1.3012382984161377,1.6120140200755635
7,7,1.2344423532485962,1.566025905921811
8,8,1.1610493659973145,1.5281574687019723
9,9,1.1012649536132812,1.4897736095991292
10,10,1.0442585945129395,1.4583885317943135
11,11,0.9967300891876221,1.4319605592821465
12,12,0.9566308856010437,1.4142396020107582
13,13,0.9221247434616089,1.4024120393346569
14,14,0.889514148235321,1.389162157402664
15,15,0.8563276529312134,1.3776914252609502
16,16,0.825962245464325,1.3686693535476435
17,17,0.796024739742279,1.3627164246606045
18,18,0.7693156003952026,1.3557999407658812
19,19,0.7427224516868591,1.350797559394211
20,20,0.7178900837898254,1.3457776679367315
21,21,0.6934908628463745,1.3409137413149974
22,22,0.6707596778869629,1.3381620313300462
23,23,0.6495320200920105,1.336594003145812
24,24,0.630782425403595,1.335477234887295
25,25,0.61297208070755,1.3351928210649333
26,26,0.596035897731781,1.33192006095511
27,27,0.5794575214385986,1.3311074679015114
28,28,0.5640406608581543,1.3308072950019212
29,29,0.5487897396087646,1.3307425076844261
30,30,0.5346084237098694,1.3306036777183659
31,31,0.5205250978469849,1.3297089123335042
32,32,0.5068598985671997,1.331665289206583
33,33,0.4937836825847626,1.3297665705446338
34,34,0.48122313618659973,1.33015004142386
35,35,0.4695098102092743,1.3285474933561732
36,36,0.4582706093788147,1.3326329715916367
37,37,0.4480991065502167,1.3317829819976306
38,38,0.4396567940711975,1.3425470571048925
39,39,0.4343782067298889,1.345000970559042
40,40,0.4330768883228302,1.3610320794777793
41,41,0.4293825626373291,1.3466785618516266
42,42,0.41265007853507996,1.3413466156506149
43,43,0.3885858654975891,1.3411597580206198
44,44,0.3787391781806946,1.3470258869108607
45,45,0.3810245990753174,1.3582648605596823
46,46,0.3739548623561859,1.3450636316518314
47,47,0.3561115860939026,1.345043620125192
48,48,0.3467499911785126,1.3571982461898053
49,49,0.34706616401672363,1.3532921212618467
50,50,0.3401646316051483,1.3518988187195824
51,51,0.3264698088169098,1.3541040889552383
52,52,0.3206096589565277,1.3569838727106813
53,53,0.31959328055381775,1.3618284131659837
54,54,0.3119523227214813,1.3565585026975537
55,55,0.3020618259906769,1.3586171885005762
56,56,0.2984493672847748,1.3667134769627305
57,57,0.2960899770259857,1.361645182625192
58,58,0.28880760073661804,1.362989707071273
59,59,0.2816428244113922,1.367848505739306
60,60,0.278865247964859,1.3668943311347337
61,61,0.2757687270641327,1.369721959848873
62,62,0.2695055902004242,1.369081340852331
63,63,0.264072984457016,1.370590585177062
64,64,0.2614922523498535,1.3757389256211578
65,65,0.2584391236305237,1.3733698110111425
66,66,0.253355473279953,1.3754314985431608
67,67,0.2488439381122589,1.378631591796875
68,68,0.24619251489639282,1.3782456194768187
69,69,0.24349217116832733,1.3820605668865267
70,70,0.23947353661060333,1.3814504654681097
71,71,0.23543456196784973,1.3830964135341957
72,72,0.23262931406497955,1.387225166696017
73,73,0.23032034933567047,1.387034306760694
74,74,0.22735072672367096,1.390033784459849
75,75,0.22389401495456696,1.3919837826588115
76,76,0.22088007628917694,1.3932024846311475
77,77,0.21857839822769165,1.3987781962410348
78,78,0.21641996502876282,1.397829149590164
79,79,0.2139415442943573,1.404067617947938
80,80,0.2114490121603012,1.4029154543016777
81,81,0.2099515199661255,1.415541476890689
82,82,0.211494579911232,1.417525869901063
83,83,0.2211124449968338,1.4623162941854508
84,84,0.24351654946804047,1.4489796122566598
85,85,0.25610464811325073,1.447570550637167
86,86,0.21986210346221924,1.4191351718589909
87,87,0.1985253244638443,1.4351536485015368
88,88,0.2233058363199234,1.4462218988137168
89,89,0.20985348522663116,1.4233766149301998
90,90,0.19212812185287476,1.4347429119172643
91,91,0.21020515263080597,1.436861132012039
92,92,0.19459959864616394,1.4361927470222848
93,93,0.19067665934562683,1.433534966140497
94,94,0.19823817908763885,1.4315973500736425
95,95,0.18327438831329346,1.4494309972544186
96,96,0.1912623941898346,1.4334574214747695
97,97,0.1845063865184784,1.4370713781137936
98,98,0.18096984922885895,1.4508698260197874
99,99,0.1842384785413742,1.4414402695952868

```


## File: `reports\linear_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7517350748181343,0.8925475031137466,0.4,0.15,0.5125,0.5625,0.2,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.08750000000000002,-0.14999999999999997,0.19999999999999996,20,7.245480351406546
2,0.842614758014679,0.8714069545269012,0.5294117647058824,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.662879614491889
3,1.2523650877615984,0.825560657417073,0.0,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,4.529887431770876
4,2.264769653479258,0.7627622981866201,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.53228318947619
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8651082083582878,0.8754298031330109,0.55,0.15,0.55,0.5625,0.3,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.04999999999999999,0.25,20,6.295954264951511
2,0.8376282423734664,0.8713630318641663,0.5882352941176471,0.5882352941176471,0.4125,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.07500000000000001,0.30000000000000004,0.30000000000000004,20,6.702544649327672
3,1.0868414079441744,0.8413489636252908,0.5,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,5.2197798405296805
4,1.569451590379079,0.8073779543240865,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.6541669438533764
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9064998745918273,0.8720436155796051,0.2,0.45,0.4625,0.475,0.0,0.0,0.05,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.03750000000000003,-0.05,-0.15000000000000002,20,6.142774112462388
2,1.276777121424675,0.8078093200922012,0.4,0.45,0.425,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.07500000000000001,-0.05,-0.2,20,4.573391150345698
3,1.7245918393135071,0.7620211154222488,0.4,0.3,0.475,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.2375,-0.9,0.19999999999999996,20,3.3055145534072685
4,3.4383003950119018,0.626533716917038,0.3,0.55,0.3375,0.4625,1.0,0.9,0.95,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.16250000000000003,1.0,0.1499999999999999,20,1.6878080166700355
5,3.987105441093445,0.721822202205658,,,0.2875,0.3625,0.0,0.0,0.7,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.175,0.0,-0.10000000000000009,20,1.4432991120927745
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\linear_seed43\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\linear_seed43\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.063492,63,0.002795,0.124189
oracle1,probeA_per_label,all,0.496032,63,,
oracle1,probeA_jaccard,all,0.378307,63,,

```


## File: `reports\linear_seed43\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.063492 |
| probeA_exact_ci_lo | 0.002795 |
| probeA_exact_ci_hi | 0.124189 |
| probeA_per_label | 0.496032 |
| probeA_jaccard | 0.378307 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\linear_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.893 | 0.400 | 0.150 | 0.512 | 0.562 | 0.7517 |
| 2 | 0.871 | 0.529 | 0.588 | 0.450 | 0.512 | 0.8426 |
| 3 | 0.826 | 0.000 | 0.667 | 0.515 | 0.471 | 1.2524 |
| 4 | 0.763 | nan | nan | 0.208 | 0.417 | 2.2648 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **2**; rollout state probe accuracy stays >= 0.50 through depth **3**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.3518.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\linear_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.875 | +0.107 |
| 2 | 1.000 | 0.871 | 0.871 | +0.129 |
| 3 | 1.000 | 0.826 | 0.841 | +0.174 |
| 4 | 1.000 | 0.763 | 0.807 | +0.237 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.512 | 0.550 | +0.050 |
| 2 | 0.512 | 0.450 | 0.412 | +0.062 |
| 3 | 0.471 | 0.515 | 0.471 | -0.044 |
| 4 | 0.417 | 0.208 | 0.250 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.550 | -0.250 |
| 2 | 0.588 | 0.529 | 0.588 | +0.059 |
| 3 | 0.667 | 0.000 | 0.500 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.137 | -3.000 | +0.375 |
| 2 | +0.000 | +0.375 | n/a |
| 3 | -0.100 | n/a | -3.000 |
| 4 | -0.232 | -0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.875 | 0.287 | 0.562 | 0.512 | 0.550 | 0.600 |
| 2 | 1.000 | 0.871 | 0.871 | 0.246 | 0.512 | 0.450 | 0.412 | 0.487 |
| 3 | 1.000 | 0.826 | 0.841 | 0.241 | 0.471 | 0.515 | 0.471 | 0.456 |
| 4 | 1.000 | 0.763 | 0.807 | 0.237 | 0.417 | 0.208 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1619
**Average Oracle Gain (state probe):** +0.0692
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\linear_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\linear_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\linear_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.473798990249634,3.12270242659772
1,1,3.124931812286377,2.423854139984631
2,2,2.292603015899658,2.0329588593029584
3,3,1.8056186437606812,1.803791358822682
4,4,1.5028027296066284,1.686751193687564
5,5,1.3437180519104004,1.6321573726466445
6,6,1.2653870582580566,1.5842236378153816
7,7,1.1954407691955566,1.5367098948994622
8,8,1.127571702003479,1.5044559416223744
9,9,1.075363039970398,1.4730133306784707
10,10,1.0219863653182983,1.4403171226626537
11,11,0.9692369699478149,1.4161632100089652
12,12,0.9268369078636169,1.4033475782050462
13,13,0.8926759958267212,1.391113781538166
14,14,0.8607675433158875,1.3744579377721569
15,15,0.8272882103919983,1.3592411729155993
16,16,0.7937148213386536,1.3447971031314037
17,17,0.7618528604507446,1.332826708183914
18,18,0.7327417731285095,1.3249610525662783
19,19,0.7062252759933472,1.3162451572105534
20,20,0.6815218329429626,1.3101558997982838
21,21,0.6581063270568848,1.301556571585233
22,22,0.6361016631126404,1.2960582795690319
23,23,0.6155379414558411,1.291505657258581
24,24,0.595582902431488,1.2865640608990778
25,25,0.5767673850059509,1.2822185578893444
26,26,0.5586187243461609,1.2760069800204918
27,27,0.5420236587524414,1.2764374779873207
28,28,0.5274990797042847,1.2705583416047643
29,29,0.5168663859367371,1.2939273021260247
30,30,0.5179003477096558,1.2994350996173796
31,31,0.5353514552116394,1.3200240838723105
32,32,0.5210318565368652,1.2561431634621543
33,33,0.46049964427948,1.2617764082111296
34,34,0.45885926485061646,1.29708737232646
35,35,0.4683185815811157,1.2488270743948515
36,36,0.422758013010025,1.254072971031314
37,37,0.4205912947654724,1.2791991937355918
38,38,0.4210967421531677,1.2442879598648822
39,39,0.38606560230255127,1.2520852010758197
40,40,0.3920612037181854,1.2646301769819417
41,41,0.3785865008831024,1.2496300369012552
42,42,0.3581661283969879,1.2515223768890882
43,43,0.3634863793849945,1.2518510662141393
44,44,0.34213244915008545,1.2565582775678792
45,45,0.33664581179618835,1.2491226196289062
46,46,0.3323926329612732,1.2462893626728997
47,47,0.3149472773075104,1.259964114329854
48,48,0.31580549478530884,1.2468349269179047
49,49,0.30351153016090393,1.24755859375
50,50,0.2955329716205597,1.2609345482998207
51,51,0.2933961749076843,1.2507730702884863
52,52,0.2806623578071594,1.2527290719454405
53,53,0.27867117524147034,1.2607529436955687
54,54,0.27141785621643066,1.258135060795018
55,55,0.26335376501083374,1.257172631435707
56,56,0.2614957094192505,1.2610381079501793
57,57,0.25258517265319824,1.265459779833184
58,58,0.248908132314682,1.261278371341893
59,59,0.24472537636756897,1.263548553967085
60,60,0.23758210241794586,1.2713217813460553
61,61,0.23538519442081451,1.2662049590564164
62,62,0.22976601123809814,1.2684443739594007
63,63,0.22508211433887482,1.2771341292584528
64,64,0.22253994643688202,1.2730435230692878
65,65,0.21702887117862701,1.2748293016777663
66,66,0.2138654738664627,1.2823566374231556
67,67,0.21075992286205292,1.2799282386654713
68,68,0.2061084657907486,1.2812552530257428
69,69,0.2036094069480896,1.2871934234118851
70,70,0.2002890408039093,1.2865545554239242
71,71,0.19648590683937073,1.28697642341989
72,72,0.1942278891801834,1.2931165851530482
73,73,0.19103984534740448,1.293317826067815
74,74,0.18786323070526123,1.2938352491034837
75,75,0.18570750951766968,1.3000855993051998
76,76,0.18279367685317993,1.3002687047739498
77,77,0.18002429604530334,1.3015877145235655
78,78,0.17795519530773163,1.3061791091668802
79,79,0.1753687858581543,1.3078968485847848
80,80,0.17289352416992188,1.307357412869813
81,81,0.17102645337581635,1.3157578765368851
82,82,0.16906999051570892,1.3118748899366035
83,83,0.1676376312971115,1.3237616116883324
84,84,0.16803081333637238,1.3211578619284707
85,85,0.17122521996498108,1.345440848928983
86,86,0.17708829045295715,1.330812047739498
87,87,0.17934949696063995,1.3448586385758197
88,88,0.16877083480358124,1.3247282934970543
89,89,0.15571515262126923,1.326629138383709
90,90,0.15612144768238068,1.3459327572681865
91,91,0.16114957630634308,1.3322806436507428
92,92,0.1561952829360962,1.334981074098681
93,93,0.14879627525806427,1.3417145775966957
94,94,0.1495441198348999,1.3385563834768828
95,95,0.15019209682941437,1.343246334888896
96,96,0.14554713666439056,1.344025658779457
97,97,0.14299023151397705,1.3454013261638704
98,98,0.14304138720035553,1.3486840920370133
99,99,0.14103789627552032,1.351836407770876

```


## File: `reports\linear_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.547534227371216,3.1573678939068905
1,1,3.143289566040039,2.470789424708632
2,2,2.3241188526153564,2.085841444672131
3,3,1.8465005159378052,1.853409688980853
4,4,1.5549120903015137,1.7064858108270364
5,5,1.3730701208114624,1.6466402147637038
6,6,1.2912684679031372,1.6088176789830944
7,7,1.2277226448059082,1.5650207019243083
8,8,1.1584501266479492,1.5263831967213115
9,9,1.099727988243103,1.4890266793673155
10,10,1.0436184406280518,1.4584316816486296
11,11,0.9946845769882202,1.4350969908667393
12,12,0.955093264579773,1.4190238577420595
13,13,0.9206522703170776,1.4093487849001025
14,14,0.8897721767425537,1.397247314453125
15,15,0.8578667640686035,1.3835409195696722
16,16,0.8261060118675232,1.37103021340292
17,17,0.7946691513061523,1.3620973180551998
18,18,0.7662612795829773,1.3554907626793034
19,19,0.7394025921821594,1.3515688786741162
20,20,0.7157735228538513,1.3433163752321338
21,21,0.6923642158508301,1.3366846803758965
22,22,0.6710613369941711,1.3332274390048668
23,23,0.6501696705818176,1.3319329433753841
24,24,0.6302686929702759,1.3324524926357582
25,25,0.6112964749336243,1.3291175717213115
26,26,0.5936398506164551,1.3273050276959528
27,27,0.5771744847297668,1.3266393942911117
28,28,0.5619783401489258,1.328447060506852
29,29,0.5474713444709778,1.328292471463563
30,30,0.5330769419670105,1.3276139556384476
31,31,0.5194072723388672,1.3271513141569544
32,32,0.5059272050857544,1.328173653024142
33,33,0.49274006485939026,1.327790057072874
34,34,0.47996532917022705,1.3271394323130123
35,35,0.4674260914325714,1.3260770703925462
36,36,0.45602089166641235,1.3326493560290726
37,37,0.44653764367103577,1.3357994204661885
38,38,0.4445658028125763,1.3775080696481172
39,39,0.4649921655654907,1.3974646896612448
40,40,0.4997044801712036,1.3846775742827868
41,41,0.44928255677223206,1.3381866705222207
42,42,0.3959372341632843,1.3723809914510758
43,43,0.43712475895881653,1.3615297411308913
44,44,0.39729949831962585,1.3524476348376664
45,45,0.379303514957428,1.3637865410476435
46,46,0.39737850427627563,1.340972650246542
47,47,0.3539227843284607,1.3716045442174694
48,48,0.3724820613861084,1.3455172679463372
49,49,0.34828874468803406,1.3491711225665983
50,50,0.34341490268707275,1.3647533479284069
51,51,0.3419913351535797,1.349781098912974
52,52,0.3223082721233368,1.3566803228659707
53,53,0.3304480016231537,1.3510371974257172
54,54,0.30895960330963135,1.3682708740234375
55,55,0.31603145599365234,1.3511019847432122
56,56,0.2994518280029297,1.3573458311987705
57,57,0.3011668026447296,1.3627579485783812
58,58,0.2911893427371979,1.3646665479316087
59,59,0.2878480851650238,1.3585457723648822
60,60,0.2832190692424774,1.3610362068551485
61,61,0.2764638066291809,1.3709284047611425
62,62,0.2750120162963867,1.3642130367091445
63,63,0.2667694389820099,1.36503288394115
64,64,0.2667771577835083,1.367839750696401
65,65,0.25847601890563965,1.375563949835105
66,66,0.25872254371643066,1.368332034251729
67,67,0.2511866092681885,1.3715483868708376
68,68,0.25074535608291626,1.3760135838242828
69,69,0.24475330114364624,1.3789482742059427
70,70,0.2431371808052063,1.3745073412285476
71,71,0.23885577917099,1.3780451289943008
72,72,0.23612608015537262,1.3845246111760374
73,73,0.2332179993391037,1.3822673109711194
74,74,0.2297355681657791,1.3816504556624616
75,75,0.2277253121137619,1.3877469672531377
76,76,0.22394207119941711,1.3918815988009092
77,77,0.22232042253017426,1.3881685851050205
78,78,0.21877923607826233,1.3922289238601435
79,79,0.21702440083026886,1.3975568677558274
80,80,0.2139817625284195,1.3971347496157787
81,81,0.21199697256088257,1.397550739225794
82,82,0.20949766039848328,1.4030076323962601
83,83,0.20726042985916138,1.4054412841796875
84,84,0.20511814951896667,1.4052666836097591
85,85,0.20289380848407745,1.4088519987512806
86,86,0.2009391188621521,1.4134766625576332
87,87,0.1987464278936386,1.4131132032050462
88,88,0.1970379650592804,1.4166035886670723
89,89,0.19502204656600952,1.420289586801998
90,90,0.19413110613822937,1.4271507888543802
91,91,0.19489410519599915,1.430402787005315
92,92,0.20352166891098022,1.4775047927606302
93,93,0.2302180677652359,1.4803095332911758
94,94,0.25833815336227417,1.4749318107229765
95,95,0.2229202389717102,1.431914532770876
96,96,0.18367353081703186,1.4548767590131917
97,97,0.21621733903884888,1.4625291668000768
98,98,0.20308531820774078,1.4390530195392546
99,99,0.18181568384170532,1.452455864577997

```


## File: `reports\linear_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7554270699620247,0.8923284560441971,0.4,0.15,0.5,0.5625,0.2,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.09999999999999998,-0.14999999999999997,0.25,20,7.210069549575086
2,0.83287333548069,0.8730126082897186,0.4117647058823529,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.740809742463157
3,1.2115034180528976,0.8290568765471963,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.682671783260453
4,2.0843040347099304,0.7721430758635203,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,2.7515362567244073
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8411295711994171,0.8781614243984223,0.5,0.15,0.5375,0.5625,0.3,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.0625,-0.04999999999999999,0.25,20,6.475437198446337
2,0.8205747067928314,0.8737680643796921,0.5882352941176471,0.5882352941176471,0.5,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.841839807601375
3,1.1243885790600496,0.8363730556824628,0.5,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.045473581546329
4,1.8870981733004253,0.7860081990559896,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.0390777770247595
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9206722065806389,0.8710367053747177,0.25,0.45,0.475,0.475,0.0,0.0,0.05,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.15000000000000002,20,6.048215556842007
2,1.28966945707798,0.8070818156003952,0.5,0.45,0.4875,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.1375,-0.05,-0.2,20,4.527672696318181
3,1.6603946030139922,0.7667847245931625,0.45,0.3,0.5,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.2625,-0.9,0.19999999999999996,20,3.433318449235027
4,3.0906944155693052,0.6458099573850632,0.4,0.55,0.4,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.22500000000000003,1.0,0.19999999999999996,20,1.8776333697655099
5,3.238003158569336,0.7365975767374039,,,0.2875,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.175,0.0,0.19999999999999996,20,1.7772020165332447
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\linear_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\linear_seed44\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\linear_seed44\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.063492,63,0.002795,0.124189
oracle1,probeA_per_label,all,0.488095,63,,
oracle1,probeA_jaccard,all,0.396825,63,,

```


## File: `reports\linear_seed44\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.063492 |
| probeA_exact_ci_lo | 0.002795 |
| probeA_exact_ci_hi | 0.124189 |
| probeA_per_label | 0.488095 |
| probeA_jaccard | 0.396825 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\linear_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.400 | 0.150 | 0.500 | 0.562 | 0.7554 |
| 2 | 0.873 | 0.412 | 0.588 | 0.463 | 0.512 | 0.8329 |
| 3 | 0.829 | 0.167 | 0.667 | 0.471 | 0.471 | 1.2115 |
| 4 | 0.772 | nan | nan | 0.167 | 0.417 | 2.0843 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.3580.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\linear_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.878 | +0.108 |
| 2 | 1.000 | 0.873 | 0.874 | +0.127 |
| 3 | 1.000 | 0.829 | 0.836 | +0.171 |
| 4 | 1.000 | 0.772 | 0.786 | +0.228 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.500 | 0.537 | +0.062 |
| 2 | 0.512 | 0.463 | 0.500 | +0.050 |
| 3 | 0.471 | 0.471 | 0.485 | +0.000 |
| 4 | 0.417 | 0.167 | 0.250 | +0.250 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.500 | -0.250 |
| 2 | 0.588 | 0.412 | 0.588 | +0.176 |
| 3 | 0.667 | 0.167 | 0.500 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.116 | -1.500 | +0.286 |
| 2 | -0.006 | -3.000 | n/a |
| 3 | -0.045 | +1.000 | -2.000 |
| 4 | -0.065 | -0.500 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.878 | 0.287 | 0.562 | 0.500 | 0.537 | 0.600 |
| 2 | 1.000 | 0.873 | 0.874 | 0.246 | 0.512 | 0.463 | 0.500 | 0.487 |
| 3 | 1.000 | 0.829 | 0.836 | 0.241 | 0.471 | 0.471 | 0.485 | 0.456 |
| 4 | 1.000 | 0.772 | 0.786 | 0.237 | 0.417 | 0.167 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1584
**Average Oracle Gain (state probe):** +0.0906
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\linear_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\linear_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\linear_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.449768304824829,3.0509513479764343
1,1,3.031238317489624,2.3335378678118595
2,2,2.1770596504211426,1.9851124247566598
3,3,1.74236261844635,1.7840363549404457
4,4,1.481698989868164,1.6913799848712858
5,5,1.355785846710205,1.6490210861456198
6,6,1.2876895666122437,1.6022352625112064
7,7,1.2136595249176025,1.55513313168385
8,8,1.143774390220642,1.5131817176693776
9,9,1.0850751399993896,1.4711448794505635
10,10,1.0291881561279297,1.434936023149334
11,11,0.9800440073013306,1.409696735319544
12,12,0.9390828609466553,1.3953477202868851
13,13,0.9031094312667847,1.3837365322425716
14,14,0.8688772916793823,1.3709783085056992
15,15,0.8361627459526062,1.358074375840484
16,16,0.8035537004470825,1.3459078679319287
17,17,0.7725100517272949,1.3329555323866547
18,18,0.7424705624580383,1.3232783333199922
19,19,0.7145848870277405,1.316193127241291
20,20,0.6883550882339478,1.3079620111184043
21,21,0.6637231707572937,1.2998379566630378
22,22,0.6405051350593567,1.2925227431000257
23,23,0.6190376281738281,1.287371651071017
24,24,0.5986884236335754,1.2830562904232838
25,25,0.5800177454948425,1.27575433449667
26,26,0.5625174641609192,1.2718740994813011
27,27,0.5461093187332153,1.2686143468637936
28,28,0.530292809009552,1.2661860731781507
29,29,0.5151050686836243,1.2618903488409323
30,30,0.5004920959472656,1.2593022330862578
31,31,0.48605847358703613,1.2557029098760886
32,32,0.4721527695655823,1.2540029306880762
33,33,0.4590275287628174,1.2511739261814805
34,34,0.44662734866142273,1.2537166407850922
35,35,0.43556588888168335,1.2515007394259092
36,36,0.42763665318489075,1.27187985279521
37,37,0.4283590614795685,1.279496865194352
38,38,0.44005805253982544,1.2979359861280098
39,39,0.4324531853199005,1.2534107145715931
40,40,0.38801848888397217,1.2515356345254867
41,41,0.37389469146728516,1.2834710293128841
42,42,0.38812440633773804,1.2554263755923412
43,43,0.36303484439849854,1.248822446729316
44,44,0.34280166029930115,1.2744903564453125
45,45,0.3528389036655426,1.2524745503409964
46,46,0.3329184651374817,1.2484030801741803
47,47,0.31883978843688965,1.2687773157338627
48,48,0.32429003715515137,1.2504517602138832
49,49,0.3054039478302002,1.250608100265753
50,50,0.29893961548805237,1.2664609815253587
51,51,0.29883456230163574,1.253462869612897
52,52,0.28254279494285583,1.2546783197121543
53,53,0.2812032401561737,1.263886998911373
54,54,0.2757093906402588,1.2572906994428792
55,55,0.2641511559486389,1.2573499835905482
56,56,0.26409727334976196,1.261323772492956
57,57,0.25541916489601135,1.2608078503217854
58,58,0.2489682137966156,1.2587226492459658
59,59,0.2476116120815277,1.2613775534707992
60,60,0.23867028951644897,1.2658298680039703
61,61,0.23563775420188904,1.2620036641105277
62,62,0.23229286074638367,1.265138970046747
63,63,0.22496455907821655,1.2712085911485016
64,64,0.2231099009513855,1.2665414028480404
65,65,0.21852566301822662,1.2705725998174948
66,66,0.2132786512374878,1.2765878145811989
67,67,0.21143315732479095,1.2723107259781634
68,68,0.2065833956003189,1.275541211737961
69,69,0.20274165272712708,1.281876485855853
70,70,0.20070278644561768,1.2790009545498207
71,71,0.19634944200515747,1.2812770155609632
72,72,0.19327843189239502,1.2895776717389216
73,73,0.1910559982061386,1.2858311387359118
74,74,0.18721261620521545,1.2903402359759222
75,75,0.1844853311777115,1.296797830550397
76,76,0.18233266472816467,1.2956869406778304
77,77,0.17905603349208832,1.2983035728579662
78,78,0.17650704085826874,1.306163725305776
79,79,0.17453736066818237,1.3029745133196722
80,80,0.17186103761196136,1.309981924588563
81,81,0.16961570084095,1.3117725810066598
82,82,0.16852720081806183,1.3217318175268955
83,83,0.16874858736991882,1.3184099041047643
84,84,0.17346976697444916,1.361176537685707
85,85,0.18704403936862946,1.3405303955078125
86,86,0.2002566158771515,1.3655316712426357
87,87,0.18432283401489258,1.3278758564933402
88,88,0.1573481559753418,1.327577184458248
89,89,0.163370281457901,1.3629655681672643
90,90,0.17240627110004425,1.3352135830238216
91,91,0.15587110817432404,1.3318582753666113
92,92,0.15259845554828644,1.3586243176069417
93,93,0.15948618948459625,1.3419114409900101
94,94,0.14891111850738525,1.3379883062644082
95,95,0.14782333374023438,1.3584555954229636
96,96,0.14997519552707672,1.3494307721247438
97,97,0.14185886085033417,1.3449444379962858
98,98,0.14412903785705566,1.3606990126312757
99,99,0.14170344173908234,1.3580248473120518

```


## File: `reports\linear_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.477013349533081,3.07566183121478
1,1,3.0568203926086426,2.373477122822746
2,2,2.207951784133911,2.00403670013928
3,3,1.7402470111846924,1.7947807937371927
4,4,1.4793387651443481,1.683149619180648
5,5,1.3430713415145874,1.6345267374007428
6,6,1.276816487312317,1.5963565013447747
7,7,1.2133615016937256,1.5589709672771517
8,8,1.1504831314086914,1.5271224975585938
9,9,1.0991874933242798,1.4900700303374743
10,10,1.044881820678711,1.4580144413181992
11,11,0.9954438209533691,1.4320155909804047
12,12,0.9517760276794434,1.4161929771548412
13,13,0.9153178334236145,1.404894344142226
14,14,0.8827284574508667,1.3913143970927253
15,15,0.8502361178398132,1.3791736540247181
16,16,0.8189092874526978,1.3703735851850667
17,17,0.7889904379844666,1.3628787681704662
18,18,0.7612493634223938,1.355958532114498
19,19,0.7346100807189941,1.3519320878826204
20,20,0.7099528908729553,1.3475414338659069
21,21,0.6860494613647461,1.3442908115074284
22,22,0.6640301942825317,1.3406509649558145
23,23,0.6437587738037109,1.3389324751056608
24,24,0.6243861317634583,1.337342309170082
25,25,0.6063244938850403,1.3333617663774333
26,26,0.588694155216217,1.3328318361376152
27,27,0.5727509260177612,1.3336887046939037
28,28,0.5573263168334961,1.33761471607646
29,29,0.5434308648109436,1.3360695760758197
30,30,0.5310061573982239,1.3435962864610016
31,31,0.5219780802726746,1.3476985243500257
32,32,0.5198127627372742,1.3676265028656507
33,33,0.5202410221099854,1.352404860199475
34,34,0.5040169358253479,1.3376363535396387
35,35,0.4694572687149048,1.3330423323834528
36,36,0.4552190601825714,1.341187774157915
37,37,0.4601123034954071,1.3470527773997822
38,38,0.44570156931877136,1.3299194085793418
39,39,0.4214707314968109,1.3332374447681865
40,40,0.41825175285339355,1.3479534211705944
41,41,0.4158915877342224,1.3343856061091188
42,42,0.39659562706947327,1.3347132948578382
43,43,0.38600820302963257,1.348569526047003
44,44,0.38571810722351074,1.3397599517321976
45,45,0.37306758761405945,1.3397707079277663
46,46,0.36005523800849915,1.3492241531121927
47,47,0.35813987255096436,1.3446777844038167
48,48,0.3509812355041504,1.3454117071433145
49,49,0.338541716337204,1.351042950739626
50,50,0.3343072533607483,1.3499895940061475
51,51,0.33043143153190613,1.3525459414622822
52,52,0.3202168643474579,1.3541571195008324
53,53,0.3140011727809906,1.3547583408043034
54,54,0.311267226934433,1.3600271256243597
55,55,0.3039533197879791,1.3579456767097848
56,56,0.2965640425682068,1.3590649464091316
57,57,0.2933250069618225,1.3660553478803792
58,58,0.28885918855667114,1.3623966154504994
59,59,0.2819388806819916,1.3638607087682506
60,60,0.27744412422180176,1.3701609627145235
61,61,0.2745033800601959,1.3674186331326845
62,62,0.269462913274765,1.3699913650262552
63,63,0.2641168236732483,1.3729298075691598
64,64,0.2607963979244232,1.3720640588979252
65,65,0.2576109766960144,1.377164496750128
66,66,0.2530617415904999,1.3761539146548412
67,67,0.2488037347793579,1.3772531728275488
68,68,0.2458728551864624,1.3826716688812757
69,69,0.24288275837898254,1.3801464643634733
70,70,0.23906943202018738,1.3834368596311475
71,71,0.2354760617017746,1.3857359338979252
72,72,0.23270758986473083,1.3866039338659069
73,73,0.2300482541322708,1.3902622910796618
74,74,0.22691000998020172,1.3907920962474385
75,75,0.22368799149990082,1.3935674448482325
76,76,0.2209632247686386,1.396695996894211
77,77,0.21862895786762238,1.3982338827164447
78,78,0.2162053883075714,1.4010359967341188
79,79,0.21359986066818237,1.404936993708376
80,80,0.2111971229314804,1.4031542168288935
81,81,0.20961524546146393,1.418641137295082
82,82,0.21011687815189362,1.4107714793721184
83,83,0.2164258360862732,1.4620022382892546
84,84,0.2346673458814621,1.4419165439293034
85,85,0.25394707918167114,1.4662948358254355
86,86,0.23097680509090424,1.4166537425557122
87,87,0.19675323367118835,1.4257880038902409
88,88,0.21395114064216614,1.4611731357261784
89,89,0.2177966684103012,1.4210305135758197
90,90,0.1920575499534607,1.428570231453317
91,91,0.2035943567752838,1.4499951972336065
92,92,0.20156404376029968,1.430798014656442
93,93,0.18632856011390686,1.4321349097079918
94,94,0.19798514246940613,1.4400727318935707
95,95,0.18718208372592926,1.445740246381916
96,96,0.18545283377170563,1.4349492807857325
97,97,0.1890038102865219,1.4366977879258453
98,98,0.17860351502895355,1.455900098456711
99,99,0.18393832445144653,1.440837547427318

```


## File: `reports\mlp\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7484399557113648,0.8911447495222091,0.35,0.15,0.475,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.125,-0.09999999999999998,0.25,20,7.2773796648537425
2,0.7892478168010711,0.8780645936727524,0.5294117647058824,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,7.113406682328628
3,0.996183248127208,0.8498814035864437,0.16666666666666666,0.6666666666666666,0.4264705882352941,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.694808542208562
4,1.368765374024709,0.826249897480011,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,4.189935127214733
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8043881759047509,0.8828081905841827,0.5,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.77121056376999
2,0.8206886410713196,0.8730108708143234,0.5882352941176471,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.840889971033644
3,0.9593773273860707,0.8538061520632576,0.3333333333333333,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,5.913286367207371
4,1.1769853432973225,0.8417754272619883,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,4.872650415063489
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.883877956867218,0.8704910725355148,0.2,0.45,0.425,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.0,-0.05,-0.2,20,6.299991892918768
2,1.1991392016410827,0.8123234987258912,0.4,0.45,0.4,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.050000000000000044,-0.05,-0.2,20,4.86949403380043
3,1.340456449985504,0.7963992059230804,0.35,0.3,0.425,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.1875,-0.9,0.19999999999999996,20,4.252777793414964
4,2.312501984834671,0.68907490670681,0.25,0.55,0.375,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.2,1.0,0.19999999999999996,20,2.5094858333000434
5,1.77137291431427,0.8156516343355179,,,0.325,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.21250000000000002,0.0,0.19999999999999996,20,3.248658538497605
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\mlp\metadata.txt`
```text
Base model:
DeepSeek-R1-Distill-Qwen-7B

Dataset:
Countdown

Transition:
mlp

Trajectory source:
shared_teacher_hidden_states

Seed:
42

```


## File: `reports\mlp\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\mlp\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.047619,63,-0.005390,0.100628
oracle1,probeA_per_label,all,0.476190,63,,
oracle1,probeA_jaccard,all,0.366402,63,,

```


## File: `reports\mlp\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.047619 |
| probeA_exact_ci_lo | -0.005390 |
| probeA_exact_ci_hi | 0.100628 |
| probeA_per_label | 0.476190 |
| probeA_jaccard | 0.366402 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\mlp\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.891 | 0.350 | 0.150 | 0.475 | 0.562 | 0.7484 |
| 2 | 0.878 | 0.529 | 0.588 | 0.463 | 0.512 | 0.7892 |
| 3 | 0.850 | 0.167 | 0.667 | 0.426 | 0.471 | 0.9962 |
| 4 | 0.826 | nan | nan | 0.292 | 0.417 | 1.3688 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **2**; rollout state probe accuracy stays >= 0.50 through depth **0**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~2 step(s)**.

Single-step transition validation MSE: 1.2362.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\mlp\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.891 | 0.883 | +0.109 |
| 2 | 1.000 | 0.878 | 0.873 | +0.122 |
| 3 | 1.000 | 0.850 | 0.854 | +0.150 |
| 4 | 1.000 | 0.826 | 0.842 | +0.174 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.475 | 0.562 | +0.088 |
| 2 | 0.512 | 0.463 | 0.438 | +0.050 |
| 3 | 0.471 | 0.426 | 0.471 | +0.044 |
| 4 | 0.417 | 0.292 | 0.250 | +0.125 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.500 | -0.200 |
| 2 | 0.588 | 0.529 | 0.588 | +0.059 |
| 3 | 0.667 | 0.167 | 0.333 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.071 | n/a | +0.429 |
| 2 | +0.040 | +0.333 | n/a |
| 3 | -0.027 | n/a | -0.500 |
| 4 | -0.098 | +0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.891 | 0.883 | 0.287 | 0.562 | 0.475 | 0.562 | 0.600 |
| 2 | 1.000 | 0.878 | 0.873 | 0.246 | 0.512 | 0.463 | 0.438 | 0.487 |
| 3 | 1.000 | 0.850 | 0.854 | 0.241 | 0.471 | 0.426 | 0.471 | 0.456 |
| 4 | 1.000 | 0.826 | 0.842 | 0.237 | 0.417 | 0.292 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1387
**Average Oracle Gain (state probe):** +0.0767
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\mlp\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\mlp\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\mlp\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2860894203186035,2.8268252513447747
1,1,2.8235669136047363,2.5399335016969773
2,2,2.479604959487915,2.287418803230661
3,3,2.1677184104919434,2.0666874119492826
4,4,1.8921501636505127,1.891662347512167
5,5,1.6696058511734009,1.7737185368772412
6,6,1.5132672786712646,1.695237331703061
7,7,1.407021403312683,1.6423888910012168
8,8,1.3352855443954468,1.607467776439229
9,9,1.2848303318023682,1.5818846655673668
10,10,1.2426918745040894,1.5580994652920082
11,11,1.2009459733963013,1.5331010662141393
12,12,1.1574300527572632,1.5049117041415856
13,13,1.1120840311050415,1.474302448210169
14,14,1.0662240982055664,1.4458985875864498
15,15,1.0219244956970215,1.4223707855724899
16,16,0.9818803668022156,1.4025713811155225
17,17,0.946262538433075,1.3856271212218239
18,18,0.9139527678489685,1.3715580174180328
19,19,0.8840715289115906,1.3589980328669313
20,20,0.8559772372245789,1.3463562512006917
21,21,0.8296536207199097,1.333921338691086
22,22,0.8038020730018616,1.3232105442735016
23,23,0.7787665128707886,1.3139395791976178
24,24,0.7550327777862549,1.3052722117939934
25,25,0.7326739430427551,1.2970955020091572
26,26,0.7114230990409851,1.2896473368660348
27,27,0.6914528608322144,1.283508801069416
28,28,0.6729753613471985,1.2781209476658555
29,29,0.6553851366043091,1.273574704029521
30,30,0.6385750770568848,1.270230652856045
31,31,0.622447669506073,1.2662205930616035
32,32,0.6067749261856079,1.2620792076235912
33,33,0.5917951464653015,1.2595570048347848
34,34,0.5772501230239868,1.2570265472912399
35,35,0.5633066296577454,1.2538773583584144
36,36,0.5494410395622253,1.2512882420274078
37,37,0.5361084938049316,1.2501589665647412
38,38,0.5234768390655518,1.248962902631916
39,39,0.5110383629798889,1.2478129902824027
40,40,0.4993172585964203,1.2457301655753714
41,41,0.48763319849967957,1.2451432024846312
42,42,0.476719468832016,1.2450927984519082
43,43,0.46553027629852295,1.2448505339075306
44,44,0.4550594389438629,1.2446629258452868
45,45,0.44468921422958374,1.2442852082799694
46,46,0.434637188911438,1.2434489766105277
47,47,0.4248923063278198,1.2435096365506533
48,48,0.4155609905719757,1.2432376048603997
49,49,0.40643298625946045,1.243287383532915
50,50,0.3977663516998291,1.2435342757428278
51,51,0.38924598693847656,1.2427245593461833
52,52,0.3809567987918854,1.2432110895876025
53,53,0.3728511333465576,1.2423295818391393
54,54,0.36488381028175354,1.2423050677190062
55,55,0.35713714361190796,1.2432991403048155
56,56,0.34967708587646484,1.243346667680584
57,57,0.34246230125427246,1.244507836513832
58,58,0.33541643619537354,1.2426042400422643
59,59,0.3286496698856354,1.24374264576396
60,60,0.32190829515457153,1.2436160728579662
61,61,0.31586429476737976,1.2442096647669056
62,62,0.30906400084495544,1.2431131581791113
63,63,0.3031068444252014,1.2402286216860912
64,64,0.29756584763526917,1.2452470122790726
65,65,0.29259437322616577,1.2406448614401895
66,66,0.28838568925857544,1.246793903288294
67,67,0.2840910851955414,1.2403324314805328
68,68,0.2777314782142639,1.2401520775966957
69,69,0.27103734016418457,1.2433200273357454
70,70,0.2669513523578644,1.2385226390400872
71,71,0.26266640424728394,1.2405020291688011
72,72,0.2584417462348938,1.2384911208856302
73,73,0.2529471516609192,1.2388396966652793
74,74,0.24895800650119781,1.2417244833023822
75,75,0.245707705616951,1.237302686347336
76,76,0.241787850856781,1.2371886206454918
77,77,0.2373896837234497,1.2383218984134863
78,78,0.2338113635778427,1.2381469226274333
79,79,0.23031190037727356,1.2391384937724128
80,80,0.22740471363067627,1.2372586609887295
81,81,0.22416995465755463,1.2389278724545338
82,82,0.2205807864665985,1.2368033987576845
83,83,0.2172020822763443,1.236862057545146
84,84,0.21438899636268616,1.237670148005251
85,85,0.21179735660552979,1.236465829317687
86,86,0.20920449495315552,1.239199028640497
87,87,0.20646829903125763,1.2359863031105918
88,88,0.20374558866024017,1.2368879474577357
89,89,0.2010079175233841,1.2367993964523565
90,90,0.19844116270542145,1.2357833111872438
91,91,0.1961854100227356,1.2372263924020235
92,92,0.19394166767597198,1.2361372650646774
93,93,0.19180138409137726,1.2361800397028688
94,94,0.18985125422477722,1.236650685795018
95,95,0.188010573387146,1.2375080546394723
96,96,0.1862826943397522,1.2356219682537142
97,97,0.18451188504695892,1.2375444506035476
98,98,0.18280598521232605,1.236345009725602
99,99,0.1812448799610138,1.2362313192398822

```


## File: `reports\mlp\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.298072338104248,2.820525622758709
1,1,2.831045389175415,2.5535778608478483
2,2,2.5153040885925293,2.3116922847560195
3,3,2.2176334857940674,2.1106429803566855
4,4,1.9601647853851318,1.9379337498399078
5,5,1.7348240613937378,1.8108320392546107
6,6,1.5634936094284058,1.7272300094854636
7,7,1.4468063116073608,1.6715333031826332
8,8,1.369111180305481,1.6332287397540983
9,9,1.313273310661316,1.6033665391265368
10,10,1.2656495571136475,1.5769520743948515
11,11,1.22067391872406,1.551985068399398
12,12,1.177693486213684,1.52496087746542
13,13,1.1343387365341187,1.495498782298604
14,14,1.0907347202301025,1.4662310490842725
15,15,1.048497200012207,1.4418590107902152
16,16,1.010654330253601,1.422870448378266
17,17,0.9762758612632751,1.407146391321401
18,18,0.9442514777183533,1.393816463282851
19,19,0.9147214293479919,1.3809714395491803
20,20,0.8872340321540833,1.3693900186507428
21,21,0.8612146973609924,1.359508952156442
22,22,0.8360181450843811,1.3498764038085938
23,23,0.8113843202590942,1.341232925164895
24,24,0.7876076698303223,1.333282970991291
25,25,0.7646743655204773,1.3258912133388832
26,26,0.7433972358703613,1.3193889680455944
27,27,0.7234256267547607,1.3127480178582864
28,28,0.704049825668335,1.3065405673668034
29,29,0.6853256225585938,1.3019439196977458
30,30,0.6675612330436707,1.2978237965067878
31,31,0.6506237983703613,1.2941630629242444
32,32,0.6345640420913696,1.2918516065253587
33,33,0.6193843483924866,1.2899398803710938
34,34,0.6044599413871765,1.288925546114562
35,35,0.5901724100112915,1.2873455110143444
36,36,0.5763458609580994,1.2855174580558402
37,37,0.562940776348114,1.2832474005026895
38,38,0.550082802772522,1.2821495181224385
39,39,0.5380381345748901,1.2829898771692494
40,40,0.5264869332313538,1.2836773981813525
41,41,0.5155299305915833,1.2838289854956455
42,42,0.5048859119415283,1.2842882500320185
43,43,0.494689404964447,1.2841100223728867
44,44,0.48490098118782043,1.284792290359247
45,45,0.4753143787384033,1.2864339859759222
46,46,0.4662177860736847,1.2866981381275615
47,47,0.4575521945953369,1.2870094424388447
48,48,0.4491352140903473,1.288436764576396
49,49,0.44092684984207153,1.2891990786693135
50,50,0.4329559803009033,1.2895122590612194
51,51,0.4250967502593994,1.289185821032915
52,52,0.41745761036872864,1.289033983574539
53,53,0.41005444526672363,1.2895667904713115
54,54,0.40293803811073303,1.2907442186699538
55,55,0.39594289660453796,1.2910059944528047
56,56,0.38917338848114014,1.290312720126793
57,57,0.3826027512550354,1.2903942670978483
58,58,0.3762497007846832,1.2906785558481686
59,59,0.370102196931839,1.2912395039542777
60,60,0.36408528685569763,1.29101812644083
61,61,0.3583381474018097,1.2935936099193135
62,62,0.35283833742141724,1.2940463707095287
63,63,0.3472714126110077,1.2931092059026
64,64,0.3421444594860077,1.294862090564165
65,65,0.33683520555496216,1.2956010161853226
66,66,0.3319152891635895,1.2946427142033812
67,67,0.32695746421813965,1.2951407510726178
68,68,0.322202205657959,1.297025086449795
69,69,0.31759166717529297,1.2969320328509222
70,70,0.31302690505981445,1.297300745229252
71,71,0.30869317054748535,1.2973417688588627
72,72,0.3043747544288635,1.2977976564501152
73,73,0.30022311210632324,1.2989139244204662
74,74,0.2961629033088684,1.3006709364594007
75,75,0.2922491431236267,1.3013225617955944
76,76,0.2884030044078827,1.3016184822457735
77,77,0.2847006022930145,1.3024046850986168
78,78,0.2810458242893219,1.3046464763703893
79,79,0.2775866389274597,1.303864525966957
80,80,0.274138480424881,1.3054864602010758
81,81,0.2708137035369873,1.3052870953669313
82,82,0.2675904929637909,1.306672143154457
83,83,0.26451531052589417,1.3087682254978867
84,84,0.2617682218551636,1.307098638815958
85,85,0.2597951591014862,1.316053171626857
86,86,0.2589995265007019,1.3109079579837988
87,87,0.25724583864212036,1.3114466432665215
88,88,0.25219544768333435,1.312773157338627
89,89,0.24776425957679749,1.3120946415135117
90,90,0.2471930831670761,1.315836796995069
91,91,0.24511165916919708,1.3140637757348232
92,92,0.24056439101696014,1.3142302466220543
93,93,0.238830104470253,1.3175141381435707
94,94,0.2375689148902893,1.3147855664862962
95,95,0.23409679532051086,1.3175531606205175
96,96,0.23187609016895294,1.3190355144563268
97,97,0.23060977458953857,1.3156050385021774
98,98,0.22807595133781433,1.3219431892770235
99,99,0.2257753610610962,1.3216002417392418

```


## File: `reports\mlp_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7347442373633385,0.8936283409595489,0.35,0.15,0.5375,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.0625,-0.09999999999999998,0.19999999999999996,20,7.4130308712642305
2,0.8018624752759933,0.8764720916748047,0.47058823529411764,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,7.001500715086652
3,0.984418462304508,0.8528834230759564,0.0,0.6666666666666666,0.5294117647058824,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.0735294117647059,0.35294117647058826,0.2941176470588235,17,5.762867203607013
4,1.3627923528353374,0.8289438883463541,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,4.208299312517709
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8066168993711471,0.8823344320058822,0.5,0.15,0.55,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.09999999999999998,0.25,20,6.752501365027502
2,0.8417939931154251,0.8693173736333847,0.47058823529411764,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.669376046825992
3,0.9992016764248118,0.8470361408065347,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,5.677605437311118
4,1.1104228794574738,0.8453614215056101,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,5.164733389088078
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8737493276596069,0.8733441323041916,0.2,0.45,0.4625,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.03750000000000003,-0.05,-0.2,20,6.373022314659121
2,1.2112935304641723,0.8118271827697754,0.45,0.45,0.4,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.050000000000000044,-0.05,-0.2,20,4.820632688304592
3,1.326008415222168,0.8005892276763916,0.35,0.3,0.45,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.21250000000000002,-0.9,0.19999999999999996,20,4.29911556977795
4,2.3214591085910796,0.6903035998344421,0.35,0.55,0.325,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.15000000000000002,1.0,0.19999999999999996,20,2.4998032267485693
5,1.601481169462204,0.8301547884941101,,,0.325,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.21250000000000002,0.0,0.19999999999999996,20,3.5932896700139763
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\mlp_seed42\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\mlp_seed42\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.095238,63,0.022171,0.168306
oracle1,probeA_per_label,all,0.507937,63,,
oracle1,probeA_jaccard,all,0.370370,63,,

```


## File: `reports\mlp_seed42\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.095238 |
| probeA_exact_ci_lo | 0.022171 |
| probeA_exact_ci_hi | 0.168306 |
| probeA_per_label | 0.507937 |
| probeA_jaccard | 0.370370 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\mlp_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.894 | 0.350 | 0.150 | 0.537 | 0.562 | 0.7347 |
| 2 | 0.876 | 0.471 | 0.588 | 0.463 | 0.512 | 0.8019 |
| 3 | 0.853 | 0.000 | 0.667 | 0.529 | 0.471 | 0.9844 |
| 4 | 0.829 | nan | nan | 0.208 | 0.417 | 1.3628 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2394.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\mlp_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.882 | +0.106 |
| 2 | 1.000 | 0.876 | 0.869 | +0.124 |
| 3 | 1.000 | 0.853 | 0.847 | +0.147 |
| 4 | 1.000 | 0.829 | 0.845 | +0.171 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.537 | 0.550 | +0.025 |
| 2 | 0.512 | 0.463 | 0.463 | +0.050 |
| 3 | 0.471 | 0.529 | 0.500 | -0.059 |
| 4 | 0.417 | 0.208 | 0.333 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.500 | -0.200 |
| 2 | 0.588 | 0.471 | 0.471 | +0.118 |
| 3 | 0.667 | 0.000 | 0.333 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.096 | -1.000 | +0.429 |
| 2 | +0.055 | +0.000 | +0.000 |
| 3 | +0.038 | -1.000 | -1.000 |
| 4 | -0.106 | -1.500 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.882 | 0.287 | 0.562 | 0.537 | 0.550 | 0.600 |
| 2 | 1.000 | 0.876 | 0.869 | 0.246 | 0.512 | 0.463 | 0.463 | 0.487 |
| 3 | 1.000 | 0.853 | 0.847 | 0.241 | 0.471 | 0.529 | 0.500 | 0.456 |
| 4 | 1.000 | 0.829 | 0.845 | 0.237 | 0.417 | 0.208 | 0.333 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1370
**Average Oracle Gain (state probe):** +0.0561
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\mlp_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\mlp_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\mlp_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.3286380767822266,2.8200210821433145
1,1,2.822308301925659,2.532857676021388
2,2,2.4865009784698486,2.2707947277631915
3,3,2.1637978553771973,2.0652893566694415
4,4,1.8978708982467651,1.9022119240682633
5,5,1.685105562210083,1.788270793977331
6,6,1.5316882133483887,1.7135492543705175
7,7,1.4291799068450928,1.6671230128554047
8,8,1.3625636100769043,1.63485592701396
9,9,1.3135594129562378,1.6102877757588372
10,10,1.2715473175048828,1.5859862780961833
11,11,1.2296808958053589,1.5566886526639345
12,12,1.1840994358062744,1.522563371501985
13,13,1.134629487991333,1.4902311231269212
14,14,1.085972547531128,1.4629379022316855
15,15,1.0411717891693115,1.4384310362768955
16,16,0.9998141527175903,1.4147501460841445
17,17,0.9624283313751221,1.3938870039142546
18,18,0.9286254644393921,1.3794957145315703
19,19,0.8986238837242126,1.3688110601706582
20,20,0.8705886006355286,1.3580372294441598
21,21,0.8429640531539917,1.3458392033811475
22,22,0.8156896829605103,1.3334703289094518
23,23,0.7884190082550049,1.321626256723873
24,24,0.7616121768951416,1.3122648645619877
25,25,0.7365407347679138,1.3043112832991803
26,26,0.7136057615280151,1.2947101280337474
27,27,0.6920283436775208,1.285337604460169
28,28,0.672311007976532,1.2786602583087858
29,29,0.6537232398986816,1.2741047593413806
30,30,0.6362932324409485,1.268143575699603
31,31,0.6190634965896606,1.2633259257332223
32,32,0.6026971936225891,1.2608092261142418
33,33,0.5868134498596191,1.2584608734631149
34,34,0.5714294910430908,1.2549393450627562
35,35,0.5568460822105408,1.2520689417104252
36,36,0.5429412722587585,1.2530139860559681
37,37,0.5297497510910034,1.250976812644083
38,38,0.5169343948364258,1.2477204369716957
39,39,0.5044084191322327,1.2478670214043288
40,40,0.4924579858779907,1.2487272669057377
41,41,0.48059773445129395,1.2463411424980788
42,42,0.46915289759635925,1.2442274249967982
43,43,0.4581061899662018,1.2426917904713115
44,44,0.4472413659095764,1.2428336221663678
45,45,0.43683627247810364,1.2438897304847591
46,46,0.42663806676864624,1.2433075201315957
47,47,0.4167173206806183,1.2425564625224128
48,48,0.40709173679351807,1.243427339147349
49,49,0.3978527784347534,1.2414175565125511
50,50,0.3886769413948059,1.2413052418192878
51,51,0.3801364302635193,1.2430274838306865
52,52,0.3714349865913391,1.2431973316630378
53,53,0.36335188150405884,1.2411706643026383
54,54,0.3551316261291504,1.2413422631435707
55,55,0.3472093641757965,1.241457079277664
56,56,0.3393828570842743,1.239470935258709
57,57,0.33186987042427063,1.239076958327997
58,58,0.32452377676963806,1.23876702980917
59,59,0.3175528049468994,1.238388186595479
60,60,0.31072574853897095,1.2366518114433913
61,61,0.3041490614414215,1.2359255180984248
62,62,0.2977372705936432,1.2366096621654072
63,63,0.2916161119937897,1.233853949875128
64,64,0.28562599420547485,1.2368701872278431
65,65,0.2800111472606659,1.2335739135742188
66,66,0.2745484411716461,1.2367885151847464
67,67,0.2696550488471985,1.2330977643122438
68,68,0.26521196961402893,1.2389825289366676
69,69,0.26116156578063965,1.2339420005923412
70,70,0.2559581398963928,1.2349073066086065
71,71,0.24997419118881226,1.2345553538838372
72,72,0.2448352873325348,1.234252179255251
73,73,0.24141639471054077,1.2371562269867444
74,74,0.23803913593292236,1.233019093998143
75,75,0.23344695568084717,1.2340205458344007
76,76,0.22871318459510803,1.2353480604828382
77,77,0.2252802550792694,1.2333218934106045
78,78,0.22264160215854645,1.237609112849001
79,79,0.21927551925182343,1.234205777527856
80,80,0.21541379392147064,1.2341148501536885
81,81,0.21191100776195526,1.2387018672755508
82,82,0.20940591394901276,1.23274543637135
83,83,0.20695465803146362,1.2399259473456712
84,84,0.20402419567108154,1.2343644939485143
85,85,0.20088940858840942,1.2367733814677253
86,86,0.1982194036245346,1.2373622206390882
87,87,0.1961938440799713,1.2376326263928024
88,88,0.19416795670986176,1.2375575831679047
89,89,0.19178195297718048,1.2390460655337474
90,90,0.1893179714679718,1.2346715458103867
91,91,0.18729571998119354,1.24310302734375
92,92,0.18549755215644836,1.2350673988217213
93,93,0.18310114741325378,1.2400385121830175
94,94,0.18049706518650055,1.238004840788294
95,95,0.17832070589065552,1.2362403244268698
96,96,0.17675936222076416,1.2427480728899847
97,97,0.1752738654613495,1.2359999359631149
98,98,0.1735161691904068,1.2399066862512806
99,99,0.17164920270442963,1.239399268978932

```


## File: `reports\mlp_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2405190467834473,2.794680736103996
1,1,2.7928333282470703,2.4885368972528177
2,2,2.42737078666687,2.212135940301614
3,3,2.091857433319092,2.0040733462474387
4,4,1.8312478065490723,1.8478599923555967
5,5,1.6297669410705566,1.7389591404649078
6,6,1.4853793382644653,1.6755756315637806
7,7,1.396022081375122,1.6373986416175716
8,8,1.33942449092865,1.6117613745517418
9,9,1.2982933521270752,1.5905586617891905
10,10,1.2610399723052979,1.5650139480340677
11,11,1.2188657522201538,1.534974895539831
12,12,1.1723062992095947,1.5049478499615778
13,13,1.1265127658843994,1.476790256187564
14,14,1.0837352275848389,1.4530211902055583
15,15,1.0444492101669312,1.4330299252369365
16,16,1.008631944656372,1.415619646916624
17,17,0.9764536619186401,1.4009455696481172
18,18,0.9477985501289368,1.3877483430455944
19,19,0.9202165007591248,1.3749718587906634
20,20,0.8930166959762573,1.3633751791031634
21,21,0.8663605451583862,1.3529949500912526
22,22,0.8407472968101501,1.3428414766905739
23,23,0.8155000805854797,1.33295765861136
24,24,0.7906317710876465,1.3242677782402663
25,25,0.7677454948425293,1.3154847191982582
26,26,0.7462475299835205,1.307862078557249
27,27,0.7262309193611145,1.3015667024205944
28,28,0.7076115608215332,1.2958824282786885
29,29,0.6898286938667297,1.2908935546875
30,30,0.6730250120162964,1.2861158027023565
31,31,0.6564818620681763,1.2825576281938396
32,32,0.6406599879264832,1.2797592663374104
33,33,0.6254147887229919,1.2771754030321465
34,34,0.6105501651763916,1.274505865378458
35,35,0.5962384939193726,1.272249315605789
36,36,0.5829029083251953,1.2680607780081328
37,37,0.5696391463279724,1.2664822437724128
38,38,0.5574650168418884,1.2658346207415472
39,39,0.5452470183372498,1.2654221331486937
40,40,0.5337750315666199,1.2640478415567367
41,41,0.5222623944282532,1.2637605510774206
42,42,0.5112267732620239,1.2640709798844134
43,43,0.5006636381149292,1.2656209976946722
44,44,0.4901063144207001,1.266143048395876
45,45,0.479924738407135,1.2657198046074538
46,46,0.47006601095199585,1.266724133100666
47,47,0.4603712856769562,1.2682244973104508
48,48,0.45103731751441956,1.2681316938556608
49,49,0.44188371300697327,1.2678790483318392
50,50,0.4330585300922394,1.2685929595446976
51,51,0.4245341420173645,1.2713280349481302
52,52,0.41640913486480713,1.2713833167904713
53,53,0.40833142399787903,1.2708269963498975
54,54,0.40075862407684326,1.272166142698194
55,55,0.3931291401386261,1.2737696913422132
56,56,0.3860588073730469,1.2737154100762038
57,57,0.3789764642715454,1.274078619284708
58,58,0.37203365564346313,1.275452910876665
59,59,0.3654744625091553,1.2766623575179303
60,60,0.3589402139186859,1.2765680531986425
61,61,0.3526070713996887,1.2768387090964395
62,62,0.3466321527957916,1.2794422087122181
63,63,0.3406638503074646,1.2794131919985912
64,64,0.3349045217037201,1.2792320876825052
65,65,0.3294200003147125,1.2798877153240267
66,66,0.32403501868247986,1.280866404048732
67,67,0.31887373328208923,1.2819263896004098
68,68,0.31383445858955383,1.281840214963819
69,69,0.308988094329834,1.2826345474993597
70,70,0.3042106330394745,1.2838612540823515
71,71,0.2996130585670471,1.284074752057185
72,72,0.2952089011669159,1.2858684102042777
73,73,0.2909119427204132,1.2862093565893955
74,74,0.28701576590538025,1.2880926913902409
75,75,0.28295668959617615,1.2876466844902663
76,76,0.27926164865493774,1.2897012429159196
77,77,0.2756378948688507,1.2888409974145107
78,78,0.27218878269195557,1.2904890717053024
79,79,0.2688278555870056,1.2940098496734118
80,80,0.2655235230922699,1.2896670982485912
81,81,0.2621432840824127,1.298992094446401
82,82,0.25903162360191345,1.2902274209944928
83,83,0.2558375597000122,1.298401879482582
84,84,0.2527175545692444,1.297179425349001
85,85,0.24956609308719635,1.2970408455270235
86,86,0.24656997621059418,1.3004915831518955
87,87,0.24399766325950623,1.2961104346103356
88,88,0.24152140319347382,1.3045327858846696
89,89,0.23891358077526093,1.298828125
90,90,0.23602648079395294,1.3026189335056992
91,91,0.2332375943660736,1.3049835455222207
92,92,0.23101991415023804,1.3010824234759222
93,93,0.2288672775030136,1.3077194964299437
94,94,0.22696077823638916,1.3032997006275615
95,95,0.22502298653125763,1.3106660686555456
96,96,0.22302943468093872,1.3062036232870133
97,97,0.22128517925739288,1.3102044277503841
98,98,0.2194022536277771,1.3118972778320312
99,99,0.2173655778169632,1.3082535540471312

```


## File: `reports\mlp_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7458958730101586,0.8920401424169541,0.35,0.15,0.5125,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.08750000000000002,-0.09999999999999998,0.19999999999999996,20,7.302201166601363
2,0.7785200327634811,0.8802536100149154,0.47058823529411764,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,7.211427397850479
3,1.0271344780921936,0.8475425979670357,0.16666666666666666,0.6666666666666666,0.5294117647058824,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.0735294117647059,0.35294117647058826,0.2941176470588235,17,5.5232036233240835
4,1.3778541286786397,0.8283084134260813,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,4.16229701110758
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8024085536599159,0.8828462392091752,0.5,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.787915818213453
2,0.8238886773586274,0.8726117163896561,0.5294117647058824,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.814319517104153
3,0.9349751139388365,0.8568151663331425,0.5,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,6.067619112492242
4,0.994559109210968,0.8592170079549154,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,5.766412542429203
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8757711470127105,0.8729764848947525,0.25,0.45,0.4375,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.012500000000000011,-0.05,-0.2,20,6.358309452860133
2,1.1886647909879684,0.8161162853240966,0.45,0.45,0.3625,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.012500000000000011,-0.05,-0.2,20,4.912403591288477
3,1.425722175836563,0.788203838467598,0.45,0.3,0.45,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.21250000000000002,-0.9,0.19999999999999996,20,3.998439191137124
4,2.3627329766750336,0.6889656037092209,0.45,0.55,0.375,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.2,1.0,0.19999999999999996,20,2.456134919904239
5,1.758066976070404,0.8156131237745285,,,0.3375,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.22500000000000003,0.0,0.19999999999999996,20,3.273246026049004
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\mlp_seed43\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\mlp_seed43\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.079365,63,0.012081,0.146649
oracle1,probeA_per_label,all,0.500000,63,,
oracle1,probeA_jaccard,all,0.359788,63,,

```


## File: `reports\mlp_seed43\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.079365 |
| probeA_exact_ci_lo | 0.012081 |
| probeA_exact_ci_hi | 0.146649 |
| probeA_per_label | 0.500000 |
| probeA_jaccard | 0.359788 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\mlp_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.350 | 0.150 | 0.512 | 0.562 | 0.7459 |
| 2 | 0.880 | 0.471 | 0.588 | 0.450 | 0.512 | 0.7785 |
| 3 | 0.848 | 0.167 | 0.667 | 0.529 | 0.471 | 1.0271 |
| 4 | 0.828 | nan | nan | 0.250 | 0.417 | 1.3779 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2495.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\mlp_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | +0.108 |
| 2 | 1.000 | 0.880 | 0.873 | +0.120 |
| 3 | 1.000 | 0.848 | 0.857 | +0.152 |
| 4 | 1.000 | 0.828 | 0.859 | +0.172 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.512 | 0.562 | +0.050 |
| 2 | 0.512 | 0.450 | 0.463 | +0.062 |
| 3 | 0.471 | 0.529 | 0.471 | -0.059 |
| 4 | 0.417 | 0.250 | 0.333 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.500 | -0.200 |
| 2 | 0.588 | 0.471 | 0.529 | +0.118 |
| 3 | 0.667 | 0.167 | 0.500 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.078 | n/a | +0.429 |
| 2 | +0.060 | -0.250 | -1.000 |
| 3 | -0.065 | n/a | -2.000 |
| 4 | -0.220 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | 0.287 | 0.562 | 0.512 | 0.562 | 0.600 |
| 2 | 1.000 | 0.880 | 0.873 | 0.246 | 0.512 | 0.450 | 0.463 | 0.487 |
| 3 | 1.000 | 0.848 | 0.857 | 0.241 | 0.471 | 0.529 | 0.471 | 0.456 |
| 4 | 1.000 | 0.828 | 0.859 | 0.237 | 0.417 | 0.250 | 0.333 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1380
**Average Oracle Gain (state probe):** +0.0551
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\mlp_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\mlp_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\mlp_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.267935276031494,2.799407208552126
1,1,2.8100833892822266,2.5357465900358607
2,2,2.504091262817383,2.292244582879739
3,3,2.205570697784424,2.085326398005251
4,4,1.9382981061935425,1.9261470857213756
5,5,1.7243086099624634,1.8106070346519596
6,6,1.5651999711990356,1.7279117771836578
7,7,1.4535584449768066,1.6702180455942623
8,8,1.3772187232971191,1.6324950671586833
9,9,1.3218551874160767,1.6062101770619877
10,10,1.2764309644699097,1.5807352535060195
11,11,1.2321298122406006,1.5515214263415726
12,12,1.186069369316101,1.5202701756211578
13,13,1.1405175924301147,1.490949662005315
14,14,1.096422553062439,1.4654242093445824
15,15,1.0545504093170166,1.4440079986071976
16,16,1.0159069299697876,1.4251806540567367
17,17,0.9805086851119995,1.4081278316310195
18,18,0.9469848275184631,1.392439670250064
19,19,0.91559237241745,1.3786042010197874
20,20,0.8862864971160889,1.3663832867731813
21,21,0.8583267331123352,1.354065941982582
22,22,0.8309171795845032,1.3425011556656634
23,23,0.8043002486228943,1.330180058713819
24,24,0.7792468667030334,1.319351071217021
25,25,0.755526065826416,1.3096055828157018
26,26,0.732944667339325,1.299209719798604
27,27,0.7108396887779236,1.291108678598873
28,28,0.6902452707290649,1.2848775894915472
29,29,0.6710281372070312,1.2789164058497695
30,30,0.6526522636413574,1.2735082907754867
31,31,0.6355149149894714,1.2697513767930328
32,32,0.6189554333686829,1.2658806472528177
33,33,0.6024667620658875,1.261566412253458
34,34,0.5873024463653564,1.2582421224625384
35,35,0.5723830461502075,1.2562509755619238
36,36,0.5586609840393066,1.2538424632588372
37,37,0.5452715158462524,1.2513371451956328
38,38,0.5326074361801147,1.2492713302862448
39,39,0.5203024744987488,1.2477462018122438
40,40,0.5083506107330322,1.246978509621542
41,41,0.49694910645484924,1.246173420890433
42,42,0.4857181906700134,1.2437666595959274
43,43,0.4745115339756012,1.2425114365874743
44,44,0.46448275446891785,1.243164312644083
45,45,0.4540145695209503,1.2432386054367315
46,46,0.4440467059612274,1.242246534003586
47,47,0.4346367120742798,1.2415581374871927
48,48,0.4252658486366272,1.241130516177318
49,49,0.41618725657463074,1.2407920712330303
50,50,0.40744996070861816,1.2413232521932633
51,51,0.39889565110206604,1.2402568879674694
52,52,0.3907041549682617,1.241135018770812
53,53,0.38263869285583496,1.240007119100602
54,54,0.3748253583908081,1.2407776879482582
55,55,0.36704373359680176,1.2403444383965163
56,56,0.35969075560569763,1.238801799836706
57,57,0.35239771008491516,1.2401160568487448
58,58,0.34539565443992615,1.2387197525774847
59,59,0.33853238821029663,1.2378735151447233
60,60,0.3318410813808441,1.2377579485783812
61,61,0.3255474865436554,1.2374630287045338
62,62,0.3195103108882904,1.2378049756659837
63,63,0.3141390383243561,1.2372989341860912
64,64,0.3097009062767029,1.2414883472880378
65,65,0.30538854002952576,1.2361555255827357
66,66,0.29864099621772766,1.2385821733318392
67,67,0.29115352034568787,1.2384135762199027
68,68,0.28594163060188293,1.2382652407786885
69,69,0.2826154828071594,1.242204134581519
70,70,0.27707570791244507,1.236909960137039
71,71,0.2708207964897156,1.2378600073642418
72,72,0.2666938602924347,1.2423856141137295
73,73,0.26317134499549866,1.2384528488409323
74,74,0.2581467628479004,1.2396773041271774
75,75,0.25304052233695984,1.2412688458552126
76,76,0.2496214061975479,1.2395993842453252
77,77,0.24618428945541382,1.2426141207335426
78,78,0.24178142845630646,1.2413685282722848
79,79,0.23740436136722565,1.2406143438620645
80,80,0.2342340648174286,1.2441187373927383
81,81,0.2313271015882492,1.2416783317190703
82,82,0.22773794829845428,1.2417542504482582
83,83,0.2238844782114029,1.2435516607565957
84,84,0.2205846756696701,1.2425882308209528
85,85,0.21790176630020142,1.2444375460265114
86,86,0.21517518162727356,1.2437460227090804
87,87,0.21214886009693146,1.2446070186427383
88,88,0.20907141268253326,1.246092249135502
89,89,0.2062399983406067,1.2444805708087858
90,90,0.20364108681678772,1.2473262098969007
91,91,0.20125585794448853,1.246968378786181
92,92,0.19900120794773102,1.2477154340900358
93,93,0.19677478075027466,1.2481266709624743
94,94,0.1944376528263092,1.2478437580046107
95,95,0.19204235076904297,1.2488293256915983
96,96,0.18963263928890228,1.2494543106829534
97,97,0.1873469203710556,1.2486576017786244
98,98,0.1852744072675705,1.2510109573114114
99,99,0.18325619399547577,1.2494862040535348

```


## File: `reports\mlp_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.302522897720337,2.8367414630827357
1,1,2.84743070602417,2.580047607421875
2,2,2.546755313873291,2.34133035628522
3,3,2.2548840045928955,2.1349347224001023
4,4,1.994018316268921,1.9655624139504355
5,5,1.7726194858551025,1.8301836858030225
6,6,1.592383861541748,1.7309650358606556
7,7,1.4607285261154175,1.6672763511782787
8,8,1.375197410583496,1.6309141565541752
9,9,1.3200438022613525,1.6095008224737448
10,10,1.27828049659729,1.590879596647669
11,11,1.2407196760177612,1.5691592107053662
12,12,1.2030752897262573,1.543995716532723
13,13,1.1617722511291504,1.51684320168417
14,14,1.1173299551010132,1.490386462602459
15,15,1.0733320713043213,1.4630828607277793
16,16,1.0308893918991089,1.4392142374007428
17,17,0.9909740686416626,1.420316852507044
18,18,0.954482913017273,1.4053127101210297
19,19,0.9216555953025818,1.3900961954085553
20,20,0.8915392160415649,1.3755855872982838
21,21,0.8630073666572571,1.3636652211673925
22,22,0.8355136513710022,1.35374638291656
23,23,0.8095722794532776,1.3430699833103867
24,24,0.7843092679977417,1.33326283439261
25,25,0.7599762082099915,1.3251765516937757
26,26,0.7364839911460876,1.3175557871333887
27,27,0.7141919136047363,1.3084461649910348
28,28,0.6931869983673096,1.3007744961097591
29,29,0.6733778715133667,1.2961986103995902
30,30,0.6546301245689392,1.2918868768410605
31,31,0.6368712186813354,1.2871330136158428
32,32,0.6199512481689453,1.282590522140753
33,33,0.6038998961448669,1.2795842905513575
34,34,0.5893278121948242,1.2788968946112962
35,35,0.5744129419326782,1.2769049972784323
36,36,0.560420036315918,1.2751852567078636
37,37,0.5467714071273804,1.2745931656634222
38,38,0.5337215662002563,1.2718236954485784
39,39,0.52129727602005,1.2700450459464652
40,40,0.5094190239906311,1.2719836625896517
41,41,0.4981030225753784,1.2715504130379098
42,42,0.4872353672981262,1.2694647116739242
43,43,0.4770296812057495,1.2716277075595543
44,44,0.4666244387626648,1.2717075035220287
45,45,0.4570474922657013,1.2704847992443649
46,46,0.44755813479423523,1.2729462170210042
47,47,0.43840840458869934,1.2747717685386784
48,48,0.42949560284614563,1.2747582607581966
49,49,0.42097988724708557,1.2750192861087988
50,50,0.41268816590309143,1.2744893558689805
51,51,0.4046980142593384,1.2766338410924694
52,52,0.39698606729507446,1.278913529192815
53,53,0.3893657922744751,1.2766778664510758
54,54,0.3818085193634033,1.2776061511430583
55,55,0.3747410178184509,1.280701183881916
56,56,0.36773088574409485,1.2799022236808402
57,57,0.3610393702983856,1.2805423423892162
58,58,0.35454317927360535,1.2828670564245006
59,59,0.3484589755535126,1.2822395699923155
60,60,0.3422846496105194,1.2831344604492188
61,61,0.3365653157234192,1.2843722984439037
62,62,0.33087560534477234,1.2848953497214395
63,63,0.32555481791496277,1.2865628101786628
64,64,0.3202686905860901,1.286388084536693
65,65,0.3152705729007721,1.2859081831134733
66,66,0.3102931082248688,1.289140419881852
67,67,0.3055999279022217,1.2888939028880635
68,68,0.30100685358047485,1.2894368406201973
69,69,0.29662466049194336,1.2902661933273565
70,70,0.2922630310058594,1.2899294993916497
71,71,0.28804531693458557,1.291624225553919
72,72,0.2839744985103607,1.2924674612576845
73,73,0.2800723612308502,1.2926695776767418
74,74,0.2763698697090149,1.293547833552126
75,75,0.2726968824863434,1.2938700191310195
76,76,0.2690700590610504,1.2960327648725667
77,77,0.26560887694358826,1.2960335153048155
78,78,0.2622106075286865,1.2962686507428278
79,79,0.2590506076812744,1.2990227370965677
80,80,0.2560470700263977,1.2988934126056608
81,81,0.253256231546402,1.30154293873271
82,82,0.25080618262290955,1.2971841780865778
83,83,0.24844737350940704,1.3013085537269466
84,84,0.24588678777217865,1.3040477565077484
85,85,0.24273675680160522,1.3010031278016136
86,86,0.23964394629001617,1.3044249737849
87,87,0.23712602257728577,1.3026638343685963
88,88,0.23509949445724487,1.3066133592949538
89,89,0.23284408450126648,1.3050049328413167
90,90,0.23037292063236237,1.3044273501536885
91,91,0.2279542088508606,1.3095583055840163
92,92,0.22587670385837555,1.3050749731845543
93,93,0.22400672733783722,1.3109180888191598
94,94,0.22213207185268402,1.3081820128393955
95,95,0.22002926468849182,1.3094267297963627
96,96,0.21797004342079163,1.3120927654328893
97,97,0.21617864072322845,1.3114362622870774
98,98,0.21451930701732635,1.3139311993708376
99,99,0.21278415620326996,1.3117883400838883

```


## File: `reports\mlp_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7441744193434715,0.8923472166061401,0.35,0.15,0.525,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.19999999999999996,20,7.319092906825681
2,0.8090398460626602,0.8753514796495437,0.4117647058823529,0.5882352941176471,0.4125,0.5125,0.5,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.07500000000000001,0.25,0.30000000000000004,20,6.9393871282962705
3,1.0106042939073898,0.8477605090421789,0.0,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.613545188003888
4,1.1650014718373616,0.8445407748222351,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,4.92277328413708
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8015805408358574,0.8832018584012985,0.45,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.794927566952077
2,0.7997877612709999,0.8764950841665268,0.35294117647058826,0.5882352941176471,0.4875,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,0.0,0.30000000000000004,0.30000000000000004,20,7.019663173044845
3,0.9860276527264539,0.8496432199197657,0.3333333333333333,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,5.753462243531757
4,1.0930899679660797,0.8477148910363516,,,0.375,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.16666666666666666,0.0,0.16666666666666663,6,5.246629545244631
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8742806941270829,0.8734545409679413,0.25,0.45,0.45,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,6.3691489472415
2,1.213216844201088,0.8120377540588379,0.4,0.45,0.3875,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.03750000000000003,-0.05,-0.2,20,4.812990535037138
3,1.3712614834308625,0.7939174950122834,0.35,0.3,0.4375,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.2,-0.9,0.19999999999999996,20,4.157240243688088
4,2.263488394021988,0.6974710017442703,0.35,0.55,0.3625,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.1875,1.0,0.19999999999999996,20,2.5638262540896704
5,1.6364949941635132,0.8230887174606323,,,0.35,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.2375,0.0,0.19999999999999996,20,3.5164090103996126
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\mlp_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\mlp_seed44\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\mlp_seed44\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.095238,63,0.022171,0.168306
oracle1,probeA_per_label,all,0.507937,63,,
oracle1,probeA_jaccard,all,0.350529,63,,

```


## File: `reports\mlp_seed44\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.095238 |
| probeA_exact_ci_lo | 0.022171 |
| probeA_exact_ci_hi | 0.168306 |
| probeA_per_label | 0.507937 |
| probeA_jaccard | 0.350529 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\mlp_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.350 | 0.150 | 0.525 | 0.562 | 0.7442 |
| 2 | 0.875 | 0.412 | 0.588 | 0.412 | 0.512 | 0.8090 |
| 3 | 0.848 | 0.000 | 0.667 | 0.485 | 0.471 | 1.0106 |
| 4 | 0.845 | nan | nan | 0.292 | 0.417 | 1.1650 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2444.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\mlp_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | +0.108 |
| 2 | 1.000 | 0.875 | 0.876 | +0.125 |
| 3 | 1.000 | 0.848 | 0.850 | +0.152 |
| 4 | 1.000 | 0.845 | 0.848 | +0.155 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.525 | 0.562 | +0.037 |
| 2 | 0.512 | 0.412 | 0.487 | +0.100 |
| 3 | 0.471 | 0.485 | 0.515 | -0.015 |
| 4 | 0.417 | 0.292 | 0.375 | +0.125 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.450 | -0.200 |
| 2 | 0.588 | 0.412 | 0.353 | +0.176 |
| 3 | 0.667 | 0.000 | 0.333 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.078 | n/a | +0.333 |
| 2 | -0.009 | -3.000 | +0.250 |
| 3 | -0.013 | +0.667 | -1.000 |
| 4 | -0.021 | -2.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | 0.287 | 0.562 | 0.525 | 0.562 | 0.600 |
| 2 | 1.000 | 0.875 | 0.876 | 0.246 | 0.512 | 0.412 | 0.487 | 0.487 |
| 3 | 1.000 | 0.848 | 0.850 | 0.241 | 0.471 | 0.485 | 0.515 | 0.456 |
| 4 | 1.000 | 0.845 | 0.848 | 0.237 | 0.417 | 0.292 | 0.375 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1350
**Average Oracle Gain (state probe):** +0.0619
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\mlp_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\mlp_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\mlp_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2327730655670166,2.7674490506531764
1,1,2.762316942214966,2.4547309250128073
2,2,2.392094612121582,2.1928180632044056
3,3,2.069675922393799,1.9792100249743851
4,4,1.7987982034683228,1.819063530593622
5,5,1.593103289604187,1.723203565253586
6,6,1.46112859249115,1.6693567995165215
7,7,1.3816152811050415,1.6300441554335297
8,8,1.3252047300338745,1.5987771456358864
9,9,1.2788115739822388,1.573914824939165
10,10,1.2362102270126343,1.5488567664975026
11,11,1.192788004875183,1.5192667226322363
12,12,1.1462290287017822,1.4906343553887038
13,13,1.1009091138839722,1.4653900646772542
14,14,1.0586497783660889,1.443737717925525
15,15,1.0195894241333008,1.4242017151879482
16,16,0.9831456542015076,1.4073966604764345
17,17,0.9498981833457947,1.3925419791800078
18,18,0.9187107682228088,1.3776465243980534
19,19,0.8883231282234192,1.3622031290023053
20,20,0.8591300249099731,1.3479539214587601
21,21,0.8301348090171814,1.3349356729476178
22,22,0.8026717305183411,1.3237272168769212
23,23,0.7762961387634277,1.312744390769083
24,24,0.7510298490524292,1.3019179047131149
25,25,0.7273929715156555,1.2933907430680072
26,26,0.7060506939888,1.2872181876761015
27,27,0.6859888434410095,1.2814723780897797
28,28,0.6672171354293823,1.2746012953461194
29,29,0.6497028470039368,1.270201135854252
30,30,0.6330810785293579,1.2664352166848105
31,31,0.6165367960929871,1.2620186727555072
32,32,0.6008901000022888,1.2575753634093239
33,33,0.5854103565216064,1.2553618384189293
34,34,0.5707365870475769,1.2533776955526383
35,35,0.55644291639328,1.2511087636478613
36,36,0.5432230830192566,1.2490419481621413
37,37,0.5302028656005859,1.2474068813636654
38,38,0.5173761248588562,1.2450524001825052
39,39,0.5048884749412537,1.244936083183914
40,40,0.4931597411632538,1.244417284355789
41,41,0.48125556111335754,1.2426620233254355
42,42,0.46999627351760864,1.2429448112112578
43,43,0.45943814516067505,1.2448147633036628
44,44,0.4487841725349426,1.2448116365026256
45,45,0.43847575783729553,1.244099726442431
46,46,0.4285513460636139,1.2446341592757428
47,47,0.4187465012073517,1.2442837074154713
48,48,0.40924397110939026,1.243010724177126
49,49,0.40019330382347107,1.2428982844118213
50,50,0.39125728607177734,1.2426146210217086
51,51,0.3825504779815674,1.242474540335233
52,52,0.37408506870269775,1.2423370861616292
53,53,0.3658316731452942,1.2411461501825052
54,54,0.3578453063964844,1.2414458227939293
55,55,0.3503054976463318,1.2403011634701588
56,56,0.342835396528244,1.240739791119685
57,57,0.3355486989021301,1.2410849899542136
58,58,0.32848960161209106,1.2402794009349385
59,59,0.3216426372528076,1.240538800349001
60,60,0.3150646686553955,1.2395260920290088
61,61,0.3087066113948822,1.240509533491291
62,62,0.3024384677410126,1.240220241859311
63,63,0.2965830862522125,1.2399264476338372
64,64,0.2906571626663208,1.240631103515625
65,65,0.285004585981369,1.2398013755923412
66,66,0.279649943113327,1.2411916764056097
67,67,0.2744150757789612,1.2410470931256403
68,68,0.26926690340042114,1.2378890240778688
69,69,0.264373242855072,1.2419190953989498
70,70,0.2597942054271698,1.239074206743084
71,71,0.2554694414138794,1.2418750700403431
72,72,0.25109273195266724,1.2393429865602588
73,73,0.24684196710586548,1.2416481893570697
74,74,0.24266692996025085,1.239168511062372
75,75,0.23831576108932495,1.2392330482357838
76,76,0.2337983101606369,1.2398322683865908
77,77,0.22989743947982788,1.2388241877321338
78,78,0.22613948583602905,1.2391581300829277
79,79,0.2227829247713089,1.2406501144659323
80,80,0.22070960700511932,1.2416114181768698
81,81,0.21796658635139465,1.239884798644019
82,82,0.21562953293323517,1.2443941460281123
83,83,0.21139712631702423,1.2369817514888575
84,84,0.20845694839954376,1.2434611085985527
85,85,0.2051420360803604,1.2378490010245902
86,86,0.20181980729103088,1.2389870315301614
87,87,0.19902585446834564,1.2437228843814037
88,88,0.19633902609348297,1.2383399087874616
89,89,0.19415277242660522,1.2426587714523565
90,90,0.192170187830925,1.2403151715388063
91,91,0.18919284641742706,1.2434751166672002
92,92,0.1881590634584427,1.243041742043417
93,93,0.184870183467865,1.2413537697713883
94,94,0.18246163427829742,1.2425490832719646
95,95,0.17979897558689117,1.242463909211706
96,96,0.17738337814807892,1.244733091260566
97,97,0.17568808794021606,1.2430975241739242
98,98,0.17422670125961304,1.24372426017386
99,99,0.17274223268032074,1.2443588757124104

```


## File: `reports\mlp_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.263411045074463,2.7920349621381915
1,1,2.7923760414123535,2.498685742987961
2,2,2.4426918029785156,2.245456883164703
3,3,2.12640118598938,2.029945623679239
4,4,1.851228952407837,1.865478515625
5,5,1.6383033990859985,1.7440955990650615
6,6,1.4815292358398438,1.6606865554559427
7,7,1.3745641708374023,1.6149417064228997
8,8,1.312411904335022,1.5917833672195185
9,9,1.275299310684204,1.576599371237833
10,10,1.246645212173462,1.5587869863041113
11,11,1.2155653238296509,1.53595220847208
12,12,1.1797438859939575,1.5110388583824284
13,13,1.140494704246521,1.4866031584192494
14,14,1.0990980863571167,1.4634677073994622
15,15,1.057684302330017,1.4408643128441982
16,16,1.018231749534607,1.4201887787365524
17,17,0.9828295707702637,1.4029758640977203
18,18,0.9521333575248718,1.389393915895556
19,19,0.9245441555976868,1.3784102142834274
20,20,0.8992729783058167,1.3686897402904072
21,21,0.875594973564148,1.3594406628217854
22,22,0.8529478311538696,1.3498313778736553
23,23,0.830567479133606,1.3399730744909069
24,24,0.8081187009811401,1.331178758965164
25,25,0.7860308885574341,1.3233700111264088
26,26,0.7642250061035156,1.3155467549308402
27,27,0.7436730861663818,1.3093486848424694
28,28,0.7242656946182251,1.3037793519066982
29,29,0.7058826088905334,1.298999348624808
30,30,0.688615083694458,1.29399046350698
31,31,0.6721667647361755,1.289396442350794
32,32,0.6564215421676636,1.2863798297819544
33,33,0.6413283944129944,1.2847317554911628
34,34,0.626444399356842,1.2821542708600153
35,35,0.6120616793632507,1.279786156826332
36,36,0.5983657836914062,1.2788662519611296
37,37,0.584523618221283,1.2774753257876537
38,38,0.5717360973358154,1.2754770497806738
39,39,0.5591100454330444,1.2749553742955944
40,40,0.5472186207771301,1.2753549794681738
41,41,0.5357116460800171,1.2759009189293034
42,42,0.5244393348693848,1.276153314309042
43,43,0.5137184858322144,1.27578860423604
44,44,0.5032526254653931,1.2754093858062243
45,45,0.4930139183998108,1.2752830630443135
46,46,0.48307889699935913,1.2760520059554303
47,47,0.47337695956230164,1.2776573056080303
48,48,0.4639345407485962,1.2790432288998463
49,49,0.4548347294330597,1.2792738617443649
50,50,0.4459763765335083,1.2806916784067623
51,51,0.4373616576194763,1.2826735699763063
52,52,0.42925018072128296,1.2816219642514088
53,53,0.4210401177406311,1.2839185370773565
54,54,0.41327258944511414,1.2850251745005123
55,55,0.40567171573638916,1.284587547427318
56,56,0.398327112197876,1.2856989375880508
57,57,0.3912232518196106,1.286009991755251
58,58,0.384308397769928,1.2869531600201716
59,59,0.3776601552963257,1.287640555960233
60,60,0.37119346857070923,1.2880716792872695
61,61,0.3649361729621887,1.2877319836225667
62,62,0.35885894298553467,1.2882240170338115
63,63,0.353054016828537,1.2889107876136654
64,64,0.34737473726272583,1.2899317506883965
65,65,0.3418669104576111,1.2916033385229893
66,66,0.33654049038887024,1.2912033581342854
67,67,0.3314157724380493,1.2930935718974128
68,68,0.32650068402290344,1.292662823786501
69,69,0.32190728187561035,1.2951214899782275
70,70,0.3170872628688812,1.2936806600601947
71,71,0.31277239322662354,1.2959499671810963
72,72,0.3083682656288147,1.2982700535508453
73,73,0.3041675388813019,1.2988406322041497
74,74,0.30015406012535095,1.2974143106429303
75,75,0.2961548864841461,1.2976859671170595
76,76,0.29194530844688416,1.3008537917840677
77,77,0.287934809923172,1.3023611600281761
78,78,0.28415796160697937,1.3010729180007685
79,79,0.28067222237586975,1.3027091104476178
80,80,0.2773756980895996,1.3023138827964909
81,81,0.2742115557193756,1.3042668827244492
82,82,0.2709978520870209,1.3045536729155993
83,83,0.26788586378097534,1.304993676357582
84,84,0.2647753655910492,1.3050497086321722
85,85,0.261470228433609,1.30604115470511
86,86,0.2582976222038269,1.30777102611104
87,87,0.2552628517150879,1.3086090087890625
88,88,0.2525273263454437,1.3085896226226306
89,89,0.25002291798591614,1.3090004842789447
90,90,0.24774983525276184,1.313157253578061
91,91,0.245594322681427,1.310150146484375
92,92,0.2436712384223938,1.3163889900582735
93,93,0.24163664877414703,1.3110893124439678
94,94,0.23905879259109497,1.3158474281185963
95,95,0.2358855903148651,1.3144507486312116
96,96,0.2330462783575058,1.314289655841765
97,97,0.2309645563364029,1.318464060298732
98,98,0.229164257645607,1.3149078869428792
99,99,0.22737957537174225,1.318449927158043

```


## File: `reports\multiseed_linear_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7587254464626312,0.8916218936443329,0.5667760968208313,0.32484579682350156,0.45,0.15,0.525,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.19999999999999996,20,7.178725379627492
2,0.8277763485908508,0.8731832712888717,0.5667760968208313,0.30640717446804044,0.47058823529411764,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.78231596165683
3,1.205806669066934,0.8284962668138391,0.5667760968208313,0.26172016999300785,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.704794690952647
4,2.1232856710751853,0.7684605518976847,0.5667760968208313,0.2016844550768534,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,2.7010204609747044
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8363187670707702,0.8786539196968078,0.5610660910606384,0.3175878286361694,0.6,0.15,0.5125,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.08750000000000002,-0.09999999999999998,0.25,20,6.512686111023964
2,0.8319200843572616,0.8723046600818634,0.5610660910606384,0.31123856902122493,0.6470588235294118,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.7485337201196165
3,1.1613777314915377,0.8332799532834221,0.5610660910606384,0.27221386222278365,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.884778363759029
4,1.9573414127031963,0.7839478254318237,0.5610660910606384,0.2228817343711853,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,2.930014153304036
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed42\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,1.0122331768274306,0.8583914279937744,0.5589107871055603,0.29948064088821413,0.35,0.15,0.55,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.04999999999999999,0.25,20,5.3808566478344755
2,0.9888877153396607,0.84873625934124,0.5589107871055603,0.28982547223567967,0.29411764705882354,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,5.677328835864212
3,1.1383345688090605,0.8322457986719468,0.5589107871055603,0.2733350115663865,0.3333333333333333,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.983660314275305
4,1.464249610900879,0.8183408379554749,0.5589107871055603,0.25943005084991455,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,3.916707916036257
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9052079662680625,0.8721217930316925,0.5730632543563843,0.2990585386753082,0.25,0.45,0.5,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.07500000000000001,-0.05,-0.2,20,6.151541022897078
2,1.2418081760406494,0.812265083193779,0.5730632543563843,0.23920182883739471,0.45,0.45,0.4375,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.08750000000000002,-0.05,-0.2,20,4.70217637598082
3,1.5967497646808624,0.7742615401744842,0.5730632543563843,0.20119828581809995,0.4,0.3,0.475,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.2375,-0.9,0.19999999999999996,20,3.5701670815773947
4,3.167183244228363,0.6390329658985138,0.5730632543563843,0.0659697115421295,0.35,0.55,0.3625,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.1875,1.0,0.19999999999999996,20,1.8322876914774997
5,3.5649105429649355,0.7311459809541703,0.5730632543563843,0.158082726597786,,,0.2875,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.175,0.0,0.19999999999999996,20,1.6142300352476104
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_linear_seed42\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9248135641217232,0.8670866698026657,0.560093343257904,0.30699332654476164,0.35,0.15,0.525,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.04999999999999999,0.25,20,5.889491493199599
2,1.0127511948347092,0.8450578063726425,0.560093343257904,0.28496446311473844,0.35294117647058826,0.5882352941176471,0.5,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.012500000000000011,0.30000000000000004,0.30000000000000004,20,5.543553807059229
3,1.275463945725385,0.8165730518453261,0.560093343257904,0.2564797085874221,0.16666666666666666,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.447850395108584
4,1.8458486199378967,0.7874423861503601,0.560093343257904,0.22734904289245605,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,3.1069926212375076
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.450 | 0.150 | 0.525 | 0.562 | 0.7587 |
| 2 | 0.873 | 0.471 | 0.588 | 0.438 | 0.512 | 0.8278 |
| 3 | 0.828 | 0.167 | 0.667 | 0.471 | 0.471 | 1.2058 |
| 4 | 0.768 | nan | nan | 0.167 | 0.417 | 2.1233 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.075 | 0.037 | +0.325 |
| Shuffled Action | -0.075 | 0.037 | +0.307 |
| Constant Action | -0.050 | 0.012 | +0.299 |
| Blind | -0.088 | 0.050 | +0.318 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.3586.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_linear_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.879 | +0.108 |
| 2 | 1.000 | 0.873 | 0.872 | +0.127 |
| 3 | 1.000 | 0.828 | 0.833 | +0.172 |
| 4 | 1.000 | 0.768 | 0.784 | +0.232 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.525 | 0.512 | +0.037 |
| 2 | 0.512 | 0.438 | 0.512 | +0.075 |
| 3 | 0.471 | 0.471 | 0.500 | +0.000 |
| 4 | 0.417 | 0.167 | 0.292 | +0.250 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.450 | 0.600 | -0.300 |
| 2 | 0.588 | 0.471 | 0.647 | +0.118 |
| 3 | 0.667 | 0.167 | 0.333 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.107 | +0.250 | +0.333 |
| 2 | +0.007 | n/a | +3.000 |
| 3 | -0.029 | +1.000 | -0.500 |
| 4 | -0.072 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.879 | 0.287 | 0.562 | 0.525 | 0.512 | 0.600 |
| 2 | 1.000 | 0.873 | 0.872 | 0.246 | 0.512 | 0.438 | 0.512 | 0.487 |
| 3 | 1.000 | 0.828 | 0.833 | 0.241 | 0.471 | 0.471 | 0.500 | 0.456 |
| 4 | 1.000 | 0.768 | 0.784 | 0.237 | 0.417 | 0.167 | 0.292 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1596
**Average Oracle Gain (state probe):** +0.0906
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_linear_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_linear_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_linear_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.5232813358306885,3.0784426829853997
1,1,3.061244010925293,2.3878606577388575
2,2,2.2403244972229004,2.029964259413422
3,3,1.7933316230773926,1.8156505647252819
4,4,1.5140321254730225,1.7165387263063525
5,5,1.3788529634475708,1.6749196287061348
6,6,1.3168095350265503,1.6206438658667393
7,7,1.2374018430709839,1.567597936411373
8,8,1.1598652601242065,1.5260500048027663
9,9,1.0993807315826416,1.4842356697457735
10,10,1.0412166118621826,1.453754737728932
11,11,0.9957539439201355,1.4266500004002305
12,12,0.9542943239212036,1.4085471981861553
13,13,0.9181751608848572,1.3954060038582223
14,14,0.8842596411705017,1.3816063052318135
15,15,0.8511653542518616,1.367311071176998
16,16,0.8182157278060913,1.3544940635806224
17,17,0.7854738235473633,1.3438072829950052
18,18,0.7560673356056213,1.3318641537525615
19,19,0.7265509366989136,1.323320107381852
20,20,0.6993833184242249,1.3155884039206582
21,21,0.6730312705039978,1.3073567875096055
22,22,0.6492671966552734,1.2988300010806224
23,23,0.6268006563186646,1.292359399013832
24,24,0.6061885356903076,1.2882297703477203
25,25,0.5871606469154358,1.2824039146548412
26,26,0.5691555738449097,1.2748278008132685
27,27,0.5516728758811951,1.2711516833696208
28,28,0.5351634621620178,1.2701318459432633
29,29,0.5199247002601624,1.2674638091540726
30,30,0.504911482334137,1.2636970144803408
31,31,0.4902103841304779,1.2610631223584785
32,32,0.4756787121295929,1.2584272290839524
33,33,0.462110698223114,1.2556156095911244
34,34,0.44944843649864197,1.2540703445184427
35,35,0.4369819760322571,1.2524386546650872
36,36,0.4251714050769806,1.2507559354188011
37,37,0.4137410521507263,1.2509680576011784
38,38,0.40372127294540405,1.2567527645924053
39,39,0.39789727330207825,1.267412904833184
40,40,0.40243738889694214,1.2961758472880378
41,41,0.420651376247406,1.2965205458344007
42,42,0.4150506258010864,1.2597887633276768
43,43,0.36139920353889465,1.258068272324859
44,44,0.3516237437725067,1.279644450203317
45,45,0.3700995147228241,1.2610107171730918
46,46,0.3352540135383606,1.257061942678983
47,47,0.3236881494522095,1.27032720847208
48,48,0.3347918391227722,1.2551627237288678
49,49,0.30532291531562805,1.2631935995133197
50,50,0.30493786931037903,1.2654826680167777
51,51,0.3038899600505829,1.2545682563156377
52,52,0.281889408826828,1.2702580436331328
53,53,0.2884024977684021,1.2623309776431224
54,54,0.2754635214805603,1.2605425725217725
55,55,0.2666817307472229,1.2725404833183913
56,56,0.26852676272392273,1.263793445024334
57,57,0.25341159105300903,1.269014702468622
58,58,0.25429266691207886,1.272353250472272
59,59,0.24717190861701965,1.271497507564357
60,60,0.23948034644126892,1.2746211818007172
61,61,0.23979289829730988,1.2736741363025101
62,62,0.22980231046676636,1.2808595250864498
63,63,0.22896374762058258,1.2792917470462988
64,64,0.22392967343330383,1.2806218882076075
65,65,0.2181147038936615,1.2880900648773694
66,66,0.21734707057476044,1.2851532482710042
67,67,0.21047022938728333,1.2891102775198515
68,68,0.20879092812538147,1.293485047387295
69,69,0.20503957569599152,1.293212890625
70,70,0.20055095851421356,1.2955064617219518
71,71,0.19916898012161255,1.2984103843814037
72,72,0.1944286823272705,1.301788580222208
73,73,0.1922878623008728,1.3012956713066726
74,74,0.1895606368780136,1.3049685368772412
75,75,0.18587914109230042,1.3100525902920082
76,76,0.18426373600959778,1.3084310312740137
77,77,0.18093577027320862,1.3126958628169825
78,78,0.1784941554069519,1.317352795210041
79,79,0.17649687826633453,1.3165937329902024
80,80,0.1734810620546341,1.3192101150262552
81,81,0.17159827053546906,1.323790753474001
82,82,0.16930101811885834,1.3239878670113985
83,83,0.16688548028469086,1.3256593297739498
84,84,0.16505755484104156,1.330614308841893
85,85,0.16279318928718567,1.3309032252577484
86,86,0.16081278026103973,1.333599278184234
87,87,0.1589294970035553,1.336533718421811
88,88,0.15687046945095062,1.3391646088146774
89,89,0.15512831509113312,1.3393502157242572
90,90,0.15329909324645996,1.3450249843910091
91,91,0.1514773964881897,1.3431380225009606
92,92,0.14997807145118713,1.3511209956935195
93,93,0.14860548079013824,1.34632435783011
94,94,0.14772263169288635,1.363326651151063
95,95,0.14818719029426575,1.3500226130251025
96,96,0.15129242837429047,1.3898708155897797
97,97,0.15794388949871063,1.3604906426101435
98,98,0.16426262259483337,1.3984038556208376
99,99,0.15833428502082825,1.3586328225057633

```


## File: `reports\multiseed_linear_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.4580087661743164,3.0653456390881146
1,1,3.0447001457214355,2.3464755699282787
2,2,2.1839418411254883,1.9878092281153945
3,3,1.7344741821289062,1.7977502541463883
4,4,1.489260196685791,1.698744852034772
5,5,1.3653203248977661,1.6524376791031634
6,6,1.3012382984161377,1.6120140200755635
7,7,1.2344423532485962,1.5660260309938525
8,8,1.1610493659973145,1.528157343629931
9,9,1.1012648344039917,1.4897736095991292
10,10,1.04425847530365,1.458388406722272
11,11,0.9967300891876221,1.4319605592821465
12,12,0.9566307663917542,1.4142396020107582
13,13,0.9221246242523193,1.4024120393346569
14,14,0.8895140886306763,1.389162157402664
15,15,0.8563276529312134,1.3776913001889088
16,16,0.8259621858596802,1.368669228475602
17,17,0.7960246205329895,1.3627164246606045
18,18,0.7693154811859131,1.3557999407658812
19,19,0.7427224516868591,1.3507976844662526
20,20,0.7178900837898254,1.3457776679367315
21,21,0.693490743637085,1.340913866387039
22,22,0.6707596182823181,1.3381620313300462
23,23,0.6495320200920105,1.3365938780737705
24,24,0.630782425403595,1.3354773599593366
25,25,0.6129720211029053,1.3351928210649333
26,26,0.5960358381271362,1.33192006095511
27,27,0.5794575214385986,1.3311074679015114
28,28,0.5640406012535095,1.3308071699298796
29,29,0.5487897396087646,1.3307425076844261
30,30,0.5346084237098694,1.3306036777183659
31,31,0.5205250382423401,1.3297089123335042
32,32,0.5068598389625549,1.3316651641345414
33,33,0.4937836527824402,1.3297664454725922
34,34,0.48122310638427734,1.3301501664959017
35,35,0.4695097804069519,1.3285474933561732
36,36,0.4582706689834595,1.3326330966636784
37,37,0.44809919595718384,1.3317832321417136
38,38,0.4396570324897766,1.342547432321017
39,39,0.43437859416007996,1.345001220703125
40,40,0.43307721614837646,1.3610317042616547
41,41,0.4293820559978485,1.346677436203253
42,42,0.41264864802360535,1.3413458652183659
43,43,0.3885851502418518,1.3411602583087858
44,44,0.3787395656108856,1.3470261370549437
45,45,0.38102471828460693,1.3582643602715163
46,46,0.37395426630973816,1.34506350657979
47,47,0.35611119866371155,1.3450437451972337
48,48,0.34675005078315735,1.3571979960457223
49,49,0.3470659852027893,1.3532919961898053
50,50,0.3401643633842468,1.351898693647541
51,51,0.326469749212265,1.3541040889552383
52,52,0.32060950994491577,1.35698374763864
53,53,0.31959301233291626,1.361828288093942
54,54,0.3119523823261261,1.3565585026975537
55,55,0.30206185579299927,1.3586169383564934
56,56,0.2984490394592285,1.366713351890689
57,57,0.2960898280143738,1.361645432769275
58,58,0.28880783915519714,1.362989957215356
59,59,0.28164276480674744,1.3678481305231813
60,60,0.27886486053466797,1.3668943311347337
61,61,0.2757686376571655,1.369722460137039
62,62,0.2695058584213257,1.3690812157802894
63,63,0.26407289505004883,1.3705904601050205
64,64,0.26149189472198486,1.3757390506931992
65,65,0.2584390938282013,1.373369936083184
66,66,0.2533557713031769,1.3754317486872438
67,67,0.24884384870529175,1.378631091508709
68,68,0.24619214236736298,1.3782456194768187
69,69,0.24349205195903778,1.3820610671746927
70,70,0.23947378993034363,1.3814502153240267
71,71,0.2354346066713333,1.3830966636782787
72,72,0.2326289713382721,1.3872250416239753
73,73,0.23032011091709137,1.3870344318327357
74,74,0.2273508608341217,1.3900344098200563
75,75,0.22389420866966248,1.39198365758677
76,76,0.2208799570798874,1.3932027347752305
77,77,0.21857808530330658,1.3987780711689934
78,78,0.2164197862148285,1.397829149590164
79,79,0.21394167840480804,1.4040684934522285
80,80,0.21144936978816986,1.4029152041575947
81,81,0.20995202660560608,1.4155438532594775
82,82,0.21149598062038422,1.417528371341893
83,83,0.22111663222312927,1.46232292300365
84,84,0.2435227483510971,1.44897711081583
85,85,0.25610119104385376,1.4475562924244365
86,86,0.21984997391700745,1.419141300389024
87,87,0.1985277384519577,1.4351556496542008
88,88,0.22330988943576813,1.446212893626729
89,89,0.20984572172164917,1.4233818679559427
90,90,0.19213029742240906,1.434742786845223
91,91,0.21020632982254028,1.4368547533379226
92,92,0.19459408521652222,1.4361980000480277
93,93,0.19068053364753723,1.433533465275999
94,94,0.19823601841926575,1.4315950987768955
95,95,0.18327325582504272,1.449432122902792
96,96,0.19126392900943756,1.4334566710425205
97,97,0.18450331687927246,1.437072503762167
98,98,0.1809718757867813,1.4508673245789574
99,99,0.18423697352409363,1.4414422707479508

```


## File: `reports\multiseed_linear_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.751734958589077,0.8925475746393203,0.5667760968208313,0.32577147781848903,0.4,0.15,0.5125,0.5625,0.2,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.08750000000000002,-0.14999999999999997,0.19999999999999996,20,7.245481344799113
2,0.8426082193851471,0.871407949924469,0.5667760968208313,0.3046318531036377,0.5294117647058824,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.662931374947255
3,1.2523492609753328,0.8255621966193704,0.5667760968208313,0.25878609979853906,0.0,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,4.529944634233426
4,2.264716386795044,0.7627657254536947,0.5667760968208313,0.1959896286328634,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.532342714305401
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.865130840241909,0.8754267513751983,0.5610660910606384,0.3143606603145599,0.55,0.15,0.55,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.04999999999999999,0.25,20,6.295789452110483
2,0.8376157730817795,0.8713648825883865,0.5610660910606384,0.3102987915277481,0.5882352941176471,0.5882352941176471,0.4125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.07500000000000001,0.30000000000000004,0.30000000000000004,20,6.702644484682594
3,1.0868084535879248,0.8413524943239549,0.5610660910606384,0.28028640326331644,0.5,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,5.2199380637983275
4,1.569354812304179,0.8073844710985819,0.5610660910606384,0.24631838003794349,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.654392236289834
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed43\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9616783663630486,0.8648374289274215,0.5589107871055603,0.30592664182186124,0.3,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,5.663724805715639
2,0.950974041223526,0.853962504863739,0.5589107871055603,0.29505171775817873,0.35294117647058826,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,5.9036740209085385
3,1.1634302805451786,0.8287736528060016,0.5589107871055603,0.26986286570044127,0.16666666666666666,0.6666666666666666,0.5441176470588235,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.08823529411764702,0.35294117647058826,0.2941176470588235,17,4.876160531323826
4,1.5829979181289673,0.805232991774877,0.5589107871055603,0.24632220466931665,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,3.6228967684600875
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9065004095435143,0.8720436930656433,0.5730632543563843,0.29898043870925906,0.2,0.45,0.4625,0.475,0.0,0.0,0.05,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.03750000000000003,-0.05,-0.15000000000000002,20,6.142770461135597
2,1.2767688840627671,0.8078105270862579,0.5730632543563843,0.2347472727298736,0.4,0.45,0.425,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.07500000000000001,-0.05,-0.2,20,4.573420563240413
3,1.7245627284049987,0.7620243072509766,0.5730632543563843,0.1889610528945923,0.4,0.3,0.475,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.2375,-0.9,0.19999999999999996,20,3.3055703648729873
4,3.4382385492324827,0.626537661254406,0.5730632543563843,0.05347440689802174,0.3,0.55,0.3375,0.4625,1.0,0.9,0.95,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.16250000000000003,1.0,0.1499999999999999,20,1.6878383486069781
5,3.9870136618614196,0.7218256652355194,0.5730632543563843,0.1487624108791351,,,0.2875,0.3625,0.0,0.0,0.7,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.175,0.0,-0.10000000000000009,20,1.4433323182389646
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_linear_seed43\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8987120151519775,0.8706511646509171,0.560093343257904,0.310557821393013,0.3,0.15,0.525,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.25,20,6.060541671704951
2,1.0090200453996658,0.8460196435451508,0.560093343257904,0.28592630028724675,0.35294117647058826,0.5882352941176471,0.4875,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.0,0.30000000000000004,0.30000000000000004,20,5.564052733467722
3,1.2997813540346481,0.8157661276705125,0.560093343257904,0.25567278441260843,0.16666666666666666,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.364636250036696
4,1.9673248529434204,0.7816300491491953,0.560093343257904,0.22153670589129126,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.9151454237402548
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.893 | 0.400 | 0.150 | 0.512 | 0.562 | 0.7517 |
| 2 | 0.871 | 0.529 | 0.588 | 0.450 | 0.512 | 0.8426 |
| 3 | 0.826 | 0.000 | 0.667 | 0.515 | 0.471 | 1.2523 |
| 4 | 0.763 | nan | nan | 0.208 | 0.417 | 2.2647 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **2**; rollout state probe accuracy stays >= 0.50 through depth **3**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.088 | 0.050 | +0.326 |
| Shuffled Action | -0.075 | 0.037 | +0.311 |
| Constant Action | -0.037 | 0.000 | +0.306 |
| Blind | -0.050 | 0.012 | +0.314 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.3518.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_linear_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.875 | +0.107 |
| 2 | 1.000 | 0.871 | 0.871 | +0.129 |
| 3 | 1.000 | 0.826 | 0.841 | +0.174 |
| 4 | 1.000 | 0.763 | 0.807 | +0.237 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.512 | 0.550 | +0.050 |
| 2 | 0.512 | 0.450 | 0.412 | +0.062 |
| 3 | 0.471 | 0.515 | 0.471 | -0.044 |
| 4 | 0.417 | 0.208 | 0.250 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.550 | -0.250 |
| 2 | 0.588 | 0.529 | 0.588 | +0.059 |
| 3 | 0.667 | 0.000 | 0.500 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.137 | -3.000 | +0.375 |
| 2 | +0.000 | +0.375 | n/a |
| 3 | -0.100 | n/a | -3.000 |
| 4 | -0.232 | -0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.893 | 0.875 | 0.287 | 0.562 | 0.512 | 0.550 | 0.600 |
| 2 | 1.000 | 0.871 | 0.871 | 0.246 | 0.512 | 0.450 | 0.412 | 0.487 |
| 3 | 1.000 | 0.826 | 0.841 | 0.241 | 0.471 | 0.515 | 0.471 | 0.456 |
| 4 | 1.000 | 0.763 | 0.807 | 0.237 | 0.417 | 0.208 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1619
**Average Oracle Gain (state probe):** +0.0692
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_linear_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_linear_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_linear_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.473798990249634,3.12270242659772
1,1,3.124931573867798,2.423854139984631
2,2,2.292603015899658,2.0329588593029584
3,3,1.8056185245513916,1.8037912337506403
4,4,1.5028027296066284,1.6867510686155225
5,5,1.3437180519104004,1.6321573726466445
6,6,1.2653868198394775,1.5842237628874232
7,7,1.1954405307769775,1.5367098948994622
8,8,1.127571702003479,1.5044559416223744
9,9,1.075363039970398,1.4730133306784707
10,10,1.0219862461090088,1.4403168725185707
11,11,0.9692369699478149,1.4161632100089652
12,12,0.9268369078636169,1.4033475782050462
13,13,0.8926759958267212,1.3911139066102074
14,14,0.8607674837112427,1.3744579377721569
15,15,0.8272882699966431,1.3592411729155993
16,16,0.7937148809432983,1.3447972282034453
17,17,0.7618528604507446,1.332826708183914
18,18,0.7327417135238647,1.3249610525662783
19,19,0.7062252759933472,1.3162451572105534
20,20,0.6815217733383179,1.3101560248703252
21,21,0.6581062078475952,1.301556571585233
22,22,0.6361016631126404,1.2960582795690319
23,23,0.6155379414558411,1.2915055321865394
24,24,0.595582902431488,1.2865640608990778
25,25,0.5767673254013062,1.2822185578893444
26,26,0.5586186647415161,1.2760069800204918
27,27,0.5420235395431519,1.2764371027711963
28,28,0.5274988412857056,1.2705580914606813
29,29,0.516865611076355,1.293925175901319
30,30,0.517898440361023,1.2994340990410476
31,31,0.5353500843048096,1.3200283363217213
32,32,0.5210360288619995,1.2561445392546107
33,33,0.46050214767456055,1.2617746572025488
34,34,0.45885658264160156,1.297089373479124
35,35,0.46832042932510376,1.2488278248271003
36,36,0.42275965213775635,1.2540720955270235
37,37,0.4205898940563202,1.2792010698162142
38,38,0.42109835147857666,1.2442877097207992
39,39,0.38606593012809753,1.2520850760037783
40,40,0.3920610249042511,1.2646311775582735
41,41,0.37858736515045166,1.2496297867571722
42,42,0.35816600918769836,1.2515226270331712
43,43,0.36348676681518555,1.251851191286181
44,44,0.342132568359375,1.2565586527840036
45,45,0.33664608001708984,1.2491226196289062
46,46,0.33239272236824036,1.2462891125288167
47,47,0.31494733691215515,1.2599647396900615
48,48,0.3158060312271118,1.2468346767738216
49,49,0.3035111725330353,1.24755859375
50,50,0.29553353786468506,1.2609345482998207
51,51,0.2933961749076843,1.250773320432569
52,52,0.2806622385978699,1.252729197017482
53,53,0.2786718010902405,1.2607520681912783
54,54,0.2714173495769501,1.2581359362993083
55,55,0.26335424184799194,1.2571725063636654
56,56,0.2614958882331848,1.261036982301806
57,57,0.2525847852230072,1.2654609054815573
58,58,0.24890883266925812,1.2612782462698515
59,59,0.24472501873970032,1.263547803534836
60,60,0.2375822365283966,1.2713222816342213
61,61,0.23538564145565033,1.266205084128458
62,62,0.22976556420326233,1.2684438736712347
63,63,0.2250826209783554,1.2771341292584528
64,64,0.2225399762392044,1.2730438982854124
65,65,0.21702860295772552,1.274828926461642
66,66,0.21386602520942688,1.2823561371349899
67,67,0.2107597291469574,1.2799292392418034
68,68,0.2061084359884262,1.2812548778096184
69,69,0.2036098688840866,1.2871925479075947
70,70,0.2002887725830078,1.2865555560002562
71,71,0.19648605585098267,1.2869760482037653
72,72,0.19422820210456848,1.2931154595046748
73,73,0.1910395622253418,1.2933189517161885
74,74,0.18786343932151794,1.2938347488153177
75,75,0.18570773303508759,1.3000845987288678
76,76,0.1827934831380844,1.3002697053502819
77,77,0.18002451956272125,1.3015872142353997
78,78,0.17795540392398834,1.3061782336625896
79,79,0.17536863684654236,1.3078980993051998
80,80,0.1728937029838562,1.3073567875096055
81,81,0.17102673649787903,1.3157575013207607
82,82,0.16907000541687012,1.3118758905129355
83,83,0.167637899518013,1.323761236472208
84,84,0.16803139448165894,1.3211578619284707
85,85,0.1712256520986557,1.3454412241451075
86,86,0.1770879477262497,1.330809546298668
87,87,0.17934733629226685,1.3448552616306992
88,88,0.16876813769340515,1.3247290439293034
89,89,0.15571478009223938,1.32662838795146
90,90,0.15612173080444336,1.345931631619813
91,91,0.16114848852157593,1.3322808937948258
92,92,0.15619471669197083,1.334980823954598
93,93,0.14879649877548218,1.3417138271644466
94,94,0.14954322576522827,1.3385562584048412
95,95,0.15019163489341736,1.343247085321145
96,96,0.14554773271083832,1.344024908347208
97,97,0.14298953115940094,1.3454007008036628
98,98,0.14304065704345703,1.3486855929015114
99,99,0.14103855192661285,1.351835657338627

```


## File: `reports\multiseed_linear_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.5475337505340576,3.1573676437628073
1,1,3.14328932762146,2.470789674852715
2,2,2.3241186141967773,2.085841444672131
3,3,1.8465005159378052,1.8534095639088115
4,4,1.5549119710922241,1.7064856857549948
5,5,1.3730700016021729,1.6466402147637038
6,6,1.291268229484558,1.6088175539110527
7,7,1.2277226448059082,1.5650209520683913
8,8,1.1584500074386597,1.5263831967213115
9,9,1.099727988243103,1.4890266793673155
10,10,1.0436184406280518,1.4584316816486296
11,11,0.9946844577789307,1.4350969908667393
12,12,0.955093264579773,1.4190238577420595
13,13,0.9206522703170776,1.4093487849001025
14,14,0.8897719979286194,1.3972474395251664
15,15,0.8578667044639587,1.3835410446417136
16,16,0.8261060118675232,1.37103021340292
17,17,0.7946690917015076,1.3620973180551998
18,18,0.7662612199783325,1.3554908877513447
19,19,0.7394025325775146,1.3515688786741162
20,20,0.7157735228538513,1.3433163752321338
21,21,0.6923640966415405,1.3366846803758965
22,22,0.6710612773895264,1.3332274390048668
23,23,0.6501696109771729,1.3319329433753841
24,24,0.6302686333656311,1.3324526177077998
25,25,0.6112964749336243,1.32911744664927
26,26,0.5936397910118103,1.3273050276959528
27,27,0.5771744847297668,1.3266393942911117
28,28,0.561978280544281,1.3284471855788935
29,29,0.547471284866333,1.328292471463563
30,30,0.5330768823623657,1.3276139556384476
31,31,0.5194072127342224,1.3271511890849128
32,32,0.5059271454811096,1.328173903168225
33,33,0.49274003505706787,1.327789806928791
34,34,0.4799653887748718,1.3271404328893444
35,35,0.46742644906044006,1.3260773205366292
36,36,0.4560222625732422,1.3326571104956455
37,37,0.44654354453086853,1.335816680407915
38,38,0.4445897936820984,1.3775718563892803
39,39,0.4650554358959198,1.3974719438396517
40,40,0.4997129440307617,1.3845449979188011
41,41,0.4491548240184784,1.3382313212410348
42,42,0.3959680199623108,1.372406756291624
43,43,0.4371569752693176,1.361439063900807
44,44,0.3972165286540985,1.3525125472272028
45,45,0.37936094403266907,1.3637605260630123
46,46,0.3973471522331238,1.3409503874231556
47,47,0.3539126515388489,1.3716308093461833
48,48,0.37250763177871704,1.3454903774574154
49,49,0.34824976325035095,1.349192509885694
50,50,0.3434472382068634,1.3647213294857838
51,51,0.34196171164512634,1.3498039870965677
52,52,0.32232365012168884,1.356671817967149
53,53,0.33043816685676575,1.3510279420946465
54,54,0.3089587390422821,1.368275501688973
55,55,0.31603485345840454,1.3511006089507556
56,56,0.2994425892829895,1.357349458287974
57,57,0.30117496848106384,1.3627436903656507
58,58,0.2911789119243622,1.3646781796314678
59,59,0.28785595297813416,1.3585417700595543
60,60,0.28321102261543274,1.3610367071433145
61,61,0.2764689326286316,1.3709226514472337
62,62,0.2750069200992584,1.3642192903112194
63,63,0.2667718529701233,1.365031132932569
64,64,0.26677459478378296,1.3678376244716957
65,65,0.258476585149765,1.375564199979188
66,66,0.25872156023979187,1.3683341604764345
67,67,0.25118592381477356,1.3715482617987962
68,68,0.2507452666759491,1.3760115826716188
69,69,0.24475222826004028,1.378949900142482
70,70,0.2431372106075287,1.3745084668769212
71,71,0.23885491490364075,1.3780448788502178
72,72,0.23612579703330994,1.3845241108878714
73,73,0.2332174926996231,1.3822684366194928
74,74,0.22973503172397614,1.381651081022669
75,75,0.22772495448589325,1.3877468421810963
76,76,0.22394149005413055,1.3918822241611168
77,77,0.22232002019882202,1.3881690853931865
78,78,0.2187788337469101,1.3922291740042265
79,79,0.21702377498149872,1.3975579934042008
80,80,0.21398164331912994,1.3971344994716957
81,81,0.2119961380958557,1.3975513645860016
82,82,0.20949766039848328,1.4030087580446338
83,83,0.20725952088832855,1.4054422847560195
84,84,0.2051180899143219,1.4052664334656761
85,85,0.20289307832717896,1.4088532494716957
86,86,0.20093883574008942,1.4134779132780482
87,87,0.19874589145183563,1.4131130781330046
88,88,0.197037473320961,1.4166049644595287
89,89,0.1950218826532364,1.4202908375224128
90,90,0.194131001830101,1.427155291447874
91,91,0.19489672780036926,1.4304111668320953
92,92,0.20353157818317413,1.4775340596183402
93,93,0.2302444577217102,1.4803196641265368
94,94,0.25835126638412476,1.474897415911565
95,95,0.22288721799850464,1.4319287909836065
96,96,0.18367800116539001,1.45489501953125
97,97,0.21624323725700378,1.4624978987897028
98,98,0.20305752754211426,1.4390770333712217
99,99,0.18182967603206635,1.4524573654424948

```


## File: `reports\multiseed_linear_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7554293960332871,0.8923281311988831,0.5667760968208313,0.3255520343780518,0.4,0.15,0.5,0.5625,0.2,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.09999999999999998,-0.14999999999999997,0.25,20,7.210047222534201
2,0.8328847318887711,0.8730110496282577,0.5667760968208313,0.3062349528074264,0.4117647058823529,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.740717564840051
3,1.2115291549878962,0.8290544082136715,0.5667760968208313,0.2622783113928402,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.682572261332071
4,2.08437180519104,0.7721383074919382,0.5667760968208313,0.20536221067110694,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,2.7514467561812204
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8411303281784057,0.8781612426042557,0.5610660910606384,0.3170951515436172,0.5,0.15,0.5375,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.0625,-0.04999999999999999,0.25,20,6.475431257467673
2,0.8205735355615615,0.8737681627273559,0.5610660910606384,0.3127020716667175,0.5882352941176471,0.5882352941176471,0.5,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.84184963129187
3,1.1243831620496862,0.836373465902665,0.5610660910606384,0.2753073748420266,0.5,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.045497839544056
4,1.887078086535136,0.7860091825326284,0.5610660910606384,0.22494309147199,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.0391100839915874
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed44\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9313929900527,0.8682799816131592,0.5589107871055603,0.3093691945075989,0.3,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,5.847887708906105
2,0.8957263767719269,0.8622966289520264,0.5589107871055603,0.3033858418464661,0.4117647058823529,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.267807767325863
3,1.0775027976316565,0.8403551578521729,0.5589107871055603,0.28144437074661255,0.16666666666666666,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,5.265019104739942
4,1.480315089225769,0.8149999181429545,0.5589107871055603,0.25608913103739417,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,3.874200893992108
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9206801667809487,0.8710357904434204,0.5730632543563843,0.2979725360870361,0.25,0.45,0.475,0.475,0.0,0.0,0.05,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.15000000000000002,20,6.048163238077094
2,1.289692258834839,0.8070790141820907,0.5730632543563843,0.23401575982570644,0.5,0.45,0.4875,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.1375,-0.05,-0.2,20,4.527592554640552
3,1.660439583659172,0.7667798370122909,0.5730632543563843,0.19371658265590663,0.45,0.3,0.5,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.2625,-0.9,0.19999999999999996,20,3.4332254563681883
4,3.0907593727111817,0.6458056896924973,0.5730632543563843,0.07274243533611302,0.4,0.55,0.4,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.22500000000000003,1.0,0.19999999999999996,20,1.8775938775081373
5,3.2380683541297914,0.7365942686796189,0.5730632543563843,0.16353101432323458,,,0.2875,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.175,0.0,0.19999999999999996,20,1.7771662120985
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_linear_seed44\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8945617213845253,0.871246549487114,0.560093343257904,0.31115320622920994,0.3,0.15,0.525,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.25,20,6.0886593831229305
2,1.0020571678876877,0.8472233265638351,0.560093343257904,0.2871299833059311,0.35294117647058826,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,5.602715016314308
3,1.2695456673117245,0.819041865713456,0.560093343257904,0.258948522455552,0.16666666666666666,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.4685850702434315
4,1.824842095375061,0.7890259027481079,0.560093343257904,0.22893255949020386,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,3.142758519547279
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_linear_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `linear`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 0

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.400 | 0.150 | 0.500 | 0.562 | 0.7554 |
| 2 | 0.873 | 0.412 | 0.588 | 0.463 | 0.512 | 0.8329 |
| 3 | 0.829 | 0.167 | 0.667 | 0.471 | 0.471 | 1.2115 |
| 4 | 0.772 | nan | nan | 0.167 | 0.417 | 2.0844 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.100 | 0.062 | +0.326 |
| Shuffled Action | -0.075 | 0.037 | +0.311 |
| Constant Action | -0.037 | 0.000 | +0.309 |
| Blind | -0.062 | 0.025 | +0.317 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.3580.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_linear_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.878 | +0.108 |
| 2 | 1.000 | 0.873 | 0.874 | +0.127 |
| 3 | 1.000 | 0.829 | 0.836 | +0.171 |
| 4 | 1.000 | 0.772 | 0.786 | +0.228 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.500 | 0.537 | +0.062 |
| 2 | 0.512 | 0.463 | 0.500 | +0.050 |
| 3 | 0.471 | 0.471 | 0.485 | +0.000 |
| 4 | 0.417 | 0.167 | 0.250 | +0.250 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.500 | -0.250 |
| 2 | 0.588 | 0.412 | 0.588 | +0.176 |
| 3 | 0.667 | 0.167 | 0.500 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.116 | -1.500 | +0.286 |
| 2 | -0.006 | -3.000 | n/a |
| 3 | -0.045 | +1.000 | -2.000 |
| 4 | -0.065 | -0.500 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.878 | 0.287 | 0.562 | 0.500 | 0.537 | 0.600 |
| 2 | 1.000 | 0.873 | 0.874 | 0.246 | 0.512 | 0.463 | 0.500 | 0.487 |
| 3 | 1.000 | 0.829 | 0.836 | 0.241 | 0.471 | 0.471 | 0.485 | 0.456 |
| 4 | 1.000 | 0.772 | 0.786 | 0.237 | 0.417 | 0.167 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1584
**Average Oracle Gain (state probe):** +0.0906
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_linear_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_linear_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_linear_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.44976806640625,3.0509513479764343
1,1,3.031238555908203,2.3335378678118595
2,2,2.1770594120025635,1.9851124247566598
3,3,1.742362380027771,1.7840364800124873
4,4,1.4816991090774536,1.6913799848712858
5,5,1.355785846710205,1.6490210861456198
6,6,1.287689447402954,1.6022352625112064
7,7,1.213659405708313,1.5551330066118083
8,8,1.143774390220642,1.5131817176693776
9,9,1.0850750207901,1.471145004522605
10,10,1.0291881561279297,1.434936273293417
11,11,0.9800439476966858,1.409696735319544
12,12,0.9390827417373657,1.3953477202868851
13,13,0.9031093120574951,1.3837367823866547
14,14,0.8688772916793823,1.3709783085056992
15,15,0.8361626863479614,1.358074375840484
16,16,0.8035536408424377,1.3459078679319287
17,17,0.7725101113319397,1.3329555323866547
18,18,0.7424706220626831,1.3232782082479508
19,19,0.7145849466323853,1.316193127241291
20,20,0.6883550882339478,1.3079621361904457
21,21,0.6637231111526489,1.2998378315909964
22,22,0.6405051350593567,1.292522618027984
23,23,0.6190376281738281,1.2873715259989753
24,24,0.5986884832382202,1.2830562904232838
25,25,0.5800177454948425,1.27575433449667
26,26,0.5625174641609192,1.2718740994813011
27,27,0.5461093187332153,1.2686143468637936
28,28,0.530292809009552,1.2661860731781507
29,29,0.5151050090789795,1.2618905989850153
30,30,0.5004920363426208,1.2593022330862578
31,31,0.4860585033893585,1.2557027848040472
32,32,0.47215282917022705,1.2540029306880762
33,33,0.459027498960495,1.251173801109439
34,34,0.44662731885910034,1.2537167658571338
35,35,0.4355659484863281,1.2515008644979508
36,36,0.427636981010437,1.271880853371542
37,37,0.4283599257469177,1.2794972404104765
38,38,0.4400586783885956,1.2979343601914703
39,39,0.43245160579681396,1.2534097139952614
40,40,0.38801684975624084,1.2515361348136527
41,41,0.3738956153392792,1.2834707791688011
42,42,0.3881243169307709,1.2554253750160091
43,43,0.36303362250328064,1.2488225718013575
44,44,0.3428020477294922,1.274490231373271
45,45,0.35283881425857544,1.2524740500528304
46,46,0.33291780948638916,1.2484030801741803
47,47,0.318839967250824,1.2687769405177383
48,48,0.32428979873657227,1.2504516351418418
49,49,0.3054036796092987,1.250608100265753
50,50,0.29893958568573,1.2664607313812757
51,51,0.29883435368537903,1.253462869612897
52,52,0.28254279494285583,1.2546781946401127
53,53,0.2812030017375946,1.263886998911373
54,54,0.2757093608379364,1.2572906994428792
55,55,0.26415109634399414,1.2573498585185066
56,56,0.26409703493118286,1.261324022637039
57,57,0.2554193139076233,1.2608074751056608
58,58,0.2489679753780365,1.2587227743180072
59,59,0.24761152267456055,1.2613778036148822
60,60,0.23867042362689972,1.2658294927878457
61,61,0.23563742637634277,1.262003789182569
62,62,0.23229295015335083,1.2651393452628714
63,63,0.22496454417705536,1.271208215932377
64,64,0.22310961782932281,1.2665414028480404
65,65,0.21852582693099976,1.2705728499615778
66,66,0.21327851712703705,1.2765875644371159
67,67,0.21143300831317902,1.272310851050205
68,68,0.20658357441425323,1.2755413368100026
69,69,0.20274144411087036,1.281876485855853
70,70,0.2007026970386505,1.2790008294777793
71,71,0.1963495910167694,1.2812775158491292
72,72,0.1932782381772995,1.2895776717389216
73,73,0.19105595350265503,1.2858310136638704
74,74,0.18721270561218262,1.2903406111920466
75,75,0.18448516726493835,1.2967977054783555
76,76,0.18233266472816467,1.2956869406778304
77,77,0.17905613780021667,1.2983039480740908
78,78,0.1765069216489792,1.306163975449859
79,79,0.17453740537166595,1.3029738879594646
80,80,0.17186132073402405,1.3099840508132685
81,81,0.16961611807346344,1.3117720807184938
82,82,0.16852855682373047,1.3217385714171364
83,83,0.16875270009040833,1.3184151571305072
84,84,0.17347970604896545,1.3611935474833503
85,85,0.18705874681472778,1.3405286444992315
86,86,0.20025452971458435,1.3654932741258965
87,87,0.1842915415763855,1.3278771072137552
88,88,0.1573367863893509,1.3275851890689037
89,89,0.1633862555027008,1.3629548119716957
90,90,0.17239871621131897,1.3352092055023694
91,91,0.15585698187351227,1.3318606517353997
92,92,0.15260639786720276,1.3586168132844518
93,93,0.1594821810722351,1.3419119412781761
94,94,0.14890378713607788,1.3379906826331966
95,95,0.14782771468162537,1.3584462150198515
96,96,0.1499698907136917,1.3494359000784453
97,97,0.14185741543769836,1.3449459388607838
98,98,0.14413027465343475,1.3606915083087858
99,99,0.14169883728027344,1.358030725698002

```


## File: `reports\multiseed_linear_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.477013111114502,3.07566183121478
1,1,3.0568201541900635,2.373477372966829
2,2,2.207951784133911,2.0040368252113216
3,3,1.7402472496032715,1.7947807937371927
4,4,1.4793387651443481,1.683149619180648
5,5,1.3430712223052979,1.6345267374007428
6,6,1.2768162488937378,1.596356626416816
7,7,1.2133615016937256,1.5589709672771517
8,8,1.1504830121994019,1.5271224975585938
9,9,1.0991874933242798,1.490070155409516
10,10,1.0448817014694214,1.4580144413181992
11,11,0.9954437613487244,1.4320155909804047
12,12,0.9517760872840881,1.4161929771548412
13,13,0.9153178334236145,1.4048942190701845
14,14,0.8827284574508667,1.3913143970927253
15,15,0.8502360582351685,1.3791735289526768
16,16,0.818909227848053,1.3703735851850667
17,17,0.7889904379844666,1.3628787681704662
18,18,0.761249303817749,1.3559586571865394
19,19,0.7346101403236389,1.3519319628105788
20,20,0.7099528908729553,1.3475414338659069
21,21,0.6860494613647461,1.3442908115074284
22,22,0.664030134677887,1.3406509649558145
23,23,0.6437587738037109,1.3389324751056608
24,24,0.624386191368103,1.337342309170082
25,25,0.6063244938850403,1.3333617663774333
26,26,0.588694155216217,1.3328318361376152
27,27,0.5727509260177612,1.3336887046939037
28,28,0.5573263168334961,1.33761471607646
29,29,0.5434308648109436,1.3360694510037783
30,30,0.5310060977935791,1.34359616138896
31,31,0.5219780206680298,1.347698399277984
32,32,0.5198127031326294,1.3676265028656507
33,33,0.5202411413192749,1.3524052354155993
34,34,0.5040172338485718,1.3376364786116803
35,35,0.4694574177265167,1.3330423323834528
36,36,0.45521900057792664,1.341188024301998
37,37,0.4601123631000519,1.3470529024718239
38,38,0.4457016587257385,1.3299192835073002
39,39,0.4214707612991333,1.333237569840228
40,40,0.4182518422603607,1.3479535462426357
41,41,0.4158916175365448,1.3343854810370774
42,42,0.3965955972671509,1.3347134199298796
43,43,0.38600829243659973,1.348569526047003
44,44,0.3857181966304779,1.3397598266601562
45,45,0.3730674088001251,1.339770582855725
46,46,0.36005526781082153,1.3492244032562757
47,47,0.3581400215625763,1.3446777844038167
48,48,0.3509811758995056,1.345411582071273
49,49,0.3385416567325592,1.351043200883709
50,50,0.33430740237236023,1.3499895940061475
51,51,0.33043140172958374,1.3525458163902409
52,52,0.3202167749404907,1.354157244572874
53,53,0.31400129199028015,1.3547583408043034
54,54,0.31126734614372253,1.3600268754802767
55,55,0.3039532005786896,1.3579456767097848
56,56,0.2965640723705292,1.3590650714811732
57,57,0.2933252155780792,1.3660552228083376
58,58,0.28885915875434875,1.362396490378458
59,59,0.281938761472702,1.363860583696209
60,60,0.2774443030357361,1.3701609627145235
61,61,0.2745034098625183,1.3674186331326845
62,62,0.26946279406547546,1.3699909898101306
63,63,0.2641168236732483,1.3729300577132428
64,64,0.26079654693603516,1.3720639338258838
65,65,0.2576110064983368,1.3771639964619622
66,66,0.2530616521835327,1.3761541647989242
67,67,0.2488037645816803,1.3772530477555072
68,68,0.24587298929691315,1.382671793953317
69,69,0.24288274347782135,1.380146589435515
70,70,0.2390693724155426,1.3834366094870645
71,71,0.23547609150409698,1.3857361840420082
72,72,0.232707679271698,1.3866039338659069
73,73,0.23004832863807678,1.3902622910796618
74,74,0.22690996527671814,1.3907920962474385
75,75,0.22368796169757843,1.393567569920274
76,76,0.22096329927444458,1.396695996894211
77,77,0.21862906217575073,1.3982340077884863
78,78,0.2162054181098938,1.4010358716620774
79,79,0.21359984576702118,1.4049376190685836
80,80,0.21119722723960876,1.4031539666848105
81,81,0.20961564779281616,1.4186428883036628
82,82,0.21011796593666077,1.4107727300925332
83,83,0.2164284884929657,1.4620064907386654
84,84,0.23467113077640533,1.4419147929207223
85,85,0.25394490361213684,1.4662822035492444
86,86,0.23096656799316406,1.416655743708376
87,87,0.1967518925666809,1.4257893796826973
88,88,0.21395455300807953,1.4611673824122695
89,89,0.21779192984104156,1.42103138908011
90,90,0.19205628335475922,1.4285704815974
91,91,0.20359550416469574,1.4499919453605277
92,92,0.20156121253967285,1.4307991403048155
93,93,0.18632872402668,1.4321347846359502
94,94,0.197984516620636,1.4400719814613216
95,95,0.187181293964386,1.4457398711657914
96,96,0.1854524463415146,1.434949405857774
97,97,0.18900319933891296,1.4366987885021774
98,98,0.17860330641269684,1.4558993480244622
99,99,0.1839374452829361,1.4408376724993597

```


## File: `reports\multiseed_mlp_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7347443625330925,0.893628278374672,0.5667760968208313,0.32685218155384066,0.35,0.15,0.5375,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.0625,-0.09999999999999998,0.19999999999999996,20,7.413029478596612
2,0.8018624067306519,0.8764720648527146,0.5667760968208313,0.30969596803188326,0.47058823529411764,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,7.001501373059851
3,0.9844185639830196,0.8528833669774672,0.5667760968208313,0.28610727015663595,0.0,0.6666666666666666,0.5294117647058824,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.0735294117647059,0.35294117647058826,0.2941176470588235,17,5.762866551386227
4,1.3627944787343342,0.8289436598618826,0.5667760968208313,0.26216756304105127,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,4.208292689441165
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8066169574856759,0.8823343634605407,0.5610660910606384,0.3212682723999023,0.5,0.15,0.55,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.09999999999999998,0.25,20,6.752500760297014
2,0.841794091463089,0.8693173259496689,0.5610660910606384,0.3082512348890305,0.47058823529411764,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.669375324281317
3,0.999201609807856,0.8470360496464897,0.5610660910606384,0.2859699585858513,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,5.677605759694807
4,1.1104223827521007,0.8453613718350729,0.5610660910606384,0.28429528077443444,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,5.164735627765904
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed42\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8923957884311676,0.8723622173070907,0.5589107871055603,0.31345143020153043,0.3,0.15,0.575,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.04999999999999999,0.25,20,6.103437162411715
2,0.872515293955803,0.865157762169838,0.5589107871055603,0.30624697506427767,0.35294117647058826,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.4345470854452715
3,0.9546124847496257,0.8548243115929997,0.5589107871055603,0.29591352448743935,0.16666666666666666,0.6666666666666666,0.45588235294117646,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.0,0.35294117647058826,0.2941176470588235,17,5.942801823327642
4,1.107910007238388,0.8493453065554301,0.5589107871055603,0.29043451944986975,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,5.176447549529605
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.873749478161335,0.8733440309762954,0.5730632543563843,0.30028077661991115,0.2,0.45,0.4625,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.03750000000000003,-0.05,-0.2,20,6.373021189631005
2,1.2112938165664673,0.811827102303505,0.5730632543563843,0.2387638479471207,0.45,0.45,0.4,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.050000000000000044,-0.05,-0.2,20,4.8206314512774195
3,1.3260086983442307,0.8005891263484954,0.5730632543563843,0.22752587199211116,0.35,0.3,0.45,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.21250000000000002,-0.9,0.19999999999999996,20,4.299114669834676
4,2.3214606285095214,0.6903034225106239,0.5730632543563843,0.11724016815423965,0.35,0.55,0.325,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.15000000000000002,1.0,0.19999999999999996,20,2.4998015489839713
5,1.6014830470085144,0.830154636502266,0.5730632543563843,0.2570913821458817,,,0.325,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.21250000000000002,0.0,0.19999999999999996,20,3.5932854126518086
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_mlp_seed42\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.847090245783329,0.8766189932823181,0.560093343257904,0.316525650024414,0.3,0.15,0.525,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.25,20,6.429871723589245
2,0.8819941371679306,0.8635502517223358,0.560093343257904,0.30345690846443174,0.29411764705882354,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.365394626949534
3,0.9845441264264724,0.8511588678640478,0.560093343257904,0.29106552460614377,0.16666666666666666,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.762131592346747
4,1.1676496068636577,0.8446482519308726,0.560093343257904,0.2845549086729685,,,0.16666666666666666,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,-0.041666666666666685,0.0,0.16666666666666663,6,4.911608763756592
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.894 | 0.350 | 0.150 | 0.537 | 0.562 | 0.7347 |
| 2 | 0.876 | 0.471 | 0.588 | 0.463 | 0.512 | 0.8019 |
| 3 | 0.853 | 0.000 | 0.667 | 0.529 | 0.471 | 0.9844 |
| 4 | 0.829 | nan | nan | 0.208 | 0.417 | 1.3628 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.062 | 0.025 | +0.327 |
| Shuffled Action | -0.075 | 0.037 | +0.317 |
| Constant Action | -0.025 | -0.012 | +0.313 |
| Blind | -0.050 | 0.012 | +0.321 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2394.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_mlp_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.882 | +0.106 |
| 2 | 1.000 | 0.876 | 0.869 | +0.124 |
| 3 | 1.000 | 0.853 | 0.847 | +0.147 |
| 4 | 1.000 | 0.829 | 0.845 | +0.171 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.537 | 0.550 | +0.025 |
| 2 | 0.512 | 0.463 | 0.463 | +0.050 |
| 3 | 0.471 | 0.529 | 0.500 | -0.059 |
| 4 | 0.417 | 0.208 | 0.333 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.500 | -0.200 |
| 2 | 0.588 | 0.471 | 0.471 | +0.118 |
| 3 | 0.667 | 0.000 | 0.333 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.096 | -1.000 | +0.429 |
| 2 | +0.055 | +0.000 | +0.000 |
| 3 | +0.038 | -1.000 | -1.000 |
| 4 | -0.106 | -1.500 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.882 | 0.287 | 0.562 | 0.537 | 0.550 | 0.600 |
| 2 | 1.000 | 0.876 | 0.869 | 0.246 | 0.512 | 0.463 | 0.463 | 0.487 |
| 3 | 1.000 | 0.853 | 0.847 | 0.241 | 0.471 | 0.529 | 0.500 | 0.456 |
| 4 | 1.000 | 0.829 | 0.845 | 0.237 | 0.417 | 0.208 | 0.333 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1370
**Average Oracle Gain (state probe):** +0.0561
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_mlp_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_mlp_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_mlp_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.3286378383636475,2.8200210821433145
1,1,2.82230806350708,2.532857676021388
2,2,2.4865007400512695,2.2707947277631915
3,3,2.1637978553771973,2.0652892315974003
4,4,1.8978708982467651,1.9022121742123463
5,5,1.6851054430007935,1.7882709190493724
6,6,1.5316880941390991,1.7135492543705175
7,7,1.4291799068450928,1.6671230128554047
8,8,1.3625634908676147,1.63485592701396
9,9,1.3135594129562378,1.6102877757588372
10,10,1.2715474367141724,1.585986153024142
11,11,1.2296807765960693,1.5566886526639345
12,12,1.1840994358062744,1.5225634965740267
13,13,1.134629487991333,1.4902312481989626
14,14,1.085972547531128,1.4629379022316855
15,15,1.0411717891693115,1.4384310362768955
16,16,0.9998140335083008,1.4147501460841445
17,17,0.9624283313751221,1.3938870039142546
18,18,0.9286254644393921,1.3794955894595287
19,19,0.8986238241195679,1.3688110601706582
20,20,0.8705886006355286,1.3580371043721184
21,21,0.8429641127586365,1.345839328453189
22,22,0.815689742565155,1.3334704539814934
23,23,0.7884190082550049,1.3216261316518314
24,24,0.7616122364997864,1.3122649896340293
25,25,0.7365407943725586,1.3043112832991803
26,26,0.7136057615280151,1.294710253105789
27,27,0.6920284032821655,1.2853374793881276
28,28,0.6723110675811768,1.2786602583087858
29,29,0.6537231802940369,1.2741048844134222
30,30,0.6362931728363037,1.2681434506275615
31,31,0.6190634369850159,1.2633259257332223
32,32,0.6026971936225891,1.2608092261142418
33,33,0.5868135094642639,1.2584609985351562
34,34,0.5714294910430908,1.2549393450627562
35,35,0.556846022605896,1.2520688166383838
36,36,0.5429412126541138,1.2530139860559681
37,37,0.5297498106956482,1.250976812644083
38,38,0.5169344544410706,1.2477204369716957
39,39,0.5044084787368774,1.2478670214043288
40,40,0.4924580156803131,1.2487272669057377
41,41,0.48059776425361633,1.2463410174260374
42,42,0.46915292739868164,1.2442274249967982
43,43,0.4581061899662018,1.2426917904713115
44,44,0.44724133610725403,1.2428334970943262
45,45,0.43683624267578125,1.2438896054127178
46,46,0.42663809657096863,1.2433073950595543
47,47,0.4167173504829407,1.2425563374503714
48,48,0.40709173679351807,1.243427339147349
49,49,0.3978527784347534,1.2414175565125511
50,50,0.3886769413948059,1.2413052418192878
51,51,0.3801364302635193,1.2430274838306865
52,52,0.3714349865913391,1.2431973316630378
53,53,0.36335188150405884,1.2411706643026383
54,54,0.3551316261291504,1.2413422631435707
55,55,0.3472094237804413,1.241457079277664
56,56,0.33938291668891907,1.2394710603307506
57,57,0.33186987042427063,1.239076958327997
58,58,0.32452377676963806,1.23876702980917
59,59,0.31755274534225464,1.2383880615234375
60,60,0.31072571873664856,1.236651936515433
61,61,0.3041490614414215,1.2359255180984248
62,62,0.2977373003959656,1.2366096621654072
63,63,0.2916161119937897,1.2338538248030866
64,64,0.28562599420547485,1.2368701872278431
65,65,0.2800111174583435,1.2335737885021774
66,66,0.2745484411716461,1.236788390112705
67,67,0.2696550786495209,1.2330977643122438
68,68,0.26521196961402893,1.2389822787925846
69,69,0.26116153597831726,1.2339420005923412
70,70,0.25595805048942566,1.2349073066086065
71,71,0.24997416138648987,1.2345552288117956
72,72,0.2448352873325348,1.234252179255251
73,73,0.2414163202047348,1.2371562269867444
74,74,0.2380390763282776,1.233019093998143
75,75,0.23344700038433075,1.2340205458344007
76,76,0.22871321439743042,1.2353478103387552
77,77,0.22528022527694702,1.2333218934106045
78,78,0.22264157235622406,1.237609362993084
79,79,0.219275563955307,1.2342056524558145
80,80,0.21541383862495422,1.2341148501536885
81,81,0.21191099286079407,1.2387017422035091
82,82,0.20940588414669037,1.2327455614433913
83,83,0.20695465803146362,1.2399261974897542
84,84,0.20402425527572632,1.2343642438044313
85,85,0.200889453291893,1.2367733814677253
86,86,0.19821935892105103,1.2373623457111296
87,87,0.19619379937648773,1.2376326263928024
88,88,0.19416794180870056,1.2375579583840293
89,89,0.19178196787834167,1.239045690317623
90,90,0.18931794166564941,1.2346715458103867
91,91,0.18729567527770996,1.2431031524157914
92,92,0.18549761176109314,1.2350673988217213
93,93,0.18310125172138214,1.2400387623271003
94,94,0.18049713969230652,1.238004590644211
95,95,0.17832070589065552,1.2362401993548284
96,96,0.17675937712192535,1.2427480728899847
97,97,0.17527388036251068,1.2359998108910732
98,98,0.1735161989927292,1.2399065611792393
99,99,0.17164921760559082,1.239399268978932

```


## File: `reports\multiseed_mlp_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2405192852020264,2.794680986248079
1,1,2.792833089828491,2.4885371473969005
2,2,2.427370548248291,2.212135940301614
3,3,2.091857433319092,2.004073221175397
4,4,1.8312478065490723,1.8478598672835553
5,5,1.6297669410705566,1.7389590153928662
6,6,1.4853794574737549,1.6755756315637806
7,7,1.396022081375122,1.6373986416175716
8,8,1.3394242525100708,1.6117613745517418
9,9,1.2982933521270752,1.590558786861232
10,10,1.2610399723052979,1.5650139480340677
11,11,1.2188657522201538,1.534974895539831
12,12,1.1723061800003052,1.5049478499615778
13,13,1.1265126466751099,1.476790256187564
14,14,1.0837352275848389,1.4530211902055583
15,15,1.0444492101669312,1.4330301753810195
16,16,1.008631944656372,1.415619646916624
17,17,0.9764534831047058,1.4009455696481172
18,18,0.9477984309196472,1.3877482179735527
19,19,0.9202165007591248,1.374971983862705
20,20,0.8930167555809021,1.363375304175205
21,21,0.866360604763031,1.3529949500912526
22,22,0.8407472372055054,1.3428414766905739
23,23,0.815500020980835,1.33295765861136
24,24,0.7906316518783569,1.324267903312308
25,25,0.7677453756332397,1.3154845941262168
26,26,0.7462475299835205,1.307862078557249
27,27,0.7262309789657593,1.3015668274926357
28,28,0.7076115012168884,1.295882303206647
29,29,0.689828634262085,1.2908935546875
30,30,0.6730250120162964,1.2861158027023565
31,31,0.6564818024635315,1.2825576281938396
32,32,0.6406600475311279,1.2797592663374104
33,33,0.6254147291183472,1.277175528104188
34,34,0.6105501651763916,1.2745059904504994
35,35,0.5962384939193726,1.272249315605789
36,36,0.5829029679298401,1.2680606529360912
37,37,0.5696390867233276,1.2664822437724128
38,38,0.5574650168418884,1.2658346207415472
39,39,0.5452470183372498,1.2654221331486937
40,40,0.5337750315666199,1.2640478415567367
41,41,0.5222623348236084,1.2637604260053792
42,42,0.5112267732620239,1.264070854812372
43,43,0.5006635189056396,1.2656211227667136
44,44,0.4901062250137329,1.2661429233238346
45,45,0.479924738407135,1.2657199296794954
46,46,0.47006601095199585,1.266724133100666
47,47,0.46037131547927856,1.2682244973104508
48,48,0.45103734731674194,1.2681316938556608
49,49,0.44188371300697327,1.2678790483318392
50,50,0.4330584704875946,1.2685930846167393
51,51,0.4245341420173645,1.2713280349481302
52,52,0.41640907526016235,1.2713833167904713
53,53,0.40833139419555664,1.2708269963498975
54,54,0.4007585942745209,1.272166142698194
55,55,0.3931291401386261,1.2737694411981302
56,56,0.3860588073730469,1.2737154100762038
57,57,0.3789764642715454,1.2740787443567494
58,58,0.37203362584114075,1.2754527858046234
59,59,0.36547449231147766,1.2766623575179303
60,60,0.3589401841163635,1.276567928126601
61,61,0.35260704159736633,1.2768387090964395
62,62,0.3466321527957916,1.2794422087122181
63,63,0.340663880109787,1.2794131919985912
64,64,0.3349045217037201,1.2792320876825052
65,65,0.3294200301170349,1.2798877153240267
66,66,0.32403501868247986,1.280866404048732
67,67,0.31887373328208923,1.2819263896004098
68,68,0.31383445858955383,1.281840214963819
69,69,0.308988094329834,1.2826345474993597
70,70,0.3042106330394745,1.2838612540823515
71,71,0.29961302876472473,1.284074752057185
72,72,0.2952089011669159,1.2858682851322363
73,73,0.2909119427204132,1.2862093565893955
74,74,0.28701573610305786,1.2880926913902409
75,75,0.28295665979385376,1.2876466844902663
76,76,0.27926161885261536,1.2897012429159196
77,77,0.2756378948688507,1.2888408723424694
78,78,0.27218878269195557,1.2904890717053024
79,79,0.2688278555870056,1.2940098496734118
80,80,0.2655234932899475,1.2896670982485912
81,81,0.2621432840824127,1.298992094446401
82,82,0.25903165340423584,1.2902274209944928
83,83,0.2558375597000122,1.298401879482582
84,84,0.2527175545692444,1.2971793002769596
85,85,0.24956606328487396,1.2970408455270235
86,86,0.24656996130943298,1.3004915831518955
87,87,0.24399764835834503,1.296110559682377
88,88,0.24152138829231262,1.3045327858846696
89,89,0.23891356587409973,1.298828125
90,90,0.23602651059627533,1.3026190585777409
91,91,0.2332375943660736,1.3049834204501793
92,92,0.23101989924907684,1.3010824234759222
93,93,0.22886726260185242,1.3077194964299437
94,94,0.22696076333522797,1.3032997006275615
95,95,0.22502300143241882,1.3106660686555456
96,96,0.22302944958209991,1.3062037483590547
97,97,0.2212851643562317,1.3102044277503841
98,98,0.2194022834300995,1.3118972778320312
99,99,0.21736563742160797,1.3082535540471312

```


## File: `reports\multiseed_mlp_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7475928485393524,0.8923725098371506,0.5667760968208313,0.32559641301631925,0.4,0.15,0.5625,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.19999999999999996,20,7.285625630759072
2,0.7790270507335663,0.8805482238531113,0.5667760968208313,0.31377212703227997,0.4117647058823529,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,7.206734010639449
3,1.0332163309349733,0.8478225960451014,0.5667760968208313,0.28104649922427005,0.16666666666666666,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,5.49069216686476
4,1.4119989673296611,0.8277252614498138,0.5667760968208313,0.26094916462898254,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,4.061644643348748
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.802408616244793,0.8828461945056916,0.5610660910606384,0.32178010344505314,0.5,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.787915169929901
2,0.823888412117958,0.8726116776466369,0.5610660910606384,0.3115455865859985,0.5294117647058824,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.814321768766341
3,0.9349748772733352,0.8568151488023645,0.5610660910606384,0.2957490577417261,0.5,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,6.067620588358239
4,0.9945588807264963,0.859216958284378,0.5610660910606384,0.2981508672237396,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,5.7664137872653685
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed43\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9084201142191887,0.8713143527507782,0.5589107871055603,0.3124035656452179,0.4,0.15,0.5875,0.5625,0.3,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.012499999999999956,-0.04999999999999999,0.25,20,5.995773908388256
2,0.870723882317543,0.8660576164722442,0.5589107871055603,0.3071468293666839,0.4117647058823529,0.5882352941176471,0.525,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.03750000000000003,0.30000000000000004,0.30000000000000004,20,6.4477854067660525
3,0.9973922827664543,0.8493603362756617,0.5589107871055603,0.29044954917010135,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,5.687905263519863
4,1.1717549761136372,0.8403998812039694,0.5589107871055603,0.28148909409840905,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,4.894400415597036
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8860853612422943,0.8726241379976273,0.5730632543563843,0.299560883641243,0.25,0.45,0.475,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.2,20,6.284297407807611
2,1.198817077279091,0.815801352262497,0.5730632543563843,0.24273809790611267,0.4,0.45,0.3625,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.012500000000000011,-0.05,-0.2,20,4.870802376398561
3,1.4611327260732652,0.7853487014770508,0.5730632543563843,0.21228544712066655,0.45,0.3,0.45,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.21250000000000002,-0.9,0.19999999999999996,20,3.9015370374328464
4,2.4815728187561037,0.6814877927303314,0.5730632543563843,0.10842453837394717,0.45,0.55,0.35,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.175,1.0,0.19999999999999996,20,2.3385132328948837
5,1.8818023025989532,0.8112014710903168,0.5730632543563843,0.23813821673393254,,,0.325,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.21250000000000002,0.0,0.19999999999999996,20,3.0580181900496237
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_mlp_seed43\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8544470354914665,0.8764377892017364,0.560093343257904,0.3163444459438324,0.35,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.374510522536522
2,0.8801637291908264,0.8646099746227265,0.560093343257904,0.30451663136482243,0.35294117647058826,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.378632242538734
3,1.041376482037937,0.8453580456621507,0.560093343257904,0.2852647024042466,0.16666666666666666,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,5.447667498539436
4,1.339026262362798,0.8332618276278178,0.560093343257904,0.2731684843699137,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,4.282991456753535
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.400 | 0.150 | 0.562 | 0.562 | 0.7476 |
| 2 | 0.881 | 0.412 | 0.588 | 0.450 | 0.512 | 0.7790 |
| 3 | 0.848 | 0.167 | 0.667 | 0.515 | 0.471 | 1.0332 |
| 4 | 0.828 | nan | nan | 0.250 | 0.417 | 1.4120 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.037 | 0.000 | +0.326 |
| Shuffled Action | -0.037 | 0.000 | +0.316 |
| Constant Action | -0.012 | -0.025 | +0.312 |
| Blind | -0.037 | 0.000 | +0.322 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2556.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_mlp_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | +0.108 |
| 2 | 1.000 | 0.881 | 0.873 | +0.119 |
| 3 | 1.000 | 0.848 | 0.857 | +0.152 |
| 4 | 1.000 | 0.828 | 0.859 | +0.172 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.562 | 0.562 | +0.000 |
| 2 | 0.512 | 0.450 | 0.463 | +0.062 |
| 3 | 0.471 | 0.515 | 0.471 | -0.044 |
| 4 | 0.417 | 0.250 | 0.333 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.500 | -0.250 |
| 2 | 0.588 | 0.412 | 0.529 | +0.176 |
| 3 | 0.667 | 0.167 | 0.500 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.081 | n/a | +0.286 |
| 2 | +0.062 | -0.250 | -2.000 |
| 3 | -0.063 | n/a | -2.000 |
| 4 | -0.224 | -1.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | 0.287 | 0.562 | 0.562 | 0.562 | 0.600 |
| 2 | 1.000 | 0.881 | 0.873 | 0.246 | 0.512 | 0.450 | 0.463 | 0.487 |
| 3 | 1.000 | 0.848 | 0.857 | 0.241 | 0.471 | 0.515 | 0.471 | 0.456 |
| 4 | 1.000 | 0.828 | 0.859 | 0.237 | 0.417 | 0.250 | 0.333 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1379
**Average Oracle Gain (state probe):** +0.0463
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_mlp_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_mlp_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_mlp_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.267935037612915,2.799406958408043
1,1,2.8100831508636475,2.5357463398917774
2,2,2.5040910243988037,2.292244582879739
3,3,2.205570697784424,2.085326398005251
4,4,1.938297986984253,1.9261470857213756
5,5,1.7243086099624634,1.8106070346519596
6,6,1.5651999711990356,1.7279116521116162
7,7,1.4535584449768066,1.6702181706663037
8,8,1.3772187232971191,1.6324950671586833
9,9,1.321855068206787,1.6062101770619877
10,10,1.2764308452606201,1.580735378578061
11,11,1.2321298122406006,1.5515215514136143
12,12,1.186069369316101,1.5202701756211578
13,13,1.1405177116394043,1.4909495369332735
14,14,1.0964224338531494,1.465424334416624
15,15,1.054550290107727,1.4440079986071976
16,16,1.015906810760498,1.4251806540567367
17,17,0.98050856590271,1.4081280817751025
18,18,0.9469848275184631,1.392439670250064
19,19,0.9155922532081604,1.3786040759477458
20,20,0.8862864375114441,1.36638316170114
21,21,0.85832679271698,1.354065941982582
22,22,0.8309170603752136,1.3425011556656634
23,23,0.8043003082275391,1.330167426437628
24,24,0.7792515158653259,1.319336938076332
25,25,0.7555304765701294,1.3096657424676614
26,26,0.7329641580581665,1.2992628754162399
27,27,0.7108453512191772,1.291090543152856
28,28,0.6902409791946411,1.284823058081455
29,29,0.6710370779037476,1.278791458880315
30,30,0.6526572704315186,1.2734696435146644
31,31,0.6355078816413879,1.2698104107966188
32,32,0.6189865469932556,1.2659363043112832
33,33,0.6024453639984131,1.2616280727699154
34,34,0.5873110294342041,1.2582289898981813
35,35,0.572403073310852,1.2563205156169954
36,36,0.5586004257202148,1.2540765981205175
37,37,0.5452406406402588,1.2514046840980404
38,38,0.5325444936752319,1.2487781712266266
39,39,0.5202866196632385,1.2480288646260247
40,40,0.5084138512611389,1.2475267003794186
41,41,0.49683964252471924,1.2461127609503073
42,42,0.4856531322002411,1.2439122434522285
43,43,0.4745732545852661,1.242433141489498
44,44,0.46436503529548645,1.24368161060771
45,45,0.45404052734375,1.2433118976530482
46,46,0.4440450072288513,1.2424849213146774
47,47,0.4344947934150696,1.242052547267226
48,48,0.4253460466861725,1.2405187888223617
49,49,0.4159895181655884,1.24116828793385
50,50,0.4073619246482849,1.2414769657322617
51,51,0.3986462950706482,1.2402461317719007
52,52,0.3905481994152069,1.239930199795082
53,53,0.3824675977230072,1.2409909357790088
54,54,0.37459486722946167,1.2406581190765882
55,55,0.36697468161582947,1.2400423894163037
56,56,0.35953566431999207,1.2390561963691087
57,57,0.35241782665252686,1.2400106211177637
58,58,0.345437228679657,1.2392970851210297
59,59,0.3394113779067993,1.2403089179367315
60,60,0.33343881368637085,1.238036609086834
61,61,0.3277797996997833,1.239655541591957
62,62,0.32066261768341064,1.2362393238505378
63,63,0.3133164048194885,1.2364286829213627
64,64,0.30759212374687195,1.2387317594934681
65,65,0.3028804063796997,1.2365640108702614
66,66,0.2973560690879822,1.2380190990010247
67,67,0.29103896021842957,1.2368664350665983
68,68,0.2856484651565552,1.2376912851802637
69,69,0.28107550740242004,1.2401032995005123
70,70,0.2765764594078064,1.237138466756852
71,71,0.27117204666137695,1.238362171610848
72,72,0.2662312984466553,1.2388094292312373
73,73,0.26192745566368103,1.2392937081759092
74,74,0.25797632336616516,1.2408220885229893
75,75,0.25391536951065063,1.2382883791063652
76,76,0.24939070641994476,1.240555184786437
77,77,0.24503394961357117,1.240264892578125
78,78,0.24114863574504852,1.2406666239754098
79,79,0.23765237629413605,1.24272593513864
80,80,0.23439264297485352,1.240463882196145
81,81,0.23108607530593872,1.2431165351242315
82,82,0.22762148082256317,1.2420384141265368
83,83,0.22405444085597992,1.2422564146948643
84,84,0.22064527869224548,1.2440673327836833
85,85,0.21743758022785187,1.2426580210201075
86,86,0.21442952752113342,1.2441298688044313
87,87,0.21160611510276794,1.2440391915743467
88,88,0.2089391052722931,1.2453112993084017
89,89,0.20646855235099792,1.2463357644002946
90,90,0.2043086588382721,1.247233781658235
91,91,0.20250263810157776,1.2474843009573515
92,92,0.20076881349086761,1.2512601008180713
93,93,0.19881747663021088,1.2462093165663422
94,94,0.19627073407173157,1.2523377215276
95,95,0.19302664697170258,1.2463587776559297
96,96,0.18973952531814575,1.249390398869749
97,97,0.1873626857995987,1.2514523365458503
98,98,0.18608595430850983,1.2476197539782914
99,99,0.18492086231708527,1.2555577012359118

```


## File: `reports\multiseed_mlp_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.302522897720337,2.8367414630827357
1,1,2.84743070602417,2.580047607421875
2,2,2.546755313873291,2.34133035628522
3,3,2.2548840045928955,2.1349349725441855
4,4,1.994018316268921,1.965562539022477
5,5,1.772619366645813,1.830183560730981
6,6,1.592383861541748,1.7309649107886143
7,7,1.460728406906128,1.6672763511782787
8,8,1.3751972913742065,1.6309140314821338
9,9,1.3200435638427734,1.6095008224737448
10,10,1.2782803773880005,1.590879596647669
11,11,1.2407195568084717,1.5691593357774078
12,12,1.2030752897262573,1.543995716532723
13,13,1.1617722511291504,1.51684320168417
14,14,1.1173298358917236,1.490386462602459
15,15,1.0733319520950317,1.4630827356557377
16,16,1.0308893918991089,1.4392143624727842
17,17,0.9909740686416626,1.420316852507044
18,18,0.954482913017273,1.4053128351930713
19,19,0.9216555953025818,1.3900963204805967
20,20,0.8915392160415649,1.3755855872982838
21,21,0.8630073070526123,1.3636652211673925
22,22,0.8355136513710022,1.35374638291656
23,23,0.8095722198486328,1.3430698582383453
24,24,0.7843093276023865,1.3332627093205687
25,25,0.7599762678146362,1.3251765516937757
26,26,0.7364839911460876,1.3175559122054303
27,27,0.7141919732093811,1.3084461649910348
28,28,0.69318687915802,1.3007744961097591
29,29,0.6733778715133667,1.2961986103995902
30,30,0.6546301245689392,1.2918868768410605
31,31,0.6368712782859802,1.2871330136158428
32,32,0.6199512481689453,1.282590522140753
33,33,0.6038998961448669,1.2795842905513575
34,34,0.5893278121948242,1.2788968946112962
35,35,0.5744129419326782,1.276905122350474
36,36,0.5604200959205627,1.2751853817799053
37,37,0.5467714071273804,1.2745931656634222
38,38,0.5337215065956116,1.2718238205206198
39,39,0.5212972164154053,1.2700451710185066
40,40,0.5094190239906311,1.2719836625896517
41,41,0.49810299277305603,1.2715505381099512
42,42,0.4872353672981262,1.2694647116739242
43,43,0.4770296514034271,1.2716277075595543
44,44,0.46662437915802,1.2717075035220287
45,45,0.4570474624633789,1.2704847992443649
46,46,0.44755810499191284,1.2729463420930456
47,47,0.43840837478637695,1.2747716434666367
48,48,0.42949560284614563,1.2747582607581966
49,49,0.4209798574447632,1.2750192861087988
50,50,0.41268813610076904,1.274489480941022
51,51,0.4046979546546936,1.2766335909483864
52,52,0.39698606729507446,1.2789134041207735
53,53,0.3893657922744751,1.2766781165951588
54,54,0.38180848956108093,1.2776061511430583
55,55,0.3747410476207733,1.2807013089539574
56,56,0.36773088574409485,1.2799022236808402
57,57,0.3610393702983856,1.2805423423892162
58,58,0.35454317927360535,1.2828670564245006
59,59,0.3484589159488678,1.2822395699923155
60,60,0.342284619808197,1.2831344604492188
61,61,0.3365653157234192,1.2843722984439037
62,62,0.33087560534477234,1.2848953497214395
63,63,0.32555481791496277,1.2865628101786628
64,64,0.3202686607837677,1.286388084536693
65,65,0.3152705430984497,1.2859081831134733
66,66,0.3102930784225464,1.2891405449538935
67,67,0.3055998980998993,1.2888939028880635
68,68,0.30100682377815247,1.2894368406201973
69,69,0.29662466049194336,1.2902661933273565
70,70,0.292263001203537,1.2899294993916497
71,71,0.28804534673690796,1.291624225553919
72,72,0.2839744985103607,1.2924674612576845
73,73,0.2800723612308502,1.2926695776767418
74,74,0.2763698697090149,1.2935479586241676
75,75,0.272696852684021,1.2938700191310195
76,76,0.269070029258728,1.2960327648725667
77,77,0.26560884714126587,1.2960335153048155
78,78,0.2622106075286865,1.2962686507428278
79,79,0.259050577878952,1.2990227370965677
80,80,0.2560470402240753,1.2988934126056608
81,81,0.2532562017440796,1.3015430638047516
82,82,0.25080618262290955,1.2971843031586194
83,83,0.24844735860824585,1.3013085537269466
84,84,0.24588674306869507,1.3040477565077484
85,85,0.24273671209812164,1.3010031278016136
86,86,0.2396439164876938,1.3044250988569417
87,87,0.23712602257728577,1.3026638343685963
88,88,0.23509946465492249,1.3066133592949538
89,89,0.2328440546989441,1.3050049328413167
90,90,0.23037287592887878,1.3044273501536885
91,91,0.2279541790485382,1.3095583055840163
92,92,0.22587670385837555,1.3050749731845543
93,93,0.22400671243667603,1.3109182138912012
94,94,0.22213202714920044,1.3081820128393955
95,95,0.22002924978733063,1.3094267297963627
96,96,0.21797004342079163,1.3120927654328893
97,97,0.21617862582206726,1.3114363873591188
98,98,0.21451932191848755,1.3139313244428792
99,99,0.21278414130210876,1.3117883400838883

```


## File: `reports\multiseed_mlp_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7441744327545166,0.892347127199173,0.5667760968208313,0.3255710303783417,0.35,0.15,0.525,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.19999999999999996,20,7.3190926467735915
2,0.8090398833155632,0.8753514438867569,0.5667760968208313,0.3085753470659256,0.4117647058823529,0.5882352941176471,0.4125,0.5125,0.5,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.07500000000000001,0.25,0.30000000000000004,20,6.939386867705162
3,1.0106044551905464,0.8477604073636672,0.5667760968208313,0.2809843105428359,0.0,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,5.613544236623977
4,1.1650018692016602,0.8445406456788381,0.5667760968208313,0.27776454885800683,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,4.922771536837556
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8015805631875992,0.8832017719745636,0.5610660910606384,0.3221356809139252,0.45,0.15,0.5625,0.5625,0.25,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.25,20,6.7949272585041065
2,0.7997875198721885,0.8764950513839722,0.5610660910606384,0.3154289603233338,0.35294117647058826,0.5882352941176471,0.4875,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.0,0.30000000000000004,0.30000000000000004,20,7.019665351400994
3,0.9860272127039292,0.8496432164136101,0.5610660910606384,0.28857712535297164,0.3333333333333333,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,5.753464754166819
4,1.093090335528056,0.8477148016293844,0.5610660910606384,0.28664871056874597,,,0.375,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.16666666666666666,0.0,0.16666666666666663,6,5.246627708311014
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed44\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8380402401089668,0.8787720143795014,0.5589107871055603,0.31986122727394106,0.5,0.15,0.55,0.5625,0.3,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.04999999999999999,0.19999999999999996,20,6.499307978316509
2,0.8457772105932235,0.8689068645238877,0.5589107871055603,0.30999607741832735,0.35294117647058826,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.637966442477136
3,1.009387359899633,0.8457478354958927,0.5589107871055603,0.2868370483903324,0.16666666666666666,0.6666666666666666,0.4117647058823529,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.04411764705882354,0.35294117647058826,0.2941176470588235,17,5.62031291486105
4,1.1412313580513,0.8429764807224274,0.5589107871055603,0.28406569361686707,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,5.025307096241464
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8742807537317276,0.8734544515609741,0.5730632543563843,0.30039119720458984,0.25,0.45,0.45,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,6.369148485750479
2,1.2132167667150497,0.8120376825332641,0.5730632543563843,0.23897442817687986,0.4,0.45,0.3875,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.03750000000000003,-0.05,-0.2,20,4.812990744175592
3,1.3712615966796875,0.7939173996448516,0.5730632543563843,0.22085414528846736,0.35,0.3,0.4375,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.2,-0.9,0.19999999999999996,20,4.157239917739548
4,2.2634889602661135,0.6974708706140518,0.5730632543563843,0.1244076162576675,0.35,0.55,0.3625,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.1875,1.0,0.19999999999999996,20,2.5638255705789423
5,1.6364958703517913,0.8230886071920395,0.5730632543563843,0.2500253528356552,,,0.35,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.2375,0.0,0.19999999999999996,20,3.5164070839896615
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_mlp_seed44\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8409855768084527,0.8773220717906952,0.560093343257904,0.31722872853279116,0.35,0.15,0.525,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.07499999999999996,-0.09999999999999998,0.19999999999999996,20,6.476545815875575
2,0.9290553748607635,0.8562489420175552,0.560093343257904,0.29615559875965114,0.29411764705882354,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.042955989110053
3,1.0401546779800863,0.8424559726434595,0.560093343257904,0.28236262938555545,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,5.454066529757045
4,1.1432494521141052,0.8451468348503113,0.560093343257904,0.2850534915924072,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,5.016436291714995
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_mlp_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `mlp`
- **Transition Parameters:** 2,108,992
- **d_model:** 2048
- **Bottleneck:** 512
- **Layers:** 1

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.892 | 0.350 | 0.150 | 0.525 | 0.562 | 0.7442 |
| 2 | 0.875 | 0.412 | 0.588 | 0.412 | 0.512 | 0.8090 |
| 3 | 0.848 | 0.000 | 0.667 | 0.485 | 0.471 | 1.0106 |
| 4 | 0.845 | nan | nan | 0.292 | 0.417 | 1.1650 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | -0.075 | 0.037 | +0.326 |
| Shuffled Action | -0.075 | 0.037 | +0.317 |
| Constant Action | -0.050 | 0.012 | +0.320 |
| Blind | -0.037 | 0.000 | +0.322 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2444.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_mlp_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | +0.108 |
| 2 | 1.000 | 0.875 | 0.876 | +0.125 |
| 3 | 1.000 | 0.848 | 0.850 | +0.152 |
| 4 | 1.000 | 0.845 | 0.848 | +0.155 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.525 | 0.562 | +0.037 |
| 2 | 0.512 | 0.412 | 0.487 | +0.100 |
| 3 | 0.471 | 0.485 | 0.515 | -0.015 |
| 4 | 0.417 | 0.292 | 0.375 | +0.125 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.450 | -0.200 |
| 2 | 0.588 | 0.412 | 0.353 | +0.176 |
| 3 | 0.667 | 0.000 | 0.333 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.078 | n/a | +0.333 |
| 2 | -0.009 | -3.000 | +0.250 |
| 3 | -0.013 | +0.667 | -1.000 |
| 4 | -0.021 | -2.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.892 | 0.883 | 0.287 | 0.562 | 0.525 | 0.562 | 0.600 |
| 2 | 1.000 | 0.875 | 0.876 | 0.246 | 0.512 | 0.412 | 0.487 | 0.487 |
| 3 | 1.000 | 0.848 | 0.850 | 0.241 | 0.471 | 0.485 | 0.515 | 0.456 |
| 4 | 1.000 | 0.845 | 0.848 | 0.237 | 0.417 | 0.292 | 0.375 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1350
**Average Oracle Gain (state probe):** +0.0619
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_mlp_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_mlp_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_mlp_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.2327730655670166,2.767448800509093
1,1,2.7623164653778076,2.4547309250128073
2,2,2.392094612121582,2.1928180632044056
3,3,2.0696756839752197,1.9792097748303024
4,4,1.7987982034683228,1.819063530593622
5,5,1.5931029319763184,1.723203315109503
6,6,1.4611284732818604,1.6693567995165215
7,7,1.3816152811050415,1.6300441554335297
8,8,1.3252047300338745,1.5987771456358864
9,9,1.2788115739822388,1.5739146998671234
10,10,1.2362101078033447,1.5488565163534196
11,11,1.1927878856658936,1.5192665975601947
12,12,1.1462289094924927,1.4906343553887038
13,13,1.1009089946746826,1.4653903148213372
14,14,1.0586496591567993,1.4437375928534837
15,15,1.0195891857147217,1.4242017151879482
16,16,0.9831456542015076,1.407396535404393
17,17,0.9498980641365051,1.3925419791800078
18,18,0.9187105894088745,1.3776463993260117
19,19,0.8883230686187744,1.3622031290023053
20,20,0.8591299653053284,1.3479539214587601
21,21,0.8301346302032471,1.3349357980196592
22,22,0.8026717305183411,1.3237272168769212
23,23,0.7762960195541382,1.312744390769083
24,24,0.7510297298431396,1.3019179047131149
25,25,0.7273929119110107,1.2933907430680072
26,26,0.7060506343841553,1.2872181876761015
27,27,0.68598872423172,1.2814725031618213
28,28,0.6672171354293823,1.2746012953461194
29,29,0.6497029066085815,1.2702010107822106
30,30,0.6330810189247131,1.2664352166848105
31,31,0.6165367364883423,1.2620187978275488
32,32,0.6008901000022888,1.2575753634093239
33,33,0.5854103565216064,1.2553617133468877
34,34,0.5707365274429321,1.2533776955526383
35,35,0.5564428567886353,1.2511086385758197
36,36,0.5432230234146118,1.2490419481621413
37,37,0.5302028656005859,1.247406756291624
38,38,0.5173760652542114,1.2450524001825052
39,39,0.5048884749412537,1.244936083183914
40,40,0.4931597411632538,1.244417284355789
41,41,0.48125556111335754,1.242662148397477
42,42,0.46999627351760864,1.2429448112112578
43,43,0.45943817496299744,1.2448147633036628
44,44,0.448784202337265,1.2448116365026256
45,45,0.43847575783729553,1.244099726442431
46,46,0.4285513162612915,1.2446342843477842
47,47,0.4187464416027069,1.2442837074154713
48,48,0.4092439115047455,1.2430105991050846
49,49,0.4001932740211487,1.2428982844118213
50,50,0.39125728607177734,1.2426146210217086
51,51,0.382550448179245,1.242474540335233
52,52,0.37408506870269775,1.2423372112336706
53,53,0.3658316433429718,1.2411462752545466
54,54,0.3578453063964844,1.2414458227939293
55,55,0.350305438041687,1.2403011634701588
56,56,0.342835396528244,1.2407396660476435
57,57,0.3355486989021301,1.2410848648821722
58,58,0.3284895718097687,1.2402794009349385
59,59,0.32164257764816284,1.240539050493084
60,60,0.31506460905075073,1.2395260920290088
61,61,0.3087065815925598,1.2405096585633324
62,62,0.3024384379386902,1.240220241859311
63,63,0.29658305644989014,1.2399264476338372
64,64,0.2906571328639984,1.240631103515625
65,65,0.285004585981369,1.2398013755923412
66,66,0.279649943113327,1.241191801477651
67,67,0.2744150459766388,1.2410470931256403
68,68,0.26926687359809875,1.2378890240778688
69,69,0.264373242855072,1.2419190953989498
70,70,0.2597942054271698,1.239074206743084
71,71,0.255469411611557,1.2418750700403431
72,72,0.25109273195266724,1.2393428614882172
73,73,0.2468419373035431,1.2416481893570697
74,74,0.24266692996025085,1.239168511062372
75,75,0.23831573128700256,1.2392330482357838
76,76,0.2337983250617981,1.2398323934586322
77,77,0.2298974245786667,1.2388241877321338
78,78,0.22613947093486786,1.2391581300829277
79,79,0.2227829098701477,1.240649989393891
80,80,0.22070957720279694,1.2416115432489114
81,81,0.21796655654907227,1.239884798644019
82,82,0.2156294882297516,1.2443942711001537
83,83,0.21139709651470184,1.2369817514888575
84,84,0.20845696330070496,1.2434612336705944
85,85,0.20514202117919922,1.2378490010245902
86,86,0.2018197923898697,1.2389871566022028
87,87,0.19902580976486206,1.2437228843814037
88,88,0.19633899629116058,1.23833978371542
89,89,0.19415274262428284,1.2426587714523565
90,90,0.1921701580286026,1.2403151715388063
91,91,0.18919286131858826,1.2434752417392418
92,92,0.1881590634584427,1.243041742043417
93,93,0.1848701685667038,1.2413540199154713
94,94,0.18246163427829742,1.2425492083440062
95,95,0.17979899048805237,1.242463909211706
96,96,0.17738334834575653,1.244733091260566
97,97,0.17568808794021606,1.2430975241739242
98,98,0.17422665655612946,1.2437243852459017
99,99,0.17274218797683716,1.2443590007844518

```


## File: `reports\multiseed_mlp_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.263410806655884,2.7920349621381915
1,1,2.7923760414123535,2.498685742987961
2,2,2.4426920413970947,2.245456883164703
3,3,2.126400947570801,2.029945623679239
4,4,1.8512285947799683,1.865478515625
5,5,1.638303518295288,1.744095724137103
6,6,1.4815291166305542,1.660686680527984
7,7,1.3745640516281128,1.6149417064228997
8,8,1.3124116659164429,1.5917833672195185
9,9,1.2752991914749146,1.5765994963098744
10,10,1.2466449737548828,1.5587869863041113
11,11,1.2155652046203613,1.53595220847208
12,12,1.1797438859939575,1.5110389834544697
13,13,1.1404948234558105,1.4866031584192494
14,14,1.0990980863571167,1.4634675823274206
15,15,1.0576841831207275,1.4408641877721569
16,16,1.0182316303253174,1.4201886536645107
17,17,0.9828294515609741,1.4029759891697617
18,18,0.9521332383155823,1.3893940409675973
19,19,0.9245439171791077,1.3784102142834274
20,20,0.8992728590965271,1.3686898653624489
21,21,0.8755947947502136,1.3594406628217854
22,22,0.8529478311538696,1.3498315029456966
23,23,0.8305674195289612,1.3399730744909069
24,24,0.8081185817718506,1.331178758965164
25,25,0.7860307693481445,1.3233701361984502
26,26,0.7642250061035156,1.3155467549308402
27,27,0.7436730265617371,1.3093486848424694
28,28,0.7242656350135803,1.3037793519066982
29,29,0.7058826684951782,1.298999348624808
30,30,0.6886149048805237,1.29399046350698
31,31,0.6721667051315308,1.289396442350794
32,32,0.6564215421676636,1.2863797047099128
33,33,0.6413283944129944,1.2847316304191214
34,34,0.626444399356842,1.2821542708600153
35,35,0.612061619758606,1.279786156826332
36,36,0.5983657240867615,1.2788662519611296
37,37,0.5845235586166382,1.2774753257876537
38,38,0.5717359781265259,1.2754771748527152
39,39,0.5591099858283997,1.2749553742955944
40,40,0.5472186207771301,1.2753549794681738
41,41,0.5357116460800171,1.2759009189293034
42,42,0.52443927526474,1.276153314309042
43,43,0.5137184858322144,1.27578860423604
44,44,0.5032526254653931,1.275409510878266
45,45,0.4930138885974884,1.2752830630443135
46,46,0.4830789268016815,1.2760520059554303
47,47,0.47337692975997925,1.2776574306800716
48,48,0.4639345407485962,1.2790432288998463
49,49,0.45483464002609253,1.2792737366723232
50,50,0.4459763765335083,1.2806918034788037
51,51,0.4373616874217987,1.2826735699763063
52,52,0.4292501211166382,1.2816219642514088
53,53,0.42104005813598633,1.2839185370773565
54,54,0.41327255964279175,1.2850251745005123
55,55,0.40567171573638916,1.2845874223552767
56,56,0.3983270525932312,1.2856988125160091
57,57,0.3912232518196106,1.286009991755251
58,58,0.3843083679676056,1.2869530349481302
59,59,0.3776601254940033,1.2876406810322747
60,60,0.37119340896606445,1.2880716792872695
61,61,0.3649362027645111,1.2877319836225667
62,62,0.3588589131832123,1.2882240170338115
63,63,0.3530540466308594,1.288910912685707
64,64,0.34737473726272583,1.2899317506883965
65,65,0.3418669104576111,1.2916034635950306
66,66,0.33654049038887024,1.2912032330622438
67,67,0.3314158022403717,1.2930935718974128
68,68,0.32650068402290344,1.292662823786501
69,69,0.32190725207328796,1.2951214899782275
70,70,0.3170872926712036,1.2936806600601947
71,71,0.31277239322662354,1.2959500922531377
72,72,0.3083682656288147,1.2982700535508453
73,73,0.3041675388813019,1.2988406322041497
74,74,0.30015403032302856,1.2974143106429303
75,75,0.2961548864841461,1.2976859671170595
76,76,0.29194530844688416,1.3008537917840677
77,77,0.287934809923172,1.3023611600281761
78,78,0.28415796160697937,1.3010729180007685
79,79,0.28067225217819214,1.3027089853755762
80,80,0.2773756980895996,1.3023140078685322
81,81,0.274211585521698,1.3042667576524078
82,82,0.2709978520870209,1.3045536729155993
83,83,0.26788583397865295,1.304993676357582
84,84,0.26477527618408203,1.3050498337042136
85,85,0.26147013902664185,1.3060410296330687
86,86,0.2582976222038269,1.30777102611104
87,87,0.2552628517150879,1.3086090087890625
88,88,0.25252729654312134,1.3085896226226306
89,89,0.25002291798591614,1.3090003592069033
90,90,0.24774982035160065,1.3131571285060195
91,91,0.245594322681427,1.3101502715564164
92,92,0.2436712682247162,1.316388864986232
93,93,0.24163666367530823,1.3110894375160091
94,94,0.23905883729457855,1.3158476782626793
95,95,0.2358856350183487,1.3144507486312116
96,96,0.23304630815982819,1.3142897809138063
97,97,0.2309645265340805,1.318464060298732
98,98,0.2291642278432846,1.3149078869428792
99,99,0.22737956047058105,1.3184498020860016

```


## File: `reports\multiseed_transformer_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7374177813529968,0.8945990145206452,0.5667760968208313,0.32782291769981386,0.35,0.15,0.6,0.5625,0.2,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,0.0,-0.14999999999999997,0.19999999999999996,20,7.386154438393182
2,0.8382326975464821,0.8712622970342636,0.5667760968208313,0.3044862002134323,0.4117647058823529,0.5882352941176471,0.5,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.6977114566905955
3,1.293735633878147,0.8181956059792462,0.5667760968208313,0.25141950915841493,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.385032510804086
4,2.3679478963216147,0.7528911530971527,0.5667760968208313,0.1861150562763214,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.421944355691832
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8097779333591462,0.8825434178113938,0.5610660910606384,0.32147732675075535,0.4,0.15,0.5625,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.09999999999999998,0.19999999999999996,20,6.726142309282738
2,0.9311441659927369,0.8573381125926971,0.5610660910606384,0.29627202153205867,0.35294117647058826,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.029400115227193
3,1.425560193903306,0.8029012539807487,0.5610660910606384,0.2418351629201103,0.16666666666666666,0.6666666666666666,0.4411764705882353,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.01470588235294118,0.35294117647058826,0.2941176470588235,17,3.979539299149513
4,2.5409632325172424,0.7440049548943838,0.5610660910606384,0.18293886383374536,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.257033068670963
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed42\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9644763365387916,0.8642714411020279,0.5589107871055603,0.3053606539964676,0.4,0.15,0.575,0.5625,0.3,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.04999999999999999,0.19999999999999996,20,5.6472941972189314
2,0.9376819759607316,0.8552541971206665,0.5589107871055603,0.2963434100151062,0.4117647058823529,0.5882352941176471,0.5,0.5125,0.6,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.012500000000000011,0.35,0.30000000000000004,20,5.987361264972049
3,1.1511231853681452,0.828207321026746,0.5589107871055603,0.2692965339211857,0.3333333333333333,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,4.928293415554026
4,1.7770475347836812,0.7808109720547994,0.5589107871055603,0.22190018494923913,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.227284543497934
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8754494741559029,0.8769952714443207,0.5730632543563843,0.3039320170879364,0.2,0.45,0.475,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.2,20,6.3606457061616535
2,1.2337322890758515,0.8118789613246917,0.5730632543563843,0.23881570696830745,0.5,0.45,0.425,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.07500000000000001,-0.05,-0.2,20,4.732956347646642
3,1.7215090811252594,0.7540328085422516,0.5730632543563843,0.18096955418586735,0.3,0.3,0.4125,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.175,-0.9,0.19999999999999996,20,3.3114338517772115
4,3.2678229808807373,0.6175306662917137,0.5730632543563843,0.04446741193532944,0.35,0.55,0.325,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.15000000000000002,1.0,0.19999999999999996,20,1.775858395331237
5,3.3534437894821165,0.7105202913284302,0.5730632543563843,0.13745703697204592,,,0.25,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.1375,0.0,0.19999999999999996,20,1.7160227016399656
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_transformer_seed42\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9122805297374725,0.8689174622297287,0.560093343257904,0.3088241189718246,0.3,0.15,0.575,0.5625,0.2,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.14999999999999997,0.19999999999999996,20,5.97040213086416
2,0.9594920247793197,0.8523672133684158,0.560093343257904,0.29227387011051176,0.4117647058823529,0.5882352941176471,0.5125,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.30000000000000004,0.30000000000000004,20,5.851263581915644
3,1.3213847279548645,0.8107349907650667,0.560093343257904,0.25064164750716267,0.16666666666666666,0.6666666666666666,0.4411764705882353,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.29327863030606
4,2.084279775619507,0.7671722869078318,0.560093343257904,0.20707894364992774,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.7515682439339826
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 0.350 | 0.150 | 0.600 | 0.562 | 0.7374 |
| 2 | 0.871 | 0.412 | 0.588 | 0.500 | 0.512 | 0.8382 |
| 3 | 0.818 | 0.167 | 0.667 | 0.471 | 0.471 | 1.2937 |
| 4 | 0.753 | nan | nan | 0.208 | 0.417 | 2.3679 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **2**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | +0.000 | -0.037 | +0.328 |
| Shuffled Action | -0.025 | -0.012 | +0.309 |
| Constant Action | -0.025 | -0.012 | +0.305 |
| Blind | -0.037 | 0.000 | +0.321 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~2 step(s)**.

Single-step transition validation MSE: 1.2872.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_transformer_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.883 | +0.105 |
| 2 | 1.000 | 0.871 | 0.857 | +0.129 |
| 3 | 1.000 | 0.818 | 0.803 | +0.182 |
| 4 | 1.000 | 0.753 | 0.744 | +0.247 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.600 | 0.562 | -0.037 |
| 2 | 0.512 | 0.500 | 0.475 | +0.012 |
| 3 | 0.471 | 0.471 | 0.441 | +0.000 |
| 4 | 0.417 | 0.208 | 0.250 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.400 | -0.200 |
| 2 | 0.588 | 0.412 | 0.353 | +0.176 |
| 3 | 0.667 | 0.167 | 0.167 | +0.500 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.103 | n/a | +0.200 |
| 2 | +0.098 | +0.667 | +0.250 |
| 3 | +0.078 | +1.000 | +0.000 |
| 4 | +0.035 | -0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.883 | 0.287 | 0.562 | 0.600 | 0.562 | 0.600 |
| 2 | 1.000 | 0.871 | 0.857 | 0.246 | 0.512 | 0.500 | 0.475 | 0.487 |
| 3 | 1.000 | 0.818 | 0.803 | 0.241 | 0.471 | 0.471 | 0.441 | 0.456 |
| 4 | 1.000 | 0.753 | 0.744 | 0.237 | 0.417 | 0.208 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1658
**Average Oracle Gain (state probe):** +0.0458
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_transformer_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_transformer_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_transformer_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.988105535507202,4.009040207159324
1,1,4.1091485023498535,3.3090620197233607
2,2,3.321253776550293,2.9205459844870645
3,3,2.8715968132019043,2.6043816238153177
4,4,2.4986989498138428,2.333981373271004
5,5,2.169788122177124,2.1122311451396003
6,6,1.8908488750457764,1.9491677206070697
7,7,1.6894599199295044,1.8361121005699284
8,8,1.5482985973358154,1.7605064267017803
9,9,1.4535256624221802,1.7068487698914574
10,10,1.3876310586929321,1.6712706518954918
11,11,1.3393480777740479,1.6497562596055328
12,12,1.3014540672302246,1.6277283215131917
13,13,1.2696348428726196,1.6032899950371413
14,14,1.234097957611084,1.572172571401127
15,15,1.1911842823028564,1.5379265957191341
16,16,1.143682837486267,1.5071172245213242
17,17,1.09994637966156,1.4819068283331198
18,18,1.0618605613708496,1.4607949178726947
19,19,1.023587942123413,1.4423011404569033
20,20,0.9907776117324829,1.4251736500224128
21,21,0.9633713960647583,1.4096752229284069
22,22,0.935771107673645,1.3956413894403177
23,23,0.9094994068145752,1.383715645211642
24,24,0.8857768774032593,1.372314703269083
25,25,0.8627215623855591,1.3605786933273565
26,26,0.8384600281715393,1.3495653496413935
27,27,0.8162406086921692,1.3404368416207735
28,28,0.7938172221183777,1.3307567658971569
29,29,0.7726207971572876,1.3209576215900358
30,30,0.7511121034622192,1.3125857994204662
31,31,0.7295059561729431,1.306154595046747
32,32,0.712346613407135,1.301957052262103
33,33,0.6941131353378296,1.2972782322617828
34,34,0.6768537163734436,1.2934978047355277
35,35,0.6610534191131592,1.290149000824475
36,36,0.6477226614952087,1.2864104724321208
37,37,0.6336562633514404,1.2827397330862578
38,38,0.6224037408828735,1.28004267958344
39,39,0.6096486449241638,1.278165973600794
40,40,0.5983026027679443,1.2753980042504482
41,41,0.5865355730056763,1.2707479508196722
42,42,0.5747510194778442,1.268186475409836
43,43,0.5648090243339539,1.2669613947633838
44,44,0.5582150816917419,1.263688259437436
45,45,0.5469653606414795,1.26330316262167
46,46,0.5397948026657104,1.2651092029008708
47,47,0.5285613536834717,1.2657323118116035
48,48,0.5214764475822449,1.2617427638319672
49,49,0.5137144923210144,1.2567176193487448
50,50,0.5074061751365662,1.258587821585233
51,51,0.4953930675983429,1.2610860105420723
52,52,0.4884642958641052,1.25399905345479
53,53,0.48112186789512634,1.2555480706887168
54,54,0.4739079773426056,1.2607014140144723
55,55,0.46671241521835327,1.2593900336593877
56,56,0.4594332277774811,1.255417995765561
57,57,0.45283788442611694,1.253324414862961
58,58,0.4456704258918762,1.259412046338691
59,59,0.4384590983390808,1.2594149229956455
60,60,0.43234121799468994,1.2572648095302894
61,61,0.4247139096260071,1.2532691330206198
62,62,0.41847679018974304,1.2528253774173925
63,63,0.4121303856372833,1.2563338983254355
64,64,0.4059444069862366,1.259188042312372
65,65,0.3990078568458557,1.255110318543481
66,66,0.39281919598579407,1.2570338014696465
67,67,0.38754355907440186,1.259737358718622
68,68,0.37936967611312866,1.2608969016153304
69,69,0.3771754503250122,1.2567921622854765
70,70,0.3693794012069702,1.2598660578493213
71,71,0.3639076054096222,1.2650326588114753
72,72,0.36070096492767334,1.2634957735655739
73,73,0.35348108410835266,1.261383556928791
74,74,0.3490070104598999,1.2616662197425716
75,75,0.34338971972465515,1.268583704213627
76,76,0.3398135006427765,1.270951943319352
77,77,0.333605021238327,1.2624798133725026
78,78,0.3264530599117279,1.264772508965164
79,79,0.3243292272090912,1.271190580774526
80,80,0.3200884461402893,1.268158959560707
81,81,0.31643614172935486,1.2650611752369365
82,82,0.31162431836128235,1.2713361646308274
83,83,0.30689674615859985,1.2747372486552253
84,84,0.3049965798854828,1.2749813892802253
85,85,0.2986047565937042,1.277832281394083
86,86,0.296627938747406,1.2741611668320953
87,87,0.29021719098091125,1.275193136246478
88,88,0.2860594391822815,1.2742967449250768
89,89,0.28477150201797485,1.2761920866419056
90,90,0.2809302508831024,1.280057563156378
91,91,0.27945107221603394,1.2773022260822233
92,92,0.27451688051223755,1.285566611368148
93,93,0.2730100750923157,1.2847291289782914
94,94,0.2668076157569885,1.282315863937628
95,95,0.2639479637145996,1.2811712046138575
96,96,0.25935620069503784,1.2874379392530098
97,97,0.2567569613456726,1.2922705978643698
98,98,0.2539784610271454,1.2844120713530993
99,99,0.2521474063396454,1.287153025142482

```


## File: `reports\multiseed_transformer_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.131489276885986,4.332992803855021
1,1,4.455928325653076,3.4464931800717213
2,2,3.4871151447296143,2.982668016777664
3,3,2.9580905437469482,2.6254129878810195
4,4,2.551292896270752,2.367857135710169
5,5,2.23822021484375,2.1563605636846823
6,6,1.9756388664245605,1.9944923275806865
7,7,1.7703171968460083,1.8822700625560322
8,8,1.619866132736206,1.8141126789030482
9,9,1.5263272523880005,1.7628731649430072
10,10,1.4543360471725464,1.7256061991707223
11,11,1.4019354581832886,1.696121340892354
12,12,1.3574352264404297,1.6732663013896003
13,13,1.3240914344787598,1.6516871217821465
14,14,1.2886359691619873,1.6242142974353226
15,15,1.2489954233169556,1.594375485279521
16,16,1.2024296522140503,1.5635588599033043
17,17,1.1599990129470825,1.537523988817559
18,18,1.1215269565582275,1.5142803504818776
19,19,1.0865803956985474,1.4960429707511527
20,20,1.0552465915679932,1.4801765817110655
21,21,1.0327543020248413,1.4672197435722976
22,22,1.0046324729919434,1.4562760650134476
23,23,0.9801933765411377,1.4481761494620902
24,24,0.9604292511940002,1.4405558851898694
25,25,0.9362484216690063,1.4328558249551742
26,26,0.9203926920890808,1.4282134009189293
27,27,0.8996928930282593,1.4255263531794313
28,28,0.8819271922111511,1.4196250790455303
29,29,0.8614965677261353,1.412111626296747
30,30,0.8420272469520569,1.4075306126328766
31,31,0.8250259757041931,1.4056926789830944
32,32,0.8095225095748901,1.3988426083424053
33,33,0.7942562103271484,1.397061957687628
34,34,0.7764336466789246,1.399182804295274
35,35,0.7647622227668762,1.3948570626680967
36,36,0.7479682564735413,1.3915926823850537
37,37,0.7336972951889038,1.3918557088883197
38,38,0.7225838899612427,1.3867126214699668
39,39,0.7074848413467407,1.3861216560738985
40,40,0.6972164511680603,1.3852411489017675
41,41,0.6848983764648438,1.3782711341732838
42,42,0.6732257008552551,1.3799626084624743
43,43,0.6607910990715027,1.378572807937372
44,44,0.6518636345863342,1.3752603999903945
45,45,0.6420164108276367,1.3807858326396003
46,46,0.6325812935829163,1.3764518362576845
47,47,0.6244439482688904,1.3723187055744108
48,48,0.6135423183441162,1.3755439383084658
49,49,0.6015930771827698,1.3769900212522412
50,50,0.5925207734107971,1.374143756803919
51,51,0.5866899490356445,1.3743328657306608
52,52,0.5781761407852173,1.3756153544441598
53,53,0.5672215223312378,1.3707530537589652
54,54,0.5606128573417664,1.3703703333119877
55,55,0.5513190031051636,1.3711337730532787
56,56,0.5435331463813782,1.3699593465836322
57,57,0.5369243025779724,1.371599416263768
58,58,0.5286304950714111,1.3699392099849512
59,59,0.5221172571182251,1.3698785500448258
60,60,0.5140098929405212,1.373761411573066
61,61,0.508087694644928,1.3740887251056608
62,62,0.5010775327682495,1.377713563012295
63,63,0.4950653314590454,1.3770399249967982
64,64,0.4873536229133606,1.376009581518955
65,65,0.4810398519039154,1.3776307653208248
66,66,0.47542089223861694,1.3800077594694544
67,67,0.46779969334602356,1.3817703997502562
68,68,0.46179822087287903,1.381149667208312
69,69,0.4552364945411682,1.3847318555487962
70,70,0.4491044282913208,1.3815220066758453
71,71,0.4431845545768738,1.3837105172579405
72,72,0.43802526593208313,1.3889577896868597
73,73,0.43417665362358093,1.3874838156778304
74,74,0.430062472820282,1.3889655441534323
75,75,0.4218490421772003,1.3917914218589909
76,76,0.4183262288570404,1.387481189164959
77,77,0.4117410480976105,1.3860816330206198
78,78,0.4054885506629944,1.3974867023405482
79,79,0.404742568731308,1.3966290833520107
80,80,0.4013376235961914,1.4080143912893828
81,81,0.4055192172527313,1.415180018690766
82,82,0.41187241673469543,1.410828387150999
83,83,0.4033944010734558,1.4017771736520235
84,84,0.38251492381095886,1.3918695918849258
85,85,0.3768933415412903,1.4002426647749104
86,86,0.37642887234687805,1.4109447041495902
87,87,0.3718174695968628,1.396802683345607
88,88,0.36456093192100525,1.3975426095430967
89,89,0.3604116439819336,1.4149882832511527
90,90,0.35401517152786255,1.4016208336001537
91,91,0.3503310978412628,1.405595998295018
92,92,0.3469558358192444,1.4079447261622695
93,93,0.34154605865478516,1.403076422019083
94,94,0.3400895297527313,1.4187814681256403
95,95,0.3381514549255371,1.4135527063588627
96,96,0.3330100178718567,1.4115887000912526
97,97,0.32739824056625366,1.4220542282354636
98,98,0.32661059498786926,1.4113691986584274
99,99,0.3225853741168976,1.4294701247918802

```


## File: `reports\multiseed_transformer_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7363465994596481,0.8945491135120391,0.5667760968208313,0.32777301669120784,0.35,0.15,0.6125,0.5625,0.4,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,0.012500000000000067,0.050000000000000044,0.19999999999999996,20,7.396899262775736
2,0.8422335758805275,0.8702411264181137,0.5667760968208313,0.3034650295972824,0.47058823529411764,0.5882352941176471,0.425,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.0625,0.30000000000000004,0.30000000000000004,20,6.665895189300939
3,1.272395372390747,0.8251287270994747,0.5667760968208313,0.2583526302786434,0.0,0.6666666666666666,0.4264705882352941,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.458577057131288
4,2.283124844233195,0.7839499910672506,0.5667760968208313,0.2171738942464193,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.5119248544617534
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.817195662856102,0.880837419629097,0.5610660910606384,0.3197713285684586,0.55,0.15,0.55,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.09999999999999998,0.19999999999999996,20,6.665088749558848
2,0.8737567335367202,0.8651024371385574,0.5610660910606384,0.30403634607791896,0.47058823529411764,0.5882352941176471,0.525,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.03750000000000003,0.30000000000000004,0.30000000000000004,20,6.425404836658456
3,1.213964094133938,0.8243744162952199,0.5610660910606384,0.2633083252345815,0.3333333333333333,0.6666666666666666,0.45588235294117646,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.0,0.35294117647058826,0.2941176470588235,17,4.673180073739059
4,1.9699038465817769,0.7786184052626292,0.5610660910606384,0.21755231420199073,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.911328921977615
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed43\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9979203909635543,0.8589423507452011,0.5589107871055603,0.3000315636396408,0.3,0.15,0.575,0.5625,0.4,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,0.050000000000000044,0.19999999999999996,20,5.458032191757681
2,0.97677541077137,0.8489084422588349,0.5589107871055603,0.2899976551532746,0.4117647058823529,0.5882352941176471,0.45,0.5125,0.6,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.35,0.30000000000000004,20,5.747729395947949
3,1.179989187156453,0.8274404406547546,0.5589107871055603,0.26852965354919434,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.807732881529551
4,1.721894105275472,0.8012032210826874,0.5589107871055603,0.24229243397712708,,,0.3333333333333333,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.12499999999999997,0.0,0.16666666666666663,6,3.3306566440396628
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8711518377065659,0.8765803724527359,0.5730632543563843,0.3035171180963516,0.2,0.45,0.45,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,6.392024556145009
2,1.2143743872642516,0.814670866727829,0.5730632543563843,0.24160761237144468,0.5,0.45,0.375,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,4.808402688756268
3,1.6487890392541886,0.7707661271095276,0.5730632543563843,0.19770287275314335,0.35,0.3,0.425,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.1875,-0.9,0.19999999999999996,20,3.457485046090977
4,3.190389394760132,0.6540046900510788,0.5730632543563843,0.08094143569469447,0.35,0.55,0.325,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.15000000000000002,1.0,0.19999999999999996,20,1.8189600569085755
5,3.5235782623291017,0.7491312801837922,0.5730632543563843,0.17606802582740788,,,0.3125,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.2,0.0,0.19999999999999996,20,1.6331652777370291
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_transformer_seed43\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9214872762560844,0.866921791434288,0.560093343257904,0.306828448176384,0.3,0.15,0.5875,0.5625,0.35,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.012499999999999956,0.0,0.19999999999999996,20,5.910750760249065
2,0.967028146982193,0.8511496186256409,0.560093343257904,0.2910562753677368,0.4117647058823529,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,5.805664250053228
3,1.296340037794674,0.8174950059722451,0.560093343257904,0.25740166271434106,0.16666666666666666,0.6666666666666666,0.4411764705882353,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.376222788422399
4,1.946991542975108,0.7954479257265726,0.560093343257904,0.23535458246866858,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,2.945589600920934
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 0.350 | 0.150 | 0.613 | 0.562 | 0.7363 |
| 2 | 0.870 | 0.471 | 0.588 | 0.425 | 0.512 | 0.8422 |
| 3 | 0.825 | 0.000 | 0.667 | 0.426 | 0.471 | 1.2724 |
| 4 | 0.784 | nan | nan | 0.250 | 0.417 | 2.2831 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | +0.013 | -0.050 | +0.328 |
| Shuffled Action | -0.012 | -0.025 | +0.307 |
| Constant Action | -0.025 | -0.012 | +0.300 |
| Blind | -0.050 | 0.012 | +0.320 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2966.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_transformer_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.881 | +0.105 |
| 2 | 1.000 | 0.870 | 0.865 | +0.130 |
| 3 | 1.000 | 0.825 | 0.824 | +0.175 |
| 4 | 1.000 | 0.784 | 0.779 | +0.216 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.613 | 0.550 | -0.050 |
| 2 | 0.512 | 0.425 | 0.525 | +0.087 |
| 3 | 0.471 | 0.426 | 0.456 | +0.044 |
| 4 | 0.417 | 0.250 | 0.250 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.550 | -0.200 |
| 2 | 0.588 | 0.471 | 0.471 | +0.118 |
| 3 | 0.667 | 0.000 | 0.333 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.115 | +5.000 | +0.500 |
| 2 | +0.038 | +8.000 | +0.000 |
| 3 | +0.004 | -2.000 | -1.000 |
| 4 | +0.024 | +0.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.881 | 0.287 | 0.562 | 0.613 | 0.550 | 0.600 |
| 2 | 1.000 | 0.870 | 0.865 | 0.246 | 0.512 | 0.425 | 0.525 | 0.487 |
| 3 | 1.000 | 0.825 | 0.824 | 0.241 | 0.471 | 0.426 | 0.456 | 0.456 |
| 4 | 1.000 | 0.784 | 0.779 | 0.237 | 0.417 | 0.250 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1565
**Average Oracle Gain (state probe):** +0.0621
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_transformer_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_transformer_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_transformer_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.8975813388824463,4.051217000992572
1,1,4.162201404571533,3.3558144491226947
2,2,3.3916573524475098,2.9192704998078893
3,3,2.886620044708252,2.581693555487961
4,4,2.482503652572632,2.289704369716957
5,5,2.135690450668335,2.0734723200563523
6,6,1.8763561248779297,1.9225881607806097
7,7,1.6905614137649536,1.8312159053614883
8,8,1.5695934295654297,1.7577377069191855
9,9,1.4775960445404053,1.7115811207255378
10,10,1.4119192361831665,1.6835499747854765
11,11,1.3662337064743042,1.6606081352859248
12,12,1.3279156684875488,1.6368200583536117
13,13,1.2913085222244263,1.6135403992699795
14,14,1.252520203590393,1.5835489992235527
15,15,1.2096134424209595,1.554335922491355
16,16,1.166045904159546,1.5251847564196976
17,17,1.125091314315796,1.4975849839507556
18,18,1.0853822231292725,1.473207567558914
19,19,1.0474427938461304,1.4525479176005378
20,20,1.0127055644989014,1.4350023113313268
21,21,0.9845819473266602,1.4212894127017162
22,22,0.9590612649917603,1.4085805924212347
23,23,0.9347063302993774,1.3972750804463372
24,24,0.9100219011306763,1.3861791892129867
25,25,0.8881150484085083,1.3752041175717213
26,26,0.8655931949615479,1.3646107658011015
27,27,0.8443621397018433,1.3540424284387806
28,28,0.8222391605377197,1.3442580426325563
29,29,0.7998189926147461,1.3363603685722976
30,30,0.779114305973053,1.3284543146852588
31,31,0.7572926878929138,1.3212700515496927
32,32,0.740015983581543,1.3152630915407275
33,33,0.7192291617393494,1.309144066982582
34,34,0.7001386880874634,1.3050786002737578
35,35,0.6865206360816956,1.300139630427126
36,36,0.6718636155128479,1.2951825251344775
37,37,0.6558594107627869,1.2924400704805967
38,38,0.6427828073501587,1.2893844354348105
39,39,0.6289997100830078,1.2832118800429047
40,40,0.6174302697181702,1.2773533805471953
41,41,0.6051636338233948,1.2734309962538422
42,42,0.5934228301048279,1.2721595138799948
43,43,0.5819352865219116,1.2713806902776
44,44,0.5712813138961792,1.2704552822425716
45,45,0.5642162561416626,1.2668113083135886
46,46,0.554125964641571,1.2640020651895492
47,47,0.5444278120994568,1.2647501210697363
48,48,0.5347400307655334,1.2656410092213115
49,49,0.5259475708007812,1.2635434260133838
50,50,0.518828272819519,1.2597632486312116
51,51,0.5096285343170166,1.261001461842021
52,52,0.5025399923324585,1.263446620253266
53,53,0.49705860018730164,1.2665640408875511
54,54,0.49258655309677124,1.2654926737800973
55,55,0.4804939031600952,1.2603489610015368
56,56,0.4728286564350128,1.2592525794857838
57,57,0.46652188897132874,1.2624472946417136
58,58,0.45849859714508057,1.263474886534644
59,59,0.45232874155044556,1.2623599943567494
60,60,0.4455152750015259,1.2608497494556865
61,61,0.440508097410202,1.2599534832063268
62,62,0.43218451738357544,1.264558635774206
63,63,0.42472508549690247,1.2657345631083503
64,64,0.41906362771987915,1.2630465147925205
65,65,0.41446664929389954,1.2647358628570056
66,66,0.4076839089393616,1.2654296374711833
67,67,0.40043988823890686,1.2675714961818008
68,68,0.3949510157108307,1.2704231387279072
69,69,0.388952374458313,1.2661915763479765
70,70,0.3848172724246979,1.2651519775390625
71,71,0.37801769375801086,1.269991264968622
72,72,0.3705950379371643,1.270736319119813
73,73,0.36665692925453186,1.2700744378762168
74,74,0.3626629412174225,1.2721452556672643
75,75,0.3573254942893982,1.270187753145812
76,76,0.3503780961036682,1.2711548101706582
77,77,0.34889742732048035,1.2744831022669056
78,78,0.3428143262863159,1.2784986652311732
79,79,0.33706194162368774,1.2769430191790472
80,80,0.33139747381210327,1.2782482710040983
81,81,0.32757771015167236,1.2775322335665342
82,82,0.32229083776474,1.2824927158043034
83,83,0.322706401348114,1.2860845346919825
84,84,0.3182889521121979,1.2785248052878457
85,85,0.3097200095653534,1.2831648529553024
86,86,0.30641138553619385,1.289885724177126
87,87,0.30545955896377563,1.2859409519883453
88,88,0.30185467004776,1.2845386442590931
89,89,0.29703912138938904,1.2926646998671234
90,90,0.28984782099723816,1.2893580452340547
91,91,0.2878658175468445,1.2881512251056608
92,92,0.2853516936302185,1.292407801893891
93,93,0.27848514914512634,1.2970040743468239
94,94,0.27960139513015747,1.294893858862705
95,95,0.27571818232536316,1.2967170340115908
96,96,0.2702978849411011,1.2986126258725026
97,97,0.27271467447280884,1.3060195172419313
98,98,0.26403623819351196,1.3017963346887806
99,99,0.2602495849132538,1.2966324853115394

```


## File: `reports\multiseed_transformer_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.287034034729004,4.306293425012807
1,1,4.414597034454346,3.4774299996798157
2,2,3.5335278511047363,3.0364797623431095
3,3,3.0192477703094482,2.722987941054047
4,4,2.650759696960449,2.4529579037525613
5,5,2.325164318084717,2.2144942987160605
6,6,2.0384955406188965,2.033952181456519
7,7,1.8195934295654297,1.9041845603067367
8,8,1.6526597738265991,1.81625241138896
9,9,1.5313979387283325,1.7526191336209658
10,10,1.4507648944854736,1.714311818607518
11,11,1.3902523517608643,1.691604239041688
12,12,1.353577971458435,1.6711697187580046
13,13,1.319338321685791,1.6482775328589268
14,14,1.28170645236969,1.6296869496830175
15,15,1.25148606300354,1.606416796074539
16,16,1.2150852680206299,1.579816599361232
17,17,1.1752803325653076,1.5530750712410348
18,18,1.1370190382003784,1.5277194664126537
19,19,1.1012483835220337,1.5056900274558145
20,20,1.0681169033050537,1.4866888327676742
21,21,1.0403828620910645,1.4720989289830944
22,22,1.012779712677002,1.4609767726210297
23,23,0.9911808371543884,1.4514292732613985
24,24,0.9687036275863647,1.44357424876729
25,25,0.9497891664505005,1.4381588795146003
26,26,0.9320623874664307,1.4324716036436989
27,27,0.9115282297134399,1.4287328251072617
28,28,0.8945580124855042,1.4225318783619365
29,29,0.8743337988853455,1.4172200687596055
30,30,0.8534363508224487,1.410550727218878
31,31,0.8365601897239685,1.405715567166688
32,32,0.8170934915542603,1.4009024197938011
33,33,0.7985274791717529,1.3963216562740137
34,34,0.7819361090660095,1.3955315761878841
35,35,0.7651584148406982,1.3945690217565319
36,36,0.7494685649871826,1.391009846671683
37,37,0.7361850738525391,1.3841650290567367
38,38,0.7217519283294678,1.383668493051998
39,39,0.7079728841781616,1.3867523943791624
40,40,0.6966084241867065,1.3855763419729765
41,41,0.6839389204978943,1.3836906308033428
42,42,0.673328697681427,1.3840018100425846
43,43,0.6600286960601807,1.3845907742859886
44,44,0.6485635042190552,1.3823933835889473
45,45,0.6382133960723877,1.3785017670178024
46,46,0.6287797689437866,1.376392427037974
47,47,0.6186964511871338,1.3783086557857325
48,48,0.61142498254776,1.3761796794953893
49,49,0.6045917868614197,1.373475621958248
50,50,0.5932016968727112,1.3705789534772028
51,51,0.5823017358779907,1.369615273397477
52,52,0.5743154287338257,1.371732367843878
53,53,0.5672791600227356,1.3737990582575563
54,54,0.5618200302124023,1.3705787033331198
55,55,0.5526400208473206,1.3694800705206198
56,56,0.5443109273910522,1.3700845436971696
57,57,0.5377517938613892,1.3725564675252946
58,58,0.5310848951339722,1.370760057793289
59,59,0.5230519771575928,1.3693745097175973
60,60,0.5162594318389893,1.3722217747422516
61,61,0.509005069732666,1.372010778208248
62,62,0.502917468547821,1.3707274139904586
63,63,0.49763596057891846,1.3743161060771003
64,64,0.48891907930374146,1.3725804813572617
65,65,0.4803292155265808,1.3773295918449027
66,66,0.4783676266670227,1.3795531225986168
67,67,0.4729219675064087,1.3793425012807377
68,68,0.4657335579395294,1.3780740206358864
69,69,0.4589761197566986,1.3769461209656761
70,70,0.45223867893218994,1.382549723640817
71,71,0.44753530621528625,1.3811649259973744
72,72,0.44295698404312134,1.3806880263031507
73,73,0.4355020225048065,1.3760586097592213
74,74,0.4305625259876251,1.3834857627993724
75,75,0.4249245226383209,1.3877874905945824
76,76,0.42046090960502625,1.384948230180584
77,77,0.41304200887680054,1.3819317426837858
78,78,0.40754419565200806,1.3900361608286373
79,79,0.4032059907913208,1.384981749487705
80,80,0.40039098262786865,1.3963240326428024
81,81,0.39526429772377014,1.409791039638832
82,82,0.4080769419670105,1.451586488817559
83,83,0.44521141052246094,1.4746866695216445
84,84,0.47433874011039734,1.40176516673604
85,85,0.39821022748947144,1.4128362937051742
86,86,0.40310433506965637,1.4017290209160476
87,87,0.39195477962493896,1.3893980432729252
88,88,0.38285166025161743,1.3886319770187627
89,89,0.37795984745025635,1.3994818515464909
90,90,0.3709484934806824,1.4043966824891136
91,91,0.3694571852684021,1.39099371237833
92,92,0.3570539355278015,1.3912287227443008
93,93,0.35944634675979614,1.390393991939357
94,94,0.3515999913215637,1.3948701952324538
95,95,0.34458091855049133,1.399354027920082
96,96,0.344607412815094,1.3894718357774078
97,97,0.3351040482521057,1.3889671700899717
98,98,0.33389896154403687,1.3913038910412399
99,99,0.3287968933582306,1.3971204914030482

```


## File: `reports\multiseed_transformer_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7326967597007752,0.8950793504714966,0.5667760968208313,0.32830325365066526,0.4,0.15,0.6,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,0.0,-0.09999999999999998,0.19999999999999996,20,7.433746016448676
2,0.8303376346826553,0.8722527235746383,0.5667760968208313,0.30547662675380705,0.35294117647058826,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.761395012374007
3,1.230914154473473,0.8251674701185787,0.5667760968208313,0.2583913732977474,0.3333333333333333,0.6666666666666666,0.5,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.608828970179549
4,2.1088493863741555,0.7740650475025177,0.5667760968208313,0.2072889506816864,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.7195104966357997
5,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
6,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
7,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,
8,,,0.5667760968208313,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8062186628580094,0.8825147360563278,0.5610660910606384,0.32144864499568937,0.55,0.15,0.575,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.09999999999999998,0.19999999999999996,20,6.755836685028162
2,0.8530033886432647,0.8693215519189834,0.5610660910606384,0.308255460858345,0.47058823529411764,0.5882352941176471,0.425,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.0625,0.30000000000000004,0.30000000000000004,20,6.581733222255313
3,1.205183530555052,0.8293193789089427,0.5610660910606384,0.26825328784830427,0.3333333333333333,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.707227298674295
4,1.952649474143982,0.7897213498751322,0.5610660910606384,0.22865525881449378,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.93705455997557
5,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
6,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
7,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,
8,,,0.5610660910606384,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed44\coherence_constant_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9692032992839813,0.8628217220306397,0.5589107871055603,0.30391093492507937,0.35,0.15,0.575,0.5625,0.3,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.04999999999999999,0.19999999999999996,20,5.619751421311027
2,0.9418677657842636,0.8545310258865356,0.5589107871055603,0.2956202387809753,0.47058823529411764,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,5.960752608467214
3,1.1507515539141262,0.8300052846179289,0.5589107871055603,0.2710944975123686,0.16666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.929884991808366
4,1.7662916382153828,0.7875693043073019,0.5589107871055603,0.22865851720174157,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,3.246937209000786
5,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
6,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
7,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,
8,,,0.5589107871055603,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8666682854294777,0.8775756329298019,0.5730632543563843,0.3045123785734176,0.2,0.45,0.475,0.475,0.0,0.0,0.0,0.2,5.568423938751221,0.27208997011184693,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.2,20,6.425092543904253
2,1.2193886041641235,0.8150484412908554,0.5730632543563843,0.24198518693447113,0.5,0.45,0.4125,0.5,0.0,0.05,0.0,0.05,5.8392010688781735,0.21551565043628215,0.4,0.35,0.05,0.2,0.0625,-0.05,-0.2,20,4.788630178220237
3,1.633970522880554,0.7691002309322357,0.5730632543563843,0.1960369765758514,0.45,0.3,0.4,0.4375,0.0,0.15,1.0,1.0,5.700663447380066,0.23710520789027215,0.3,0.2375,0.9,0.8,0.16250000000000003,-0.9,0.19999999999999996,20,3.4888410577507054
4,3.036438798904419,0.6479363679885864,0.5730632543563843,0.07487311363220217,0.4,0.55,0.35,0.4625,1.0,0.9,1.0,1.0,5.803190875053406,0.22019456401467324,0.35,0.175,0.0,0.8,0.175,1.0,0.19999999999999996,20,1.9111832180339883
5,3.252507770061493,0.7325339823961258,0.5730632543563843,0.1594707280397415,,,0.3375,0.3625,0.0,0.0,1.0,1.0,5.7545856714248655,0.2303200677037239,,0.1125,0.0,0.8,0.22500000000000003,0.0,0.19999999999999996,20,1.7692765331398632
6,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
7,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,
8,,,0.5730632543563843,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\multiseed_transformer_seed44\coherence_shuffled_depth.csv`
```csv
depth,mse,cosine_similarity,random_pair_cosine,mean_centered_cosine,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.9077251464128494,0.8690466046333313,0.560093343257904,0.3089532613754272,0.3,0.15,0.6,0.5625,0.25,0.45,0.7,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,0.0,-0.09999999999999998,0.19999999999999996,20,6.0003643616294
2,0.9503716856241227,0.8537309408187866,0.560093343257904,0.2936375975608826,0.4117647058823529,0.5882352941176471,0.4875,0.5125,0.55,0.65,1.0,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.0,0.30000000000000004,0.30000000000000004,20,5.907415831778263
3,1.2634034226922428,0.8186550736427307,0.560093343257904,0.25856173038482666,0.3333333333333333,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.490309835359161
4,1.8874878485997517,0.7851232488950094,0.560093343257904,0.22502990563710534,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.0384503117850885
5,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
6,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
7,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,
8,,,0.560093343257904,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\multiseed_transformer_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 0.400 | 0.150 | 0.600 | 0.562 | 0.7327 |
| 2 | 0.872 | 0.353 | 0.588 | 0.450 | 0.512 | 0.8303 |
| 3 | 0.825 | 0.333 | 0.667 | 0.500 | 0.471 | 1.2309 |
| 4 | 0.774 | nan | nan | 0.250 | 0.417 | 2.1088 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

### Action Control Ablation (Depth 1)

| Transition Mode | Semantic Gain | Oracle Gap | Mean-Centered Cosine |
|---|---|---|---|
| Oracle | -0.037 | 0.000 | +nan |
| True Action | +0.000 | -0.037 | +0.328 |
| Shuffled Action | +0.000 | -0.037 | +0.309 |
| Constant Action | -0.025 | -0.012 | +0.304 |
| Blind | -0.025 | -0.012 | +0.321 |

_Note: Semantic gain is `Probe_A(h_1) - Probe_A(Identity)`. Oracle Gap is `Oracle_Probe_A(h_1) - Probe_A(h_1)`._


See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2897.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\multiseed_transformer_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.883 | +0.105 |
| 2 | 1.000 | 0.872 | 0.869 | +0.128 |
| 3 | 1.000 | 0.825 | 0.829 | +0.175 |
| 4 | 1.000 | 0.774 | 0.790 | +0.226 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.600 | 0.575 | -0.037 |
| 2 | 0.512 | 0.450 | 0.425 | +0.062 |
| 3 | 0.471 | 0.500 | 0.471 | -0.029 |
| 4 | 0.417 | 0.250 | 0.250 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.550 | -0.250 |
| 2 | 0.588 | 0.353 | 0.471 | +0.235 |
| 3 | 0.667 | 0.333 | 0.333 | +0.333 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.107 | -2.000 | +0.375 |
| 2 | +0.022 | +0.286 | -1.000 |
| 3 | -0.024 | n/a | +0.000 |
| 4 | -0.074 | +0.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.883 | 0.287 | 0.562 | 0.600 | 0.575 | 0.600 |
| 2 | 1.000 | 0.872 | 0.869 | 0.246 | 0.512 | 0.450 | 0.425 | 0.487 |
| 3 | 1.000 | 0.825 | 0.829 | 0.241 | 0.471 | 0.500 | 0.471 | 0.456 |
| 4 | 1.000 | 0.774 | 0.790 | 0.237 | 0.417 | 0.250 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1584
**Average Oracle Gain (state probe):** +0.0406
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\multiseed_transformer_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\multiseed_transformer_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\multiseed_transformer_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.226391315460205,3.8842005495165215
1,1,3.981693744659424,3.3235291027631915
2,2,3.3592569828033447,2.945342017001793
3,3,2.925642490386963,2.617927176053407
4,4,2.5395655632019043,2.3397679563428535
5,5,2.1999423503875732,2.1050174900742826
6,6,1.9134324789047241,1.9367115458504098
7,7,1.7027523517608643,1.8240913015897158
8,8,1.5602667331695557,1.7470152808017418
9,9,1.467488169670105,1.6995616975377819
10,10,1.4049625396728516,1.672666956166752
11,11,1.3655592203140259,1.6566724933561732
12,12,1.3277239799499512,1.6397254818775615
13,13,1.291024088859558,1.6185407794889857
14,14,1.2576466798782349,1.5900140981205175
15,15,1.2118566036224365,1.5594406127929688
16,16,1.167618989944458,1.5305941222143955
17,17,1.1246750354766846,1.5035057693231302
18,18,1.0828557014465332,1.4790406774301998
19,19,1.0451322793960571,1.4566732938172386
20,20,1.0114394426345825,1.4377401383196722
21,21,0.9809972047805786,1.421862617867892
22,22,0.9526880979537964,1.4073301221503587
23,23,0.927641749382019,1.3947936511430583
24,24,0.901800274848938,1.383208227939293
25,25,0.8794377446174622,1.3719252289318649
26,26,0.8567638993263245,1.362993959520684
27,27,0.8357728123664856,1.3507285196273053
28,28,0.8111107349395752,1.3408896024109886
29,29,0.7900477647781372,1.3337329801965931
30,30,0.7697048187255859,1.3274586161629098
31,31,0.7508599162101746,1.3211399766265368
32,32,0.7337058782577515,1.313462929647477
33,33,0.7161287069320679,1.3064663996461963
34,34,0.6983935832977295,1.3018710026975537
35,35,0.6821478605270386,1.2980316662397542
36,36,0.6671731472015381,1.2936976698578382
37,37,0.6535252928733826,1.2895938060322747
38,38,0.6418160796165466,1.2868918747198386
39,39,0.6292403340339661,1.2836171134573515
40,40,0.615079939365387,1.279573034067623
41,41,0.6045528650283813,1.275909423828125
42,42,0.5919222235679626,1.274034718998143
43,43,0.5861613750457764,1.2726047703477203
44,44,0.5731631517410278,1.2726943219294313
45,45,0.5628438591957092,1.2689856857549948
46,46,0.5546144843101501,1.267715329029521
47,47,0.5464872717857361,1.2646086645908043
48,48,0.536905825138092,1.2638882496317878
49,49,0.527898907661438,1.2628226358382428
50,50,0.5189650058746338,1.2582606331246797
51,51,0.5109919905662537,1.257983223336642
52,52,0.5039381980895996,1.260729054935643
53,53,0.49687659740448,1.257814751296747
54,54,0.48772650957107544,1.2572274129898822
55,55,0.4831041693687439,1.2587225241739242
56,56,0.4740447402000427,1.257096962850602
57,57,0.46608924865722656,1.2572102781201973
58,58,0.45950397849082947,1.2590792296362705
59,59,0.45261213183403015,1.2588108250352203
60,60,0.4443892240524292,1.2588535996734118
61,61,0.43666231632232666,1.2574555443935707
62,62,0.4316514730453491,1.2575853691726435
63,63,0.42395803332328796,1.2591767858286373
64,64,0.4173535704612732,1.2611975748030866
65,65,0.41255468130111694,1.2597721287461578
66,66,0.40764713287353516,1.2607289298636015
67,67,0.4004020392894745,1.26008668493052
68,68,0.3956945240497589,1.2619989113729508
69,69,0.38845735788345337,1.2617565217565319
70,70,0.38057056069374084,1.2637950709608734
71,71,0.37641575932502747,1.2652542864690062
72,72,0.3703828454017639,1.2616206935194672
73,73,0.3645748496055603,1.2638250882508324
74,74,0.360992431640625,1.2669802806416497
75,75,0.3553979694843292,1.265257538342085
76,76,0.3478032946586609,1.2649681216380635
77,77,0.34484025835990906,1.2679819826219902
78,78,0.3423677086830139,1.2675611152023565
79,79,0.33184799551963806,1.2674397953221055
80,80,0.33110764622688293,1.2672760760197874
81,81,0.3266715407371521,1.269393045394147
82,82,0.32110610604286194,1.268781693255315
83,83,0.3170606195926666,1.2718400798860143
84,84,0.3125170171260834,1.2759325621558018
85,85,0.308141827583313,1.2700948246189805
86,86,0.30365726351737976,1.2766382186139216
87,87,0.29840871691703796,1.2810536368948515
88,88,0.2959737181663513,1.2776565551757812
89,89,0.29221904277801514,1.2744503333920338
90,90,0.28798210620880127,1.2823251192686989
91,91,0.2857879400253296,1.283233642578125
92,92,0.2790089249610901,1.2823236184042008
93,93,0.2791013717651367,1.2869968101626537
94,94,0.2751922309398651,1.2894030711689934
95,95,0.2683197557926178,1.2861301859871286
96,96,0.26656487584114075,1.285492193503458
97,97,0.2642207443714142,1.2855801191486296
98,98,0.26131612062454224,1.2881257104091957
99,99,0.25737231969833374,1.2897272579005508

```


## File: `reports\multiseed_transformer_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.108949661254883,4.55602026767418
1,1,4.732741355895996,3.5855647853163424
2,2,3.647576332092285,3.0677312632076075
3,3,3.0682783126831055,2.770105830958632
4,4,2.705390691757202,2.5223758885117826
5,5,2.4007763862609863,2.2966173515945183
6,6,2.119060516357422,2.101465944383965
7,7,1.8792296648025513,1.9631260105820953
8,8,1.7015726566314697,1.869953968485848
9,9,1.5791491270065308,1.8007597376088627
10,10,1.4926445484161377,1.7474505315061475
11,11,1.4268264770507812,1.7131597800332992
12,12,1.379086971282959,1.6881707613585426
13,13,1.3399115800857544,1.665008795065958
14,14,1.303626537322998,1.6412553630891393
15,15,1.2677192687988281,1.6164034233718623
16,16,1.2284517288208008,1.5919748525150488
17,17,1.1912126541137695,1.5656174206342854
18,18,1.153937578201294,1.5392923824122695
19,19,1.1166410446166992,1.515217890504931
20,20,1.0782568454742432,1.4947726140256787
21,21,1.0421853065490723,1.4789243604316087
22,22,1.0189263820648193,1.4640245281282018
23,23,0.9932374954223633,1.4528333319992315
24,24,0.9686731100082397,1.4449718037589652
25,25,0.9488693475723267,1.4382271688492572
26,26,0.926365315914154,1.4319900762839395
27,27,0.9059460163116455,1.4266847704277663
28,28,0.8873352408409119,1.4222837354316087
29,29,0.8669026494026184,1.4182387805375896
30,30,0.8460146188735962,1.4155058313588627
31,31,0.830974817276001,1.4125413738313268
32,32,0.8134756684303284,1.4090721255443135
33,33,0.7965925931930542,1.4056821729316087
34,34,0.7790672779083252,1.402100609951332
35,35,0.7612743377685547,1.3984303708936348
36,36,0.747527003288269,1.396360303534836
37,37,0.7323481440544128,1.3957069271900615
38,38,0.7194089889526367,1.3940676079421748
39,39,0.7057019472122192,1.3954130078925462
40,40,0.6944940090179443,1.3952996926229508
41,41,0.6816953420639038,1.3931830984647158
42,42,0.6683635115623474,1.390636631699859
43,43,0.6574523448944092,1.3864823638415726
44,44,0.6472762227058411,1.387956337850602
45,45,0.6373383402824402,1.3911620593461833
46,46,0.6261736750602722,1.388103422571401
47,47,0.6162022948265076,1.3856438808753841
48,48,0.6078745722770691,1.3862477286917265
49,49,0.5982027053833008,1.3854932941374232
50,50,0.5892086625099182,1.3851322111536244
51,51,0.5828474164009094,1.3847751304751537
52,52,0.5729979276657104,1.3885760698162142
53,53,0.5665003061294556,1.3866954866002819
54,54,0.5584477782249451,1.3867004894819417
55,55,0.552116870880127,1.391358172307249
56,56,0.5456129312515259,1.3920965976402409
57,57,0.5396824479103088,1.3896291764056097
58,58,0.5311422944068909,1.3878447735895876
59,59,0.5239964723587036,1.3878402709960938
60,60,0.520263135433197,1.3848076492059427
61,61,0.512239396572113,1.3860906382076075
62,62,0.5044727921485901,1.3886179689501152
63,63,0.4984458386898041,1.3862644883452868
64,64,0.49130088090896606,1.3850127673539958
65,65,0.4824928045272827,1.385975697001473
66,66,0.48022717237472534,1.3941147601018187
67,67,0.4764487147331238,1.3948015306816726
68,68,0.4725300371646881,1.3892989862160605
69,69,0.46648290753364563,1.391431089307441
70,70,0.4555458724498749,1.3897509965740267
71,71,0.44815075397491455,1.3911356691454277
72,72,0.442775696516037,1.3930585267113857
73,73,0.4390484094619751,1.3905817250736425
74,74,0.42993253469467163,1.400147484951332
75,75,0.4302779734134674,1.3972928406762295
76,76,0.4231310188770294,1.3917026207095287
77,77,0.41369321942329407,1.4029505995453382
78,78,0.41216036677360535,1.4023199863121159
79,79,0.404705673456192,1.3970953519227074
80,80,0.39945363998413086,1.4001429823578382
81,81,0.39384782314300537,1.4030058813876793
82,82,0.3896724283695221,1.4016050745229252
83,83,0.38499462604522705,1.4076418016777663
84,84,0.37959128618240356,1.4053189637231045
85,85,0.3745886981487274,1.4063022801133453
86,86,0.3705579340457916,1.4080201446032914
87,87,0.3661634027957916,1.412294231477331
88,88,0.3597053289413452,1.4100398079293672
89,89,0.3573625683784485,1.4168367229524206
90,90,0.35642698407173157,1.422623055880187
91,91,0.35399356484413147,1.426893765809106
92,92,0.3569587171077728,1.4720467739417904
93,93,0.38974013924598694,1.5144315625800462
94,94,0.44876718521118164,1.5018085417200306
95,95,0.42082342505455017,1.4128671864994238
96,96,0.3392895460128784,1.4590525392626152
97,97,0.3940446078777313,1.4131237092565319
98,98,0.3333619236946106,1.4416111180039703
99,99,0.3574894070625305,1.4179070894835426

```


## File: `reports\transformer\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.736598764359951,0.894507372379303,0.4,0.15,0.6125,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,0.012500000000000067,-0.09999999999999998,0.19999999999999996,20,7.394367161056372
2,0.8394410997629166,0.8709937274456024,0.4117647058823529,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.688069830785806
3,1.2734325861229616,0.8208198897978839,0.0,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.4549455800498166
4,2.1655954917271933,0.7697034974892935,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.6482499356180784
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.817132793366909,0.8810890853404999,0.5,0.15,0.5375,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.0625,-0.09999999999999998,0.19999999999999996,20,6.665601672422725
2,0.8812031745910645,0.8636460870504379,0.35294117647058826,0.5882352941176471,0.45,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.03749999999999998,0.30000000000000004,0.30000000000000004,20,6.3711081121007
3,1.2260235267526962,0.8234177827835083,0.16666666666666666,0.6666666666666666,0.4117647058823529,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.04411764705882354,0.35294117647058826,0.2941176470588235,17,4.62721370940235
4,2.0710790356000266,0.7690909405549368,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.769106356136623
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8826506391167641,0.8755313485860825,0.15,0.45,0.4375,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.012500000000000011,-0.05,-0.2,20,6.308751974807603
2,1.2570623695850371,0.8090034931898117,0.45,0.45,0.4375,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.08750000000000002,-0.05,-0.2,20,4.645116526728116
3,1.7126843810081482,0.7589603751897812,0.4,0.3,0.4125,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.175,-0.9,0.19999999999999996,20,3.3284961822227808
4,3.2399171113967897,0.635369636118412,0.2,0.55,0.3625,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.1875,1.0,0.19999999999999996,20,1.7911541471253787
5,3.65866881608963,0.728294751048088,,,0.3125,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.2,0.0,0.19999999999999996,20,1.5728632549750476
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\transformer\metadata.txt`
```text
Base model:
DeepSeek-R1-Distill-Qwen-7B

Dataset:
Countdown

Transition:
transformer

Trajectory source:
shared_teacher_hidden_states

Seed:
42

```


## File: `reports\transformer\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\transformer\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.126984,63,0.044106,0.209862
oracle1,probeA_per_label,all,0.539683,63,,
oracle1,probeA_jaccard,all,0.378307,63,,

```


## File: `reports\transformer\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.126984 |
| probeA_exact_ci_lo | 0.044106 |
| probeA_exact_ci_hi | 0.209862 |
| probeA_per_label | 0.539683 |
| probeA_jaccard | 0.378307 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\transformer\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 0.400 | 0.150 | 0.613 | 0.562 | 0.7366 |
| 2 | 0.871 | 0.412 | 0.588 | 0.450 | 0.512 | 0.8394 |
| 3 | 0.821 | 0.000 | 0.667 | 0.485 | 0.471 | 1.2734 |
| 4 | 0.770 | nan | nan | 0.208 | 0.417 | 2.1656 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2891.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\transformer\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.881 | +0.105 |
| 2 | 1.000 | 0.871 | 0.864 | +0.129 |
| 3 | 1.000 | 0.821 | 0.823 | +0.179 |
| 4 | 1.000 | 0.770 | 0.769 | +0.230 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.613 | 0.537 | -0.050 |
| 2 | 0.512 | 0.450 | 0.450 | +0.062 |
| 3 | 0.471 | 0.485 | 0.412 | -0.015 |
| 4 | 0.417 | 0.208 | 0.250 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.500 | -0.250 |
| 2 | 0.588 | 0.412 | 0.353 | +0.176 |
| 3 | 0.667 | 0.000 | 0.167 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.113 | +3.000 | +0.286 |
| 2 | +0.054 | +0.000 | +0.250 |
| 3 | -0.015 | +1.250 | -0.333 |
| 4 | +0.003 | -0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.881 | 0.287 | 0.562 | 0.613 | 0.537 | 0.600 |
| 2 | 1.000 | 0.871 | 0.864 | 0.246 | 0.512 | 0.450 | 0.450 | 0.487 |
| 3 | 1.000 | 0.821 | 0.823 | 0.241 | 0.471 | 0.485 | 0.412 | 0.456 |
| 4 | 1.000 | 0.770 | 0.769 | 0.237 | 0.417 | 0.208 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1610
**Average Oracle Gain (state probe):** +0.0515
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\transformer\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\transformer\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\transformer\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.946916341781616,3.9126671963050716
1,1,4.0204691886901855,3.33286147821145
2,2,3.355743885040283,2.888681130331071
3,3,2.848907232284546,2.590116907338627
4,4,2.502209186553955,2.3220950267353997
5,5,2.1786208152770996,2.1070103879834785
6,6,1.9098703861236572,1.9482872134349385
7,7,1.7081531286239624,1.8342195104380123
8,8,1.5727028846740723,1.7619101102234886
9,9,1.47916579246521,1.714600860095415
10,10,1.414994716644287,1.6876078120997695
11,11,1.3752115964889526,1.6648482025646774
12,12,1.337943196296692,1.6398798207767675
13,13,1.2992984056472778,1.6132714943807633
14,14,1.2579519748687744,1.587265139720479
15,15,1.2172691822052002,1.5601551493660348
16,16,1.1754140853881836,1.5303272184778431
17,17,1.131488561630249,1.5012069452004355
18,18,1.0899674892425537,1.476866174916752
19,19,1.0547497272491455,1.454531059890497
20,20,1.02179753780365,1.4375377717565319
21,21,0.9939424395561218,1.423462414350666
22,22,0.9684776663780212,1.40998902868052
23,23,0.9409546256065369,1.3992199506915983
24,24,0.9173586964607239,1.3899328513223617
25,25,0.8936715126037598,1.3785021422339268
26,26,0.8682663440704346,1.366835672347272
27,27,0.8462382555007935,1.3562374427670338
28,28,0.8238536715507507,1.3462480638847976
29,29,0.8016750812530518,1.3374120993692367
30,30,0.7799709439277649,1.3297489353867828
31,31,0.7579812407493591,1.3233084756819928
32,32,0.7396538257598877,1.3163244528848617
33,33,0.7203075289726257,1.3076056808721823
34,34,0.7017236351966858,1.3003617583728226
35,35,0.6859386563301086,1.2948996121766136
36,36,0.6687518954277039,1.2897088723104508
37,37,0.6532647013664246,1.286268765809106
38,38,0.6400303840637207,1.2810716472688268
39,39,0.624248206615448,1.2780686675525101
40,40,0.6097032427787781,1.272400902920082
41,41,0.5978350639343262,1.267869667928727
42,42,0.5888239741325378,1.2635045286084785
43,43,0.5757690668106079,1.2591460181064293
44,44,0.5633508563041687,1.2596638163582223
45,45,0.5523561239242554,1.2559415473312627
46,46,0.5435577630996704,1.2520244160636527
47,47,0.534027099609375,1.2511021348296618
48,48,0.5268931984901428,1.2543586355740908
49,49,0.5198640823364258,1.251055733102267
50,50,0.5078152418136597,1.2465859084832864
51,51,0.5021446347236633,1.2489726582511527
52,52,0.4931747317314148,1.2502026167072233
53,53,0.4834236800670624,1.2464084312564037
54,54,0.47755110263824463,1.2425507092085042
55,55,0.469959557056427,1.2478982894147028
56,56,0.46629056334495544,1.2528884137263063
57,57,0.4583909809589386,1.2493622576604124
58,58,0.4532235264778137,1.2465034860079405
59,59,0.4424944519996643,1.2464285678550846
60,60,0.43534427881240845,1.2474918052798412
61,61,0.43130603432655334,1.2484481061091188
62,62,0.4219413995742798,1.2481516853707735
63,63,0.4144679605960846,1.2482945176421618
64,64,0.4104464054107666,1.2488630951428024
65,65,0.4020443260669708,1.2508333550124873
66,66,0.3962878882884979,1.2529474477298925
67,67,0.39053547382354736,1.2535205278240267
68,68,0.3828827142715454,1.2514250708408043
69,69,0.37809255719184875,1.2565338885197874
70,70,0.37368592619895935,1.2569470014728483
71,71,0.36678940057754517,1.2548713058721823
72,72,0.36145174503326416,1.252853268482646
73,73,0.35550200939178467,1.2552897969230277
74,74,0.34870612621307373,1.2597878878233864
75,75,0.34685149788856506,1.2629305730100537
76,76,0.34056922793388367,1.2576108838691087
77,77,0.33145707845687866,1.2584718798027663
78,78,0.3315889537334442,1.2668240656618213
79,79,0.32722386717796326,1.2650187758148694
80,80,0.32054513692855835,1.263351565501729
81,81,0.31361785531044006,1.264310742987961
82,82,0.31031301617622375,1.2645785222288037
83,83,0.30798354744911194,1.2647029689100922
84,84,0.30073773860931396,1.2722872124343623
85,85,0.296534925699234,1.271148681640625
86,86,0.29360562562942505,1.2673557469102203
87,87,0.2900855243206024,1.2739447922003073
88,88,0.28697866201400757,1.274538133965164
89,89,0.28223058581352234,1.2773237384733607
90,90,0.27701425552368164,1.2773999073466316
91,91,0.2767906188964844,1.2759536993308145
92,92,0.27419546246528625,1.28057611184042
93,93,0.2675492763519287,1.2834645255667265
94,94,0.26905909180641174,1.272981737480789
95,95,0.2629168927669525,1.282054088154777
96,96,0.25955426692962646,1.2834685278720543
97,97,0.2539551854133606,1.2892501080622438
98,98,0.25503382086753845,1.282074474897541
99,99,0.2548848092556,1.2890670025934938

```


## File: `reports\transformer\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.002492427825928,4.511295506211578
1,1,4.665741920471191,3.5470093273725665
2,2,3.570535898208618,3.0186077180455944
3,3,2.989933490753174,2.6995139200179303
4,4,2.615719795227051,2.4350423343846055
5,5,2.2938942909240723,2.210012217037013
6,6,2.010181427001953,2.0450435700963756
7,7,1.7998170852661133,1.9261403318311348
8,8,1.649062991142273,1.8423802110015368
9,9,1.5388493537902832,1.7851472448130123
10,10,1.4668493270874023,1.7494266697617828
11,11,1.4142712354660034,1.719878149814293
12,12,1.368739128112793,1.693115734663166
13,13,1.3303289413452148,1.6691734439036885
14,14,1.293771505355835,1.6439610465628203
15,15,1.257677435874939,1.6198067586930072
16,16,1.2198076248168945,1.589994837026127
17,17,1.1750833988189697,1.56109869284708
18,18,1.1341005563735962,1.5373161190845928
19,19,1.097167730331421,1.510469780593622
20,20,1.0588572025299072,1.4901988545402152
21,21,1.0281952619552612,1.474838632052062
22,22,1.0012972354888916,1.465899483102267
23,23,0.9779110550880432,1.4547762010918288
24,24,0.9538374543190002,1.4445699472896387
25,25,0.936832845211029,1.4363166934154072
26,26,0.9166066646575928,1.432950629562628
27,27,0.8959691524505615,1.425373890360848
28,28,0.8791691064834595,1.4204501793032787
29,29,0.8586525321006775,1.4139451824250768
30,30,0.837879478931427,1.4087614465932377
31,31,0.821227490901947,1.4044555914206582
32,32,0.8028395771980286,1.401532157522733
33,33,0.7868059873580933,1.3973631311635502
34,34,0.7720281481742859,1.3969018654745133
35,35,0.7574897408485413,1.394824543937308
36,36,0.7403204441070557,1.3919530149366035
37,37,0.7285529971122742,1.3914527267706198
38,38,0.7143197655677795,1.3921608846695697
39,39,0.6993533372879028,1.3880697781922386
40,40,0.6897113919258118,1.387375753433978
41,41,0.6779962778091431,1.3907693331358864
42,42,0.6687265634536743,1.3830784031602203
43,43,0.6573288440704346,1.3799799934762422
44,44,0.6461829543113708,1.3837022625032018
45,45,0.6352864503860474,1.383284396812564
46,46,0.6252762675285339,1.3818224297195185
47,47,0.6173186898231506,1.382355736904457
48,48,0.6093758940696716,1.376420193031186
49,49,0.6018391847610474,1.3778261278496413
50,50,0.5915355086326599,1.3770071561219261
51,51,0.5826693773269653,1.3759709342581328
52,52,0.5733498334884644,1.3721376012583248
53,53,0.5653145909309387,1.3733950755635247
54,54,0.5587452054023743,1.3831879662685707
55,55,0.5537656545639038,1.3777231935594902
56,56,0.5471893548965454,1.381807671218622
57,57,0.5394206643104553,1.3725524652199668
58,58,0.5290583372116089,1.3746866945360527
59,59,0.5232452154159546,1.3909226714587601
60,60,0.5186747908592224,1.3775802362160605
61,61,0.5107762813568115,1.3762149498110912
62,62,0.5023953318595886,1.3741212438364498
63,63,0.49519944190979004,1.374998499135502
64,64,0.48652827739715576,1.380857623991419
65,65,0.4828433394432068,1.3769446201011784
66,66,0.4783661961555481,1.3757359238921618
67,67,0.46978673338890076,1.3765213763127562
68,68,0.4640115797519684,1.3807217957543545
69,69,0.4576796293258667,1.3799142055824154
70,70,0.4532862603664398,1.378281515152728
71,71,0.44659656286239624,1.3840333281970414
72,72,0.44099661707878113,1.3795348620805583
73,73,0.4341322183609009,1.3877440905961833
74,74,0.429514616727829,1.3862712422355277
75,75,0.42445236444473267,1.3872005275038422
76,76,0.4199715256690979,1.385693034187692
77,77,0.4107647240161896,1.3867980456743083
78,78,0.4056772291660309,1.3896907118500257
79,79,0.40063995122909546,1.3926891889728483
80,80,0.3957911729812622,1.3862819984310963
81,81,0.39134976267814636,1.386598430696081
82,82,0.3834175765514374,1.3915370253265882
83,83,0.3818433880805969,1.3956292574522926
84,84,0.3798905313014984,1.395112834992956
85,85,0.3758261203765869,1.40255862376729
86,86,0.3769208490848541,1.4259505975441855
87,87,0.39384618401527405,1.4917107253778177
88,88,0.4439377188682556,1.5035386632700436
89,89,0.47289586067199707,1.4030316462282275
90,90,0.372531533241272,1.437153925661181
91,91,0.39234763383865356,1.411436487416752
92,92,0.3653543293476105,1.406126428823002
93,93,0.37200385332107544,1.3874591764856556
94,94,0.35450008511543274,1.40162220939261
95,95,0.3532849848270416,1.4161321921426742
96,96,0.3531247675418854,1.4082191342213115
97,97,0.3429390490055084,1.3933883416848105
98,98,0.33758994936943054,1.3891123787301485
99,99,0.3383670449256897,1.3917457705638447

```


## File: `reports\transformer_seed42\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7585051104426384,0.8908125817775726,0.35,0.15,0.6,0.5625,0.4,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,0.0,0.050000000000000044,0.19999999999999996,20,7.180810833139173
2,0.8229287430644036,0.8731510609388351,0.35294117647058826,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.822268320753796
3,1.2116236283498651,0.825780896579518,0.0,0.6666666666666666,0.4264705882352941,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.682207195617478
4,2.013198753197988,0.7764655351638794,,,0.20833333333333334,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.0,0.0,0.16666666666666663,6,2.8487192893553956
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed42\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8089511945843697,0.8823682844638825,0.5,0.15,0.5625,0.5625,0.3,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.03749999999999998,-0.04999999999999999,0.19999999999999996,20,6.733016466903628
2,0.8680663526058197,0.8666419506072998,0.35294117647058826,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.467524835161295
3,1.2555196039816912,0.8195595530902638,0.0,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.518506005839017
4,2.2170957922935486,0.7627033193906149,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.586734475558469
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed42\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8771660536527633,0.8750475108623504,0.2,0.45,0.475,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.04999999999999999,-0.05,-0.2,20,6.348198199650582
2,1.1816006869077682,0.8183781832456589,0.5,0.45,0.375,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,4.9417719986001085
3,1.5265436977148057,0.77690649330616,0.45,0.3,0.4125,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.175,-0.9,0.19999999999999996,20,3.734359803831325
4,2.8130434691905974,0.6524870321154594,0.35,0.55,0.3125,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.1375,1.0,0.19999999999999996,20,2.062958156878607
5,2.8159192204475403,0.737486720085144,,,0.2,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.08750000000000001,0.0,0.19999999999999996,20,2.043590491220075
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed42\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\transformer_seed42\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\transformer_seed42\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.111111,63,0.032884,0.189338
oracle1,probeA_per_label,all,0.515873,63,,
oracle1,probeA_jaccard,all,0.379630,63,,

```


## File: `reports\transformer_seed42\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.111111 |
| probeA_exact_ci_lo | 0.032884 |
| probeA_exact_ci_hi | 0.189338 |
| probeA_per_label | 0.515873 |
| probeA_jaccard | 0.379630 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\transformer_seed42\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.891 | 0.350 | 0.150 | 0.600 | 0.562 | 0.7585 |
| 2 | 0.873 | 0.353 | 0.588 | 0.438 | 0.512 | 0.8229 |
| 3 | 0.826 | 0.000 | 0.667 | 0.426 | 0.471 | 1.2116 |
| 4 | 0.776 | nan | nan | 0.208 | 0.417 | 2.0132 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2886.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\transformer_seed42\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.891 | 0.882 | +0.109 |
| 2 | 1.000 | 0.873 | 0.867 | +0.127 |
| 3 | 1.000 | 0.826 | 0.820 | +0.174 |
| 4 | 1.000 | 0.776 | 0.763 | +0.224 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.600 | 0.562 | -0.037 |
| 2 | 0.512 | 0.438 | 0.475 | +0.075 |
| 3 | 0.471 | 0.426 | 0.471 | +0.044 |
| 4 | 0.417 | 0.208 | 0.250 | +0.208 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.350 | 0.500 | -0.200 |
| 2 | 0.588 | 0.353 | 0.353 | +0.235 |
| 3 | 0.667 | 0.000 | 0.000 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.072 | n/a | +0.429 |
| 2 | +0.049 | -1.000 | +0.000 |
| 3 | +0.034 | n/a | +0.000 |
| 4 | +0.058 | -0.250 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.891 | 0.882 | 0.287 | 0.562 | 0.600 | 0.562 | 0.600 |
| 2 | 1.000 | 0.873 | 0.867 | 0.246 | 0.512 | 0.438 | 0.475 | 0.487 |
| 3 | 1.000 | 0.826 | 0.820 | 0.241 | 0.471 | 0.426 | 0.471 | 0.456 |
| 4 | 1.000 | 0.776 | 0.763 | 0.237 | 0.417 | 0.208 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1584
**Average Oracle Gain (state probe):** +0.0725
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\transformer_seed42\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\transformer_seed42\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\transformer_seed42\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.9831717014312744,4.009930219806608
1,1,4.109736919403076,3.319608094262295
2,2,3.3361992835998535,2.9008573938588627
3,3,2.849261522293091,2.595335913486168
4,4,2.4891300201416016,2.326629138383709
5,5,2.1612613201141357,2.105340426085425
6,6,1.882382869720459,1.9461164630827357
7,7,1.6844515800476074,1.8336624395651895
8,8,1.5454429388046265,1.7577223230580814
9,9,1.4475030899047852,1.7080101888687884
10,10,1.3842391967773438,1.6730095284884092
11,11,1.3392460346221924,1.6487150348600794
12,12,1.3034638166427612,1.6261761774782275
13,13,1.2672131061553955,1.6013217363201204
14,14,1.2321405410766602,1.5709209754818776
15,15,1.1888879537582397,1.5383786060771003
16,16,1.144856333732605,1.5073514844550462
17,17,1.098086953163147,1.481719220270876
18,18,1.0591926574707031,1.4596375012006917
19,19,1.0244557857513428,1.4396118414206582
20,20,0.9913108944892883,1.422227327940894
21,21,0.9616958498954773,1.4097427618308145
22,22,0.9378131628036499,1.397706578989498
23,23,0.910180389881134,1.3860253506019466
24,24,0.8855766654014587,1.3745494905065319
25,25,0.8602156043052673,1.3619396022108734
26,26,0.8372069001197815,1.3509356389280225
27,27,0.8137259483337402,1.3420940461705944
28,28,0.7928802371025085,1.333466326604124
29,29,0.7679232358932495,1.3247728191438268
30,30,0.7493538856506348,1.3160830638447747
31,31,0.7290799617767334,1.3081261056368467
32,32,0.7119781970977783,1.302930613033107
33,33,0.6939883232116699,1.2981927590292008
34,34,0.6770047545433044,1.2954200369412783
35,35,0.6588956713676453,1.2910106221183402
36,36,0.6443713307380676,1.2865535548475922
37,37,0.631266176700592,1.2813237925044825
38,38,0.6190884113311768,1.2788533695408555
39,39,0.6082624793052673,1.2767252687548027
40,40,0.5969972014427185,1.2748868348168545
41,41,0.5835545063018799,1.272554241242956
42,42,0.5736008882522583,1.271512891425461
43,43,0.5627674460411072,1.2700265352843239
44,44,0.5542678833007812,1.2707857225762038
45,45,0.5452345013618469,1.2684999059458248
46,46,0.5372927188873291,1.2635948306224385
47,47,0.5269955396652222,1.2596225425845287
48,48,0.5210789442062378,1.2616659695984886
49,49,0.5104268193244934,1.2655977342949538
50,50,0.5028718113899231,1.2619151131051485
51,51,0.49448928236961365,1.2591342613345287
52,52,0.4881521463394165,1.260785337354316
53,53,0.4811173379421234,1.263764928598873
54,54,0.4718293249607086,1.2607234266937757
55,55,0.4660038650035858,1.2577244492827868
56,56,0.45700445771217346,1.2607769575275358
57,57,0.4535221457481384,1.2604138733910732
58,58,0.44657430052757263,1.2557639450323386
59,59,0.43946340680122375,1.2567475115666624
60,60,0.4323132634162903,1.2595177322137552
61,61,0.42544838786125183,1.2590410826636143
62,62,0.41754844784736633,1.2592303166623975
63,63,0.4129374027252197,1.2621476220302894
64,64,0.4070971608161926,1.2600366561139216
65,65,0.39852583408355713,1.2542952240490524
66,66,0.393259197473526,1.2564334556704662
67,67,0.38912883400917053,1.2649663706294825
68,68,0.38375145196914673,1.264406047883581
69,69,0.3755761682987213,1.259643054399334
70,70,0.3731299936771393,1.2622913298059681
71,71,0.36597928404808044,1.2669290011046364
72,72,0.3607974648475647,1.2701853767770235
73,73,0.35498473048210144,1.2688967595334912
74,74,0.3489089608192444,1.2668884777631917
75,75,0.34433624148368835,1.269746874199539
76,76,0.339958131313324,1.2760833990378457
77,77,0.3354778289794922,1.2708116124887936
78,78,0.3283877968788147,1.2657318115234375
79,79,0.32537221908569336,1.2715073882556351
80,80,0.3217173218727112,1.272035192270748
81,81,0.31526198983192444,1.2670072962026127
82,82,0.3102550506591797,1.2690693589507556
83,83,0.305988073348999,1.2764742491675205
84,84,0.30462008714675903,1.275288190998015
85,85,0.29791566729545593,1.2744508336801998
86,86,0.29575616121292114,1.2748468117635758
87,87,0.2909209430217743,1.2752016411452998
88,88,0.28695201873779297,1.2779418445024333
89,89,0.28452068567276,1.2808004910828636
90,90,0.28174540400505066,1.2791111430183786
91,91,0.27782219648361206,1.282946602242892
92,92,0.27467942237854004,1.2777108364417904
93,93,0.27188000082969666,1.2823731469326332
94,94,0.26741987466812134,1.287007441286181
95,95,0.26218804717063904,1.2849271180199795
96,96,0.2595078945159912,1.2891905737704918
97,97,0.2609902620315552,1.284424828701332
98,98,0.2536908686161041,1.300809516281378
99,99,0.25670126080513,1.2886411322922002

```


## File: `reports\transformer_seed42\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.1259894371032715,4.312902231685451
1,1,4.439151287078857,3.4358020219646517
2,2,3.4714043140411377,2.9861645307697233
3,3,2.958670139312744,2.6237675401030995
4,4,2.54874849319458,2.365660370373335
5,5,2.2309725284576416,2.1586553855020494
6,6,1.977904200553894,1.9948237684906507
7,7,1.7666631937026978,1.8809759421426742
8,8,1.6185215711593628,1.8079383725025615
9,9,1.522674322128296,1.7594411881243597
10,10,1.4513180255889893,1.7207148817719007
11,11,1.395890474319458,1.695695845807185
12,12,1.3575025796890259,1.673842258140689
13,13,1.3231662511825562,1.649656452116419
14,14,1.2890892028808594,1.6211585373174948
15,15,1.2452805042266846,1.590679356309234
16,16,1.2020769119262695,1.5612574092677383
17,17,1.160300374031067,1.536192596935835
18,18,1.1202067136764526,1.5143664000464268
19,19,1.0893319845199585,1.4958237194624104
20,20,1.0580413341522217,1.4804373569175846
21,21,1.030963659286499,1.4659186191246159
22,22,1.004300832748413,1.456757092085041
23,23,0.9815067648887634,1.446949067662974
24,24,0.9587657451629639,1.4385018270523822
25,25,0.9392420053482056,1.433440161533043
26,26,0.9203327298164368,1.42787232946177
27,27,0.9039118885993958,1.4221517844278304
28,28,0.8812775015830994,1.4173846635662142
29,29,0.863491952419281,1.413177490234375
30,30,0.8453913927078247,1.4089341710825436
31,31,0.8260653614997864,1.4061930922211194
32,32,0.8113631010055542,1.4038135966316598
33,33,0.7970874905586243,1.3976267830270235
34,34,0.7803642749786377,1.3958368770411758
35,35,0.7644625902175903,1.3972933409643955
36,36,0.751610279083252,1.3919632708440062
37,37,0.7350660562515259,1.3921166091668802
38,38,0.7236043214797974,1.3888970046746927
39,39,0.7117878198623657,1.3840487120581455
40,40,0.6972238421440125,1.3855525782850922
41,41,0.6879531741142273,1.3841771610447617
42,42,0.674026370048523,1.381031098912974
43,43,0.6623935103416443,1.378965784291752
44,44,0.6537154316902161,1.3767762731333248
45,45,0.641623318195343,1.377643647741099
46,46,0.6315291523933411,1.3736112000512295
47,47,0.6216274499893188,1.3747796230628841
48,48,0.6109194755554199,1.3777677192062627
49,49,0.6029966473579407,1.3716980981045082
50,50,0.5946552753448486,1.3736039458728226
51,51,0.5853285193443298,1.3753604576235912
52,52,0.576568067073822,1.3733554277263704
53,53,0.5704958438873291,1.3744416784067623
54,54,0.5607346892356873,1.3726241314997438
55,55,0.5520629286766052,1.3739710323146133
56,56,0.5439003109931946,1.3749244564869365
57,57,0.5384470820426941,1.3759385405993851
58,58,0.5294852256774902,1.377779350906122
59,59,0.5221675634384155,1.3774943117235527
60,60,0.5157572627067566,1.379071220022733
61,61,0.5095818042755127,1.3723026963530993
62,62,0.5010896921157837,1.3784077128425973
63,63,0.49706709384918213,1.3749502213274847
64,64,0.48962152004241943,1.3767670178022542
65,65,0.4815024137496948,1.3785355364690062
66,66,0.4771343469619751,1.3785429157194544
67,67,0.4676482379436493,1.3834747564597207
68,68,0.46329110860824585,1.3815392666175716
69,69,0.4579833149909973,1.3865437742139473
70,70,0.4506087005138397,1.380748936387359
71,71,0.442922979593277,1.3810837542424437
72,72,0.4383869767189026,1.3886163430135758
73,73,0.4344671070575714,1.3844752077196465
74,74,0.42871686816215515,1.394746874199539
75,75,0.4242470860481262,1.392412154400935
76,76,0.4190583825111389,1.3889855556800716
77,77,0.41249507665634155,1.3899591164510758
78,78,0.40595871210098267,1.388671875
79,79,0.40316858887672424,1.3912205930616035
80,80,0.3973363935947418,1.3891651591316598
81,81,0.39112386107444763,1.3950930736103997
82,82,0.38462889194488525,1.3957916009621543
83,83,0.3839274048805237,1.4028740554559427
84,84,0.3812054991722107,1.405684549300397
85,85,0.37641680240631104,1.4104516701620133
86,86,0.37024402618408203,1.4084582719646517
87,87,0.3728539049625397,1.4341233050236937
88,88,0.38763752579689026,1.4669904865202357
89,89,0.411985844373703,1.460981400286565
90,90,0.4273223280906677,1.4375212622470543
91,91,0.39130282402038574,1.4125322435722976
92,92,0.3583741784095764,1.4445323006051485
93,93,0.3904036283493042,1.3949765064677253
94,94,0.34206610918045044,1.4212287527615908
95,95,0.3699372112751007,1.4003982543945312
96,96,0.3371219038963318,1.4215500628361937
97,97,0.35106855630874634,1.4053920057953382
98,98,0.3302454948425293,1.4089882022044697
99,99,0.3348486125469208,1.4010712670498207

```


## File: `reports\transformer_seed43\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7357936844229698,0.8946233868598938,0.4,0.15,0.6,0.5625,0.3,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,0.0,-0.04999999999999999,0.19999999999999996,20,7.40245782121569
2,0.8323258593678474,0.8718349128961563,0.35294117647058826,0.5882352941176471,0.4375,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.04999999999999999,0.30000000000000004,0.30000000000000004,20,6.745243621662847
3,1.238675255985821,0.8282624342862297,0.0,0.6666666666666666,0.4411764705882353,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.01470588235294118,0.35294117647058826,0.2941176470588235,17,4.579951721506605
4,2.1725316643714905,0.7888087431589762,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.6397949524019837
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed43\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8078670591115952,0.8823232054710388,0.55,0.15,0.575,0.5625,0.3,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.04999999999999999,0.19999999999999996,20,6.742051990642611
2,0.8495850652456284,0.8685230255126953,0.5882352941176471,0.5882352941176471,0.475,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.012500000000000011,0.30000000000000004,0.30000000000000004,20,6.608214908324519
3,1.2093026322477005,0.8200536300154293,0.5,0.6666666666666666,0.4264705882352941,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,-0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.691193684491943
4,1.8195081154505413,0.7791571418444315,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,3.151971718532977
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed43\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8733845263719558,0.8766283243894577,0.2,0.45,0.45,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.025000000000000022,-0.05,-0.2,20,6.375684242683279
2,1.2113339364528657,0.8149743288755417,0.45,0.45,0.4125,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.0625,-0.05,-0.2,20,4.820471888359972
3,1.6144418478012086,0.7736942857503891,0.35,0.3,0.425,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.1875,-0.9,0.19999999999999996,20,3.5310428996264154
4,3.095461428165436,0.6586707085371017,0.35,0.55,0.3375,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.16250000000000003,1.0,0.19999999999999996,20,1.8747418131648863
5,3.3807533264160154,0.7544293850660324,,,0.3125,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.2,0.0,0.19999999999999996,20,1.7021607870607223
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed43\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\transformer_seed43\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\transformer_seed43\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.111111,63,0.032884,0.189338
oracle1,probeA_per_label,all,0.547619,63,,
oracle1,probeA_jaccard,all,0.428571,63,,

```


## File: `reports\transformer_seed43\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.111111 |
| probeA_exact_ci_lo | 0.032884 |
| probeA_exact_ci_hi | 0.189338 |
| probeA_per_label | 0.547619 |
| probeA_jaccard | 0.428571 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\transformer_seed43\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.895 | 0.400 | 0.150 | 0.600 | 0.562 | 0.7358 |
| 2 | 0.872 | 0.353 | 0.588 | 0.438 | 0.512 | 0.8323 |
| 3 | 0.828 | 0.000 | 0.667 | 0.441 | 0.471 | 1.2387 |
| 4 | 0.789 | nan | nan | 0.250 | 0.417 | 2.1725 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **1**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~1 step(s)**.

Single-step transition validation MSE: 1.2950.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\transformer_seed43\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.882 | +0.105 |
| 2 | 1.000 | 0.872 | 0.869 | +0.128 |
| 3 | 1.000 | 0.828 | 0.820 | +0.172 |
| 4 | 1.000 | 0.789 | 0.779 | +0.211 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.600 | 0.575 | -0.037 |
| 2 | 0.512 | 0.438 | 0.475 | +0.075 |
| 3 | 0.471 | 0.441 | 0.426 | +0.029 |
| 4 | 0.417 | 0.250 | 0.250 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.550 | -0.250 |
| 2 | 0.588 | 0.353 | 0.588 | +0.235 |
| 3 | 0.667 | 0.000 | 0.500 | +0.667 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.105 | -2.000 | +0.375 |
| 2 | +0.025 | -1.000 | n/a |
| 3 | +0.046 | +0.333 | -3.000 |
| 4 | +0.044 | +0.000 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.895 | 0.882 | 0.287 | 0.562 | 0.600 | 0.575 | 0.600 |
| 2 | 1.000 | 0.872 | 0.869 | 0.246 | 0.512 | 0.438 | 0.475 | 0.487 |
| 3 | 1.000 | 0.828 | 0.820 | 0.241 | 0.471 | 0.441 | 0.426 | 0.456 |
| 4 | 1.000 | 0.789 | 0.779 | 0.237 | 0.417 | 0.250 | 0.250 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1541
**Average Oracle Gain (state probe):** +0.0584
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\transformer_seed43\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\transformer_seed43\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\transformer_seed43\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,3.9045751094818115,4.056257904553022
1,1,4.179919242858887,3.3665616395043547
2,2,3.3926823139190674,2.922975633965164
3,3,2.8961129188537598,2.5925758236744363
4,4,2.4942991733551025,2.2941784467853483
5,5,2.1426022052764893,2.0774539884973744
6,6,1.881136178970337,1.926179729524206
7,7,1.692319631576538,1.833515730060515
8,8,1.5708118677139282,1.7596713206807122
9,9,1.477528691291809,1.7109666417856686
10,10,1.409532070159912,1.6829229886414574
11,11,1.3662939071655273,1.6609702188460553
12,12,1.327392339706421,1.6394285608510502
13,13,1.292099952697754,1.6122375238137168
14,14,1.2513065338134766,1.5821306822729893
15,15,1.211014986038208,1.5537472083920338
16,16,1.1687567234039307,1.5233907230564805
17,17,1.1224180459976196,1.4973494732966188
18,18,1.0861622095108032,1.4725551917904713
19,19,1.0492260456085205,1.4505365090292008
20,20,1.0148515701293945,1.4332332923764088
21,21,0.9839121103286743,1.4200557020844007
22,22,0.9577561020851135,1.4084312564036885
23,23,0.9320705533027649,1.397706078701332
24,24,0.9118756651878357,1.3847997696673284
25,25,0.887930154800415,1.3734653660508453
26,26,0.8653261065483093,1.3643524920354124
27,27,0.8446065783500671,1.3538574468894082
28,28,0.8246766328811646,1.3447145555840163
29,29,0.7991056442260742,1.3361611288101947
30,30,0.7802187204360962,1.3283600103659707
31,31,0.7586885690689087,1.3209867633757044
32,32,0.7380572557449341,1.3159207203349128
33,33,0.7216364145278931,1.3092466260566087
34,34,0.7041965126991272,1.3059162077356556
35,35,0.6872453093528748,1.3031521156185963
36,36,0.673047661781311,1.2977267406025872
37,37,0.6592907905578613,1.2895152607902152
38,38,0.6443468332290649,1.2839977076796234
39,39,0.63139808177948,1.2804012611264088
40,40,0.6183556318283081,1.2770089321449154
41,41,0.6060596704483032,1.2738423581983223
42,42,0.5938507914543152,1.2735740786693135
43,43,0.5840336680412292,1.2703632292200306
44,44,0.5731699466705322,1.2675548616002819
45,45,0.5643959641456604,1.2684413722304047
46,46,0.5556505918502808,1.2623431096311475
47,47,0.546606183052063,1.2618360675749232
48,48,0.536026120185852,1.265122335465228
49,49,0.5272440910339355,1.2672276731397285
50,50,0.5211375951766968,1.2611688082335426
51,51,0.5105067491531372,1.2584438636654713
52,52,0.5067455172538757,1.260804973664831
53,53,0.49598339200019836,1.2646964651639345
54,54,0.49000877141952515,1.263713273845735
55,55,0.4822905659675598,1.2604027419793802
56,56,0.4760392904281616,1.2589815483718623
57,57,0.4671797454357147,1.2604332595575052
58,58,0.4606249928474426,1.258909131659836
59,59,0.4532792568206787,1.2591202532658812
60,60,0.44462844729423523,1.260670020932057
61,61,0.44281822443008423,1.258257381251601
62,62,0.43221020698547363,1.260194371958248
63,63,0.42731228470802307,1.26434576315958
64,64,0.4216538369655609,1.2603307004834785
65,65,0.4149371087551117,1.2614596007300205
66,66,0.4099177122116089,1.2659968391793672
67,67,0.4031372666358948,1.2658592599337217
68,68,0.3953583538532257,1.2672086621894212
69,69,0.3909340500831604,1.2682342529296875
70,70,0.38664063811302185,1.262911436987705
71,71,0.37899577617645264,1.268146577428599
72,72,0.37415212392807007,1.2696894661324922
73,73,0.37072479724884033,1.2617850381819928
74,74,0.3637429475784302,1.2656581440909964
75,75,0.3598206639289856,1.2735360567686989
76,76,0.3548060655593872,1.2690948736472207
77,77,0.34717315435409546,1.2710182315013447
78,78,0.34609028697013855,1.2756242595735143
79,79,0.3403375446796417,1.2742084440637806
80,80,0.3342592716217041,1.2731986124007428
81,81,0.33379438519477844,1.2771576428022542
82,82,0.32500898838043213,1.2830869330734502
83,83,0.31995096802711487,1.2777173401879482
84,84,0.31875017285346985,1.2783165603387552
85,85,0.31187909841537476,1.2847750304175205
86,86,0.3087101876735687,1.2829339699667008
87,87,0.30517324805259705,1.2885099317206712
88,88,0.30158731341362,1.2876689473136527
89,89,0.2953455150127411,1.2827418593109632
90,90,0.2889884114265442,1.2857354586241676
91,91,0.29082876443862915,1.2841425411036758
92,92,0.28611305356025696,1.2890509933721823
93,93,0.2787545323371887,1.2902155391505508
94,94,0.2786872982978821,1.2897223800909323
95,95,0.2725081145763397,1.2888683881915983
96,96,0.27075570821762085,1.2927567528896644
97,97,0.26717203855514526,1.3005125952548668
98,98,0.26252424716949463,1.2936445142402024
99,99,0.2617762088775635,1.2950346899814293

```


## File: `reports\transformer_seed43\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.293334007263184,4.324599969582479
1,1,4.444897651672363,3.5024949370837604
2,2,3.5615274906158447,3.027191162109375
3,3,3.012420177459717,2.7277634417424435
4,4,2.660025119781494,2.4562563036308913
5,5,2.3301868438720703,2.2249525726818646
6,6,2.055187702178955,2.037082359439037
7,7,1.8219060897827148,1.906969289310643
8,8,1.6538286209106445,1.8176063162381533
9,9,1.5372552871704102,1.7567693991739242
10,10,1.4512251615524292,1.7175374265576973
11,11,1.3954015970230103,1.6929156193967725
12,12,1.3540873527526855,1.6720978783779457
13,13,1.3189222812652588,1.6504426549692623
14,14,1.2837848663330078,1.631499493708376
15,15,1.2505139112472534,1.6070005072922002
16,16,1.2147274017333984,1.5797571901415215
17,17,1.1749039888381958,1.5541129190413678
18,18,1.1400890350341797,1.5306539066502305
19,19,1.1041475534439087,1.507994855036501
20,20,1.0699717998504639,1.4880042154280866
21,21,1.0405128002166748,1.47059443739594
22,22,1.0162473917007446,1.4594598989017675
23,23,0.9917224645614624,1.4510060294729765
24,24,0.9688084125518799,1.4454416994188652
25,25,0.9484630227088928,1.4399929359310963
26,26,0.9298291206359863,1.4360491643186475
27,27,0.9117885828018188,1.4300442054623463
28,28,0.8891944289207458,1.4237898529553024
29,29,0.8734427690505981,1.41508171206615
30,30,0.8557782173156738,1.4093968125640368
31,31,0.8357129096984863,1.406781431104316
32,32,0.8158905506134033,1.4046365706647028
33,33,0.7988102436065674,1.3974534331775101
34,34,0.7814696431159973,1.392549358430456
35,35,0.7676538825035095,1.3935271716508708
36,36,0.752399742603302,1.395035165255187
37,37,0.7353677153587341,1.3931969814613216
38,38,0.7228297591209412,1.3877525954950052
39,39,0.7073255777359009,1.385682152920082
40,40,0.6952173113822937,1.3825293368980534
41,41,0.6824021339416504,1.382841516713627
42,42,0.6723476052284241,1.3823913824362832
43,43,0.6581340432167053,1.381164300637167
44,44,0.6492083668708801,1.3768555688076332
45,45,0.637955367565155,1.375943543481045
46,46,0.6289169788360596,1.3767202408587347
47,47,0.6184362769126892,1.3757832011238473
48,48,0.610327422618866,1.3790468309746413
49,49,0.6045600771903992,1.3789631577788806
50,50,0.5957840085029602,1.378129427550269
51,51,0.5893895030021667,1.3772220298892162
52,52,0.5791758894920349,1.3740672127145235
53,53,0.5683919191360474,1.3759280345478997
54,54,0.5625345706939697,1.3735067648965804
55,55,0.5568249821662903,1.3681222884381403
56,56,0.5469573140144348,1.3672826798235784
57,57,0.5372903347015381,1.3711980600826075
58,58,0.5344365835189819,1.3718526871477972
59,59,0.529403030872345,1.3664655841764857
60,60,0.5188986659049988,1.3721391021228226
61,61,0.513389527797699,1.3733130283043034
62,62,0.5061200261116028,1.374999499711834
63,63,0.5016211271286011,1.3785805624039447
64,64,0.4931505024433136,1.3720037741739242
65,65,0.4864008128643036,1.371186303310707
66,66,0.47970274090766907,1.3754304979668288
67,67,0.47207945585250854,1.3733431706663037
68,68,0.46462151408195496,1.3770596863793545
69,69,0.46104300022125244,1.3783016517514088
70,70,0.45446309447288513,1.3755635746189805
71,71,0.4477125108242035,1.3762567238729508
72,72,0.4417377710342407,1.3823124619780993
73,73,0.43707630038261414,1.38286015244781
74,74,0.43277618288993835,1.3749824899141905
75,75,0.42836901545524597,1.3913436639504355
76,76,0.4261418282985687,1.3888784940125511
77,77,0.4213966727256775,1.384886194448002
78,78,0.4161534011363983,1.3829926037397542
79,79,0.4051905572414398,1.386188819760182
80,80,0.39751148223876953,1.3849307200947747
81,81,0.396064430475235,1.390810106621414
82,82,0.3904185891151428,1.3953039450723617
83,83,0.38760143518447876,1.3910248553166624
84,84,0.3814944326877594,1.3936329825979765
85,85,0.37533867359161377,1.3930097486152024
86,86,0.37161505222320557,1.3897172271228226
87,87,0.3677787482738495,1.4034826560098617
88,88,0.3633647561073303,1.399140404873207
89,89,0.35925111174583435,1.3949678764968623
90,90,0.35656872391700745,1.4146688492571722
91,91,0.35745713114738464,1.4080965636206455
92,92,0.35775309801101685,1.422975633965164
93,93,0.3613519072532654,1.434069649117892
94,94,0.36586353182792664,1.4300582135309938
95,95,0.36319616436958313,1.4081335849449284
96,96,0.3390853703022003,1.4149426319560066
97,97,0.33735811710357666,1.4210765400870902
98,98,0.33611273765563965,1.4057121902215677
99,99,0.3299493193626404,1.4114428660908684

```


## File: `reports\transformer_seed44\coherence_action_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.7380437552928925,0.893699124455452,0.4,0.15,0.575,0.5625,0.25,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.025000000000000022,-0.09999999999999998,0.19999999999999996,20,7.37988997941783
2,0.8236397564411163,0.8729872465133667,0.4117647058823529,0.5882352941176471,0.4625,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,-0.024999999999999967,0.30000000000000004,0.30000000000000004,20,6.816378944970699
3,1.2152569153729607,0.827056372866911,0.3333333333333333,0.6666666666666666,0.5147058823529411,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.05882352941176466,0.35294117647058826,0.2941176470588235,17,4.668208671990019
4,2.09912117322286,0.7747823297977448,,,0.25,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.04166666666666666,0.0,0.16666666666666663,6,2.7321138935186484
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed44\coherence_blind_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.824271610379219,0.880249235033989,0.5,0.15,0.55,0.5625,0.3,0.45,0.7,0.75,5.446681714057922,0.28723630383610727,0.5,0.6,0.35,0.5,-0.04999999999999993,-0.04999999999999999,0.19999999999999996,20,6.607872508859175
2,0.8510125517845154,0.8691343188285827,0.5294117647058824,0.5882352941176471,0.525,0.5125,0.55,0.65,1.0,0.95,5.61424069404602,0.24624466970562936,0.4117647058823529,0.4875,0.25,0.7,0.03750000000000003,0.30000000000000004,0.30000000000000004,20,6.597130303511199
3,1.2442391795270584,0.8241116825272056,0.5,0.6666666666666666,0.4852941176470588,0.47058823529411764,0.35294117647058826,0.058823529411764705,1.0,1.0,5.673072871039896,0.24148888798320994,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.02941176470588236,0.35294117647058826,0.2941176470588235,17,4.559471333474854
4,2.1403077046076455,0.7751516103744507,,,0.2916666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.735038121541341,0.23732860138018927,,0.20833333333333334,0.0,0.8333333333333334,0.08333333333333334,0.0,0.16666666666666663,6,2.6795390724403667
5,,,,,,,,,,,,,,,,,,,,0,
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed44\coherence_ood_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples,dynamics_gain
1,0.8709316045045853,0.8758100092411041,0.2,0.45,0.4875,0.475,0.0,0.0,0.0,0.2,5.568423962593078,0.2720899984240532,0.3,0.425,0.05,0.2,0.0625,-0.05,-0.2,20,6.393640940106407
2,1.209949505329132,0.8157121300697326,0.5,0.45,0.4,0.5,0.0,0.05,0.0,0.05,5.839201188087463,0.2155156686902046,0.4,0.35,0.05,0.2,0.050000000000000044,-0.05,-0.2,20,4.82598749978337
3,1.600602352619171,0.7732988983392716,0.45,0.3,0.4375,0.4375,0.0,0.15,1.0,1.0,5.700663423538208,0.23710524737834932,0.3,0.2375,0.9,0.8,0.2,-0.9,0.19999999999999996,20,3.5615738126398706
4,3.029014527797699,0.6488577216863632,0.4,0.55,0.375,0.4625,1.0,0.9,1.0,1.0,5.803190970420838,0.2201945848762989,0.35,0.175,0.0,0.8,0.2,1.0,0.19999999999999996,20,1.9158676583304983
5,3.3465423226356505,0.7299038141965866,,,0.3125,0.3625,0.0,0.0,1.0,1.0,5.7545857429504395,0.23032009825110436,,0.1125,0.0,0.8,0.2,0.0,0.19999999999999996,20,1.71956162156595
6,,,,,,,,,,,,,,,,,,,,0,
7,,,,,,,,,,,,,,,,,,,,0,
8,,,,,,,,,,,,,,,,,,,,0,

```


## File: `reports\transformer_seed44\coherence_oracle_depth.csv`
```csv
depth,mse,cosine_similarity,operator_accuracy,teacher_operator_accuracy,state_probe_accuracy,teacher_state_probe_accuracy,dist_probe_accuracy,teacher_dist_probe_accuracy,reach_probe_accuracy,teacher_reach_probe_accuracy,identity_mse,identity_cosine_similarity,identity_operator_accuracy,identity_state_probe_accuracy,identity_dist_probe_accuracy,identity_reach_probe_accuracy,semantic_gain_state,semantic_gain_dist,semantic_gain_reach,n_samples
1,0.0,0.9999998450279236,0.15,0.15,0.5625,0.5625,0.45,0.45,0.75,0.75,5.446681618690491,0.28723626732826235,0.5,0.6,0.35,0.5,-0.03749999999999998,0.10000000000000003,0.25,20
2,0.0,0.9999999016523361,0.5882352941176471,0.5882352941176471,0.5125,0.5125,0.65,0.65,0.95,0.95,5.614240741729736,0.24624463841319083,0.4117647058823529,0.4875,0.25,0.7,0.024999999999999967,0.4,0.25,20
3,0.0,0.9999998527414659,0.6666666666666666,0.6666666666666666,0.47058823529411764,0.47058823529411764,0.058823529411764705,0.058823529411764705,1.0,1.0,5.673072814941406,0.24148884766242085,0.3333333333333333,0.45588235294117646,0.0,0.7058823529411765,0.01470588235294118,0.058823529411764705,0.2941176470588235,17
4,0.0,0.9999998609224955,,,0.4166666666666667,0.4166666666666667,0.0,0.0,1.0,1.0,5.7350380420684814,0.23732855916023254,,0.20833333333333334,0.0,0.8333333333333334,0.20833333333333334,0.0,0.16666666666666663,6
5,,,,,,,,,,,,,,,,,,,,0
6,,,,,,,,,,,,,,,,,,,,0
7,,,,,,,,,,,,,,,,,,,,0
8,,,,,,,,,,,,,,,,,,,,0

```


## File: `reports\transformer_seed44\oracle_audit_coverage.md`
```md
# Oracle Audit — Coverage

| Field | Value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_95ci | [0.000000, 0.057471] |
| total_state_instances | 83 |

## Swap pairs by delta_depth

| delta_depth | n_pairs |
|---|---|
| all | 0 |

```


## File: `reports\transformer_seed44\oracle_audit_metrics.csv`
```csv
oracle,metric,delta_depth,value,n,ci_lo,ci_hi
oracle0,probeA_exact,all,0.238095,63,,
oracle0,probeA_per_label,all,0.607143,63,,
oracle0,probeA_jaccard,all,0.500000,63,,
oracle0,probeB_accuracy,all,0.460317,63,,
oracle0,probeC_accuracy,all,0.380952,63,,
oracle0,probeD_accuracy,all,0.698413,63,,
coverage,coverage_rate,all,0.000000,63,0.000000,0.057471
oracle1,probeA_exact,all,0.126984,63,0.044106,0.209862
oracle1,probeA_per_label,all,0.519841,63,,
oracle1,probeA_jaccard,all,0.406085,63,,

```


## File: `reports\transformer_seed44\oracle_audit_summary.md`
```md
# Oracle Audit — Summary

## 1. Coverage

| metric | value |
|---|---|
| total_states | 63 |
| states_with_swap_partner | 0 |
| states_without_swap_partner | 63 |
| coverage_rate | 0.000000 |
| coverage_rate_ci_lo | 0.000000 |
| coverage_rate_ci_hi | 0.057471 |
| total_state_instances | 83 |

## 2. Oracle 0 — Teacher Ceiling

| metric | value |
|---|---|
| probeA_exact | 0.238095 |
| probeA_per_label | 0.607143 |
| probeA_jaccard | 0.500000 |
| probeB_accuracy | 0.460317 |
| probeC_accuracy | 0.380952 |
| probeD_accuracy | 0.698413 |
| n | 63 |

## 3. Oracle 1 — True-State Transition

| metric | value |
|---|---|
| probeA_exact | 0.126984 |
| probeA_exact_ci_lo | 0.044106 |
| probeA_exact_ci_hi | 0.209862 |
| probeA_per_label | 0.519841 |
| probeA_jaccard | 0.406085 |
| n | 63 |

## 4. Oracle 2A — Representation Agreement

| metric | all |  |
|---|---|
| probeA_pred_exact | nan |
| probeA_pred_per_label | nan |
| probeA_pred_jaccard | nan |
| probeB_pred_abs_diff | nan |
| probeB_pred_agreement | nan |
| probeC_pred_agreement | nan |
| probeD_pred_agreement | nan |
| gtA_exact | nan |
| gtA_per_label | nan |
| gtA_jaccard | nan |
| gtB_abs_diff | nan |
| gtB_agreement | nan |
| gtC_agreement | nan |
| gtD_agreement | nan |

## 4b. Oracle 2A — Representation Agreement (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_pred_exact | nan | nan | nan | 0 |
| probeA_pred_per_label | nan | nan | nan | 0 |
| probeA_pred_jaccard | nan | nan | nan | 0 |
| probeB_pred_abs_diff | nan | nan | nan | 0 |
| probeB_pred_agreement | nan | nan | nan | 0 |
| probeC_pred_agreement | nan | nan | nan | 0 |
| probeD_pred_agreement | nan | nan | nan | 0 |
| gtA_exact | nan | nan | nan | 0 |
| gtA_per_label | nan | nan | nan | 0 |
| gtA_jaccard | nan | nan | nan | 0 |
| gtB_abs_diff | nan | nan | nan | 0 |
| gtB_agreement | nan | nan | nan | 0 |
| gtC_agreement | nan | nan | nan | 0 |
| gtD_agreement | nan | nan | nan | 0 |

## 5. Oracle 2B — Swapped-State Transition

| metric | all |  |
|---|---|
| probeA_exact | nan |
| probeA_per_label | nan |
| probeA_jaccard | nan |

## 6. Oracle 2B — Delta

| metric | all |  |
|---|---|
| oracle1_paired_exact | nan |
| delta_transition_accuracy | nan |

## 6b. Oracle 2B — Transition + Delta (95% CI, all pairs)

| metric | mean | ci_lo | ci_hi | n |
|---|---|---|---|---|
| probeA_exact | nan | nan | nan | 0 |
| probeA_per_label | nan | nan | nan | 0 |
| probeA_jaccard | nan | nan | nan | 0 |
| oracle1_paired_exact | nan | nan | nan | 0 |
| delta_transition_accuracy | nan | nan | nan | 0 |

```


## File: `reports\transformer_seed44\phase_a_report.md`
```md
# Phase A Report — Is Latent Planning Viable?

- **Teacher model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- **Hidden layer probed:** -1
- **Hidden dim:** 2048
- **Trajectories:** train=20 val=20 test=20
- **Transition Architecture:** `transformer`
- **Transition Parameters:** 2,635,840
- **d_model:** 256
- **Bottleneck:** 256
- **Layers:** 2

---

## 1. Do hidden states contain reasoning information?

**Yes.** The following quantities are linearly decodable from the frozen hidden state (above majority-class baseline):
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

See `probe_results.csv` and `probe_report.md` for full metrics.

## 2. How quickly does coherence decay?

Rollouts were evaluated to depth 4 (limited by solution length in the data).

| depth | cosine | operator acc | teacher op acc | probe acc | teacher probe acc | MSE |
|---|---|---|---|---|---|---|
| 1 | 0.894 | 0.400 | 0.150 | 0.575 | 0.562 | 0.7380 |
| 2 | 0.873 | 0.412 | 0.588 | 0.463 | 0.512 | 0.8236 |
| 3 | 0.827 | 0.333 | 0.667 | 0.515 | 0.471 | 1.2153 |
| 4 | 0.775 | nan | nan | 0.250 | 0.417 | 2.0991 |

Cosine similarity stays >= 0.90 through depth **0**; rollout operator accuracy stays >= 0.50 through depth **0**; rollout state probe accuracy stays >= 0.50 through depth **3**.

See `coherence_depth.csv` / `coherence_depth.png`.

## 3. What is the effective planning horizon?

Taking the depth at which the rolled-out latent still resembles the teacher (cosine >= 0.9) and retains operations (acc >= 0.5) and state information (acc >= 0.5), the **effective horizon is ~3 step(s)**.

Single-step transition validation MSE: 1.2945.

## 4. Is latent planning worth pursuing?

**Mixed / representation-limited bottleneck.** The representation encodes reasoning information, but the learned dynamics lose coherence by depth 3. The bottleneck is *dynamics*, not representation — worth pursuing only with a stronger transition model.

**Bottleneck diagnosis:** dynamics.

_Decision rule: 'viable' requires both (a) task information linearly present in the representation and (b) rollout coherence surviving beyond depth 3._

```


## File: `reports\transformer_seed44\phase_b_oracle_report.md`
```md
# Phase B Report — Oracle Transition Diagnostic

The Oracle Transition returns the **exact teacher hidden state** at each depth. It performs no learning and no prediction. It establishes the theoretical maximum coherence achievable under perfect dynamics.

## 0. Sanity Check

✅ **PASS**: Oracle cosine ≈ 1.0 and Oracle MSE ≈ 0.0 at all depths. The Oracle is correctly returning the exact teacher state.

## 1. How much coherence survives under perfect transitions?

Since the Oracle returns the exact teacher state, its cosine and MSE are trivially perfect. The informative metric is **probe accuracy**: how much task information do the *teacher states themselves* contain at each depth?

| Depth | Oracle State Acc | Oracle Op Acc | n_samples |
|---|---|---|---|
| 1 | 0.562 | 0.150 | 20 |
| 2 | 0.512 | 0.588 | 20 |
| 3 | 0.471 | 0.667 | 17 |
| 4 | 0.417 | nan | 6 |

## 2. Transition Learning Error vs Representation Limitations

The **Oracle Gain** at each depth measures how much coherence the learned transition model leaves on the table.

### Cosine Similarity Gain

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain (vs Action) |
|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.880 | +0.106 |
| 2 | 1.000 | 0.873 | 0.869 | +0.127 |
| 3 | 1.000 | 0.827 | 0.824 | +0.173 |
| 4 | 1.000 | 0.775 | 0.775 | +0.225 |

### State Probe Accuracy Gain (Probe A)

*Probe A Definition: Countdown (Remaining operands (25, 50, 75, 100) encoding)*

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.562 | 0.575 | 0.550 | -0.012 |
| 2 | 0.512 | 0.463 | 0.525 | +0.050 |
| 3 | 0.471 | 0.515 | 0.485 | -0.044 |
| 4 | 0.417 | 0.250 | 0.292 | +0.167 |

### Operator Accuracy Gain (Probe C)

| Depth | Oracle | Action-Cond. | Blind | Oracle Gain |
|---|---|---|---|---|
| 1 | 0.150 | 0.400 | 0.500 | -0.250 |
| 2 | 0.588 | 0.412 | 0.529 | +0.176 |
| 3 | 0.667 | 0.333 | 0.500 | +0.333 |
| 4 | nan | nan | nan | +nan |

## 2b. Does the Action Vector actually drive dynamics?

The **Action Gain** measures how much of the theoretically available planning signal (Oracle - Blind) is captured by the action conditioning. Formula: `(Action - Blind) / (Oracle - Blind)`. Higher is better.

| Depth | Cosine Action Gain | State Acc Action Gain | Op Acc Action Gain |
|---|---|---|---|
| 1 | +0.112 | +2.000 | +0.286 |
| 2 | +0.029 | +5.000 | -2.000 |
| 3 | +0.017 | -2.000 | -1.000 |
| 4 | -0.002 | -0.333 | n/a |

## 3. Full Depth-by-Depth Comparison

| Depth | Oracle Cos | Action Cos | Blind Cos | Identity Cos | Oracle State | Action State | Blind State | Identity State |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.894 | 0.880 | 0.287 | 0.562 | 0.575 | 0.550 | 0.600 |
| 2 | 1.000 | 0.873 | 0.869 | 0.246 | 0.512 | 0.463 | 0.525 | 0.487 |
| 3 | 1.000 | 0.827 | 0.824 | 0.241 | 0.471 | 0.515 | 0.485 | 0.456 |
| 4 | 1.000 | 0.775 | 0.775 | 0.237 | 0.417 | 0.250 | 0.292 | 0.208 |

## 4. Bottleneck Diagnosis

**Average Oracle Gain (cosine):** +0.1579
**Average Oracle Gain (state probe):** +0.0400
**Average Oracle Gain (operator):** +nan

**Average Oracle State Probe Accuracy:** 0.4906
**Average Oracle Operator Accuracy:** 0.4683

### Interpretation

**Case C — Representation Instability.** Even under perfect (Oracle) transitions, the teacher hidden states yield low probe accuracy. The frozen hidden state is not a stable planning state. The bottleneck is in the **representation itself**, not the learned dynamics.

---

_This report was generated by the Phase B Oracle Transition Diagnostic. The Oracle performs no learning; it simply returns the teacher state at each depth. It answers one question: is latent planning limited by the learned dynamics, or by the representation itself?_

```


## File: `reports\transformer_seed44\probe_report.md`
```md
# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.607 | 0.238 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.460 | n/a | 0.317 | 0.388 | 0.766 | encoded |
| C:next_operation | 0.381 | n/a | 0.429 | 0.378 | 0.534 | weak / not encoded |
| D:reachable_within_2 | 0.698 | n/a | 0.635 | 0.800 | 0.837 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_operands
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._

```


## File: `reports\transformer_seed44\probe_results.csv`
```csv
probe,accuracy,f1,auc,exact_accuracy
A:remaining_operands,0.6071428571428571,0.0,,0.23809523809523808
B:distance_to_solution,0.4603174603174603,0.38840319572026893,0.7657004222227092,
C:next_operation,0.38095238095238093,0.3782608695652174,0.5337080415638917,
D:reachable_within_2,0.6984126984126984,0.8,0.8369565217391304,

```


## File: `reports\transformer_seed44\transition_action_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.225070953369141,3.904700357405866
1,1,3.9939401149749756,3.327632216156506
2,2,3.3641610145568848,2.9523370461385756
3,3,2.9352095127105713,2.6222128946273053
4,4,2.5447843074798584,2.345193581502946
5,5,2.2079503536224365,2.111086235671747
6,6,1.9185941219329834,1.9401102535060195
7,7,1.7091971635818481,1.8237474785476435
8,8,1.5627844333648682,1.7487844248287012
9,9,1.4687732458114624,1.7025132726450436
10,10,1.4095598459243774,1.6738786541047643
11,11,1.364107608795166,1.6557126905097337
12,12,1.3308238983154297,1.6402505343077614
13,13,1.2963032722473145,1.6184784936123207
14,14,1.2566964626312256,1.5908248150934938
15,15,1.21449613571167,1.5609654911228867
16,16,1.1693778038024902,1.5305772374887936
17,17,1.1275590658187866,1.5031600702004355
18,18,1.0853807926177979,1.4792978255475153
19,19,1.0448524951934814,1.4597032890945185
20,20,1.0131803750991821,1.4413383358814678
21,21,0.9830880165100098,1.4235332051261527
22,22,0.9531463384628296,1.4072435722976435
23,23,0.926425039768219,1.3940938730708887
24,24,0.9018499255180359,1.3825841184522285
25,25,0.8813037276268005,1.3710794917872695
26,26,0.8572443723678589,1.359647406906378
27,27,0.8354405760765076,1.3489224793481045
28,28,0.8116056323051453,1.3411132312211833
29,29,0.7901232242584229,1.3330020591860912
30,30,0.7705013751983643,1.3244711453797386
31,31,0.7504755258560181,1.3178308205526383
32,32,0.7333704233169556,1.3129742731813525
33,33,0.7174026966094971,1.3085048237784964
34,34,0.7000213861465454,1.3028999703829405
35,35,0.6853979825973511,1.2962994184650358
36,36,0.6672467589378357,1.2926339321449154
37,37,0.6564888954162598,1.2873316280177383
38,38,0.6416612863540649,1.285828637295082
39,39,0.6298254728317261,1.2852317935130635
40,40,0.6197759509086609,1.2800157890945185
41,41,0.6066511273384094,1.2768046895011527
42,42,0.5946394801139832,1.275238537397541
43,43,0.5844243168830872,1.2735795818391393
44,44,0.574926495552063,1.2708952856845543
45,45,0.5665230751037598,1.268444248887359
46,46,0.5561756491661072,1.2720171818967725
47,47,0.5469594597816467,1.2664747394499232
48,48,0.5387592315673828,1.2648688144371159
49,49,0.5300593972206116,1.2620748301021387
50,50,0.5213826298713684,1.2597009627545466
51,51,0.5135670900344849,1.2597885131835938
52,52,0.5059224963188171,1.260301308553727
53,53,0.499165415763855,1.260699037645684
54,54,0.4916895925998688,1.2601706082703636
55,55,0.4841187596321106,1.257752965708248
56,56,0.4756825268268585,1.2566767207911758
57,57,0.4682305157184601,1.2569352447009476
58,58,0.459693044424057,1.2589457777679944
59,59,0.4541063606739044,1.2581554475377819
60,60,0.4480131268501282,1.260038782338627
61,61,0.4398064911365509,1.262601508468878
62,62,0.4350472092628479,1.2603853569656123
63,63,0.4244997501373291,1.2602602849241162
64,64,0.4189927279949188,1.262452422595415
65,65,0.4151260554790497,1.2589173864145748
66,66,0.40705686807632446,1.2581009161276895
67,67,0.4002666771411896,1.2610883869108607
68,68,0.3939395844936371,1.2572126544889857
69,69,0.38722530007362366,1.2611426681768698
70,70,0.3849438428878784,1.2650066438268444
71,71,0.3782435953617096,1.2643552686347337
72,72,0.3706531524658203,1.2646696997470543
73,73,0.36836352944374084,1.265440143522669
74,74,0.36246371269226074,1.2666597209992956
75,75,0.3561607897281647,1.265942808057441
76,76,0.35177797079086304,1.2659973394675332
77,77,0.3431830108165741,1.2691873018858864
78,78,0.33911392092704773,1.2675487330702484
79,79,0.3342861831188202,1.266981906578189
80,80,0.32952404022216797,1.2731545870421364
81,81,0.3263055980205536,1.2710713871189805
82,82,0.3191429376602173,1.2741047593413806
83,83,0.3187731206417084,1.2762692560915088
84,84,0.3109672963619232,1.2742394619300716
85,85,0.3089527487754822,1.2700539260614114
86,86,0.3017631471157074,1.2751857569960297
87,87,0.29928985238075256,1.2784874087474385
88,88,0.2951664328575134,1.276565926973937
89,89,0.290717214345932,1.2797761510630123
90,90,0.289440780878067,1.2767516589555583
91,91,0.28620028495788574,1.2793989337858607
92,92,0.27686482667922974,1.2849398753682122
93,93,0.2770996689796448,1.2806656634221312
94,94,0.2709179222583771,1.28109866282979
95,95,0.27114105224609375,1.2885259409419825
96,96,0.2679818868637085,1.2876709484663167
97,97,0.2625187039375305,1.2896030613633453
98,98,0.259847491979599,1.2874860919889857
99,99,0.2541678845882416,1.2944552312131787

```


## File: `reports\transformer_seed44\transition_blind_log.csv`
```csv
epoch,step,loss,eval_loss
0,0,4.115538120269775,4.524001324763064
1,1,4.675642490386963,3.5575486480212604
2,2,3.627753734588623,3.052396680487961
3,3,3.052551746368408,2.7601793633132683
4,4,2.6950244903564453,2.5095580054111166
5,5,2.3839704990386963,2.2822961025550716
6,6,2.1069884300231934,2.0914795672307247
7,7,1.8675458431243896,1.9546036016745645
8,8,1.696366548538208,1.8596774241963372
9,9,1.5660042762756348,1.790847653248271
10,10,1.4814200401306152,1.7445528624487705
11,11,1.422731637954712,1.7137114728083376
12,12,1.3787503242492676,1.6874552242091445
13,13,1.3401135206222534,1.6648165593381787
14,14,1.302122712135315,1.6407285596503587
15,15,1.2668803930282593,1.613392188900807
16,16,1.2283709049224854,1.587299159315766
17,17,1.1897878646850586,1.5594417384413422
18,18,1.1501507759094238,1.5341976978739753
19,19,1.1108989715576172,1.5114560987128587
20,20,1.0719947814941406,1.4929044129418545
21,21,1.043112874031067,1.4777792008196722
22,22,1.0148789882659912,1.4622337466380635
23,23,0.9907541871070862,1.4517144375160091
24,24,0.9679868221282959,1.4431912781762295
25,25,0.9460858702659607,1.4377901671362705
26,26,0.9238935112953186,1.4314952912877819
27,27,0.9050357937812805,1.4254608154296875
28,28,0.8849613070487976,1.421740547555392
29,29,0.8635497689247131,1.4183504698706455
30,30,0.8457996249198914,1.4165246682088883
31,31,0.8293903470039368,1.4126696977459017
32,32,0.8096765875816345,1.4083109370997695
33,33,0.7941526770591736,1.405241919345543
34,34,0.7762139439582825,1.4045900438652663
35,35,0.7604588270187378,1.4009867183497695
36,36,0.745280385017395,1.3972243011974899
37,37,0.7311112880706787,1.3975579934042008
38,38,0.7154219746589661,1.3968255715292008
39,39,0.7043524980545044,1.3939056396484375
40,40,0.6913546323776245,1.391068630531186
41,41,0.6806448698043823,1.3922370535428408
42,42,0.6674497723579407,1.3934984050813268
43,43,0.6566819548606873,1.3920266823690446
44,44,0.6444987654685974,1.3917981757492315
45,45,0.6345653533935547,1.3920032938972848
46,46,0.6256265044212341,1.3867102451011784
47,47,0.6155098080635071,1.3877701055808145
48,48,0.6067177653312683,1.3892041816086065
49,49,0.5964576601982117,1.387789366675205
50,50,0.588954746723175,1.3885574340820312
51,51,0.5821192860603333,1.3921693895683913
52,52,0.5753002166748047,1.394087119180648
53,53,0.5675288438796997,1.3922491855308659
54,54,0.5588333010673523,1.391450475473873
55,55,0.5521038174629211,1.38882321217021
56,56,0.5416814684867859,1.3878676617731813
57,57,0.5350568890571594,1.3850918128842213
58,58,0.5270072817802429,1.3843611420178024
59,59,0.5209569931030273,1.3865166335809427
60,60,0.5131111741065979,1.3887061447393698
61,61,0.5084193348884583,1.390563464555584
62,62,0.5034773945808411,1.394488850577933
63,63,0.49955204129219055,1.3920924702628714
64,64,0.49049878120422363,1.385512180015689
65,65,0.4817928373813629,1.3882069822217598
66,66,0.47754886746406555,1.3873075891713627
67,67,0.46891096234321594,1.3876976888687884
68,68,0.46282345056533813,1.3904544017353997
69,69,0.4564758539199829,1.38996949743052
70,70,0.45136797428131104,1.3900226530481556
71,71,0.44417881965637207,1.393172342269147
72,72,0.4388670325279236,1.3949334816854508
73,73,0.4326658844947815,1.396164065501729
74,74,0.42668160796165466,1.39797848560771
75,75,0.4225473701953888,1.3995260019771387
76,76,0.415662556886673,1.3987459276543288
77,77,0.4098511040210724,1.4032074975185707
78,78,0.41089239716529846,1.4117341588755123
79,79,0.4093616008758545,1.4177040975601947
80,80,0.4104931354522705,1.4209227014760502
81,81,0.4080459177494049,1.4259385906282018
82,82,0.4061357080936432,1.40862786965292
83,83,0.38774996995925903,1.4073456310835042
84,84,0.3810197412967682,1.4151239864161758
85,85,0.380737841129303,1.4136386308513704
86,86,0.37579911947250366,1.4089989584000384
87,87,0.36779889464378357,1.4107347081919186
88,88,0.3615928888320923,1.410948956599001
89,89,0.3617207109928131,1.4115939531169954
90,90,0.3565625548362732,1.4130595472992444
91,91,0.34889882802963257,1.4143023881755892
92,92,0.34732571244239807,1.417357397861168
93,93,0.345991849899292,1.4168972578205046
94,94,0.33837854862213135,1.421311300308978
95,95,0.33331766724586487,1.4180800641169313
96,96,0.33044368028640747,1.4170203287093366
97,97,0.3258870244026184,1.4238817809058018
98,98,0.32405754923820496,1.4247812990282402
99,99,0.32137829065322876,1.4197247614626025

```

