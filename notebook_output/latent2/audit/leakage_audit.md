# Phase 5 — Data Leakage Audit

Ordered by severity. Each is traced to code.

## L1 — Phase C.3 fits PCA + scaler on the FULL set, then splits — CONFIRMED, HIGH

`scripts/run_phase_c3_geometry.py`:
```
147  scaler = StandardScaler().fit(X)          # X = ALL states
148  X_scaled = scaler.transform(X)
151  pca = PCA(n_components=100)
152  X_pca = pca.fit_transform(X_scaled)        # PCA sees test rows
157  idx_tr, idx_te = train_test_split(indices, test_size=0.2, ...)   # split AFTER
```
The scaler statistics and principal axes are computed using the test rows, then probes are trained/evaluated on slices of that leaked basis. The "Depth is CONCENTRATED/DISTRIBUTED across PCs" verdict (`:207`) is computed on contaminated features. **This directly contradicts the sibling script** `run_phase_c4a_pca_ablation.py:225-232`, which correctly fits scaler+PCA on **train only**. So the repo demonstrably knows the right way and got it wrong in C.3. Any C.3 number is invalid as stated.

## L2 — Phase D trains the projection on train+test, then evaluates on the same states — CONFIRMED, HIGH

`scripts/run_phase_d.py`:
```
220  all_trajs = train_trajs + test_trajs
223  state_groups = gather_state_groups(all_trajs)      # train+test combined
226  pos/neg = build_contrastive_dataset(state_groups)  # pairs include test states
234  model = train_projection(...)                       # P(h) trained on test states
...
240  df_h, top1_h ... = compute_intrinsic_noise(state_groups)        # eval on trained states
241  df_z, top1_z ... = compute_intrinsic_noise(z_groups)            # eval on trained states
257  probe_res = evaluate_probes(model, train_trajs, test_trajs)     # test X pushed through P trained on test
```
The canonicalization projection `P(h)` is trained on contrastive pairs drawn from the **combined** pool, then "retrieval improvement h→z" and "probes maintained" are measured on those same states. The headline verdict `ENTANGLED` vs `ABSENT` (`:289`) is a train-on-test result: the projection is rewarded for memorizing the exact states it will be scored on. Top-1(z) is inflated relative to any held-out measurement. **Invalid as a generalization claim.**

## L3 — Deterministic causal LM ⇒ shared prefixes ⇒ identical hidden states — CONFIRMED, HIGH (confound that masquerades as signal)

The teacher is a **frozen, deterministic** LM. Hidden state `h_t` depends only on tokens `≤ t` (causal attention). Two different problems that share the prompt header **and the first k reasoning lines** therefore have **byte-identical** hidden states at every position up to step k.

The Countdown generator draws from a tiny pool (`generate_countdown_dataset.py:15-25`: 4 large + duplicated 1–10), so prefix collisions are common. Consequences:
- **Phase C "within-state" similarity** (`intrinsic_noise.py`): pairs reaching the same symbolic state often do so via an identical text prefix → cosine = 1.0 *trivially*, not because the representation is Markovian. The `same_hist_mask` (`:209-216`) only removes pairs with **identical full action histories**, not shared **prefixes**, so prefix-identical pairs survive and inflate within-state similarity and Top-1 retrieval.
- **Probe train/test**: Phase A dedups only at `(target, sorted(numbers))` problem level (`run_phase_a.py:102-132`). Two *distinct* problems sharing a prefix still leak identical sub-state vectors across the train/test boundary.

This is not a coding bug — it's a measurement confound that makes "the representation encodes the symbolic state" partly tautological.

## L4 — Phase A dedup is necessary but insufficient — CONFIRMED, MEDIUM

`run_phase_a.py:101-132` removes val/test trajectories whose `(target, sorted(numbers))` appears in train. Good that it exists. Gaps:
- Hash ignores the **solution path**; two entries with same numbers/target but different op sequences are treated as duplicates and dropped, possibly discarding legitimate held-out variety.
- It dedups trajectories but the **probe feature rows** are per-sub-state; sub-state overlap (L3) is unaddressed.
- `val_hashes` is defined only inside `if "val" in trajs:` (`:111`) yet read at `:119`/`:127`. Since `splits` always contains `"val"`, it's currently safe, but it's a latent `NameError` if the split list ever changes.

## L5 — Phase C / C4a retrieval database overlaps the query set — CONFIRMED, MEDIUM

- `run_phase_c.py:110` builds `state_groups` from `train+test` combined and computes retrieval over the union (`compute_intrinsic_noise`), masking only same-trajectory/same-history — so a test query can retrieve a train neighbor of the *same problem* (allowed), inflating Top-1.
- `run_phase_c4a_pca_ablation.py` is better (query=test slice, DB=train slice, PCA on train), but it still measures retrieval into a database that, by L3, contains prefix-identical vectors.

## L6 — Probe label B/C/D leak position, not reasoning — see metric_inflation_audit §M3

Distance-to-solution (B) and reachable-in-2 (D) are deterministic functions of token position; the generator fixes lengths (IID 2–4, OOD=5). "Decoding distance" ≈ "decoding position," which the frozen LM encodes trivially. Counted as a leakage-adjacent confound.

## Verdict

Two **confirmed train-on-test leaks** (C.3 PCA, Phase D projection) invalidate those phases' headline verdicts as written. A deeper, structural confound (L3: deterministic-prefix identity) inflates every within-state/retrieval/Markovianity number in Phase C across the board. Phase A's dedup is real but operates at the wrong granularity. **Proven: leakage exists.**
