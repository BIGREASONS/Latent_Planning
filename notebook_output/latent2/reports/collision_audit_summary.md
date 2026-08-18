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
