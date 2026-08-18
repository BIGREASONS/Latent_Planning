# Representation Probe Report

Linear probes fit on **14968** teacher states, evaluated on **2903** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_numbers | 0.981 | 0.933 | 0.444 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.531 | n/a | 0.334 | 0.511 | 0.809 | encoded |
| C:next_operation | 0.532 | n/a | 0.337 | 0.531 | 0.717 | encoded |
| D:reachable_within_2 | 0.762 | n/a | 0.668 | 0.815 | 0.841 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_numbers
- B:distance_to_solution
- C:next_operation
- D:reachable_within_2


_Note: probe A (remaining numbers) is restricted to the fixed set {25,50,75,100} so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._
