# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_numbers | 0.607 | 0.190 | 0.456 | 0.000 | n/a | encoded |
| B:distance_to_solution | 0.413 | n/a | 0.317 | 0.351 | 0.738 | encoded |
| C:next_operation | 0.365 | n/a | 0.429 | 0.353 | 0.535 | weak / not encoded |
| D:reachable_within_2 | 0.667 | n/a | 0.635 | 0.769 | 0.786 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- A:remaining_numbers
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- C:next_operation

_Note: probe A (remaining numbers) is restricted to the fixed set {25,50,75,100} so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._
