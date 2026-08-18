# Representation Probe Report

Linear probes fit on **52** teacher states, evaluated on **63** held-out states.

Each probe is a logistic regression on standardized hidden states (linear only). A score well above the majority-class baseline means the information is *linearly decodable* from the frozen representation.


## Results

| Probe | Accuracy | Exact Match | Chance | F1 | AUC | Verdict |
|---|---|---|---|---|---|---|
| A:remaining_operands | 0.761 | 0.016 | 0.829 | 0.000 | n/a | weak / not encoded |
| B:distance_to_solution | 0.429 | n/a | 0.317 | 0.373 | 0.747 | encoded |
| C:next_operation | 0.397 | n/a | 0.429 | 0.389 | 0.560 | weak / not encoded |
| D:reachable_within_2 | 0.683 | n/a | 0.635 | 0.792 | 0.835 | encoded |

## Interpretation

**Linearly encoded in the hidden state:**
- B:distance_to_solution
- D:reachable_within_2

**Weak or not linearly encoded:**
- A:remaining_operands
- C:next_operation

_Note: probe A (remaining numbers) evaluates the fixed domain vocabulary so each label has a consistent meaning across problems. AUC is reported as macro one-vs-rest for multiclass probes and averaged over labels for the multi-label probe; 'n/a' indicates a degenerate or missing class in the evaluation split._
