# Appendix

## A. Hyperparameters
[Placeholder]

## B. Game of 24 Leak-Free Benchmark
The classic Game of 24 involves combining 4 numbers drawn from 1-13. Because random sampling with replacement guarantees massive train/test overlap, we exhaustively enumerated the pool.

*   Total unique 4-card multisets: 1,820
*   Solvable (integer-only intermediate steps): 1,346 (74.0%)

Our benchmark partitions this finite 1,346-hand pool into strictly mutually exclusive train, validation, and test splits (80/10/10 ratio), ensuring zero evaluation leakage.

## C. Positive Control: FSM Construction
[Placeholder for details on the synthetic state machine used to validate the pipeline.]
