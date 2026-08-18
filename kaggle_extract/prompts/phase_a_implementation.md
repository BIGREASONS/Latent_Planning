You are the lead research engineer on this project.

The repository infrastructure already exists and passes tests.

Your job is to implement ALL research-critical components required to determine whether latent planning is viable.

Do NOT implement MCTS yet.

Do NOT implement VQ-VAE yet.

Do NOT implement policy search yet.

The goal is to build the complete diagnostic framework that determines whether latent planning is worth pursuing.

====================================================
PROJECT GOAL
============

Research Question:

Can frozen LM hidden states support multi-step latent planning?

We want to determine:

1. Does the hidden state contain task-relevant information?
2. Can a learned transition model preserve that information?
3. What is the effective coherence horizon?
4. Is the bottleneck representation or dynamics?

====================================================
EXISTING INFRASTRUCTURE
=======================

Already available:

* TinyLlama loading
* QLoRA training
* Countdown dataset generation
* Hidden-state extraction
* Plotting utilities
* Training pipeline
* Tests

Use existing infrastructure.

====================================================
PART 1
SYMBOLIC ACTION REPRESENTATION
==============================

Implement a symbolic Countdown action parser.

Convert reasoning steps:

"75 * 11 = 825"

into:

{
op: MUL,
arg1: 75,
arg2: 11,
result: 825
}

Support:

ADD
SUB
MUL
DIV

Create:

datasets/action_parser.py

Include tests.

====================================================
PART 2
TEACHER TRAJECTORY DATASET
==========================

Create a dataset builder that constructs:

(h_t, action_t, h_t+1)

from extracted hidden states.

Support:

* train
* validation
* test

Create:

datasets/trajectory_dataset.py

====================================================
PART 3
TRANSITION MODEL
================

Implement:

T(h_t, a_t) -> h_t+1

Architecture:

Action Embedding

Concatenate:

[h_t ; action_embedding]

Network:

Linear -> ReLU -> Linear

Configurable hidden dimension.

Loss:

MSE

Create:

models/transition_model.py

training/train_transition.py

====================================================
PART 4
DIAGNOSTIC DECODER
==================

Implement a lightweight decoder:

hidden_state -> next reasoning token

Purpose:

Diagnostic only.

Not used for planning.

Train on teacher hidden states.

Create:

models/diagnostic_decoder.py

training/train_decoder.py

====================================================
PART 5
LATENT COHERENCE EVALUATION
===========================

Implement rollout evaluation.

Procedure:

Start from teacher hidden state.

Apply ground-truth symbolic actions.

Depth:

1..8

Repeatedly apply transition model.

Measure:

1. Hidden-state MSE
2. Cosine similarity
3. Decoder token accuracy

Output:

coherence_depth.csv

Columns:

depth
mse
cosine_similarity
token_accuracy

Generate:

coherence_depth.png

====================================================
PART 6
REPRESENTATION PROBES
=====================

Implement linear probes.

Probe A:

Predict remaining numbers.

Probe B:

Predict distance-to-solution.

Probe C:

Predict next symbolic operation.

Probe D:

Predict whether solution is reachable within 2 steps.

Use:

Linear layer only.

No deep network.

Create:

evaluation/probes.py

====================================================
PART 7
PROBE EVALUATION
================

Generate:

probe_results.csv

Columns:

probe
accuracy
f1
auc

Generate:

probe_report.md

Explain:

* what information is encoded
* what is not encoded

====================================================
PART 8
MASTER REPORT
=============

Generate:

reports/phase_a_report.md

Must answer:

1. Do hidden states contain reasoning information?
2. How quickly does coherence decay?
3. What is the effective planning horizon?
4. Is latent planning worth pursuing?

====================================================
DO NOT IMPLEMENT
================

* MCTS
* Retrieval baseline
* Oracle transitions
* VQ-VAE
* Policy networks
* Value networks
* Search

====================================================
SUCCESS CRITERIA
================

When complete I should have:

* action parser
* trajectory dataset
* transition model
* diagnostic decoder
* probes
* coherence_depth.csv
* coherence_depth.png
* probe_results.csv
* phase_a_report.md

The objective is to determine whether latent planning survives beyond depth 3.

Prioritize correctness, reproducibility, tests, and scientific rigor over feature count.
