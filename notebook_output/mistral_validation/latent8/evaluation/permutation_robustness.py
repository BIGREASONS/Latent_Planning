"""Permutation robustness analysis (V5 Component 7).

Tests whether **equivalent reasoning states map to the same discrete code**.

For each problem we have ``K`` distinct valid solutions; encoding each one as
its own hidden-state trajectory yields ``K`` discrete code trajectories through
the same problem. Two states from *different* solutions are considered
"equivalent" iff they share the same *symbolic state* (target + sorted
available numbers, via :func:`evaluation.intrinsic_noise.get_symbolic_states`):
they represent the same logical situation reached by a different reasoning
order.

Metric
------
**Code consistency** = fraction of equivalent cross-solution state *pairs* that
received the same code. We compare against:

* a **random-baseline** consistency = ``1 / num_codes`` (what you'd get by
  assigning codes uniformly at random), and
* a **within-solution** consistency floor (states from the *same* solution at
  the same symbolic content — these are trivially identical by construction,
  so they are excluded from the cross-solution metric but reported as a sanity
  ceiling).

High cross-solution consistency (well above the random baseline) is evidence
that the codebook captures reusable *state content*, not the surface form of
a particular reasoning trace.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_processing.discrete_trajectory_dataset import DiscreteTrajectory
from data_processing.trajectory_dataset import (
    Trajectory,
    HEADER_TEMPLATE,
    _state_end_chars,
    _token_index_for_char_end,
)


def build_multi_solution_trajectories(
    model,
    tokenizer,
    problem: Dict,
    layer: int = -1,
) -> List[Trajectory]:
    """Build one :class:`Trajectory` per solution in a multi-solution problem.

    Reuses the alignment helpers (``_state_end_chars`` /
    ``_token_index_for_char_end``) so the resulting trajectories are
    byte-compatible with the single-solution :func:`build_trajectory`. Requires
    a fast tokenizer.
    """
    from data_processing.action_parser import parse_solution

    numbers = problem["numbers"]
    target = problem["target"]
    solutions = problem["solutions"]
    header = HEADER_TEMPLATE.format(numbers=numbers, target=target)

    out: List[Trajectory] = []
    for steps in solutions:
        if not steps:
            continue
        try:
            actions = parse_solution(steps)
        except ValueError:
            continue
        full_text = header + "\n".join(steps)
        enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0].detach().cpu().float()
        input_ids = enc["input_ids"][0].detach().cpu()
        end_chars = _state_end_chars(header, steps)
        try:
            state_indices = [
                _token_index_for_char_end(offsets, ec) for ec in end_chars
            ]
        except ValueError:
            continue
        state_indices = torch.tensor(state_indices, dtype=torch.long)
        op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
        operands = torch.tensor(
            [[a.arg1, a.arg2] for a in actions], dtype=torch.float32
        )
        out.append(
            Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=op_ids,
                operands=operands,
                numbers=list(numbers),
                target=int(target),
            )
        )
    return out


def evaluate_permutation_robustness(
    discrete_groups: List[List[DiscreteTrajectory]],
    num_codes: int,
    seed: int = 0,
) -> Dict:
    """Compute cross-solution code consistency for equivalent symbolic states.

    Args:
        discrete_groups: one list of :class:`DiscreteTrajectory` per problem;
            each inner list has one trajectory per solution of that problem.
            Symbolic states are recomputed per trajectory via
            :func:`get_symbolic_states` to decide equivalence across solutions.
        num_codes: codebook cardinality (for the random baseline).
        seed: RNG seed for the random-assignment baseline.

    Returns a dict with ``cross_consistency``, ``within_consistency`` (sanity
    ceiling), ``random_baseline`` (``1/num_codes``), ``n_pairs_cross``,
    ``n_problems_with_match``, and ``mean_solutions_per_problem``.
    """
    from evaluation.intrinsic_noise import get_symbolic_states

    # Map: (problem_idx, symbolic_state) -> list[(solution_idx, code)]
    by_symbol: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)

    n_solutions_total = 0
    for p_idx, sol_disc_trajs in enumerate(discrete_groups):
        for s_idx, disc in enumerate(sol_disc_trajs):
            n_solutions_total += 1
            sym_states = get_symbolic_states(
                _disc_to_trajectory(disc)
            )
            for d, info in enumerate(sym_states):
                if info is None or d >= disc.codes.shape[0]:
                    continue
                sym, _hist = info
                by_symbol[(p_idx, sym)].append((s_idx, int(disc.codes[d].item())))

    cross_match, cross_total = 0, 0
    within_match, within_total = 0, 0
    problems_with_match = set()
    for (p_idx, _sym), items in by_symbol.items():
        n = len(items)
        if n < 2:
            continue
        problems_with_match.add(p_idx)
        for i in range(n):
            for j in range(i + 1, n):
                sol_i, code_i = items[i]
                sol_j, code_j = items[j]
                same = 1 if code_i == code_j else 0
                if sol_i == sol_j:
                    within_match += same
                    within_total += 1
                else:
                    cross_match += same
                    cross_total += 1

    # Random baseline: P(two random codes agree) = 1/num_codes under uniform
    # assignment, but the empirical code distribution is usually skewed, so we
    # also report the data-driven baseline sum_k p_k^2.
    rng = np.random.RandomState(seed)
    # Aggregate all codes seen across the matched groups for the empirical null.
    all_codes_seen = [c for items in by_symbol.values() for (_, c) in items]
    if all_codes_seen:
        counts = np.bincount(np.asarray(all_codes_seen), minlength=num_codes).astype(np.float64)
        p = counts / counts.sum()
        empirical_baseline = float((p * p).sum())
    else:
        empirical_baseline = 0.0
    uniform_baseline = 1.0 / num_codes

    n_problems = len(discrete_groups)
    return {
        "cross_consistency": (cross_match / cross_total) if cross_total else float("nan"),
        "within_consistency": (within_match / within_total) if within_total else float("nan"),
        "uniform_random_baseline": uniform_baseline,
        "empirical_random_baseline": empirical_baseline,
        "n_pairs_cross": cross_total,
        "n_problems_with_match": len(problems_with_match),
        "n_problems_total": n_problems,
        "mean_solutions_per_problem": (n_solutions_total / n_problems) if n_problems else 0.0,
    }


def _disc_to_trajectory(disc: DiscreteTrajectory):
    """Lightweight stand-in carrying only the fields ``get_symbolic_states`` reads.

    ``get_symbolic_states`` only touches ``target``, ``numbers``, ``op_ids``,
    ``operands`` and ``num_steps`` — so we synthesize a minimal object instead
    of round-tripping through the continuous ``Trajectory``. ``num_steps`` is a
    plain int attribute here (the consumer reads it as ``traj.num_steps``,
    which is valid whether it is a property or an attribute).
    """

    class _Stub:
        pass

    s = _Stub()
    s.target = disc.target
    s.numbers = disc.numbers
    s.op_ids = disc.op_ids
    s.operands = disc.operands
    s.num_steps = int(disc.op_ids.shape[0])
    return s


def save_permutation_robustness_report(metrics: Dict, csv_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
