"""Collision Statistics Audit.

Diagnostic answering a single question: is the Oracle 2A coverage collapse
(4/3004 states with a swap partner) caused by a genuine *lack of state
collisions* in the dataset, or by an implementation issue downstream?

This script trains nothing, runs no probes, no Oracle, no Phase A. It only reads
the existing trajectory artifacts and counts how often each symbolic state recurs
across trajectories.

Symbolic-state extraction is reused verbatim from the canonical implementation
``evaluation.intrinsic_noise.get_symbolic_states`` — the same function used by
the Oracle Audit, ``intrinsic_noise.py`` and Phase C. No symbolic-state logic is
duplicated here.

Definitions (matching the Oracle Audit exactly)
------------------------------------------------
``get_symbolic_states`` returns, per aligned state, a tuple
``(symbolic_state, action_history)`` where:

* ``symbolic_state = (target, tuple(sorted(available_numbers)))`` — the *target
  is already embedded* in the symbolic state, so grouping by ``symbolic_state``
  is identical to grouping by ``(symbolic_state, target)``.
* ``action_history`` is the tuple of ``(op_id, arg1, arg2)`` applied so far.

A valid Oracle 2A swap pair (the audit's coverage requirement) is two occurrences
of the *same symbolic state* drawn from *different trajectories* with *different
action histories*.

Inputs
------
    reports/trajectories/train.pt
    reports/trajectories/test.pt

Outputs
-------
    reports/collision_audit.csv          (one row per unique symbolic state)
    reports/collision_audit_summary.md   (global stats + distributions + coverage)
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.trajectory_dataset import load_trajectories
from evaluation.intrinsic_noise import get_symbolic_states


# --------------------------------------------------------------------------- #
# Occurrence collection
# --------------------------------------------------------------------------- #
def collect_occurrences(trajs, split, occ_by_state):
    """Append every aligned state's occurrence into ``occ_by_state``.

    ``occ_by_state`` maps a symbolic state -> list of occurrence dicts. Each
    occurrence records the originating trajectory id, depth, action-history
    length, history (for diversity counting) and split. States invalidated by a
    teacher arithmetic mistake (``get_symbolic_states`` returns ``None``) are
    skipped, exactly as the Oracle Audit skips them.

    ``traj_id`` is namespaced by split (``"train:0"``, ``"test:0"``, ...) so that
    train and test trajectories are never conflated as the "same" trajectory.
    """
    for ti, traj in enumerate(trajs):
        sym_list = get_symbolic_states(traj)  # len N+1; (sym, hist) or None
        traj_id = f"{split}:{ti}"
        for depth, info in enumerate(sym_list):
            if info is None:
                continue
            sym, hist = info
            occ_by_state[sym].append(
                {
                    "traj_id": traj_id,
                    "depth": depth,
                    "action_history_len": len(hist),
                    "history": hist,
                    "split": split,
                }
            )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def compute_statistics(occ_by_state):
    """Return (per_state_rows, summary) computed from grouped occurrences."""
    per_state_rows = []
    occ_counts = []         # occurrences per symbolic state
    unique_hist_counts = []  # unique histories per symbolic state

    # Accumulators for anchor-level (occurrence-level) coverage.
    anchors_with_partner_a = 0
    anchors_with_partner_b = 0

    for sym, occs in occ_by_state.items():
        target, numbers = sym
        n_occ = len(occs)

        unique_histories = {o["history"] for o in occs}
        unique_trajs = {o["traj_id"] for o in occs}
        n_unique_hist = len(unique_histories)
        n_unique_traj = len(unique_trajs)

        # Oracle 2A swap candidates *within this symbolic state*, computed in a
        # single O(n^2) pass per state. The dataset's collisions are sparse, so
        # n is tiny for almost every state.
        #   (A) same target + same symbolic state + different trajectory.
        #   (B) (A) AND different action history (the exact Oracle 2A partner rule).
        # cand_* count ordered anchor->partner pairs (comparable to the Oracle
        # Audit's per-anchor partner search). anchors_with_partner_* count
        # occurrences that have >=1 partner (the audit's coverage numerator).
        cand_a = 0
        cand_b = 0
        for i in range(n_occ):
            oi = occs[i]
            has_a = False
            has_b = False
            for j in range(n_occ):
                if i == j:
                    continue
                oj = occs[j]
                if oi["traj_id"] == oj["traj_id"]:
                    continue
                cand_a += 1
                has_a = True
                if oi["history"] != oj["history"]:
                    cand_b += 1
                    has_b = True
            if has_a:
                anchors_with_partner_a += 1
            if has_b:
                anchors_with_partner_b += 1

        split_set = sorted({o["split"] for o in occs})

        per_state_rows.append(
            {
                "target": target,
                "numbers": "|".join(str(x) for x in numbers),
                "num_available": len(numbers),
                "occurrences": n_occ,
                "unique_trajectories": n_unique_traj,
                "unique_histories": n_unique_hist,
                "swap_candidates_A": cand_a,
                "swap_candidates_B": cand_b,
                "splits": "|".join(split_set),
            }
        )
        occ_counts.append(n_occ)
        unique_hist_counts.append(n_unique_hist)

    occ_counts_sorted = sorted(occ_counts)
    total_states = sum(occ_counts)            # total aligned state instances
    unique_states = len(occ_by_state)         # distinct symbolic states

    def _frac(cond_count):
        return (cond_count / unique_states) if unique_states else 0.0

    ge2 = sum(1 for c in occ_counts if c >= 2)
    ge5 = sum(1 for c in occ_counts if c >= 5)
    ge10 = sum(1 for c in occ_counts if c >= 10)
    ge20 = sum(1 for c in occ_counts if c >= 20)

    hist_ge2 = sum(1 for c in unique_hist_counts if c >= 2)
    hist_ge5 = sum(1 for c in unique_hist_counts if c >= 5)
    hist_ge10 = sum(1 for c in unique_hist_counts if c >= 10)

    # Oracle coverage estimate: how many *symbolic states* admit >=1 valid swap
    # candidate, and how many anchors (occurrences) have >=1 partner.
    states_with_cand_a = sum(1 for r in per_state_rows if r["swap_candidates_A"] > 0)
    states_with_cand_b = sum(1 for r in per_state_rows if r["swap_candidates_B"] > 0)
    total_cand_a = sum(r["swap_candidates_A"] for r in per_state_rows)
    total_cand_b = sum(r["swap_candidates_B"] for r in per_state_rows)

    summary = {
        "total_states": total_states,
        "unique_symbolic_states": unique_states,
        "avg_occurrences_per_state": (total_states / unique_states) if unique_states else 0.0,
        "median_occurrences": _median(occ_counts_sorted),
        "max_occurrences": max(occ_counts) if occ_counts else 0,
        "states_ge2_occ": ge2,
        "states_ge5_occ": ge5,
        "states_ge10_occ": ge10,
        "states_ge20_occ": ge20,
        "frac_ge2_occ": _frac(ge2),
        "frac_ge5_occ": _frac(ge5),
        "frac_ge10_occ": _frac(ge10),
        "frac_ge20_occ": _frac(ge20),
        "states_ge2_unique_hist": hist_ge2,
        "states_ge5_unique_hist": hist_ge5,
        "states_ge10_unique_hist": hist_ge10,
        "states_with_swap_candidates_A": states_with_cand_a,
        "states_with_swap_candidates_B": states_with_cand_b,
        "total_swap_candidates_A": total_cand_a,
        "total_swap_candidates_B": total_cand_b,
        "anchors_with_partner_A": anchors_with_partner_a,
        "anchors_with_partner_B": anchors_with_partner_b,
        "anchor_coverage_rate_A": _safe_div(anchors_with_partner_a, total_states),
        "anchor_coverage_rate_B": _safe_div(anchors_with_partner_b, total_states),
    }
    return per_state_rows, summary


def _safe_div(a, b):
    return (a / b) if b else 0.0


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "target",
    "numbers",
    "num_available",
    "occurrences",
    "unique_trajectories",
    "unique_histories",
    "swap_candidates_A",
    "swap_candidates_B",
    "splits",
]


def write_csv(per_state_rows, path):
    # Most-collided states first for easy inspection.
    rows = sorted(per_state_rows, key=lambda r: (-r["occurrences"], -r["unique_histories"]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_summary_md(summary, split_counts, path):
    s = summary
    lines = ["# Collision Statistics Audit — Summary\n"]

    lines.append(
        "Reuses `evaluation.intrinsic_noise.get_symbolic_states` (the same symbolic-"
        "state extraction used by the Oracle Audit, intrinsic_noise.py and Phase C). "
        "Grouping key is the symbolic state `(target, sorted(available_numbers))`; "
        "the target is already embedded in that key.\n"
    )

    lines.append("## Inputs\n")
    lines.append("| split | trajectories |")
    lines.append("|---|---|")
    for split, n in split_counts.items():
        lines.append(f"| {split} | {n} |")
    lines.append("")

    lines.append("## Global\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in [
        "total_states",
        "unique_symbolic_states",
        "avg_occurrences_per_state",
        "median_occurrences",
        "max_occurrences",
    ]:
        lines.append(f"| {k} | {_fmt(s[k])} |")
    lines.append("")

    lines.append("## Collision distribution\n")
    lines.append("| threshold | states | fraction of unique states |")
    lines.append("|---|---|---|")
    lines.append(f"| >= 2 occurrences | {s['states_ge2_occ']} | {_fmt(s['frac_ge2_occ'])} |")
    lines.append(f"| >= 5 occurrences | {s['states_ge5_occ']} | {_fmt(s['frac_ge5_occ'])} |")
    lines.append(f"| >= 10 occurrences | {s['states_ge10_occ']} | {_fmt(s['frac_ge10_occ'])} |")
    lines.append(f"| >= 20 occurrences | {s['states_ge20_occ']} | {_fmt(s['frac_ge20_occ'])} |")
    lines.append("")

    lines.append("## History diversity\n")
    lines.append("Unique action histories reaching each symbolic state.\n")
    lines.append("| threshold | states |")
    lines.append("|---|---|")
    lines.append(f"| >= 2 unique histories | {s['states_ge2_unique_hist']} |")
    lines.append(f"| >= 5 unique histories | {s['states_ge5_unique_hist']} |")
    lines.append(f"| >= 10 unique histories | {s['states_ge10_unique_hist']} |")
    lines.append("")

    lines.append("## Oracle Coverage Estimate\n")
    lines.append(
        "Valid Oracle 2A swap candidates. **A** = same target + same symbolic "
        "state + different trajectory. **B** = A *and* different action history "
        "(this is the exact Oracle 2A partner rule). Candidate counts are ordered "
        "anchor->partner pairs.\n"
    )
    lines.append("| metric | A (diff trajectory) | B (diff trajectory + diff history) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| symbolic states with >=1 candidate | {s['states_with_swap_candidates_A']} "
        f"| {s['states_with_swap_candidates_B']} |"
    )
    lines.append(
        f"| total swap candidate pairs | {s['total_swap_candidates_A']} "
        f"| {s['total_swap_candidates_B']} |"
    )
    lines.append(
        f"| anchors (state instances) with >=1 partner | {s['anchors_with_partner_A']} "
        f"| {s['anchors_with_partner_B']} |"
    )
    lines.append(
        f"| anchor coverage rate | {_fmt(s['anchor_coverage_rate_A'])} "
        f"| {_fmt(s['anchor_coverage_rate_B'])} |"
    )
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Collision Statistics Audit")
    ap.add_argument("--reports_dir", default="reports",
                    help="Holds trajectories/ and receives outputs.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default <reports_dir>).")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    help="Trajectory splits to include (default train test).")
    ap.add_argument("--sample", type=int, default=0,
                    help="Lightweight validation: cap trajectories loaded PER SPLIT "
                         "to this many (0 = use all). Outputs are suffixed _sample.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.reports_dir
    traj_dir = os.path.join(args.reports_dir, "trajectories")

    occ_by_state = collections.defaultdict(list)
    split_counts = {}
    for split in args.splits:
        path = os.path.join(traj_dir, f"{split}.pt")
        if not os.path.exists(path):
            print(f"ERROR: trajectories not found: {path}")
            sys.exit(1)
        print(f"[Collision Audit] Loading {path}")
        trajs = load_trajectories(path)
        if args.sample > 0:
            trajs = trajs[: args.sample]
        print(f"  {len(trajs)} trajectories ({split})")
        split_counts[split] = len(trajs)
        collect_occurrences(trajs, split, occ_by_state)

    print("[Collision Audit] Computing statistics...")
    per_state_rows, summary = compute_statistics(occ_by_state)

    suffix = "_sample" if args.sample > 0 else ""
    csv_path = os.path.join(out_dir, f"collision_audit{suffix}.csv")
    md_path = os.path.join(out_dir, f"collision_audit_summary{suffix}.md")

    write_csv(per_state_rows, csv_path)
    write_summary_md(summary, split_counts, md_path)

    print("\n[Collision Audit] Done.")
    print(f"  total state instances     : {summary['total_states']}")
    print(f"  unique symbolic states     : {summary['unique_symbolic_states']}")
    print(f"  states >=2 occ             : {summary['states_ge2_occ']}")
    print(f"  states with swap cand (B)  : {summary['states_with_swap_candidates_B']}")
    print(f"  anchor coverage rate (B)   : {summary['anchor_coverage_rate_B']:.6f}")
    for p in (csv_path, md_path):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
