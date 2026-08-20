"""Deterministic cross-domain reasoning-trace generators for V5 transfer.

Three small CoT datasets, each emitted as JSONL with a ``problem`` (the prompt
header) and ``solution`` (a list of newline-joined step strings). They share
the *shape* of the Countdown data (a header followed by one-line steps) so the
generic trajectory extractor (:mod:`data_processing.transfer_trajectory`) can
treat them identically — only the symbolic content differs.

* **Algebra** — linear ``solve for x``: each step isolates a sub-expression.
* **Logic** — propositional modus-ponens chains over a few facts.
* **Graph** — Dijkstra-style shortest-path reasoning over a small weighted
  graph.

All generators are seeded and rejection-sampled so outputs are reproducible
and every step is internally valid. They are deliberately small (default
~2000 problems each) — enough to measure transfer statistics, not to train a
domain expert.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

DOMAINS = ["algebra", "logic", "graph"]


# --------------------------------------------------------------------------- #
# Algebra: solve a*x + b = c for x. CoT isolates the unknown step by step.
# --------------------------------------------------------------------------- #
def _algebra_problem(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(2, 9)
    x = rng.randint(-9, 9)
    b = rng.randint(-20, 20)
    c = a * x + b
    steps = [
        f"Equation: {a} * x + {b} = {c}",
        f"Subtract {b}: {a} * x = {c - b}",
        f"Divide by {a}: x = {(c - b) // a}",
    ]
    header = f"Problem: Solve for x.\nSolution:\n"
    return {
        "domain": "algebra",
        "problem": header,
        "numbers": [a, b, c],
        "target": (c - b) // a,
        "solution": steps,
        "cot": "\n".join(steps),
    }


# --------------------------------------------------------------------------- #
# Logic: chain of modus ponens. If P then Q; P; therefore Q (extended).
# --------------------------------------------------------------------------- #
def _logic_problem(rng: random.Random) -> Dict[str, Any]:
    atoms = ["A", "B", "C", "D", "E", "F"]
    n = rng.randint(3, 5)
    chain = rng.sample(atoms, n)
    steps = []
    for i in range(n - 1):
        steps.append(f"If {chain[i]} then {chain[i + 1]}.")
    steps.append(f"{chain[0]} is true.")
    for i in range(n - 1):
        steps.append(f"Therefore {chain[i + 1]}.")
    header = "Problem: Determine the conclusion.\nSolution:\n"
    return {
        "domain": "logic",
        "problem": header,
        "numbers": [n],
        "target": n - 1,
        "solution": steps,
        "cot": "\n".join(steps),
    }


# --------------------------------------------------------------------------- #
# Graph: shortest path on a small weighted graph (Dijkstra-style narrative).
# --------------------------------------------------------------------------- #
def _graph_problem(rng: random.Random) -> Dict[str, Any]:
    import heapq

    n_nodes = rng.randint(4, 6)
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    # Random connected graph via a spanning tree, plus a few extra edges.
    adj = {v: {} for v in nodes}
    for i in range(1, n_nodes):
        j = rng.randint(0, i - 1)
        w = rng.randint(1, 9)
        adj[nodes[i]][nodes[j]] = w
        adj[nodes[j]][nodes[i]] = w
    for _ in range(rng.randint(1, 3)):
        u, v = rng.sample(nodes, 2)
        if v not in adj[u]:
            w = rng.randint(1, 9)
            adj[u][v] = w
            adj[v][u] = w

    src, dst = rng.sample(nodes, 2)
    # Dijkstra from src.
    dist = {v: float("inf") for v in nodes}
    dist[src] = 0
    prev = {src: None}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path + steps.
    if dist[dst] == float("inf"):
        return _graph_problem(rng)  # connected graph guarantees this is rare
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    steps = [
        f"Graph edges: "
        + ", ".join(f"{u}-{v}={w}" for u in adj for v, w in adj[u].items() if u < v)
    ]
    steps.append(f"Start at {src}, end at {dst}.")
    for i in range(len(path) - 1):
        w = adj[path[i]][path[i + 1]]
        steps.append(f"Move {path[i]} -> {path[i + 1]} (cost {w}).")
    steps.append(f"Total cost: {int(dist[dst])}.")
    header = "Problem: Find the shortest path.\nSolution:\n"
    return {
        "domain": "graph",
        "problem": header,
        "numbers": [int(dist[dst])],
        "target": int(dist[dst]),
        "solution": steps,
        "cot": "\n".join(steps),
    }


_GENERATORS = {
    "algebra": _algebra_problem,
    "logic": _logic_problem,
    "graph": _graph_problem,
}


def generate_transfer_dataset(
    domain: str, num_samples: int, output_file: str, seed: int = 0
) -> None:
    """Generate ``num_samples`` problems of ``domain`` to ``output_file``."""
    if domain not in _GENERATORS:
        raise ValueError(f"Unknown domain {domain!r}; choose from {DOMAINS}")
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    gen = _GENERATORS[domain]
    with open(output_file, "w") as f:
        for _ in range(num_samples):
            f.write(json.dumps(gen(rng)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate V5 cross-domain reasoning datasets"
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--num_samples", type=int, default=2000, help="Per domain")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for domain in args.domains:
        out = os.path.join(args.output_dir, f"transfer_{domain}.jsonl")
        print(f"[{domain}] generating {args.num_samples} problems -> {out}")
        # Per-domain deterministic seed offset, mirroring the Countdown generator.
        generate_transfer_dataset(
            domain, args.num_samples, out, seed=args.seed + DOMAINS.index(domain)
        )
    print("Done!")


if __name__ == "__main__":
    main()
