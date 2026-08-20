"""Teacher trajectory dataset.

Builds ``(h_t, action_t, h_{t+1})`` transition tuples from a frozen language
model's hidden states, aligned to the boundaries of each symbolic reasoning
step.

State alignment
---------------
For a problem with an ``N``-step solution we define a sequence of *latent
states* ``s_0, s_1, ..., s_N`` where:

* ``s_0`` is the hidden state at the **last token of the prompt header**
  (``"...Solution:\n"``) — the model's encoding of the problem before any step.
* ``s_i`` (i >= 1) is the hidden state at the **last token of reasoning step i**
  (the final token of that line, i.e. the last digit of the step's result).

A transition ``i`` is then ``(s_{i-1}, action_i, s_i)``: applying the symbolic
action of step ``i`` should carry the latent state forward by one step. Token
alignment is computed from the fast tokenizer's ``offset_mapping`` so it is
robust to sub-word merges.

Each :class:`Trajectory` also retains the full per-token hidden states and input
ids so the diagnostic decoder can be trained on every position, and so the
coherence rollout can look up the teacher token that follows any state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from data_processing.action_parser import Action, parse_solution

HEADER_TEMPLATE = (
    "Problem: Given the numbers {numbers}, reach the target {target}.\nSolution:\n"
)
IGNORE_TOKEN = -100  # used for "no next token" (terminal state)


def format_header(numbers: List[int], target: int) -> str:
    """Prompt header, matching the format used by training / extraction."""
    return HEADER_TEMPLATE.format(numbers=numbers, target=target)


@dataclass
class Trajectory:
    """One teacher trajectory through latent space for a single problem."""

    all_hidden: torch.Tensor  # (T, H) hidden state for every token
    input_ids: torch.Tensor  # (T,) token ids
    state_indices: torch.Tensor  # (N+1,) indices into all_hidden for s_0..s_N
    op_ids: torch.Tensor  # (N,) action op id for steps 1..N
    operands: torch.Tensor  # (N, 2) float [arg1, arg2] for steps 1..N
    numbers: List[int]
    target: int

    @property
    def num_steps(self) -> int:
        return int(self.op_ids.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.all_hidden.shape[1])

    @property
    def states(self) -> torch.Tensor:
        """(N+1, H) the aligned latent states s_0..s_N."""
        return self.all_hidden[self.state_indices]

    def next_token_after_state(self, i: int) -> int:
        """Teacher token id immediately following state ``s_i`` (or IGNORE)."""
        idx = int(self.state_indices[i].item())
        nxt = idx + 1
        if nxt < self.input_ids.shape[0]:
            return int(self.input_ids[nxt].item())
        return IGNORE_TOKEN


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #
def _state_end_chars(header: str, steps: List[str]) -> List[int]:
    """Exclusive character offsets marking the end of s_0..s_N in the prompt."""
    ends = [len(header)]  # s_0 = end of header
    c = len(header)
    for i, step in enumerate(steps):
        c += len(step)
        ends.append(c)
        c += 1  # the "\n" joining this step to the next
    return ends


def _token_index_for_char_end(offsets: List[tuple], char_end: int) -> int:
    """Index of the last non-empty token whose span ends at/before ``char_end``."""
    best = None
    for i, (a, b) in enumerate(offsets):
        if b > a and b <= char_end:
            best = i
    if best is None:  # pragma: no cover - would mean an empty prompt
        raise ValueError(f"No token found for char_end={char_end}")
    return best


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_trajectory(
    model,
    tokenizer,
    problem: Dict[str, Any],
    layer: int = -1,
) -> Optional[Trajectory]:
    """Build a single :class:`Trajectory` from a problem dict.

    The problem dict must contain ``numbers``, ``target`` and ``solution``
    (a list of step strings). Returns ``None`` if the solution is empty or the
    steps cannot be parsed.
    """
    numbers = problem["numbers"]
    target = problem["target"]
    steps = problem["solution"]
    if not steps:
        return None

    try:
        actions: List[Action] = parse_solution(steps)
    except ValueError:
        return None

    header = format_header(numbers, target)
    full_text = header + "\n".join(steps)

    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0].detach().cpu().float()  # (T, H)
    input_ids = enc["input_ids"][0].detach().cpu()

    end_chars = _state_end_chars(header, steps)
    state_indices = [_token_index_for_char_end(offset_mapping, ec) for ec in end_chars]
    state_indices = torch.tensor(state_indices, dtype=torch.long)

    op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
    operands = torch.tensor([[a.arg1, a.arg2] for a in actions], dtype=torch.float32)

    return Trajectory(
        all_hidden=hidden,
        input_ids=input_ids,
        state_indices=state_indices,
        op_ids=op_ids,
        operands=operands,
        numbers=list(numbers),
        target=int(target),
    )


def build_trajectories(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    layer: int = -1,
    batch_size: int = 32,
) -> List[Trajectory]:
    """Build trajectories for a list of problems, skipping unparseable ones, in batches."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with offset_mapping support is required.")

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    trajectories: List[Trajectory] = []

    from tqdm import tqdm

    for i in tqdm(range(0, len(problems), batch_size), desc="Extracting"):
        batch = problems[i : i + batch_size]

        valid_problems = []
        valid_actions = []
        valid_texts = []
        valid_headers = []

        for problem in batch:
            numbers = problem["numbers"]
            target = problem["target"]
            steps = problem["solution"]
            if not steps:
                continue
            try:
                actions: List[Action] = parse_solution(steps)
            except ValueError:
                continue

            header = format_header(numbers, target)
            full_text = header + "\n".join(steps)

            valid_problems.append(problem)
            valid_actions.append(actions)
            valid_texts.append(full_text)
            valid_headers.append(header)

        if not valid_texts:
            continue

        enc = tokenizer(
            valid_texts, padding=True, return_tensors="pt", return_offsets_mapping=True
        )
        offset_mappings = enc.pop("offset_mapping").tolist()
        attention_mask = enc["attention_mask"]
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)

        hidden_batch = outputs.hidden_states[layer].detach().cpu().float()
        input_ids_batch = enc["input_ids"].detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()

        for b_idx in range(len(valid_texts)):
            problem = valid_problems[b_idx]
            actions = valid_actions[b_idx]
            header = valid_headers[b_idx]
            steps = problem["solution"]

            pad_mask = attention_mask_cpu[b_idx].bool()

            hidden = hidden_batch[b_idx][pad_mask]
            input_ids = input_ids_batch[b_idx][pad_mask]

            valid_offsets = [
                offset_mappings[b_idx][t_idx]
                for t_idx in range(len(offset_mappings[b_idx]))
                if pad_mask[t_idx]
            ]

            end_chars = _state_end_chars(header, steps)
            try:
                state_indices = [
                    _token_index_for_char_end(valid_offsets, ec) for ec in end_chars
                ]
            except ValueError:
                continue

            state_indices = torch.tensor(state_indices, dtype=torch.long)
            op_ids = torch.tensor([a.op_id for a in actions], dtype=torch.long)
            operands = torch.tensor(
                [[a.arg1, a.arg2] for a in actions], dtype=torch.float32
            )

            traj = Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=op_ids,
                operands=operands,
                numbers=list(problem["numbers"]),
                target=int(problem["target"]),
            )
            if traj.num_steps > 0:
                trajectories.append(traj)

    return trajectories


def load_problems(path: str) -> List[Dict[str, Any]]:
    """Load problems from a JSONL file (one problem per line)."""
    problems = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def save_trajectories(trajectories: List[Trajectory], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(trajectories, path)


def load_trajectories(path: str) -> List[Trajectory]:
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------- #
# Flat transition view (for the transition model)
# --------------------------------------------------------------------------- #
class TransitionDataset(torch.utils.data.Dataset):
    """Flat ``(h_t, op_id, operands, h_{t+1})`` view over trajectories."""

    def __init__(self, trajectories: List[Trajectory]):
        self.h_t: List[torch.Tensor] = []
        self.h_next: List[torch.Tensor] = []
        self.op_ids: List[int] = []
        self.operands: List[torch.Tensor] = []
        for traj in trajectories:
            states = traj.states  # (N+1, H)
            for i in range(traj.num_steps):
                self.h_t.append(states[i])
                self.h_next.append(states[i + 1])
                self.op_ids.append(int(traj.op_ids[i].item()))
                self.operands.append(traj.operands[i])

    def __len__(self) -> int:
        return len(self.h_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "h_t": self.h_t[idx],
            "op_id": torch.tensor(self.op_ids[idx], dtype=torch.long),
            "operands": self.operands[idx],
            "h_next": self.h_next[idx],
        }


class DecoderDataset(torch.utils.data.Dataset):
    """Per-token ``(hidden_state, next_token_id)`` view for the decoder."""

    def __init__(self, trajectories: List[Trajectory]):
        self.hidden: List[torch.Tensor] = []
        self.targets: List[int] = []
        for traj in trajectories:
            T = traj.all_hidden.shape[0]
            for p in range(T - 1):
                self.hidden.append(traj.all_hidden[p])
                self.targets.append(int(traj.input_ids[p + 1].item()))

    def __len__(self) -> int:
        return len(self.hidden)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "hidden": self.hidden[idx],
            "target": torch.tensor(self.targets[idx], dtype=torch.long),
        }
