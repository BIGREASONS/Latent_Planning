"""Generic hidden-state trajectory extractor for non-Countdown CoT.

The Countdown-specific :func:`data_processing.trajectory_dataset.build_trajectory`
relies on the Countdown action parser to delimit steps. The transfer datasets
(algebra / logic / graph) have no symbolic action grammar — but they *do* share
the Countdown format of "header + one step per line", and the exact same
token-alignment helpers (``_state_end_chars`` / ``_token_index_for_char_end``)
work on any such text.

This module produces :class:`~data_processing.trajectory_dataset.Trajectory`
objects that are duck-type-compatible with the Countdown trajectories, so all
downstream V5 machinery (VQ encoding, discrete transition dataset, etc.)
operates on them without modification. ``op_ids`` / ``operands`` are filled
with zeros (there is no action semantics to encode) — they are present only to
satisfy the dataclass; downstream transfer analysis does not use them.

State alignment
---------------
For a problem with ``N`` step lines we define ``N+1`` latent states:
``s_0`` at the last token of the header, and ``s_i`` (i>=1) at the last token
of step ``i``'s line — exactly as in the Countdown pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from data_processing.trajectory_dataset import (
    HEADER_TEMPLATE,
    Trajectory,
    _state_end_chars,
    _token_index_for_char_end,
)


def build_transfer_trajectory(
    model,
    tokenizer,
    problem: Dict[str, Any],
    layer: int = -1,
) -> Optional[Trajectory]:
    """Build a :class:`Trajectory` from a transfer-domain problem dict.

    Requires ``problem`` (the header text) and ``solution`` (list of step
    strings). Returns ``None`` if the problem has no steps or alignment fails.
    """
    header = problem.get("problem", HEADER_TEMPLATE)
    steps = problem.get("solution", [])
    if not steps:
        return None

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
        state_indices = [_token_index_for_char_end(offsets, ec) for ec in end_chars]
    except ValueError:
        return None
    state_indices = torch.tensor(state_indices, dtype=torch.long)

    n_steps = len(steps)
    # No symbolic action grammar: zero op_ids / operands (unused downstream).
    op_ids = torch.zeros(n_steps, dtype=torch.long)
    operands = torch.zeros((n_steps, 2), dtype=torch.float32)

    return Trajectory(
        all_hidden=hidden,
        input_ids=input_ids,
        state_indices=state_indices,
        op_ids=op_ids,
        operands=operands,
        numbers=list(problem.get("numbers", [])),
        target=int(problem.get("target", 0)),
    )


def build_transfer_trajectories(
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    layer: int = -1,
    batch_size: int = 32,
) -> List[Trajectory]:
    """Batched transfer-trajectory extraction (mirrors ``build_trajectories``)."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with offset_mapping is required.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    out: List[Trajectory] = []
    from tqdm import tqdm

    for i in tqdm(range(0, len(problems), batch_size), desc="Extracting transfer"):
        batch = problems[i : i + batch_size]
        valid = []
        for problem in batch:
            if problem.get("solution"):
                valid.append(problem)
        if not valid:
            continue

        texts = [p["problem"] + "\n".join(p["solution"]) for p in valid]
        enc = tokenizer(
            texts, padding=True, return_tensors="pt", return_offsets_mapping=True
        )
        offset_mappings = enc.pop("offset_mapping").tolist()
        attention_mask = enc["attention_mask"]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden_batch = outputs.hidden_states[layer].detach().cpu().float()
        input_ids_batch = enc["input_ids"].detach().cpu()
        attention_cpu = attention_mask.detach().cpu()

        for b_idx, problem in enumerate(valid):
            pad_mask = attention_cpu[b_idx].bool()
            hidden = hidden_batch[b_idx][pad_mask]
            input_ids = input_ids_batch[b_idx][pad_mask]
            valid_offsets = [
                offset_mappings[b_idx][t]
                for t in range(len(offset_mappings[b_idx]))
                if pad_mask[t]
            ]
            header = problem["problem"]
            steps = problem["solution"]
            end_chars = _state_end_chars(header, steps)
            try:
                state_indices = [
                    _token_index_for_char_end(valid_offsets, ec) for ec in end_chars
                ]
            except ValueError:
                continue
            state_indices = torch.tensor(state_indices, dtype=torch.long)
            n_steps = len(steps)
            traj = Trajectory(
                all_hidden=hidden,
                input_ids=input_ids,
                state_indices=state_indices,
                op_ids=torch.zeros(n_steps, dtype=torch.long),
                operands=torch.zeros((n_steps, 2), dtype=torch.float32),
                numbers=list(problem.get("numbers", [])),
                target=int(problem.get("target", 0)),
            )
            if traj.num_steps > 0:
                out.append(traj)
    return out
