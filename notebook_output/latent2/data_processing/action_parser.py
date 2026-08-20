"""Symbolic action parser for Countdown reasoning steps.

Converts a textual reasoning step such as ``"75 * 11 = 825"`` into a structured
symbolic action::

    Action(op=Op.MUL, arg1=75, arg2=11, result=825)

Supported operations: ADD (+), SUB (-), MUL (*), DIV (/).

The parser is the bridge between the teacher's natural-language chain-of-thought
and the symbolic action space used by the transition model and probes. It is
deliberately strict about structure but, by default, lenient about arithmetic
correctness: we record exactly what the teacher wrote (including mistakes) and
expose :meth:`Action.is_arithmetically_valid` so callers can filter if desired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class Op(Enum):
    """The four supported Countdown operations."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


# Stable, contiguous integer ids for embedding lookups. The order here is the
# canonical action id space used throughout the project (transition model action
# embedding, probe C labels, etc.).
OP_TO_ID = {Op.ADD: 0, Op.SUB: 1, Op.MUL: 2, Op.DIV: 3}
ID_TO_OP = {i: op for op, i in OP_TO_ID.items()}

_SYMBOL_TO_OP = {op.value: op for op in Op}

# "<int> <op> <int> = <int>" with arbitrary surrounding/inner whitespace.
_STEP_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)\s*$")


@dataclass(frozen=True)
class Action:
    """A single symbolic Countdown action."""

    op: Op
    arg1: int
    arg2: int
    result: int

    @property
    def op_id(self) -> int:
        return OP_TO_ID[self.op]

    def is_arithmetically_valid(self) -> bool:
        """Whether ``arg1 op arg2`` actually equals ``result``."""
        try:
            return apply_op(self.op, self.arg1, self.arg2) == self.result
        except (ZeroDivisionError, ValueError):
            return False


def apply_op(op: Op, a: int, b: int) -> int:
    """Apply a symbolic op to two integers.

    Division is integer division and requires an exact (remainder-free) result,
    matching the Countdown rule that only whole-number intermediate values are
    legal.
    """
    if op == Op.ADD:
        return a + b
    if op == Op.SUB:
        return a - b
    if op == Op.MUL:
        return a * b
    if op == Op.DIV:
        if b == 0 or a % b != 0:
            raise ValueError(f"Illegal division: {a} / {b}")
        return a // b
    raise ValueError(f"Unknown op: {op!r}")


def parse_step(step: str, validate: bool = False) -> Action:
    """Parse a single reasoning step string into an :class:`Action`.

    Args:
        step: e.g. ``"75 * 11 = 825"``.
        validate: if True, raise ``ValueError`` when the stated result does not
            match the arithmetic of the operands.

    Raises:
        ValueError: if the string is not a well-formed ``a op b = c`` equation
            with a supported operator, or (when ``validate``) is arithmetically
            inconsistent.
    """
    match = _STEP_RE.match(step)
    if match is None:
        raise ValueError(f"Malformed reasoning step: {step!r}")

    arg1_s, symbol, arg2_s, result_s = match.groups()
    op = _SYMBOL_TO_OP.get(symbol)
    if op is None:  # pragma: no cover - regex already restricts the symbol set
        raise ValueError(f"Unsupported operator {symbol!r} in step: {step!r}")

    action = Action(
        op=op,
        arg1=int(arg1_s),
        arg2=int(arg2_s),
        result=int(result_s),
    )

    if validate and not action.is_arithmetically_valid():
        raise ValueError(f"Arithmetically invalid step: {step!r}")

    return action


def parse_solution(steps: List[str], validate: bool = False) -> List[Action]:
    """Parse a list of reasoning-step strings into a list of actions."""
    return [parse_step(s, validate=validate) for s in steps]
