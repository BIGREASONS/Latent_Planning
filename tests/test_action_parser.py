import pytest

from data_processing.action_parser import (
    Op,
    Action,
    OP_TO_ID,
    ID_TO_OP,
    parse_step,
    parse_solution,
    apply_op,
)


def test_parse_mul():
    action = parse_step("75 * 11 = 825")
    assert action.op == Op.MUL
    assert action.arg1 == 75
    assert action.arg2 == 11
    assert action.result == 825


@pytest.mark.parametrize(
    "step,op,a,b,r",
    [
        ("3 + 4 = 7", Op.ADD, 3, 4, 7),
        ("20 - 2 = 18", Op.SUB, 20, 2, 18),
        ("6 * 7 = 42", Op.MUL, 6, 7, 42),
        ("100 / 4 = 25", Op.DIV, 100, 4, 25),
    ],
)
def test_parse_all_ops(step, op, a, b, r):
    action = parse_step(step)
    assert action.op == op
    assert action.arg1 == a
    assert action.arg2 == b
    assert action.result == r


def test_parse_tolerates_extra_whitespace():
    action = parse_step("  25   -   5   =   20  ")
    assert action.op == Op.SUB
    assert action.arg1 == 25
    assert action.arg2 == 5
    assert action.result == 20


def test_parse_solution_list():
    steps = ["10 * 8 = 80", "80 - 50 = 30", "30 + 100 = 130"]
    actions = parse_solution(steps)
    assert len(actions) == 3
    assert [a.op for a in actions] == [Op.MUL, Op.SUB, Op.ADD]
    assert actions[-1].result == 130


def test_op_id_roundtrip():
    # Stable, contiguous ids 0..3 for embedding lookups.
    assert sorted(OP_TO_ID.values()) == [0, 1, 2, 3]
    for op in Op:
        assert ID_TO_OP[OP_TO_ID[op]] == op


def test_apply_op_matches_arithmetic():
    assert apply_op(Op.ADD, 3, 4) == 7
    assert apply_op(Op.SUB, 20, 2) == 18
    assert apply_op(Op.MUL, 6, 7) == 42
    assert apply_op(Op.DIV, 100, 4) == 25


def test_malformed_raises():
    with pytest.raises(ValueError):
        parse_step("not an equation")
    with pytest.raises(ValueError):
        parse_step("3 ^ 4 = 81")  # unsupported operator
    with pytest.raises(ValueError):
        parse_step("3 + 4")  # missing result


def test_result_mismatch_can_be_validated():
    # By default we trust the teacher text; strict mode flags arithmetic errors.
    bad = parse_step("3 + 4 = 99")  # no raise by default
    assert bad.result == 99
    assert not bad.is_arithmetically_valid()
    with pytest.raises(ValueError):
        parse_step("3 + 4 = 99", validate=True)
