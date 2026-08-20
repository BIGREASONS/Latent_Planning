import os
import json
import pytest
from scripts.generate_countdown_dataset import (
    generate_countdown_problem,
    generate_dataset,
)


def test_generate_countdown_problem():
    problem = generate_countdown_problem()
    assert "numbers" in problem
    assert "target" in problem
    assert "solution" in problem
    assert "cot" in problem
    assert len(problem["numbers"]) == 6
    assert isinstance(problem["target"], int)
    assert isinstance(problem["cot"], str)
    assert len(problem["solution"]) > 0


def test_generate_dataset(tmp_path):
    output_file = os.path.join(tmp_path, "test_dataset.jsonl")
    generate_dataset(10, output_file)

    assert os.path.exists(output_file)
    with open(output_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 10

        # Verify first line parses
        data = json.loads(lines[0])
        assert "numbers" in data
