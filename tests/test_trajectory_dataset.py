import torch

from models.model_loader import load_model, load_tokenizer
from data_processing.trajectory_dataset import (
    build_trajectories,
    TransitionDataset,
    DecoderDataset,
    format_header,
    _state_end_chars,
    _token_index_for_char_end,
)

TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

PROBLEMS = [
    {
        "numbers": [25, 100, 50, 2, 5, 4],
        "target": 18,
        "solution": ["25 - 5 = 20", "20 - 2 = 18"],
    },
    {"numbers": [2, 4, 8, 1, 3, 6], "target": 8, "solution": ["2 * 4 = 8"]},
]


def test_state_end_chars_alignment():
    header = format_header([1, 2, 3], 6)
    steps = ["1 + 2 = 3", "3 + 3 = 6"]
    ends = _state_end_chars(header, steps)
    # s_0 ends at end of header; one extra state per step.
    assert ends[0] == len(header)
    assert len(ends) == len(steps) + 1
    full = header + "\n".join(steps)
    # Each step end char should point just past that step's text.
    assert full[: ends[1]].endswith("1 + 2 = 3")
    assert full[: ends[2]].endswith("3 + 3 = 6")


def test_token_index_monotonic():
    offsets = [(0, 0), (0, 3), (3, 5), (5, 9)]
    assert _token_index_for_char_end(offsets, 9) == 3
    assert _token_index_for_char_end(offsets, 5) == 2
    assert _token_index_for_char_end(offsets, 3) == 1


def test_build_trajectories_shapes():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)

    trajs = build_trajectories(model, tokenizer, PROBLEMS, layer=-1)
    assert len(trajs) == 2

    t0 = trajs[0]
    # Two steps -> three states s_0, s_1, s_2.
    assert t0.num_steps == 2
    assert t0.state_indices.shape[0] == 3
    assert t0.states.shape == (3, t0.hidden_dim)
    assert t0.op_ids.tolist() == [1, 1]  # SUB, SUB
    # State indices are strictly increasing along the sequence.
    assert torch.all(t0.state_indices[1:] > t0.state_indices[:-1])
    # state_indices are valid positions in the token sequence.
    assert int(t0.state_indices.max()) < t0.all_hidden.shape[0]


def test_transition_and_decoder_datasets():
    model = load_model(model_id=TEST_MODEL, device_map="cpu")
    tokenizer = load_tokenizer(model_id=TEST_MODEL)
    trajs = build_trajectories(model, tokenizer, PROBLEMS, layer=-1)

    tds = TransitionDataset(trajs)
    # 2 transitions from traj0 + 1 from traj1 = 3.
    assert len(tds) == 3
    sample = tds[0]
    H = trajs[0].hidden_dim
    assert sample["h_t"].shape == (H,)
    assert sample["h_next"].shape == (H,)
    assert sample["operands"].shape == (2,)
    assert sample["op_id"].dtype == torch.long

    dds = DecoderDataset(trajs)
    assert len(dds) > 0
    d0 = dds[0]
    assert d0["hidden"].shape == (H,)
    assert d0["target"].dtype == torch.long
