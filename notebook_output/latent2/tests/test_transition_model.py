import torch

from models.transition_model import TransitionModel, ActionEncoder, transition_loss
from training.train_transition import train_transition_model, TransitionTrainConfig
from data_processing.trajectory_dataset import Trajectory


def _toy_trajectories(n=8, H=16, steps=3, seed=0):
    """Synthetic trajectories with a learnable linear dynamics per op."""
    g = torch.Generator().manual_seed(seed)
    trajs = []
    # Fixed per-op shift vectors so a model can actually learn the dynamics.
    op_shifts = torch.randn(4, H, generator=g)
    for _ in range(n):
        T = steps + 5
        all_hidden = torch.zeros(T, H)
        state_indices = torch.arange(steps + 1)
        op_ids = torch.randint(0, 4, (steps,), generator=g)
        operands = torch.randint(1, 50, (steps, 2), generator=g).float()
        s = torch.randn(H, generator=g)
        all_hidden[0] = s
        for i in range(steps):
            s = s + op_shifts[op_ids[i]]
            all_hidden[i + 1] = s
        trajs.append(Trajectory(
            all_hidden=all_hidden,
            input_ids=torch.zeros(T, dtype=torch.long),
            state_indices=state_indices,
            op_ids=op_ids,
            operands=operands,
            numbers=[1, 2, 3, 4, 5, 6],
            target=10,
        ))
    return trajs


def test_action_encoder_dim():
    enc = ActionEncoder(op_embed_dim=8)
    assert enc.output_dim == 10
    out = enc(torch.tensor([0, 1]), torch.tensor([[10.0, 20.0], [3.0, 4.0]]))
    assert out.shape == (2, 10)


def test_forward_shapes_and_delta():
    model = TransitionModel(hidden_dim=16, mlp_hidden_dim=32)
    h = torch.randn(5, 16)
    op = torch.randint(0, 4, (5,))
    operands = torch.randint(1, 10, (5, 2)).float()
    out = model(h, op, operands)
    assert out.shape == (5, 16)


def test_training_reduces_loss():
    trajs = _toy_trajectories(n=16, H=16, steps=3)
    cfg = TransitionTrainConfig(epochs=60, mlp_hidden_dim=64, lr=5e-3, batch_size=16)
    model = train_transition_model(trajs, val_trajs=trajs, config=cfg)
    history = cfg.history
    assert history[-1]["loss"] < history[0]["loss"]
    # On clean synthetic dynamics the model should fit well.
    assert history[-1]["loss"] < 0.1
    assert "eval_loss" in history[-1]
