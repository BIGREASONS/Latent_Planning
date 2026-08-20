import torch

from models.diagnostic_decoder import DiagnosticDecoder
from training.train_decoder import train_decoder_model, DecoderTrainConfig
from data_processing.trajectory_dataset import Trajectory


def test_decoder_shapes():
    dec = DiagnosticDecoder(hidden_dim=8, vocab_size=5)
    h = torch.randn(3, 8)
    assert dec(h).shape == (3, 5)
    assert dec.predict(h).shape == (3,)


def _decodable_trajectories(n=12, H=8, vocab=5, T=6, seed=0):
    """Trajectories where each hidden state linearly determines the next token."""
    g = torch.Generator().manual_seed(seed)
    token_vecs = torch.randn(vocab, H, generator=g)  # fixed embedding per token
    trajs = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (T,), generator=g)
        # hidden[p] encodes the token that comes at p+1.
        all_hidden = torch.zeros(T, H)
        for p in range(T - 1):
            all_hidden[p] = token_vecs[ids[p + 1]]
        trajs.append(
            Trajectory(
                all_hidden=all_hidden,
                input_ids=ids,
                state_indices=torch.arange(min(3, T)),
                op_ids=torch.zeros(min(2, T - 1), dtype=torch.long),
                operands=torch.zeros(min(2, T - 1), 2),
                numbers=[1, 2, 3, 4, 5, 6],
                target=10,
            )
        )
    return trajs


def test_decoder_learns_mapping():
    trajs = _decodable_trajectories()
    cfg = DecoderTrainConfig(epochs=80, lr=5e-2, batch_size=32)
    model = train_decoder_model(trajs, vocab_size=5, val_trajs=trajs, config=cfg)
    hist = cfg.history
    assert hist[-1]["accuracy"] > hist[0]["accuracy"]
    # Mapping is deterministic and linearly separable -> should be learned well.
    assert hist[-1]["accuracy"] > 0.9
