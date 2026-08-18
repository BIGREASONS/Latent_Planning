import torch

from models.vq_state import VQStateQuantizer


def test_forward_shapes():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    h = torch.randn(5, 8)
    z_q, indices, info = vq(h, training=False)
    assert z_q.shape == (5, 8)
    assert indices.shape == (5,)
    assert indices.dtype == torch.int64
    for key in ("commitment_loss", "codebook_loss", "perplexity", "active_codes"):
        assert key in info


def test_straight_through_gradient_flows():
    """STE: decoder-side gradient reaches the encoder input."""
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    vq.eval()
    h = torch.randn(3, 8, requires_grad=True)
    z_q, _, _ = vq(h, training=False)
    z_q.sum().backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_encode_decode_roundtrip():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=16)
    h = torch.randn(7, 8)
    z = vq.encode(h)
    assert z.shape == (7,)
    decoded = vq.decode(z)
    assert decoded.shape == (7, 8)
    # Decoded vectors must equal the codebook entries they were indexed from.
    assert torch.allclose(decoded, vq.codebook[z])


def test_ema_recovers_cluster_centers():
    """With EMA on, the codebook should converge to the data cluster centers."""
    torch.manual_seed(0)
    # Use a relatively fast EMA decay so codebook adapts within the loop.
    vq = VQStateQuantizer(hidden_dim=8, num_codes=4, ema_decay=0.9, epsilon=1e-5)
    centers = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    h = centers.repeat(50, 1) + 0.01 * torch.randn(200, 8)
    vq.train()
    # Run enough steps for dead-code revival to activate all codes.
    for _ in range(500):
        vq(h, training=True)
    # All 4 codes should be active (perplexity ~ 4).
    _, _, info = vq(h, training=False)
    assert int(info["active_codes"]) == 4, (
        f"Expected 4 active codes, got {int(info['active_codes'])}"
    )
    assert float(info["perplexity"]) > 3.0
    # Commitment loss -> ~0 once the codebook matches the centers.
    assert float(info["commitment_loss"]) < 0.1


def test_diagnostics_in_range():
    vq = VQStateQuantizer(hidden_dim=8, num_codes=10)
    h = torch.randn(20, 8)
    _, _, info = vq(h, training=False)
    assert 1.0 <= float(info["perplexity"]) <= 10.0
    assert 1 <= int(info["active_codes"]) <= 10
