"""Verify the pos_encoder auto-resize hook on GTrXL.

Ckpts trained in late 2025 saved pos_encoder with shape [1, 100, 128]; the
current default constructs [1, 48, 128] (constraint floor of mem_len+32 with
mem_len=16). Loading raised:

    RuntimeError: size mismatch for gtrxl.pos_encoder: copying a param
    with shape [1, 100, 128] from checkpoint, the shape in current model
    is [1, 48, 128]

The pre-hook reshapes self.pos_encoder to match the checkpoint before
load_state_dict's strict check runs.
"""

import torch

from src.networks.gtrxl import GTrXL


def _build(max_seq_len: int) -> GTrXL:
    return GTrXL(
        input_dim=64, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=0.0, mem_len=16,
        max_seq_len=max_seq_len,
    )


def test_load_smaller_state_into_larger_model():
    """Ckpt has [1, 32, 128] but model wants [1, 64, 128] — must adapt."""
    src = _build(max_seq_len=32)
    dst = _build(max_seq_len=64)
    state = src.state_dict()
    dst.load_state_dict(state, strict=True)
    assert dst.pos_encoder.shape == (1, 32, 128)


def test_load_larger_state_into_smaller_model():
    """Real-world failure shape: ckpt [1, 100, 128], model [1, 48, 128]."""
    src = _build(max_seq_len=100)
    dst = _build(max_seq_len=48)
    state = src.state_dict()
    dst.load_state_dict(state, strict=True)
    assert dst.pos_encoder.shape == (1, 100, 128)
    # Weights are actually copied, not zeroed:
    assert torch.allclose(dst.pos_encoder, src.pos_encoder)


def test_no_resize_when_shapes_already_match():
    src = _build(max_seq_len=64)
    dst = _build(max_seq_len=64)
    pre_id = id(dst.pos_encoder)
    dst.load_state_dict(src.state_dict(), strict=True)
    assert dst.pos_encoder.shape == (1, 64, 128)
    # Sanity: still a Parameter, still trainable
    assert dst.pos_encoder.requires_grad
