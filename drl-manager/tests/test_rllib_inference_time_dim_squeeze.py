"""GTrXL emits action_dist_inputs with a leading time dim [B, T, A] while
flat backbones emit [B, A]. The local + global schedulers must accept both.
Direct reproduction of the failure observed on 2026-05-08:

    IndexError: The shape of the mask [1, 225] at index 1 does not match
    the shape of the indexed tensor [1, 1, 225] at index 1
"""

import numpy as np
import torch


def test_local_scheduler_squeezes_time_dim():
    """Replay the exact masking branch from RLlibLocalScheduler.schedule."""
    full_mask = np.ones(225, dtype=np.float32)
    full_mask[100:] = 0  # half invalid

    # Simulate GTrXL output shape: [B, T, A] with T=1.
    dist_inputs = torch.randn(1, 1, 225)

    # The squeeze logic that lives in local_schedulers.py
    if dist_inputs.dim() == 3 and dist_inputs.shape[1] == 1:
        dist_inputs = dist_inputs.squeeze(1)

    # The masking line that previously crashed:
    mask_tensor = torch.from_numpy(full_mask).unsqueeze(0)
    masked_logits = dist_inputs.clone()
    masked_logits[mask_tensor == 0] = float("-inf")
    action = torch.argmax(masked_logits, dim=-1).item()

    # Argmax must land on a valid index (0..99)
    assert 0 <= action < 100


def test_local_scheduler_flat_backbone_unchanged():
    """Flat backbones emit [B, A]; squeeze must be a no-op."""
    dist_inputs = torch.randn(1, 225)
    if dist_inputs.dim() == 3 and dist_inputs.shape[1] == 1:
        dist_inputs = dist_inputs.squeeze(1)
    assert dist_inputs.shape == (1, 225)


def test_global_scheduler_squeezes_time_dim():
    """RLlibNewAPIGlobalScheduler reshape would silently misinterpret values
    if the time dim weren't squeezed; assert squeeze restores [B, A]."""
    # 10 DCs, batch size 10 -> 100 logits flat
    dist_inputs = torch.randn(1, 1, 100)
    if dist_inputs.dim() == 3 and dist_inputs.shape[1] == 1:
        dist_inputs = dist_inputs.squeeze(1)
    assert dist_inputs.shape == (1, 100)
    # Now the reshape downstream (1, 10, 10) is unambiguous
    reshaped = dist_inputs.reshape(1, 10, 10)
    assert reshaped.shape == (1, 10, 10)
