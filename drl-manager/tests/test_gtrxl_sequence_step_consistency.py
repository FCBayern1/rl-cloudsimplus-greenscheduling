"""The sequence path and the single-step rollout path must be the same function.

PPO's ratio exp(new_logp - old_logp) compares a log-prob recorded during rollout (one step at
a time) with one recomputed on a training chunk. If the two paths encode the same state
differently, the ratio compares two different functions and the gradient is corrupted. Until
2026-09-10 the positional encoding was indexed by position inside the chunk, so rollout always
saw position 0 while training saw 0..T-1.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.networks.gtrxl import GTrXL


def _model(seed=0, d_model=16, layers=2, mem=4, max_seq=32):
    torch.manual_seed(seed)
    m = GTrXL(input_dim=8, d_model=d_model, num_layers=layers, nhead=2,
              dim_feedforward=32, mem_len=mem, max_seq_len=max_seq)
    m.eval()
    # a non-zero positional parameter, so a position-dependent encoding would show up
    with torch.no_grad():
        m.pos_encoder.copy_(torch.randn_like(m.pos_encoder))
    return m


def test_sequence_matches_step_by_step_rollout():
    m = _model()
    torch.manual_seed(1)
    x = torch.randn(3, 7, 8)                       # (B, T, input_dim)

    with torch.no_grad():
        seq_out, seq_state = m(x)                  # the training path: one call, T = 7
        step_out, state = [], None
        for t in range(x.shape[1]):                # the rollout path: T = 1, carrying state
            o, state = m(x[:, t:t + 1, :], state=state)
            step_out.append(o)
        step_out = torch.cat(step_out, dim=1)

    torch.testing.assert_close(seq_out, step_out, rtol=1e-5, atol=1e-6)
    for a, b in zip(seq_state, state):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_a_chunk_boundary_does_not_change_the_answer():
    """Splitting a trajectory into two training chunks must give the same outputs as one."""
    m = _model(seed=2)
    torch.manual_seed(3)
    x = torch.randn(2, 6, 8)

    with torch.no_grad():
        whole, _ = m(x)
        first, st = m(x[:, :4, :])
        second, _ = m(x[:, 4:, :], state=st)

    torch.testing.assert_close(whole, torch.cat([first, second], dim=1), rtol=1e-5, atol=1e-6)


def test_position_index_is_not_used_beyond_the_first_row():
    """Rows of pos_encoder past the first must not affect the output; if they do, the two
    paths can diverge again."""
    m = _model(seed=4)
    torch.manual_seed(5)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        before, _ = m(x)
        m.pos_encoder[:, 1:, :] += 10.0            # poison every row except the first
        after, _ = m(x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_state_carries_between_calls_rather_than_resetting():
    """A regression guard for the evaluation fault fixed earlier: dropping the state must
    change the answer, so a silent state loss cannot pass unnoticed."""
    m = _model(seed=6)
    torch.manual_seed(7)
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        _, st = m(x)
        with_state, _ = m(x, state=st)
        without_state, _ = m(x)
    assert not torch.allclose(with_state, without_state, rtol=1e-4, atol=1e-5)
