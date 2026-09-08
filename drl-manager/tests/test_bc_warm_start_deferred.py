"""Checkpoint restore through the BC warm-start path when part of the module is built after
the base setup (the CRD modules' Q-ensemble heads)."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.rlmodule_gtrxl_models import bc_consume_pending, bc_warm_start_load


class _Base(nn.Module):
    BC_DEFERRED_PREFIXES = ("q_heads.",)

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 2)


class _Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_heads = nn.ModuleList([nn.Linear(2, 1), nn.Linear(2, 1)])


def _full_state(seed):
    torch.manual_seed(seed)
    m = _Base(); m.q_heads = _Heads()
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_deferred_keys_are_held_back_then_loaded_strictly(tmp_path):
    state = _full_state(1)
    p = tmp_path / "s.pt"; torch.save(state, p)
    m = _Base()                                   # heads do not exist yet
    n_now = bc_warm_start_load(m, str(p))
    assert n_now == 2                             # trunk weight and bias
    assert set(m._bc_pending_state) == {k for k in state if k.startswith("q_heads.")}
    m.q_heads = _Heads()                          # the subclass builds them later ...
    n_later = bc_consume_pending(m, "q_heads", "q_heads.")
    assert n_later == 4
    for k, v in m.state_dict().items():           # ... and every tensor is the checkpoint's
        torch.testing.assert_close(v, state[k])
    assert m._bc_pending_state == {}


def test_a_plain_bc_file_without_heads_is_accepted_loudly(tmp_path, caplog):
    m = _Base()
    torch.save({k: v for k, v in m.state_dict().items()}, tmp_path / "s.pt")
    bc_warm_start_load(m, str(tmp_path / "s.pt"))
    m.q_heads = _Heads()
    with caplog.at_level("WARNING"):
        assert bc_consume_pending(m, "q_heads", "q_heads.") == 0
    assert "stays at its initialisation" in caplog.text


def test_unconsumed_deferred_keys_raise(tmp_path):
    m = _Base()
    m.BC_DEFERRED_PREFIXES = ("q_heads.", "other.")
    state = dict(m.state_dict()); state["other.w"] = torch.zeros(1)
    torch.save(state, tmp_path / "s.pt")
    bc_warm_start_load(m, str(tmp_path / "s.pt"))
    m.q_heads = _Heads()
    with pytest.raises(RuntimeError):
        bc_consume_pending(m, "q_heads", "q_heads.")


def test_keys_outside_the_deferred_prefixes_still_fail_strictly(tmp_path):
    m = _Base()
    state = dict(m.state_dict()); state["stray.w"] = torch.zeros(1)
    torch.save(state, tmp_path / "s.pt")
    with pytest.raises(RuntimeError):
        bc_warm_start_load(m, str(tmp_path / "s.pt"))
