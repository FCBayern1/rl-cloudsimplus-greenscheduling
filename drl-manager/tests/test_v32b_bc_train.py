"""Unit tests for the V3.2B BC trainer's pure functions (loss masking,
dataset loading, defer metrics) - the module-level train loop is exercised
by the overnight chain itself."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v32b_bc_train import load_dataset, collate, masked_slot_ce, defer_metrics


def _fake_npz(path, T=3, slots=4):
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        obs_a=rng.normal(size=(T, slots)).astype(np.float32),
        obs_b=rng.normal(size=(T, 2)).astype(np.float32),
        actions=rng.integers(0, 3, size=(T, slots)).astype(np.int16),
        real_mask=np.array([[True, True, False, False]] * T),
    )


class TestLoadCollate:
    def test_roundtrip_and_alignment(self, tmp_path):
        _fake_npz(tmp_path / "teacher_ep000.npz")
        _fake_npz(tmp_path / "teacher_ep001.npz")
        steps, actions, mask = load_dataset(tmp_path)
        assert len(steps) == 6 and actions.shape == (6, 4)
        batch = collate(steps, [0, 5])
        assert batch["a"].shape == (2, 4) and batch["b"].shape == (2, 2)

    def test_empty_dir_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_dataset(tmp_path)


class TestMaskedCE:
    def test_padded_slots_never_contribute(self):
        B, S, O = 2, 3, 4
        logits = torch.randn(B, S * O)
        actions = torch.zeros(B, S, dtype=torch.long)
        m_all = torch.ones(B, S, dtype=torch.bool)
        m_first = torch.tensor([[True, False, False]] * B)
        # corrupt the padded slots' logits wildly: masked loss must not move
        loss1 = masked_slot_ce(logits, actions, m_first, S)
        logits2 = logits.clone()
        logits2.view(B, S, O)[:, 1:, :] = 100.0
        loss2 = masked_slot_ce(logits2, actions, m_first, S)
        assert torch.allclose(loss1, loss2)
        assert not torch.allclose(masked_slot_ce(logits, actions, m_all, S),
                                  loss1)

    def test_perfect_logits_low_loss(self):
        S, O = 2, 3
        actions = torch.tensor([[1, 2]])
        logits = torch.full((1, S * O), -10.0)
        logits.view(1, S, O)[0, 0, 1] = 10.0
        logits.view(1, S, O)[0, 1, 2] = 10.0
        m = torch.ones(1, S, dtype=torch.bool)
        assert masked_slot_ce(logits, actions, m, S) < 1e-3


class TestDeferMetrics:
    def test_recall_counts_only_real_slots(self):
        S, O, defer = 2, 3, 2
        actions = torch.tensor([[defer, defer]])
        logits = torch.full((1, S * O), 0.0)
        logits.view(1, S, O)[0, 0, defer] = 5.0     # predicts defer on slot 0
        m = torch.tensor([[True, False]])           # slot 1 is padding
        acc, prec, rec = defer_metrics(logits, actions, m, S, defer)
        assert rec == 1.0 and acc == 1.0
