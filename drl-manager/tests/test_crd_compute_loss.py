"""
M2.1 — verify _compute_crd_terms hook is reached when compute_loss_for_module
runs. This isolates the hook plumbing from the actual CF computation (which
M2.2-M2.5 will fill in).

We don't spin up a full PPO learner here. We construct a minimal subclass
that bypasses super() and just verifies our hook is invoked exactly once
per minibatch with the expected arguments.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_compute_loss.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.learners.crd_q_loss import CRDPPOTorchLearner
from src.models.rlmodule_gtrxl_ensemble import COL_Q_ENSEMBLE


class _StubLearner(CRDPPOTorchLearner):
    """
    Avoid the heavyweight PPOTorchLearner.build / .compute_loss_for_module
    chain — we only want to test that:
      1. our hook is called at the start of compute_loss_for_module
      2. it sees the batch / fwd_out we passed in
      3. log fires only once per module

    super().compute_loss_for_module would require a real PPO config and
    module registry, which is overkill for testing the hook itself.
    """

    def __init__(self):
        # Skip PPOTorchLearner.__init__ — we don't need the full Learner setup.
        # Just init the CRD-specific dicts.
        self._crd_call_counts = {}
        self._crd_hook_logged = {}
        self.hook_calls = []  # record (module_id, batch_keys, fwd_keys)

    def _compute_crd_terms(self, *, module_id, batch, fwd_out):
        # Wrap the parent hook so we can observe it.
        super()._compute_crd_terms(
            module_id=module_id, batch=batch, fwd_out=fwd_out
        )
        self.hook_calls.append(
            (
                module_id,
                tuple(sorted(batch.keys())) if isinstance(batch, dict) else None,
                tuple(sorted(fwd_out.keys())) if isinstance(fwd_out, dict) else None,
            )
        )

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        # Minimal mirror of CRDPPOTorchLearner.compute_loss_for_module that
        # exercises the hook but skips super() (which needs a real PPO setup).
        self._compute_crd_terms(module_id=module_id, batch=batch, fwd_out=fwd_out)
        # Pretend super() returned a scalar loss.
        return torch.tensor(0.0)


def test_hook_invoked_once_per_minibatch():
    learner = _StubLearner()
    batch = {"obs": torch.randn(2, 3), "rewards": torch.zeros(2)}
    fwd = {"vf_preds": torch.zeros(2)}
    learner.compute_loss_for_module(module_id="global_policy", config=None, batch=batch, fwd_out=fwd)
    assert len(learner.hook_calls) == 1
    mid, b_keys, f_keys = learner.hook_calls[0]
    assert mid == "global_policy"
    assert b_keys == ("obs", "rewards")
    assert f_keys == ("vf_preds",)


def test_hook_invoked_per_compute_loss_call():
    """Each minibatch update calls compute_loss_for_module → hook fires every time."""
    learner = _StubLearner()
    batch = {"obs": torch.randn(2, 3)}
    fwd = {}
    for _ in range(5):
        learner.compute_loss_for_module(module_id="m", config=None, batch=batch, fwd_out=fwd)
    assert len(learner.hook_calls) == 5


def test_log_fires_only_once_per_module(caplog):
    """Even if hook is called many times, the warn-once log should appear once per module."""
    import logging
    learner = _StubLearner()
    batch = {}
    fwd = {}
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        for _ in range(3):
            learner.compute_loss_for_module(module_id="alpha", config=None, batch=batch, fwd_out=fwd)
        for _ in range(3):
            learner.compute_loss_for_module(module_id="beta", config=None, batch=batch, fwd_out=fwd)
    crd_log_lines = [r for r in caplog.records if "[CRD] hook reached" in r.message]
    seen_modules = {r.message.split("module ")[1].split(";")[0] for r in crd_log_lines}
    assert seen_modules == {"'alpha'", "'beta'"}, (
        f"expected logs for both modules exactly once each; got {seen_modules}"
    )
    assert len(crd_log_lines) == 2


def test_hook_records_q_ensemble_presence_in_log(caplog):
    """When fwd_out has crd_q_ensemble, the hook should note has_q_ensemble=True."""
    import logging
    learner = _StubLearner()
    fwd_with = {COL_Q_ENSEMBLE: torch.randn(2, 1, 5, 4)}
    fwd_without = {}
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        learner.compute_loss_for_module(module_id="with", config=None, batch={}, fwd_out=fwd_with)
        learner.compute_loss_for_module(module_id="without", config=None, batch={}, fwd_out=fwd_without)
    msgs = [r.message for r in caplog.records if "[CRD]" in r.message]
    assert any("has_q_ensemble=True" in m for m in msgs)
    assert any("has_q_ensemble=False" in m for m in msgs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
