"""
Regression test for GreenEnergyLoggerCallback.on_learn_on_batch.

Bug: the SampleBatch branch built its log label with `policy.id`, but RLlib
`Policy` objects have no `.id` attribute. Because the label was evaluated
*outside* the inner `_check_and_log_seq_lens` try/except, this raised
`AttributeError: 'PPOTorchPolicy' object has no attribute 'id'` on the very
first minibatch SGD step (do_minibatch_sgd calls learn_on_batch per policy
with a plain SampleBatch), crashing training.

This reproduces the crash with a minimal fake policy + SampleBatch and
verifies the callback now runs without raising.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray.rllib.policy.sample_batch import SampleBatch  # noqa: E402
from src.callbacks import rllib_green_energy_logger as LOG  # noqa: E402


class _FakePolicy:
    """Mimics an RLlib Policy: notably it has NO `.id` attribute."""
    pass


def _make_callback():
    cb = LOG.GreenEnergyLoggerCallback.__new__(LOG.GreenEnergyLoggerCallback)
    return cb


def test_on_learn_on_batch_with_sample_batch_does_not_raise():
    cb = _make_callback()
    batch = SampleBatch(
        {
            SampleBatch.SEQ_LENS: np.array([2, 2], dtype=np.int32),
            "obs": np.zeros((4, 3), dtype=np.float32),
        }
    )
    # Before the fix this raised AttributeError on `policy.id`.
    cb.on_learn_on_batch(policy=_FakePolicy(), train_batch=batch, result={})


def test_on_learn_on_batch_label_falls_back_to_class_name():
    """The label must degrade gracefully when the policy has no id.

    Whether the batch is judged valid or invalid, the callback emits `msg_base`
    (which embeds the label) — only the log level differs. Capture every level
    via a handler so the assertion does not depend on SampleBatch.count
    semantics for the valid/invalid branch.
    """
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    LOG.logger.addHandler(handler)
    prev_level = LOG.logger.level
    LOG.logger.setLevel(logging.DEBUG)

    cb = _make_callback()
    batch = SampleBatch(
        {
            SampleBatch.SEQ_LENS: np.array([2, 2], dtype=np.int32),
            "obs": np.zeros((4, 3), dtype=np.float32),
        }
    )
    try:
        cb.on_learn_on_batch(policy=_FakePolicy(), train_batch=batch, result={})
    finally:
        LOG.logger.removeHandler(handler)
        LOG.logger.setLevel(prev_level)

    joined = "\n".join(records)
    assert "_FakePolicy" in joined, f"expected class-name label in logs, got: {joined!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
