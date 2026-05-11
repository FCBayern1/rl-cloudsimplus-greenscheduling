"""
Regression test: HierarchicalMultiDCEnv.close() must silence py4j's own
ERROR-level loggers so that post-shutdown ConnectionResetError noise
(emitted by py4j background threads / Java-proxy finalizers after we have
explicitly closed the gateway and killed the JVM) does not pollute the
evaluation/test output.

Background:
  Running `python -m src.baselines.evaluate ... --episodes N` always
  produced a misleading

      ERROR - Exception while sending command.
      ConnectionResetError: [Errno 104] Connection reset by peer

  ~400 ms after the run had already saved its CSV and printed its summary.
  The error is benign (py4j's callback-server thread reading from a socket
  whose other end — the JVM — has just been killed by SIGTERM), but it
  looks like a failure to a casual reader.  close() now lifts py4j's
  loggers to CRITICAL after the gateway client is closed but *before* we
  send SIGTERM, so the noise is suppressed without hiding real problems
  that occur during normal operation.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def restore_py4j_log_levels():
    """close() mutates module-global py4j logger levels — restore around the test."""
    names = ["py4j.java_gateway", "py4j.clientserver"]
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def _build_env_skeleton():
    """
    Build the bare minimum of HierarchicalMultiDCEnv state needed to invoke
    close() without going through the full constructor (which spawns a JVM).
    """
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

    env = HierarchicalMultiDCEnv.__new__(HierarchicalMultiDCEnv)
    env.java_env = MagicMock()
    env.gateway = MagicMock()
    env.java_process = None
    return env


def test_close_silences_py4j_loggers(restore_py4j_log_levels):
    """After close(), py4j's own loggers must be at CRITICAL."""
    # Pre-condition: py4j loggers are at a chatty level (default is WARNING but
    # tests / scripts often set INFO).
    logging.getLogger("py4j.java_gateway").setLevel(logging.INFO)
    logging.getLogger("py4j.clientserver").setLevel(logging.INFO)

    env = _build_env_skeleton()
    env.close()

    assert logging.getLogger("py4j.java_gateway").level == logging.CRITICAL
    assert logging.getLogger("py4j.clientserver").level == logging.CRITICAL


def test_close_does_not_silence_unrelated_loggers(restore_py4j_log_levels):
    """The fix must not bleed into other loggers — only py4j is suppressed."""
    other = logging.getLogger("gym_cloudsimplus.envs.hierarchical_multidc_env")
    other.setLevel(logging.INFO)

    env = _build_env_skeleton()
    env.close()

    assert other.level == logging.INFO, (
        "close() must only mute py4j; the env's own logger must remain untouched"
    )


def test_close_calls_gateway_and_java_env_close(restore_py4j_log_levels):
    """The silencing must not skip the actual cleanup work."""
    env = _build_env_skeleton()
    java_env_mock = env.java_env
    gateway_mock = env.gateway

    env.close()

    java_env_mock.close.assert_called_once()
    gateway_mock.close.assert_called_once()
    # Should null out references after close
    assert env.gateway is None
    assert env.java_env is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
