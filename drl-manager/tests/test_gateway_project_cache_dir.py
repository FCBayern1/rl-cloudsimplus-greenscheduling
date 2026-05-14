"""
Regression test for _launch_java_gateway's GRADLE_PROJECT_CACHE_DIR hook.

Why this exists:
  Running the A1 ablation as a SLURM job array, every task but one crashed
  during gateway startup with

      Timeout waiting to lock file hash cache
      (.../cloudsimplus-gateway/.gradle/8.14/fileHashes)

  because the project-local <gateway>/.gradle directory sits on shared
  Lustre and Gradle's lockfiles aren't cross-node safe. The fix injects
  ``--project-cache-dir`` into the gradlew command from the
  ``GRADLE_PROJECT_CACHE_DIR`` env var so each array task can point at
  node-local storage. This test pins that contract so the flag does not
  silently disappear from the cmd in a future refactor.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gym_cloudsimplus.envs import hierarchical_multidc_env as env_mod


class _FakeProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return None


def _capture_popen_cmd(monkeypatch, tmp_path, env_value):
    """Drive _launch_java_gateway just far enough to capture the gradlew cmd."""
    captured: dict = {}

    fake_gateway = tmp_path / "cloudsimplus-gateway"
    fake_gateway.mkdir()
    (fake_gateway / "gradlew").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(fake_gateway / "gradlew", 0o755)

    monkeypatch.chdir(tmp_path)

    if env_value is None:
        monkeypatch.delenv("GRADLE_PROJECT_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("GRADLE_PROJECT_CACHE_DIR", str(env_value))

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(env_mod.subprocess, "Popen", fake_popen)

    # _launch_java_gateway opens a log file in self.config["gateway_log_dir"]
    # and reads self.java_process — minimum surface to reach the cmd build.
    instance = MagicMock()
    instance.config = {"gateway_log_dir": str(tmp_path / "logs")}
    instance._java_log_file = None

    with pytest.raises(RuntimeError, match="stop after capture"):
        env_mod.HierarchicalMultiDCEnv._launch_java_gateway(instance, port=12345)

    return captured


def test_project_cache_dir_injected_when_env_set(monkeypatch, tmp_path):
    cache_dir = tmp_path / "node_local_cache"
    captured = _capture_popen_cmd(monkeypatch, tmp_path, cache_dir)

    cmd = captured["cmd"]
    assert "--project-cache-dir" in cmd, f"expected flag missing: {cmd}"
    idx = cmd.index("--project-cache-dir")
    assert cmd[idx + 1] == str(cache_dir), f"flag value wrong: {cmd[idx + 1]!r}"
    # Directory must be auto-created so gradle does not refuse to write to it.
    assert cache_dir.is_dir()


def test_project_cache_dir_omitted_when_env_unset(monkeypatch, tmp_path):
    captured = _capture_popen_cmd(monkeypatch, tmp_path, env_value=None)
    cmd = captured["cmd"]
    assert "--project-cache-dir" not in cmd, (
        f"flag must not appear when GRADLE_PROJECT_CACHE_DIR is unset: {cmd}"
    )
