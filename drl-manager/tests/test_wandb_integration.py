"""
Unit tests for the wandb integration helper.

Run from drl-manager/:
    .venv/bin/python -m pytest tests/test_wandb_integration.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.wandb_integration import (
    build_wandb_callbacks,
    get_wandb_config,
    upload_run_artifacts,
)


def test_defaults_off():
    """No wandb block → disabled, callback list is empty."""
    cfg = get_wandb_config({})
    assert cfg["enabled"] is False
    assert cfg["project"]  # default project is set
    callbacks = build_wandb_callbacks(
        {}, experiment_name="exp", run_name="run-1"
    )
    assert callbacks == []


def test_enabled_returns_wandb_callback():
    """When enabled and wandb is installed, we get a WandbLoggerCallback."""
    pytest.importorskip("wandb")
    from ray.air.integrations.wandb import WandbLoggerCallback

    env_config = {
        "wandb": {
            "enabled": True,
            "project": "unit-test-project",
            "entity": "",
            "tags": ["a"],
            "mode": "offline",  # avoid network in CI
        }
    }
    callbacks = build_wandb_callbacks(
        env_config, experiment_name="exp1", run_name="run-x"
    )
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], WandbLoggerCallback)
    assert callbacks[0].project == "unit-test-project"


def test_disabled_mode_yields_no_callback():
    env_config = {
        "wandb": {"enabled": True, "project": "x", "mode": "disabled"}
    }
    assert build_wandb_callbacks(
        env_config, experiment_name="exp", run_name="r"
    ) == []


def test_explicit_disabled_yields_no_callback():
    env_config = {"wandb": {"enabled": False, "project": "x"}}
    assert build_wandb_callbacks(
        env_config, experiment_name="exp", run_name="r"
    ) == []


def test_upload_run_artifacts_disabled_is_noop(tmp_path):
    """Should not import wandb when disabled."""
    (tmp_path / "monitor.csv").write_text("step,reward\n0,1\n")
    # Should not raise even with no wandb run alive
    upload_run_artifacts(
        {"wandb": {"enabled": False}},
        str(tmp_path),
        experiment_name="exp",
        run_name="r",
    )


def test_upload_run_artifacts_log_artifacts_false(tmp_path):
    """log_artifacts=false bypasses the upload path entirely."""
    (tmp_path / "monitor.csv").write_text("step,reward\n0,1\n")
    upload_run_artifacts(
        {
            "wandb": {
                "enabled": True,
                "project": "x",
                "log_artifacts": False,
                "mode": "offline",
            }
        },
        str(tmp_path),
        experiment_name="exp",
        run_name="r",
    )


def test_tags_appended_with_experiment_name():
    """The experiment_name is auto-appended to the tag list passed to wandb."""
    pytest.importorskip("wandb")
    callbacks = build_wandb_callbacks(
        {
            "wandb": {
                "enabled": True,
                "project": "x",
                "tags": ["foo", "bar"],
                "mode": "offline",
            }
        },
        experiment_name="my_exp_2026",
        run_name="r",
    )
    assert len(callbacks) == 1
    init_tags = callbacks[0].kwargs.get("tags", [])
    assert "foo" in init_tags
    assert "bar" in init_tags
    assert "my_exp_2026" in init_tags
