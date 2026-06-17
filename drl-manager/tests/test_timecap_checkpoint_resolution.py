"""
Regression test for HierarchicalMultiDCEnv._resolve_timecap_checkpoint.

Bug: timecap.checkpoint was passed to TimeCAP_GreenPredictor verbatim, so a
relative path only resolved when the process cwd happened to be drl-manager.
Inside Ray worker actors the cwd is the Ray session dir, so loading failed with
`FileNotFoundError: TimeCAP checkpoint not found: timecap_prediction/...`.

The resolver now turns a relative path into an existing absolute file by
searching known base dirs (drl-manager, repo root, cwd), independent of the
process cwd.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

_resolve = HierarchicalMultiDCEnv._resolve_timecap_checkpoint


def test_absolute_existing_path_returned_as_is(tmp_path):
    ckpt = tmp_path / "ckpt_best.pth"
    ckpt.write_bytes(b"x")
    assert _resolve(str(ckpt)) == ckpt


def test_absolute_missing_path_raises(tmp_path):
    missing = tmp_path / "nope.pth"
    with pytest.raises(FileNotFoundError):
        _resolve(str(missing))


def test_relative_path_resolves_against_search_dir(tmp_path):
    # File lives under a base dir; config holds only the relative tail.
    rel = Path("sub/ckpt_best.pth")
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    other = tmp_path / "elsewhere"
    other.mkdir()

    # `other` first, real base second -> must skip `other` and find it in tmp_path.
    resolved = _resolve(str(rel), search_dirs=[other, tmp_path])
    assert resolved == target
    assert resolved.is_absolute()


def test_relative_path_not_found_lists_candidates(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        _resolve("does/not/exist.pth", search_dirs=[tmp_path])
    msg = str(exc.value)
    assert "does/not/exist.pth" in msg
    assert str(tmp_path) in msg  # candidate base is reported for debugging


def test_real_config_relative_checkpoint_resolves_independent_of_cwd(monkeypatch):
    """The exact relative path used in config.yml must resolve via the default
    search dirs (drl-manager / repo root) even when cwd is unrelated."""
    rel = ("timecap_prediction/TimeCAP/model/"
           "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth")
    drl_manager = Path(__file__).resolve().parents[1]
    if not (drl_manager / rel).is_file():
        pytest.skip("TimeCAP checkpoint not present in this checkout")

    monkeypatch.chdir(Path(drl_manager.anchor))  # cwd = filesystem root, not drl-manager
    resolved = _resolve(rel)
    assert resolved == drl_manager / rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
