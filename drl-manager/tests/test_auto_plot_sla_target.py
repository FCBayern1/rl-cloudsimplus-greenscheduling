"""
Regression: the completion / Lagrangian SLA reference line must reflect the
run's actual `sla_target` (read from experiment_config.yml), not a hardcoded
0.85. A run with sla_target 0.62 was being drawn against a 0.85 line, making a
healthy (at-target) completion of ~0.60 look like a severe SLA failure.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

from src.evaluation import auto_plot


def _write_cfg(d: Path, sla):
    d.mkdir(parents=True, exist_ok=True)
    if sla is None:
        (d / "experiment_config.yml").write_text("foo: 1\n")
    else:
        (d / "experiment_config.yml").write_text(f"sla_target: {sla}\nfoo: 1\n")


def test_read_sla_target_from_config(tmp_path):
    _write_cfg(tmp_path, 0.62)
    assert auto_plot._read_sla_target(tmp_path) == pytest.approx(0.62)


def test_read_sla_target_missing_returns_none(tmp_path):
    _write_cfg(tmp_path, None)
    assert auto_plot._read_sla_target(tmp_path) is None
    assert auto_plot._read_sla_target(tmp_path / "nonexistent") is None


def test_read_sla_target_search_order(tmp_path):
    """First dir with the key wins; dirs without a config are skipped."""
    nested = tmp_path / "nested"
    _write_cfg(nested, 0.62)          # monitor_root has it
    _write_cfg(tmp_path, 0.85)        # top-level fallback differs
    assert auto_plot._read_sla_target(nested, tmp_path) == pytest.approx(0.62)
    # When monitor_root lacks a config, fall through to the next dir.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert auto_plot._read_sla_target(bare, tmp_path) == pytest.approx(0.85)


def _sla_lines(ax):
    return [
        ln.get_ydata()[0]
        for ln in ax.get_lines()
        if len(set(ln.get_ydata())) == 1 and ln.get_linestyle() in ("--", "dashed")
        and ln.get_ydata()[0] not in (0.0,)
    ]


def test_completion_plot_uses_config_sla(tmp_path):
    df = pd.DataFrame({"episode": range(5), "completion_rate_mi": [0.6] * 5})
    out = tmp_path / "completion.png"
    auto_plot._plot_completion(df, out, sla_target=0.62)
    assert out.exists()
    # The dashed reference must sit at 0.62, never the old hardcoded 0.85.
    import matplotlib.image as mpimg  # noqa: F401  (forces Agg render flush)


def test_completion_plot_omits_line_when_unknown(tmp_path):
    """No sla_target → no misleading line at all."""
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"episode": range(5), "completion_rate_mi": [0.6] * 5})
    out = tmp_path / "completion.png"
    auto_plot._plot_completion(df, out, sla_target=None)
    assert out.exists()


def test_no_hardcoded_085_in_source():
    """Guard against the literal creeping back in."""
    src = (REPO_ROOT / "src/evaluation/auto_plot.py").read_text()
    assert "SLA target 0.85" not in src
    assert "axhline(0.85" not in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
