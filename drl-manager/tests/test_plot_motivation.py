"""Tests for the motivation figure script."""
import importlib.util, sys
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "plot_motivation", Path(__file__).resolve().parents[1] / "scripts" / "plot_motivation.py")
pm = importlib.util.module_from_spec(_S); sys.modules["plot_motivation"] = pm
_S.loader.exec_module(pm)


def test_build_bars_keeps_order_and_skips_missing():
    labels, h, c = pm.build_bars({"timecap": 0.39, "shuffle": 0.52})
    assert h == [0.39, 0.52]                      # oracle skipped, order kept
    assert "TimeCAP" in labels[0] and "Corrupted" in labels[1]


def test_build_bars_all_present():
    labels, h, _ = pm.build_bars({"oracle": 0.34, "timecap": 0.39, "shuffle": 0.52})
    assert h == [0.34, 0.39, 0.52]


def test_render_writes_file(tmp_path):
    out = tmp_path / "m.pdf"
    pm.render({"timecap": 0.393, "shuffle": 0.52}, 0.464, "stochastic", out)
    assert out.exists() and out.stat().st_size > 0


def test_render_without_reference_line(tmp_path):
    out = tmp_path / "m.pdf"
    pm.render({"timecap": 0.393}, None, "stochastic", out)
    assert out.exists()


def test_render_empty_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        pm.render({"oracle": None}, None, "stochastic", tmp_path / "x.pdf")


def test_cli(tmp_path):
    out = tmp_path / "cli.pdf"
    rc = pm.main(["--timecap", "0.393", "--no-forecast", "0.464",
                  "--decode", "stochastic", "--out", str(out)])
    assert rc == 0 and out.exists()
