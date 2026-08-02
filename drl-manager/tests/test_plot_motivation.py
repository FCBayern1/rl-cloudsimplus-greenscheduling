"""Tests for the motivation figure script."""
import importlib.util, sys
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "plot_motivation", Path(__file__).resolve().parents[1] / "scripts" / "plot_motivation.py")
pm = importlib.util.module_from_spec(_S); sys.modules["plot_motivation"] = pm
_S.loader.exec_module(pm)


def test_build_bars_keeps_order_and_skips_missing():
    labels, h, c, _e = pm.build_bars({"timecap": 0.39, "shuffle": 0.52})
    assert h == [0.39, 0.52]                      # oracle skipped, order kept
    assert "TimeCAP" in labels[0] and "Corrupted" in labels[1]


def test_build_bars_all_present():
    labels, h, _f, _e = pm.build_bars({"oracle": 0.34, "timecap": 0.39, "shuffle": 0.52})
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


def test_build_bars_no_forecast_leads_when_present():
    labels, h, fills, edges = pm.build_bars(
        {"no_forecast": 0.464, "timecap": 0.397, "shuffle": 0.478})
    assert h == [0.464, 0.397, 0.478]             # grey baseline bar first
    assert "No" in labels[0]
    assert fills[0] == "#f5f5f5" and edges[0] == "#666666"


def test_cli_no_forecast_as_bar(tmp_path):
    out = tmp_path / "m3.png"
    rc = pm.main(["--timecap", "0.397", "--shuffle", "0.478",
                  "--no-forecast", "0.464", "--no-forecast-as-bar",
                  "--out", str(out)])
    assert rc == 0 and out.exists() and out.stat().st_size > 0


def test_cli_no_forecast_as_bar_requires_value(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        pm.main(["--timecap", "0.397", "--no-forecast-as-bar",
                 "--out", str(tmp_path / "x.png")])
