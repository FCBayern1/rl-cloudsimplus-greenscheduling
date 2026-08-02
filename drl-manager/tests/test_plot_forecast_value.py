"""Tests for the forecast-value figure script."""
import importlib.util, sys
from pathlib import Path

import pytest

_S = importlib.util.spec_from_file_location(
    "plot_forecast_value",
    Path(__file__).resolve().parents[1] / "scripts" / "plot_forecast_value.py")
pfv = importlib.util.module_from_spec(_S); sys.modules["plot_forecast_value"] = pfv
_S.loader.exec_module(pfv)


def test_parse_seeds():
    assert pfv.parse_seeds("0.09,0.07, 0.14") == [0.09, 0.07, 0.14]


def test_parse_seeds_empty_raises():
    with pytest.raises(ValueError):
        pfv.parse_seeds(" , ")


def test_render_two_panels(tmp_path):
    out = tmp_path / "fv.png"
    pfv.render([0.0916, 0.0704, 0.1441], [0.0940, 0.0820, 0.0855],
               [100, 99.6, 100], [91.4, 100, 100], out)
    assert out.exists() and out.stat().st_size > 0


def test_render_carbon_only(tmp_path):
    out = tmp_path / "fv1.png"
    pfv.render([0.10, 0.12], [0.08, 0.09], None, None, out)
    assert out.exists() and out.stat().st_size > 0


def test_cli(tmp_path):
    out = tmp_path / "fv2.png"
    rc = pfv.main(["--without", "0.0916,0.0704,0.1441",
                   "--with", "0.0940,0.0820,0.0855",
                   "--without-comp", "100,99.6,100",
                   "--with-comp", "91.4,100,100",
                   "--out", str(out)])
    assert rc == 0 and out.exists()
