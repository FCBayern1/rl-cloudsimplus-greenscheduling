"""Tests for the EU-CRD mechanism-diagnostics figure script."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "plot_crd_mechanism",
    Path(__file__).resolve().parents[1] / "scripts" / "plot_crd_mechanism.py",
)
plot_crd_mechanism = importlib.util.module_from_spec(_SPEC)
sys.modules["plot_crd_mechanism"] = plot_crd_mechanism
_SPEC.loader.exec_module(plot_crd_mechanism)

STEP_COL = plot_crd_mechanism.STEP_COL
COLS = plot_crd_mechanism.COLS


def _df(rows=4, **overrides):
    data = {
        STEP_COL: [8000 * (i + 1) for i in range(rows)],
        COLS["c_t"]: [0.55, 0.58, 0.60, 0.61][:rows],
        COLS["rho_routing"]: [0.89, 0.90, 0.88, 0.89][:rows],
        COLS["rho_forecast"]: [0.11, 0.10, 0.12, 0.11][:rows],
        COLS["sigma2"]: [5.9, 4.1, 3.2, 2.7][:rows],
        COLS["tau"]: [9.6, 6.5, 5.0, 4.4][:rows],
        COLS["dq"]: [-0.002, 0.02, -0.01, -0.014][:rows],
        COLS["reweight"]: [0.0, 0.0, 1.0, 1.0][:rows],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_load_run_reads_csv(tmp_path):
    p = tmp_path / "progress.csv"
    _df().to_csv(p, index=False)
    got = plot_crd_mechanism.load_run(p)
    assert len(got) == 4


def test_load_run_rejects_csv_without_step_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"something": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing"):
        plot_crd_mechanism.load_run(p)


def test_extract_scales_steps_to_1e5_units():
    x, y = plot_crd_mechanism.extract(_df(), "c_t")
    assert x == pytest.approx([0.08, 0.16, 0.24, 0.32])
    assert y == pytest.approx([0.55, 0.58, 0.60, 0.61])


def test_extract_drops_nan_rows():
    df = _df()
    df.loc[1, COLS["dq"]] = float("nan")
    x, y = plot_crd_mechanism.extract(df, "dq")
    assert len(x) == 3 and len(y) == 3
    # the NaN row (step 16000 -> 0.16) is gone, the others survive
    assert x == pytest.approx([0.08, 0.24, 0.32])


def test_extract_unknown_metric_returns_empty():
    assert plot_crd_mechanism.extract(_df(), "does_not_exist") == ([], [])


def test_extract_accepts_raw_column_name():
    x, _ = plot_crd_mechanism.extract(_df(), COLS["c_t"])
    assert len(x) == 4


def test_find_warmup_end_returns_first_step_where_reweight_on():
    assert plot_crd_mechanism.find_warmup_end(_df()) == pytest.approx(0.24)


def test_find_warmup_end_none_when_never_on():
    df = _df()
    df[COLS["reweight"]] = 0.0
    assert plot_crd_mechanism.find_warmup_end(df) is None


def test_find_warmup_end_none_when_column_missing():
    df = _df().drop(columns=[COLS["reweight"]])
    assert plot_crd_mechanism.find_warmup_end(df) is None


def test_render_writes_file(tmp_path):
    out = tmp_path / "sub" / "fig.pdf"
    got = plot_crd_mechanism.render([_df(), _df()], ["seed 1", "seed 2"], out)
    assert got.exists() and got.stat().st_size > 0


def test_render_tolerates_run_missing_crd_columns(tmp_path):
    bare = pd.DataFrame({STEP_COL: [8000, 16000]})
    out = tmp_path / "fig.pdf"
    plot_crd_mechanism.render([_df(), bare], ["crd", "ablation"], out)
    assert out.exists()


def test_cli_end_to_end(tmp_path):
    csv = tmp_path / "progress.csv"
    _df().to_csv(csv, index=False)
    out = tmp_path / "cli.pdf"
    rc = plot_crd_mechanism.main(["--run", str(csv), "--label", "s1", "--out", str(out)])
    assert rc == 0 and out.exists()


def test_cli_returns_error_when_no_match(tmp_path):
    rc = plot_crd_mechanism.main(["--run", str(tmp_path / "nope_*.csv")])
    assert rc == 1


def test_smooth_preserves_length_and_starts_at_first_value():
    vals = [1.0, 2.0, 3.0, 10.0]
    out = plot_crd_mechanism.smooth(vals, span=3)
    assert len(out) == len(vals)
    assert out[0] == pytest.approx(1.0)
    # EMA lags a jump instead of following it fully
    assert out[-1] < 10.0


def test_smooth_empty_is_empty():
    assert plot_crd_mechanism.smooth([]) == []


def _df_with_dispersion(rows=4):
    df = _df(rows)
    df[COLS["rho_p10"]] = [0.55, 0.60, 0.62, 0.63][:rows]
    df[COLS["rho_p90"]] = [0.99, 0.99, 0.98, 0.98][:rows]
    df[COLS["w_std"]] = [0.20, 0.18, 0.15, 0.14][:rows]
    return df


def test_has_dispersion_detects_new_and_old_runs():
    assert plot_crd_mechanism.has_dispersion(_df_with_dispersion())
    assert not plot_crd_mechanism.has_dispersion(_df())


def test_render_with_dispersion_band(tmp_path):
    out = tmp_path / "band.pdf"
    plot_crd_mechanism.render([_df_with_dispersion()], ["seed 4"], out)
    assert out.exists() and out.stat().st_size > 0


def test_render_mixed_old_and_new_runs(tmp_path):
    out = tmp_path / "mixed.pdf"
    plot_crd_mechanism.render([_df(), _df_with_dispersion()], ["s1", "s4"], out)
    assert out.exists()
