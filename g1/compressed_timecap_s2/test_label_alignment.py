"""T1-T5 of the retrain prereg §2.2: plan A label alignment.

One file, five tests, named exactly as the prereg names them, so a reviewer can map the
frozen requirement to the check without hunting.

What plan A fixes: the k=0 audit established that training builds `y[0]` as the row AFTER
the last history row (`label_len = 0` ⇒ `r_begin = s_end`), while the deployed consumer
treats `forecast[0]` as the CURRENT row -- Java's future window sums `series[i]` from
`i = currentIdx`, inclusive, and the Python provider asks for the forecast right after
`update(t)` has pushed row `t`. Plan A moves the training label back one row so the two
name the same row. `label_start_offset = 1` is that move.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_dataset as cd  # noqa: E402
import train_timecap_clean as tc  # noqa: E402

SEQ, PRED = 8, 12
REPO = cd._REPO
JAVA_PROVIDER = os.path.join(
    REPO, "cloudsimplus-gateway/src/main/java/exe/edu/cspg/energy/GreenEnergyProvider.java")
PY_PROVIDER = os.path.join(REPO, "drl-manager/src/prediction/timecap_godeye_provider.py")
PY_PREDICTOR = os.path.join(REPO, "drl-manager/timecap_prediction/predictor.py")


def _write(tmp, turbine, year, n):
    """Rows carry their own index in every feature, so an absolute row is identifiable
    from the value alone and an off-by-one cannot hide."""
    cols = cd.feature_columns()
    data = {c: np.arange(n, dtype=float) for c in cols}
    data["TurbID"] = turbine
    data["Tmstamp"] = pd.date_range("2020-01-01", periods=n, freq="10min")
    p = os.path.join(tmp, f"Turbine_{turbine}_{year}.csv")
    pd.DataFrame(data).to_csv(p, index=False)
    return p


@pytest.fixture()
def files(tmp_path):
    d = str(tmp_path)
    _write(d, 1, 2020, 400)
    return cd.file_specs([1], [2020], split_dir=d)


# ---------------------------------------------------------------------------------------
# T1  label_start_offset=1 => y[0] is exactly the row x[-1] sits on
# ---------------------------------------------------------------------------------------
def test_T1_plan_a_label_starts_on_the_last_history_row(files):
    ds = cd.CleanWindowDataset(files, "train", SEQ, PRED, scale=False,
                               label_start_offset=1)
    assert len(ds) > 0
    for i in (0, 1, len(ds) // 2, len(ds) - 1):
        x, y = ds[i]
        assert float(y[0, 0]) == pytest.approx(float(x[-1, 0])), \
            "plan A: y[0] must be the last history row itself"
        name, lrow = ds.label_first_row(i)
        hname, hrow = ds.history_last_row(i)
        assert (name, lrow) == (hname, hrow)


# ---------------------------------------------------------------------------------------
# T2  label_start_offset=0 reproduces the stock convention bit for bit
# ---------------------------------------------------------------------------------------
def test_T2_offset_zero_is_the_stock_convention(files):
    ds = cd.CleanWindowDataset(files, "train", SEQ, PRED, scale=False,
                               label_start_offset=0)
    for i in (0, 3, len(ds) - 1):
        x, y = ds[i]
        assert float(y[0, 0]) == pytest.approx(float(x[-1, 0]) + 1.0), \
            "offset 0 must keep y[0] one row after the history"
    assert cd.LABEL_START_OFFSET == 0, "the module default must stay the stock convention"
    default = cd.CleanWindowDataset(files, "train", SEQ, PRED, scale=False)
    assert np.array_equal(default[0][1], ds[0][1])


# ---------------------------------------------------------------------------------------
# T3  span and window count follow the border arithmetic under both offsets
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("off", [0, 1])
def test_T3_span_and_window_count(files, off):
    ds = cd.CleanWindowDataset(files, "train", SEQ, PRED, label_start_offset=off)
    assert ds.span == SEQ + PRED - off
    for split in cd.SPLITS:
        d = cd.CleanWindowDataset(files, split, SEQ, PRED, label_start_offset=off)
        want = sum(max(0, (hi - lo) - d.span + 1)
                   for name in d.borders for lo, hi in [d.borders[name][split]])
        assert len(d) == want
        for i in range(len(d)):
            _, start, end = d.window_rows(i)
            assert end - start == d.span


def test_T3b_offset_must_stay_inside_the_history(files):
    with pytest.raises(ValueError, match="label_start_offset must lie"):
        cd.CleanWindowDataset(files, "train", SEQ, PRED, label_start_offset=SEQ)
    with pytest.raises(ValueError, match="label_start_offset must lie"):
        cd.CleanWindowDataset(files, "train", SEQ, PRED, label_start_offset=-1)


# ---------------------------------------------------------------------------------------
# T4  boundaries stay clean under both offsets
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("off", [0, 1])
def test_T4_no_window_crosses_a_file_or_split_boundary(tmp_path, off):
    d = str(tmp_path)
    _write(d, 1, 2020, 400)
    _write(d, 2, 2020, 400)
    fs = cd.file_specs([1, 2], [2020], split_dir=d)
    for split in cd.SPLITS:
        ds = cd.CleanWindowDataset(fs, split, SEQ, PRED, label_start_offset=off)
        a = ds.audit()
        assert a["cross_file_windows"] == 0 and a["cross_split_windows"] == 0
        assert a["split_row_overlaps"] == []
        assert a["label_start_offset"] == off and a["span"] == SEQ + PRED - off
        for i in range(len(ds)):
            name, start, end = ds.window_rows(i)
            lo, hi = ds.borders[name][split]
            assert lo <= start and end <= hi


# ---------------------------------------------------------------------------------------
# T5  end-to-end: the row training calls y[0] is the row deployment reads first
# ---------------------------------------------------------------------------------------
def _java_future_window_starts_at_current_row():
    """Java sums series[i] from i = currentIdx, i.e. offset 0 from the current row."""
    src = open(JAVA_PROVIDER).read()
    body = src[src.index("computeFutureTrendFeatures"):]
    body = body[:body.index("private int simTimeToRowIndex")]
    short_loop = re.search(r"for\s*\(int i\s*=\s*(currentIdx[^;]*);\s*i\s*<\s*shortEndIdx",
                           body)
    long_loop = re.search(r"for\s*\(int i\s*=\s*(currentIdx[^;]*);\s*i\s*<\s*longEndIdx",
                          body)
    assert short_loop and long_loop, "the Java future-window loops moved; re-audit k=0"
    starts = {short_loop.group(1).strip(), long_loop.group(1).strip()}
    assert starts == {"currentIdx"}, \
        f"Java's future window no longer starts at the current row: {starts}"
    assert "series[currentIdx]" in body, "short_trend no longer anchors on the current row"
    return 0


def _python_consumption_offset_from_the_pushed_row():
    """update(t) pushes row t, and the features are taken from forecast[0] onward, so the
    consumer treats forecast[0] as the row it just pushed: offset 0."""
    prov = open(PY_PROVIDER).read()
    pred = open(PY_PREDICTOR).read()
    assert "self.predictor.update(simulation_step)" in prov
    assert "get_feature_at_time(tid, float(simulation_step))" in pred, \
        "update() no longer pushes the row at simulation_step; re-audit k=0"
    step_and_get = prov[prov.index("def step_and_get"):]
    step_and_get = step_and_get[:step_and_get.index("def get_features")]
    assert "self.update(simulation_step)" in step_and_get, \
        "step_and_get no longer pushes the current row before forecasting"
    return 0


def test_T5_training_label_and_deployment_window_name_the_same_row(files):
    java_offset = _java_future_window_starts_at_current_row()
    py_offset = _python_consumption_offset_from_the_pushed_row()
    assert java_offset == py_offset == 0

    plan_a = cd.CleanWindowDataset(files, "train", SEQ, PRED, label_start_offset=1)
    for i in (0, 5, len(plan_a) - 1):
        _, last_history = plan_a.history_last_row(i)
        _, label_first = plan_a.label_first_row(i)
        # deployment: standing on `last_history`, the first row read is
        # last_history + java_offset. training: y[0] is `label_first`.
        assert label_first == last_history + java_offset, \
            "plan A does not line the label up with the deployed window"


def test_T5b_the_check_discriminates(files):
    """A test that passes under both conventions would prove nothing."""
    stock = cd.CleanWindowDataset(files, "train", SEQ, PRED, label_start_offset=0)
    _, last_history = stock.history_last_row(0)
    _, label_first = stock.label_first_row(0)
    assert label_first == last_history + 1, \
        "the stock convention must still be one row late; otherwise T5 tests nothing"


def test_T5c_wrapper_passes_the_knob_through(files):
    ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED, scale=False,
                                label_start_offset=1)
    assert ds.label_start_offset == 1
    x, y, _, _ = ds[0]
    assert float(y[0, 0]) == pytest.approx(float(x[-1, 0]))

    exp = tc.CleanExpTimeCAP.__new__(tc.CleanExpTimeCAP)
    exp.clean_files = files
    exp.args = tc.SimpleNamespace(seq_len=SEQ, pred_len=PRED, batch_size=4,
                                  num_workers=0, drop_last=False, scale=True,
                                  label_start_offset=1)
    got, _ = tc.CleanExpTimeCAP._get_data(exp, "train")
    assert got.label_start_offset == 1


def test_T5d_label_len_is_not_the_mechanism(tmp_path):
    """Plan A must not be implemented by moving Code/'s label_len: that changes the LENGTH
    of y (r_end = r_begin + label_len + pred_len), not where it starts."""
    d = str(tmp_path)
    _write(d, 1, 2020, 400)
    args = tc.build_clean_args([1], [2020], res_dir=str(tmp_path / "res"), epochs=1,
                               batch_size=4, lr=5e-5, patience=5, use_gpu=False, gpu=0,
                               num_workers=0, split_dir=d, label_start_offset=1)
    assert args.label_len == 0, "plan A must not be routed through label_len"
    assert args.label_start_offset == 1
