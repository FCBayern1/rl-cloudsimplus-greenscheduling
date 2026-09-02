"""Tests for the boundary-clean loader.

Each test names a way the legacy pipeline earned STOP_DATA_PIPELINE. The counters here are
the audit's counters, recomputed against this loader instead of against `Dataset_Custom`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_dataset as cd  # noqa: E402

SEQ, PRED = 8, 12          # small window so a tiny fixture still exercises the borders
SPAN = SEQ + PRED


@pytest.fixture(scope="module")
def cols():
    return cd.feature_columns()


def _write(tmp, turbine, year, n, offset=0.0):
    """One synthetic turbine-year file whose rows are identifiable by value."""
    cols = cd.feature_columns()
    data = {c: np.arange(n, dtype=float) + offset for c in cols}
    data["TurbID"] = turbine
    data["Tmstamp"] = pd.date_range("2020-01-01", periods=n, freq="10min")
    p = os.path.join(tmp, f"Turbine_{turbine}_{year}.csv")
    pd.DataFrame(data).to_csv(p, index=False)
    return p


@pytest.fixture()
def two_files(tmp_path):
    d = str(tmp_path)
    # Two turbines with disjoint value ranges: any window that straddles them is visible.
    _write(d, 1, 2020, 200, offset=0.0)
    _write(d, 2, 2020, 200, offset=10_000.0)
    return cd.file_specs([1, 2], [2020], split_dir=d)


class TestColumnContract:
    def test_order_is_the_deployed_predictor_order_bit_for_bit(self, cols):
        sys.path.insert(0, os.path.join(cd._REPO, "drl-manager"))
        from timecap_prediction.predictor import TimeCAP_GreenPredictor
        assert cols == list(TimeCAP_GreenPredictor.DEFAULT_FEATURE_COLUMNS)
        assert len(cols) == 13

    def test_patv_is_the_last_column(self, cols):
        assert cols[-1] == "Patv"
        assert cols.index("Patv") == 12

    def test_dataset_lays_the_columns_out_in_that_order(self, two_files, cols):
        ds = cd.CleanWindowDataset(two_files, "train", SEQ, PRED, scale=False)
        assert ds.columns == cols
        x, y = ds[0]
        assert x.shape == (SEQ, 13) and y.shape == (PRED, 13)

    def test_missing_feature_column_is_loud(self, tmp_path, cols):
        p = _write(str(tmp_path), 3, 2020, 100)
        df = pd.read_csv(p).drop(columns=["Patv"])
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing feature columns"):
            cd.CleanWindowDataset([cd.FileSpec(3, 2020, p)], "train", SEQ, PRED)


class TestNoWindowCrossesABoundary:
    def test_cross_boundary_count_is_zero_in_every_split(self, two_files):
        for split in cd.SPLITS:
            a = cd.CleanWindowDataset(two_files, split, SEQ, PRED).audit()
            assert a["cross_file_windows"] == 0, split
            assert a["cross_split_windows"] == 0, split

    def test_no_window_mixes_two_turbines(self, two_files):
        """The 478 the audit found: a sample that starts in one turbine and ends in another."""
        ds = cd.CleanWindowDataset(two_files, "train", SEQ, PRED, scale=False)
        for i in range(len(ds)):
            x, y = ds[i]
            block = np.concatenate([x, y])[:, 0]
            # Values are contiguous within a file and 10000 apart between files.
            assert block.max() - block.min() == SPAN - 1, f"window {i} straddles two files"

    def test_every_window_sits_inside_its_own_split(self, two_files):
        for split in cd.SPLITS:
            ds = cd.CleanWindowDataset(two_files, split, SEQ, PRED)
            for i in range(len(ds)):
                name, start, end = ds.window_rows(i)
                lo, hi = ds.borders[name][split]
                assert lo <= start and end <= hi

    def test_window_count_is_the_border_arithmetic(self, two_files):
        for split in cd.SPLITS:
            ds = cd.CleanWindowDataset(two_files, split, SEQ, PRED)
            want = sum(max(0, (hi - lo) - SPAN + 1)
                       for name in ds.borders
                       for lo, hi in [ds.borders[name][split]])
            assert len(ds) == want

    def test_a_split_shorter_than_one_window_yields_nothing(self, tmp_path):
        p = _write(str(tmp_path), 4, 2020, 60)      # val slice is 6 rows, span is 20
        ds = cd.CleanWindowDataset([cd.FileSpec(4, 2020, p)], "val", SEQ, PRED)
        assert len(ds) == 0


class TestSplitsDoNotOverlap:
    def test_borders_are_contiguous_exhaustive_and_disjoint(self):
        b = cd.split_borders(1000)
        assert b["train"] == (0, 700) and b["val"] == (700, 800) and b["test"] == (800, 1000)
        assert b["train"][1] == b["val"][0] and b["val"][1] == b["test"][0]
        assert b["test"][1] == 1000

    def test_no_seq_len_pullback(self):
        """Legacy val started 96 rows before train ended, so the two shared 96 rows."""
        b = cd.split_borders(157683)
        assert b["val"][0] == b["train"][1], "the pull-back is back"

    def test_audit_reports_no_row_overlap(self, two_files):
        for split in cd.SPLITS:
            assert cd.CleanWindowDataset(two_files, split, SEQ, PRED).audit()[
                "split_row_overlaps"] == []

    def test_window_row_sets_are_disjoint_across_splits(self, two_files):
        seen = {}
        for split in cd.SPLITS:
            ds = cd.CleanWindowDataset(two_files, split, SEQ, PRED)
            rows = set()
            for i in range(len(ds)):
                name, start, end = ds.window_rows(i)
                rows |= {(name, r) for r in range(start, end)}
            for other, prev in seen.items():
                assert not (rows & prev), f"{split} and {other} share rows"
            seen[split] = rows

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="ratios must sum to 1"):
            cd.split_borders(100, (0.7, 0.1, 0.1))


class TestScaler:
    def test_fitted_on_the_train_segment_only(self, two_files):
        ds = cd.CleanWindowDataset(two_files, "test", SEQ, PRED)
        for name, s in ds.scalers.items():
            assert s["fit_rows"] == ds.borders[name]["train"]
            arr = ds.data[name][slice(*ds.borders[name]["train"])]
            assert np.allclose(s["mean"], arr.mean(axis=0))
            assert np.allclose(s["std"], np.where(arr.std(axis=0) == 0, 1.0,
                                                  arr.std(axis=0)))

    def test_val_and_test_rows_cannot_move_the_scaler(self, tmp_path):
        d = str(tmp_path)
        p = _write(d, 5, 2020, 200)
        before = cd.CleanWindowDataset([cd.FileSpec(5, 2020, p)], "train",
                                       SEQ, PRED).scalers["Turbine_5_2020"]["mean"].copy()
        df = pd.read_csv(p)
        df.loc[140:, cd.feature_columns()] = 1e6          # wreck val and test only
        df.to_csv(p, index=False)
        after = cd.CleanWindowDataset([cd.FileSpec(5, 2020, p)], "train",
                                      SEQ, PRED).scalers["Turbine_5_2020"]["mean"]
        assert np.allclose(before, after), "the scaler saw rows outside train"

    def test_each_file_is_scaled_against_itself(self, two_files):
        ds = cd.CleanWindowDataset(two_files, "train", SEQ, PRED)
        a = ds.scalers["Turbine_1_2020"]["mean"]
        b = ds.scalers["Turbine_2_2020"]["mean"]
        assert not np.allclose(a, b), "one scaler is being shared across files"

    def test_scaling_is_reversible_to_the_raw_rows(self, two_files):
        raw = cd.CleanWindowDataset(two_files, "train", SEQ, PRED, scale=False)
        scaled = cd.CleanWindowDataset(two_files, "train", SEQ, PRED, scale=True)
        name, start, _ = raw.window_rows(3)
        s = scaled.scalers[name]
        x_raw, _ = raw[3]
        x_sc, _ = scaled[3]
        assert np.allclose(x_sc * s["std"] + s["mean"], x_raw, atol=1e-3)


class TestYearIsolation:
    def test_the_forbidden_year_is_refused(self, tmp_path):
        p = _write(str(tmp_path), 6, 2022, 200)
        with pytest.raises(ValueError, match="excluded from every"):
            cd.CleanWindowDataset([cd.FileSpec(6, 2022, p)], "train", SEQ, PRED)

    def test_unknown_split_is_refused(self, two_files):
        with pytest.raises(ValueError, match="unknown split"):
            cd.CleanWindowDataset(two_files, "validation", SEQ, PRED)

    def test_missing_file_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cd.file_specs([99], [2020], split_dir=str(tmp_path))


class TestAgainstTheRealSplitFiles:
    """One pass over a real SDWPF file at the production window size."""

    def test_real_turbine_year_is_boundary_clean(self):
        files = cd.file_specs([12], [2020])
        for split in cd.SPLITS:
            ds = cd.CleanWindowDataset(files, split, cd.SEQ_LEN, cd.PRED_LEN)
            a = ds.audit()
            assert a["cross_file_windows"] == 0 and a["cross_split_windows"] == 0
            assert a["split_row_overlaps"] == []
            assert a["scaler_fit_is_train_only"] and a["patv_is_last"]
            assert len(ds) > 0, split

    def test_manifest_covers_the_three_splits_with_source_hashes(self):
        m = cd.manifest([12], [2020])
        assert set(m["splits"]) == set(cd.SPLITS)
        assert len(m["file_sha256"]["Turbine_12_2020"]) == 64
        assert all(m["splits"][s]["cross_file_windows"] == 0 for s in cd.SPLITS)
