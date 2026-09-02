"""Tests for the clean training wrapper.

The claim under test is narrow and it is the whole point of the wrapper: what reaches the
model is the boundary-clean loader's windows, and only those. A wrapper that quietly falls
back to `Dataset_Custom`, or that lets a DataLoader assemble a batch spanning two turbines,
would reproduce exactly the defect `timecap_data_audit.json` stopped the pipeline for.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_dataset as cd  # noqa: E402
import train_timecap_clean as tc  # noqa: E402

SEQ, PRED = 8, 12
BAND = 10_000.0          # value gap between the two synthetic files


def _write(tmp, turbine, year, n, offset):
    cols = cd.feature_columns()
    data = {c: np.arange(n, dtype=float) + offset for c in cols}
    data["TurbID"] = turbine
    data["Tmstamp"] = pd.date_range("2020-01-01", periods=n, freq="10min")
    p = os.path.join(tmp, f"Turbine_{turbine}_{year}.csv")
    pd.DataFrame(data).to_csv(p, index=False)
    return p


@pytest.fixture()
def files(tmp_path):
    d = str(tmp_path)
    _write(d, 1, 2020, 400, 0.0)
    _write(d, 2, 2020, 400, BAND)
    return cd.file_specs([1, 2], [2020], split_dir=d)


class TestBatchProvenance:
    def test_every_batch_item_comes_from_one_file(self, files):
        """Assembled through a real DataLoader, not by inspecting the index."""
        ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED, scale=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=7, shuffle=True, num_workers=0)
        seen = 0
        for bx, by, bxm, bym in loader:
            block = torch.cat([bx, by], dim=1)[:, :, 0]      # (B, seq+pred)
            for row in block:
                span = float(row.max() - row.min())
                assert span == pytest.approx(SEQ + PRED - 1), \
                    "a window spans two files"
                seen += 1
        assert seen == len(ds)

    def test_every_batch_item_stays_inside_one_split(self, files):
        for split in cd.SPLITS:
            ds = tc.CleanDatasetAdapter(files, split, SEQ, PRED)
            if len(ds) == 0:
                continue
            ds.fetched = []
            loader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=True,
                                                 num_workers=0)
            for _ in loader:
                pass
            assert sorted(ds.fetched) == list(range(len(ds))), \
                "the loader did not draw exactly the clean index"
            for i in ds.fetched:
                name, start, end = ds.window_rows(i)
                lo, hi = ds.borders(name)
                assert lo <= start and end <= hi, f"{split} item {i} escaped its split"

    def test_the_three_splits_never_share_a_row(self, files):
        seen = {}
        for split in cd.SPLITS:
            ds = tc.CleanDatasetAdapter(files, split, SEQ, PRED)
            rows = set()
            for i in range(len(ds)):
                name, start, end = ds.window_rows(i)
                rows |= {(name, r) for r in range(start, end)}
            for other, prev in seen.items():
                assert not (rows & prev), f"{split} and {other} share rows"
            seen[split] = rows

    def test_audit_counters_are_clean(self, files):
        for split in cd.SPLITS:
            a = tc.CleanDatasetAdapter(files, split, SEQ, PRED).audit()
            assert a["cross_file_windows"] == 0
            assert a["cross_split_windows"] == 0
            assert a["split_row_overlaps"] == []
            assert a["scaler_fit_is_train_only"]


class TestTupleCompatibility:
    def test_shape_matches_the_stock_loader_tuple(self, files):
        ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED)
        x, y, xm, ym = ds[0]
        assert x.shape == (SEQ, 13) and y.shape == (PRED, 13)
        # Dataset_Custom sizes BOTH marks from seq_x; mirrored so the batch is
        # byte-compatible with the stock finetune path rather than merely similar.
        assert xm.shape == (SEQ, 1) and ym.shape == (SEQ, 1)
        assert torch.count_nonzero(xm) == 0 and torch.count_nonzero(ym) == 0

    def test_label_window_starts_after_the_history(self, files):
        """label_len = 0, so y[0] is the row after x[-1]. Same convention as Code/."""
        ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED, scale=False)
        x, y, _, _ = ds[0]
        assert float(y[0, 0]) == pytest.approx(float(x[-1, 0]) + 1.0)

    def test_default_collate_produces_batched_tensors(self, files):
        ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED)
        bx, by, bxm, bym = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
        assert bx.shape == (4, SEQ, 13) and by.shape == (4, PRED, 13)
        assert bx.dtype == torch.float32


class TestWiring:
    def test_inference_flag_maps_to_the_test_split(self):
        assert tc.FLAG_TO_SPLIT["inference"] == "test"
        assert tc.FLAG_TO_SPLIT["train"] == "train"

    def test_build_clean_args_keeps_the_stock_model_shape(self, tmp_path):
        d = str(tmp_path)
        _write(d, 1, 2020, 400, 0.0)
        args = tc.build_clean_args([1], [2020], res_dir=str(tmp_path), epochs=1,
                                   batch_size=4, lr=5e-5, patience=5, use_gpu=False,
                                   gpu=0, num_workers=0, split_dir=d)
        assert args.seq_len == 96 and args.label_len == 0 and args.pred_len == 144
        assert args.enc_in == 13 and args.d_model == 736 and args.features == "MS"
        assert [f.name for f in args.clean_files] == ["Turbine_1_2020"]

    def test_get_data_uses_the_clean_loader_not_dataset_custom(self, files, tmp_path):
        """The one substitution the wrapper is allowed to make."""
        exp = tc.CleanExpTimeCAP.__new__(tc.CleanExpTimeCAP)
        exp.clean_files = files
        exp.args = tc.SimpleNamespace(seq_len=SEQ, pred_len=PRED, batch_size=4,
                                      num_workers=0, drop_last=False, scale=True)
        ds, loader = tc.CleanExpTimeCAP._get_data(exp, "train")
        assert isinstance(ds, tc.CleanDatasetAdapter)
        assert isinstance(ds.inner, cd.CleanWindowDataset)
        assert len(ds) == len(loader.dataset)

    def test_inverse_transform_refuses_when_scalers_differ(self, files):
        ds = tc.CleanDatasetAdapter(files, "train", SEQ, PRED)
        with pytest.raises(RuntimeError, match="ambiguous with per-file scalers"):
            ds.inverse_transform(np.zeros((2, 13)))

    def test_inverse_transform_round_trips_for_a_single_file(self, tmp_path):
        d = str(tmp_path)
        _write(d, 1, 2020, 400, 0.0)
        f = cd.file_specs([1], [2020], split_dir=d)
        raw = tc.CleanDatasetAdapter(f, "train", SEQ, PRED, scale=False)
        sc = tc.CleanDatasetAdapter(f, "train", SEQ, PRED, scale=True)
        x_raw, _, _, _ = raw[2]
        x_sc, _, _, _ = sc[2]
        back = sc.inverse_transform(x_sc.numpy())
        assert np.allclose(back, x_raw.numpy(), atol=1e-3)
