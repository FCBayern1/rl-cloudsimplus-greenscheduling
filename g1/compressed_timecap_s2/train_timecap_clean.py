#!/usr/bin/env python3
"""TimeCAP fine-tuning on the boundary-clean loader. Wrapper only: Code/ is not touched.

`timecap_data_audit.json` returned STOP_DATA_PIPELINE against the stock path, which
concatenates every turbine and year into one CSV and lets `Dataset_Custom` slide windows
across the joins (478 cross-turbine windows in train alone) while pulling the validation
border back by seq_len so train and val share 96 rows.

Everything about the model, the optimiser and the training loop stays exactly as
`Exp_TimeCAP` has it. The only substitution is `_get_data`, which returns a DataLoader over
`clean_dataset.CleanWindowDataset` instead of `Dataset_Custom`. That keeps the comparison
between a stock-trained and a clean-trained checkpoint attributable to the data boundaries
and to nothing else.

The adapter reproduces `Dataset_Custom.__getitem__`'s tuple shape exactly, including the
detail that BOTH marks are zeros of shape (seq_len, 1) -- the y mark uses seq_x's length in
the stock loader, and the finetune path never reads either, so mirroring it keeps the batch
byte-compatible rather than merely compatible in spirit.

This module does not start a real training run. It exists so that, if Stage A' asks for a
retrained predictor, the data foundation is already laid and tested. A 1-epoch smoke is the
most it is meant to do, and a smoke's numbers are not evidence of anything.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DRLMANAGER = _REPO / "drl-manager"
_CODE = _DRLMANAGER / "Code"

# Same ordering discipline as timecap_prediction/train_timecap.py: Code/ must win over the
# identically named packages under drl-manager/.
for _p in (str(_DRLMANAGER), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if sys.path[0] != str(_CODE):
    sys.path.remove(str(_CODE))
    sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_HERE))

import clean_dataset as cd                                       # noqa: E402
from timecap_prediction.train_timecap import build_args, save_model_args  # noqa: E402
from exp.exp_TimeCAP import Exp_TimeCAP                          # noqa: E402
from Arguments.load_setting import get_setting_str               # noqa: E402
from utils.tools import init_logger, make_dir, set_seed          # noqa: E402

FLAG_TO_SPLIT = {"train": "train", "val": "val", "test": "test", "inference": "test"}


class CleanDatasetAdapter(torch.utils.data.Dataset):
    """`CleanWindowDataset` in the tuple shape the stock finetune loop consumes."""

    def __init__(self, files, split, seq_len, pred_len, ratios=cd.RATIOS, scale=True,
                 label_start_offset=cd.LABEL_START_OFFSET):
        self.inner = cd.CleanWindowDataset(files, split, seq_len, pred_len,
                                           ratios=ratios, scale=scale,
                                           label_start_offset=label_start_offset)
        self.seq_len, self.pred_len, self.split = seq_len, pred_len, split
        self.label_start_offset = self.inner.label_start_offset
        # Set by the tests to record which items a real DataLoader pass actually fetched.
        self.fetched = None

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        if self.fetched is not None:
            self.fetched.append(int(i))
        x, y = self.inner[i]
        x = torch.from_numpy(np.ascontiguousarray(x))
        y = torch.from_numpy(np.ascontiguousarray(y))
        marks = torch.zeros((x.shape[0], 1))
        return x, y, marks, marks.clone()

    # -- provenance, for the batch-boundary test and for any audit artifact -------------
    def window_rows(self, i):
        return self.inner.window_rows(i)

    def label_first_row(self, i):
        return self.inner.label_first_row(i)

    def history_last_row(self, i):
        return self.inner.history_last_row(i)

    def borders(self, name):
        return self.inner.borders[name][self.split]

    def audit(self):
        return self.inner.audit()

    def inverse_transform(self, data):
        """Only well defined with a single file: the scalers are per file by design."""
        names = sorted(self.inner.scalers)
        if len(names) != 1:
            raise RuntimeError(
                "inverse_transform is ambiguous with per-file scalers; "
                f"this dataset holds {len(names)} files. Run with --inverse off, or "
                "build the dataset from one file.")
        s = self.inner.scalers[names[0]]
        return np.asarray(data) * s["std"] + s["mean"]


class CleanExpTimeCAP(Exp_TimeCAP):
    """`Exp_TimeCAP` with the data source swapped and nothing else changed."""

    def __init__(self, args, logger, model_dir, test_dir, setting):
        self.clean_files = args.clean_files
        super().__init__(args, logger, model_dir, test_dir, setting)

    def _get_data(self, flag):
        split = FLAG_TO_SPLIT[flag]
        ds = CleanDatasetAdapter(
            self.clean_files, split, self.args.seq_len, self.args.pred_len,
            scale=getattr(self.args, "scale", True),
            label_start_offset=getattr(self.args, "label_start_offset",
                                       cd.LABEL_START_OFFSET))
        shuffle = flag not in ("test", "inference")
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.args.batch_size, shuffle=shuffle,
            num_workers=self.args.num_workers, drop_last=self.args.drop_last,
            pin_memory=True)
        print(flag, len(ds))
        return ds, loader


def build_clean_args(turbine_ids, years, res_dir, epochs, batch_size, lr, patience,
                     use_gpu, gpu, num_workers=4, split_dir=cd.SPLIT_DIR,
                     label_start_offset=cd.LABEL_START_OFFSET):
    """The stock args, with the data source replaced by a file list.

    build_args is reused rather than restated so the model, optimiser and schedule stay
    identical to the stock run; only root_path/data_path become inert and clean_files
    carries the real source.
    """
    files = cd.file_specs(turbine_ids, years, split_dir)
    args = build_args(data_csv=str(_DRLMANAGER / "timecap_prediction/data/turbines_merged.csv"),
                      res_dir=res_dir, epochs=epochs, batch_size=batch_size, lr=lr,
                      patience=patience, use_gpu=use_gpu, gpu=gpu,
                      num_workers=num_workers)
    args.clean_files = files
    args.data = "clean_per_file"          # inert: _get_data never consults data_dict
    args.scale = True
    # Construction-side knob only. args.label_len stays 0: plan A's shift is carried by
    # the dataset, because moving label_len would change the LENGTH of y, not its start.
    args.label_start_offset = int(label_start_offset)
    return args


def main():
    ap = argparse.ArgumentParser(
        description="1-epoch smoke of the clean loader; not a training entry point")
    ap.add_argument("--turbine-id", action="append", type=int, required=True)
    ap.add_argument("--year", action="append", type=int, required=True)
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--label-start-offset", type=int, default=cd.LABEL_START_OFFSET,
                    help="0 = stock label convention; 1 = plan A (y[0] is the last "
                         "history row, matching deployed consumption). Retrain prereg "
                         "§2.1; NOT residual_calibration.py's --label-offset.")
    a = ap.parse_args()

    args = build_clean_args(a.turbine_id, a.year, a.res_dir, a.epochs, a.batch_size,
                            a.lr, a.patience, use_gpu=not a.no_gpu, gpu=a.gpu,
                            num_workers=a.num_workers,
                            label_start_offset=a.label_start_offset)
    set_seed(a.seed)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    args.device = torch.device(f"cuda:{args.gpu}") if args.use_gpu else torch.device("cpu")
    print("使用 GPU" if args.use_gpu else "使用 CPU")
    print(f"files: {[f.name for f in args.clean_files]}")
    print(f"label_start_offset: {args.label_start_offset} "
          f"({'plan A, y[0] = last history row' if args.label_start_offset == 1 else 'stock'})")

    # make_dir takes the args namespace and derives res_dir/<model>/{test,model,log}
    # itself; the stock entry point calls it exactly this way.
    test_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)
    setting = get_setting_str(args)

    exp = CleanExpTimeCAP(args, logger, model_dir, test_dir, setting)
    for split in ("train", "val", "test"):
        print(f"[audit] {split}: " + ", ".join(
            f"{k}={v}" for k, v in CleanDatasetAdapter(
                args.clean_files, split, args.seq_len, args.pred_len).audit().items()
            if k in ("n_windows", "cross_file_windows", "cross_split_windows",
                     "split_row_overlaps", "scaler_fit_is_train_only",
                     "label_start_offset", "span")))
    exp.finetune()
    mse, mae = exp.Inference()
    print(f"\n评估结果 — MSE: {mse:.4f}  MAE: {mae:.4f}   (smoke only; not evidence)")
    ckpt = Path(exp.best_checkpoints_path)
    if ckpt.exists():
        print(f"  Checkpoint : {ckpt}")
        print(f"  Args JSON  : {save_model_args(args, ckpt)}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt}")


if __name__ == "__main__":
    main()
