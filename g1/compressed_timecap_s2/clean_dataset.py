#!/usr/bin/env python3
"""Boundary-clean TimeCAP data loading for Scheme 2. Built, not used: nothing here trains.

`timecap_data_audit.json` returned STOP_DATA_PIPELINE against the legacy path. Two things
put it there, and both are structural rather than incidental:

  * `prepare_turbine_data.py` concatenates every turbine and year into one CSV, and
    `Dataset_Custom` then splits that single file 7:1:2 BY ROW and slides windows freely
    inside each split. A window is therefore free to start in one turbine and end in the
    next. The audit counted 478 such windows in the train split alone. A sample that
    straddles two turbines is not a forecasting example; it teaches the model a
    discontinuity that exists nowhere in the world.
  * the legacy validation border is pulled back by seq_len (train ends at 110378, val
    starts at 110282) so the first val window carries history. That is a deliberate
    legacy convenience and it makes train and val share 96 rows.

This module replaces both with the semantics the audit asked for:

    one file per (turbine, year)      no concatenation, ever
    split inside each file            each file is cut 7:1:2 on its own rows
    windows never cross a boundary    a window lies inside one split of one file
    strictly disjoint splits          no seq_len pull-back, so no shared rows
    scaler per file, train only       fitted on that file's train segment and nothing else

The cost is real and stated rather than hidden: forbidding the pull-back drops
seq_len + pred_len - 1 candidate windows at each internal split border, and per-file
scaling means each site is normalised against itself. Both are choices, and both are
checked by the tests rather than asserted in prose.

Column order is taken from `predictor.TimeCAP_GreenPredictor.DEFAULT_FEATURE_COLUMNS` at
import time rather than restated, because a loader that agrees with a hand-copied list but
not with the deployed predictor is the failure this whole audit exists to catch.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SPLIT_DIR = os.path.join(_REPO, "cloudsimplus-gateway/src/main/resources/windProduction/split")

SEQ_LEN = 96
PRED_LEN = 144
RATIOS = (0.7, 0.1, 0.2)
SPLITS = ("train", "val", "test")


def feature_columns() -> List[str]:
    """The deployed predictor's column order, read from the predictor itself."""
    import sys
    p = os.path.join(_REPO, "drl-manager")
    if p not in sys.path:
        sys.path.insert(0, p)
    from timecap_prediction.predictor import TimeCAP_GreenPredictor
    return list(TimeCAP_GreenPredictor.DEFAULT_FEATURE_COLUMNS)


@dataclasses.dataclass(frozen=True)
class FileSpec:
    turbine_id: int
    year: int
    path: str

    @property
    def name(self) -> str:
        return f"Turbine_{self.turbine_id}_{self.year}"


def file_specs(turbine_ids: Sequence[int], years: Sequence[int],
               split_dir: str = SPLIT_DIR) -> List[FileSpec]:
    out = []
    for t in turbine_ids:
        for y in years:
            p = os.path.join(split_dir, f"Turbine_{t}_{y}.csv")
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            out.append(FileSpec(int(t), int(y), p))
    return out


def split_borders(n_rows: int, ratios: Sequence[float] = RATIOS) -> Dict[str, Tuple[int, int]]:
    """Row ranges of one file's three splits: contiguous, exhaustive, disjoint.

    No seq_len pull-back. The legacy loader starts validation 96 rows before training
    ends so the first val window has history; that is exactly what makes the two splits
    share rows, and a window that must not cross a border cannot be given one for free.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    n_train = int(n_rows * ratios[0])
    n_val = int(n_rows * ratios[1])
    return {"train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n_rows)}


class CleanWindowDataset:
    """Sliding windows over one split, guaranteed inside one file and one split.

    Item i is (x, y) with x of shape (seq_len, n_features) and y of shape
    (pred_len, n_features), scaled by the owning file's train-fitted scaler.
    """

    def __init__(self, files: Sequence[FileSpec], split: str,
                 seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN,
                 ratios: Sequence[float] = RATIOS, scale: bool = True,
                 forbid_years: Sequence[int] = (2022,)):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        bad = [f.name for f in files if f.year in tuple(forbid_years)]
        if bad:
            raise ValueError(f"years {tuple(forbid_years)} are excluded from every "
                             f"split; refused: {bad}")
        self.files = list(files)
        self.split, self.seq_len, self.pred_len = split, int(seq_len), int(pred_len)
        self.ratios, self.scale = tuple(ratios), bool(scale)
        self.columns = feature_columns()

        self.data: Dict[str, np.ndarray] = {}
        self.borders: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self.scalers: Dict[str, Dict[str, np.ndarray]] = {}
        self.index: List[Tuple[str, int]] = []          # (file name, absolute start row)

        span = self.seq_len + self.pred_len
        for f in self.files:
            df = pd.read_csv(f.path)
            missing = [c for c in self.columns if c not in df.columns]
            if missing:
                raise ValueError(f"{f.name} is missing feature columns {missing}")
            arr = df[self.columns].to_numpy(dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            b = split_borders(len(arr), self.ratios)
            lo_tr, hi_tr = b["train"]
            mu = arr[lo_tr:hi_tr].mean(axis=0)
            sd = arr[lo_tr:hi_tr].std(axis=0)
            sd[sd == 0.0] = 1.0
            self.data[f.name] = arr
            self.borders[f.name] = b
            self.scalers[f.name] = {"mean": mu, "std": sd,
                                    "fit_rows": (int(lo_tr), int(hi_tr))}
            lo, hi = b[split]
            # The only window rule in this module: it must end inside its own split.
            for start in range(lo, max(lo, hi - span) + 1):
                if start + span <= hi:
                    self.index.append((f.name, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        name, start = self.index[i]
        arr = self.data[name]
        if self.scale:
            s = self.scalers[name]
            arr = (arr - s["mean"]) / s["std"]
        x = arr[start:start + self.seq_len]
        y = arr[start + self.seq_len:start + self.seq_len + self.pred_len]
        return x.astype(np.float32), y.astype(np.float32)

    # -- auditing ------------------------------------------------------------------
    def window_rows(self, i: int) -> Tuple[str, int, int]:
        name, start = self.index[i]
        return name, start, start + self.seq_len + self.pred_len

    def audit(self) -> Dict:
        """The counters the legacy pipeline failed, recomputed against this loader."""
        span = self.seq_len + self.pred_len
        cross_file = 0                       # impossible by construction; counted anyway
        cross_split = 0
        for i in range(len(self)):
            name, start, end = self.window_rows(i)
            lo, hi = self.borders[name][self.split]
            if end - start != span:
                cross_file += 1
            if start < lo or end > hi:
                cross_split += 1
        overlaps = []
        for name, b in self.borders.items():
            r = [b[s] for s in SPLITS]
            for a, c in zip(r, r[1:]):
                if a[1] > c[0]:
                    overlaps.append((name, a, c))
        return {
            "split": self.split,
            "n_files": len(self.files),
            "n_windows": len(self),
            "cross_file_windows": cross_file,
            "cross_split_windows": cross_split,
            "split_row_overlaps": overlaps,
            "scaler_fit_is_train_only": all(
                s["fit_rows"] == self.borders[n]["train"]
                for n, s in self.scalers.items()),
            "feature_columns": list(self.columns),
            "patv_is_last": self.columns[-1] == "Patv",
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "files": sorted(self.borders),
        }


def manifest(turbine_ids: Sequence[int], years: Sequence[int],
             split_dir: str = SPLIT_DIR, **kw) -> Dict:
    """One audit record covering all three splits, with the source files' SHA256."""
    files = file_specs(turbine_ids, years, split_dir)
    out = {"turbine_ids": list(map(int, turbine_ids)), "years": list(map(int, years)),
           "split_dir": os.path.relpath(split_dir, _REPO), "splits": {},
           "file_sha256": {}}
    for f in files:
        h = hashlib.sha256()
        with open(f.path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["file_sha256"][f.name] = h.hexdigest()
    for s in SPLITS:
        out["splits"][s] = CleanWindowDataset(files, s, **kw).audit()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="audit only; this module never trains")
    ap.add_argument("--turbine-id", action="append", type=int, required=True)
    ap.add_argument("--year", action="append", type=int, required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    m = manifest(a.turbine_id, a.year)
    text = json.dumps(m, indent=2, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)
