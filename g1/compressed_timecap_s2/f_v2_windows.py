"""F_FITS_V2 §2 + Addendum A: twelve hash-drawn 2021 windows (6 training, 2 validation, 4 test),
footprint 1200 rows (an episode touches about 1007 rows: 96 rows of TimeCAP history, the 13-row
head, tz up to 108, planner horizon 669, candidate horizon 121; at 2922 the 2021 file has no
free position), at least 1200 rows from every read window and from each other, accepted in
draw order, never replaced. Pure draw + a record with hashes. Usage: python f_v2_windows.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_run as lr  # noqa: E402

TAG = "f-v2:2021"
FOOTPRINT = 1200
N_ROWS = 52559                      # Turbine_*_2021.csv rows
MAX_TZ, HEAD = 108, 13              # largest tz offset; obs row -> file row shift (+1 clock, +12 skip)
SPLIT = {"train": 6, "val": 2, "test": 4}
OUT = os.path.join(HERE, "stage_a_out", "f_v2", "windows.json")


def read_windows():
    """Every offset already read on this fleet's 2021 file."""
    dev = json.load(open(os.path.join(lr.OUT, "scene_v2_dev.json")))
    man = json.load(open(os.path.join(lr.OUT, "scene_v1_manifest.json")))
    read = set(dev["dev_offsets"]) | set(dev["candidates"]["offsets"]) | set(man["pool_2021"]["windows"])
    return sorted(read)


def draw(read, tag=TAG, n_total=12, footprint=FOOTPRINT, n_rows=N_ROWS, max_offset=None):
    """Pure: seeded uniform draws over [0, max_offset], accepted iff >= footprint from every
    read window and every accepted one; first n_total accepted, in draw order."""
    max_offset = (n_rows - footprint - MAX_TZ - HEAD) if max_offset is None else int(max_offset)
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    accepted, tried = [], 0
    while len(accepted) < n_total and tried < 100000:
        cand = int(rng.integers(0, max_offset + 1)); tried += 1
        if all(abs(cand - r) >= footprint for r in list(read) + accepted):
            accepted.append(cand)
    if len(accepted) < n_total:
        raise RuntimeError("could not draw enough windows")
    return {"seed": seed, "tag": tag, "max_offset": max_offset, "tried": tried, "offsets": accepted,
            "train": accepted[:SPLIT["train"]], "val": accepted[SPLIT["train"]:SPLIT["train"] + SPLIT["val"]],
            "test": accepted[SPLIT["train"] + SPLIT["val"]:n_total]}


def main():
    read = read_windows()
    d = draw(read)
    d["read_windows"] = read
    d["footprint"] = FOOTPRINT
    d["record_sha256"] = hashlib.sha256(json.dumps({k: d[k] for k in ("offsets", "train", "val", "test", "read_windows")}, sort_keys=True).encode()).hexdigest()[:16]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        prev = json.load(open(OUT))
        if prev.get("record_sha256") != d["record_sha256"]:
            raise RuntimeError("windows.json exists with a different record; windows are never redrawn")
        print("windows already drawn:", prev["record_sha256"]); return
    json.dump(d, open(OUT, "w"), indent=1)
    print(json.dumps({k: d[k] for k in ("train", "val", "test", "tried", "record_sha256")}))


if __name__ == "__main__":
    main()
