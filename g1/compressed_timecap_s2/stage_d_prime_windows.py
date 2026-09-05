"""Stage D' judgement windows, deterministic (STAGE_D_PRIME_DESIGN §16 Q4).

Rule, fixed before any window is chosen and reading no green, carbon or policy value:
  - excluded: every read window (window_preflight.READ_K on the 1009-row schedule), the
    six Stage D judgement offsets, and the Stage D training windows, each with its FULL
    footprint (the eval footprint for read/judgement windows, the training footprint for
    training windows);
  - legal offsets: every offset on the PRE-spaced grid whose eval-footprint interval lies
    inside a free gap;
  - order: sha256("stage-d-prime-judgement-v1:" + offset), ascending;
  - greedy: take offsets in that order, keeping those whose interval overlaps none already
    taken; stop at six;
  - fewer than six -> STOP_WINDOW_SPLIT.
Old training windows may keep serving training; old judgement windows serve development
and smoke only. Pure `select()`; the CLI writes stage_a_out/stage_d_prime_windows.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_preflight as wp  # noqa: E402

TAG = "stage-d-prime-judgement-v1"
N_WINDOWS = 6


def candidates(taken, length, lo=wp.PRE, hi=wp.ROWS_IN_FILE):
    out = []
    for gs, ge in wp.free_gaps(taken, lo=lo, hi=hi):
        o = gs + wp.PRE
        while o + length <= ge:
            out.append(o)
            o += wp.PRE
    return out


def select(read_intervals, eval_len, n=N_WINDOWS):
    """Pure. read_intervals: [(start, end), ...] already excluded; eval_len: footprint rows."""
    cands = candidates(read_intervals, eval_len)
    order = sorted(cands, key=lambda o: hashlib.sha256(f"{TAG}:{o}".encode()).hexdigest())
    chosen = []
    for o in order:
        iv = (o - wp.PRE, o + eval_len)
        if all(not wp.overlaps(iv, (c - wp.PRE, c + eval_len)) for c in chosen) and \
                all(not wp.overlaps(iv, t) for t in read_intervals):
            chosen.append(o)
        if len(chosen) == n:
            break
    return {"tag": TAG, "n_candidates": len(cands), "windows": sorted(chosen),
            "status": "OK" if len(chosen) == n else "STOP_WINDOW_SPLIT"}


def main():
    prev = json.load(open(os.path.join(HERE, "stage_a_out", "stage_d_windows.json")))
    eval_len = int(prev["eval_footprint_rows"])
    train_len = int(prev["train_footprint_rows"])
    read = [tuple(w["rows"]) for w in prev["read_windows"]]
    read += [tuple(w["rows"]) for w in prev["eval_windows"]]      # the six Stage D judgement windows, read
    read += [tuple(w["rows"]) for w in prev["train_windows"]]     # training windows keep their footprint
    res = select(read, eval_len)
    res.update({"eval_footprint_rows": eval_len, "train_footprint_rows": train_len,
                "excluded_intervals": sorted(read),
                "eval_windows": [{"offset": o, "rows": (o - wp.PRE, o + eval_len)} for o in res["windows"]],
                "source": "stage_d_windows.json (read + judgement + train), rule " + TAG})
    with open(os.path.join(HERE, "stage_a_out", "stage_d_prime_windows.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("status", "n_candidates", "windows")}))


if __name__ == "__main__":
    main()
