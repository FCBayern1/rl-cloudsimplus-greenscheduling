"""Stage D' formal judgement windows on the 2020 series of the same five turbines
(STAGE_D_PRIME_DESIGN §20, Codex 2026-09-05). Cross-year one-shot confirmation.

Frozen rule:
  1. read_2020_intervals.json lists every 2020 offset ever used in this repository for the
     HZ turbines (123, 10, 51, 53, 112) with its full 2,922-row footprint; SHA frozen.
  2. legal offsets: the PRE-spaced grid inside [PRE, ROWS_2020 - footprint] not touching
     any read interval;
  3. order by sha256("stage-d-prime-judgement-v1:2020:" + offset), greedy six
     non-overlapping windows;
  4. no green, carbon or policy value of any candidate is read here or anywhere before the
     judgement;
  5. every judgement block must carry wind_csv_year = 2020 for the simulator and the
     forecast provider, and the error audit must be a 2020 audit (checked fail-fast by the
     generator, gen_stage_d.py eval_dprime_2020);
  6. fewer than six -> STOP_WINDOW_SPLIT; no fallback to 2021, no fewer windows;
  7. training stays on the frozen 2021 training windows.

Usage: python stage_d_prime_windows_2020.py scan    # build read_2020_intervals.json from the repo
       python stage_d_prime_windows_2020.py select  # only after margin, P0' and the dev smoke pass
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import window_preflight as wp  # noqa: E402
import stage_d_prime_windows as sw  # noqa: E402

YEAR = 2020
ROWS_2020 = 32225
HZ_TURBINES = (123, 10, 51, 53, 112)
TAG = f"stage-d-prime-judgement-v1:{YEAR}"
READ_FILE = os.path.join(HERE, "stage_a_out", "read_2020_intervals.json")
OUT_FILE = os.path.join(HERE, "stage_a_out", "stage_d_prime_windows_2020.json")


def scan_repo(repo=REPO):
    """Every tracked yml/json/md that mentions wind_csv_year 2020 (or a Turbine_<hz>_2020 file)
    together with at least one HZ turbine id; returns the offsets it names. The scan is
    conservative: any offset-like number in such a file is treated as read."""
    tracked = subprocess.run(["git", "ls-files", "*.yml", "*.yaml", "*.json", "*.md"], cwd=repo,
                             capture_output=True, text=True).stdout.split()
    hz = {str(t) for t in HZ_TURBINES}
    hits = []
    for rel in tracked:
        p = os.path.join(repo, rel)
        try:
            txt = open(p, errors="ignore").read()
        except Exception:
            continue
        if "2020" not in txt:
            continue
        year_hit = re.search(r"wind_csv_year\s*[:=]\s*2020", txt) or any(f"Turbine_{t}_2020" in txt for t in hz)
        if not year_hit:
            continue
        turb_hit = any(re.search(rf"\b{t}\b", txt) for t in hz)
        offs = sorted({int(x) for x in re.findall(r"(?:offset|reset-skip k=\d+ off)\s*[:=]?\s*(\d{3,5})", txt)})
        hits.append({"file": rel, "hz_turbine_mentioned": turb_hit, "offsets_named": offs})
    return hits


def build_read_file(eval_len):
    hits = scan_repo()
    intervals = []
    for h in hits:
        if h["hz_turbine_mentioned"]:
            for o in h["offsets_named"]:
                intervals.append([o - wp.PRE, o + eval_len])
    doc = {"year": YEAR, "turbines": list(HZ_TURBINES), "eval_footprint_rows": eval_len,
           "scan": hits, "read_intervals": sorted(intervals)}
    text = json.dumps(doc, indent=2, sort_keys=True)
    doc["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    with open(READ_FILE, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    return doc


def select_2020(read_intervals, eval_len, rows=ROWS_2020, n=6):
    """Pure. Same greedy hash rule as stage_d_prime_windows.select, on the 2020 length and tag."""
    cands = []
    for gs, ge in wp.free_gaps(read_intervals, lo=wp.PRE, hi=rows):
        o = gs + wp.PRE
        while o + eval_len <= ge:
            cands.append(o)
            o += wp.PRE
    order = sorted(cands, key=lambda o: hashlib.sha256(f"{TAG}:{o}".encode()).hexdigest())
    chosen = []
    for o in order:
        iv = (o - wp.PRE, o + eval_len)
        if all(not wp.overlaps(iv, (c - wp.PRE, c + eval_len)) for c in chosen) and \
                all(not wp.overlaps(iv, tuple(t)) for t in read_intervals):
            chosen.append(o)
        if len(chosen) == n:
            break
    return {"tag": TAG, "rows": rows, "n_candidates": len(cands), "windows": sorted(chosen),
            "status": "OK" if len(chosen) == n else "STOP_WINDOW_SPLIT"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    prev = json.load(open(os.path.join(HERE, "stage_a_out", "stage_d_windows.json")))
    eval_len = int(prev["eval_footprint_rows"])
    if mode == "scan":
        doc = build_read_file(eval_len)
        print(json.dumps({"files_scanned_with_2020_and_hz_turbine": [h["file"] for h in doc["scan"] if h["hz_turbine_mentioned"]],
                          "read_intervals": doc["read_intervals"], "sha256": doc["sha256"]}, indent=1))
        return
    if not os.path.exists(READ_FILE):
        raise SystemExit("run `scan` first; the read set must be frozen before selection")
    rd = json.load(open(READ_FILE))
    res = select_2020([tuple(x) for x in rd["read_intervals"]], eval_len)
    res.update({"eval_footprint_rows": eval_len, "read_file_sha256": rd.get("sha256"),
                "eval_windows": [{"offset": o, "rows": (o - wp.PRE, o + eval_len)} for o in res["windows"]],
                "year": YEAR, "turbines": list(HZ_TURBINES)})
    with open(OUT_FILE, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("status", "n_candidates", "windows", "read_file_sha256")}))


if __name__ == "__main__":
    main()
