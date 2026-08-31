"""Frozen window table, selection rule and artefact hashes for the planner gate.

Codex 2026-08-30: the script, the hashes, the selection rule and the window table are
written before any confirmation run, so the windows cannot be chosen after seeing a
result. Offsets follow the simulator's own schedule, read from the Java source rather
than restated here.

    offset(k) = (1009 * k) mod green_episode_offset_range

1009 is prime, so for a range not divisible by 1009 the schedule cycles through the
whole span. The range is a configured value, not derived by the code, so the safe bound
is also reported: rows minus the episode, the warmup and the largest time-zone shift.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
STRIDE = 1009
WARMUP_ROWS = 13                      # measured, matches the planner's alignment

CALIBRATION_2021 = {"low": 19, "mid": 56, "high": 34}
CONFIRMATION_2020 = {"low": 27, "mid": 71, "high": 13}


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def turbine_rows(wd, turbine, year):
    f = wd / f"Turbine_{turbine}_{year}.csv"
    if not f.is_file():
        return None, None
    with open(f) as fh:
        n = sum(1 for _ in fh) - 1
    return n, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config_C.yml"))
    ap.add_argument("--experiment", default="experiment_g1eval_matchedvan")
    ap.add_argument("--jar", default=str(
        REPO / "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    blk = yaml.safe_load(open(args.config))[args.experiment]
    dcs = sorted(blk["datacenters"], key=lambda x: x["datacenter_id"])
    rng21 = int(blk["green_episode_offset_range"])
    episode = int(blk["max_episode_length"])
    tz = [int(d.get("time_zone_offset_rows", 0)) for d in dcs]
    turbines = sorted({t for d in dcs for t in (d.get("turbine_ids") or [])})
    wd = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"

    out = {
        "experiment": args.experiment,
        "stride": STRIDE,
        "episode_steps": episode,
        "warmup_rows": WARMUP_ROWS,
        "max_tz_offset_rows": max(tz) if tz else 0,
        "turbines": turbines,
        "configured_range_2021": rng21,
    }

    print(f"experiment {args.experiment}")
    print(f"  episode {episode} steps, warmup {WARMUP_ROWS} rows, "
          f"tz offsets {tz}, turbines {turbines}")

    rows = {}
    for year in (2020, 2021, 2022):
        n, f = turbine_rows(wd, turbines[0], year)
        rows[year] = n
        state = "USABLE" if (n or 0) > episode else "STUB, NOT USABLE"
        print(f"  {year}: {n} rows per turbine  [{state}]")
    out["rows_per_turbine"] = rows

    reserve = rows[2021] - rng21
    safe_2020 = rows[2020] - reserve
    out["reserve_rows_2021"] = reserve
    out["safe_range_2020_same_reserve"] = safe_2020

    print(f"\n  2021 reserve = {rows[2021]} - {rng21} = {reserve} rows")
    print(f"  applying the same reserve to 2020 gives range {safe_2020}")

    def table(label, year, ks, rng):
        print(f"\n{label}  year={year}  range={rng}")
        entries = {}
        for name, k in ks.items():
            off = (STRIDE * k) % rng
            end = off + WARMUP_ROWS + max(tz) + episode
            fits = end <= rows[year]
            entries[name] = {"k": k, "offset": off, "last_row_touched": end, "fits": fits}
            print(f"  {name:>5}  k={k:>3}  offset={off:>6}  last_row={end:>6}  "
                  f"{'ok' if fits else 'OVERRUNS THE TRACE'}")
        offs = sorted(e["offset"] for e in entries.values())
        span = WARMUP_ROWS + max(tz) + episode
        overlap = any(offs[i + 1] - offs[i] < span for i in range(len(offs) - 1))
        print(f"  windows disjoint: {not overlap}  (each spans {span} rows)")
        entries["_disjoint"] = not overlap
        return entries

    out["calibration_2021"] = table("CALIBRATION (2021)", 2021, CALIBRATION_2021, rng21)

    print("\nCONFIRMATION (2020) is NOT frozen: the range is undecided, see the report.")
    for rng, why in ((safe_2020, "same reserve as 2021"), (24669, "as stated by Codex")):
        out.setdefault("confirmation_2020_candidates", {})[str(rng)] = table(
            f"  candidate range {rng} ({why})", 2020, CONFIRMATION_2020, rng)

    print("\nartefact hashes")
    arte = {}
    for label, path in [("config", args.config), ("jar", args.jar)]:
        p = Path(path)
        if p.is_file():
            arte[label] = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
            print(f"  {label:<10} {arte[label]['sha256']}  {p}")
        else:
            print(f"  {label:<10} MISSING  {p}")
    for t in turbines:
        for year in (2020, 2021):
            n, f = turbine_rows(wd, t, year)
            if f is None:
                continue
            arte[f"turbine_{t}_{year}"] = {"sha256": sha256(f), "rows": n}
    for t in turbines:
        print(f"  T{t:<9} 2021 {arte[f'turbine_{t}_2021']['sha256'][:16]}…  "
              f"2020 {arte[f'turbine_{t}_2020']['sha256'][:16]}…")
    out["artefacts"] = arte

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
