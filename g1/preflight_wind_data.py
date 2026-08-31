"""Refuse placeholder wind data before it can be mistaken for a held-out year.

Codex 2026-08-30: the 2022 files in windProduction/simplified are two-row stubs with
zero power. Data like that must be rejected outright, never zero-filled, cycled or used
as a sanity check, and never described as held-out. A year passes only when every
turbine the experiment uses has enough rows to carry a window and a power series that
actually varies.
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def series(path):
    with open(path) as f:
        return [float(r["power_kw"] or 0.0) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="+", type=int)
    ap.add_argument("--config", default=str(REPO / "config_C.yml"))
    ap.add_argument("--experiment", default="experiment_g1eval_matchedvan")
    args = ap.parse_args()

    blk = yaml.safe_load(open(args.config))[args.experiment]
    dcs = sorted(blk["datacenters"], key=lambda x: x["datacenter_id"])
    turbines = sorted({t for d in dcs for t in (d.get("turbine_ids") or [])})
    episode = int(blk["max_episode_length"])
    tz = max(int(d.get("time_zone_offset_rows", 0)) for d in dcs)
    need = episode + tz + 13 + 288          # window, timezone, warmup, forecast reserve
    wd = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"

    bad = 0
    for year in args.years:
        print(f"year {year}  (each turbine needs at least {need} rows)")
        for t in turbines:
            f = wd / f"Turbine_{t}_{year}.csv"
            if not f.is_file():
                print(f"  T{t:<4} REJECT  file missing")
                bad += 1
                continue
            v = series(f)
            n = len(v)
            span = (max(v) - min(v)) if v else 0.0
            mean = (sum(v) / n) if n else 0.0
            why = []
            if n < need:
                why.append(f"only {n} rows")
            if span <= 0.0:
                why.append("power never varies")
            if mean <= 0.0:
                why.append("mean power is zero")
            if why:
                print(f"  T{t:<4} REJECT  rows={n:<7} span={span:<10.3f} mean={mean:<10.3f}"
                      f"  [{'; '.join(why)}]")
                bad += 1
            else:
                print(f"  T{t:<4} ok      rows={n:<7} span={span:<10.3f} mean={mean:<10.3f}")
    if bad:
        print(f"\n{bad} turbine-year(s) rejected. Placeholder data must not be used as a "
              f"calibration, confirmation or held-out set.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
