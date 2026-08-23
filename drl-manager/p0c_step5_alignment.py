"""P0-C step 5: does the green series the simulator serves at a registered
offset actually start where the artifact says it does?

reset_skip advances a counter inside Java. Nothing so far has checked that the
counter lands on the row the artifact names, that each datacentre applies its
own timezone shift, or that TimeCAP's history is drawn from the same place. An
off-by-one here would silently move every registered window and no downstream
metric would look wrong.

The test is scale-free on purpose. COMPRESSED mode divides raw turbine kW by a
configured divisor before it reaches the observation, so comparing magnitudes
would test the divisor, not the alignment. Cross-correlating the observed
per-DC series against the CSV at a range of lags tests the alignment directly:
the peak must sit at lag 0.
"""
import csv
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("EVAL_CONFIG_PATH",
                      str(pathlib.Path(__file__).resolve().parent.parent / "config_C.yml"))

from src.baselines.evaluate import load_config          # noqa: E402
from gym_cloudsimplus.envs import HierarchicalMultiDCEnv  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
WIND = ROOT.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ART = json.loads((ROOT / "calib" / "p0c_green_windows.json").read_text())
ARMS = ("experiment_g1eval_matchedvan", "experiment_g1eval_knSV3b")
STEPS = 900
LAGS = range(-24, 25)


def steps_per_row(series):
    """The green series is piecewise constant: it only moves when the simulator
    crosses into the next CSV row. The run length of those constant segments is
    the step-to-row rate, which must be measured rather than assumed - one env
    step is not one row."""
    runs, cur = [], 1
    for a, b in zip(series, series[1:]):
        if abs(a - b) < 1e-9:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    if len(runs) < 3:
        return None
    runs = sorted(runs[1:-1]) or runs      # drop the partial first/last segment
    return runs[len(runs) // 2]


def _green(obs):
    """reset()/step() return {"global": {...}, "local_i": {...}}; the green
    series lives on the global observation."""
    g = obs.get("global", obs)
    return np.asarray(g["dc_current_green_power_w"], dtype=float).copy()


def csv_series(turbines, start, n):
    out = np.zeros(n)
    for t in turbines:
        with open(WIND / f"Turbine_{t}_2021.csv") as fh:
            col = [float(r["power_kw"] or 0) for r in csv.DictReader(fh)]
        out += np.array(col[start:start + n])
    return out


def probe(experiment, k, offset):
    cfg = load_config(experiment)
    # The env only auto-launches a gateway when these two are set; evaluate.py
    # fills them in at the CLI layer, which this probe bypasses.
    cfg["py4j_port"] = None
    cfg.setdefault("gateway_log_dir", f"/tmp/p0c_step5_gateways_k{k}")
    env = HierarchicalMultiDCEnv(config=cfg)
    try:
        for _ in range(k):
            env.reset(seed=20260823)
        obs, _ = env.reset(seed=20260823)
        series = [_green(obs)]
        for _ in range(STEPS - 1):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            series.append(_green(obs))
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return np.array(series)


def main():
    warm = 0   # search absolute lag; the measured value is the answer, not an input
    expect = int(ART["safe_domain"]["warmup_rows"])
    print(f"artifact records warmup_rows={expect} (measured); config "
          f"simulation_warmup_rows={load_config(ARMS[0]).get('simulation_warmup_rows', 0)}")
    rows = []
    for arm in ARMS:
     for w in ART["windows"]:
        k, offset = w["episode_index_k"], w["offset_rows"]
        obs_series = probe(arm, k, offset)
        n = len(obs_series)
        for dc in load_config(arm)["datacenters"]:
            turbines = dc.get("turbine_ids") or []
            if not turbines:
                continue
            i, tz = dc["datacenter_id"], dc["time_zone_offset_rows"]
            raw = obs_series[:, i]
            if np.std(raw) < 1e-9:
                rows.append((f'{arm.split("_")[-1]}/{w["stratum"]}', i, "FLAT", None, None, None))
                continue
            spr = steps_per_row(raw)
            if not spr:
                rows.append((f'{arm.split("_")[-1]}/{w["stratum"]}', i, "NO-STEP", None, None, None))
                continue
            # one sample per row, read at the middle of each constant segment
            got = raw[spr // 2::spr]
            m = len(got)
            best, bestr = None, -2.0
            for lag in LAGS:
                start = offset + warm + tz + lag
                if start < 0:
                    continue
                ref = csv_series(turbines, start, m)
                if np.std(ref) < 1e-9:
                    continue
                r = float(np.corrcoef(got, ref)[0, 1])
                if r > bestr:
                    best, bestr = lag, r
            rows.append((f'{arm.split("_")[-1]}/{w["stratum"]}', i, f"tz={tz}", best, bestr, spr))
    print(f"{'arm/window':<20}{'DC':>4}{'tz':>8}{'steps/row':>11}{'best lag':>10}{'corr':>9}   verdict")
    ok = True
    for stratum, dc, tz, lag, r, spr in rows:
        v = "PASS" if lag == expect and r is not None and r > 0.9999 else "FAIL"
        ok &= v == "PASS"
        print(f"{stratum:<20}{dc:>4}{tz:>8}{str(spr):>11}{str(lag):>10}"
              f"{('%.4f' % r) if r is not None else '   n/a':>9}   {v}")
    print("\nP0-C step 5:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
