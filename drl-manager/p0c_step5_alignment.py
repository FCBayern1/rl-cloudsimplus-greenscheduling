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
STEPS = 300
LAGS = range(-6, 7)


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
        series = [np.asarray(obs["dc_current_green_power_w"], dtype=float).copy()]
        for _ in range(STEPS - 1):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            series.append(np.asarray(obs["dc_current_green_power_w"], dtype=float).copy())
            if term or trunc:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return np.array(series)


def main():
    warm = ART["safe_domain"]["warmup_rows"]
    rows = []
    for w in ART["windows"]:
        k, offset = w["episode_index_k"], w["offset_rows"]
        obs_series = probe("experiment_p0cprobe_van", k, offset)
        n = len(obs_series)
        for dc in load_config("experiment_p0cprobe_van")["datacenters"]:
            turbines = dc.get("turbine_ids") or []
            if not turbines:
                continue
            i, tz = dc["datacenter_id"], dc["time_zone_offset_rows"]
            got = obs_series[:, i]
            if np.std(got) < 1e-9:
                rows.append((w["stratum"], i, "FLAT", None, None))
                continue
            best, bestr = None, -2.0
            for lag in LAGS:
                ref = csv_series(turbines, offset + warm + tz + lag, n)
                if np.std(ref) < 1e-9:
                    continue
                r = float(np.corrcoef(got, ref)[0, 1])
                if r > bestr:
                    best, bestr = lag, r
            rows.append((w["stratum"], i, f"tz={tz}", best, bestr))
    print(f"{'window':<8}{'DC':>4}{'tz':>8}{'best lag':>10}{'corr':>9}   verdict")
    ok = True
    for stratum, dc, tz, lag, r in rows:
        v = "PASS" if lag == 0 and r is not None and r > 0.99 else "FAIL"
        ok &= v == "PASS"
        print(f"{stratum:<8}{dc:>4}{tz:>8}{str(lag):>10}{('%.4f' % r) if r is not None else '   n/a':>9}   {v}")
    print("\nP0-C step 5:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
