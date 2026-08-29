"""Offline screen: can a forecast possibly help on this scenario?

Twelve testbeds were built and trained before anyone asked whether the physics
allowed a forecast to matter. This computes, without running the simulator, the
three quantities that decide it, and it is calibrated against C-regime, where
the answer is now known to be no.

  schedulable fraction   how much of the energy a scheduler can move at all.
                         Idle hosts burn 51.4 W each whatever the policy does,
                         so a fleet that is mostly idle has little to arbitrate.

  green surplus          green supply divided by the movable demand. Above one
                         there is no scarcity, so timing cannot pay: whenever
                         work runs, green is already there.

  granularity            job runtime against the time the greenest site stays
                         greenest. A job pinned for longer than the ranking
                         survives cannot be placed well.

A scenario needs all three in range. C-regime fails on the first two.
"""
import argparse
import csv
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "cloudsimplus-gateway/src/main/resources"
WIND = RES / "windProduction/simplified"
IDLE_W, PEAK_W, HOST_PES = 51.4, 214.0, 64


def wind(turbines, year=2021):
    acc = None
    for t in turbines:
        f = WIND / f"Turbine_{t}_{year}.csv"
        if not f.is_file():
            return None
        v = np.array([float(x["power_kw"] or 0) for x in csv.DictReader(open(f))])
        acc = v if acc is None else acc + v
    return acc


def screen(cfg, key, divisor=None, verbose=True):
    b = cfg[key]
    dcs = b["datacenters"]
    steps = float(b.get("max_episode_length") or 7200)
    # Seconds of simulated time per env step. C-regime runs at 1 s/step while
    # TB12 runs at 600 s/step, so every duration has to be carried in seconds
    # before anything is compared across scenarios.
    dt = float(b.get("simulation_timestep") or 1.0)
    ep_sec = steps * dt
    trace = RES / b["cloudlet_trace_file"]
    if not trace.is_file():
        return None
    rows = list(csv.DictReader(open(trace)))
    mips = dcs[0].get("vm_pe_mips", 40000)
    ln = np.array([float(r["length"]) for r in rows])
    pes = np.array([float(r["pes_required"]) for r in rows])
    # Cloudlets run at 0.5 CPU utilisation, so wall time is twice the ideal
    # length/MIPS figure while power is charged on the allocated PEs. Verified
    # against the measured marginal energy on C-regime (424 Wh here, 452 Wh
    # measured by the independent offline calculation).
    UTIL = 0.5
    run = ln / (pes * mips) / UTIL                 # seconds of CPU time
    # marginal energy: the work itself, at the per-core dynamic power
    dyn_per_pe = (PEAK_W - IDLE_W) / HOST_PES
    e_marg = float((run * pes * dyn_per_pe).sum() / 3600.0)          # Wh
    # static energy: every host that stays up, for the whole episode
    # Idle hosts power down, so the fleet size is an upper bound on what stays
    # up. Calibrated on C-regime, where the measured total (1104 Wh) minus the
    # marginal term leaves 664 Wh of static draw against a 30-host fleet over
    # 7200 steps. This is a fitted proxy for ranking scenarios, not a power
    # model, and it is only comparable between scenarios of similar shape.
    hosts = sum(v for d in dcs for k, v in d.items() if k.startswith("host_count_"))
    CAL = 664.0 / (30.0 * 7200.0)          # per host per second, fitted on C-regime
    e_static = hosts * ep_sec * CAL                                   # Wh
    frac_sched = e_marg / (e_marg + e_static)
    # green available, in the same units the simulator serves
    div = divisor if divisor is not None else float(b.get("compressed_power_divisor") or 60.0)
    green = 0.0
    tz = []
    for d in dcs:
        t = d.get("turbine_ids") or []
        if not t:
            continue
        w = wind(t)
        if w is None:
            continue
        off = int(d.get("time_zone_offset_rows", 0)) + 13
        nrow = int(np.ceil(ep_sec / 600.0)) if dt >= 600 else int(steps)
        seg = w[off:off + nrow]
        # each CSV row covers 600 s of supply
        green += float(seg.sum() * 1000.0 / div * (600.0 if dt >= 600 else 1.0) / 3600.0)
        tz.append(d["datacenter_id"])
    surplus = green / e_marg if e_marg > 0 else float("inf")
    # granularity: how long the greenest DC stays greenest
    flips = np.nan
    if len(tz) > 1:
        segs = []
        for d in dcs:
            t = d.get("turbine_ids") or []
            if not t:
                continue
            w = wind(t)
            off = int(d.get("time_zone_offset_rows", 0)) + 13
            nrow = int(np.ceil(ep_sec / 600.0)) if dt >= 600 else int(steps)
            s2 = w[off:off + nrow]
            segs.append(s2 / max(s2.mean(), 1e-9))
        best = np.array(segs).argmax(0)
        # hold time in SECONDS, so it is comparable with job runtime
        row_sec = 600.0 if dt >= 600 else ep_sec / max(len(best), 1)
        flips = float(len(best) * row_sec / max(int((best[1:] != best[:-1]).sum()), 1))
    return dict(key=key, jobs=len(rows), dcs=len(dcs), green_dcs=len(tz),
                dt=dt, ep_h=ep_sec/3600.0, run_med=float(np.median(run)), e_marg=e_marg, e_static=e_static,
                frac_sched=frac_sched, green=green, surplus=surplus, hold=flips)


def show(r):
    verdict = []
    if r["frac_sched"] < 0.5:
        verdict.append("静态占比过高")
    if r["surplus"] > 1.5:
        verdict.append(f"绿电过剩{r['surplus']:.1f}x")
    if r["green_dcs"] > 1 and r["hold"] == r["hold"] and r["hold"] < r["run_med"]:
        verdict.append("粒度错配")
    print(f"{r['key'][:40]:<42}{r['jobs']:>6}{r['dcs']:>3}{r['ep_h']:>7.1f}{r['run_med']/60:>8.1f}"
          f"{100*r['frac_sched']:>8.1f}%{r['surplus']:>9.2f}"
          f"{(r['hold']/60 if r['hold']==r['hold'] else 0):>9.1f}   "
          f"{'; '.join(verdict) if verdict else '★ 三项全过'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_C.yml"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    keys = [a.only] if a.only else [k for k in cfg if k.startswith("experiment_")]
    print(f"{'实验':<42}{'作业':>6}{'DC':>3}{'集时h':>7}{'作业min':>8}{'可调度':>9}{'绿电倍':>9}{'保持min':>9}   判读")
    out = []
    for k in keys:
        try:
            r = screen(cfg, k)
        except Exception:
            r = None
        if r:
            out.append(r)
    seen = set()
    for r in sorted(out, key=lambda x: (-x["frac_sched"], x["surplus"])):
        sig = (r["jobs"], r["dcs"], round(r["frac_sched"], 3), round(r["surplus"], 2))
        if sig in seen:
            continue
        seen.add(sig)
        show(r)
