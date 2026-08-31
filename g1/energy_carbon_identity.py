"""Per-datacentre energy and carbon decomposition, checked against the global ledger.

Codex 2026-08-30: before any carbon difference between two arms can be attributed, the
ledger has to be shown to close. The identity is

    C = sum_d ( E_green,d * c_green,d + E_brown,d * c_brown,d )

reconstructed from the per-step per-datacentre trace that AUDIT_TRACE already writes,
and compared with the episode totals the simulator reports. When it closes, the carbon
gap between two arms can be split into a spatial part (which site), an efficiency part
(how much energy per unit of work) and a temporal part (when the work ran). When it does
not close, no attribution is meaningful and the trace or the meter is wrong.

Usage: energy_carbon_identity.py TRACE.csv TOTALS.csv [--experiment NAME]
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml


def load_factors(cfg_path, experiment):
    blk = yaml.safe_load(open(cfg_path))[experiment]
    dcs = sorted(blk["datacenters"], key=lambda x: x["datacenter_id"])
    return (np.array([d["brown_carbon_factor"] for d in dcs], dtype=float),
            np.array([d.get("green_carbon_factor", 0.01) for d in dcs], dtype=float),
            float(blk.get("simulation_timestep", 1.0)),
            [d.get("name", f"DC{d['datacenter_id']}") for d in dcs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("totals")
    ap.add_argument("--config", default=os.environ.get("EVAL_CONFIG_PATH", "config_C.yml"))
    ap.add_argument("--experiment", default=os.environ.get(
        "ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan"))
    args = ap.parse_args()

    cb, cg, dt, names = load_factors(args.config, args.experiment)
    n = len(cb)

    rows = list(csv.DictReader(open(args.trace)))
    power = np.array([[float(r[f"power_w_dc{d}"]) for d in range(n)] for r in rows])
    green = np.array([[float(r[f"green_w_dc{d}"]) for d in range(n)] for r in rows])

    # Green covers demand up to the amount generated; the rest is brown. Anything
    # generated above demand is spilled, which is what the waste column tracks.
    used_green = np.minimum(power, green)
    used_brown = power - used_green
    wh = dt / 3600.0
    e_green = used_green.sum(axis=0) * wh
    e_brown = used_brown.sum(axis=0) * wh
    e_spill = np.maximum(0.0, green - power).sum(axis=0) * wh

    c_per_dc = (e_green * cg + e_brown * cb) / 1000.0
    tot = list(csv.DictReader(open(args.totals)))[0]

    print(f"trace {args.trace}  steps={len(rows)}  timestep={dt}s")
    print(f"{'site':>5} {'brown_f':>8} {'green_Wh':>10} {'brown_Wh':>10} {'spill_Wh':>10} "
          f"{'carbon_kg':>10} {'share':>7} {'recv':>6}")
    for d in range(n):
        recv = tot.get(f"received_dc_{d}", "")
        share = c_per_dc[d] / c_per_dc.sum() if c_per_dc.sum() else 0.0
        print(f"{d:>5} {cb[d]:>8.2f} {e_green[d]:>10.2f} {e_brown[d]:>10.2f} {e_spill[d]:>10.2f} "
              f"{c_per_dc[d]:>10.6f} {share:>7.1%} {recv:>6}")

    print(f"{'sum':>5} {'':>8} {e_green.sum():>10.2f} {e_brown.sum():>10.2f} {e_spill.sum():>10.2f} "
          f"{c_per_dc.sum():>10.6f}")

    def cmp(label, got, want):
        want = float(want)
        rel = abs(got - want) / abs(want) if want else float("inf")
        flag = "MATCH" if rel < 1e-6 else ("close" if rel < 0.02 else "MISMATCH")
        print(f"  {label:<22} reconstructed={got:>12.6f}  reported={want:>12.6f}  "
              f"rel={rel:>8.3%}  {flag}")
        return rel

    print("\nidentity against the episode ledger")
    rels = [
        cmp("total carbon kg", c_per_dc.sum(), tot["total_carbon_kg"]),
        cmp("green used Wh", e_green.sum(), tot["green_used_wh"]),
        cmp("brown used Wh", e_brown.sum(), tot["brown_used_wh"]),
        cmp("total energy Wh", e_green.sum() + e_brown.sum(), tot["total_energy_wh"]),
        cmp("green waste Wh", e_spill.sum(), tot["green_waste_wh"]),
    ]
    print("\nweighted brown intensity of routing "
          f"{(e_brown * cb).sum() / max(e_brown.sum(), 1e-12):.4f} kg/kWh")
    return 0 if max(rels[:4]) < 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())
