"""Algorithm-comparison table for Stage D (STAGE_D_LONGRUN_PREREG Addendum F).

Addendum F: the workstation alone decides gates 1-5; the expanded comparison of EU-CRD
against CCA-PG and the risk-sensitive objectives is computed entirely on one platform,
including the N_V, V, N_E and E rows that serve as its references and denominators, and
results are never substituted across platforms. This reader therefore refuses to produce a
table whose rows do not all carry the same platform tag, where the tag is the GPU string
recorded in each seed's freeze manifest.

Quantities, per line X and per seed, pooled over the six cells and six judgement windows
(carbon intensity = sum carbon / sum completed MI, as in the verdict reader):

    forecast_value(X)   = (C_ref0 - C_X0) / C_ref0     with C_ref0 the matched no-forecast
    corruption_increment(X) = (C_X1 - C_X0) / C_X0
    containment(X)      = 1 - corruption_increment(X) / corruption_increment(V)

Matched no-forecast: N_V for V and for every risk line (they share V's backbone and
observation, asserted by the generator's tests), N_E for E, N_C for C.

Usage: python stage_d_comparison.py [<results_root_prefix>]
       default prefix drl-manager/results/stage_d_longrun, which with the _cca and _risk
       suffixes covers the three preregistrations.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

# line -> (results suffix, matched no-forecast line)
LINE_SOURCE = {
    "NV": ("", None), "V": ("", "NV"), "NE": ("", None), "E": ("", "NE"),
    "NC": ("_cca", None), "C": ("_cca", "NC"),
    "RCV": ("_risk", "NV"), "RRS": ("_risk", "NV"), "RMV": ("_risk", "NV"), "RDC": ("_risk", "NV"),
}
CLEAN_TIER = {L: ("hollow" if L.startswith("N") else "godeye") for L in LINE_SOURCE}
CORRUPT_TIER = "calibrated_shrink_v1"
CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]


def pooled_intensity(rows):
    c = sum(r["carbon"] for r in rows)
    mi = sum(r["mi"] for r in rows)
    return c / mi if mi > 0 else None


def compare(table, platforms):
    """table: {(line, seed, tier): [row, ...]}; platforms: {seed_or_key: platform tag}.
    Returns the comparison record, or a refusal when the platform tags disagree."""
    tags = sorted(set(platforms.values()))
    if len(tags) != 1:
        return {"status": "REFUSED_MIXED_PLATFORMS", "platforms": tags}
    lines = sorted({L for (L, _s, _t) in table})
    seeds = sorted({s for (_L, s, _t) in table})
    C = {}
    for L in lines:
        for s in seeds:
            for tier in {CLEAN_TIER[L], CORRUPT_TIER}:
                rows = table.get((L, s, tier))
                if rows:
                    C[(L, s, tier)] = pooled_intensity(rows)
    out = {"status": "OK", "platform": tags[0], "seeds": seeds, "lines": lines, "per_seed": {}}
    for s in seeds:
        rec = {}
        vinc = None
        v0, v1 = C.get(("V", s, "godeye")), C.get(("V", s, CORRUPT_TIER))
        if v0 and v1:
            vinc = (v1 - v0) / v0
        for L in lines:
            ref = LINE_SOURCE[L][1]
            c0 = C.get((L, s, CLEAN_TIER[L]))
            c1 = C.get((L, s, CORRUPT_TIER))
            r0 = C.get((ref, s, CLEAN_TIER[ref])) if ref else None
            e = {"clean": c0, "corrupt": c1}
            if c0 and r0:
                e["forecast_value"] = (r0 - c0) / r0
            if c0 and c1:
                e["corruption_increment"] = (c1 - c0) / c0
                if vinc:
                    e["containment_vs_vanilla"] = 1.0 - e["corruption_increment"] / vinc
            rec[L] = e
        out["per_seed"][s] = rec
    med = {}
    for L in lines:
        for k in ("clean", "corrupt", "forecast_value", "corruption_increment", "containment_vs_vanilla"):
            vals = [out["per_seed"][s][L].get(k) for s in seeds if out["per_seed"][s][L].get(k) is not None]
            if vals:
                med.setdefault(L, {})[k] = st.median(vals)
    out["median"] = med
    return out


def platform_tag(results_root, seed):
    p = os.path.join(results_root, f"seed_{seed}", "freeze.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("gpu")


def load(prefix):
    import ladder_v2_verdict as lv
    mi_per_job = lv._mi_per_job()
    table, platforms = {}, {}
    for L, (suffix, _ref) in LINE_SOURCE.items():
        root = prefix + suffix
        for d in sorted(glob.glob(os.path.join(root, "seed_*"))):
            seed = int(os.path.basename(d).split("_")[1])
            tag = platform_tag(root, seed)
            if tag:
                platforms[f"{root}:{seed}"] = tag
            for f in glob.glob(os.path.join(d, f"{L}_final", "*.csv")):
                b = os.path.basename(f)[:-4]
                cell = "_".join(b.split("_")[:5])
                tier = b[len(cell) + 1:].rsplit("_k", 1)[0]
                if tier not in (CLEAN_TIER[L], CORRUPT_TIER):
                    continue
                rows = list(csv.DictReader(open(f)))
                if not rows:
                    continue
                r = rows[-1]
                g = lambda k, d=0.0: float(r.get(k, d) or d)  # noqa: E731
                table.setdefault((L, seed, tier), []).append(
                    {"carbon": g("total_carbon_kg"), "mi": g("total_finished_cloudlets") * mi_per_job[cell]})
    return table, platforms


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "drl-manager/results/stage_d_longrun")
    table, platforms = load(prefix)
    out = compare(table, platforms)
    dest = os.path.join(os.path.dirname(prefix), "stage_d_comparison.json")
    with open(dest, "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    if out["status"] != "OK":
        print(json.dumps(out, indent=2))
        return
    print(f"platform: {out['platform']}   seeds: {out['seeds']}")
    print(f"{'line':5s} {'clean':>10s} {'corrupt':>10s} {'fcst value':>11s} {'corr incr':>10s} {'containment':>12s}")
    for L in out["lines"]:
        m = out["median"].get(L, {})
        fmt = lambda k, p=False: ("%9.4f" % m[k] if not p else "%10.1f%%" % (100 * m[k])) if k in m else "         -"  # noqa: E731
        print(f"{L:5s} {fmt('clean')} {fmt('corrupt')} {fmt('forecast_value', True)} "
              f"{fmt('corruption_increment', True)} {fmt('containment_vs_vanilla', True)}")


if __name__ == "__main__":
    main()
