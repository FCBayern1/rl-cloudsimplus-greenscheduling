"""Search turbine assignments and green scaling for a testbed where the
forecast could pay.

The C-regime diagnosis says why it cannot pay there. Green covers the movable
load 2.4 times over, so three quarters of the time placement is irrelevant, and
in the quarter that is short all three sites are short together, so there is
nowhere to move. The gap between a clairvoyant allocation and a greedy one that
only reads current green is 4.8 points.

Two knobs decide that gap and they are coupled. Scaling green down creates
scarcity but also desynchronises the sites, which lets routing substitute for
waiting. Turbine choice is the free variable: 182 series are available and the
current testbed uses five.

This computes a FLUID UPPER BOUND on the gap, where movable load may be split
across sites instantly. The real per-job, capacity-constrained gap is smaller,
so a configuration whose bound falls short is dead and one that clears it still
has to survive the constrained model.
"""
import csv
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WIND = ROOT / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
MOVABLE_W = 204.0          # dynamic power of the concurrent jobs, C-regime
EP = 7200
WARMUP = 13


def load_all(year=2021, offset=19171):
    out = {}
    for f in sorted(WIND.glob(f"Turbine_*_{year}.csv")):
        tid = int(f.stem.split("_")[1])
        # Only the real SDWPF series (1-134). The 7xxx/8xxx/9xxx families are
        # synthetic profiles built for earlier experiments, several of them
        # shaped specifically to make a forecast pay. Screening on those would
        # be marking our own homework.
        if tid > 134:
            continue
        v = np.array([float(x["power_kw"] or 0) for x in csv.DictReader(open(f))])
        seg = v[offset + WARMUP: offset + WARMUP + EP]
        if len(seg) == EP and seg.std() > 0:
            out[tid] = seg * 1000.0            # W before scaling
    return out


def evaluate(series, div):
    """series: list of per-DC arrays in W. Returns the screening quantities."""
    G = np.array(series) / div
    tot = G.sum(0)
    kappa = float(tot.sum() / (MOVABLE_W * EP))
    blind = float(np.minimum(G.max(0), MOVABLE_W).sum() / (MOVABLE_W * EP))
    omni = float(np.minimum(tot, MOVABLE_W).sum() / (MOVABLE_W * EP))
    lack = tot < MOVABLE_W
    med = np.median(G, axis=1, keepdims=True)
    sync = float(np.mean((G[:, lack] < med).all(0))) if lack.any() else 0.0
    return dict(kappa=kappa, blind=blind, omni=omni, gap=100 * (omni - blind),
                lack=float(lack.mean()), sync=sync)


def main():
    n_dc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    S = load_all()
    ids = sorted(S)
    print(f"载入 {len(ids)} 台涡轮,搜索 {n_dc} 个 DC 的组合", flush=True)
    # A single turbine per DC keeps the search tractable; the scale knob is the
    # divisor, so per-DC fleet size adds nothing the divisor cannot express.
    rng = np.random.default_rng(20260828)
    combos = list(itertools.combinations(ids, n_dc))
    if len(combos) > 40000:
        idx = rng.choice(len(combos), 40000, replace=False)
        combos = [combos[i] for i in idx]
    print(f"评估 {len(combos)} 个组合 × divisor 网格", flush=True)
    DIVS = [3000, 5000, 7000, 10000, 15000, 22000]
    best = []
    for k, combo in enumerate(combos):
        ser = [S[t] for t in combo]
        for div in DIVS:
            r = evaluate(ser, div)
            if r["gap"] >= 8.0:
                best.append((r["gap"], combo, div, r))
        if k % 5000 == 0 and k:
            print(f"  ... {k}/{len(combos)}  当前候选 {len(best)}", flush=True)
    best.sort(key=lambda x: -x[0])
    print(f"\n间隙 ≥8pp 的组合: {len(best)}\n")
    print(f"{'涡轮':<22}{'divisor':>8}{'κ':>7}{'盲态':>8}{'全知':>8}{'间隙pp':>9}{'缺绿%':>8}{'同步%':>8}")
    seen = set()
    for gap, combo, div, r in best:
        if combo in seen:
            continue
        seen.add(combo)
        print(f"{str(list(combo)):<22}{div:>8}{r['kappa']:>7.2f}"
              f"{100*r['blind']:>7.1f}%{100*r['omni']:>7.1f}%{gap:>9.2f}"
              f"{100*r['lack']:>7.1f}%{100*r['sync']:>7.1f}%")
        if len(seen) >= topn:
            break


if __name__ == "__main__":
    main()
