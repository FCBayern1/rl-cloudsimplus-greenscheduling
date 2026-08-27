"""T1: redundancy and conditional value of the forecast features.

Two layers, per the frozen spec in reports/T1_T2_SPEC_FROZEN.md.

  Layer 1  redundancy   predict each forecast feature from the blind-visible
                        variables, blocked out-of-fold R^2. Reported twice, once
                        for the exogenous subset (green power, green ratio,
                        demand, time) and once for the full blind set, because
                        queue and utilisation depend on the acting policy while
                        the exogenous ones do not.

  Layer 2  conditional  does adding the forecast improve prediction of a
           value        decision-relevant target over blind alone?
                        Reported as dR2 = R2(blind+forecast) - R2(blind).

Every split is a contiguous time block. Random row splits would let wind
autocorrelation leak the answer across the fold boundary and inflate R^2.
"""
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
T1 = ROOT / "g1/t1"
WIND = ROOT / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ART = json.loads((ROOT / "drl-manager/calib/p0c_green_windows.json").read_text())

FCST = ["dc_future_short_mean", "dc_future_short_trend",
        "dc_future_long_mean", "dc_future_long_peak_timing"]
EXOG = ["dc_current_green_power_w", "dc_current_power_w",
        "dc_green_ratio", "dc_cumulative_wasted_green_wh"]
ACTION_DEP = ["dc_queue_sizes", "dc_utilizations", "dc_available_pes", "dc_ram_utilizations"]
NFOLD = 5
DC_TURBINES = {0: [12, 36], 1: [95, 91], 2: [96], 3: [], 4: []}
DC_TZ = {0: 0, 1: 18, 2: 54, 3: 72, 4: 108}
WARMUP = 13


def load(k):
    rows = list(csv.DictReader(open(T1 / f"obs_k{k}.csv")))
    out = {}
    for r in rows:
        out.setdefault(int(r["dc"]), []).append(r)
    return out


def wind_series(turbines):
    if not turbines:
        return None
    acc = None
    for t in turbines:
        v = np.array([float(x["power_kw"] or 0)
                      for x in csv.DictReader(open(WIND / f"Turbine_{t}_2021.csv"))])
        acc = v if acc is None else acc + v
    return acc


def ridge_blocked_r2(X, y, nfold=NFOLD, lam=1e-6):
    """Out-of-fold R^2 with contiguous blocks."""
    n = len(y)
    idx = np.arange(n)
    bounds = np.linspace(0, n, nfold + 1).astype(int)
    pred = np.empty(n)
    for f in range(nfold):
        te = (idx >= bounds[f]) & (idx < bounds[f + 1])
        tr = ~te
        if tr.sum() < X.shape[1] + 2 or te.sum() < 2:
            pred[te] = y[tr].mean() if tr.sum() else 0.0
            continue
        Xtr = np.column_stack([np.ones(tr.sum()), X[tr]])
        Xte = np.column_stack([np.ones(te.sum()), X[te]])
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0; mu[0], sd[0] = 0.0, 1.0
        A = (Xtr - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = ((Xte - mu) / sd) @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build(k):
    """Per-DC design matrices plus the realised-future target."""
    data = load(k)
    offset = next(w["offset_rows"] for w in ART["windows"] if w["episode_index_k"] == k)
    out = {}
    for dc, rows in data.items():
        if not DC_TURBINES[dc]:
            continue
        n = len(rows)
        cols = {c: np.array([float(r[c]) for r in rows]) for c in EXOG + FCST + ACTION_DEP}
        step = np.array([float(r["step"]) for r in rows])
        cols["step_norm"] = step / max(step.max(), 1.0)
        ws = wind_series(DC_TURBINES[dc])
        base = offset + WARMUP + DC_TZ[dc]
        fut = {}
        for H in (1, 3, 6, 12, 24):
            fut[H] = np.array([ws[base + int(s) + 1: base + int(s) + 1 + H].mean()
                               if base + int(s) + 1 + H <= len(ws) else np.nan for s in step])
        out[dc] = (cols, fut)
    return out


def main():
    ks = [w["episode_index_k"] for w in ART["windows"]]
    per_dc = {}
    for k in ks:
        for dc, v in build(k).items():
            per_dc.setdefault(dc, []).append(v)

    print("=" * 78)
    print("第一层 冗余度:用盲态可见变量预测 forecast 特征(blocked OOF R²)")
    print("=" * 78)
    print(f"{'特征':<30}{'外生子集':>12}{'全盲集':>12}")
    l1 = {}
    for f in FCST:
        r_ex, r_all = [], []
        for dc, chunks in per_dc.items():
            for cols, _ in chunks:
                y = cols[f]
                if np.std(y) < 1e-12:
                    continue
                Xe = np.column_stack([cols[c] for c in EXOG] + [cols["step_norm"]])
                Xa = np.column_stack([cols[c] for c in EXOG + ACTION_DEP] + [cols["step_norm"]])
                r_ex.append(ridge_blocked_r2(Xe, y)); r_all.append(ridge_blocked_r2(Xa, y))
        l1[f] = (float(np.median(r_ex)), float(np.median(r_all))) if r_ex else (float("nan"),) * 2
        print(f"{f:<30}{l1[f][0]:>12.4f}{l1[f][1]:>12.4f}")

    print()
    print("=" * 78)
    print("第二层 条件价值:ΔR² = R²(blind+forecast) − R²(blind),目标=未来 H 行实现绿电")
    print("=" * 78)
    print(f"{'视界 H':<10}{'R²(blind)':>12}{'R²(+fcst)':>12}{'ΔR²':>10}{'各 DC ΔR²':>28}")
    for H in (1, 3, 6, 12, 24):
        rb, rf, per = [], [], []
        for dc, chunks in per_dc.items():
            db, df = [], []
            for cols, fut in chunks:
                y = fut[H]; m = ~np.isnan(y)
                if m.sum() < 100:
                    continue
                Xb = np.column_stack([cols[c][m] for c in EXOG] + [cols["step_norm"][m]])
                Xf = np.column_stack([cols[c][m] for c in EXOG + FCST] + [cols["step_norm"][m]])
                db.append(ridge_blocked_r2(Xb, y[m])); df.append(ridge_blocked_r2(Xf, y[m]))
            if db:
                rb += db; rf += df
                per.append(f"dc{dc}:{np.median(df)-np.median(db):+.3f}")
        b, ff = float(np.median(rb)), float(np.median(rf))
        print(f"{H:<10}{b:>12.4f}{ff:>12.4f}{ff-b:>10.4f}{'  '.join(per):>28}")

    print()
    print("判据(冻结):第一层 R² > 0.7 为高冗余;第二层 ΔR² < 0.05 为信息稀薄。")
    print("两条同时成立才支持『被大量使用但几乎不携带独立信息』。")


if __name__ == "__main__":
    main()
