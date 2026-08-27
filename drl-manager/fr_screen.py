#!/usr/bin/env python3
"""预报特征筛选 步骤 2:blocked OOF ΔR²(工单 ad79b80,口径同 T1_T2_SPEC_FROZEN)。

问题:交付给策略的预报,相对"只看当前可观测量"是否携带**增量**信息。

- 目标 T-a:未来 H 行内每 DC 的**实现**绿电均值(真值,来自 CSV)
- 目标 T-b:oracle 排序标签(未来 H 行内实现绿电最高的 DC)
- blind 基线(仅当前可观测):当前绿电(= persistence)、绿电占比、时间索引
- 报告量:ΔR² = R²(blind + 候选) − R²(blind)
- **一律 blocked OOF,按时间连续分 5 块,禁止随机行切分**(风电自相关会虚高 R²)

判据(工单冻结):任一候选 ΔR² ≥ 0.05 ⇒ 有救;全部 < 0.05(含
`forecast − persistence` 残差)⇒ 预报器/视界不带增量信息。

附:**最佳滞后诊断**(非判据)。档案里有"−8 步 TimeCAP 滞后"的记录,
故对关键候选同时报契约对齐与最佳滞后对齐 —— 直接区分"没信息"与"信息错位"。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 3, 6, 12, 24, 144]      # 行;现役 short=3, long=144
N_FOLDS = 5
DR2_MIN = 0.05
ROWS_PER_DAY = 144


# ---------------------------------------------------------------- 纯函数
def blocked_oof_r2(X, y, n_folds=N_FOLDS, alpha=1.0):
    """按时间连续分块的 out-of-fold R²(禁止随机切分)。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < n_folds * 2 or X.shape[1] == 0:
        return float("nan")
    bounds = [round(k * n / n_folds) for k in range(n_folds + 1)]
    oof = np.full(n, np.nan)
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        te = np.zeros(n, dtype=bool)
        te[lo:hi] = True
        tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    ok = ~np.isnan(oof)
    if ok.sum() < 2:
        return float("nan")
    ss_res = float(((y[ok] - oof[ok]) ** 2).sum())
    ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def blocked_oof_accuracy(X, y_cls, n_folds=N_FOLDS):
    """T-b:多类 oracle 排序标签的 blocked OOF 准确率。"""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, dtype=float)
    y_cls = np.asarray(y_cls)
    n = len(y_cls)
    if n < n_folds * 2 or X.shape[1] == 0 or len(set(y_cls.tolist())) < 2:
        return float("nan")
    bounds = [round(k * n / n_folds) for k in range(n_folds + 1)]
    oof = np.full(n, -1)
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        te = np.zeros(n, dtype=bool)
        te[lo:hi] = True
        tr = ~te
        if len(set(y_cls[tr].tolist())) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
        m.fit(X[tr], y_cls[tr])
        oof[te] = m.predict(X[te])
    ok = oof >= 0
    return float((oof[ok] == y_cls[ok]).mean()) if ok.sum() else float("nan")


def incumbent_features(pred_d, short_rows=3, long_rows=144):
    """现役四特征(未做 Java 的 [0,1] 裁剪 —— 裁剪只会丢信息,不裁是对
    现役特征更有利的对照)。pred_d: (n, 144)。"""
    sm = pred_d[:, :short_rows].mean(axis=1)
    st = pred_d[:, short_rows - 1] - pred_d[:, 0]
    lm = pred_d[:, :long_rows].mean(axis=1)
    pt = pred_d[:, :long_rows].argmax(axis=1) / float(long_rows)
    return np.column_stack([sm, st, lm, pt])


def crossing_time(pred_i, pred_j, horizon=144):
    """t*: 第一个 pred_i[h] > pred_j[h] 的 h;从不超过记为 horizon。"""
    gt = pred_i[:, :horizon] > pred_j[:, :horizon]
    any_gt = gt.any(axis=1)
    first = gt.argmax(axis=1).astype(float)
    first[~any_gt] = float(horizon)
    return first


# ---------------------------------------------------------------- 主流程
def build(dat, lag=0):
    """返回 blind 特征、pred(可选滞后)、真值窗口。lag>0 表示 pred[h] 实际
    对应 T0+1+h+lag(诊断用)。"""
    T0 = dat["T0"]
    pred = dat["pred"].astype(float)           # (n, D, 144)
    series = dat["series"].astype(float)       # (D, n_rows)
    n, D, P = pred.shape
    n_rows = series.shape[1]
    # 真值窗口(契约:pred[h] = T0+1+h)
    idx = T0[:, None] + 1 + np.arange(P)[None, :]
    idx = np.clip(idx, 0, n_rows - 1)
    actual = np.stack([series[d][idx] for d in range(D)], axis=1)   # (n, D, 144)
    cur = np.stack([series[d][np.clip(T0, 0, n_rows - 1)]
                    for d in range(D)], axis=1)                     # (n, D)
    tot = cur.sum(axis=1, keepdims=True)
    share = np.divide(cur, tot, out=np.zeros_like(cur), where=tot > 0)
    rod = (T0 % ROWS_PER_DAY) / ROWS_PER_DAY
    time_feat = np.column_stack([np.sin(2 * np.pi * rod), np.cos(2 * np.pi * rod),
                                 T0 / float(n_rows)])
    if lag:
        pred = np.roll(pred, -lag, axis=2)
    return actual, cur, share, time_feat, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="calib/fr_timecap_year2021.npz")
    ap.add_argument("--json-out", default="../local_eval_rt/audit/fr_screen.json")
    ap.add_argument("--lag-scan", default="-12,-8,-6,-3,0,3,6,8,12")
    a = ap.parse_args()

    dat = np.load(a.npz)
    if int(dat["done"]) != 1:
        print(f"[FR] 注意:采集尚未完成(done=0),按现有 {len(dat['T0'])} 事件分析",
              flush=True)
    actual, cur, share, time_feat, pred = build(dat)
    n, D, P = pred.shape
    print(f"[FR] 事件 n={n}  DC={D}  pred_len={P}", flush=True)

    out = {"n_events": int(n), "n_dc": int(D), "horizons": HORIZONS,
           "dr2_min": DR2_MIN, "n_folds": N_FOLDS, "results": {}}

    for H in HORIZONS:
        y_all = actual[:, :, :H].mean(axis=2)             # (n, D)  T-a 目标
        for d in range(D):
            y = y_all[:, d]
            blind = np.column_stack([cur[:, d], share[:, d], time_feat])
            r2_blind = blocked_oof_r2(blind, y)
            pm = pred[:, d, :H].mean(axis=1)              # 预报的 H 窗均值
            cands = {
                "incumbent4": incumbent_features(pred[:, d, :]),
                "pred_mean_H": pm[:, None],
                "resid_pred_minus_persistence": (pm - cur[:, d])[:, None],
                "crossdc_diff": (pm - np.delete(
                    pred[:, :, :H].mean(axis=2), d, axis=1).mean(axis=1))[:, None],
                "crossdc_rank": (pred[:, :, :H].mean(axis=2).argsort(
                    axis=1).argsort(axis=1)[:, d]).astype(float)[:, None],
                "crossing_time": np.column_stack(
                    [crossing_time(pred[:, d, :], pred[:, j, :])
                     for j in range(D) if j != d]),
            }
            row = {"r2_blind": r2_blind}
            for name, F in cands.items():
                r2 = blocked_oof_r2(np.column_stack([blind, F]), y)
                row[name] = {"r2": r2, "dr2": r2 - r2_blind}
            out["results"][f"H{H}_DC{d}"] = row
            print(f"[FR] H={H:>3} DC{d}  blind R²={r2_blind:+.4f} | " +
                  "  ".join(f"{k}:{v['dr2']:+.4f}" for k, v in row.items()
                            if k != "r2_blind"), flush=True)

    # T-b:oracle 排序标签(仅在有绿电的 DC 间)
    out["ranking"] = {}
    for H in HORIZONS:
        y_cls = actual[:, :, :H].mean(axis=2).argmax(axis=1)
        blind = np.column_stack([cur, share, time_feat])
        acc_b = blocked_oof_accuracy(blind, y_cls)
        pm_all = pred[:, :, :H].mean(axis=2)
        acc_f = blocked_oof_accuracy(np.column_stack([blind, pm_all]), y_cls)
        acc_r = blocked_oof_accuracy(
            np.column_stack([blind, pm_all - cur]), y_cls)
        maj = float(np.bincount(y_cls).max() / len(y_cls))
        out["ranking"][f"H{H}"] = {"majority": maj, "acc_blind": acc_b,
                                   "acc_blind_plus_pred": acc_f,
                                   "acc_blind_plus_resid": acc_r,
                                   "dacc_pred": acc_f - acc_b,
                                   "dacc_resid": acc_r - acc_b}
        print(f"[FR] T-b H={H:>3} 多数类={maj:.4f} blind={acc_b:.4f} "
              f"+pred={acc_f:.4f}({acc_f-acc_b:+.4f}) "
              f"+resid={acc_r:.4f}({acc_r-acc_b:+.4f})", flush=True)

    # 最佳滞后诊断(非判据):只对关键候选 resid 在 H=3/24 上扫
    out["lag_diagnostic"] = {}
    for H in (3, 24):
        best = {}
        for L in [int(x) for x in a.lag_scan.split(",")]:
            _, cur_l, share_l, tf_l, pred_l = build(dat, lag=L)
            drs = []
            for d in range(D):
                y = actual[:, d, :H].mean(axis=1)
                blind = np.column_stack([cur_l[:, d], share_l[:, d], tf_l])
                pmL = pred_l[:, d, :H].mean(axis=1)
                r2b = blocked_oof_r2(blind, y)
                r2r = blocked_oof_r2(
                    np.column_stack([blind, (pmL - cur_l[:, d])[:, None]]), y)
                drs.append(r2r - r2b)
            best[str(L)] = float(np.mean(drs))
        bl = max(best, key=lambda k: best[k])
        out["lag_diagnostic"][f"H{H}"] = {"per_lag_mean_dr2": best,
                                          "best_lag": int(bl),
                                          "best_dr2": best[bl]}
        print(f"[FR] 滞后诊断 H={H}: 最佳 lag={bl} 平均ΔR²={best[bl]:+.4f} "
              f"(契约 lag=0 为 {best.get('0', float('nan')):+.4f})", flush=True)

    dr2s = [v[k]["dr2"] for v in out["results"].values()
            for k in v if k != "r2_blind" and np.isfinite(v[k]["dr2"])]
    out["max_dr2"] = float(max(dr2s)) if dr2s else float("nan")
    out["verdict"] = ("有救:存在 ΔR² ≥ 0.05 的候选" if out["max_dr2"] >= DR2_MIN
                      else "预报器/视界不带增量信息:全部候选 ΔR² < 0.05")
    print(f"\n[FR] 最大 ΔR² = {out['max_dr2']:+.4f} -> {out['verdict']}", flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("FR SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
