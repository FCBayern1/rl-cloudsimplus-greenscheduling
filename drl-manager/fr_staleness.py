#!/usr/bin/env python3
"""预报特征筛选 附加诊断:**缓存陈旧**对增量信息的损耗(工单 ad79b80)。

fr_screen.py 只在**生成事件**上取样(陈旧=0),量的是"预报本身有多少信息"。
但 `forecast_every: 6` 意味着策略在两次生成之间看到的是**最旧可达 5 行
(50 分钟)**的预报。在 30 分钟视界上零陈旧 ΔR² 仅 ~0.045 的前提下,陈旧
足以把它抹平 —— 这是可证伪、且可修(把 forecast_every 调小)的假设。

做法:对陈旧 k∈{0..5},站在 T0+k 这一步:
  - 特征仍来自 T0 生成的那份预报,窗口右移 k(h = k .. k+H-1);
  - blind 用 T0+k 的当前绿电;
  - 目标是 T0+k 之后 H 行的真实均值。
口径与主表一致:blocked OOF,5 个时间连续块,禁止随机切分。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fr_screen import ROWS_PER_DAY, blocked_oof_r2  # noqa: E402

HORIZONS = [3, 6, 12, 24]
STALENESS = [0, 1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="calib/fr_timecap_year2021.npz")
    ap.add_argument("--json-out", default="../local_eval_rt/audit/fr_staleness.json")
    a = ap.parse_args()
    d = np.load(a.npz)
    T0 = d["T0"].astype(int)
    pred = d["pred"].astype(float)          # (n, D, 144) 对应 T0+1+h
    series = d["series"].astype(float)      # (D, n_rows)
    n, D, P = pred.shape
    n_rows = series.shape[1]
    out = {"n_events": int(n), "horizons": HORIZONS, "staleness": STALENESS,
           "results": {}}
    print(f"[STALE] n={n} DC={D}", flush=True)

    for H in HORIZONS:
        for k in STALENESS:
            if k + H > P:
                continue
            drs = []
            for dc in range(D):
                t = T0 + k                                   # 策略所处的步
                cur = series[dc][np.clip(t, 0, n_rows - 1)]
                idx = np.clip(t[:, None] + 1 + np.arange(H)[None, :],
                              0, n_rows - 1)
                y = series[dc][idx].mean(axis=1)             # 真值:T0+k 之后 H 行
                tot = np.stack([series[j][np.clip(t, 0, n_rows - 1)]
                                for j in range(D)], axis=1).sum(axis=1)
                share = np.divide(cur, tot, out=np.zeros_like(cur), where=tot > 0)
                rod = (t % ROWS_PER_DAY) / ROWS_PER_DAY
                blind = np.column_stack([cur, share, np.sin(2*np.pi*rod),
                                         np.cos(2*np.pi*rod), t / float(n_rows)])
                pm = pred[:, dc, k:k + H].mean(axis=1)       # 陈旧预报的对应窗口
                r2b = blocked_oof_r2(blind, y)
                r2f = blocked_oof_r2(np.column_stack([blind, pm[:, None]]), y)
                drs.append(r2f - r2b)
            out["results"][f"H{H}_k{k}"] = {
                "per_dc_dr2": [float(x) for x in drs],
                "mean_dr2": float(np.mean(drs))}
            print(f"[STALE] H={H:>3} 陈旧={k} 行  平均ΔR²={np.mean(drs):+.4f}  "
                  f"逐DC {[round(x,4) for x in drs]}", flush=True)
        base = out["results"][f"H{H}_k0"]["mean_dr2"]
        worst = out["results"][f"H{H}_k5"]["mean_dr2"] if f"H{H}_k5" in out["results"] else float("nan")
        out["results"][f"H{H}_summary"] = {
            "dr2_fresh": base, "dr2_stalest": worst,
            "retained_frac": (worst / base) if base > 0 else float("nan")}
        print(f"[STALE] H={H:>3} 新鲜 {base:+.4f} -> 最旧 {worst:+.4f} "
              f"(保留 {100*worst/base if base>0 else float('nan'):.1f}%)", flush=True)

    pathlib.Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("STALENESS DONE", flush=True)


if __name__ == "__main__":
    main()
