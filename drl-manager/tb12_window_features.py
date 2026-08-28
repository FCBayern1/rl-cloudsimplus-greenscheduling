#!/usr/bin/env python3
"""TB12 窗长特征探针(预注册 PREREG_WINDOW_FEATURES.md,阈值先冻)。

回答:现役四特征都不表达"窗长",而 TB12 的决策问的是「这个绿窗**装不装得下**
我、值不值得等下一个」。补上窗长特征,分歧作业上的判别力起不起来?

仪器沿用表示审计 Run 2(每作业等权 + 作业内类别平衡 BCE,G4 健康门已过),
同一批 60 offset / 5 时间块 / blocked OOF / 42 个分歧作业 / 同 ck0 与超参。
fc 侧 = q_fc ⊕ 窗长特征;nofc 侧 = q_nofc(不变,对照)。

阈值(**冻结**):θ_dc = 该 DC 在训练分布 T100+101/2021 **全年**绿电的 60 分位数。
窗长特征全部按**真实未来风**计算(godeye 口径)—— 若连真实未来都撑不起判别力,
预报器版本更不可能。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import load_scaled, ROW_S  # noqa: E402

THRESHOLD_PCTL = 60.0        # 冻结
W1_MEDIAN_MIN = 0.05         # 冻结判据
W2_ACC_GAIN_MIN = 0.03       # 冻结判据


# ---------------------------------------------------------------- 纯函数
def green_threshold(series, pctl=THRESHOLD_PCTL):
    """θ = 全年绿电序列的 pctl 分位数(冻结定义)。"""
    return float(np.percentile(np.asarray(series, dtype=float), pctl))


def window_features_at(series, t, theta, runtime_rows):
    """在行 t 上,按阈值 theta 给出五个窗长特征。
    series 为该 DC 的全年绿电序列(W);t 为绝对行号;runtime_rows 为作业时长。"""
    n = len(series)
    in_window = series[t % n] > theta
    # 当前窗剩余
    rem = 0
    if in_window:
        u = t
        while u - t < 4 * max(1, runtime_rows) and series[u % n] > theta:
            rem += 1
            u += 1
    # 下一个窗起点
    onset = 0
    if not in_window:
        u, cap = t, 8 * max(1, runtime_rows)
        while u - t < cap and series[u % n] <= theta:
            u += 1
        onset = u - t if u - t < cap else cap
    else:
        u = t + rem
        cap = 8 * max(1, runtime_rows)
        while u - t < cap and series[u % n] <= theta:
            u += 1
        onset = u - t if u - t < cap else cap
    # 下一个窗的 时长 × 平均强度
    v, tot = t + onset, 0.0
    size_len = 0
    while size_len < 4 * max(1, runtime_rows) and series[v % n] > theta:
        tot += float(series[v % n])
        size_len += 1
        v += 1
    nxt_size = tot / 1000.0                       # kW·行,压量纲
    rt = max(1.0, float(runtime_rows))
    return np.array([float(rem), float(onset), nxt_size,
                     rem / rt, (size_len / rt)], dtype=np.float64)


def probe_verdict(median_gap, acc_fc, acc_nofc, degen_fc_ok, degen_nofc_ok,
                  median_min=W1_MEDIAN_MIN, gain_min=W2_ACC_GAIN_MIN):
    """W1/W2/W3 机械合成(判据冻结,不因结果回调)。"""
    gain = acc_fc - acc_nofc
    w1 = bool(median_gap >= median_min)
    w2 = bool(gain >= gain_min)
    w3 = bool(degen_fc_ok and degen_nofc_ok)
    ok = bool(w1 and w2 and w3)
    return ok, {
        "W1_median_gap": {"ok": w1, "median": float(median_gap), "min": median_min},
        "W2_acc_gain": {"ok": w2, "acc_fc": float(acc_fc),
                        "acc_nofc": float(acc_nofc), "gain": float(gain),
                        "min": gain_min},
        "W3_no_degeneracy": {"ok": w3, "fc": bool(degen_fc_ok),
                             "nofc": bool(degen_nofc_ok)},
        "ALL_PASS": ok,
    }


def window_stats(series, theta, cap=20000):
    """窗长分布(附加报告项):连续 > θ 的段长。"""
    s = np.asarray(series[:cap], dtype=float) > theta
    lens, run = [], 0
    for v in s:
        if v:
            run += 1
        elif run:
            lens.append(run)
            run = 0
    if run:
        lens.append(run)
    a = np.asarray(lens, dtype=float)
    if not len(a):
        return {"n_windows": 0}
    return {"n_windows": int(len(a)), "median_rows": float(np.median(a)),
            "mean_rows": float(a.mean()),
            "cv": float(a.std() / max(1e-9, a.mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", default="100,101")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--json-out", default="../local_eval_rt/audit/tb12_window_stats.json")
    a = ap.parse_args()
    turb = tuple(int(x) for x in a.turbines.split(","))
    series = load_scaled(turb, a.year)
    theta = green_threshold(series)
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    rt_rows = int(round(art["rt_h"] * 3600.0 / ROW_S))
    st = window_stats(series, theta)
    st.update({"theta_w": theta, "pctl": THRESHOLD_PCTL,
               "job_runtime_rows": rt_rows,
               "median_window_over_runtime": st.get("median_rows", 0) / max(1, rt_rows)})
    print(f"[WIN] θ({THRESHOLD_PCTL:.0f}分位) = {theta:.4f} W", flush=True)
    print(f"[WIN] 窗数={st['n_windows']} 中位窗长={st.get('median_rows',0):.1f} 行 "
          f"CV={st.get('cv',float('nan')):.3f}", flush=True)
    print(f"[WIN] 作业时长={rt_rows} 行  中位窗长/作业时长 = "
          f"{st['median_window_over_runtime']:.3f}", flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print("WINDOW STATS DONE", flush=True)


if __name__ == "__main__":
    main()
