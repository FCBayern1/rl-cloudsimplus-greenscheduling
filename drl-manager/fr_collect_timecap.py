#!/usr/bin/env python3
"""预报特征筛选 步骤 1:全年采集 TimeCAP 预测(工单 ad79b80)。

只在**生成事件**上采样(零缓存陈旧),契约:输入截至 T0 的最近 seq_len 行,
输出 pred[h] = T0+1+h 行的功率。

为什么顺序跑全年而不是跳转采样:实测 `warmup(start_step=X)` 后跳到同一绝对步,
逐涡轮预测与顺序跑最大差 **807 kW** —— 跳转会改变预测,采样会静默污染。
故顺序覆盖全年(~3.9h CPU),同时得到全年均匀的 ~8760 个事件,正好够按年
切 5 个连续块做 blocked OOF。

产出 npz:T0(生成步)、pred(n, n_dc, 144)、以及各 DC 的真实合成序列
(真值/当前绿电/时间特征在分析阶段从序列派生,便于改定义而不必重跑)。
增量落盘,崩了不丢。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from prediction.timecap_godeye_provider import TimeCAPGodEyeProvider  # noqa: E402

CSV_DIR = REPO.parent / "cloudsimplus-gateway/src/main/resources/windProduction/split"
CKPT = REPO / "timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth"
# knSV3b 的 DC/涡轮/时区映射(config_C.yml 实读):DC3/DC4 无涡轮,恒零绿电,不入分析
DC_ASSIGN = {0: [12, 36], 1: [95, 91], 2: [96]}
TZ = {0: 0, 1: 18, 2: 54}
YEAR = 2021
PRED = 144
OUT = REPO / "calib/fr_timecap_year2021.npz"
FLUSH_EVERY = 250


def main():
    patv = {t: pd.read_csv(CSV_DIR / f"Turbine_{t}_{YEAR}.csv")["Patv"].to_numpy(float)
            for v in DC_ASSIGN.values() for t in v}
    n_rows = min(len(a) for a in patv.values())
    # 各 DC 的真实合成序列(已按各自时区行偏移对齐)
    series = np.zeros((len(DC_ASSIGN), n_rows), dtype=np.float64)
    for d, tids in DC_ASSIGN.items():
        for t in tids:
            a = patv[t]
            series[d] += np.roll(a, -TZ[d])[:n_rows]
    paths = {t: str(CSV_DIR / f"Turbine_{t}_{YEAR}.csv")
             for v in DC_ASSIGN.values() for t in v}
    prov = TimeCAPGodEyeProvider(
        dc_assignments=DC_ASSIGN, turbine_csv_paths=paths,
        checkpoint_path=str(CKPT), forecast_every=6, device="cpu",
        csv_start_offset=0, dc_tz_offsets=TZ,
        simulation_warmup_rows=0, forecast_shift=None)
    print(f"[FR] rows={n_rows} seq_len={prov.seq_len} pred_len={PRED} "
          f"DC={list(DC_ASSIGN)}", flush=True)

    t0s, preds = [], []
    last_id, t_start = None, time.time()
    end = n_rows - PRED - 2                     # 保证真值窗口完整
    for T in range(prov.seq_len, end):
        prov.step_and_get(T)
        raw = prov._last_per_t_pred
        if raw is None or id(raw) == last_id:
            continue
        last_id = id(raw)
        p = np.zeros((len(DC_ASSIGN), PRED), dtype=np.float32)
        ok = True
        for d, tids in DC_ASSIGN.items():
            for t in tids:
                arr = raw.get(t)
                if arr is None or arr.size < PRED:
                    ok = False
                    break
                p[d] += arr[:PRED]
            if not ok:
                break
        if not ok:
            continue
        t0s.append(T)
        preds.append(p)
        if len(t0s) % FLUSH_EVERY == 0:
            np.savez_compressed(OUT, T0=np.asarray(t0s, dtype=np.int64),
                                pred=np.asarray(preds, dtype=np.float32),
                                series=series.astype(np.float32),
                                dc_ids=np.asarray(sorted(DC_ASSIGN)),
                                tz=np.asarray([TZ[d] for d in sorted(DC_ASSIGN)]),
                                pred_len=PRED, n_rows=n_rows, done=0)
            el = time.time() - t_start
            print(f"[FR] {len(t0s)} 事件  T={T}/{end}  已用 {el/60:.1f} min  "
                  f"预计总 {el/max(1,T-prov.seq_len)*(end-prov.seq_len)/3600:.2f} h",
                  flush=True)
    np.savez_compressed(OUT, T0=np.asarray(t0s, dtype=np.int64),
                        pred=np.asarray(preds, dtype=np.float32),
                        series=series.astype(np.float32),
                        dc_ids=np.asarray(sorted(DC_ASSIGN)),
                        tz=np.asarray([TZ[d] for d in sorted(DC_ASSIGN)]),
                        pred_len=PRED, n_rows=n_rows, done=1)
    print(f"[FR] COLLECT DONE 事件={len(t0s)} 用时 "
          f"{(time.time()-t_start)/3600:.2f} h -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
