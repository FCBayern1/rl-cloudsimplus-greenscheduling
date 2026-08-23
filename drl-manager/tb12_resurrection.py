#!/usr/bin/env python3
"""TB12 复活测试:污染预报 -> clairvoyant 崩溃 -> 因果审计回退 -> 性能复活。

臂:
  clair_true     认证 clairvoyant(对真实未来规划)
  clair_corrupt  对污染序列规划(displaced=别处的风 / stale=2020 的风),无审计
  repaired       同污染,但因果残差监视器触发后回退 greenfollow:
                   监视器每行比较"上一行对本行的 1 步预报"与实测,
                   连续 m=3 行 |Δ|>eps 即触发(全部 tau<=t 信息)
  greenfollow    冻结盲参照(复活的目标线)
  nowait         基线

执行:释放时刻离线预计算,tb12_run.run_episode 仿真计分(接线哨兵在环)。
诊断集 T110+111;认证集 T114+115 不触碰(收兵条款)。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage15_continuous import coordinated_blind_contig, coordinated_clair_contig
from tb12_run import FrozenPolicies, load_scaled, run_episode, ROW_S

EPS_W = 1.0        # 残差阈(W):godeye 干净时残差恒 0,任何持续失配都是异常
M_CONSEC = 3       # 连续失配行数(检测延迟 = 3 行 = 30 分钟)


def corrupt_series(w, mode, off, turbines, year):
    if mode == "displaced":                      # 别处的风(+7000 行,循环)
        return np.roll(w, -7000)
    if mode == "stale":                          # 去年的同一位置
        return load_scaled(turbines, year - 1)
    raise ValueError(mode)


def trigger_row(w_true, w_corrupt, off, max_rows=300):
    """因果监视器:行 k 时比较 w_corrupt[off+k](昨步对本行的预报) 与实测
    w_true[off+k];连续 M 行失配 -> 在行 k 触发。返回触发行(相对 episode)。"""
    consec = 0
    for k in range(max_rows):
        if abs(float(w_corrupt[off + k]) - float(w_true[off + k])) > EPS_W:
            consec += 1
            if consec >= M_CONSEC:
                return k
        else:
            consec = 0
    return None                                   # 未触发(污染不可见)


def repaired_releases(jobs, w_true_ep, w_corrupt_ep, trig):
    """触发前:跟随 corrupt-clair 计划(已释放的覆水难收);
    触发后:未释放作业按 greenfollow 在【实测】风上重排(到达钳制到触发时刻)。"""
    plan = coordinated_clair_contig(jobs, w_corrupt_ep, 1.0)
    if trig is None:
        return plan
    t_trig = trig * ROW_S
    held = [i for i, r in enumerate(plan) if r >= t_trig]
    if not held:
        return plan
    jobs_held = [(j[0], j[1], j[2], max(j[3], t_trig), j[4], j[5])
                 for i, j in enumerate(jobs) if i in held]
    re_rel = coordinated_blind_contig(jobs_held, w_true_ep, 1.0)
    out = list(plan)
    for i, r in zip(held, re_rel):
        out[i] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", default="110,111")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--mode", default="displaced", choices=("displaced", "stale"))
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    from src.baselines.evaluate import load_config
    cfg = load_config("experiment_tb12_iso")
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg.setdefault("output_dir", "/tmp/tb12_gw")
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    arrivals = np.array([j[1] for j in art["jobs"]], float)
    pol = FrozenPolicies(art["rt_h"] * 3600.0, art["slack_h"] * 3600.0)
    turbines = tuple(int(x) for x in a.turbines.split(","))
    w = load_scaled(turbines, a.year)
    wc = corrupt_series(w, a.mode, 0, turbines, a.year)
    results = []
    for off in [int(x) for x in a.offsets.split(",")]:
        w_ep, wc_ep = w[off:off + 300], wc[off:off + 300]
        trig = trigger_row(w, wc, off)
        jobs = pol._mkjobs(arrivals)
        plans = {
            "nowait": [j[3] for j in jobs],
            "greenfollow": coordinated_blind_contig(jobs, w_ep, 1.0),
            "clair_true": coordinated_clair_contig(jobs, w_ep, 1.0),
            "clair_corrupt": coordinated_clair_contig(jobs, wc_ep, 1.0),
            "repaired": repaired_releases(jobs, w_ep, wc_ep, trig),
        }
        for arm, rel in plans.items():
            ges, steps = run_episode(cfg, off, rel, arrivals, turbines,
                                     ref_series=w)
            rec = {"offset": off, "arm": arm, "mode": a.mode,
                   "trigger_row": trig,
                   "carbon_kg": ges.get("total_carbon_emission_kg"),
                   "green_wh": ges.get("total_green_energy_wh"),
                   "finished": ges.get("total_finished_cloudlets"),
                   "ontime": ges.get("ontime_mi_share"), "steps": steps,
                   "releases": [float(r) for r in rel]}
            results.append(rec)
            print(f"[RES off={off:>6} {arm:>14}] carbon={rec['carbon_kg']:.5f} "
                  f"trig={trig} finished={rec['finished']} ontime={rec['ontime']:.3f}",
                  flush=True)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(
            {"mode": a.mode, "eps_w": EPS_W, "m_consec": M_CONSEC,
             "planning_turbines": list(turbines),
             "environment_turbines": list(turbines),
             "results": results}, indent=1))
    print("RESURRECTION RUN DONE", flush=True)


if __name__ == "__main__":
    main()
