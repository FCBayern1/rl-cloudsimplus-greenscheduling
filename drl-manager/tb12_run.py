#!/usr/bin/env python3
"""tb12 五臂 runner:离线预计算释放时刻 -> 仿真执行 + 物理计分。

臂(全部作用于同一 episode 的同一 trace/风偏移,天然配对):
  nowait        到达即放
  greenfollow   因果协调盲(绿电跟随,stage15)
  hazard        窗龄 hazard 盲,theta 于 2020 冻结
  dpcont        连续功率 DP 盲(2020 冻结表)
  clair         在线到达 clairvoyant(可读未来风,不可读未来作业)

执行器把 batch 槽位按【到达序】映射到作业(broker 队列序 = 到达序;
作业同 MI 同 rt,错配只影响 ontime 归属,不影响碳;烟测用 ontime 验证)。
全局动作:0 = 路由 DC0,1(=num_dc) = 持有。
"""
import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage15_continuous import (binarize, blind_release, coordinated_blind_contig,
                                coordinated_clair_contig, fit_prob_table)
from dp_blind import fit_tables_cont, solve_dp_cont, dp_cont_releases

_REPO = pathlib.Path(__file__).resolve().parent.parent
_W = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ROW_S = 600.0
SCALE = 9.05562658195e-5          # 2020 冻结
P_JOB = 28.75
W_PER_PE_EFF = P_JOB              # 每作业功率(pes_eff=1 口径)
THETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.1)


def load_scaled(turbines, year):
    tot = None
    for t in turbines:
        p = np.array([float(r["power_kw"]) for r in
                      csv.DictReader(open(_W / f"Turbine_{t}_{year}.csv"))])
        tot = p if tot is None else tot[:len(p)] + p[:len(tot)]
    return tot * 1000.0 * SCALE


class FrozenPolicies:
    """2020 拟合冻结的盲参数(theta、F、DP 表)。judgment 年份只读。"""

    def __init__(self, rt_s, slack_s, cal_turbines=(100, 101)):
        w20 = load_scaled(cal_turbines, 2020)
        self.rt, self.slack = rt_s, slack_s
        self.thr = P_JOB                     # 窗口 = 绿电足以整机供电
        s20, e20 = binarize(w20, self.thr)
        self.F = fit_prob_table(e20 - s20, rt_s)
        # theta 于 2020 选定
        jobs20 = self._mkjobs(np.arange(5) * 4 * 3600.0)
        best = (1e30, None)
        from stage15_continuous import fluid_carbon
        for th in THETAS:
            rel = [blind_release(s20, e20, self.F, j[3], j[2], j[3] + j[4], th)
                   for j in jobs20]
            c = fluid_carbon(jobs20, rel, w20, 1.0)[0]
            if c < best[0]:
                best = (c, th)
        self.theta = best[1]
        ct, P, _ = fit_tables_cont(w20, 1.0, P_JOB, self.thr, rt_s)
        _, self.dp_policy = solve_dp_cont(ct, P, int(slack_s // ROW_S))

    def _mkjobs(self, arrivals):
        return [(self.rt * 40000.0, P_JOB / 2.541, self.rt, float(a),
                 self.slack, 0) for a in arrivals]

    def releases(self, arm, w_ep, arrivals):
        """w_ep: 该 episode 的 scaled 序列(行 0 = episode 第 0 步)。"""
        jobs = self._mkjobs(arrivals)
        if arm == "nowait":
            return [j[3] for j in jobs]
        if arm == "greenfollow":
            return coordinated_blind_contig(jobs, w_ep, 1.0)
        if arm == "hazard":
            s_, e_ = binarize(w_ep, self.thr)
            return [blind_release(s_, e_, self.F, j[3], j[2],
                                  j[3] + j[4], self.theta) for j in jobs]
        if arm == "dpcont":
            return dp_cont_releases(jobs, w_ep, 1.0, P_JOB, self.thr,
                                    self.dp_policy)
        if arm == "clair":
            return coordinated_clair_contig(jobs, w_ep, 1.0)
        raise ValueError(arm)


def run_episode(env_cfg, offset_rows, release_times, arrivals, max_steps=300):
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    cfg = dict(env_cfg)
    cfg["green_episode_offset_range"] = 0
    for dc in cfg["datacenters"]:
        dc["time_zone_offset_rows"] = int(offset_rows)   # 偏移经涡轮 tz 注入
    env = HierarchicalMultiDCEnv(cfg)
    try:
        obs, _ = env.reset(seed=1)
        done, t, ges = False, 0, {}
        batch = env.global_routing_batch_size
        order = np.argsort(arrivals)                     # 槽位 = 到达序
        rel_sorted = [release_times[i] for i in order]
        released = 0
        while not done and t < max_steps:
            g = obs["global"]
            mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
            now = t * ROW_S
            acts = []
            k = 0
            for i in range(batch):
                if mi[i] <= 0:
                    acts.append(0)
                    continue
                j = released + k                          # 队列中第 k 个未释放作业
                hold = (j < len(rel_sorted) and now < rel_sorted[j] - 1e-9)
                acts.append(1 if hold else 0)
                k += 1
            released += sum(1 for i in range(batch)
                            if mi[i] > 0 and acts[i] == 0)
            local = {0: drain_action(env.get_local_action_masks(0))}
            obs, _, term, trunc, info = env.step({"global": acts, "local": local})
            done = term or trunc
            ges = info.get("global_energy_stats") or ges
            t += 1
        return ges, t
    finally:
        env.close()


def verify_frozen_jar():
    """P0.5-2(Codex):正式跑必须显式 GATEWAY_LIBS 指向冻结 installDist,
    且实际 jar SHA256 与 calib/tb12_jar_manifest.json 一致,不一致立即退出。"""
    import hashlib, os
    man_p = pathlib.Path(__file__).resolve().parent / "calib/tb12_jar_manifest.json"
    if not man_p.exists():
        sys.exit("tb12_jar_manifest.json 不存在 —— 先冻结 jar 再跑正式格")
    man = json.loads(man_p.read_text())
    libs = os.environ.get("GATEWAY_LIBS", "").strip()
    if not libs:
        sys.exit("正式跑必须设置 GATEWAY_LIBS 指向冻结 installDist(P0.5-2)")
    jar = pathlib.Path(libs) / "cloudsimplus-gateway.jar"
    if not jar.is_file():
        sys.exit(f"GATEWAY_LIBS 下无 cloudsimplus-gateway.jar: {libs}")
    sha = hashlib.sha256(jar.read_bytes()).hexdigest()
    if sha != man["jar_sha256"]:
        sys.exit(f"jar SHA 不一致!\n  实际   {sha}\n  冻结   {man['jar_sha256']}")
    print(f"[TB12] jar SHA 哨兵通过: {sha[:16]}… (source={man['source_code_commit'][:12]})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_tb12_smoke")
    ap.add_argument("--turbines", default="100,101")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--offsets", default="8000,26000,44000")
    ap.add_argument("--arms", default="nowait,greenfollow,hazard,dpcont,clair")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    if os.environ.get("TB12_REQUIRE_FROZEN_JAR", "1") != "0":
        verify_frozen_jar()

    from src.baselines.evaluate import load_config
    cfg = load_config(a.experiment)
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg.setdefault("output_dir", "/tmp/tb12_gw")
    pathlib.Path("/tmp/tb12_gw").mkdir(exist_ok=True)

    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    arrivals = np.array([j[1] for j in art["jobs"]], dtype=float)
    rt_s, slack_s = art["rt_h"] * 3600.0, art["slack_h"] * 3600.0
    pol = FrozenPolicies(rt_s, slack_s)
    print(f"[TB12] frozen theta*={pol.theta} thr={pol.thr}W scale={SCALE}",
          flush=True)

    turbines = tuple(int(x) for x in a.turbines.split(","))
    w_year = load_scaled(turbines, a.year)
    results = []
    for off in [int(x) for x in a.offsets.split(",")]:
        w_ep = w_year[off:off + 300]
        for arm in a.arms.split(","):
            rel = pol.releases(arm, w_ep, arrivals)
            ges, steps = run_episode(cfg, off, rel, arrivals)
            rec = {"offset": off, "arm": arm, "steps": steps,
                   "carbon_kg": ges.get("total_carbon_emission_kg"),
                   "green_wh": ges.get("total_green_energy_wh"),
                   "brown_wh": ges.get("total_brown_energy_wh"),
                   "energy_wh": ges.get("total_energy_wh"),
                   "finished": ges.get("total_finished_cloudlets"),
                   "ontime": ges.get("ontime_mi_share"),
                   "miss": ges.get("deadline_miss_rate"),
                   "releases": [float(r) for r in rel]}
            results.append(rec)
            print(f"[TB12 off={off:>6} {arm:>11}] carbon={rec['carbon_kg']:.5f} "
                  f"green={rec['green_wh']:.1f} finished={rec['finished']} "
                  f"ontime={rec['ontime']:.3f} steps={steps}", flush=True)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(
            {"artifact": art, "results": results}, indent=1))
    print("TB12 RUN DONE", flush=True)


if __name__ == "__main__":
    main()
