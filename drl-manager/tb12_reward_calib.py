#!/usr/bin/env python3
"""P1/P2(Codex 裁定):per-step 碳分布标定 + 奖励真值表,零训练。

- 环境 = experiment_tb12_rl_fc(未来修复训练的同一物理),gradlew 路径。
- 风 = 训练分布 T100+101/2021(= rl block csv_year;fixed_max 冻结自
  训练分布,不碰 held-out 判决对 T110/111+)。
- 逐步碳 = info.global_energy_stats.total_carbon_emission_kg 的一阶差分
  (与判决同一本账,天然满足恒等式 3)。
- 四轨:nowait / greenfollow / clair(FrozenPolicies)+ always_defer
  (rel=+inf,交给 latest_start backstop —— 正是 RL 坍缩解的确定性替身)。
- 产出:分布 p50/p90/p99/max、建议 fixed_max(=max/1.5,对 3.0 封顶留 2x
  余量)、无封顶等比奖励下的真值表(cap rate、排序、ontime)。只提案不改配置。
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import FrozenPolicies, load_scaled, ROW_S  # noqa: E402

CAP = 3.0  # Java CARBON_RATIO_MAX,两种归一化模式都套


def propose_fixed_max(step_kgs, cap=CAP, headroom=2.0):
    """建议分母:最坏步比值 = cap/headroom(留 headroom 倍到封顶)。"""
    mx = float(np.max(step_kgs))
    return mx * headroom / cap


def truth_table(per_arm_kg, per_arm_steps, fixed_max, cap=CAP):
    """无折扣 ΣĈ(含封顶)与物理 kg 的同序检查 + cap 命中率。"""
    rows = {}
    for arm, kgs in per_arm_kg.items():
        ratios = np.asarray(kgs) / fixed_max
        rows[arm] = {
            "phys_kg": float(np.sum(kgs)),
            "sum_chat": float(np.sum(np.minimum(ratios, cap))),
            "cap_hits": int(np.sum(ratios > cap)),
            "max_ratio": float(ratios.max()) if len(kgs) else 0.0,
            "steps": per_arm_steps[arm],
        }
    by_phys = sorted(rows, key=lambda a: rows[a]["phys_kg"])
    by_rew = sorted(rows, key=lambda a: rows[a]["sum_chat"])
    rows["_order_match"] = by_phys == by_rew
    rows["_order_phys"] = by_phys
    return rows


def assert_year_consistency(cfg, cli_year):
    """Codex 2026-08-25:experiment csv_year == CLI year == 离线参考序列年份,
    启动前锁死,不允许只靠运行期接线哨兵事后发现年份漂移。"""
    cfg_year = int(cfg.get("csv_year", -1))
    if cfg_year != int(cli_year):
        sys.exit(f"年份不一致: experiment csv_year={cfg_year} != --year {cli_year}"
                 " —— 标定必须对齐训练分布")


def episode_step_kg(env_cfg, off, releases, arrivals, turbines, ref_series):
    """与 tb12_run.run_episode 同构的循环,但逐步记录累计碳的差分。
    不改判决 harness 本体(已认证,保持字节不动)。"""
    import copy
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    cfg = copy.deepcopy(env_cfg)
    cfg["green_episode_offset_range"] = 0
    for dc in cfg["datacenters"]:
        dc["time_zone_offset_rows"] = int(off)
        if dc.get("turbine_ids"):
            dc["turbine_ids"] = list(turbines)
    env = HierarchicalMultiDCEnv(cfg)
    try:
        obs, _ = env.reset(seed=1)
        g0 = float(np.asarray(obs["global"]["dc_current_green_power_w"]).reshape(-1)[0])
        expect = float(ref_series[int(off)])
        if abs(g0 - expect) > 1e-3:
            sys.exit(f"接线哨兵失败 off={off}: {g0} != {expect}")
        batch = env.global_routing_batch_size
        order = np.argsort(arrivals)
        rel_sorted = [releases[i] for i in order]
        released, done, t = 0, False, 0
        prev_kg, step_kgs, ges = 0.0, [], {}
        while not done and t < 300:
            g = obs["global"]
            mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
            now = t * ROW_S
            acts, k = [], 0
            for i in range(batch):
                if mi[i] <= 0:
                    acts.append(0)
                    continue
                j = released + k
                hold = (j < len(rel_sorted) and now < rel_sorted[j] - 1e-9)
                acts.append(1 if hold else 0)
                k += 1
            released += sum(1 for i in range(batch) if mi[i] > 0 and acts[i] == 0)
            local = {0: drain_action(env.get_local_action_masks(0))}
            obs, _, term, trunc, info = env.step({"global": acts, "local": local})
            done = term or trunc
            ges = info.get("global_energy_stats") or ges
            cur = float(ges.get("total_carbon_emission_kg", prev_kg))
            step_kgs.append(cur - prev_kg)
            prev_kg = cur
            t += 1
        # 自检:差分和 == 终值(同一本账,应逐位)
        tot = float(ges.get("total_carbon_emission_kg", 0.0))
        if abs(sum(step_kgs) - tot) > 1e-9 + 1e-6 * abs(tot):
            sys.exit(f"差分自检失败 off={off}: {sum(step_kgs)} != {tot}")
        return np.asarray(step_kgs), ges, t
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_tb12_rl_fc")
    ap.add_argument("--turbines", default="100,101")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--offsets", default="4000,12000,20000,28000,36000,44000")
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    from src.baselines.evaluate import load_config
    cfg = load_config(a.experiment)
    assert_year_consistency(cfg, a.year)
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg.setdefault("output_dir", "/tmp/tb12_gw")
    pathlib.Path("/tmp/tb12_gw").mkdir(exist_ok=True)

    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    arrivals = np.array([j[1] for j in art["jobs"]], dtype=float)
    pol = FrozenPolicies(art["rt_h"] * 3600.0, art["slack_h"] * 3600.0)
    turbines = tuple(int(x) for x in a.turbines.split(","))
    w = load_scaled(turbines, a.year)

    offsets = [int(x) for x in a.offsets.split(",")]
    arms = ["nowait", "greenfollow", "clair"]
    all_kg, per_arm_kg, per_arm_steps, recs = [], {}, {}, []
    for arm in arms + ["always_defer"]:
        per_arm_kg[arm], per_arm_steps[arm] = [], 0
    for off in offsets:
        w_ep = w[off:off + 300]
        plans = {arm: pol.releases(arm, w_ep, arrivals) for arm in arms}
        plans["always_defer"] = [1e12] * len(arrivals)
        for arm, rel in plans.items():
            kgs, ges, t = episode_step_kg(cfg, off, rel, arrivals,
                                          turbines, ref_series=w)
            fin, ot = ges.get("total_finished_cloudlets"), ges.get("ontime_mi_share")
            all_kg.append(kgs)
            per_arm_kg[arm].extend(kgs.tolist())
            per_arm_steps[arm] += t
            recs.append({"offset": off, "arm": arm, "steps": t,
                         "carbon_kg": float(kgs.sum()), "finished": fin,
                         "ontime": ot,
                         "env_cap_count": ges.get("global_carbon_cap_count"),
                         "env_max_ratio": ges.get("global_carbon_max_ratio")})
            print(f"[CALIB off={off:>6} {arm:>13}] kg={kgs.sum():.5f} "
                  f"step_kg max={kgs.max():.3e} finished={fin} ontime={ot:.3f} "
                  f"steps={t}", flush=True)

    flat = np.concatenate(all_kg)
    dist = {q: float(np.percentile(flat, q)) for q in (50, 90, 99, 99.9)}
    dist["max"] = float(flat.max())
    prop = propose_fixed_max(flat)
    tt = truth_table(per_arm_kg, per_arm_steps, prop)
    print(f"\nper-step kg 分布: p50={dist[50]:.3e} p90={dist[90]:.3e} "
          f"p99={dist[99]:.3e} max={dist['max']:.3e}")
    print(f"现行 fixed_max=2e-05 → max 比值 {dist['max']/2e-05:.1f}(封顶 3.0)")
    print(f"建议 fixed_max={prop:.6e} → max 比值 {dist['max']/prop:.2f},"
          f"封顶余量 2x")
    print(f"真值表同序: {tt['_order_match']}  物理排序: {tt['_order_phys']}")
    for arm in tt["_order_phys"]:
        r = tt[arm]
        print(f"  {arm:>13} phys={r['phys_kg']:.5f} ΣĈ={r['sum_chat']:.2f} "
              f"cap_hits={r['cap_hits']} max_ratio={r['max_ratio']:.2f}")
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"experiment": a.experiment, "turbines": list(turbines),
         "year": a.year, "offsets": offsets, "distribution": dist,
         "current_fixed_max": 2e-05, "proposed_fixed_max": prop,
         "truth_table": tt, "results": recs}, indent=1))
    print("REWARD CALIB DONE", flush=True)


if __name__ == "__main__":
    main()
