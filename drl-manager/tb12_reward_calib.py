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


def audit_defer_charges(slot_events, ledger_sum=None, n_jobs=None,
                        base=0.5, tol=1e-6):
    """Codex Step 1 新哨兵(v3 修复语义),机械正确版(2026-08-26 修订):

    槽事件的作业归因只在**首次强制路由之前**有效(backstop 强制路由不经过
    runner 的 released 计数,之后队列前移,槽→作业映射失效;首次强制的时刻
    由事件流中的 forced_mark 标记)。route 槽的 per-slot reward 混有 per-action
    差分奖励,不能用于等待成本守恒 —— 守恒改用 Java 自己的
    defer_urgency_cost_sum 账本(只含 base+urgency)。

    S1 首次 defer 精确 −base(信任窗内,首增量=0 ⇒ 恰为 −base);
    S2 重复 defer 不重复收 base(信任窗内,合法增量每步 ≤ w·Δt/window=0.333<base);
    S3 账本守恒/重现次数不变性:TB12 几何下(释放/强制点距 deadline 均 >
       urgency 窗)全部增量为 0 ⇒ ledger_sum 必须是 −base 的整数倍,倍数 n 落在
       [信任窗内见过 defer 的作业数, n_jobs];从未 defer 的作业贡献 0 由此覆盖。
    返回 (ok, detail)。"""
    cutoff = min((e["t"] for e in slot_events if e.get("forced_mark")),
                 default=float("inf"))
    trusted = [e for e in slot_events
               if not e.get("forced_mark") and e["t"] < cutoff]
    by_job = {}
    for e in trusted:
        by_job.setdefault(e["job"], []).append(e)
    fails, detail = [], {}
    for j, evs in sorted(by_job.items()):
        defers = [e for e in evs if e["act"] == "defer"]
        d = {"n_defer_trusted": len(defers)}
        if defers:
            first = defers[0]["r"]
            d["first_defer_r"] = first
            if abs(first - (-base)) > tol:
                fails.append((j, "first_defer_not_minus_base", first))
            for e in defers[1:]:
                if e["r"] <= -base + tol:
                    fails.append((j, "repeat_base_charge", e["r"]))
        detail[j] = d
    n_deferred_trusted = sum(1 for j in by_job
                             if any(e["act"] == "defer" for e in by_job[j]))
    if ledger_sum is not None:
        mult = -ledger_sum / base
        d_l = {"ledger_sum": ledger_sum, "base_multiple": mult}
        if abs(mult - round(mult)) > tol:
            fails.append(("ledger", "not_integer_multiple_of_base", ledger_sum))
        else:
            n = int(round(mult))
            hi = n_jobs if n_jobs is not None else float("inf")
            if not (n_deferred_trusted <= n <= hi):
                fails.append(("ledger", "multiple_out_of_range",
                              (n, n_deferred_trusted, hi)))
        detail["_ledger"] = d_l
    return len(fails) == 0, {"fails": fails, "per_job": detail,
                             "trust_cutoff_t": cutoff}


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
        prev_kg, step_kgs, ges, slot_events = 0.0, [], {}, []
        while not done and t < 300:
            g = obs["global"]
            mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
            now = t * ROW_S
            acts, k, slotmap = [], 0, {}
            for i in range(batch):
                if mi[i] <= 0:
                    acts.append(0)
                    continue
                j = released + k
                slotmap[i] = j                      # 槽 i -> 到达序作业 j
                hold = (j < len(rel_sorted) and now < rel_sorted[j] - 1e-9)
                acts.append(1 if hold else 0)
                k += 1
            released += sum(1 for i in range(batch) if mi[i] > 0 and acts[i] == 0)
            local = {0: drain_action(env.get_local_action_masks(0))}
            obs, _, term, trunc, info = env.step({"global": acts, "local": local})
            csv_rw = info.get("per_slot_reward_csv") or ""
            if csv_rw:
                vals = [float(x) for x in csv_rw.split(",")]
                for i, j in slotmap.items():
                    if i < len(vals):
                        slot_events.append({"t": t, "job": j,
                                            "act": "defer" if acts[i] == 1 else "route",
                                            "r": vals[i]})
            done = term or trunc
            ges = info.get("global_energy_stats") or ges
            if int(ges.get("deadline_forced_count", 0) or 0) > 0 and \
                    not any(e.get("forced_mark") for e in slot_events):
                slot_events.append({"t": t, "forced_mark": True})
            cur = float(ges.get("total_carbon_emission_kg", prev_kg))
            step_kgs.append(cur - prev_kg)
            prev_kg = cur
            t += 1
        # 自检:差分和 == 终值(同一本账,应逐位)
        tot = float(ges.get("total_carbon_emission_kg", 0.0))
        if abs(sum(step_kgs) - tot) > 1e-9 + 1e-6 * abs(tot):
            sys.exit(f"差分自检失败 off={off}: {sum(step_kgs)} != {tot}")
        return np.asarray(step_kgs), ges, t, slot_events
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
    audit_results = {}
    for arm in arms + ["always_defer"]:
        per_arm_kg[arm], per_arm_steps[arm] = [], 0
    for off in offsets:
        w_ep = w[off:off + 300]
        plans = {arm: pol.releases(arm, w_ep, arrivals) for arm in arms}
        plans["always_defer"] = [1e12] * len(arrivals)
        for arm, rel in plans.items():
            kgs, ges, t, sev = episode_step_kg(cfg, off, rel, arrivals,
                                               turbines, ref_series=w)
            fin, ot = ges.get("total_finished_cloudlets"), ges.get("ontime_mi_share")
            all_kg.append(kgs)
            per_arm_kg[arm].extend(kgs.tolist())
            per_arm_steps[arm] += t
            aud_ok, aud = audit_defer_charges(
                sev, ledger_sum=float(ges.get("defer_urgency_cost_sum", 0.0) or 0.0),
                n_jobs=len(arrivals))
            audit_results[f"{arm}@{off}"] = {"ok": aud_ok, **aud}
            recs.append({"offset": off, "arm": arm, "steps": t,
                         "carbon_kg": float(kgs.sum()), "finished": fin,
                         "ontime": ot, "defer_audit_ok": aud_ok,
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
    aud_all = all(v["ok"] for v in audit_results.values())
    n_fail = sum(1 for v in audit_results.values() if not v["ok"])
    print(f"defer 计费哨兵(首 defer −base/不重复收/守恒/立即 route 零): "
          f"{'PASS 全格' if aud_all else f'**FAIL** {n_fail} 格'}")
    if not aud_all:
        for k, v in audit_results.items():
            if not v["ok"]:
                print(f"  {k}: {v['fails'][:3]}")
    for arm in tt["_order_phys"]:
        r = tt[arm]
        print(f"  {arm:>13} phys={r['phys_kg']:.5f} ΣĈ={r['sum_chat']:.2f} "
              f"cap_hits={r['cap_hits']} max_ratio={r['max_ratio']:.2f}")
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"experiment": a.experiment, "turbines": list(turbines),
         "year": a.year, "offsets": offsets, "distribution": dist,
         "current_fixed_max": 2e-05, "proposed_fixed_max": prop,
         "truth_table": tt, "defer_audit": audit_results,
         "results": recs}, indent=1))
    print("REWARD CALIB DONE", flush=True)


if __name__ == "__main__":
    main()
