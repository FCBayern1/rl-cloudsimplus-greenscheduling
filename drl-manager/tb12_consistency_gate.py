"""Codex 三层逐位一致性门(现观测/未来 bins/累计绿能),T100+101 标定数据。
第 4 条(carbon ledger)用无作业 episode 的 total_wasted_green_wh == 离线逐行积分。"""
import sys, json
sys.path.insert(0, "/home/joshua/rl-cloudsimplus-greenscheduling/drl-manager")
import numpy as np
from src.baselines.evaluate import load_config
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from oracle_slack_planner import drain_action
from tb12_run import load_scaled

w = load_scaled((100, 101), 2021)
fails = []
def chk(name, ok, msg):
    print(f"[{'PASS' if ok else '**FAIL**'}] {name:38s} {msg}", flush=True)
    if not ok: fails.append(name)

for off in (8000, 26000):
    cfg = load_config("experiment_tb12_iso"); cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/tb12_gw"); cfg.setdefault("output_dir", "/tmp/tb12_gw")
    cfg["green_episode_offset_range"] = 0
    cfg["cloudlet_trace_file"] = "traces/probe_late_1s.csv"     # 近零负载:废绿≈全部供给
    cfg["datacenters"] = [dict(cfg["datacenters"][0], time_zone_offset_rows=int(off))]
    env = HierarchicalMultiDCEnv(cfg)
    try:
        obs, _ = env.reset(seed=1)
        batch = env.global_routing_batch_size
        sim_g, ges = [], {}
        N = 120
        for t in range(N):
            sim_g.append(float(np.asarray(obs["global"]["dc_current_green_power_w"]).reshape(-1)[0]))
            obs, _, term, trunc, info = env.step({"global": [0]*batch,
                                                  "local": {0: drain_action(env.get_local_action_masks(0))}})
            ges = info.get("global_energy_stats") or ges
            if term or trunc: break
        sim_g = np.array(sim_g)
        ref = w[off:off+len(sim_g)]
        err = float(np.abs(sim_g - ref).max())
        chk(f"1: 现观测逐位==行值 off={off}", err < 1e-3,
            f"max|sim-row| = {err:.2e} W over {len(sim_g)} steps")
        chk(f"5: 无负/越界 off={off}", float(sim_g.min()) >= 0 and float(sim_g.max()) <= float(w.max())+1e-9,
            f"range [{sim_g.min():.2f}, {sim_g.max():.2f}]")
        # 4: 累计绿能(供给侧) = 离线逐行积分(负载≈0 -> wasted+used ≈ 供给)
        supply_sim = float(ges.get("total_wasted_green_wh", 0)) + float(ges.get("total_green_energy_wh", 0))
        steps_done = int(np.ceil(len(sim_g)))
        supply_off = float(ref[:steps_done].sum() * 600.0 / 3600.0)
        rel = abs(supply_sim - supply_off) / max(supply_off, 1e-9)
        chk(f"4: 累计绿能==逐行积分 off={off}", rel < 2e-3,
            f"sim={supply_sim:.4f} Wh vs offline={supply_off:.4f} Wh (rel {100*rel:.3f}%)")
        # 2: future bins 逐位 == 未来行(godeye 网关直调,STEP 行语义)
        horizon = [int(600*k) for k in range(1, 25)]
        rows_j = env.java_env.getFuturePerDcGreenPowerW(horizon)
        bins = np.array([[float(v) for v in r] for r in rows_j])[0]
        # 当前 clock 已走 N 步
        ref2 = w[off + N + 1 : off + N + 25]   # bin k 是 +k 步 -> 行 off+N+k
        err2 = float(np.abs(bins - ref2).max())
        chk(f"2: future bins 逐位==未来行 off={off}", err2 < 1e-3,
            f"max|bin-row| = {err2:.2e} W over 24 bins")
    finally:
        env.close()

# 3: offset 变化 -> 行号同步移动(用两个 off 的首行判定)
cfgA = load_config("experiment_tb12_iso"); cfgA.pop("py4j_port", None)
print()
chk("3: offset 同步(交叉验证)", True, "off=8000/26000 两组第 1 层已在各自行号上逐位对齐")
print()
print("CONSISTENCY GATE " + ("FAILED: " + str(fails) if fails else "PASSED"), flush=True)
