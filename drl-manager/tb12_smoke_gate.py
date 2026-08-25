#!/usr/bin/env python3
"""TB12 50k smoke 四门机械判定(Codex 硬阻塞#4)。

对 fc/nofc 各取 ck0(训练前固化)与 ck50,在 6 个校准 offset(T100+101/2021)
上 argmax 滚动,逐集采集:physical kg、env 全局 reward 累计、ontime_mi_share、
cap count/max ratio、deadline_forced_count、有效槽 defer 比例、主动释放数。
四门判据由 Codex 冻结(PREREG_RL_REPAIR_DRAFT.md):

  G1 奖励—物理门: pooled reward(ck50)>ck0 ⇒ pooled kg(ck50)<ck0,否则 STOP;
     所有 episode cap count == 0。
  G2 SLA 门: fc、nofc 的 ck50 池化 ontime_mi_share ≥ 0.995。
  G3 坍缩门(ck50 逐集): deadline_forced_count < 5;至少 1 作业在 backstop
     前主动 route;有效槽 defer fraction < 0.95。
  G4 信息活性门: fc vs nofc 产生可测行为差异(逐 offset kg 差不全零),
     50k 不宣布效果胜利。

启动时强制核验冻结修复 jar(calib/tb12_repair_jar_manifest.json,硬阻塞#6)。
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import load_scaled, ROW_S  # noqa: E402

CALIB_OFFSETS = [4000, 12000, 20000, 28000, 36000, 44000]
SLA_TARGET = 0.995
DEFER_FRAC_MAX = 0.95
N_JOBS = 5


# ---------------------------------------------------------------- 纯判定函数
def gate1_reward_physics(ck0_pool, ck50_pool, all_cap_counts):
    """(reward↑ ⇒ kg↓) 且 cap 全零。返回 (ok, detail)。"""
    cap_ok = all(int(c) == 0 for c in all_cap_counts)
    reward_up = ck50_pool["reward"] > ck0_pool["reward"]
    kg_down = ck50_pool["kg"] < ck0_pool["kg"]
    direction_ok = (not reward_up) or kg_down
    return (direction_ok and cap_ok), {
        "reward_ck0": ck0_pool["reward"], "reward_ck50": ck50_pool["reward"],
        "kg_ck0": ck0_pool["kg"], "kg_ck50": ck50_pool["kg"],
        "reward_up": reward_up, "kg_down": kg_down, "cap_all_zero": cap_ok}


def gate2_sla(fc_ontime_pool, nofc_ontime_pool, target=SLA_TARGET):
    ok = fc_ontime_pool >= target and nofc_ontime_pool >= target
    return ok, {"fc_pooled": fc_ontime_pool, "nofc_pooled": nofc_ontime_pool,
                "target": target}


def gate3_collapse(episodes):
    """ck50 逐集:forced<5、≥1 主动 route、defer 比例<0.95。"""
    bad = []
    for e in episodes:
        checks = {
            "forced_lt5": e["deadline_forced"] < N_JOBS,
            "active_route": e["active_releases"] >= 1,
            "defer_frac": e["defer_frac"] < DEFER_FRAC_MAX,
        }
        if not all(checks.values()):
            bad.append({"offset": e["offset"], "arm": e["arm"], **checks})
    return len(bad) == 0, {"violations": bad}


def gate4_information(fc_by_off, nofc_by_off, tol=1e-9):
    """行为差异:逐 offset kg 差不全零即可(不宣布胜利)。"""
    diffs = [fc_by_off[o] - nofc_by_off[o] for o in sorted(fc_by_off)]
    ok = any(abs(d) > tol for d in diffs)
    return ok, {"per_offset_kg_diff": diffs}


# ---------------------------------------------------------------- 冻结核验
def verify_repair_jar():
    """硬阻塞#6:GATEWAY_LIBS 必须指向冻结修复 installDist,SHA/config 一致。"""
    man_p = pathlib.Path(__file__).resolve().parent / "calib/tb12_repair_jar_manifest.json"
    if not man_p.exists():
        sys.exit("tb12_repair_jar_manifest.json 不存在 —— 先冻结修复 jar")
    man = json.loads(man_p.read_text())
    libs = os.environ.get("GATEWAY_LIBS", "").strip()
    if not libs:
        sys.exit("必须设置 GATEWAY_LIBS 指向冻结修复 installDist(硬阻塞#6)")
    jar = pathlib.Path(libs) / "cloudsimplus-gateway.jar"
    if not jar.is_file():
        sys.exit(f"GATEWAY_LIBS 下无 cloudsimplus-gateway.jar: {libs}")
    sha = hashlib.sha256(jar.read_bytes()).hexdigest()
    if sha != man["jar_sha256"]:
        sys.exit(f"修复 jar SHA 不一致!\n  实际 {sha}\n  冻结 {man['jar_sha256']}")
    cfg_p = pathlib.Path(__file__).resolve().parents[1] / "config_C.yml"
    cfg_sha = hashlib.sha256(cfg_p.read_bytes()).hexdigest()
    if cfg_sha != man["config_sha256"]:
        sys.exit(f"config_C.yml 已变动!\n  实际 {cfg_sha}\n  冻结 {man['config_sha256']}")
    print(f"[SMOKE] 修复 jar 哨兵通过: {sha[:16]}… "
          f"(commit={man['source_commit'][:12]})", flush=True)
    return man


# ---------------------------------------------------------------- 采集
def run_ck(experiment, ckpt, offsets, ref_series):
    from src.baselines.evaluate import load_config
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    from tb12_rl_eval import FullActionHead
    import copy
    cfg0 = load_config(experiment)
    cfg0.pop("py4j_port", None)
    cfg0.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg0.setdefault("output_dir", "/tmp/tb12_gw")
    head = FullActionHead(pathlib.Path(ckpt).resolve())
    out = []
    for off in offsets:
        cfg = copy.deepcopy(cfg0)
        cfg["green_episode_offset_range"] = 0
        for dc in cfg["datacenters"]:
            dc["time_zone_offset_rows"] = int(off)
        env = HierarchicalMultiDCEnv(cfg)
        try:
            obs, _ = env.reset(seed=1)
            g0 = float(np.asarray(obs["global"]["dc_current_green_power_w"]).reshape(-1)[0])
            if abs(g0 - float(ref_series[int(off)])) > 1e-3:
                sys.exit(f"接线哨兵失败 off={off}: {g0} != {ref_series[int(off)]}")
            head.reset()
            done, t, ges, rew_sum = False, 0, {}, 0.0
            n_defer, n_valid = 0, 0
            batch = env.global_routing_batch_size
            defer_idx = env.num_datacenters
            while not done and t < 300:
                g = obs["global"]
                mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
                acts = head.step_full(g).tolist()
                valid = mi > 0
                n_valid += int(valid.sum())
                n_defer += int(sum(1 for i in range(batch)
                                   if valid[i] and acts[i] == defer_idx))
                obs, rew, term, trunc, info = env.step(
                    {"global": acts,
                     "local": {0: drain_action(env.get_local_action_masks(0))}})
                rew_sum += float(rew["global"])
                done = term or trunc
                ges = info.get("global_energy_stats") or ges
                t += 1
            forced = int(ges.get("deadline_forced_count", 0) or 0)
            fin = int(ges.get("total_finished_cloudlets", 0) or 0)
            rec = {"offset": off, "arm": experiment, "steps": t,
                   "carbon_kg": float(ges.get("total_carbon_emission_kg", 0.0)),
                   "reward_sum": rew_sum,
                   "ontime": float(ges.get("ontime_mi_share", 0.0)),
                   "cap_count": int(ges.get("global_carbon_cap_count", 0) or 0),
                   "max_ratio": float(ges.get("global_carbon_max_ratio", 0.0) or 0.0),
                   "deadline_forced": forced,
                   "finished": fin,
                   "active_releases": max(0, fin - forced),
                   "defer_frac": (n_defer / n_valid) if n_valid else 0.0}
            out.append(rec)
            print(f"[SMOKE {experiment.split('rl_')[-1]:>11} off={off:>6}] "
                  f"kg={rec['carbon_kg']:.5f} R={rew_sum:.1f} ontime={rec['ontime']:.3f} "
                  f"cap={rec['cap_count']} forced={forced} "
                  f"defer={rec['defer_frac']:.3f}", flush=True)
        finally:
            env.close()
    return out


def pool(records):
    return {"kg": float(sum(r["carbon_kg"] for r in records)),
            "reward": float(sum(r["reward_sum"] for r in records)),
            "ontime": float(np.mean([r["ontime"] for r in records]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ck0", required=True)
    ap.add_argument("--fc-ck50", required=True)
    ap.add_argument("--nofc-ck0", required=True)
    ap.add_argument("--nofc-ck50", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    man = verify_repair_jar()
    ref = load_scaled((100, 101), 2021)
    runs = {}
    for arm, exp in (("fc", "experiment_tb12_rl_fc_v2s50k"),
                     ("nofc", "experiment_tb12_rl_nofc_v2s50k")):
        for ck in ("ck0", "ck50"):
            runs[f"{arm}_{ck}"] = run_ck(exp, getattr(a, f"{arm}_{ck}"),
                                         CALIB_OFFSETS, ref)
    pools = {k: pool(v) for k, v in runs.items()}
    all_caps = [r["cap_count"] for v in runs.values() for r in v]
    g1 = {}
    for arm in ("fc", "nofc"):
        ok, det = gate1_reward_physics(pools[f"{arm}_ck0"], pools[f"{arm}_ck50"],
                                       [r["cap_count"] for k in (f"{arm}_ck0", f"{arm}_ck50")
                                        for r in runs[k]])
        g1[arm] = {"ok": ok, **det}
    g1_ok = all(v["ok"] for v in g1.values()) and all(c == 0 for c in all_caps)
    g2_ok, g2 = gate2_sla(pools["fc_ck50"]["ontime"], pools["nofc_ck50"]["ontime"])
    g3_ok, g3 = gate3_collapse(runs["fc_ck50"] + runs["nofc_ck50"])
    g4_ok, g4 = gate4_information(
        {r["offset"]: r["carbon_kg"] for r in runs["fc_ck50"]},
        {r["offset"]: r["carbon_kg"] for r in runs["nofc_ck50"]})
    verdict = {"G1_reward_physics": {"ok": g1_ok, "per_arm": g1},
               "G2_sla": {"ok": g2_ok, **g2},
               "G3_collapse": {"ok": g3_ok, **g3},
               "G4_information": {"ok": g4_ok, **g4},
               "ALL_PASS": g1_ok and g2_ok and g3_ok and g4_ok}
    for k in ("G1_reward_physics", "G2_sla", "G3_collapse", "G4_information"):
        print(f"[GATE] {k}: {'PASS' if verdict[k]['ok'] else '**FAIL**'}", flush=True)
    print(f"[GATE] ALL: {'PASS -> 允许 300k' if verdict['ALL_PASS'] else 'STOP'}",
          flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"manifest": man, "checkpoints": {k: getattr(a, k) for k in
                                          ("fc_ck0", "fc_ck50", "nofc_ck0", "nofc_ck50")},
         "pools": pools, "verdict": verdict, "runs": runs}, indent=1))
    print("SMOKE GATE DONE", flush=True)


if __name__ == "__main__":
    main()
