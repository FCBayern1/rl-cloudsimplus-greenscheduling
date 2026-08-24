#!/usr/bin/env python3
"""TB12 RL 配对 eval:restore checkpoint,60 分层偏移 argmax,与封套同表。

与判决同一批偏移、同一环境(接线哨兵在环)。RL 的动作是【全联合 argmax】:
per-slot 在全部 num_dc+1 个选项(含 hold)上取 argmax —— 不像 SQT2 的
route-only(那里 defer 交给规则门,这里释放时机正是策略的学习目标)。
每臂用【自己的 experiment block】建环境(forecast_mode 是唯一差异,
obs 形状相同、内容不同 —— 与训练时一致)。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from h1_matched_headroom import ModuleHead
from tb12_run import load_scaled, ROW_S


class FullActionHead(ModuleHead):
    """全联合 argmax(含 hold 选项),沿用父类的 recurrent state 机制。"""

    def step_full(self, obs):
        from ray.rllib.core.columns import Columns
        batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                               for k, v in obs.items()
                               if np.asarray(v).dtype.kind in "ifub"}}
        state = self.state
        if state is None:
            init = self.module.get_initial_state()
            if init:
                state = {k: torch.as_tensor(np.asarray(v))[None, ...]
                         for k, v in init.items()}
        if state:
            batch[Columns.STATE_IN] = state
        with torch.no_grad():
            out = self.module.forward_inference(batch)
        self.state = out.get(Columns.STATE_OUT) or None
        logits = out[Columns.ACTION_DIST_INPUTS].detach().reshape(-1)
        n_opt = logits.numel() // self.n_slots
        return logits.reshape(self.n_slots, n_opt).argmax(-1).numpy()


def run_arm(experiment, ckpt, offsets, turbines, year, ref_series):
    from src.baselines.evaluate import load_config
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    import copy
    cfg0 = load_config(experiment)
    cfg0.pop("py4j_port", None)
    cfg0.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg0.setdefault("output_dir", "/tmp/tb12_gw")
    head = FullActionHead(pathlib.Path(ckpt))
    out = []
    for off in offsets:
        cfg = copy.deepcopy(cfg0)
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
            head.reset()
            done, t, ges = False, 0, {}
            while not done and t < 300:
                acts = head.step_full(obs["global"]).tolist()
                obs, _, term, trunc, info = env.step(
                    {"global": acts,
                     "local": {0: drain_action(env.get_local_action_masks(0))}})
                done = term or trunc
                ges = info.get("global_energy_stats") or ges
                t += 1
            rec = {"offset": off,
                   "carbon_kg": ges.get("total_carbon_emission_kg"),
                   "green_wh": ges.get("total_green_energy_wh"),
                   "finished": ges.get("total_finished_cloudlets"),
                   "ontime": ges.get("ontime_mi_share"), "steps": t}
            out.append(rec)
            print(f"[RLEVAL {experiment.split('_')[-1]:>5} off={off:>6}] "
                  f"carbon={rec['carbon_kg']:.5f} finished={rec['finished']} "
                  f"ontime={rec['ontime']:.3f}", flush=True)
        finally:
            env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-ckpt", required=True)
    ap.add_argument("--nofc-ckpt", required=True)
    ap.add_argument("--turbines", default="110,111")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    offsets = json.load(open(pathlib.Path(__file__).resolve().parent
                             / "calib/tb12_offsets.json"))["formal_offsets_T110_111_2021"]
    turbines = tuple(int(x) for x in a.turbines.split(","))
    ref = load_scaled(turbines, a.year)
    res = {"rl_fc": run_arm("experiment_tb12_rl_fc", a.fc_ckpt, offsets,
                            turbines, a.year, ref),
           "rl_nofc": run_arm("experiment_tb12_rl_nofc", a.nofc_ckpt, offsets,
                              turbines, a.year, ref)}
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"turbines": list(turbines), "fc_ckpt": a.fc_ckpt,
         "nofc_ckpt": a.nofc_ckpt, "results": res}, indent=1))
    print("RL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
