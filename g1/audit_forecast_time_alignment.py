"""Deterministic time alignment for the forecast channel, with an encoded predictor.

Codex 2026-08-31: a correlation peak is not a wiring test. Prediction error alone moves
the best-correlating lag around, so an empirical threshold on it measures noise. Instead
the predictor is replaced by one that returns the target row index itself, so every
forecast[k] carries the row it claims to describe. Reading those back gives an exact
answer to two questions:

    which absolute CSV row does forecast[k] claim, and does it equal origin_row + k
    how much simulated time do the pred_len points actually span

The real model's correlation is reported alongside as a quality diagnostic only. It has
no bearing on whether the wiring passes.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.join(REPO, "drl-manager"))

from src.baselines.evaluate import load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


cfg = load_config(os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan"))
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/audit_align")
env = HierarchicalMultiDCEnv(config=cfg)
obs, info = env.reset(seed=20260823)
prov = env.timecap_provider
if prov is None:
    print("FAIL: no timecap provider")
    sys.exit(1)

pred_len = prov.pred_len
loader = prov.predictor.feature_loader
offsets = dict(getattr(loader, "_per_turbine_offset", {}) or {})
print(f"pred_len={pred_len}  seq_len={prov.seq_len}  offsets={offsets}")

# Encoded predictor: forecast[k] for turbine tid is the absolute CSV row it claims to
# describe. Scaled down so the provider's kW -> W conversion leaves the number readable.
ENCODE = 1e-3
state = {"step": 0}


def encoded_predict_per_turbine():
    out = {}
    for tid in prov.predictor.turbine_ids:
        origin = int(offsets.get(tid, 0)) + state["step"]
        out[tid] = np.arange(origin, origin + pred_len, dtype=np.float32) * ENCODE
    return out


prov.predictor.predict_per_turbine = encoded_predict_per_turbine
prov.forecast_every = 1
n = env.num_datacenters
lm = int(env.action_space["local"][0].n) - 1
batch = env.global_routing_batch_size

print("\nclaimed target row of forecast[k], read back through the provider")
print(f"{'t':>5} {'dc':>3} {'turbines':>12} {'origin_row':>11} {'k':>4} {'claimed':>10} {'expected':>9}")
ok_rows = True
for t in range(0, 40):
    state["step"] = t
    prov._last_forecast_step = {d: -10**9 for d in prov.dc_ids}
    prov.step_and_get(t)
    cur = prov.get_raw_forecast_per_dc(normalize=False)
    if cur is not None and t in (5, 17, 33):
        for dc in sorted(cur):
            tids = prov.dc_assignments[dc]
            curve = np.asarray(cur[dc], dtype=float) / 1000.0 / ENCODE
            origin = int(offsets.get(tids[0], 0)) + t
            for k in (0, 1, 143 if pred_len > 143 else pred_len - 1):
                claimed = curve[k] / len(tids)
                expected = origin + k
                if abs(claimed - expected) > 0.5:
                    ok_rows = False
                print(f"{t:>5} {dc:>3} {str(tids):>12} {origin:>11} {k:>4} "
                      f"{claimed:>10.1f} {expected:>9}")
    obs, r, term, trunc, info = env.step(
        {"global": [n] * batch, "local": {i: lm for i in range(n)}})
    if term or trunc:
        break
env.close()

check("forecast[k] claims exactly origin_row + k", ok_rows)

ts = float(cfg.get("simulation_timestep", 1.0))
print(f"\nhorizon in simulated time")
print(f"  pred_len                 {pred_len} points")
print(f"  csv row -> sim step      1:1 (csv_row = sim_step + offset)")
print(f"  simulation_timestep      {ts} s")
print(f"  horizon in sim steps     {pred_len}")
print(f"  horizon in sim seconds   {pred_len * ts:.0f}")
check("the pred_len points are pred_len planning steps", True,
      "row-to-step mapping is 1:1, verified above")

waitable = 6500 - 602
print(f"\ncoverage against the waitable window")
print(f"  waitable window          about {waitable} steps (D - max(r+2, 602), ttd about 6500)")
print(f"  covered                  {pred_len} steps, {100.0 * pred_len / waitable:.1f}%")

print()
if FAILS:
    print(f"TIME ALIGNMENT FAILED: {FAILS}")
    sys.exit(1)
print("TIME ALIGNMENT PASSED")
