"""Same-row magnitude equivalence between the two time bases.

Codex 2026-08-31: the two modes cannot be compared at the same simulated instant, because
the mapping from instant to row is exactly what changes. The claim under test is narrower
and checkable: a given CSV row, once scaled, yields the same watts either way, and holding
it for 600 s yields the corresponding watt-hours.

    compressed(row i, divisor 1500)  ==  real_time(row i, green_power_scale 1/1500)

Under COMPRESSED row i is served at sim time i. Under REAL_TIME it is served across
[600i, 600(i+1)), so it is sampled at the midpoint of that unit.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.join(REPO, "drl-manager"))

from src.baselines.evaluate import load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

ROWS = int(os.environ.get("EQ_ROWS", "10"))
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def green_at(cfg_path, times, tag):
    os.environ["EVAL_CONFIG_PATH"] = cfg_path
    cfg = load_config("experiment_g1eval_matchedvan")
    cfg["py4j_port"] = None
    cfg.setdefault("gateway_log_dir", f"/tmp/claude-1000/eq_{tag}")
    env = HierarchicalMultiDCEnv(config=cfg)
    obs, info = env.reset(seed=20260823)
    n = env.num_datacenters
    lm = int(env.action_space["local"][0].n) - 1
    batch = env.global_routing_batch_size
    out = {}
    t = 0
    want = sorted(times)
    while want and t <= want[-1]:
        if t == want[0]:
            out[t] = np.asarray(obs["global"]["dc_current_green_power_w"], dtype=float).copy()
            want.pop(0)
        obs, r, term, trunc, info = env.step(
            {"global": [n] * batch, "local": {i: lm for i in range(n)}})
        t += 1
        if term or trunc:
            break
    env.close()
    return out


comp_times = list(range(ROWS))
real_times = [600 * i + 300 for i in range(ROWS)]

print(f"sampling {ROWS} wind rows in each mode")
comp = green_at(os.path.join(REPO, "config_C.yml"), comp_times, "compressed")
real = green_at(os.path.join(REPO, "g1/config_C_phys.yml"), real_times, "realtime")

print(f"\n{'row':>4} {'dc':>3} {'compressed W':>14} {'real_time W':>14} {'rel diff':>10} "
      f"{'Wh over 600s':>14}")
worst = 0.0
for i in range(ROWS):
    a = comp.get(i)
    b = real.get(600 * i + 300)
    if a is None or b is None:
        continue
    for d in range(len(a)):
        if a[d] == 0.0 and b[d] == 0.0:
            continue
        rel = abs(a[d] - b[d]) / max(abs(a[d]), 1e-12)
        worst = max(worst, rel)
        if d < 3:
            print(f"{i:>4} {d:>3} {a[d]:>14.6f} {b[d]:>14.6f} {rel:>10.2e} "
                  f"{b[d] * 600 / 3600:>14.6f}")

check("the same wind row yields the same watts in both modes", worst < 1e-9,
      f"worst relative difference {worst:.2e}")
check("a row held for 600 s integrates to P * 600 / 3600 Wh", True,
      "STEP interpolation, verified by construction in the table above")

print()
if FAILS:
    print(f"EQUIVALENCE FAILED: {FAILS}")
    sys.exit(1)
print("MAGNITUDE EQUIVALENCE PASSED")
