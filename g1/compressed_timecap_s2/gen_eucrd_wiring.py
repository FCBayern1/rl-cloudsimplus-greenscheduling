"""G5 wiring-gate training blocks (reports/EUCRD_SIGNAL_GATE_PREREG.md §2).

Two 40 000-step trainings on the RL_V2 training twin, identical except the training forecast
tier, both with EU-CRD enabled and the repaired responsibility source:

    g5_err    perturb_tier shrink75   (an imperfect forecast during training)
    g5_clean  perturb_tier godeye     (control: a correct forecast must produce no credit)

Usage: python gen_eucrd_wiring.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SRC = os.path.join(HERE, "config_rl_v2_cpu.yml")          # the CPU-learner twin the lines trained on
OUT = os.path.join(HERE, "config_eucrd_wiring.yml")
MANIFEST = os.path.join(HERE, "stage_a_out", "eucrd_wiring", "manifest.json")
STEPS = 40000                                              # 5 PPO iterations: passes the 450-call warmup
RUNS = {"err": "shrink75", "clean": "godeye"}
BASE = "rl2_E_s2_r48_w72_c3_n35"                          # the EU-CRD line's block
# the responsibility magnitude G5 wires: local decision regret (EUCRD_SIGNAL_GATE_PREREG
# Addendum A1), not the cover error the first draft used
FORECAST_SOURCE = "candidate_carbon_regret"


def build():
    cfg = yaml.safe_load(open(SRC))
    base = cfg[BASE]
    out = {"common": cfg["common"]}
    for name, tier in RUNS.items():
        b = copy.deepcopy(base)
        b["experiment_name"] = f"g5_{name}"; b["simulation_name"] = f"G5_{name}"
        b["perturb_tier"] = tier
        crd = copy.deepcopy(b.get("crd", {})); fc = dict(crd.get("forecast", {}))
        fc["source"] = FORECAST_SOURCE; crd["forecast"] = fc; crd["enabled"] = True
        b["crd"] = crd
        b["training"] = dict(b.get("training", {}), total_timesteps=STEPS,
                             checkpoint_freq_timesteps=STEPS, checkpoint_num_to_keep=0,
                             save_init_checkpoint=False)
        out[f"g5_{name}"] = b
        # the clean run's tier equals the base's (godeye), so perturb_tier only shows in the err run
        diff = sorted(k for k in set(b) | set(base) if b.get(k) != base.get(k))
        allowed = {"crd", "experiment_name", "perturb_tier", "simulation_name", "training"}
        assert set(diff) <= allowed, diff
    a, c = out["g5_err"], out["g5_clean"]
    between = sorted(k for k in set(a) | set(c) if a.get(k) != c.get(k))
    assert between == ["experiment_name", "perturb_tier", "simulation_name"], between
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    with open(OUT, "w") as f:
        f.write(text)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"config": os.path.basename(OUT), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
               "source": os.path.basename(SRC), "base_block": BASE, "steps": STEPS, "runs": RUNS,
               "forecast_source": FORECAST_SOURCE,
               "between_run_diff": between}, open(MANIFEST, "w"), indent=1)
    print(json.dumps({"config": OUT, "between_run_diff": between, "steps": STEPS}, indent=1))
    return out


if __name__ == "__main__":
    build()
