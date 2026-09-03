"""Boot one simulation and read back idle_host_power_down_effective from the JVM."""
import os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "drl-manager"))
from src.baselines.evaluate import load_config  # noqa
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa
cfg = load_config(os.environ["ORACLE_EXPERIMENT"])
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/idle_flag_probe")
print("python-side config top-level idle_host_power_down =", cfg.get("idle_host_power_down"),
      "| DC0 =", (cfg.get("datacenters") or [{}])[0].get("idle_host_power_down"))
env = HierarchicalMultiDCEnv(config=cfg)
obs, info = env.reset(seed=42)
print("reset info key =", info.get("idle_host_power_down_effective"))
n = env.num_datacenters
ids = info.get("planner", {}).get("batch_cloudlet_ids", [0] * 128)
obs, r, term, trunc, info = env.step({"global": [n] * len(ids), "local": {i: 0 for i in range(n)}})
print("JVM idle_host_power_down_effective (after step) =", info.get("idle_host_power_down_effective"))
print("cpu_util_effective (control) =", info.get("cloudlet_cpu_utilization_effective"))
env.close()
