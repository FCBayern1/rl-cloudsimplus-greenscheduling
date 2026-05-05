"""
M0 end-to-end smoke probe: reset + step a real env and dump info["crd"] so we
can confirm the data plumbing (Java helpers + Py4J bridge + Python wiring) is
intact. No RLlib, no PPO — just the env wiring.

Run from the repo's drl-manager directory:
    cd drl-manager
    .venv/bin/python -m tests.probe_crd_info experiment_multi_5dc_carbon_v2
"""
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.training.train_rlmodule_gtrxl import load_config
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv


def _silence_py4j_after_close():
    """
    After env.close() kills the JVM, Python's GC may finalize Py4J connection
    pool objects, which try to reconnect to the dead gateway and spam
    ConnectionResetError / ConnectionRefusedError tracebacks. Silence those.
    """
    for name in ("py4j", "py4j.java_gateway", "root"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False


def _summarize(label, value):
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        return f"{label}: list[{len(value)}] min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g}"
    return f"{label}: {value!r}"


def main():
    if len(sys.argv) < 2:
        print("usage: probe_crd_info.py <experiment_name>")
        sys.exit(2)
    experiment = sys.argv[1]

    cfg_path = REPO_ROOT.parent / "config.yml"
    all_cfg = load_config(str(cfg_path))
    if experiment not in all_cfg:
        print(f"experiment {experiment!r} not in config.yml")
        sys.exit(1)
    env_cfg = dict(all_cfg[experiment])  # shallow copy so we can override

    # Force the env to auto-launch its own JVM on a free port, instead of
    # assuming a pre-launched gateway on the configured 25333. (py4j_port=0
    # triggers _find_free_port + _launch_java_gateway in env init.)
    env_cfg["py4j_port"] = 0

    # _launch_java_gateway() requires a log directory. Entrypoint scripts
    # normally set this; for the probe we drop logs in tests/probe_logs/.
    log_dir = REPO_ROOT / "tests" / "probe_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg["gateway_log_dir"] = str(log_dir)

    print("[probe] py4j_port set to 0 → env will auto-launch a fresh gateway")
    print(f"[probe] gateway logs → {log_dir}")
    print(f"[probe] launching env with experiment={experiment}")
    env = HierarchicalMultiDCEnv(env_cfg)
    print(f"[probe] env created, num_datacenters={env.num_datacenters}")

    crd = None
    try:
        obs, info = env.reset()
        print("\n[probe] === info from reset ===")
        crd = info.get("crd", "<missing>")
        if isinstance(crd, dict):
            print(f"  crd keys: {sorted(crd.keys())}")
            for k, v in crd.items():
                print("  " + _summarize(k, v))
        else:
            print(f"  WARNING: info['crd'] = {crd!r}")

        # Build a no-op-ish action: route every cloudlet to DC 0, schedule VM 0.
        batch_size = env.global_routing_batch_size
        action_dict = {
            "global": [0] * batch_size,
            "local": {dc_id: 0 for dc_id in range(env.num_datacenters)},
        }

        n_steps = 3
        for i in range(n_steps):
            try:
                obs, reward, terminated, truncated, info = env.step(action_dict)
            except Exception as e:
                print(f"[probe] step {i} failed: {e}")
                break
            crd = info.get("crd", "<missing>")
            print(f"\n[probe] === step {i+1} reward={reward!r} ===")
            if isinstance(crd, dict):
                print(f"  crd keys: {sorted(crd.keys())}")
                for k, v in crd.items():
                    print("  " + _summarize(k, v))
            else:
                print(f"  WARNING: info['crd'] = {crd!r}")
            if terminated or truncated:
                break

        # Exercise the analytical CF endpoints via the gateway.
        if env.java_env is not None and isinstance(crd, dict):
            actual = crd.get("actual_wind_w")
            if actual is not None:
                actual_arr = list(actual)
                zero_arr = [0.0] * len(actual_arr)
                try:
                    carbon_actual = float(env.java_env.computeCounterfactualCarbonKg(actual_arr))
                    carbon_zero = float(env.java_env.computeCounterfactualCarbonKg(zero_arr))
                    waste_actual = float(env.java_env.computeCounterfactualWasteRatio(actual_arr))
                    waste_zero = float(env.java_env.computeCounterfactualWasteRatio(zero_arr))
                    print("\n[probe] === gateway CF API ===")
                    print(f"  carbon kg under actual wind = {carbon_actual:.6f}")
                    print(f"  carbon kg under zero wind    = {carbon_zero:.6f}")
                    print(f"  waste ratio under actual    = {waste_actual:.4f}")
                    print(f"  waste ratio under zero wind  = {waste_zero:.4f}")
                    if carbon_zero < carbon_actual:
                        print("  WARNING: zero-wind carbon should be >= actual-wind carbon")
                    else:
                        print("  ✓ zero-wind carbon >= actual-wind carbon (sanity OK)")
                except Exception as e:
                    print(f"[probe] gateway CF call failed: {e}")
    finally:
        # Silence Py4J before close so the JVM-shutdown reconnect-spam from
        # finalizers doesn't drown the probe's verdict.
        _silence_py4j_after_close()
        try:
            env.close()
        except Exception:
            pass

    print("\n[probe] done.")
    # Hard-exit to bypass Python's atexit / GC: any lingering Py4J connection
    # pool objects would try to reconnect to the now-dead JVM and spam errors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
