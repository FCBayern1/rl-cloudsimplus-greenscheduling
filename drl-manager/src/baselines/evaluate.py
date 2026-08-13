"""
Usage:
    # Start the Java Gateway
    cd cloudsimplus-gateway && ./gradlew run

    # Start Evaluation
    cd drl-manager
    python -m src.baselines.evaluate --global random --local random --experiment experiment_multi_dc_10
"""
import argparse
import sys
import csv
import logging
import time
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gym_cloudsimplus.envs import HierarchicalMultiDCEnv, HierarchicalMultiDCEnvSimple
from src.baselines.global_schedulers import GLOBAL_SCHEDULERS
from src.baselines.local_schedulers import LOCAL_SCHEDULERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# py4j logs full tracebacks at ERROR level whenever its connection retry
# loop hits a (briefly) unreachable gateway — typical during teardown when
# Algorithm.from_checkpoint's env_runner env tries to close after our eval
# env has already killed the JVM. The retries are harmless, so silence them.
logging.getLogger("py4j").setLevel(logging.CRITICAL)
logging.getLogger("py4j.java_gateway").setLevel(logging.CRITICAL)
logging.getLogger("py4j.clientserver").setLevel(logging.CRITICAL)

# RLlib's pre-flight env validation logs ERROR when it inspects our env BEFORE
# reset() has populated `env.agents`. The check fails but RLlib continues, and
# our actual rollout uses a separately-constructed eval env, so this validation
# noise is misleading. RLModule Catalog warnings are similarly by-design noise.
#
# Note: setLevel() doesn't survive — Ray re-configures these loggers when
# ray.rllib imports finish. A logging Filter on the root handler is robust:
# it drops records by name regardless of any later setLevel calls.
class _DropByName(logging.Filter):
    def __init__(self, prefixes):
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record):  # True = keep, False = drop
        return not record.name.startswith(self._prefixes)


_NOISY_LOGGER_PREFIXES = (
    "py4j",
    "ray.rllib.env.multi_agent_env_runner",
    "ray.rllib.utils.pre_checks.env",
    "ray.rllib.core.rl_module.rl_module",
)
_drop_filter = _DropByName(_NOISY_LOGGER_PREFIXES)
for _h in logging.getLogger().handlers:
    _h.addFilter(_drop_filter)

# Ray emits a flood of RayDeprecationWarning / DeprecationWarning every run
# (UnifiedLogger, JsonLogger, CSVLogger, TBXLogger, RLModule(config=...)).
# These are internal API drift warnings, not actionable from our code.
import warnings as _warnings
try:
    from ray._private.utils import RayDeprecationWarning as _RayDepWarn
    _warnings.filterwarnings("ignore", category=_RayDepWarn)
except Exception:
    pass
_warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\..*")
_warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray$")

# Suppress the "Exception ignored in: <function LearnerGroup.__del__>" trace
# that fires at interpreter shutdown. Root cause: LearnerGroup.__del__ tries
# to talk to Ray's GCS, which is already torn down at that point, and the
# resulting subprocess.Popen call raises "preexec_fn not supported at
# interpreter shutdown". Results are already on disk by this point — the
# noise is purely cosmetic.
try:
    from ray.rllib.core.learner.learner_group import LearnerGroup as _LG
    _orig_lg_del = _LG.__del__
    def _safe_lg_del(self):
        try:
            _orig_lg_del(self)
        except Exception:
            pass
    _LG.__del__ = _safe_lg_del
except Exception:
    pass


def load_config(experiment_name: str) -> dict:
    """Load configuration for specified experiment from config.yml
    (or EVAL_CONFIG_PATH if set — lets eval use an alternate config like config_C.yml
    without touching the default)."""
    import os as _os
    _override = _os.environ.get("EVAL_CONFIG_PATH")
    config_path = Path(_override) if _override else (Path(__file__).parent.parent.parent.parent / "config.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        all_config = yaml.safe_load(f)

    # Start with common config
    config = all_config.get('common', {}).copy()

    # Override with experiment-specific config
    if experiment_name in all_config:
        config.update(all_config[experiment_name])
    else:
        logger.warning(f"Experiment '{experiment_name}' not found in config.yml, using common config only")

    return config


def _parse_scalar(v: str):
    """Parse a CLI override value into bool/int/float/str."""
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    # int
    try:
        if low.startswith("0") and len(low) > 1 and low[1].isdigit():
            # keep as string to avoid octal ambiguity
            raise ValueError
        return int(s)
    except Exception:
        pass
    # float
    try:
        return float(s)
    except Exception:
        pass
    return s


def _apply_overrides(config: dict, overrides: List[str]) -> dict:
    """
    Apply overrides like:
      --override max_cloudlets_to_create_from_workload_file=100000
      --override green_energy.enabled=true
    """
    if not overrides:
        return config
    cfg = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --override '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        key_path = [p for p in k.strip().split(".") if p]
        if not key_path:
            raise ValueError(f"Invalid --override '{item}'. Empty key.")
        value = _parse_scalar(v)
        # set nested dicts
        cur = cfg
        for p in key_path[:-1]:
            nxt = cur.get(p, None)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[key_path[-1]] = value
    return cfg


def collect_metrics(info: Dict[str, Any], num_dcs: int) -> Dict[str, Any]:
    """
    Collect evaluation metrics from info dict.

    Collects all metrics except rewards (which are excluded for baseline evaluation).
    """
    # Global energy statistics
    gs = info.get('global_energy_stats', {})
    if isinstance(gs, str):
        # If gs is a string representation, parse it
        gs = {}

    # Datacenter-specific metrics
    dc_metrics = info.get('datacenter_energy_metrics', {})
    if isinstance(dc_metrics, str):
        dc_metrics = {}

    # Primary metrics
    metrics = {
        # Energy metrics (global)
        'green_waste_wh': _safe_get(gs, 'total_wasted_green_wh', 0),
        'green_used_wh': _safe_get(gs, 'total_green_energy_wh', 0),
        'brown_used_wh': _safe_get(gs, 'total_brown_energy_wh', 0),
        'total_energy_wh': _safe_get(gs, 'total_energy_wh', 0),
        'green_ratio': _safe_get(gs, 'green_energy_ratio', 0),
        'total_carbon_kg': _safe_get(gs, 'total_carbon_emission_kg', 0),
        'carbon_intensity': _safe_get(gs, 'carbon_intensity_kg_per_kwh', 0),

        # Workload metrics
        'total_cloudlets': _safe_get(info, 'total_cloudlets', 0),
        'remaining_cloudlets': _safe_get(info, 'remaining_cloudlets', 0),
        # Java Multi-DC info uses "cloudlets_routed"; keep backward compat with "routed_cloudlets".
        'routed_cloudlets': _safe_get(info, 'cloudlets_routed', _safe_get(info, 'routed_cloudlets', 0)),
        # Fix A: how many cloudlets the deadline backstop force-routed (vs proactively routed by the
        # policy). High share ⇒ the heuristic backstop is doing the routing, not the learned policy.
        'deadline_forced_count': _safe_get(gs, 'deadline_forced_count', 0),
        'total_finished_cloudlets': _safe_get(gs, 'total_finished_cloudlets', 0),
    }

    # Extra debug counters from Java global energy stats (if present)
    # Note: these can differ from total_cloudlets if the simulator loads a fixed trace list vs dynamically injected list.
    metrics['total_created_cloudlets'] = _safe_get(gs, 'total_created_cloudlets', 0)
    metrics['total_finished_cloudlets'] = _safe_get(gs, 'total_finished_cloudlets', 0)

    # Derived global metrics
    total_green = metrics['green_used_wh'] + metrics['green_waste_wh']
    metrics['waste_ratio'] = metrics['green_waste_wh'] / total_green if total_green > 0 else 0.0

    # Routed cloudlets: cloudlets that have been dispatched to DCs (but may not have finished execution)
    routed = metrics['total_cloudlets'] - metrics['remaining_cloudlets']
    metrics['routed_cloudlets_count'] = routed
    metrics['routed_rate'] = (
        routed / metrics['total_cloudlets'] if metrics['total_cloudlets'] > 0 else 0.0
    )
    # Keep 'completion_rate' as alias for backward compatibility (same as routed_rate)
    metrics['completion_rate'] = metrics['routed_rate']

    # 2026-05-25: MI-weighted completion rate — the SAME metric the RL trainer
    # logs (monitor.csv::completion_rate_mi).  Needed so baseline vs RL c/c
    # comparisons use a consistent completion denominator (cloudlet-count
    # finished_rate is NOT comparable to the RL's MI-weighted completion).
    metrics['completion_rate_mi'] = _safe_get(gs, 'completion_rate_mi', 0.0)
    # Carbon per unit MI-completion — the headline "c/c" metric.
    metrics['carbon_per_completion_mi'] = (
        metrics['total_carbon_kg'] / metrics['completion_rate_mi']
        if metrics['completion_rate_mi'] > 0 else 0.0
    )

    # Per-DC metrics + global mean completion time
    total_finished = 0
    weighted_completion_sum = 0.0
    total_received = 0

    for dc_id in range(num_dcs):
        dc = dc_metrics.get(dc_id, {}) if isinstance(dc_metrics, dict) else {}
        if isinstance(dc, str):
            dc = {}

        mean_ct = _safe_get(dc, 'mean_completion_time', 0.0)
        finished = _safe_get(dc, 'cloudlets_finished', 0)
        received = _safe_get(dc, 'cloudlets_received', 0)
        green_ratio_dc = _safe_get(dc, 'green_energy_ratio', 0.0)

        metrics[f'completion_time_dc_{dc_id}'] = float(mean_ct)
        metrics[f'finished_dc_{dc_id}'] = int(finished)
        metrics[f'received_dc_{dc_id}'] = int(received)
        metrics[f'green_ratio_dc_{dc_id}'] = float(green_ratio_dc)

        total_finished += int(finished)
        total_received += int(received)
        weighted_completion_sum += float(mean_ct) * int(finished)

    # Global mean completion time across all finished cloudlets (weighted by per-DC counts)
    metrics['mean_completion_time'] = (
        weighted_completion_sum / total_finished if total_finished > 0 else 0.0
    )
    metrics['total_received_cloudlets'] = int(total_received)
    metrics['sum_finished_dc'] = int(total_finished)

    # Finished rate: cloudlets that have actually completed execution across all DCs
    metrics['finished_rate'] = (
        total_finished / metrics['total_cloudlets'] if metrics['total_cloudlets'] > 0 else 0.0
    )

    # Carbon per finished cloudlet (kg CO2 per completed task)
    metrics['carbon_per_finished_cloudlet'] = (
        metrics['total_carbon_kg'] / total_finished if total_finished > 0 else 0.0
    )

    return metrics


def _safe_get(d: Any, key: str, default: Any) -> Any:
    """Safely get a value from a dict-like object."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    try:
        return getattr(d, key, default)
    except:
        return default


def _infer_use_new_api_from_checkpoint(checkpoint_path: str) -> bool:
    """
    Heuristic: New API Stack checkpoints contain learner_group/ (RLModule + Learner).
    """
    try:
        cp = Path(checkpoint_path)
        # Allow user to pass either ".../checkpoint_000019" or its parent.
        if cp.is_dir() and (cp / "rllib_checkpoint.json").exists():
            return (cp / "learner_group").exists()
        # If user passes a higher-level directory, try common patterns.
        if cp.is_dir():
            # Find any checkpoint_* child
            for child in sorted(cp.glob("checkpoint_*")):
                if (child / "rllib_checkpoint.json").exists():
                    return (child / "learner_group").exists()
    except Exception:
        pass
    return False


def _summarize_decision_latency(samples_ns: List[int], prefix: str) -> Dict[str, float]:
    """Reduce per-decision latency samples (ns) to mean/p50/p95/p99 in microseconds.

    Returns keys `{prefix}_us_mean`, `_us_p50`, `_us_p95`, `_us_p99`, `_count`.
    Empty input yields zeros (kept rather than NaN so CSV writers don't choke).
    """
    if not samples_ns:
        return {
            f"{prefix}_us_mean": 0.0,
            f"{prefix}_us_p50": 0.0,
            f"{prefix}_us_p95": 0.0,
            f"{prefix}_us_p99": 0.0,
            f"{prefix}_count": 0,
        }
    arr = np.asarray(samples_ns, dtype=np.float64) / 1e3  # ns -> us
    return {
        f"{prefix}_us_mean": float(arr.mean()),
        f"{prefix}_us_p50": float(np.percentile(arr, 50)),
        f"{prefix}_us_p95": float(np.percentile(arr, 95)),
        f"{prefix}_us_p99": float(np.percentile(arr, 99)),
        f"{prefix}_count": int(arr.size),
    }


def _capture_memory_mb() -> Dict[str, float]:
    """Peak process RSS and (if a CUDA policy is loaded) peak GPU allocation, in MB.

    Efficiency-overhead metric (see the eval-report template). ru_maxrss is the
    process-wide peak RSS, monotonic over the run, which is the footprint figure
    the report asks for. GPU peak is read per call and zero on CPU-only runs.
    """
    out = {"peak_cpu_rss_mb": 0.0, "peak_gpu_mem_mb": 0.0}
    try:
        import resource
        # Linux reports ru_maxrss in KiB, macOS in bytes; assume Linux (cluster).
        out["peak_cpu_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    except Exception:
        pass
    return out


def _checkpoint_label_from_path(checkpoint_path: str) -> str:
    """
    Derive a readable label from a checkpoint path for compare tables.
    Example: ".../PPO_xxx/checkpoint_000019" -> "PPO_xxx_checkpoint_000019"
    """
    try:
        p = Path(checkpoint_path)
        parts = [x for x in p.parts if x]
        if len(parts) >= 2:
            return f"{parts[-2]}_{parts[-1]}"
        return p.name or "rllib"
    except Exception:
        return "rllib"


def run_evaluation(
    global_scheduler_name: str,
    local_scheduler_name: str,
    config: dict,
    num_episodes: int = 1,
    seed: int = 42,
    global_model_path: Optional[str] = None,
    local_model_path: Optional[str] = None,
    output_csv: Optional[str] = None,
    verbose: bool = True,
    force_full_episode: bool = False,
    global_defer: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run evaluation with specified Global + Local scheduler combination.

    Args:
        global_scheduler_name: Name of global scheduler (from GLOBAL_SCHEDULERS)
        local_scheduler_name: Name of local scheduler (from LOCAL_SCHEDULERS)
        config: Environment configuration
        num_episodes: Number of episodes to run
        seed: Random seed
        global_model_path: Path to trained global RL model (if using rl scheduler)
        local_model_path: Path to trained local RL models (if using rl scheduler)
        output_csv: Path to save results CSV
        verbose: Print progress

    Returns:
        List of metrics dicts, one per episode
    """
    np.random.seed(seed)

    # Create environment
    # If env_id indicates the simplified (no God's Eye) env, use the Simple version.
    env_id = config.get("env_id", "")
    use_simple_env = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple_env:
        logger.info("Creating HierarchicalMultiDCEnvSimple for evaluation (no God's Eye features)")
        env = HierarchicalMultiDCEnvSimple(config=config)
    else:
        env = HierarchicalMultiDCEnv(config=config)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    max_vms = env.max_vms

    if verbose:
        print(f"\n{'='*60}")
        print(f"Baseline Evaluation")
        print(f"{'='*60}")
        print(f"Global Scheduler: {global_scheduler_name}")
        print(f"Local Scheduler: {local_scheduler_name}")
        print(f"Environment: {num_dcs} DCs, batch_size={batch_size}, max_vms={max_vms}")
        print(f"Episodes: {num_episodes}, Seed: {seed}")
        print(f"{'='*60}\n")

    # Create Global Scheduler (heuristic or baseline; RLlib uses run_rllib_evaluation)
    GlobalCls = GLOBAL_SCHEDULERS[global_scheduler_name]
    global_scheduler = GlobalCls(num_dcs, batch_size)
    # Fair comparison with arch-B RL: give the heuristic the same forecast-driven
    # DEFER lever (only the routing/defer DECISION differs then). Needs the env's
    # global_defer_enabled=true so the env accepts the defer action index.
    if global_defer:
        from src.baselines.global_schedulers import DeferringGlobalScheduler
        global_scheduler = DeferringGlobalScheduler(global_scheduler, num_dcs, batch_size)
        if verbose:
            print(f"Global defer ENABLED — wrapped {global_scheduler_name} with forecast-driven defer rule")

    # Create Local Schedulers (one per DC)
    LocalCls = LOCAL_SCHEDULERS[local_scheduler_name]
    local_schedulers = {
        dc_id: LocalCls(max_vms) for dc_id in range(num_dcs)
    }

    all_results = []

    for ep in range(num_episodes):
        # Reset schedulers
        global_scheduler.reset()
        for scheduler in local_schedulers.values():
            scheduler.reset()

        # Reset environment
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0
        global_decision_ns: List[int] = []
        local_decision_ns: List[int] = []

        # Efficiency overhead: reset the GPU peak counter and start the
        # wall-clock for the whole simulated episode.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        ep_wall_t0 = time.perf_counter_ns()

        while not done:
            # Convert observation for global scheduler
            global_obs = _convert_global_obs_for_scheduler(obs['global'])

            # Global scheduling: select DC for each cloudlet in batch
            t0 = time.perf_counter_ns()
            global_action = global_scheduler.schedule(global_obs)
            global_decision_ns.append(time.perf_counter_ns() - t0)

            # Local scheduling: select VM for each DC
            local_actions = {}
            for dc_id in range(num_dcs):
                # Get action mask for this DC
                mask = env.get_local_action_masks(dc_id)

                # Convert local observation for scheduler
                local_obs = _convert_local_obs_for_scheduler(obs['local'].get(dc_id, {}))

                # Schedule
                t0 = time.perf_counter_ns()
                local_actions[dc_id] = local_schedulers[dc_id].schedule(local_obs, mask)
                local_decision_ns.append(time.perf_counter_ns() - t0)

            # Execute action
            action = {'global': global_action, 'local': local_actions}
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            # When force_full_episode=True, ignore the env's natural-completion
            # signal so every algorithm runs the same number of steps. Carbon /
            # energy totals become directly comparable across algorithms that
            # would otherwise drain the workload at different speeds.
            done = truncated if force_full_episode else (terminated or truncated)

        # Collect metrics at end of episode
        metrics = collect_metrics(info, num_dcs)
        metrics['episode'] = ep + 1
        metrics['episode_length'] = steps
        metrics.update(_summarize_decision_latency(global_decision_ns, "global_decision"))
        metrics.update(_summarize_decision_latency(local_decision_ns, "local_decision"))
        # Efficiency overhead: episode wall-clock (s) and peak memory footprint.
        metrics["episode_wall_s"] = (time.perf_counter_ns() - ep_wall_t0) / 1e9
        metrics.update(_capture_memory_mb())
        all_results.append(metrics)

        if verbose:
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Steps={steps}, "
                  f"Routed={metrics['routed_rate']:.2%}, "
                  f"Finished={metrics['finished_rate']:.2%}, "
                  f"GreenRatio={metrics['green_ratio']:.2%}, "
                  f"WasteRatio={metrics['waste_ratio']:.2%}, "
                  f"Carbon={metrics['total_carbon_kg']:.4f}kg")

    # Close environment
    env.close()

    # Save results to CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        if verbose:
            print(f"\nResults saved to: {output_csv}")

    # Print summary
    if verbose:
        _print_summary(global_scheduler_name, local_scheduler_name, all_results, num_dcs)

    return all_results


def _convert_global_obs_for_scheduler(global_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert global observation to scheduler-friendly format."""
    return {
        'dc_queue_sizes': global_obs.get('dc_queue_sizes', []),
        'dc_green_ratio': global_obs.get('dc_green_ratio', []),
        'dc_utilizations': global_obs.get('dc_utilizations', []),
        'dc_available_pes': global_obs.get('dc_available_pes', []),
        'dc_current_green_power_w': global_obs.get('dc_current_green_power_w', []),
        'dc_current_power_w': global_obs.get('dc_current_power_w', []),
        # Forecast — needed by the defer rule (heuristic baselines with global defer)
        'dc_future_short_mean': global_obs.get('dc_future_short_mean', []),
        'dc_future_long_mean': global_obs.get('dc_future_long_mean', []),
        'upcoming_cloudlets_count': global_obs.get('upcoming_cloudlets_count', 0),
        'batch_cloudlet_pes': global_obs.get('batch_cloudlet_pes', []),
        'batch_cloudlet_mi': global_obs.get('batch_cloudlet_mi', []),
        'load_imbalance': global_obs.get('load_imbalance', [0]),
    }


def _convert_local_obs_for_scheduler(local_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert local observation to scheduler-friendly format."""
    return {
        'vm_loads': local_obs.get('vm_loads', []),
        'vm_available_pes': local_obs.get('vm_available_pes', []),
        'vm_types': local_obs.get('vm_types', []),
        'host_loads': local_obs.get('host_loads', []),
        'waiting_cloudlets': local_obs.get('waiting_cloudlets', 0),
        'next_cloudlet_pes': local_obs.get('next_cloudlet_pes', 0),
    }


def _print_summary(
    global_name: str,
    local_name: str,
    results: List[Dict[str, Any]],
    num_dcs: int
):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {global_name.upper()} + {local_name.upper()}")
    print(f"{'='*60}")

    # Aggregate metrics
    avg_routed_rate = np.mean([r['routed_rate'] for r in results])
    avg_finished_rate = np.mean([r['finished_rate'] for r in results])
    avg_green_ratio = np.mean([r['green_ratio'] for r in results])
    avg_waste_ratio = np.mean([r['waste_ratio'] for r in results])
    avg_carbon = np.mean([r['total_carbon_kg'] for r in results])
    avg_carbon_intensity = np.mean([r.get('carbon_intensity', 0) for r in results])
    avg_carbon_per_cloudlet = np.mean([r['carbon_per_finished_cloudlet'] for r in results])
    avg_carbon_per_mi = np.mean([r['carbon_per_completion_mi'] for r in results])
    avg_steps = np.mean([r['episode_length'] for r in results])
    total_energy = np.mean([r['total_energy_wh'] for r in results])
    avg_green_used = np.mean([r.get('green_used_wh', 0) for r in results])
    avg_brown_used = np.mean([r.get('brown_used_wh', 0) for r in results])

    print(f"Avg Episode Length: {avg_steps:.1f} steps")
    print(f"Avg Routed Rate: {avg_routed_rate:.2%}  (cloudlets dispatched to DCs)")
    print(f"Avg Finished Rate: {avg_finished_rate:.2%}  (cloudlets actually completed)")
    print(f"Avg Green Ratio: {avg_green_ratio:.2%}")
    print(f"Avg Waste Ratio: {avg_waste_ratio:.2%}")
    print(f"Avg Green Energy Used: {avg_green_used:.2f} Wh")
    print(f"Avg Brown Energy Used: {avg_brown_used:.2f} Wh")
    print(f"Avg Total Energy: {total_energy:.2f} Wh")
    print(f"Avg Carbon Emission: {avg_carbon:.4f} kg")
    print(f"Avg Carbon Intensity: {avg_carbon_intensity:.4f} kg/kWh")
    print(f"Avg Carbon/Cloudlet: {avg_carbon_per_cloudlet*1000:.4f} g/task")
    print(f"Avg Carbon/MI: {avg_carbon_per_mi:.6f} kg/mi-completion")

    if results and "global_decision_us_mean" in results[0]:
        gmean = np.mean([r["global_decision_us_mean"] for r in results])
        gp95  = np.mean([r["global_decision_us_p95"]  for r in results])
        gp99  = np.mean([r["global_decision_us_p99"]  for r in results])
        lmean = np.mean([r["local_decision_us_mean"]  for r in results])
        lp95  = np.mean([r["local_decision_us_p95"]   for r in results])
        lp99  = np.mean([r["local_decision_us_p99"]   for r in results])
        print(f"Decision latency  global: mean={gmean:.1f}us p95={gp95:.1f}us p99={gp99:.1f}us")
        print(f"Decision latency  local : mean={lmean:.1f}us p95={lp95:.1f}us p99={lp99:.1f}us")

    # Per-DC completion summary
    print(f"\nPer-DC Cloudlets Finished:")
    for dc_id in range(num_dcs):
        avg_finished = np.mean([r.get(f'finished_dc_{dc_id}', 0) for r in results])
        if any(f"received_dc_{dc_id}" in r for r in results):
            avg_received = np.mean([r.get(f"received_dc_{dc_id}", 0) for r in results])
            print(f"  DC {dc_id}: finished={avg_finished:.0f}, received={avg_received:.0f}")
        else:
            print(f"  DC {dc_id}: {avg_finished:.0f}")

    print(f"{'='*60}\n")


def compare_baselines(
    experiment_name: str,
    combinations: List[tuple],
    num_episodes: int = 1,
    seed: int = 42,
    output_dir: Optional[str] = None,
    print_table: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare multiple Global+Local scheduler combinations.

    Args:
        experiment_name: Experiment config to use
        combinations: List of (global_name, local_name) tuples
        num_episodes: Episodes per combination
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dict mapping "global_local" to results list
    """
    config = load_config(experiment_name)
    all_results = {}

    for global_name, local_name in combinations:
        combo_name = f"{global_name}_{local_name}"
        print(f"\n\n{'#'*60}")
        print(f"# Evaluating: {combo_name}")
        print(f"{'#'*60}")

        output_csv = None
        if output_dir:
            output_csv = f"{output_dir}/{combo_name}.csv"

        results = run_evaluation(
            global_scheduler_name=global_name,
            local_scheduler_name=local_name,
            config=config,
            num_episodes=num_episodes,
            seed=seed,
            output_csv=output_csv
        )

        all_results[combo_name] = results

    # Print comparison table (optional, can be suppressed to allow external aggregation)
    if print_table:
        _print_comparison_table(all_results)

    # If an output directory is provided, also save a combined summary CSV that includes
    # all episodes for all algorithm combinations. Each row is one episode with:
    #   - 'combo': "<global>_<local>"
    #   - 'episode': episode index (1-based, already in metrics)
    #   - all metric columns (green_ratio, completion_rate, mean_completion_time, etc.)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_rows: List[Dict[str, Any]] = []
        for combo_name, results in all_results.items():
            for r in results:
                row = dict(r)  # shallow copy
                row["combo"] = combo_name
                combined_rows.append(row)

        if combined_rows:
            first = combined_rows[0]
            fieldnames = ["combo"]
            if "episode" in first:
                fieldnames.append("episode")
            # Add remaining keys in a stable order
            for k in first.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

            summary_path = out_dir / "summary.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined_rows)
            print(f"[compare_baselines] Combined summary saved to: {summary_path}")

    return all_results


def _print_comparison_table(all_results: Dict[str, List[Dict[str, Any]]]):
    """Print comparison table for all combinations."""
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Combination':<30} {'Routed':>10} {'Finished':>10} {'Green Ratio':>12} {'Carbon (kg)':>12}")
    print(f"{'-'*100}")

    for combo_name, results in all_results.items():
        avg_routed = np.mean([r['routed_rate'] for r in results])
        avg_finished = np.mean([r['finished_rate'] for r in results])
        avg_green = np.mean([r['green_ratio'] for r in results])
        avg_carbon = np.mean([r['total_carbon_kg'] for r in results])

        print(f"{combo_name:<30} {avg_routed:>9.2%} {avg_finished:>9.2%} {avg_green:>11.2%} {avg_carbon:>12.4f}")

    print(f"{'='*100}\n")


def run_rllib_evaluation(
    checkpoint_path: str,
    config: dict,
    num_episodes: int = 1,
    seed: int = 42,
    output_csv: Optional[str] = None,
    verbose: bool = True,
    shared_local: bool = False,
    use_new_api: bool = False,
    py4j_port: Optional[int] = None,
    force_full_episode: bool = False,
    stochastic: bool = False,
    local_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用 RLlib 训练好的模型进行评估（Global + Local 都用 RL）。

    - 标准模式：每个 DC 有独立的本地策略（local_policy_{dc_id}），使用
      `create_rllib_schedulers` 创建全局/本地调度器；
    - 参数共享模式：如果传入 `shared_local=True`，则所有 DC 使用同一个
      `"shared_local_policy"` 作为本地调度器（适用于 parameter sharing 训练）。
    - New API Stack 模式：如果传入 `use_new_api=True`，则使用 RLModule 直接推理
      （适用于 enable_rl_module_and_learner=True 训练的模型）。

    Args:
        checkpoint_path: RLlib checkpoint 路径
        config: 环境配置
        num_episodes: 评估轮数
        seed: 随机种子
        output_csv: 结果保存路径
        verbose: 是否打印详情
        shared_local: 是否将所有本地调度器绑定到同一个 shared_local_policy
        use_new_api: 是否使用 New API Stack (RLModule) 进行推理
    """
    from src.baselines.global_schedulers import load_rllib_algorithm, RLlibGlobalScheduler
    from src.baselines.local_schedulers import create_rllib_schedulers, RLlibLocalScheduler

    np.random.seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print("RLlib Model Evaluation (Global + Local RL)")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        # Print workload config (what trace file this evaluation will use)
        wl_mode = config.get("workload_mode", "UNKNOWN")
        wl_file = config.get("cloudlet_trace_file", "UNKNOWN")
        wl_max = config.get("max_cloudlets_to_create_from_workload_file", None)
        ep_len = config.get("max_episode_length", None)
        print(f"Workload: mode={wl_mode}, trace={wl_file}, max_cloudlets={wl_max}, max_episode_length={ep_len}")
        print("Loading model...")

    # 1. 加载 RLlib 模型
    # IMPORTANT: Algorithm.from_checkpoint() may internally initialize env runners/envs.
    # If you are running training concurrently, override py4j_port to avoid colliding
    # with the training Java gateway.
    algo = load_rllib_algorithm(checkpoint_path, py4j_port_override=py4j_port)
    # The trust sentinel (TRUST_GATE_MODE) loads the EU-CRD Q-ensemble weights
    # straight from the checkpoint files; expose the path to the scheduler.
    os.environ["EVAL_CHECKPOINT_PATH"] = str(Path(checkpoint_path).resolve())

    if verbose:
        print("✓ Model loaded!")

    # 2. 创建环境（根据 env_id 选择是否使用简化版，无 God's Eye 特征）
    # Ensure evaluation env connects to the requested Java gateway instance.
    if py4j_port is not None:
        config = dict(config)
        config["py4j_port"] = int(py4j_port)
    env_id = config.get("env_id", "")
    use_simple_env = "Simple" in env_id or env_id == "HierarchicalMultiDCSimple-v0"

    if use_simple_env:
        if verbose:
            print("Creating HierarchicalMultiDCEnvSimple for RLlib evaluation (no God's Eye features)")
        env = HierarchicalMultiDCEnvSimple(config=config)
    else:
        env = HierarchicalMultiDCEnv(config=config)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    max_vms = env.max_vms

    if verbose:
        print(f"Environment: {num_dcs} DCs, batch_size={batch_size}, max_vms={max_vms}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}\n")

    # Get max_hosts from env (for parameter sharing observations)
    max_hosts = getattr(env, 'max_hosts', 16)

    # 3. 创建调度器
    if use_new_api:
        # New API Stack 模式：使用 RLModule 直接推理
        from src.baselines.local_schedulers import create_rllib_new_api_schedulers
        if verbose:
            print("Using New API Stack (RLModule) for inference.")
        global_scheduler, local_schedulers = create_rllib_new_api_schedulers(
            algo=algo,
            env=env,
            num_dcs=num_dcs,
            batch_size=batch_size,
            num_vms=max_vms,
            max_hosts=max_hosts,
            stochastic=stochastic,
        )
    elif shared_local:
        # 所有 DC 显式使用同一个本地策略 "shared_local_policy"
        if verbose:
            print("Using shared_local_policy for all local agents (parameter sharing mode).")
        global_scheduler = RLlibGlobalScheduler(num_dcs, batch_size, algo)
        local_schedulers = {
            dc_id: RLlibLocalScheduler(
                max_vms, algo, dc_id, env=env, policy_id="shared_local_policy",
                use_parameter_sharing=True,
                num_datacenters=num_dcs,
                max_hosts=max_hosts,
                max_vms=max_vms,
            )
            for dc_id in range(num_dcs)
        }
    else:
        # 标准多策略情形：每个 DC 有自己的 local_policy_{dc_id}
        global_scheduler, local_schedulers = create_rllib_schedulers(
            algo=algo,
            env=env,
            num_dcs=num_dcs,
            batch_size=batch_size,
            num_vms=max_vms,
            max_hosts=max_hosts,
        )

    # De-confound mode (2026-08-13): rllib GLOBAL + scripted LOCAL. Each arm's
    # checkpoint carries a local policy co-learned with that arm's own routing
    # distribution, so `--local rllib` cross-arm comparisons mix the global
    # policy under test with an arm-specific local (docs/V3_FORECAST_DIAGNOSIS.md
    # §2b). Passing e.g. `--local drain` here swaps every DC's local for the
    # scripted scheduler while keeping the rllib global untouched. No-op when
    # local_override is None/'rllib', so pre-registered `--local rllib` runs
    # (P3 chain) are byte-identical.
    if local_override and local_override != 'rllib':
        from src.baselines.local_schedulers import LOCAL_SCHEDULERS as _LOCAL_LS
        _OverrideCls = _LOCAL_LS[local_override]
        local_schedulers = {dc_id: _OverrideCls(max_vms) for dc_id in range(num_dcs)}
        if verbose:
            print(f"Local override ACTIVE: checkpoint local policy replaced by "
                  f"'{local_override}' for all {num_dcs} DCs")

    all_results = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        steps = 0
        global_decision_ns: List[int] = []
        local_decision_ns: List[int] = []

        # Efficiency overhead: reset the GPU peak counter and start the
        # wall-clock for the whole simulated episode.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        ep_wall_t0 = time.perf_counter_ns()

        while not done:
            # Global scheduling
            t0 = time.perf_counter_ns()
            global_action = global_scheduler.schedule(obs['global'])
            global_decision_ns.append(time.perf_counter_ns() - t0)

            # Local scheduling
            local_actions = {}
            for dc_id in range(num_dcs):
                local_obs = obs['local'].get(dc_id, {})
                action_mask = env.get_local_action_masks(dc_id)
                t0 = time.perf_counter_ns()
                local_actions[dc_id] = local_schedulers[dc_id].schedule(local_obs, action_mask)
                local_decision_ns.append(time.perf_counter_ns() - t0)

            # 执行
            action = {'global': global_action, 'local': local_actions}
            obs, rewards, terminated, truncated, info = env.step(action)
            steps += 1
            done = truncated if force_full_episode else (terminated or truncated)

        # 收集指标
        metrics = collect_metrics(info, num_dcs)
        metrics['episode'] = ep + 1
        metrics['episode_length'] = steps
        metrics.update(_summarize_decision_latency(global_decision_ns, "global_decision"))
        metrics.update(_summarize_decision_latency(local_decision_ns, "local_decision"))
        # Efficiency overhead: episode wall-clock (s) and peak memory footprint.
        metrics["episode_wall_s"] = (time.perf_counter_ns() - ep_wall_t0) / 1e9
        metrics.update(_capture_memory_mb())
        all_results.append(metrics)

        if verbose:
            # Print cloudlet counts for this episode (debugging workload mismatch)
            print(
                f"[Cloudlets] total={metrics.get('total_cloudlets')}, "
                f"created={metrics.get('total_created_cloudlets')}, "
                f"received(sum)={metrics.get('total_received_cloudlets')}, "
                f"finished(sum)={metrics.get('sum_finished_dc')}"
            )
            forced = int(metrics.get('deadline_forced_count') or 0)
            fin = int(metrics.get('total_finished_cloudlets') or 0)
            forced_share = (forced / fin) if fin else 0.0
            print(f"Episode {ep+1}/{num_episodes}: "
                  f"Steps={steps}, "
                  f"Routed={metrics['routed_rate']:.2%}, "
                  f"Finished={metrics['finished_rate']:.2%}, "
                  f"GreenRatio={metrics['green_ratio']:.2%}, "
                  f"Carbon={metrics['total_carbon_kg']:.4f}kg, "
                  f"DeadlineForced={forced} ({forced_share:.1%} of finished)")

    env.close()

    sentinel = getattr(global_scheduler, "_sentinel", None)
    if sentinel is not None:
        print(sentinel.summary())
        sentinel.close()

    # 保存结果
    if output_csv is None:
        # 默认将 RLlib 评估结果写到 drl-manager/compare_result/rllib/<timestamp>.csv
        base_dir = Path(__file__).parent.parent.parent / "compare_result" / "rllib"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = str(base_dir / f"{timestamp}.csv")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    if verbose:
        print(f"\nResults saved to: {output_csv}")

    if verbose:
        _print_summary("RLlib", "RLlib", all_results, num_dcs)

    # Properly shutdown Algorithm and Ray to avoid cleanup errors at interpreter exit
    # NOTE: In RLlib New API Stack, LearnerGroup cleanup can otherwise run during
    # interpreter shutdown and emit "Exception ignored in: LearnerGroup.__del__".
    import ray
    try:
        # Stop the algorithm first to clean up LearnerGroup while Ray is alive.
        algo.stop()
    except Exception:
        pass

    # Force GC now (before ray.shutdown) so any __del__ runs while Ray is still initialized.
    try:
        import gc
        del algo
        gc.collect()
    except Exception:
        pass

    if ray.is_initialized():
        ray.shutdown()

    return all_results


if __name__ == "__main__":
    # HPC (Isambard aarch64) Ray guard: on a 288-core node, RLlib's auto ray.init() pre-warms
    # ~288 workers that all import the heavy stack from NFS at once → none register → the eval
    # HANGS forever at "Started a local Ray instance" (same 288-worker storm the training
    # entrypoint already guards against). Cap Ray to the cgroup allocation BEFORE any RLlib code
    # auto-inits it. Gated on RAY_LIMIT_CPUS — unset on the workstation, so local evals are unchanged.
    _ray_limit = os.environ.get("RAY_LIMIT_CPUS")
    if _ray_limit:
        import ray as _ray
        if not _ray.is_initialized():
            _ray.init(num_cpus=int(_ray_limit), include_dashboard=False, ignore_reinit_error=True)
            print(f"[eval] Ray initialized with num_cpus={_ray_limit} (HPC 288-worker-storm guard)")

    parser = argparse.ArgumentParser(description="Evaluate baseline scheduling algorithms")

    parser.add_argument("--global", dest="global_sched", type=str, default='random',
                        choices=list(GLOBAL_SCHEDULERS.keys()) + ['rllib'],
                        help="Global scheduler algorithm")
    parser.add_argument("--local", dest="local_sched", type=str, default='random',
                        choices=list(LOCAL_SCHEDULERS.keys()) + ['rllib'],
                        help="Local scheduler algorithm")
    parser.add_argument("--experiment", type=str, default="experiment_multi_dc_10",
                        help="Experiment name from config.yml")
    parser.add_argument("--global-defer", dest="global_defer", action="store_true",
                        help="Give the heuristic global scheduler the forecast-driven DEFER lever "
                             "(fair comparison with arch-B RL; needs global_defer_enabled in the config)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (repeatable). Example: --override max_cloudlets_to_create_from_workload_file=100000",
    )
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RLlib checkpoint path (for rllib scheduler)")
    parser.add_argument(
        "--rllib-checkpoint",
        action="append",
        default=[],
        help="(Optional, repeatable) Additional RLlib checkpoints to evaluate in --compare mode. "
             "Example: --rllib-checkpoint /abs/path/to/checkpoint_000019",
    )
    parser.add_argument(
        "--rllib-label",
        action="append",
        default=[],
        help="(Optional, repeatable) Labels for --rllib-checkpoint entries (same order). "
             "If omitted, a label is derived from the checkpoint path.",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results CSV (for single run or rllib evaluation)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base directory to save compare_result outputs when using --compare")
    parser.add_argument("--shared-local", action="store_true",
                        help="For RLlib evaluation: treat all local agents as sharing 'shared_local_policy'")
    parser.add_argument("--stochastic", action="store_true",
                        help="For RLlib (new-API) evaluation: sample global routing choices from the policy "
                             "distribution instead of greedy argmax. Needed for a faithful iso-completion "
                             "comparison — greedy argmax collapses all 128 same-obs routing slots onto one DC.")
    parser.add_argument("--new-api", action="store_true",
                        help="For RLlib evaluation: use New API Stack (RLModule) for inference. "
                             "Required for models trained with enable_rl_module_and_learner=True (e.g., GTrXL)")
    parser.add_argument(
        "--auto-new-api",
        action="store_true",
        help="Auto-detect New API Stack from checkpoint contents (learner_group/) when running --compare. "
             "If set, each checkpoint will use New API inference if it looks like an RLModule checkpoint.",
    )
    parser.add_argument("--py4j-port", type=int, default=None,
                        help="Override py4j_port for evaluation to connect to a different Java gateway instance "
                             "(recommended when training is running on another port).")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison of multiple baseline combinations")

    args = parser.parse_args()

    config = load_config(args.experiment)
    # Apply CLI overrides (useful for evaluation-only workload tweaks without editing config.yml)
    if args.override:
        config = _apply_overrides(config, args.override)

    # Gateway lifecycle (mirrors compare_algorithms.py): explicit --py4j-port
    # connects to that gateway; omitted -> strip the config's baked-in port so
    # the env auto-launches its own gradlew subprocess on a free port. Without
    # this the CLI tries config.yml's hardcoded 25333 and fails when no
    # gateway is running there.
    if args.py4j_port is not None:
        config["py4j_port"] = int(args.py4j_port)
    else:
        config["py4j_port"] = None
    # Auto-launch requires gateway_log_dir; default to a tmp path if config.yml
    # doesn't specify one.
    config.setdefault(
        "gateway_log_dir",
        str(Path("/tmp") / f"evaluate_gateways_{datetime.now():%Y%m%d_%H%M%S}"),
    )

    if args.global_sched == 'rllib' or args.local_sched == 'rllib':
        # 使用 RLlib 模型评估
        if args.checkpoint is None:
            print("Error: --checkpoint required for rllib scheduler")
            sys.exit(1)

        run_rllib_evaluation(
            checkpoint_path=args.checkpoint,
            config=config,
            num_episodes=args.episodes,
            seed=args.seed,
            output_csv=args.output,
            shared_local=args.shared_local,
            use_new_api=args.new_api,
            py4j_port=args.py4j_port,
            stochastic=args.stochastic,
            local_override=args.local_sched,
        )
    elif args.compare:
        # 比较多个组合
        combinations = [
            ('random', 'random'),
            ('round_robin', 'round_robin'),
            ('min_queue', 'first_fit'),
            ('green_aware', 'best_fit'),
            ('green_queue_balanced', 'min_load'),
        ]

        # If user provides --output-dir, use it; otherwise create a timestamped folder
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(__file__).parent.parent.parent / "compare_result"
            output_dir = base_dir / timestamp

        # 1) 先跑所有 heuristic 组合，结果写到同一 timestamp 目录下
        all_results = compare_baselines(
            experiment_name=args.experiment,
            combinations=combinations,
            num_episodes=args.episodes,
            seed=args.seed,
            output_dir=str(output_dir),
            print_table=False,  # 暂时不打印表格，后面合并 RLlib 一起打印
        )

        # 2) Optional: evaluate one or more RLlib checkpoints (ResMLP/gMLP/GTrXL, etc.) in the same compare run.
        rllib_checkpoints: List[str] = []
        if args.checkpoint is not None:
            rllib_checkpoints.append(args.checkpoint)
        if args.rllib_checkpoint:
            rllib_checkpoints.extend(list(args.rllib_checkpoint))

        labels: List[str] = []
        if args.rllib_label:
            labels = list(args.rllib_label)
            if len(labels) != len(rllib_checkpoints):
                print("Error: --rllib-label count must match total RLlib checkpoints provided "
                      "(--checkpoint + --rllib-checkpoint).")
                sys.exit(1)

        for i, ckpt in enumerate(rllib_checkpoints):
            label = labels[i] if labels else _checkpoint_label_from_path(ckpt)
            safe_label = "".join(c if (c.isalnum() or c in "-_") else "_" for c in label)
            rllib_csv = str(output_dir / f"rllib_{safe_label}.csv")

            use_new_api = args.new_api
            if (not use_new_api) and args.auto_new_api:
                use_new_api = _infer_use_new_api_from_checkpoint(ckpt)

            rllib_results = run_rllib_evaluation(
                checkpoint_path=ckpt,
                config=config,
                num_episodes=args.episodes,
                seed=args.seed,
                output_csv=rllib_csv,
                shared_local=args.shared_local,
                use_new_api=use_new_api,
            )
            all_results[f"rllib_{safe_label}"] = rllib_results

        # 3) 打印包含所有组合（包含 RLlib）的对比表
        _print_comparison_table(all_results)
    else:
        # 单个评估
        run_evaluation(
            global_scheduler_name=args.global_sched,
            local_scheduler_name=args.local_sched,
            config=config,
            num_episodes=args.episodes,
            seed=args.seed,
            output_csv=args.output,
            global_defer=args.global_defer,
        )
