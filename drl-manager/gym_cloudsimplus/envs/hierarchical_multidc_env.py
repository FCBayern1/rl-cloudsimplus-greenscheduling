"""
debug_ga_obs.py -- 调试观测
src/baselines/evaluate.py -- 基线评估（full/Simple 都用）
tests/verify_action_mask_logic.py, tests/test_reset_gymnasium_compliance.py, tests/test_green_energy.py -- 测试

hierarchical_multidc_pettingzoo.py 直接使用

Hierarchical Multi-Datacenter Reinforcement Learning Environment

This environment implements a two-level hierarchical MARL system:
- Global Level: Routes arriving cloudlets to datacenters (Global Agent)
- Local Level: Schedules cloudlets to VMs within each datacenter (Local Agents)

Architecture:
    Python (Gymnasium) <--> Py4J <--> Java (CloudSim Plus Multi-DC Simulation)
"""

import logging
import time
import subprocess
import socket
import sys
import os
import signal
import atexit
import shutil
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)


# region agent log
def _write_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]):
    try:
        with open("/home/joshua/rl-cloudsimplus-greenscheduling/.cursor/debug-f7b29b.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "f7b29b",
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# endregion


def perturb_future_bins(bins, mode: str, capacity_w) -> "np.ndarray":
    """Apply FORECAST_PERTURB semantics to the V3.2 future-green WATT bins.

    Closes the derived-feature leak (V32B_ANNEAL_SPEC R1 item 7): the dc-level
    channels are perturbed downstream, but gain/time-to-best/best-future are
    computed from these bins, which previously always carried the clean truth.

    shuffle: reverse the DC axis (same coherent-wrong semantics as the
             normalized channels).
    anti:    A-prime per-DC static capacity mirror (Codex 2026-08-17):
             clip(H_d - G, 0, H_d), H_d = sum of the DC's turbine max power in
             simulator watts (calib/v3_anti_capacity.json). H_d = 0 stays 0 -
             a uniform ceiling would fabricate green on turbine-less DCs.
             Equals the normalized channels' 1 - G/H_d mirror exactly and is
             an involution on [0, H_d].
    """
    bins = np.asarray(bins, dtype=np.float64)
    mode = (mode or "none").strip().lower()
    if mode == "shuffle":
        return bins[::-1].copy()
    if mode == "anti":
        cap = np.asarray(capacity_w if capacity_w is not None else [],
                         dtype=np.float64).reshape(-1)
        if cap.size != bins.shape[0]:
            raise RuntimeError(
                f"anti bins perturbation needs a per-DC capacity vector of "
                f"length {bins.shape[0]}, got {cap.size} "
                f"(set v32_perturb_capacity_w from calib/v3_anti_capacity.json)")
        out = np.clip(cap[:, None] - bins, 0.0,
                      np.maximum(cap[:, None], 0.0))
        out[cap <= 0.0, :] = 0.0
        return out
    return bins


def episode_offset_rows(episode_counter, offset_range, allowlist=None):
    """Green-window offset for one episode, mirrored bit-for-bit from the Java side
    (MultiDatacenterSimulationCore.episodeOffsetFor): a non-empty allowlist
    ("13016;21088" or a list) is cycled by episode index and wins; otherwise the
    1009*k mod range schedule; range <= 0 keeps the historical fixed window."""
    allow = allowlist
    if isinstance(allow, str):
        allow = [int(x) for x in allow.replace(",", ";").split(";") if x.strip()]
    if allow:
        return max(0, int(allow[int(episode_counter) % len(allow)]))
    if offset_range and int(offset_range) > 0 and episode_counter >= 0:
        return (1009 * int(episode_counter)) % int(offset_range)
    return 0


def defer_allowed_from(time_to_deadline_sec, deadline_present, mi, pes, vm_pe_mips, cpu_utilization,
                       margin_sec, timestep_sec):
    """Deadline-safe DEFER mask, one entry per batch slot (STAGE_D_PRIME_DESIGN §3).

    A slot may still wait one more decision step iff, after that step, the job can still be
    started and finish by its deadline with `margin_sec` to spare:

        time_to_deadline - timestep - runtime - margin > 0,   runtime = mi / (mips * u)

    which is the Java backstop's own runtime unit (PerActionRewardMath.deadlineForceLatestStart)
    evaluated one step ahead, so the mask fires before the backstop ever has to. Slots without a
    deadline may always wait; padding slots (mi <= 0) are 0. Pure numpy; shared by every arm.
    """
    ttd = np.asarray(time_to_deadline_sec, dtype=np.float64).reshape(-1)
    present = np.asarray(deadline_present, dtype=np.float64).reshape(-1)
    mi_a = np.asarray(mi, dtype=np.float64).reshape(-1)
    n = ttd.shape[0]
    present = present[:n] if present.shape[0] >= n else np.concatenate([present, np.zeros(n - present.shape[0])])
    mi_a = mi_a[:n] if mi_a.shape[0] >= n else np.concatenate([mi_a, np.zeros(n - mi_a.shape[0])])
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_utilization)))
    runtime = mi_a / rate
    slack_after_wait = ttd - float(timestep_sec) - runtime - float(margin_sec)
    allowed = np.where(present >= 0.5, slack_after_wait > 0.0, True)
    allowed = np.where(mi_a > 0.0, allowed, False)
    return allowed.astype(np.float32)


def route_disallowed_defers(actions, defer_allowed, pes, dc_green_ratio, dc_available_pes, num_dcs):
    """Replace DEFER (index num_dcs) on slots with defer_allowed == 0 by the greenest DC that
    has room for the slot's PEs (else the greenest DC). Returns (new_actions, n_routed). Pure."""
    acts = [int(a) for a in actions]
    da = np.asarray(defer_allowed, dtype=np.float64).reshape(-1)
    green = np.asarray(dc_green_ratio if dc_green_ratio is not None else np.zeros(num_dcs), dtype=np.float64).reshape(-1)
    avail = np.asarray(dc_available_pes if dc_available_pes is not None else np.zeros(num_dcs), dtype=np.float64).reshape(-1)
    pes_a = np.asarray(pes if pes is not None else np.ones(len(acts)), dtype=np.float64).reshape(-1)
    order = list(np.argsort(-green[:num_dcs], kind="stable"))
    n_routed = 0
    for i, a in enumerate(acts):
        if a != num_dcs or i >= da.shape[0] or da[i] >= 0.5:
            continue
        need = float(pes_a[i]) if i < pes_a.shape[0] else 1.0
        if pes is not None and need <= 0.0:
            continue            # padding slot: no job here, Java ignores it, never counted
        choice = next((int(d) for d in order if avail[d] >= need), int(order[0]))
        acts[i] = choice
        n_routed += 1
    return acts, n_routed


class HierarchicalMultiDCEnv(gym.Env):
    """
    Hierarchical Multi-Datacenter Load Balancing Environment.

    Two-level decision making:
    1. Global Agent: Routes arriving cloudlets to datacenters
    2. Local Agents: Assign cloudlets to VMs within each datacenter

    Action Space:
        - Global: Discrete(num_datacenters) for each arriving cloudlet
        - Local: Discrete(num_vms_per_dc) for each datacenter

    Observation Space:
        - Global: Aggregated state of all datacenters (green power, queues, utilisation)
        - Local: Per-DC state (VM loads, local queues, next cloudlet)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    # Class-level guard to avoid closing the Java gateway multiple times across instances
    _java_gateway_closed = False

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hierarchical multi-datacenter environment.

        Args:
            config: Configuration dictionary containing:
                - multi_datacenter_enabled: bool
                - datacenters: List[dict] of datacenter configurations
                - py4j_port: int (optional, if not provided, a free port is found and a new Java process is launched)
                - global_routing_batch_size: int (cloudlets to route per step, default 5)
                - max_arriving_cloudlets: int (deprecated, for backward compatibility)
                - ... other CloudSim Plus settings
        """
        super(HierarchicalMultiDCEnv, self).__init__()

        self.config = config

        # When True, only build observation/action spaces without launching Java.
        # Used by training scripts that need space shapes for policy construction.
        self._spaces_only = bool(config.get("spaces_only", False))

        # Java Gateway Process Management
        self.java_process = None
        self.py4j_port = config.get("py4j_port")

        if not self._spaces_only:
            if self.py4j_port is None or self.py4j_port == 0:
                self.py4j_port = self._find_free_port()
                self._launch_java_gateway(self.py4j_port)
            else:
                logger.info(f"Using existing Java gateway on port {self.py4j_port}")
        else:
            logger.info("spaces_only mode: skipping Java gateway launch")

        # Cache DC configs and build a stable index <-> dcId mapping.
        # Internally we use dcIndex (0..N-1) for array indexing and observation spaces.
        self.dc_configs = config.get("datacenters")
        if not self.dc_configs:
            self.dc_configs = [{"datacenter_id": 0}]

        self.num_datacenters = len(self.dc_configs)
        self.dc_ids = [
            int(dc.get("datacenter_id", idx)) for idx, dc in enumerate(self.dc_configs)
        ]
        self.dc_id_to_index = {dc_id: idx for idx, dc_id in enumerate(self.dc_ids)}
        self.dc_index_to_id = {idx: dc_id for idx, dc_id in enumerate(self.dc_ids)}
        if len(self.dc_id_to_index) != len(self.dc_ids):
            logger.warning("Duplicate datacenter_id values detected in config; dcId->index mapping may be ambiguous.")
        
        # Fixed batch size for global routing decisions (key parameter)
        self.global_routing_batch_size = config.get("global_routing_batch_size", 10)
        
        # Backward compatibility: if max_arriving_cloudlets is set, use it as batch size
        if "max_arriving_cloudlets" in config and "global_routing_batch_size" not in config:
            logger.warning(
                "'max_arriving_cloudlets' is deprecated. Use 'global_routing_batch_size' instead."
            )
            self.global_routing_batch_size = config.get("max_arriving_cloudlets", 10)

        # Py4J Gateway connection
        self.gateway = None
        self.java_env = None

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # CRD framework: cached per-DC carbon factors and timestep duration.
        # These are stable for the simulation lifetime; populated lazily on first
        # access from Java to avoid extra Py4J round-trips per step.
        # Evaluator-only planner inputs from the most recent observation parse; None
        # until the first parse, and on a gateway built before getBatchCloudletIds.
        self._planner_channel: Optional[Dict[str, Any]] = None
        self._crd_green_factors: Optional[List[float]] = None
        self._crd_brown_factors: Optional[List[float]] = None
        self._crd_timestep_hours: Optional[float] = None
        # CRD framework: cache the most recent parsed global obs so
        # _collect_crd_info can pull decision-time signals (queue sizes,
        # green ratio) for the M2.3 baseline scheduler without extra Py4J calls.
        self._last_global_obs_for_crd: Dict[str, Any] = {}

        # V3.1 defer-state observations. Java always maintains the factual
        # lifecycle ledger, while this gate alone controls the Python policy
        # schema so default-off runs and legacy checkpoints remain identical.
        self.obs_v31_features = bool(config.get("obs_v31_features", False))
        self._obs_v31_wait_age_scale = max(
            1.0,
            float(config.get(
                "obs_v31_wait_age_scale_sec",
                float(config.get("max_episode_length", 7200))
                * float(config.get("simulation_timestep", 1.0)),
            )),
        )
        self._obs_v31_deadline_scale = max(
            1.0,
            float(config.get(
                "obs_v31_deadline_scale_sec",
                config.get("defer_urgency_window_sec", 3600.0),
            )),
        )
        self._obs_v31_defer_count_scale = max(
            1.0,
            # Counts above this are behaviorally all "many re-encounters";
            # scaling by the 7200-step episode would squash the typical 1..10
            # range into numerical noise.
            float(config.get("obs_v31_defer_count_scale", 32.0)),
        )
        self._obs_v31_global_count_scale = max(
            # V3/V3.1 traces contain 1200--2000 jobs. Keep that regime visible
            # while allowing larger scenarios to override the scale explicitly.
            1.0, float(config.get("obs_v31_global_deferred_count_scale", 2000.0))
        )
        self._obs_v31_global_mi_scale = max(
            1.0,
            float(config.get(
                "obs_v31_global_deferred_mi_scale",
                self._obs_v31_global_count_scale
                * float(config.get("obs_cloudlet_mi_high", 2000000)),
            )),
        )

        # V3.2 job-aligned forecast observations.  This is deliberately a
        # sibling gate to obs_v31_features: when disabled, neither the schema
        # nor the legacy forecast_mode="none" zero-fill changes.  When enabled,
        # the temporal head receives per-cloudlet features derived from that
        # arm's own information set (full=godeye/TimeCAP, none=persistence).
        # Stage D' deadline-safe DEFER mask: off by default so every frozen run stays
        # byte-identical. Needs obs_v31_features (the per-job deadline facts).
        self.defer_deadline_mask = bool(config.get("defer_deadline_mask", False))
        self._defer_mask_margin_sec = max(0.0, float(config.get("defer_deadline_mask_margin_sec", 0.0)))
        self._mask_route_count = 0     # DEFERs re-routed by the env-side mask this episode
        if self.defer_deadline_mask and not self.obs_v31_features:
            raise ValueError("defer_deadline_mask=true requires obs_v31_features=true")
        self.obs_v32_job_forecast = bool(
            config.get("obs_v32_job_forecast", False)
        )
        if self.obs_v32_job_forecast and not self.obs_v31_features:
            raise ValueError(
                "obs_v32_job_forecast=true requires obs_v31_features=true "
                "because per-cloudlet forecast windows are truncated by deadline slack"
            )
        self._v32_forecast_mode = str(
            config.get("forecast_mode", "full")
        ).strip().lower()
        if self.obs_v32_job_forecast and self._v32_forecast_mode not in ("full", "none"):
            raise ValueError(
                "obs_v32_job_forecast=true supports forecast_mode 'full' or 'none'; "
                f"got {self._v32_forecast_mode!r}"
            )
        self._v32_forecast_bin_count = int(
            config.get("obs_v32_forecast_bin_count", 16)
        )
        if self.obs_v32_job_forecast and not 12 <= self._v32_forecast_bin_count <= 20:
            raise ValueError(
                "obs_v32_forecast_bin_count must be in [12, 20] for the "
                f"pre-registered V3.2 representation; got {self._v32_forecast_bin_count}"
            )
        self._v32_forecast_horizon_steps = int(
            config.get("obs_v32_forecast_horizon_steps", 120)
        )
        if self.obs_v32_job_forecast and (
            self._v32_forecast_horizon_steps < self._v32_forecast_bin_count
        ):
            raise ValueError(
                "obs_v32_forecast_horizon_steps must be >= "
                "obs_v32_forecast_bin_count"
            )
        if self._v32_forecast_bin_count > 0:
            self._v32_forecast_offsets_steps = np.rint(np.linspace(
                1,
                max(1, self._v32_forecast_horizon_steps),
                self._v32_forecast_bin_count,
            )).astype(np.int32)
        else:
            self._v32_forecast_offsets_steps = np.zeros(0, dtype=np.int32)
        self._v32_sim_timestep_sec = max(
            1e-6, float(config.get("simulation_timestep", 1.0))
        )
        self._v32_mi_per_kg = max(
            1e3, float(config.get("mi_per_kg_factor", 3.5e6))
        )
        self._v32_vm_mips = max(
            1.0,
            float(config.get(
                "obs_v32_vm_mips",
                min(
                    [
                        float(dc.get("vm_pe_mips", 50000.0))
                        for dc in self.dc_configs
                        if float(dc.get("vm_pe_mips", 50000.0)) > 0.0
                    ]
                    or [50000.0]
                ),
            )),
        )
        self._v32_deadline_margin_sec = max(
            0.0, float(config.get("obs_v32_deadline_margin_sec", 0.0))
        )
        self._v32_green_factors = np.asarray(
            [float(dc.get("green_carbon_factor", 0.01)) for dc in self.dc_configs],
            dtype=np.float64,
        )
        self._v32_brown_factors = np.asarray(
            [float(dc.get("brown_carbon_factor", 0.55)) for dc in self.dc_configs],
            dtype=np.float64,
        )
        theoretical_carbon_high = (
            float(config.get("obs_cloudlet_mi_high", 2000000))
            / self._v32_mi_per_kg
            * max(1e-9, float(np.max(np.maximum(
                self._v32_green_factors, self._v32_brown_factors
            ))))
        )
        self._v32_job_carbon_high = max(
            1e-9,
            float(config.get(
                "obs_v32_job_carbon_high", theoretical_carbon_high
            )),
        )
        self._last_v32_job_forecast_debug: Dict[str, Any] = {}
        # V3.2 demand model (Codex ruling 2026-08-17): 'legacy' keeps the
        # persistence-demand approximation byte-identical; 'job_counterfactual_v1'
        # prices each candidate as D_cf = D_current + dP(i,d) with
        # dP = 1[D_current~0]*P_idle + pes_i*(P_peak-P_idle)/N_hostPE, the SAME
        # D_cf on the now AND future sides. Root cause treated: idle_host_power_down
        # makes idle-DC demand 0, ratio=min(1,G/D) saturates, best_future ~ 0
        # always, and forecast_gain collapses to a content-blind ~0.9818 plateau
        # (= 1 - green_factor/brown_factor).
        self._v32_demand_model = str(
            config.get("obs_v32_demand_model", "legacy")).strip().lower()
        _HOST_POWER = {  # HostProfile.java: (idle W, dynamic W per PE)
            "rs500a": (51.36, (214.0 - 51.36) / 64.0),
            "rs700a": (106.21, (430.0 - 106.21) / 128.0),
        }
        idle_w, dyn_pp = [], []
        for dc in self.dc_configs:
            spec = None
            for key in dc:
                if key.startswith("host_count_spec") and int(dc.get(key) or 0) > 0:
                    spec = ("rs700a" if "rs700a" in key else
                            "rs500a" if "rs500a" in key else None)
            iw, dp = _HOST_POWER.get(spec, (51.36, 2.54))
            idle_w.append(iw)
            dyn_pp.append(dp)
        self._v32_host_idle_w = np.asarray(idle_w, dtype=np.float64)
        self._v32_host_dyn_w_per_pe = np.asarray(dyn_pp, dtype=np.float64)

        # Define observation and action spaces
        self._setup_observation_spaces()
        self._setup_action_spaces()

        # Optional: TimeCAP-based forecast provider replaces Java's God's Eye
        # (oracle/ground-truth) future trend features when green_oracle_mode == "timecap".
        # Skipped in spaces_only mode (training scripts that only need space shapes).
        # 'perturbed_godeye' (2026-09-03, DESIGN_PILOT): the true future pushed through the
        # frozen SERIES-space ladder in src/baselines/forecast_perturb, then reduced to the
        # same four features, so a policy can be TRAINED against a degraded forecast; until
        # now the ladder only reached the planner. Deliberately NOT this env's own
        # FORECAST_PERTURB_MODE, whose shuffle/anti perturb the AGGREGATED FEATURES (DC-axis
        # reversal / value mirror) rather than the series (time permutation / time
        # reversal). The names collide and the questions differ.
        self.green_oracle_mode = str(config.get("green_oracle_mode", "godeye")).lower()
        if self.green_oracle_mode not in ("godeye", "timecap", "perturbed_godeye"):
            raise ValueError(
                f"config['green_oracle_mode']={self.green_oracle_mode!r}; "
                "expected 'godeye', 'timecap' or 'perturbed_godeye'."
            )
        self.timecap_provider = None
        self._timecap_warmup_on_reset = False
        # Defer the TimeCAP provider construction (loads a 23.8M-param model
        # plus wind/solar CSVs, ~1-2s, holds the GIL) to the first reset() call.
        # Building eagerly inside __init__ blocks Ray's actor health-probe under
        # the new API stack, so the EnvRunner actor gets marked unhealthy and
        # silently drops every sample it later produces. See regression test
        # test_timecap_lazy_build.py for the contract.
        self._timecap_pending_build = (
            self.green_oracle_mode in ("timecap", "perturbed_godeye")
            and not self._spaces_only
        )

        # 2026-05-12 Level A: flat-protocol opt-in.  When True, env.step() makes
        # a single Py4J call (`result.getStepAsFlatMap()`) and parses everything
        # in-process instead of issuing ~200 individual getter RPCs.  Targets
        # 5-8× wall-clock reduction at the env-step layer.  Defaults to True so
        # new experiments get the perf win; flip to False for A/B testing or
        # if a Java-side schema change breaks the flat parser.
        self.use_flat_obs_protocol = bool(config.get("use_flat_obs_protocol", True))

        logger.info(f"HierarchicalMultiDCEnv initialised with {self.num_datacenters} datacenters")
        logger.info(f"  global_routing_batch_size: {self.global_routing_batch_size}")
        logger.info(
            f"  green_oracle_mode: {self.green_oracle_mode} "
            f"(provider={'pending (built on first reset)' if self._timecap_pending_build else 'off'})"
        )
        logger.info(f"  use_flat_obs_protocol: {self.use_flat_obs_protocol}")

    @staticmethod
    def _resolve_timecap_checkpoint(ckpt, *, search_dirs=None):
        """Resolve a ``timecap.checkpoint`` config value to an existing absolute file.

        Relative checkpoint paths in config are written relative to the
        drl-manager package dir (where ``timecap_prediction/`` lives), i.e. the
        documented run cwd. Resolving them here — instead of relying on the
        process cwd inside ``TimeCAP_GreenPredictor`` — keeps loading working
        inside Ray worker actors, whose cwd is the Ray session dir rather than
        drl-manager. Absolute paths are used as-is.

        ``search_dirs`` is injectable for testing; by default it tries the
        drl-manager dir, then the repo root, then the process cwd.
        """
        from pathlib import Path as _Path
        ckpt_path = _Path(ckpt)
        if ckpt_path.is_absolute():
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"timecap.checkpoint not found: {ckpt_path}")
            return ckpt_path

        if search_dirs is None:
            here = _Path(__file__).resolve()
            search_dirs = [here.parents[2], here.parents[3], _Path.cwd()]

        candidates = [_Path(d) / ckpt_path for d in search_dirs]
        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"timecap.checkpoint not found. Relative path {str(ckpt)!r} did not "
            "resolve against any known base. Tried: "
            + ", ".join(str(c) for c in candidates)
        )

    def _forecast_dc_turbine_map(self, csv_dir, csv_year):
        """{dc_id: [turbine_ids]}, {turbine_id: csv_path}, {dc_id: tz_offset}.

        Same rules the TimeCAP builder applies inline: skip DCs that are not green-enabled
        or declare no turbines, and fold the per-episode green-window shift into the per-DC
        tz offset so the provider reads the slice the simulator replays. The TimeCAP path
        is deliberately left untouched rather than refactored onto this helper; a test
        asserts the two produce identical maps, which catches drift without editing a
        working path.
        """
        from pathlib import Path as _P
        dc_assignments, turbine_csv_paths, dc_tz_offsets = {}, {}, {}
        for idx, dc_cfg in enumerate(self.dc_configs):
            if not dc_cfg.get("green_energy_enabled", False):
                continue
            tids = dc_cfg.get("turbine_ids") or []
            if not tids:
                continue
            dc_id = self.dc_ids[idx]
            dc_assignments[dc_id] = [int(t) for t in tids]
            dc_tz_offsets[dc_id] = int(dc_cfg.get("time_zone_offset_rows", 0)) \
                + int(getattr(self, "_green_episode_offset_rows", 0))
            for t in tids:
                t = int(t)
                csv_path = _P(csv_dir) / f"Turbine_{t}_{csv_year}.csv"
                if not csv_path.is_file():
                    raise FileNotFoundError(
                        f"Turbine CSV not found: {csv_path} "
                        f"(DC {dc_id}, turbine_id={t})")
                turbine_csv_paths[t] = str(csv_path)
        return dc_assignments, turbine_csv_paths, dc_tz_offsets

    def _build_perturbed_godeye_provider(self, config: Dict[str, Any]):
        """PerturbedGodEyeProvider over the same DC/turbine map, with no checkpoint."""
        import sys
        from pathlib import Path as _Path
        _src_dir = _Path(__file__).resolve().parents[2] / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        from prediction.perturbed_godeye_provider import PerturbedGodEyeProvider

        cfg = config.get("timecap") or {}
        csv_dir = _Path(cfg.get(
            "csv_dir", "cloudsimplus-gateway/src/main/resources/windProduction/split"))
        if not csv_dir.is_absolute():
            csv_dir = _Path(__file__).resolve().parents[3] / csv_dir
        csv_year = int(cfg.get("csv_year", int(config.get("wind_csv_year", 2021))))
        dc_assignments, turbine_csv_paths, dc_tz_offsets = \
            self._forecast_dc_turbine_map(csv_dir, csv_year)
        if not dc_assignments:
            logger.warning(
                "green_oracle_mode='perturbed_godeye' requested but no green-enabled DC "
                "declares turbine_ids; falling back to godeye (Java oracle) for this run.")
            return None

        eparams = None
        ep_path = config.get("perturb_error_params")
        if ep_path:
            import json as _json
            blob = _json.load(open(ep_path))
            eparams = blob.get("primary_error_params", blob)

        provider = PerturbedGodEyeProvider(
            dc_assignments=dc_assignments,
            turbine_csv_paths=turbine_csv_paths,
            perturb_tier=str(config.get("perturb_tier", "godeye")),
            pred_len=int(cfg.get("pred_len", 144)),
            short_term_steps=int(config.get("forecast_short_term_rows", 3)),
            long_term_steps=int(config.get("forecast_long_term_rows", 144)),
            error_params=eparams,
            csv_start_offset=int(cfg.get("csv_start_offset", 0)),
            dc_tz_offsets=dc_tz_offsets,
            simulation_warmup_rows=int(config.get("simulation_warmup_rows", 0)),
        )
        # Nothing to warm up: the truth is read directly, there is no history buffer.
        self._timecap_warmup_on_reset = False
        logger.info("perturbed_godeye ready: tier=%s | dcs=%s | turbines=%s",
                    provider.perturb_tier, sorted(dc_assignments),
                    sorted(turbine_csv_paths))
        return provider

    def _build_timecap_provider(self, config: Dict[str, Any]):
        """
        Construct a TimeCAPGodEyeProvider from this env's dc_configs and the
        ``timecap`` block of ``config``. Returns None (and logs a warning) if
        no green-enabled DCs declare any turbine_ids.

        Expected config layout:
            green_oracle_mode: timecap
            timecap:
              checkpoint: <path>           # required
              forecast_every: 6            # 1=every step (GPU); 6=every sim hour (CPU)
              device: cpu                  # or "cuda"
              csv_dir: cloudsimplus-gateway/src/main/resources/windProduction/split
              csv_year: 2021
              csv_start_offset: 0          # CSV row at sim_step=0 (Java's tz_offset_rows
                                           # is per-DC; this is one-size-fits-all for now)
              warmup_on_reset: false
        """
        # Lazy import — avoids loading torch in spaces_only / godeye mode
        import sys
        from pathlib import Path as _Path
        _src_dir = _Path(__file__).resolve().parents[2] / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        if self.green_oracle_mode == "perturbed_godeye":
            return self._build_perturbed_godeye_provider(config)

        from prediction.timecap_godeye_provider import TimeCAPGodEyeProvider

        tc_cfg = config.get("timecap") or {}
        # The forecast history and the wind the simulator serves must come from the same
        # year. They were separate keys with separate defaults, so a 2020 evaluation would
        # silently have been fed 2021 predictions: the forecast would have been of a
        # different year's weather entirely, and nothing would have said so.
        _tc_year = int(tc_cfg.get("csv_year", 2021))
        _sim_year = int(config.get("wind_csv_year", 2021))
        if _tc_year != _sim_year:
            raise ValueError(
                f"timecap.csv_year={_tc_year} but wind_csv_year={_sim_year}. The forecast "
                f"would be built from a different year than the wind the simulator serves. "
                f"Set both to the same year.")
        ckpt = tc_cfg.get("checkpoint")
        if not ckpt:
            raise ValueError(
                "green_oracle_mode='timecap' requires config['timecap']['checkpoint'] to be set."
            )
        # Resolve to an absolute, existing path so it loads inside Ray worker
        # actors (whose cwd is the Ray session dir, not drl-manager).
        ckpt = self._resolve_timecap_checkpoint(ckpt)

        # Resolve csv_dir relative to repo root if a relative path was given
        csv_dir = _Path(tc_cfg.get(
            "csv_dir",
            "cloudsimplus-gateway/src/main/resources/windProduction/split",
        ))
        if not csv_dir.is_absolute():
            repo_root = _Path(__file__).resolve().parents[3]
            csv_dir = repo_root / csv_dir
        if not csv_dir.is_dir():
            raise FileNotFoundError(
                f"timecap.csv_dir does not exist: {csv_dir} "
                "(needs the 13-feature SDWPF split CSVs, NOT the simplified 2-col files)"
            )

        csv_year = int(tc_cfg.get("csv_year", 2021))

        # Build {dc_id: [turbine_ids]}, {turbine_id: csv_path}, {dc_id: tz_offset}
        # from dc_configs. Skip DCs that aren't green-enabled or have no turbines.
        dc_assignments: Dict[int, list] = {}
        turbine_csv_paths: Dict[int, str] = {}
        dc_tz_offsets: Dict[int, int] = {}
        for idx, dc_cfg in enumerate(self.dc_configs):
            if not dc_cfg.get("green_energy_enabled", False):
                continue
            tids = dc_cfg.get("turbine_ids") or []
            if not tids:
                continue
            dc_id = self.dc_ids[idx]
            dc_assignments[dc_id] = [int(t) for t in tids]
            # Closed-book support: mirror the Java per-episode green-window shift
            # ((1009*k) mod green_episode_offset_range, k = reset count). Java adds
            # the episode offset inside its tz-offset conversion; adding it to the
            # provider's per-DC tz offsets reproduces the identical row mapping, so
            # the TimeCAP history buffers read the SAME wind slice the simulator
            # replays. Without this the provider forecasts a different day and the
            # timecap arm silently measures garbage (the npy-desync bug class).
            dc_tz_offsets[dc_id] = int(dc_cfg.get("time_zone_offset_rows", 0)) \
                + int(getattr(self, "_green_episode_offset_rows", 0))
            for t in tids:
                t = int(t)
                csv_path = csv_dir / f"Turbine_{t}_{csv_year}.csv"
                if not csv_path.is_file():
                    raise FileNotFoundError(
                        f"Turbine CSV not found: {csv_path} "
                        f"(DC {dc_id}, turbine_id={t})"
                    )
                turbine_csv_paths[t] = str(csv_path)

        if not dc_assignments:
            logger.warning(
                "green_oracle_mode='timecap' requested but no green-enabled DCs declare "
                "turbine_ids; falling back to godeye (Java oracle) for this run."
            )
            return None

        # Pull simulation_warmup_rows from the top-level env config (matching Java's
        # plumbing in HierarchicalMultiDCGateway, which propagates this same key from
        # top-level into each per-DC params dict).
        sim_warmup_rows = int(config.get("simulation_warmup_rows", 0))

        provider = TimeCAPGodEyeProvider(
            dc_assignments        = dc_assignments,
            turbine_csv_paths     = turbine_csv_paths,
            checkpoint_path       = str(ckpt),
            forecast_every        = int(tc_cfg.get("forecast_every", 6)),
            device                = str(tc_cfg.get("device", "cpu")),
            csv_start_offset      = int(tc_cfg.get("csv_start_offset", 0)),
            dc_tz_offsets         = dc_tz_offsets,
            simulation_warmup_rows= sim_warmup_rows,
            forecast_shift        = tc_cfg.get("forecast_shift"),
        )
        self._timecap_warmup_on_reset = bool(tc_cfg.get("warmup_on_reset", False))
        logger.info(
            "TimeCAP green-oracle ready: ckpt=%s | "
            "forecast_every=%d | device=%s | dcs=%s | turbines=%s | "
            "warmup_rows=%d | dc_tz_offsets=%s",
            ckpt,
            int(tc_cfg.get("forecast_every", 6)),
            tc_cfg.get("device", "cpu"),
            sorted(dc_assignments.keys()),
            sorted(turbine_csv_paths.keys()),
            sim_warmup_rows,
            dc_tz_offsets,
        )
        return provider

    def _find_free_port(self) -> int:
        """Find a free TCP port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            return port

    def _launch_java_gateway(self, port: int):
        """
        Launch a dedicated Java CloudSim Plus Gateway process on the specified port.
        """
        # Locate the gradlew script
        # Assuming we are running from the project root or drl-manager
        # Try to find cloudsimplus-gateway directory
        
        possible_roots = [
            os.getcwd(),
            os.path.join(os.getcwd(), ".."),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        ]
        
        gateway_dir = None
        for root in possible_roots:
            candidate = os.path.join(root, "cloudsimplus-gateway")
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "gradlew")):
                gateway_dir = candidate
                break
        
        if not gateway_dir:
            raise RuntimeError("Could not find cloudsimplus-gateway directory with gradlew script.")

        gradlew_path = os.path.join(gateway_dir, "gradlew")
        
        # Prepare command
        # Use --no-daemon to avoid lingering Gradle daemons for each worker
        # Use -q to reduce noise
        # HPC/offline mode (env var GATEWAY_LIBS = dir of pre-installed jars from
        # `gradlew installDist`): launch the JVM directly, bypassing gradle. Compute
        # nodes typically have no internet and `gradlew run` either hangs resolving
        # deps or serializes 8 cold builds past the walltime. java -cp lib/* is fast,
        # offline, and free of gradle-cache contention. Unset -> original gradlew path.
        gateway_libs = os.environ.get("GATEWAY_LIBS")
        if gateway_libs:
            java_home = os.environ.get("JAVA_HOME")
            java_bin = os.path.join(java_home, "bin", "java") if java_home else "java"
            cmd = [java_bin]
            # `java -cp lib/*` scans every jar for logback.xml and the first hit
            # wins; cloudsimplus-8.5.5.jar sorts before cloudsimplus-gateway.jar,
            # so the library's DEBUG-level config silences ours (root=ERROR) and
            # floods ~1GB/hr of DEBUG per worker. Force our config explicitly.
            logback_cfg = os.path.abspath(os.path.join(
                gateway_libs, os.pardir, os.pardir, os.pardir, os.pardir,
                "src", "main", "resources", "logback.xml"))
            if os.path.isfile(logback_cfg):
                cmd.append("-Dlogback.configurationFile=" + logback_cfg)
            cmd += [
                "-cp", os.path.join(gateway_libs, "*"),
                "exe.edu.cspg.MainMultiDC",
                "--port", str(port),
            ]
        else:
            cmd = [
                gradlew_path,
                "--no-daemon",
                "-PappMainClass=exe.edu.cspg.MainMultiDC",
                "run",
                "-q",
                f"--args=--port {port}",
            ]

            # When running concurrent gateways across SLURM array tasks that share
            # the repo on Lustre, the project-local <gateway>/.gradle cache lock
            # (fileHashes.lock etc.) becomes a cross-node contention point and every
            # task but one times out. Setting GRADLE_PROJECT_CACHE_DIR (typically to
            # node-local $TMPDIR) redirects the project cache so each task gets its
            # own isolated state.
            project_cache_dir = os.environ.get("GRADLE_PROJECT_CACHE_DIR")
            if project_cache_dir:
                os.makedirs(project_cache_dir, exist_ok=True)
                cmd.insert(1, "--project-cache-dir")
                cmd.insert(2, project_cache_dir)
        
        logger.info(f"Launching Java Gateway on port {port}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        # region agent log
        _write_debug_log(
            "H2",
            "hierarchical_multidc_env.py:177",
            "launch_java_gateway",
            {
                "port": port,
                "cwd": os.getcwd(),
                "gateway_dir": gateway_dir,
                "cmd": cmd,
            },
        )
        # endregion
        
        # Launch process
        # Redirect stdout/stderr to a log file for debugging
        log_dir = self.config["gateway_log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"gateway_{port}.log")

        # Serialize gradlew run across Ray workers: concurrent builds in the same
        # cloudsimplus-gateway directory can corrupt build output and cause
        # NoClassDefFoundError (e.g. HierarchicalResetResult) at runtime.
        lock_file = None
        try:
            if sys.platform != "win32":
                lock_path = os.path.join(gateway_dir, ".py4j_gateway_launch.lock")
                lock_file = open(lock_path, "a+", encoding="utf-8")
                logger.info(
                    "Waiting for exclusive gateway build lock (multi-worker safe)..."
                )
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            self._java_log_file = open(log_file_path, "w")

            try:
                self.java_process = subprocess.Popen(
                    cmd,
                    cwd=gateway_dir,
                    stdout=self._java_log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group for easier cleanup
                )

                # Wait for the server to be ready
                # We'll poll the port until it's open
                max_retries = 600  # Gradle cold build + JVM can exceed 60s
                for i in range(max_retries):
                    if self.java_process.poll() is not None:
                        # region agent log
                        _write_debug_log(
                            "H1",
                            "hierarchical_multidc_env.py:201",
                            "java_process_exited_early",
                            {
                                "port": port,
                                "returncode": self.java_process.returncode,
                                "log_file_path": log_file_path,
                            },
                        )
                        # endregion
                        raise RuntimeError(
                            f"Java process exited prematurely with code {self.java_process.returncode}. "
                            f"Check logs at {log_file_path}"
                        )

                    if self._is_port_open(port):
                        logger.info(f"Java Gateway is ready on port {port}")
                        return

                    time.sleep(1.0)
                    if i % 30 == 0 and i > 0:
                        logger.info(
                            f"Waiting for Java Gateway on port {port} ({i}/{max_retries}s)..."
                        )

                raise RuntimeError(f"Timed out waiting for Java Gateway on port {port}")

            except Exception as e:
                logger.error(f"Failed to launch Java Gateway: {e}")
                if self.java_process:
                    self.java_process.kill()
                if self._java_log_file:
                    self._java_log_file.close()
                raise
        finally:
            if lock_file is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_file.close()

    def _is_port_open(self, port: int) -> bool:
        """Check if a TCP port is open and listening."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, socket.timeout):
                return False

    def _setup_observation_spaces(self):
        """
        Define observation spaces for global and local agents.
        """
        # Global observation space (aggregated DC-level metrics)
        global_spaces = {
            # Green energy metrics (W - Watts)
            # NOTE (2026-06-23): high was 5e6 W (5 MW) but actual green/demand are
            # ~tens–thousands of W in these regimes, so the input-scale normalization
            # (value/high) squashed them to ~1e-5 ≈ 0 — the policy was BLIND to which
            # DC is green and could never learn green-aware routing. Lowered to the
            # real scale (turbine-peak / DC-capacity) so the signal is visible.
            "dc_current_green_power_w": spaces.Box(
                low=0.0, high=float(self.config.get("obs_green_power_high", 3000.0)),
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_current_power_w": spaces.Box(
                low=0.0, high=float(self.config.get("obs_power_high", 5000.0)),
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_green_ratio": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_cumulative_wasted_green_wh": spaces.Box(
                low=0.0, high=1e6,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            # Future energy trend features (God's Eye mode)
            "dc_future_short_mean": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_short_trend": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_long_mean": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_future_long_peak_timing": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_queue_sizes": spaces.Box(
                low=0, high=10000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (queue sizes are integers)
            ),
            "dc_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_available_pes": spaces.Box(
                low=0, high=1000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (PEs are integers)
            ),
            "dc_ram_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "upcoming_cloudlets_count": spaces.Box(
                low=0, high=100000,
                shape=(1,),
                dtype=np.int32
            ),
            "batch_cloudlet_pes": spaces.Box(
                low=0, high=100,  # Max PEs for a cloudlet
                shape=(self.global_routing_batch_size,),
                dtype=np.int32
            ),
            "batch_cloudlet_mi": spaces.Box(
                low=0, high=int(self.config.get("obs_cloudlet_mi_high", 2000000)),  # Max MI/cloudlet (configurable for long-job regimes)
                shape=(self.global_routing_batch_size,),
                dtype=np.int64
            ),
            "upcoming_pes_distribution": spaces.Box(
                low=0, high=1000,
                shape=(3,),  # [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]
                dtype=np.int32
            ),
            "load_imbalance": spaces.Box(
                low=0.0, high=10.0,
                shape=(1,),
                dtype=np.float32
            ),
            "recent_completed": spaces.Box(
                low=0, high=100000,
                shape=(1,),
                dtype=np.int32
            ),
        }

        if self.obs_v31_features:
            batch_shape = (self.global_routing_batch_size,)
            global_spaces.update({
                # All continuous values below are normalized in the converter
                # and explicitly clipped to these declared bounds.
                "batch_cloudlet_wait_age": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_time_to_deadline": spaces.Box(
                    low=-1.0, high=4.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_deadline_present": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_is_deferred": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_defer_count": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "global_deferred_count": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "global_deferred_mi": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            })

        if self.obs_v31_features and self.defer_deadline_mask:
            # Stage D' deadline-safe DEFER mask (STAGE_D_PRIME_DESIGN §3): 1 = the slot may
            # still wait one more step and finish by its deadline, 0 = it must be routed now.
            global_spaces["batch_cloudlet_defer_allowed"] = spaces.Box(
                low=0.0, high=1.0, shape=(self.global_routing_batch_size,), dtype=np.float32)

        if self.obs_v32_job_forecast:
            batch_shape = (self.global_routing_batch_size,)
            global_spaces.update({
                # All four are clipped explicitly by
                # _append_v32_job_forecast_features. best_* are normalized by
                # the configured/theoretical per-cloudlet carbon upper bound.
                "batch_cloudlet_forecast_gain": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_time_to_best_green": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_best_now_carbon": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
                "batch_cloudlet_best_future_carbon": spaces.Box(
                    low=0.0, high=1.0, shape=batch_shape, dtype=np.float32),
            })

        self.global_observation_space = spaces.Dict(global_spaces)

        # Local observation spaces (per datacenter)
        # Track per-DC sizes but expose a shared max-sized space for SB3 compatibility
        dc_defaults = {
            "hosts_count": 16,
            "initial_s_vm_count": 10,
            "initial_m_vm_count": 5,
            "initial_l_vm_count": 3,
        }
        dc_configs = self.dc_configs or [dc_defaults.copy()]

        self.dc_host_counts: List[int] = [
            int(dc.get("hosts_count", dc_defaults["hosts_count"])) for dc in dc_configs
        ]
        self.max_hosts = max(self.dc_host_counts) if self.dc_host_counts else dc_defaults["hosts_count"]

        self.dc_vm_counts: List[int] = [
            int(
                dc.get("initial_s_vm_count", dc_defaults["initial_s_vm_count"]) +
                dc.get("initial_m_vm_count", dc_defaults["initial_m_vm_count"]) +
                dc.get("initial_l_vm_count", dc_defaults["initial_l_vm_count"])
            )
            for dc in dc_configs
        ]
        self.max_vms = max(self.dc_vm_counts) if self.dc_vm_counts else (
            dc_defaults["initial_s_vm_count"] +
            dc_defaults["initial_m_vm_count"] +
            dc_defaults["initial_l_vm_count"]
        )

        self.local_observation_space = spaces.Dict({
            "host_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "host_ram_usage": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "vm_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_vms,),
                dtype=np.float32
            ),
            "vm_types": spaces.Box(
                low=0, high=3,  # 0=Off, 1=Small, 2=Medium, 3=Large
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "vm_available_pes": spaces.Box(
                low=0, high=100,
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "waiting_cloudlets": spaces.Box(
                low=0, high=100000,
                shape=(1,),
                dtype=np.int32
            ),
            "next_cloudlet_pes": spaces.Box(
                low=0, high=256,
                shape=(1,),
                dtype=np.int32
            ),
        })

        # Architecture A only (local temporal lever): the local agent sees green-now
        # + forecast so it can decide hold-vs-run. Gated by reward_local_carbon_enabled
        # — in architecture B (global defer, local = pure QoS) the local agent does NOT
        # do carbon timing, so it gets NO forecast features (clean QoS obs).
        if bool(self.config.get("reward_local_carbon_enabled", False)):
            self.local_observation_space = spaces.Dict({
                **self.local_observation_space.spaces,
                "green_now": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "green_forecast_short": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "green_forecast_long": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            })

    def _setup_action_spaces(self):
        """
        Define action spaces for global and local agents.
        
        Global Agent: Routes a fixed batch of cloudlets per step.
        - Each action is a datacenter index in [0, num_datacenters - 1]
        - If fewer cloudlets are available than the routing batch size,
          extra actions are simply ignored (trimmed to queue length).
        
        Local Agents: Assign one cloudlet per DC per step.
        - Local action keys are dc_index (0..N-1)
        """
        # Global action space: fixed-size batch of routing decisions.
        # Each element is a DC index ∈ {0, ..., num_datacenters-1}.
        # Architecture B (2026-06-20, gated by global_defer_enabled): add a DEFER
        # option (index = num_datacenters) so the global agent can HOLD a cloudlet
        # for a greener moment instead of routing it now — the temporal lever at the
        # global level. Legacy routing (no defer) is unchanged when the flag is off.
        self.global_defer_enabled = bool(self.config.get("global_defer_enabled", False))
        n_global_choices = self.num_datacenters + (1 if self.global_defer_enabled else 0)
        self.global_action_space = spaces.MultiDiscrete(
            [n_global_choices] * self.global_routing_batch_size
        )

        # Local action spaces: Select VM for each datacenter's next cloudlet
        # Each datacenter has its own action space
        # Action space includes: 0 = NoAssign, 1 to max_vms = VM indices
        max_vms = getattr(self, "max_vms", 1)
        # NEW local agent (2026-06-18, gated): dispatch_rate mode replaces the
        # per-VM placement action with "how many cloudlets to release this step"
        # (0 = hold all = defer; N = burst-drain). Placement → sim best-fit.
        # See docs/Deferrable_Jobs_Lever.md. Default = legacy vm_placement.
        self.local_dispatch_mode = str(
            self.config.get("local_dispatch_mode", "vm_placement")).strip()
        if self.local_dispatch_mode == "dispatch_rate":
            self.max_dispatch_per_step = int(self.config.get("max_dispatch_per_step", 20))
            self.local_action_space = spaces.Discrete(self.max_dispatch_per_step + 1)
        else:
            self.local_action_space = spaces.Discrete(max_vms + 1)  # +1 for NoAssign option

        # Gymnasium requires self.action_space and self.observation_space
        # Combine global and local spaces into a Dict space
        self.action_space = spaces.Dict({
            "global": self.global_action_space,
            "local": spaces.Dict({
                i: self.local_action_space for i in range(self.num_datacenters)
            })
        })

        self.observation_space = spaces.Dict({
            "global": self.global_observation_space,
            "local": spaces.Dict({
                i: self.local_observation_space for i in range(self.num_datacenters)
            })
        })

    def _connect_to_java(self):
        """
        Establish Py4J connection to Java gateway with retry mechanism.

        Retries connection up to max_retries times with exponential backoff.
        If connection fails after all retries, raises RuntimeError.
        """
        if self.gateway is None:
            max_retries = self.config.get("gateway_max_retries", 5)
            retry_delay = self.config.get("gateway_retry_delay", 5.0)

            logger.info(f"Attempting to connect to Java gateway on port {self.py4j_port}...")

            retries = max_retries
            while retries > 0:
                try:
                    # Attempt connection
                    self.gateway = JavaGateway(
                        gateway_parameters=GatewayParameters(port=self.py4j_port, auto_convert=True)
                    )

                    # Test connection by calling a simple Java method
                    self.gateway.jvm.System.out.println(
                        f"Python HierarchicalMultiDCEnv connected on port {self.py4j_port}!"
                    )

                    self.java_env = self.gateway.entry_point
                    logger.info(f"Successfully connected to Java gateway on port {self.py4j_port}")

                    # Successfully connected, exit retry loop
                    break

                except (ConnectionRefusedError, Py4JNetworkError) as e:
                    retries -= 1
                    if retries > 0:
                        logger.warning(
                            f"Gateway connection failed: {e}. "
                            f"Retrying in {retry_delay} seconds... ({retries} retries left)"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries reached. Could not connect to Java gateway.")
                        raise RuntimeError(
                            f"Could not connect to Java gateway on port {self.py4j_port} "
                            f"after {max_retries} attempts. "
                            f"Make sure the Java gateway server is running:\n"
                            f"  cd cloudsimplus-gateway && ./gradlew run"
                        ) from e

                except Exception as e:
                    # Unexpected error, don't retry
                    logger.error(f"Unexpected error connecting to Java gateway: {e}")
                    raise RuntimeError(
                        f"Unexpected error connecting to Java gateway: {e}"
                    ) from e

            # Configure simulation after successful connection
            try:
                logger.info("Configuring multi-datacenter simulation...")
                self.java_env.configureSimulation(self.config)
                logger.info("Multi-datacenter simulation configured successfully")

            except Exception as e:
                logger.error(f"Failed to configure simulation: {e}")
                # Clean up gateway connection on configuration failure
                self._cleanup_gateway()
                raise RuntimeError(
                    f"Failed to configure multi-datacenter simulation. "
                    f"Check Java logs for details."
                ) from e

    def _cleanup_gateway(self):
        """
        Clean up gateway connection resources.
        """
        if self.gateway is not None:
            try:
                self.gateway.close()
                logger.info("Java gateway connection closed")
            except Exception as e:
                logger.warning(f"Error closing gateway: {e}")
            finally:
                self.gateway = None
                self.java_env = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (not used currently)

        Returns:
            observations: Dict with 'global' and 'local' observations
            info: Additional information

        Raises:
            RuntimeError: If connection to Java gateway fails or reset fails
        """
        super().reset(seed=seed)

        # Targeted perturbation curriculum: FORECAST_PERTURB_PROB in (0,1] makes the
        # perturbation an EPISODE-LEVEL lottery instead of always-on, so training can
        # mix clean and corrupted-forecast episodes (kn2 curriculum). Unset/<=0 keeps
        # the historical semantics (MODE applies to every episode).
        self._perturb_this_episode = self._draw_perturb_episode()

        # Connect to Java if not already connected (with retry mechanism)
        self._connect_to_java()

        # Reset Java simulation
        try:
            logger.debug(f"Resetting Java simulation with seed {seed}...")
            result = self.java_env.reset(seed if seed is not None else 0)
        except Exception as e:
            logger.error(f"Failed to reset Java simulation: {e}")
            raise RuntimeError(
                f"Failed to reset multi-datacenter simulation. "
                f"Check Java logs for details."
            ) from e

        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Closed-book green windows: keep the Python-side episode counter in
        # lockstep with Java's (env.reset() -> resetSimulation() is 1:1) and
        # apply the same deterministic schedule (1009*k mod range).
        self._episode_counter = getattr(self, "_episode_counter", -1) + 1
        _off_range = int(self.config.get("green_episode_offset_range", 0) or 0)
        _new_off = episode_offset_rows(self._episode_counter, _off_range,
                                       self.config.get("green_episode_offset_allowlist"))

        # Lazy-build the TimeCAP provider on the first reset — keeps __init__
        # cheap so Ray's EnvRunner actor registers before its first health probe.
        # Under closed-book windows the provider is REBUILT whenever the episode
        # offset changes: its per-DC tz offsets bake in the window shift, keeping
        # its CSV row mapping identical to the simulator's (see
        # _build_timecap_provider). Rebuild cost is one checkpoint load.
        if self._timecap_pending_build or (
            self.timecap_provider is not None
            and _new_off != getattr(self, "_green_episode_offset_rows", 0)
        ):
            self._green_episode_offset_rows = _new_off
            self.timecap_provider = self._build_timecap_provider(self.config)
            self._timecap_pending_build = False
        else:
            self._green_episode_offset_rows = _new_off

        # Reset TimeCAP rolling buffers BEFORE we parse the first observation
        # (because _convert_global_observation will push the first row).
        if self.timecap_provider is not None:
            self.timecap_provider.reset()
            if self._timecap_warmup_on_reset:
                # Pushes seq_len real CSV rows starting at sim_step=0 — eliminates
                # the cold-start zero-pad period at the cost of "looking 16h ahead"
                # at episode start. Off by default; enable only for evaluation runs.
                self.timecap_provider.warmup(start_step=0)

        # Parse observations from HierarchicalResetResult
        try:
            # Reset returns HierarchicalResetResult (only observations and info)
            observations = self._parse_hierarchical_observation_from_reset(result)
            self._last_global_obs_for_crd = observations.get("global", {})
            self._mask_route_count = 0
            info = self._parse_info_from_reset(result)
        except Exception as e:
            logger.error(f"Failed to parse reset result: {e}")
            raise RuntimeError(
                f"Failed to parse observations from Java. "
                f"Check observation structure compatibility."
            ) from e

        # Store observations for action masking
        self.last_observations = observations

        logger.info(f"Environment reset successfully for episode (seed={seed})")
        return observations, info

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one hierarchical step in the environment.

        Args:
            action: Dictionary containing:
                - 'global': List of datacenter indices for arriving cloudlets
                - 'local': Dict mapping dc_index -> vm_id

        Returns:
            observations: Dict with 'global' and 'local' observations
            rewards: Dict with 'global' and 'local' rewards
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information

        Raises:
            RuntimeError: If environment not initialized or step execution fails
            ValueError: If action format is invalid
        """
        if self.java_env is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first before calling step()."
            )

        # Validate and extract actions
        try:
            global_actions = action.get("global", [])
            local_actions_map = action.get("local", {})

            if not isinstance(global_actions, (list, np.ndarray)):
                raise ValueError(
                    f"'global' actions must be a list or array, got {type(global_actions)}"
                )
            if not isinstance(local_actions_map, dict):
                raise ValueError(
                    f"'local' actions must be a dict, got {type(local_actions_map)}"
                )

            # Explicit dcIndex usage for local actions (0..N-1).
            # If you have dcId keys, convert them to indices before calling step().
            for raw_key in local_actions_map.keys():
                try:
                    key_int = int(raw_key)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Local action key '{raw_key}' is not an integer dcIndex."
                    )
                if key_int < 0 or key_int >= self.num_datacenters:
                    raise ValueError(
                        f"Local action key '{raw_key}' is out of dcIndex range [0, {self.num_datacenters - 1}]."
                    )
        except Exception as e:
            logger.error(f"Invalid action format: {e}")
            raise ValueError(f"Invalid action format. Expected dict with 'global' and 'local' keys.") from e

        # Process global actions:
        # - Each element is a datacenter index in [0, num_datacenters - 1]
        # - Actions are one-to-one mapped to DC indices; there is no explicit NoAssign.
        # - If there are more actions than available cloudlets, extra actions are ignored.
        # Convert actions to DC indices and clamp out-of-range values
        # Architecture B: when global_defer_enabled, index == num_datacenters is the
        # DEFER action (valid) and must pass through to Java unchanged. Without this,
        # the old clamp turned every defer into "route to the last DC", silently
        # killing the temporal lever.
        max_valid = self.num_datacenters + (1 if getattr(self, "global_defer_enabled", False) else 0) - 1
        global_actions_filtered = []
        for i, action_val in enumerate(global_actions):
            action_int = int(action_val)
            if action_int < 0:
                logger.warning(f"Global action[{i}] = {action_int} < 0, clamping to 0")
                dc_index = 0
            elif action_int > max_valid:
                logger.warning(
                    f"Global action[{i}] = {action_int} > {max_valid}, clamping to {max_valid}"
                )
                dc_index = max_valid
            else:
                dc_index = action_int  # includes the defer index (== num_datacenters) when enabled
            global_actions_filtered.append(dc_index)
        
        global_actions = global_actions_filtered

        # Stage D' deadline-safe DEFER mask, shared by every algorithm: a heuristic arm
        # (or an adversarial always-defer policy) cannot be masked at the logit level, so
        # a DEFER on a slot whose defer_allowed is 0 is routed here by one fixed rule
        # (greenest DC with room), before Java's backstop ever has to fire. The RL module
        # never emits such a DEFER (its DEFER logit is masked), so for it this is a no-op.
        if getattr(self, "defer_deadline_mask", False):
            g = self._last_global_obs_for_crd or {}
            da = g.get("batch_cloudlet_defer_allowed")
            if da is not None:
                global_actions, n_routed = route_disallowed_defers(
                    global_actions, da, g.get("batch_cloudlet_pes"), g.get("dc_green_ratio"),
                    g.get("dc_available_pes"), self.num_datacenters)
                self._mask_route_count += int(n_routed)

        # Convert local actions dict to Java-compatible format
        # Apply action mapping: agent outputs 0 to num_vms -> Java expects -1 to num_vms-1
        # - action=0 → targetVmId=-1 (NoAssign)
        # - action=1 → targetVmId=0 (VM 0)
        # - action=n → targetVmId=n-1 (VM n-1)
        try:
            # Ensure every DC has an explicit local action; default to NoAssign (0)
            local_actions_java = {}
            for dc_index in range(self.num_datacenters):
                dc_id = self.dc_index_to_id.get(dc_index, dc_index)
                agent_action = local_actions_map.get(dc_index, 0)
                # dispatch_rate mode: action IS the raw release count (Java reads
                # it as a dispatch count, not a VM id) — no -1 offset.
                if getattr(self, "local_dispatch_mode", "vm_placement") == "dispatch_rate":
                    target_vm_id = int(agent_action)
                else:
                    # legacy vm_placement: action=n → targetVmId=n-1 (0→-1 NoAssign)
                    target_vm_id = int(agent_action) - 1  # 0→-1, 1→0, 2→1, ...
                local_actions_java[int(dc_id)] = target_vm_id
                logger.debug(
                    "DC index %d (dcId=%s): agent_action=%s → targetVmId=%d",
                    dc_index, dc_id, agent_action, target_vm_id
                )
        except Exception as e:
            logger.error(f"Failed to convert local actions: {e}")
            raise ValueError(f"Invalid local action format. DC IDs and VM IDs must be integers.") from e

        # Convert numpy types to Python native types for Py4J compatibility
        # Py4J cannot serialize numpy.int64, numpy.ndarray, etc.
        if isinstance(global_actions, np.ndarray):
            global_actions = global_actions.tolist()
        global_actions_python = [int(x) for x in global_actions]  # Ensure all elements are Python int
        local_actions_python = {int(k): int(v) for k, v in local_actions_java.items()}

        # Execute step in Java simulation
        try:
            # Per-step heartbeat at INFO.  This used to be at INFO, was demoted
            # to DEBUG for perf reasons, then restored on user request — the
            # actual cost is ~70-150 µs / step (~0.05% of total wall-clock),
            # not 5-10% as I had initially estimated.  Without a per-step log
            # the driver appears frozen for the entire ~5 min episode because
            # all other callbacks fire on episode end.  Keep it at INFO.
            logger.info(f"[STEP {self.current_step + 1}] Calling Java with global_actions={global_actions_python}, local_actions={local_actions_python}")
            result = self.java_env.step(global_actions_python, local_actions_python)
            logger.debug("Java step returned successfully")
        except Exception as e:
            logger.error(f"Failed to execute step in Java simulation: {e}")
            logger.debug("Java step FAILED: %s", e)
            raise RuntimeError(
                f"Failed to execute simulation step. Check Java logs for details."
            ) from e

        # Parse results.  Fast path (Level A): one Py4J call returns the
        # entire step result as a flat Map, ~5-8× cheaper than the legacy
        # 200-getter dance.  Falls back to the legacy path on any error
        # (e.g. Java side hasn't been rebuilt) so the run can still proceed.
        try:
            if self.use_flat_obs_protocol:
                try:
                    flat = result.getStepAsFlatMap()
                except Exception as flat_err:
                    logger.warning(
                        "getStepAsFlatMap unavailable (%s); reverting to legacy parser "
                        "for THIS run.  Rebuild the gateway jar to use the fast path.",
                        flat_err,
                    )
                    self.use_flat_obs_protocol = False
                    flat = None
                if flat is not None:
                    observations = self._parse_observation_from_flat(flat)
                    rewards      = self._parse_rewards_from_flat(flat)
                    terminated   = bool(flat["meta.terminated"])
                    truncated    = bool(flat["meta.truncated"])
                    self._last_global_obs_for_crd = observations.get("global", {})
                    info = self._parse_info_from_flat(flat)
                else:
                    observations = self._parse_hierarchical_observation(result)
                    rewards      = self._parse_hierarchical_rewards(result)
                    terminated   = result.isTerminated()
                    truncated    = result.isTruncated()
                    self._last_global_obs_for_crd = observations.get("global", {})
                    info = self._parse_info(result)
            else:
                observations = self._parse_hierarchical_observation(result)
                rewards      = self._parse_hierarchical_rewards(result)
                terminated   = result.isTerminated()
                truncated    = result.isTruncated()
                # Cache the global obs so _collect_crd_info can read decision-time
                # signals (queue sizes, green ratio) for the CRD baseline scheduler
                # without making extra Py4J round-trips.
                self._last_global_obs_for_crd = observations.get("global", {})
                info = self._parse_info(result)
        except Exception as e:
            logger.error(f"Failed to parse step result: {e}")
            raise RuntimeError(
                f"Failed to parse step results from Java. "
                f"Check observation/reward structure compatibility."
            ) from e

        # Update episode state
        self.current_step += 1
        all_rewards = rewards["global"] + sum(rewards["local"].values())
        self.episode_reward += all_rewards
        self.done = terminated or truncated

        # Store observations for action masking
        self.last_observations = observations

        logger.debug(
            f"Step {self.current_step}: Global reward={rewards['global']:.3f}, "
            f"Terminated={terminated}, Truncated={truncated}"
        )

        return observations, rewards, terminated, truncated, info

    def _parse_hierarchical_observation_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalResetResult into observation dict.
        This is specifically for reset() which returns HierarchicalResetResult.
        """
        # Parse global observation (GlobalObservationState)
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations (Map<Integer, ObservationState>)
        local_obs_java = result.getLocalObservations()
        local_obs = {}
        for dc_id in local_obs_java:
            try:
                obs_state = (
                    local_obs_java.get(dc_id)
                    if hasattr(local_obs_java, "get")
                    else local_obs_java[dc_id]
                )
            except Exception:
                obs_state = None
            if obs_state is not None:
                dc_id_int = int(dc_id)
                dc_index = self.dc_id_to_index.get(dc_id_int)
                if dc_index is None:
                    logger.warning("Unknown datacenter_id in reset observations: %s", dc_id_int)
                    continue
                local_obs[dc_index] = self._inject_local_forecast(
                    self._convert_local_observation(dc_index, obs_state), dc_index, global_obs)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_info_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse info from HierarchicalResetResult.
        This is specifically for reset() which returns HierarchicalResetResult.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()
        # Reset clears any cached static CRD values (factors / timestep) so the
        # next step picks up freshly-configured values.
        self._crd_green_factors = None
        self._crd_brown_factors = None
        self._crd_timestep_hours = None
        info = {}
        # Be robust to Py4J Map proxies vs auto-converted Python dicts.
        # Prefer keySet()/get() when available.
        try:
            if hasattr(info_java, "keySet") and hasattr(info_java, "get"):
                for key in info_java.keySet():
                    value = info_java.get(key)
                    info[str(key)] = self._convert_java_value(value)
                info["crd"] = self._collect_crd_info()
                info["ep_mask_route_count"] = int(getattr(self, "_mask_route_count", 0))   # ep_ prefix: reaches the result rows
                if self._planner_channel is not None:
                    info["planner"] = self._planner_channel
                return info
        except Exception:
            pass

        # Fallback: assume it behaves like a Python mapping / iterable of keys
        for key in info_java:
            try:
                value = info_java[key]
            except Exception:
                # Last resort: try .get(key) without default
                value = info_java.get(key) if hasattr(info_java, "get") else None
            info[str(key)] = self._convert_java_value(value)
        info["crd"] = self._collect_crd_info()
        info["ep_mask_route_count"] = int(getattr(self, "_mask_route_count", 0))   # ep_ prefix: reaches the result rows
        if self._planner_channel is not None:
            info["planner"] = self._planner_channel
        return info

    def _pad_batch_array(self, arr: np.ndarray, target_size: int, dtype=np.int32) -> np.ndarray:
        """
        Pad or trim array to match target_size.

        Args:
            arr: Input array from Java gateway
            target_size: Target array size (global_routing_batch_size)
            dtype: Array data type

        Returns:
            Array of exactly target_size elements
        """
        if len(arr) >= target_size:
            # Trim to target size
            return arr[:target_size]
        else:
            # Pad with zeros
            result = np.zeros(target_size, dtype=dtype)
            result[:len(arr)] = arr
            return result

    def _append_v31_global_features(
        self,
        obs: Dict[str, Any],
        *,
        wait_age,
        time_to_deadline,
        deadline_present,
        is_deferred,
        defer_count,
        global_deferred_count,
        global_deferred_mi,
    ) -> None:
        """Normalize and clip V3.1 defer-state facts into the policy schema."""
        if not self.obs_v31_features:
            return

        batch_size = self.global_routing_batch_size

        def _batch(values) -> np.ndarray:
            return self._pad_batch_array(
                np.asarray(values, dtype=np.float64), batch_size, dtype=np.float64)

        wait = np.nan_to_num(_batch(wait_age), nan=0.0, posinf=np.inf, neginf=0.0)
        ttd = np.nan_to_num(
            _batch(time_to_deadline), nan=0.0, posinf=np.inf, neginf=-np.inf)
        present = _batch(deadline_present)
        deferred = _batch(is_deferred)
        counts = np.nan_to_num(_batch(defer_count), nan=0.0, posinf=np.inf, neginf=0.0)

        obs.update({
            "batch_cloudlet_wait_age": np.clip(
                wait / self._obs_v31_wait_age_scale, 0.0, 1.0).astype(np.float32),
            "batch_cloudlet_time_to_deadline": np.clip(
                ttd / self._obs_v31_deadline_scale, -1.0, 4.0).astype(np.float32),
            "batch_cloudlet_deadline_present": np.clip(
                present, 0.0, 1.0).astype(np.float32),
            "batch_cloudlet_is_deferred": np.clip(
                deferred, 0.0, 1.0).astype(np.float32),
            "batch_cloudlet_defer_count": np.clip(
                counts / self._obs_v31_defer_count_scale, 0.0, 1.0).astype(np.float32),
            "global_deferred_count": np.array([
                np.clip(float(global_deferred_count) / self._obs_v31_global_count_scale, 0.0, 1.0)
            ], dtype=np.float32),
            "global_deferred_mi": np.array([
                np.clip(float(global_deferred_mi) / self._obs_v31_global_mi_scale, 0.0, 1.0)
            ], dtype=np.float32),
        })
        if self.defer_deadline_mask:
            mi = _batch(obs.get("batch_cloudlet_mi", np.zeros(batch_size)))
            pes = _batch(obs.get("batch_cloudlet_pes", np.zeros(batch_size)))
            u = float(self.config.get("cloudlet_cpu_utilization", 1.0) or 1.0)
            obs["batch_cloudlet_defer_allowed"] = defer_allowed_from(
                ttd, present, mi, pes, self._v32_vm_mips, u,
                self._defer_mask_margin_sec, float(self._v32_sim_timestep_sec))

    def _v32_apply_blind_persistence(self, obs: Dict[str, Any]) -> None:
        """Replace future summaries with a current-state persistence baseline.

        The old no-forecast contract used an all-zero sentinel.  V3.2 instead
        pre-registers a physical no-information baseline: future normalized
        green equals the current normalized green level, trend is flat, and
        peak timing is unknown/central.  This method is only called when the
        V3.2 observation gate is enabled; legacy runs retain byte-identical
        zero-fill behavior.
        """
        green_high = max(1e-9, float(self.config.get("obs_green_power_high", 3000.0)))
        current = np.asarray(
            obs.get("dc_current_green_power_w", np.zeros(self.num_datacenters)),
            dtype=np.float64,
        ).reshape(-1)
        current_norm = np.clip(current / green_high, 0.0, 1.0).astype(np.float32)
        obs["dc_future_short_mean"] = current_norm.copy()
        obs["dc_future_short_trend"] = np.zeros(
            self.num_datacenters, dtype=np.float32)
        obs["dc_future_long_mean"] = current_norm.copy()
        obs["dc_future_long_peak_timing"] = np.full(
            self.num_datacenters, 0.5, dtype=np.float32)

    def _v32_forecast_green_bins(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return this arm's per-DC future green-power bins in simulator Watts.

        Full/godeye reads the simulator's own future series through one
        read-only gateway call.  Full/TimeCAP reads the provider's cached raw
        forecast.  Blind never calls either source: it repeats current green
        power (persistence), preserving the information-set boundary.
        """
        current = np.asarray(
            obs.get("dc_current_green_power_w", np.zeros(self.num_datacenters)),
            dtype=np.float64,
        ).reshape(self.num_datacenters)
        if self._v32_forecast_mode == "none":
            return np.repeat(
                current[:, None], self._v32_forecast_bin_count, axis=1)

        offsets = self._v32_forecast_offsets_steps
        if self.timecap_provider is not None:
            raw_per_dc = self.timecap_provider.get_raw_forecast_per_dc(
                horizon=self._v32_forecast_horizon_steps,
                normalize=False,
            )
            if raw_per_dc is None:
                raise RuntimeError(
                    "V3.2 TimeCAP job forecast requested before a raw forecast "
                    "was available"
                )
            out = np.zeros(
                (self.num_datacenters, self._v32_forecast_bin_count),
                dtype=np.float64,
            )
            divisor = max(1e-9, float(
                self.config.get("compressed_power_divisor", 1.0)
            ))
            compressed = any(
                str(dc.get("time_scaling_mode", "")).upper() == "COMPRESSED"
                for dc in self.dc_configs
            )
            for dc_index, dc_id in enumerate(self.dc_ids):
                arr = raw_per_dc.get(dc_id)
                if arr is None or len(arr) == 0:
                    continue
                arr = np.asarray(arr, dtype=np.float64).reshape(-1)
                # TimeCAP index 0 predicts the next simulator step; offsets are
                # one-based for the same reason.
                idx = np.clip(offsets - 1, 0, len(arr) - 1)
                out[dc_index] = arr[idx]
            if compressed:
                out /= divisor
            return self._v32_maybe_perturb_bins(np.maximum(out, 0.0))

        if self.java_env is None or not hasattr(
            self.java_env, "getFuturePerDcGreenPowerW"
        ):
            raise RuntimeError(
                "V3.2 godeye job forecast requires gateway method "
                "getFuturePerDcGreenPowerW"
            )
        horizon_seconds = [
            max(1, int(round(float(step) * self._v32_sim_timestep_sec)))
            for step in offsets
        ]
        java_rows = self.java_env.getFuturePerDcGreenPowerW(horizon_seconds)
        out = np.asarray(
            [[float(value) for value in row] for row in java_rows],
            dtype=np.float64,
        )
        expected = (self.num_datacenters, self._v32_forecast_bin_count)
        if out.shape != expected:
            raise RuntimeError(
                f"gateway V3.2 forecast bins have shape {out.shape}, expected {expected}"
            )
        return self._v32_maybe_perturb_bins(np.maximum(out, 0.0))

    def _v32_maybe_perturb_bins(self, bins):
        import os as _os
        mode = str(_os.environ.get("FORECAST_PERTURB_MODE", "none"))
        if mode.strip().lower() in ("shuffle", "anti"):
            return perturb_future_bins(
                bins, mode, self.config.get("v32_perturb_capacity_w"))
        return bins

    @staticmethod
    def _v32_effective_carbon_factor(
        green_power_w: np.ndarray,
        demand_power_w: np.ndarray,
        green_factor: np.ndarray,
        brown_factor: np.ndarray,
    ) -> np.ndarray:
        """Mirror computeDcCostFeatures using persistence demand."""
        green = np.maximum(np.asarray(green_power_w, dtype=np.float64), 0.0)
        demand = np.maximum(np.asarray(demand_power_w, dtype=np.float64), 0.0)
        ratio = np.where(
            demand > 1e-9,
            np.minimum(1.0, green / np.maximum(demand, 1e-9)),
            np.where(green > 0.0, 1.0, 0.0),
        )
        return ratio * green_factor + (1.0 - ratio) * brown_factor

    def _append_v32_job_forecast_features(
        self,
        obs: Dict[str, Any],
        *,
        time_to_deadline,
        deadline_present,
        forecast_green_bins: Optional[np.ndarray] = None,
    ) -> None:
        """Append bounded per-cloudlet forecast-value features for V3.2.

        Future demand is held at its current per-DC value (persistence demand),
        and current/future candidate costs share the exact same effective-factor
        equation.  A real job may inspect only bins inside
        ``deadline-now-estimated_runtime-margin``.  Padding and jobs without a
        deadline receive the neutral no-op tuple.
        """
        if not self.obs_v32_job_forecast:
            return

        batch = self.global_routing_batch_size
        mi = self._pad_batch_array(
            np.asarray(obs.get("batch_cloudlet_mi", []), dtype=np.float64),
            batch,
            dtype=np.float64,
        )
        pes = self._pad_batch_array(
            np.asarray(obs.get("batch_cloudlet_pes", []), dtype=np.float64),
            batch,
            dtype=np.float64,
        )
        ttd = self._pad_batch_array(
            np.asarray(time_to_deadline, dtype=np.float64),
            batch,
            dtype=np.float64,
        )
        present = self._pad_batch_array(
            np.asarray(deadline_present, dtype=np.float64),
            batch,
            dtype=np.float64,
        )
        current_green = np.asarray(
            obs["dc_current_green_power_w"], dtype=np.float64).reshape(-1)
        demand = np.asarray(
            obs["dc_current_power_w"], dtype=np.float64).reshape(-1)
        available_pes = np.asarray(
            obs["dc_available_pes"], dtype=np.float64).reshape(-1)
        if forecast_green_bins is None:
            forecast_green_bins = self._v32_forecast_green_bins(obs)
        forecast_green_bins = np.asarray(forecast_green_bins, dtype=np.float64)
        expected = (self.num_datacenters, self._v32_forecast_bin_count)
        if forecast_green_bins.shape != expected:
            raise ValueError(
                f"forecast_green_bins shape {forecast_green_bins.shape}, expected {expected}"
            )

        current_factor = self._v32_effective_carbon_factor(
            current_green,
            demand,
            self._v32_green_factors,
            self._v32_brown_factors,
        )
        future_factor = self._v32_effective_carbon_factor(
            forecast_green_bins,
            demand[:, None],
            self._v32_green_factors[:, None],
            self._v32_brown_factors[:, None],
        )
        offsets_sec = (
            self._v32_forecast_offsets_steps.astype(np.float64)
            * self._v32_sim_timestep_sec
        )

        gain = np.zeros(batch, dtype=np.float64)
        relative_time = np.ones(batch, dtype=np.float64)
        best_now = np.zeros(batch, dtype=np.float64)
        best_future = np.zeros(batch, dtype=np.float64)
        raw_gain = np.zeros(batch, dtype=np.float64)
        raw_now = np.zeros(batch, dtype=np.float64)
        raw_future = np.zeros(batch, dtype=np.float64)
        slack_sec = np.zeros(batch, dtype=np.float64)

        cf_mode = self._v32_demand_model == "job_counterfactual_v1"
        for i in range(batch):
            if not np.isfinite(mi[i]) or mi[i] <= 0.0:
                continue
            feasible = np.flatnonzero(available_pes >= max(1.0, pes[i]))
            if feasible.size == 0:
                feasible = np.arange(self.num_datacenters)
            scale = max(0.0, mi[i]) / self._v32_mi_per_kg
            if cf_mode:
                # Counterfactual demand of routing THIS job to each DC: waking
                # an idle host costs its idle draw, plus the job's dynamic
                # share. The identical D_cf prices both sides of the compare.
                d_p = (np.where(demand < 1e-9, self._v32_host_idle_w, 0.0)
                       + max(1.0, pes[i]) * self._v32_host_dyn_w_per_pe)
                d_cf = demand + d_p
                ratio_now = np.minimum(1.0, current_green / d_cf)
                cf_current = (ratio_now * self._v32_green_factors
                              + (1.0 - ratio_now) * self._v32_brown_factors)
                ratio_fut = np.minimum(1.0, forecast_green_bins / d_cf[:, None])
                cf_future = (ratio_fut * self._v32_green_factors[:, None]
                             + (1.0 - ratio_fut) * self._v32_brown_factors[:, None])
                now_cost = scale * cf_current[feasible]
                job_future_factor = cf_future
                # Java reward-ledger alignment: runtime = MI / (PES x MIPS)
                runtime_sec = (max(0.0, mi[i])
                               / (max(1.0, pes[i]) * self._v32_vm_mips))
            else:
                now_cost = scale * current_factor[feasible]
                job_future_factor = future_factor
                runtime_sec = max(0.0, mi[i]) / self._v32_vm_mips
            now = float(np.min(now_cost))
            budget = (
                ttd[i] - runtime_sec - self._v32_deadline_margin_sec
                if present[i] > 0.5 and np.isfinite(ttd[i])
                else 0.0
            )
            budget = max(0.0, float(budget))
            slack_sec[i] = budget
            future = now
            best_offset = 0.0
            eligible_bins = np.flatnonzero(offsets_sec <= budget)
            if eligible_bins.size:
                candidate = scale * job_future_factor[
                    feasible[:, None], eligible_bins[None, :]
                ]
                flat_index = int(np.argmin(candidate))
                dc_pos, bin_pos = np.unravel_index(flat_index, candidate.shape)
                candidate_cost = float(candidate[dc_pos, bin_pos])
                if candidate_cost < future:
                    future = candidate_cost
                    best_offset = float(offsets_sec[eligible_bins[bin_pos]])

            improvement = max(0.0, now - future)
            raw_now[i] = now
            raw_future[i] = future
            raw_gain[i] = improvement
            best_now[i] = np.clip(now / self._v32_job_carbon_high, 0.0, 1.0)
            best_future[i] = np.clip(
                future / self._v32_job_carbon_high, 0.0, 1.0)
            gain[i] = np.clip(
                improvement / self._v32_job_carbon_high, 0.0, 1.0)
            if improvement > 1e-12 and budget > 1e-12:
                relative_time[i] = np.clip(best_offset / budget, 0.0, 1.0)

        if self._v32_forecast_mode == "none":
            # Pre-registered blind tuple.  best_now remains physically useful;
            # persistence states that waiting offers no incremental benefit.
            gain.fill(0.0)
            relative_time.fill(1.0)
            best_future[:] = best_now
            raw_gain.fill(0.0)
            raw_future[:] = raw_now

        obs.update({
            "batch_cloudlet_forecast_gain": gain.astype(np.float32),
            "batch_cloudlet_time_to_best_green": relative_time.astype(np.float32),
            "batch_cloudlet_best_now_carbon": best_now.astype(np.float32),
            "batch_cloudlet_best_future_carbon": best_future.astype(np.float32),
        })
        self._last_v32_job_forecast_debug = {
            "baseline_type": (
                "persistence" if self._v32_forecast_mode == "none" else self.green_oracle_mode
            ),
            "raw_gain_kg": raw_gain,
            "raw_best_now_kg": raw_now,
            "raw_best_future_kg": raw_future,
            "slack_sec": slack_sec,
            "forecast_offsets_sec": offsets_sec.copy(),
            "carbon_high_kg": self._v32_job_carbon_high,
            "demand_assumption": "per_dc_current_power_persistence",
        }

    def _finalize_forecast_observation(
        self,
        obs: Dict[str, Any],
        *,
        time_to_deadline,
        deadline_present,
    ) -> None:
        """Apply the arm's blind semantics, then append gated V3.2 features."""
        if self._v32_forecast_mode == "none":
            if self.obs_v32_job_forecast:
                self._v32_apply_blind_persistence(obs)
            else:
                # Exact historical behavior when the V3.2 gate is disabled.
                for key in (
                    "dc_future_short_mean",
                    "dc_future_short_trend",
                    "dc_future_long_mean",
                    "dc_future_long_peak_timing",
                ):
                    obs[key] = np.zeros(self.num_datacenters, dtype=np.float32)
        self._append_v32_job_forecast_features(
            obs,
            time_to_deadline=time_to_deadline,
            deadline_present=deadline_present,
        )

    def _collect_planner_channel(self, global_obs_java) -> Optional[Dict[str, Any]]:
        """Raw per-slot planner inputs, padded to the batch width, or None on an old jar.

        getBatchCloudletIds landed with the reservation-ledger work. A gateway built
        before that has every other field but not the ids, and a planner keyed on shape
        alone double-books a deferred cloudlet, so the whole block is withheld rather
        than served without identity.
        """
        try:
            ids = np.asarray(global_obs_java.getBatchCloudletIds(), dtype=np.int64)
        except Exception:
            return None
        n = self.global_routing_batch_size

        def pad(arr, dtype, fill):
            a = np.asarray(arr, dtype=dtype).ravel()
            if a.size >= n:
                return a[:n]
            return np.concatenate([a, np.full(n - a.size, fill, dtype=dtype)])

        return {
            "batch_cloudlet_ids": pad(ids, np.int64, -1),
            "batch_cloudlet_pes": pad(global_obs_java.getBatchCloudletPes(), np.int64, 0),
            "batch_cloudlet_mi": pad(global_obs_java.getBatchCloudletMi(), np.int64, 0),
            "batch_cloudlet_time_to_deadline": pad(
                global_obs_java.getBatchCloudletTimeToDeadline(), np.float64, 0.0),
            "batch_cloudlet_deadline_present": pad(
                global_obs_java.getBatchCloudletDeadlinePresent(), np.int64, 0),
            "batch_cloudlet_is_deferred": pad(
                global_obs_java.getBatchCloudletIsDeferred(), np.int64, 0),
            "batch_cloudlet_wait_age": pad(
                global_obs_java.getBatchCloudletWaitAge(), np.float64, 0.0),
            "current_clock": float(global_obs_java.getCurrentClock()),
        }

    def _convert_global_observation(self, global_obs_java) -> Dict[str, Any]:
        """
        Convert Java GlobalObservationState to Python dict.

        When ``green_oracle_mode == "timecap"``, the four ``dc_future_*`` keys are
        overwritten with TimeCAP forecasts instead of Java's CSV-truth God's Eye
        values. All other keys are unaffected. Java still computes its own
        oracle 4-tuple — we just discard it on the Python side, leaving Java's
        execution path bit-identical so the comparison is clean.
        """
        obs = {
            # Green energy metrics
            "dc_current_green_power_w": np.array(global_obs_java.getDcCurrentGreenPowerW(), dtype=np.float32),
            "dc_current_power_w": np.array(global_obs_java.getDcCurrentPowerW(), dtype=np.float32),
            "dc_green_ratio": np.array(global_obs_java.getDcGreenRatio(), dtype=np.float32),
            "dc_cumulative_wasted_green_wh": np.array(global_obs_java.getDcCumulativeWastedGreenWh(), dtype=np.float32),
            # Future energy trend features (filled below — godeye truth or TimeCAP forecast)
            "dc_future_short_mean": None,
            "dc_future_short_trend": None,
            "dc_future_long_mean": None,
            "dc_future_long_peak_timing": None,
            # Resource metrics
            "dc_queue_sizes": np.array(global_obs_java.getDcQueueSizes(), dtype=np.int32),
            "dc_utilizations": np.array(global_obs_java.getDcUtilizations(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getDcAvailablePes(), dtype=np.int32),
            "dc_ram_utilizations": np.array(global_obs_java.getDcRamUtilizations(), dtype=np.float32),
            "upcoming_cloudlets_count": np.array([min(int(global_obs_java.getUpcomingCloudletsCount()), 99999)], dtype=np.int32),
            "batch_cloudlet_pes": self._pad_batch_array(
                np.array(global_obs_java.getBatchCloudletPes(), dtype=np.int32),
                self.global_routing_batch_size, dtype=np.int32
            ),
            "batch_cloudlet_mi": self._pad_batch_array(
                np.array(global_obs_java.getBatchCloudletMi(), dtype=np.int64),
                self.global_routing_batch_size, dtype=np.int64
            ),
            "upcoming_pes_distribution": np.array(global_obs_java.getUpcomingCloudletsPesDistribution(), dtype=np.int32),
            "load_imbalance": np.array([global_obs_java.getLoadImbalance()], dtype=np.float32),
            "recent_completed": np.array([min(int(global_obs_java.getRecentCompletedCloudlets()), 99999)], dtype=np.int32),
        }

        # Evaluator-only planner channel. A curve planner needs raw seconds and a stable
        # identity, neither of which the policy observation carries: the v3.1 features are
        # normalized and gated behind obs_v31_features, and nothing in the observation
        # survives a cloudlet being deferred out of the batch and coming back. This block
        # is handed to the evaluator through info, never through observation_space, so no
        # checkpoint schema changes. Absent keys are left absent rather than filled, so a
        # planner reading them fails loudly instead of planning against a sentinel.
        self._planner_channel = self._collect_planner_channel(global_obs_java)

        raw_time_to_deadline = np.zeros(
            self.global_routing_batch_size, dtype=np.float64)
        raw_deadline_present = np.zeros(
            self.global_routing_batch_size, dtype=np.float64)
        if self.obs_v31_features:
            raw_time_to_deadline = np.asarray(
                global_obs_java.getBatchCloudletTimeToDeadline(), dtype=np.float64)
            raw_deadline_present = np.asarray(
                global_obs_java.getBatchCloudletDeadlinePresent(), dtype=np.float64)
            self._append_v31_global_features(
                obs,
                wait_age=global_obs_java.getBatchCloudletWaitAge(),
                time_to_deadline=raw_time_to_deadline,
                deadline_present=raw_deadline_present,
                is_deferred=global_obs_java.getBatchCloudletIsDeferred(),
                defer_count=global_obs_java.getBatchCloudletDeferCount(),
                global_deferred_count=global_obs_java.getGlobalDeferredCount(),
                global_deferred_mi=global_obs_java.getGlobalDeferredMi(),
            )

        # ── Future-trend features ────────────────────────────────────────
        if self.timecap_provider is None:
            # Oracle / God's Eye path: pass through Java's CSV-truth values
            obs["dc_future_short_mean"]       = np.array(global_obs_java.getDcFutureShortMean(),       dtype=np.float32)
            obs["dc_future_short_trend"]      = np.array(global_obs_java.getDcFutureShortTrend(),      dtype=np.float32)
            obs["dc_future_long_mean"]        = np.array(global_obs_java.getDcFutureLongMean(),        dtype=np.float32)
            obs["dc_future_long_peak_timing"] = np.array(global_obs_java.getDcFutureLongPeakTiming(),  dtype=np.float32)
        else:
            # TimeCAP forecast path — use Java's authoritative simulation clock
            # rather than self.current_step (which lags by 1 inside step()).
            sim_step = int(round(float(global_obs_java.getCurrentClock())))
            feats = self.timecap_provider.step_and_get(sim_step)

            short_mean       = np.full(self.num_datacenters, 0.5, dtype=np.float32)
            short_trend      = np.zeros(self.num_datacenters,       dtype=np.float32)
            long_mean        = np.full(self.num_datacenters, 0.5,  dtype=np.float32)
            long_peak_timing = np.full(self.num_datacenters, 0.5,  dtype=np.float32)

            for i in range(self.num_datacenters):
                dc_id = self.dc_ids[i]
                if dc_id not in feats:
                    # DC has no green energy / no turbines → leave neutral
                    continue
                f = feats[dc_id]
                short_mean[i], short_trend[i], long_mean[i], long_peak_timing[i] = f

            # E3 forecast-SHIFT probe (EVAL-ONLY, env-var gated; default off so training
            # & normal eval are untouched). Injects increasing forecast error to test
            # whether a forecast-trusting policy degrades gracefully (toward noforecast)
            # or COLLAPSES below it (over-trust → EU-CRD has room). Modes via
            # FORECAST_PERTURB_MODE, magnitude via FORECAST_PERTURB_EPS:
            #   blend   — features → (1-eps)*real + eps*neutral (uninformative): graceful test
            #   shuffle — permute per-DC forecast across DCs (points to WRONG DCs): adversarial
            #   noise   — add N(0,eps) then clip: noisy-sensor test
            short_mean, short_trend, long_mean, long_peak_timing = self._perturb_forecast(
                short_mean, short_trend, long_mean, long_peak_timing, sim_step
            )

            obs["dc_future_short_mean"]       = short_mean
            obs["dc_future_short_trend"]      = short_trend
            obs["dc_future_long_mean"]        = long_mean
            obs["dc_future_long_peak_timing"] = long_peak_timing

        self._finalize_forecast_observation(
            obs,
            time_to_deadline=raw_time_to_deadline,
            deadline_present=raw_deadline_present,
        )

        return obs

    def _draw_perturb_episode(self) -> bool:
        """Episode-level lottery for the perturbation curriculum. Returns True when
        FORECAST_PERTURB_MODE is active AND this episode is selected: always selected
        when FORECAST_PERTURB_PROB is unset/<=0 (historical always-on semantics),
        Bernoulli(prob) via the gym-seeded RNG otherwise (reproducible per seed)."""
        import os as _os
        mode = str(_os.environ.get("FORECAST_PERTURB_MODE", "none")).strip().lower()
        if mode in ("", "none"):
            return False
        try:
            prob = float(_os.environ.get("FORECAST_PERTURB_PROB", "0.0"))
        except ValueError:
            prob = 0.0
        if prob <= 0.0:
            return True
        return bool(self.np_random.random() < min(prob, 1.0))

    def _perturb_forecast(self, short_mean, short_trend, long_mean, long_peak_timing, sim_step):
        """E3 forecast-shift probe (EVAL-ONLY). Gated by env var FORECAST_PERTURB_MODE
        (unset/'none' → identity, so training & normal eval are untouched). Magnitude
        via FORECAST_PERTURB_EPS. Lets us inject increasing forecast error at eval to see
        whether a forecast-trusting policy degrades gracefully or collapses below noforecast.
        With FORECAST_PERTURB_PROB set, only episodes selected by _draw_perturb_episode
        are perturbed (training curriculum mode)."""
        import os as _os
        mode = str(_os.environ.get("FORECAST_PERTURB_MODE", "none")).strip().lower()
        if mode in ("", "none"):
            return short_mean, short_trend, long_mean, long_peak_timing
        if not getattr(self, "_perturb_this_episode", True):
            return short_mean, short_trend, long_mean, long_peak_timing
        try:
            eps = float(_os.environ.get("FORECAST_PERTURB_EPS", "0.0"))
        except ValueError:
            eps = 0.0
        neut_m, neut_t, neut_p = 0.5, 0.0, 0.5  # uninformative defaults (match init above)
        if mode == "blend" and eps > 0.0:
            a = float(min(max(eps, 0.0), 1.0))
            short_mean       = (1 - a) * short_mean       + a * neut_m
            short_trend      = (1 - a) * short_trend      + a * neut_t
            long_mean        = (1 - a) * long_mean        + a * neut_m
            long_peak_timing = (1 - a) * long_peak_timing + a * neut_p
        elif mode == "shuffle":
            # Reverse the per-DC forecast (DC0 cleanest ↔ DC4 dirtiest): coherent WRONG forecast.
            short_mean       = short_mean[::-1]
            short_trend      = short_trend[::-1]
            long_mean        = long_mean[::-1]
            long_peak_timing = long_peak_timing[::-1]
        elif mode == "anti":
            # MAXIMALLY adversarial: invert the forecast (high predicted-green where it's actually
            # low, trend flipped). If the policy over-trusts, this should hurt the most.
            short_mean       = 1.0 - short_mean
            short_trend      = -short_trend
            long_mean        = 1.0 - long_mean
            long_peak_timing = 1.0 - long_peak_timing
        elif mode == "panti" and eps > 0.0:
            # Graded coherent lie (reviewer #4, 2026-08-19): continuous path
            # from the clean forecast (a=0) to the full "anti" mirror (a=1),
            # so a severity SWEEP can be plotted instead of two discrete
            # points. a=1 reproduces `anti` byte-for-byte.
            a = float(min(max(eps, 0.0), 1.0))
            short_mean       = (1 - a) * short_mean       + a * (1.0 - short_mean)
            short_trend      = (1 - a) * short_trend      + a * (-short_trend)
            long_mean        = (1 - a) * long_mean        + a * (1.0 - long_mean)
            long_peak_timing = (1 - a) * long_peak_timing + a * (1.0 - long_peak_timing)
        elif mode == "bias" and eps != 0.0:
            # Systematic over/under-prediction of green: the stale-model
            # failure mode, information preserved but mis-levelled.
            b = float(eps)
            short_mean = np.clip(short_mean + b, 0.0, 1.0)
            long_mean  = np.clip(long_mean + b, 0.0, 1.0)
        elif mode == "pshuffle" and eps > 0.0:
            # Partial site permutation: the reversed (shuffle) forecast is
            # adopted on the first round(eps*N) datacentres and the clean one
            # kept elsewhere, so eps=1 reproduces `shuffle` byte-for-byte and
            # small eps mis-maps only a few site-to-feed assignments.
            n = int(np.asarray(short_mean).size)
            k = int(round(float(min(max(eps, 0.0), 1.0)) * n))
            if k > 0:
                sel = np.zeros(n, dtype=bool)
                sel[:k] = True
                short_mean = np.where(sel, np.asarray(short_mean)[::-1], short_mean)
                short_trend = np.where(sel, np.asarray(short_trend)[::-1], short_trend)
                long_mean = np.where(sel, np.asarray(long_mean)[::-1], long_mean)
                long_peak_timing = np.where(
                    sel, np.asarray(long_peak_timing)[::-1], long_peak_timing)
        elif mode == "noise" and eps > 0.0:
            rng = np.random.default_rng(int(sim_step) & 0x7FFFFFFF)
            short_mean       = np.clip(short_mean + rng.normal(0, eps, short_mean.shape), 0.0, 1.0)
            short_trend      = np.clip(short_trend + rng.normal(0, eps, short_trend.shape), -1.0, 1.0)
            long_mean        = np.clip(long_mean + rng.normal(0, eps, long_mean.shape), 0.0, 1.0)
            long_peak_timing = np.clip(long_peak_timing + rng.normal(0, eps, long_peak_timing.shape), 0.0, 1.0)
        return (np.asarray(short_mean, dtype=np.float32), np.asarray(short_trend, dtype=np.float32),
                np.asarray(long_mean, dtype=np.float32), np.asarray(long_peak_timing, dtype=np.float32))

    def _convert_local_observation(self, dc_id: int, local_obs_java) -> Dict[str, Any]:
        """
        Convert Java ObservationState to Python dict, padding/trimming so each DC
        matches the shared observation space while preserving its own host/VM count.
        """
        host_target = self._get_dc_host_count(dc_id)
        vm_target = self._get_dc_vm_count(dc_id)

        host_loads = np.array(local_obs_java.getHostLoads(), dtype=np.float32)[:host_target]
        host_ram_usage = np.array(local_obs_java.getHostRamUsageRatio(), dtype=np.float32)[:host_target]
        vm_loads = np.array(local_obs_java.getVmLoads(), dtype=np.float32)[:vm_target]
        vm_types = np.array(local_obs_java.getVmTypes(), dtype=np.int32)[:vm_target]
        vm_available_pes = np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32)[:vm_target]

        return {
            "host_loads": self._pad_vector(host_loads, self.max_hosts, 0.0),
            "host_ram_usage": self._pad_vector(host_ram_usage, self.max_hosts, 0.0),
            "vm_loads": self._pad_vector(vm_loads, self.max_vms, 0.0),
            "vm_types": self._pad_vector(vm_types, self.max_vms, 0),
            "vm_available_pes": self._pad_vector(vm_available_pes, self.max_vms, 0),
            "waiting_cloudlets": np.array([min(int(local_obs_java.getWaitingCloudlets()), 99999)], dtype=np.int32),
            "next_cloudlet_pes": np.array([min(int(local_obs_java.getNextCloudletPes()), 255)], dtype=np.int32),
        }

    def _inject_local_forecast(self, local_obs_dict: Dict[str, Any], dc_index: int,
                               global_obs: Dict[str, Any]) -> Dict[str, Any]:
        """In dispatch_rate mode, add this DC's green-now + short/long green forecast
        to its local obs so the local agent can decide hold-vs-run. No-op otherwise."""
        if not bool(self.config.get("reward_local_carbon_enabled", False)):
            return local_obs_dict
        gn = np.asarray(global_obs.get("dc_current_green_power_w", []), dtype=np.float32).ravel()
        fs = np.asarray(global_obs.get("dc_future_short_mean", []), dtype=np.float32).ravel()
        fl = np.asarray(global_obs.get("dc_future_long_mean", []), dtype=np.float32).ravel()
        gmax = float(gn.max()) if gn.size else 0.0
        gnow = (gn[dc_index] / gmax) if (gmax > 1e-9 and dc_index < gn.size) else 0.0
        local_obs_dict["green_now"] = np.array([np.clip(gnow, 0.0, 1.0)], dtype=np.float32)
        local_obs_dict["green_forecast_short"] = np.array(
            [np.clip(fs[dc_index] if dc_index < fs.size else 0.0, 0.0, 1.0)], dtype=np.float32)
        local_obs_dict["green_forecast_long"] = np.array(
            [np.clip(fl[dc_index] if dc_index < fl.size else 0.0, 0.0, 1.0)], dtype=np.float32)
        return local_obs_dict

    def _get_dc_host_count(self, dc_id: int) -> int:
        """Return configured host count for a datacenter (fallback to max_hosts)."""
        if hasattr(self, "dc_host_counts") and 0 <= dc_id < len(self.dc_host_counts):
            return self.dc_host_counts[dc_id]
        return getattr(self, "max_hosts", 1)

    def _get_dc_vm_count(self, dc_id: int) -> int:
        """Return configured VM count for a datacenter (fallback to max_vms)."""
        if hasattr(self, "dc_vm_counts") and 0 <= dc_id < len(self.dc_vm_counts):
            return self.dc_vm_counts[dc_id]
        return getattr(self, "max_vms", 1)

    @staticmethod
    def _pad_vector(values: np.ndarray, target_len: int, fill_value: float) -> np.ndarray:
        """
        Ensure vectors share a consistent length by trimming overflow and padding
        the tail with a provided fill_value.
        """
        current_len = values.shape[0]
        if current_len == target_len:
            return values

        if current_len > target_len:
            return values[:target_len]

        padded = np.full((target_len,), fill_value, dtype=values.dtype)
        if current_len > 0:
            padded[:current_len] = values
        return padded

    def _parse_hierarchical_observation(
        self,
        result  # HierarchicalStepResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalStepResult into observation dict.
        This is specifically for step() which returns HierarchicalStepResult.
        """
        # Parse global observation
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations
        local_obs_map_java = result.getLocalObservations()
        local_obs = {}
        if local_obs_map_java is None:
            return {"global": global_obs, "local": local_obs}

        # Be robust to Py4J Map proxies vs auto-converted Python dicts:
        # iterate actual keys provided by Java rather than assuming 0..N-1 membership works.
        try:
            keys_iter = local_obs_map_java.keySet() if hasattr(local_obs_map_java, "keySet") else local_obs_map_java
            for dc_id_raw in keys_iter:
                dc_id = int(dc_id_raw)
                dc_index = self.dc_id_to_index.get(dc_id)
                if dc_index is None:
                    logger.warning("Unknown datacenter_id in step observations: %s", dc_id)
                    continue
                try:
                    obs_state = (
                        local_obs_map_java.get(dc_id_raw)
                        if hasattr(local_obs_map_java, "get")
                        else local_obs_map_java[dc_id_raw]
                    )
                except Exception:
                    # Fallback: attempt Python-int lookup
                    obs_state = (
                        local_obs_map_java.get(dc_id)
                        if hasattr(local_obs_map_java, "get")
                        else local_obs_map_java[dc_id]
                    )

                if obs_state is not None:
                    local_obs[dc_index] = self._inject_local_forecast(
                        self._convert_local_observation(dc_index, obs_state), dc_index, global_obs)
        except Exception as e:
            logger.error("Failed to parse local observations map: %s", e)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_hierarchical_rewards(
        self,
        result
    ) -> Dict[str, Any]:
        """
        Parse hierarchical rewards from step result.

        Returns:
            {
                'global': float,
                'local': {dc_index: float}
            }
        """
        global_reward = result.getGlobalReward()

        local_rewards_java = result.getLocalRewards()
        local_rewards = {}
        for dc_index in range(self.num_datacenters):
            dc_id = self.dc_index_to_id.get(dc_index, dc_index)
            try:
                if hasattr(local_rewards_java, "get"):
                    reward_val = local_rewards_java.get(dc_id, 0.0)
                else:
                    reward_val = local_rewards_java[dc_id]
            except Exception:
                reward_val = 0.0
            local_rewards[dc_index] = reward_val

        return {
            "global": global_reward,
            "local": local_rewards
        }

    def _parse_info(self, result) -> Dict[str, Any]:
        """
        Parse additional info from step result.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()

        # Convert Java Map to Python dict with serializable values
        info = {}
        # Be robust to Py4J Map proxies vs auto-converted Python dicts.
        # Prefer keySet()/get() when available.
        try:
            if hasattr(info_java, "keySet") and hasattr(info_java, "get"):
                for key in info_java.keySet():
                    value = info_java.get(key)
                    info[str(key)] = self._convert_java_value(value)
                info["episode_step"] = self.current_step
                info["episode_reward"] = self.episode_reward
                info["crd"] = self._collect_crd_info()
                info["ep_mask_route_count"] = int(getattr(self, "_mask_route_count", 0))   # ep_ prefix: reaches the result rows
                if self._planner_channel is not None:
                    info["planner"] = self._planner_channel
                return info
        except Exception:
            pass

        # Fallback: assume it behaves like a Python mapping / iterable of keys
        for key in info_java:
            try:
                value = info_java[key]
            except Exception:
                value = info_java.get(key) if hasattr(info_java, "get") else None
            info[str(key)] = self._convert_java_value(value)

        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward
        info["crd"] = self._collect_crd_info()
        info["ep_mask_route_count"] = int(getattr(self, "_mask_route_count", 0))   # ep_ prefix: reaches the result rows
        if self._planner_channel is not None:
            info["planner"] = self._planner_channel

        return info

    # ------------------------------------------------------------------
    # 2026-05-12 Level A fast-path: parse the entire step result from a
    # single flat Map produced by HierarchicalStepResult.getStepAsFlatMap().
    # Replaces the ~200 individual Py4J getter RPCs the legacy methods do.
    # See the Java method's javadoc for the key naming convention.
    # ------------------------------------------------------------------

    def _convert_global_observation_from_flat(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Mirror of `_convert_global_observation` reading from a flat map."""
        obs = {
            "dc_current_green_power_w":     np.array(flat["g.dc_current_green_power_w"],     dtype=np.float32),
            "dc_current_power_w":           np.array(flat["g.dc_current_power_w"],           dtype=np.float32),
            "dc_green_ratio":               np.array(flat["g.dc_green_ratio"],               dtype=np.float32),
            "dc_cumulative_wasted_green_wh": np.array(flat["g.dc_cumulative_wasted_green_wh"], dtype=np.float32),
            "dc_future_short_mean":       None,
            "dc_future_short_trend":      None,
            "dc_future_long_mean":        None,
            "dc_future_long_peak_timing": None,
            "dc_queue_sizes":            np.array(flat["g.dc_queue_sizes"],         dtype=np.int32),
            "dc_utilizations":           np.array(flat["g.dc_utilizations"],        dtype=np.float32),
            "dc_available_pes":          np.array(flat["g.dc_available_pes"],       dtype=np.int32),
            "dc_ram_utilizations":       np.array(flat["g.dc_ram_utilizations"],    dtype=np.float32),
            "upcoming_cloudlets_count":  np.array(
                [min(int(flat["g.upcoming_cloudlets_count"]), 99999)], dtype=np.int32),
            "batch_cloudlet_pes":        self._pad_batch_array(
                np.array(flat["g.batch_cloudlet_pes"], dtype=np.int32),
                self.global_routing_batch_size, dtype=np.int32),
            "batch_cloudlet_mi":         self._pad_batch_array(
                np.array(flat["g.batch_cloudlet_mi"], dtype=np.int64),
                self.global_routing_batch_size, dtype=np.int64),
            "upcoming_pes_distribution": np.array(
                flat["g.upcoming_cloudlets_pes_distribution"], dtype=np.int32),
            "load_imbalance":            np.array([float(flat["g.load_imbalance"])], dtype=np.float32),
            "recent_completed":          np.array(
                [min(int(flat["g.recent_completed_cloudlets"]), 99999)], dtype=np.int32),
        }

        raw_time_to_deadline = np.zeros(
            self.global_routing_batch_size, dtype=np.float64)
        raw_deadline_present = np.zeros(
            self.global_routing_batch_size, dtype=np.float64)
        if self.obs_v31_features:
            raw_time_to_deadline = np.asarray(
                flat["g.batch_cloudlet_time_to_deadline"], dtype=np.float64)
            raw_deadline_present = np.asarray(
                flat["g.batch_cloudlet_deadline_present"], dtype=np.float64)
            self._append_v31_global_features(
                obs,
                wait_age=flat["g.batch_cloudlet_wait_age"],
                time_to_deadline=raw_time_to_deadline,
                deadline_present=raw_deadline_present,
                is_deferred=flat["g.batch_cloudlet_is_deferred"],
                defer_count=flat["g.batch_cloudlet_defer_count"],
                global_deferred_count=flat["g.global_deferred_count"],
                global_deferred_mi=flat["g.global_deferred_mi"],
            )

        # Future-trend features: same overlay rules as the legacy path.
        if self.timecap_provider is None:
            obs["dc_future_short_mean"]       = np.array(flat["g.dc_future_short_mean"],       dtype=np.float32)
            obs["dc_future_short_trend"]      = np.array(flat["g.dc_future_short_trend"],      dtype=np.float32)
            obs["dc_future_long_mean"]        = np.array(flat["g.dc_future_long_mean"],        dtype=np.float32)
            obs["dc_future_long_peak_timing"] = np.array(flat["g.dc_future_long_peak_timing"], dtype=np.float32)
        else:
            sim_step = int(round(float(flat["g.current_clock"])))
            feats = self.timecap_provider.step_and_get(sim_step)
            short_mean       = np.full(self.num_datacenters, 0.5, dtype=np.float32)
            short_trend      = np.zeros(self.num_datacenters,       dtype=np.float32)
            long_mean        = np.full(self.num_datacenters, 0.5,  dtype=np.float32)
            long_peak_timing = np.full(self.num_datacenters, 0.5,  dtype=np.float32)
            for i in range(self.num_datacenters):
                dc_id = self.dc_ids[i]
                if dc_id not in feats:
                    continue
                f = feats[dc_id]
                short_mean[i], short_trend[i], long_mean[i], long_peak_timing[i] = f
            obs["dc_future_short_mean"]       = short_mean
            obs["dc_future_short_trend"]      = short_trend
            obs["dc_future_long_mean"]        = long_mean
            obs["dc_future_long_peak_timing"] = long_peak_timing

        # The historical flat path omitted forecast_mode="none" zero-fill.
        # Preserve that exact (albeit surprising) behavior while the V3.2 gate
        # is off: default-off must remain checkpoint/experiment compatible.
        # V3.2 closes the leak explicitly and makes both paths persistence-based.
        if self.obs_v32_job_forecast:
            self._finalize_forecast_observation(
                obs,
                time_to_deadline=raw_time_to_deadline,
                deadline_present=raw_deadline_present,
            )

        return obs

    def _convert_local_observation_from_flat(self, dc_id: int, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Mirror of `_convert_local_observation` reading from a flat map."""
        p = f"l.{dc_id}."
        host_target = self._get_dc_host_count(dc_id)
        vm_target = self._get_dc_vm_count(dc_id)

        host_loads = np.array(flat[p + "host_loads"],          dtype=np.float32)[:host_target]
        host_ram_usage = np.array(flat[p + "host_ram_usage_ratio"], dtype=np.float32)[:host_target]
        vm_loads = np.array(flat[p + "vm_loads"],              dtype=np.float32)[:vm_target]
        vm_types = np.array(flat[p + "vm_types"],              dtype=np.int32)[:vm_target]
        vm_available_pes = np.array(flat[p + "vm_available_pes"], dtype=np.int32)[:vm_target]

        return {
            "host_loads":       self._pad_vector(host_loads,       self.max_hosts, 0.0),
            "host_ram_usage":   self._pad_vector(host_ram_usage,   self.max_hosts, 0.0),
            "vm_loads":         self._pad_vector(vm_loads,         self.max_vms,   0.0),
            "vm_types":         self._pad_vector(vm_types,         self.max_vms,   0),
            "vm_available_pes": self._pad_vector(vm_available_pes, self.max_vms,   0),
            "waiting_cloudlets": np.array(
                [min(int(flat[p + "waiting_cloudlets"]), 99999)], dtype=np.int32),
            "next_cloudlet_pes": np.array(
                [min(int(flat[p + "next_cloudlet_pes"]), 255)],    dtype=np.int32),
        }

    def _parse_observation_from_flat(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level obs parser for the flat path — same shape as `_parse_hierarchical_observation`."""
        global_obs = self._convert_global_observation_from_flat(flat)
        local_obs: Dict[int, Any] = {}
        for dc_index in range(self.num_datacenters):
            dc_id = self.dc_index_to_id.get(dc_index, dc_index)
            if f"l.{dc_id}.vm_loads" not in flat:
                logger.debug("Flat obs missing DC %s; skipping", dc_id)
                continue
            local_obs[dc_index] = self._convert_local_observation_from_flat(dc_id, flat)
        return {"global": global_obs, "local": local_obs}

    def _parse_rewards_from_flat(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Rewards parser — same shape as `_parse_hierarchical_rewards`."""
        global_reward = float(flat["r.global"])
        local_rewards_java = flat.get("r.local")
        local_rewards: Dict[int, float] = {}
        for dc_index in range(self.num_datacenters):
            dc_id = self.dc_index_to_id.get(dc_index, dc_index)
            try:
                if local_rewards_java is None:
                    val = 0.0
                elif hasattr(local_rewards_java, "get"):
                    val = local_rewards_java.get(dc_id, 0.0)
                else:
                    val = local_rewards_java[dc_id]
            except Exception:
                val = 0.0
            local_rewards[dc_index] = float(val if val is not None else 0.0)
        return {"global": global_reward, "local": local_rewards}

    def _parse_info_from_flat(self, flat: Dict[str, Any]) -> Dict[str, Any]:
        """Info parser — same shape as `_parse_info`."""
        info_java = flat.get("info")
        info: Dict[str, Any] = {}
        if info_java is not None:
            try:
                if hasattr(info_java, "keySet") and hasattr(info_java, "get"):
                    for key in info_java.keySet():
                        info[str(key)] = self._convert_java_value(info_java.get(key))
                else:
                    for key in info_java:
                        info[str(key)] = self._convert_java_value(
                            info_java.get(key) if hasattr(info_java, "get") else info_java[key]
                        )
            except Exception as e:
                logger.debug("Flat-path info parsing fallback: %s", e)
        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward
        info["crd"] = self._collect_crd_info()
        info["ep_mask_route_count"] = int(getattr(self, "_mask_route_count", 0))   # ep_ prefix: reaches the result rows
        if self._planner_channel is not None:
            info["planner"] = self._planner_channel
        return info

    def _collect_crd_info(self) -> Dict[str, Any]:
        """
        Snapshot the per-step state needed by the CRD callback to compute the
        forecast counterfactual analytically (no replay).

        Fields written:
        - actual_wind_w:     per-DC realized green power at end of step (W)
        - p_total_w:         per-DC total power demand at end of step (W)
        - running_max_carbon: simulator's current normalization denominator
        - timestep_hours:    duration matching the carbon formula
        - green_carbon_factor / brown_carbon_factor: per-DC kgCO2/kWh

        Predicted wind (Ŵ_t) is added by WindPredictionWrapper if present;
        the base env does not know about predictions.
        """
        if self.java_env is None:
            return {}
        try:
            self._ensure_crd_static_cache()
            actual_wind = list(self.java_env.getCurrentPerDcGreenPowerW())
            p_total = list(self.java_env.getCurrentPerDcTotalPowerW())
            running_max = float(self.java_env.getCurrentRunningMaxCarbon())
            crd: Dict[str, Any] = {
                "actual_wind_w": [float(x) for x in actual_wind],
                "p_total_w": [float(x) for x in p_total],
                "running_max_carbon": running_max,
                "timestep_hours": self._crd_timestep_hours,
                "green_carbon_factor": list(self._crd_green_factors or []),
                "brown_carbon_factor": list(self._crd_brown_factors or []),
            }
            # M2.3: pass through decision-time signals for the baseline
            # scheduler (GreenQueueBalanced needs dc_green_ratio + queue sizes).
            # These come from the just-parsed global obs; both are arrays of
            # length num_datacenters.
            glb = self._last_global_obs_for_crd or {}
            queue_sizes = glb.get("dc_queue_sizes")
            green_ratio = glb.get("dc_green_ratio")
            if queue_sizes is not None:
                crd["dc_queue_sizes"] = [int(x) for x in queue_sizes]
            if green_ratio is not None:
                crd["dc_green_ratio"] = [float(x) for x in green_ratio]
            # CRD M2.2: forecast counterfactual needs Ŵ_t (predicted wind in W
            # per DC). When `green_oracle_mode=timecap`, the provider already
            # ran a TimeCAP forward this step and we can pull the horizon-0
            # prediction out of its cache without any extra inference.
            #
            # Length alignment: the provider only forecasts for DCs that have
            # turbine assignments (e.g., 3 of 5 in a heterogeneous setup with
            # brown-only DCs). We pad to `num_datacenters` with 0.0 so the
            # predicted_wind_w list matches actual_wind_w / p_total_w / factors
            # — all of which are full per-DC arrays. Brown-only DCs have 0
            # actual wind, so 0 predicted wind is the correct match.
            if self.timecap_provider is not None:
                try:
                    pred_w_provider = self.timecap_provider.get_predicted_wind_w_per_dc(horizon=0)
                    if pred_w_provider is not None:
                        pred_w_full = [0.0] * self.num_datacenters
                        provider_dc_ids = getattr(self.timecap_provider, "dc_ids", [])
                        for src_idx, dc_id in enumerate(provider_dc_ids):
                            if src_idx >= len(pred_w_provider):
                                break
                            env_idx = self.dc_id_to_index.get(int(dc_id))
                            if env_idx is not None and 0 <= env_idx < self.num_datacenters:
                                pred_w_full[env_idx] = float(pred_w_provider[src_idx])
                        # v5.1 scale fix (config-gated crd.forecast.scale_fix):
                        # the provider returns raw kW*1000 W, but actual_wind_w
                        # from the gateway is divided by compressed_power_divisor
                        # under COMPRESSED scaling (1500x in the C-regime), so
                        # without this the forecast counterfactual compares
                        # against a 1500x-inflated hypothetical supply and
                        # R_forecast is a biased constant unrelated to error.
                        _fc_cfg = ((self.config.get("crd", {}) or {}).get("forecast", {}) or {})
                        if bool(_fc_cfg.get("scale_fix", False)):
                            _div = float(self.config.get("compressed_power_divisor", 60.0) or 60.0)
                            if _div > 0:
                                pred_w_full = [p / _div for p in pred_w_full]
                        crd["predicted_wind_w"] = pred_w_full
                except Exception as e:
                    logger.debug(f"timecap predicted_wind_w accessor failed: {e}")
            return crd
        except Exception as e:
            logger.warning(f"_collect_crd_info failed: {e}")
            return {}

    def _ensure_crd_static_cache(self) -> None:
        """Lazily fetch carbon factors and timestep duration once per simulation."""
        if self._crd_green_factors is not None:
            return
        if self.java_env is None:
            return
        self._crd_green_factors = [
            float(x) for x in self.java_env.getCurrentPerDcGreenCarbonFactor()
        ]
        self._crd_brown_factors = [
            float(x) for x in self.java_env.getCurrentPerDcBrownCarbonFactor()
        ]
        self._crd_timestep_hours = float(self.java_env.getCurrentTimestepHours())
    
    def _convert_java_value(self, value):
        """
        Convert a Java value (from Py4J) to a Python native type.

        IMPORTANT:
        - Preserves nested Maps/Lists by converting them recursively to dict/list
        - Avoids stringifying complex objects such as energy metrics maps
        """
        if value is None:
            return None

        # Already plain Python scalar
        if isinstance(value, (bool, int, float, str)):
            return value

        # Handle Java Maps (e.g., HashMap) exposed via Py4J:
        # They usually have keySet() and get() methods.
        try:
            if hasattr(value, "keySet") and hasattr(value, "get"):
                py_dict = {}
                for k in value.keySet():
                    # Try to keep numeric keys as int (e.g., DC id 0..N-1),
                    # fall back to string for non-numeric keys.
                    try:
                        py_key = int(k)
                    except (TypeError, ValueError):
                        py_key = str(k)
                    py_dict[py_key] = self._convert_java_value(value.get(k))
                return py_dict
        except Exception:
            # If anything goes wrong, fall through to other heuristics
            pass

        # Handle Java Lists or other iterable collections
        try:
            # Many Py4J Java collections are iterable but not sequences
            iterator = iter(value)
        except TypeError:
            iterator = None

        if iterator is not None:
            try:
                return [self._convert_java_value(v) for v in list(iterator)]
            except Exception:
                # If iteration fails, continue to scalar conversion attempts
                pass

        # Try numeric conversions (Integer, Long, Double, etc.)
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        # Try boolean from string representation
        try:
            s = str(value).lower()
            if s in ("true", "false"):
                return s == "true"
        except Exception:
            pass

        # Fallback: string representation for unknown complex types
        return str(value)

    def render(self):
        """
        Render the environment (not implemented for this environment).
        """
        pass

    def close(self):
        """
        Close the environment and cleanup resources.

        Safely closes the Java simulation environment and shuts down the Py4J gateway.
        This method is called automatically by Gymnasium when the environment is no longer needed.
        """
        # Close Java simulation environment
        if self.java_env is not None:
            try:
                logger.info("Closing Java simulation environment...")
                self.java_env.close()
                logger.info("Java simulation environment closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Java simulation environment: {e}")

        # Close Py4J gateway client connection (do NOT shutdown the Java server,
        # so that multiple evaluations / combinations can reuse the same JVM)
        if self.gateway is not None:
            try:
                logger.info("Closing Py4J gateway client connection...")
                self.gateway.close()
                logger.info("Py4J gateway client closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Py4J gateway client: {e}")
            finally:
                self.gateway = None
                self.java_env = None

        # Suppress py4j's own ERROR logs from now on.  Once we have explicitly
        # closed the gateway client, any further Py4JNetworkError /
        # ConnectionResetError raised by py4j's background callback-server
        # thread or by Java-proxy finalizers (which fire when Python GC runs
        # after the JVM is gone) is *expected*, not actionable, and only
        # produces noise in evaluation/test output.  Lifting the level to
        # CRITICAL silences these without hiding genuine problems that occur
        # before close().
        logging.getLogger("py4j.java_gateway").setLevel(logging.CRITICAL)
        logging.getLogger("py4j.clientserver").setLevel(logging.CRITICAL)

        # Terminate the Java process if we launched it
        if self.java_process:
            try:
                logger.info(f"Terminating Java Gateway process (PID {self.java_process.pid})...")
                os.killpg(os.getpgid(self.java_process.pid), signal.SIGTERM)
                self.java_process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating Java process: {e}")
                try:
                    os.killpg(os.getpgid(self.java_process.pid), signal.SIGKILL)
                except Exception:
                    pass
            finally:
                self.java_process = None
                if hasattr(self, '_java_log_file') and self._java_log_file:
                    self._java_log_file.close()

    def get_num_datacenters(self) -> int:
        """Get the number of datacenters in the environment."""
        return self.num_datacenters

    def get_arriving_cloudlets_count(self) -> int:
        """
        Get the number of cloudlets in global waiting queue.
        
        DEPRECATED: This method name is misleading. Use get_global_waiting_cloudlets_count() instead.
        Kept for backward compatibility with tests.
        """
        return self.get_global_waiting_cloudlets_count()
    
    def get_global_waiting_cloudlets_count(self) -> int:
        """Get the number of cloudlets in the global waiting queue (batch routing mode)."""
        if self.java_env is None:
            return 0
        return self.java_env.getGlobalWaitingCloudletsCount()

    def get_global_action_mask(self, global_obs: Dict[str, Any]) -> np.ndarray:
        """
        Generate slot-level mask for global MultiDiscrete routing action.

        Mask semantics:
        - mask[i] = 1.0: slot i corresponds to a real cloudlet in the upcoming batch
        - mask[i] = 0.0: slot i is padding (no cloudlet)

        Preferred source is batch_cloudlet_pes/mi arrays (direct slot-level signals).
        Fallback uses upcoming_cloudlets_count to derive a valid prefix length.

        Args:
            global_obs: Global observation dict.

        Returns:
            np.ndarray of shape (global_routing_batch_size,), dtype float32.
        """
        batch_size = int(self.global_routing_batch_size)
        if batch_size <= 0:
            return np.zeros((0,), dtype=np.float32)

        try:
            pes = np.asarray(global_obs.get("batch_cloudlet_pes", []), dtype=np.int64)
            mi = np.asarray(global_obs.get("batch_cloudlet_mi", []), dtype=np.int64)

            # Use slot-level batch features when available.
            if pes.size > 0 and mi.size > 0:
                use_len = min(batch_size, int(pes.size), int(mi.size))
                mask = np.zeros(batch_size, dtype=np.float32)
                if use_len > 0:
                    valid_slots = (pes[:use_len] > 0) & (mi[:use_len] > 0)
                    mask[:use_len] = valid_slots.astype(np.float32)
                return mask
        except Exception as e:
            logger.debug("Failed to build global action mask from batch arrays: %s", e)

        # Fallback: derive valid prefix from queue count.
        try:
            upcoming_count = int(global_obs.get("upcoming_cloudlets_count", 0))
            valid_len = max(0, min(batch_size, upcoming_count))
            mask = np.zeros(batch_size, dtype=np.float32)
            mask[:valid_len] = 1.0
            return mask
        except Exception as e:
            logger.warning(
                "Failed to build global action mask from upcoming_cloudlets_count (%s). "
                "Allowing all global action slots.", e
            )
            return np.ones(batch_size, dtype=np.float32)

    def get_local_action_masks(self, dc_id: int) -> np.ndarray:
        """
        Generate action mask for a specific datacenter's local agent.

        Mask logic (consistent with Single-DC environment):
        - If queue is empty: only allow action 0 (NoAssign)
        - If queue has tasks: forbid action 0, allow VMs with enough resources
        - If no VM has enough resources: allow all VMs (Java handles penalty)

        Args:
            dc_id: Datacenter index (0..N-1). Must be an index.

        Returns:
            mask: Boolean array of shape (num_vms+1,) where True = action allowed
        """
        # dispatch_rate mode: the action is "how many cloudlets to release" (0..N),
        # every value is always a legal action → all-ones mask (no masking needed).
        if str(self.config.get("local_dispatch_mode", "vm_placement")).strip() == "dispatch_rate":
            return np.ones(self.local_action_space.n, dtype=bool)

        # Enforce explicit dcIndex usage to avoid id/index ambiguity.
        dc_index = dc_id
        if dc_index < 0 or dc_index >= self.num_datacenters:
            if dc_id in self.dc_id_to_index:
                raise ValueError(
                    f"get_local_action_masks expects dcIndex. Received dcId={dc_id}. "
                    "Convert to dcIndex before calling."
                )
            raise ValueError(
                f"get_local_action_masks expects dcIndex in [0, {self.num_datacenters - 1}]. "
                f"Received {dc_id}."
            )

        # Fallback: allow all actions if environment not initialized or invalid index
        if self.java_env is None or dc_index >= self.num_datacenters or dc_index < 0:
            logger.warning(f"Cannot generate mask for DC {dc_id}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get DC state from last observation
        try:
            if not hasattr(self, 'last_observations') or 'local' not in self.last_observations:
                logger.debug(f"No observations available yet, allowing all actions for DC {dc_id}")
                return np.ones(self.local_action_space.n, dtype=bool)

            local_obs = self.last_observations["local"].get(dc_index)
            if local_obs is None:
                logger.warning(f"No observation for DC {dc_id}, allowing all actions")
                return np.ones(self.local_action_space.n, dtype=bool)

            vm_available_pes = local_obs["vm_available_pes"]
            waiting_cloudlets = int(np.asarray(local_obs["waiting_cloudlets"]).flat[0])
            next_cloudlet_pes = int(np.asarray(local_obs["next_cloudlet_pes"]).flat[0])

        except Exception as e:
            logger.error(f"Failed to extract state for DC {dc_id}: {e}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get actual VM count for this DC
        dc_vm_count = self._get_dc_vm_count(dc_index)
        
        # Initialize mask (all False)
        mask = np.zeros(self.local_action_space.n, dtype=bool)

        # Case 1: Queue is empty or next task invalid
        if waiting_cloudlets == 0 or next_cloudlet_pes == 0:
            mask[0] = True  # Only allow action 0 (NoAssign)
            logger.debug(f"DC {dc_id}: Queue empty, only NoAssign allowed")
            return mask

        # Case 2: Queue has tasks
        mask[0] = False  # Forbid explicit NoAssign (encourage assignment)

        # Check each VM's resources (only actual VMs, not padding)
        has_valid_vm = False
        for vm_idx in range(min(len(vm_available_pes), dc_vm_count)):
            available_pes = vm_available_pes[vm_idx]
            if available_pes >= next_cloudlet_pes:
                mask[vm_idx + 1] = True  # action (vm_idx+1) → targetVmId (vm_idx)
                has_valid_vm = True

        # Case 3: No VM has enough resources
        # Align with loadbalancing_env.py: Force assignment (disallow NoAssign)
        if not has_valid_vm:
            logger.debug(f"DC {dc_id}: No VM has {next_cloudlet_pes} PEs, allowing all VMs (forcing assignment)")
            mask[0] = False  # Disallow NoAssign
            mask[1:dc_vm_count+1] = True  # Allow all VMs

        logger.debug(f"DC {dc_id}: Mask generated - {np.sum(mask)}/{len(mask)} actions allowed")
        return mask
