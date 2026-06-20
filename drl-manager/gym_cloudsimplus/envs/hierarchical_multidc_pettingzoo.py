"""
src/training/train_rllib_multidc.py
src/training/train_rlmodule_gtrxl.py
src/training/train_rlmodule_multidc.py
src/training/train_rlmodule_gmlp.py
src/training/train_rlmodule_resmlp.py 直接使用

PettingZoo ParallelEnv wrapper for Hierarchical Multi-Datacenter Environment.

This module provides a PettingZoo-compatible interface for the hierarchical
multi-datacenter MARL system, enabling compatibility with PettingZoo-based
training frameworks (RLlib, CleanRL, etc.) while preserving all existing
functionality.

Architecture:
    PettingZoo ParallelEnv (this file)
        | wraps
    HierarchicalMultiDCEnv (existing, unchanged)
        | wraps
    Java CloudSim Plus Simulation (Py4J)

Agents:
    - "global_agent": Routes arriving cloudlets to datacenters
    - "local_agent_0", "local_agent_1", ...: VM scheduling per datacenter

Key Features:
    - Zero modifications to existing code
    - Zero Java changes required
    - Standard PettingZoo API compliance
    - Parameter sharing support via policy mapping
    - Action masking support
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .hierarchical_multidc_env import HierarchicalMultiDCEnv

logger = logging.getLogger(__name__)


class HierarchicalMultiDCParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for hierarchical multi-datacenter MARL.

    This environment wraps the existing HierarchicalMultiDCEnv and provides
    a standard PettingZoo API, making it compatible with various MARL frameworks.

    Agents:
        - "global_agent": Handles datacenter routing decisions
        - "local_agent_0", ..., "local_agent_N": Handle VM scheduling per DC

    All agents act simultaneously in each timestep (parallel execution).
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "hierarchical_multidc_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        """
        Initialize PettingZoo wrapper.

        Args:
            config: Configuration dictionary for HierarchicalMultiDCEnv
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()

        self.render_mode = render_mode

        # Store full config dict so the wrapper can read live-updated entries
        # (e.g. Lagrangian λ).  The inner "lagrangian" dict is shared-by-reference
        # with RLlib's algorithm.config.env_config, so callback mutations are
        # visible to step() without any explicit sync.
        self.config = config

        # Lagrangian state — filled by LagrangianCallback on each training
        # iteration via set_lagrangian_lambda().  Missing/None → λ=0 (disabled).
        lagrangian_cfg = config.get("lagrangian", {}) if isinstance(config, dict) else {}
        self._lagrangian_cfg = lagrangian_cfg if isinstance(lagrangian_cfg, dict) else {}
        self._lagrangian_enabled = bool(self._lagrangian_cfg.get("enabled", False))
        # Per-episode running accumulator for c_step — exposed to callback via
        # info at episode end (no shared-dict mutation, which doesn't survive
        # RLlib's env_config serialization across the new API stack).
        self._ep_c_step_running_sum = 0.0
        self._ep_c_step_running_count = 0
        # Per-episode running sum of λ·c_step actually subtracted from the
        # global agent's reward.  Exposed in terminal info so the logger can
        # show "global reward AFTER Lagrangian" — otherwise monitor.csv
        # reports the Java pre-penalty number, hiding the constraint pressure.
        self._ep_lagrangian_penalty_sum = 0.0

        # CTDE (Centralized Training with Decentralized Execution) support.
        # When enabled, each local agent's observation includes a "global_state"
        # field containing the flattened global agent observation. This is used
        # by a centralized critic during training; the actor ignores it.
        ctde_cfg = config.get("ctde", {})
        self.ctde_enabled = bool(ctde_cfg.get("enabled", False)) if isinstance(ctde_cfg, dict) else bool(ctde_cfg)

        # EU-CRD: when enabled, attach a per-step "crd_aux" sibling to every
        # agent's observation carrying the raw wind/power/carbon snapshot the
        # learner needs to compute R_forecast on the padded (B, T) obs grid.
        # This is the robust replacement for the infos-based forecast path
        # (infos is unreliable under PPO minibatching). The policy networks
        # only read obs["observation"]/obs["action_mask"], so the crd_aux
        # sibling is carried to the learner but never fed to the policy
        # (no oracle leakage, no input-dim change). Gated by crd.enabled, so
        # non-CRD runs keep an unchanged observation space.
        crd_cfg = config.get("crd", {}) if isinstance(config, dict) else {}
        self.crd_enabled = bool(crd_cfg.get("enabled", False)) if isinstance(crd_cfg, dict) else False

        # Wrap the base hierarchical environment (no modifications to original)
        logger.info("Creating base HierarchicalMultiDCEnv...")
        base_env = HierarchicalMultiDCEnv(config=config)

        # # Wrap with wind prediction if enabled
        # base_env = self._wrap_with_prediction_if_enabled(base_env, config)

        self.base_env = base_env

        self.num_datacenters = self.base_env.num_datacenters
        self.global_routing_batch_size = self.base_env.global_routing_batch_size

        # Shared max dimensions from base env (already computed there)
        # These will be used to define a unified observation/action space
        # for all local agents to enable parameter sharing.
        self.max_hosts = getattr(self.base_env, "max_hosts", None)
        self.max_vms = getattr(self.base_env, "max_vms", None)
        if self.max_hosts is None or self.max_vms is None:
            # Fallback: derive from per-DC counts if max_* are not available
            dc_host_counts = [self.base_env._get_dc_host_count(i) for i in range(self.num_datacenters)]
            dc_vm_counts = [self.base_env._get_dc_vm_count(i) for i in range(self.num_datacenters)]
            self.max_hosts = max(dc_host_counts) if dc_host_counts else 1
            self.max_vms = max(dc_vm_counts) if dc_vm_counts else 1

        # Compute global_state_dim for CTDE (flattened global observation)
        self.global_state_dim = self._compute_global_state_dim()

        # Local action space in base env is Discrete(max_vms + 1)
        self.max_actions = getattr(self.base_env, "local_action_space", None)
        if self.max_actions is not None:
            self.max_actions = self.base_env.local_action_space.n
        else:
            self.max_actions = self.max_vms + 1

        # Define agent names (PettingZoo requirement: flat namespace)
        self.possible_agents = self._create_agent_list()
        self.agents = self.possible_agents.copy()

        # Define observation and action spaces for each agent
        self._observation_spaces = self._create_observation_spaces()
        self._action_spaces = self._create_action_spaces()

        # Store last observations for action masking
        self._last_observations = None
        self._obs_shape_ref = {}
        self._validate_obs_shapes = bool(config.get("validate_obs_shapes", False))

        logger.info(
            f"HierarchicalMultiDCParallelEnv initialized with {len(self.agents)} agents: "
            f"{self.agents}"
        )

    def _compute_global_state_dim(self) -> int:
        """Compute the flattened dimension of the global agent's observation space."""
        total = 0
        for space in self.base_env.global_observation_space.spaces.values():
            total += int(np.prod(space.shape))
        return total

    def _flatten_global_obs(self, global_obs: Dict[str, Any]) -> np.ndarray:
        """Flatten a global observation dict into a 1D float32 array."""
        parts = []
        for key in sorted(global_obs.keys()):
            val = np.asarray(global_obs[key], dtype=np.float32).flatten()
            parts.append(val)
        return np.concatenate(parts, axis=0)

    def set_lagrangian_lambda(self, new_lam: float) -> None:
        """Update the Lagrangian multiplier for this env (called from callback).

        Writes into the shared ``lagrangian`` dict so subsequent step() calls see
        the new λ.  Used by ``algorithm.workers.foreach_env`` when num_workers>0;
        with num_workers=0 the shared-dict reference handles propagation already,
        but this method is kept for symmetry.
        """
        if not isinstance(self._lagrangian_cfg, dict):
            self._lagrangian_cfg = {}
        self._lagrangian_cfg["lambda"] = float(max(0.0, new_lam))

    def _create_agent_list(self) -> List[str]:
        """
        Create flat list of agent names.

        Returns:
            List of agent names: ["global_agent", "local_agent_0", ...]
        """
        agents = ["global_agent"]
        agents.extend([f"local_agent_{i}" for i in range(self.num_datacenters)])
        return agents

    def _create_observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        Create observation space dict for all agents.

        For action masking support, observation space is a Dict containing:
        - "observation": the original observation space
        - "action_mask": binary mask of valid actions

        Returns:
            Dict mapping agent_name -> observation_space (Dict space with mask)
        """
        obs_spaces = {}

        # Global agent observation space with slot-level action mask.
        # action_mask shape equals global_routing_batch_size:
        # - 1.0: real cloudlet exists for that slot
        # - 0.0: padding slot (no cloudlet)
        global_agent_space = {
            "observation": self.base_env.global_observation_space,
            "action_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.global_routing_batch_size,),
                dtype=np.float32
            ),
        }
        if self.crd_enabled:
            global_agent_space["crd_aux"] = self._crd_aux_space()
        obs_spaces["global_agent"] = spaces.Dict(global_agent_space)

        # Local agents observation spaces
        # NOTE: We expose a UNIFIED padded observation space for all local agents
        # to enable parameter sharing in RLlib. Heterogeneity is represented via
        # dc_id_onehot and valid_vm_mask.
        local_obs_dict = {
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
            # Extra context features for parameter sharing
            "dc_id_onehot": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "valid_vm_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_vms,),
                dtype=np.float32
            ),
        }

        # Deferrable-batch temporal lever (2026-06-20): mirror the base env's
        # per-DC green-now + forecast into the wrapper's local obs so the local
        # dispatch-rate agent can decide hold-vs-run. Gated by dispatch_rate.
        if str(self.base_env.config.get("local_dispatch_mode", "vm_placement")).strip() == "dispatch_rate":
            for _k in ("green_now", "green_forecast_short", "green_forecast_long"):
                local_obs_dict[_k] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # CTDE: add global_state to local agent observation for centralized critic
        if self.ctde_enabled:
            local_obs_dict["global_state"] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.global_state_dim,),
                dtype=np.float32
            )
            logger.info(
                f"CTDE enabled: adding global_state (dim={self.global_state_dim}) "
                f"to local agent observations for centralized critic"
            )

        unified_local_obs_space = spaces.Dict(local_obs_dict)

        for i in range(self.num_datacenters):
            local_agent_space = {
                "observation": unified_local_obs_space,
                "action_mask": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.max_actions,),
                    dtype=np.float32
                ),
            }
            if self.crd_enabled:
                local_agent_space["crd_aux"] = self._crd_aux_space()
            obs_spaces[f"local_agent_{i}"] = spaces.Dict(local_agent_space)

        return obs_spaces

    def _create_action_spaces(self) -> Dict[str, spaces.Space]:
        """
        Create action space dict for all agents.

        Each DC gets its own action space sized to its actual VM count.
        This eliminates invalid actions without needing action masking.

        Returns:
            Dict mapping agent_name -> action_space
        """
        action_spaces = {
            "global_agent": self.base_env.global_action_space
        }

        # All local agents share the same (padded) action space:
        # Discrete(max_vms + 1). Valid actions per-DC are controlled by
        # the action_mask and valid_vm_mask in the observation.
        for i in range(self.num_datacenters):
            action_spaces[f"local_agent_{i}"] = spaces.Discrete(self.max_actions)
            logger.info(
                f"DC {i}: unified action space Discrete({self.max_actions}) "
                f"(base DC VMs: {self.base_env._get_dc_vm_count(i)})"
            )

        return action_spaces

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        PettingZoo API: Get observation spaces for all agents.

        Returns:
            Dict of observation spaces
        """
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """
        PettingZoo API: Get action spaces for all agents.

        Returns:
            Dict of action spaces
        """
        return self._action_spaces

    def observation_space(self, agent: str) -> spaces.Space:
        """
        Get observation space for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Observation space for this agent
        """
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """
        Get action space for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Action space for this agent
        """
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observations: Dict[agent_name, observation]
            infos: Dict[agent_name, info_dict]
        """
        logger.debug(f"Resetting PettingZoo environment (seed={seed})...")

        # Defensive reset of per-episode Lagrangian accumulators (terminal step
        # already clears them; this guards against partial episodes from
        # truncation paths that may bypass the terminal branch).
        self._ep_c_step_running_sum = 0.0
        self._ep_c_step_running_count = 0
        self._ep_lagrangian_penalty_sum = 0.0

        # Reset base environment
        hierarchical_obs, hierarchical_info = self.base_env.reset(seed=seed, options=options)

        # Convert hierarchical format to flat agent dict format (crd_aux pulled
        # from the same info["crd"] snapshot the base env just collected).
        observations = self._hierarchical_to_flat_observations(
            hierarchical_obs, crd_info=hierarchical_info.get("crd")
        )

        # Replicate info for all agents (or customize per agent if needed)
        infos = {agent: hierarchical_info.copy() for agent in self.agents}

        # Store for action masking
        self._last_observations = observations

        # Reset agent list (all agents are active)
        self.agents = self.possible_agents.copy()

        logger.info("PettingZoo environment reset complete")
        return observations, infos

    def step(
        self,
        actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, Dict]  # infos
    ]:
        """
        Execute one environment step with actions from all agents.

        Args:
            actions: Dict[agent_name, action]
                Example: {
                    "global_agent": np.array([0, 1, 2, 0, 1]),
                    "local_agent_0": 3,
                    "local_agent_1": 1,
                    "local_agent_2": 5
                }

        Returns:
            observations: Dict[agent_name, observation]
            rewards: Dict[agent_name, reward]
            terminations: Dict[agent_name, bool] - natural episode end
            truncations: Dict[agent_name, bool] - time limit reached
            infos: Dict[agent_name, info_dict]
        """
        # Convert flat agent actions to hierarchical format
        hierarchical_actions = self._flat_to_hierarchical_actions(actions)

        logger.debug(
            f"Step with actions: global={len(hierarchical_actions['global'])} cloudlets, "
            f"local={list(hierarchical_actions['local'].values())}"
        )

        # Execute step in base environment
        (
            hierarchical_obs,
            hierarchical_rewards,
            terminated,
            truncated,
            hierarchical_info
        ) = self.base_env.step(hierarchical_actions)

        # Convert results to PettingZoo format (crd_aux from this step's
        # info["crd"] snapshot, delivered via obs for stable learner alignment).
        observations = self._hierarchical_to_flat_observations(
            hierarchical_obs, crd_info=hierarchical_info.get("crd")
        )
        rewards = self._hierarchical_to_flat_rewards(hierarchical_rewards)

        # -------------------------------------------------------------------
        # Lagrangian SLA shaping on global agent reward only.
        #   r_train_global = r_step_global − λ · c_step
        # c_step comes from Java info (global_energy_stats.sla_cost_step).
        # λ lives in self._lagrangian_cfg["lambda"] (updated between training
        # iterations by LagrangianCallback).  Local rewards untouched.
        # -------------------------------------------------------------------
        lagrangian_penalty = 0.0
        c_step = 0.0
        lam = 0.0
        ges: Dict[str, Any] = {}
        if self._lagrangian_enabled:
            try:
                ges = hierarchical_info.get("global_energy_stats", {}) or {}
                c_step = float(ges.get("sla_cost_step", 0.0) or 0.0)
            except Exception:
                c_step = 0.0
            lam = float(self._lagrangian_cfg.get("lambda", 0.0) or 0.0)
            if lam > 0.0 and c_step > 0.0 and "global_agent" in rewards:
                lagrangian_penalty = lam * c_step
                rewards["global_agent"] = rewards["global_agent"] - lagrangian_penalty

            # Accumulate per-step c_step so we can report a per-episode mean.
            self._ep_c_step_running_sum += c_step
            self._ep_c_step_running_count += 1
            # Track per-episode total λ·c_step actually subtracted; surfaced
            # at episode end so monitor.csv can show post-Lagrangian reward.
            self._ep_lagrangian_penalty_sum += lagrangian_penalty

        # All agents share the same termination/truncation status
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Replicate info for all agents.  Attach Lagrangian diagnostics so
        # callbacks can log λ/penalty/c_step without recomputing them.
        base_info = dict(hierarchical_info)
        base_info["lagrangian_lambda"] = lam
        base_info["lagrangian_c_step"] = c_step
        base_info["lagrangian_penalty"] = lagrangian_penalty
        # On terminal step, expose per-episode c_step mean so the callback's
        # on_episode_end hook can drive the dual update without needing a
        # shared-dict channel (which doesn't survive env_config serialization).
        if self._lagrangian_enabled and (terminated or truncated):
            if self._ep_c_step_running_count > 0:
                base_info["lagrangian_c_step_mean_episode"] = (
                    self._ep_c_step_running_sum / self._ep_c_step_running_count
                )
            else:
                base_info["lagrangian_c_step_mean_episode"] = 0.0
            base_info["lagrangian_penalty_episode_sum"] = self._ep_lagrangian_penalty_sum
            self._ep_c_step_running_sum = 0.0
            self._ep_c_step_running_count = 0
            self._ep_lagrangian_penalty_sum = 0.0
        infos = {agent: base_info.copy() for agent in self.agents}

        # Store for action masking
        self._last_observations = observations

        logger.debug(
            f"Step result: rewards={[f'{k}:{v:.2f}' for k, v in rewards.items()]}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        return observations, rewards, terminations, truncations, infos

    def _crd_aux_space(self) -> spaces.Dict:
        """
        EU-CRD auxiliary observation channel (a sibling of "observation").

        Carries the raw per-DC wind/power/carbon snapshot the learner needs to
        compute R_forecast on the padded (B, T) grid — the same quantities
        `HierarchicalMultiDCEnv._collect_crd_info` puts in info["crd"], but
        delivered through obs (which PPO minibatching keeps aligned) instead of
        infos (which it does not). Never read by the policy networks.
        """
        n = self.num_datacenters
        box = lambda shape: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        return spaces.Dict({
            "crd_actual_green_w": box((n,)),
            "crd_predicted_green_w": box((n,)),
            "crd_total_power_w": box((n,)),
            "crd_green_factor": box((n,)),
            "crd_brown_factor": box((n,)),
            "crd_timestep_hours": box((1,)),
        })

    def _build_crd_aux(self, crd_info: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Pack a crd_info snapshot (from base_env._collect_crd_info, surfaced via
        info["crd"]) into the fixed-shape crd_aux obs dict. Missing fields
        default to zeros; a missing predicted_wind_w defaults to the actual
        wind so R_forecast = 0 (the correct "no forecast signal" attribution).
        """
        n = self.num_datacenters
        crd = crd_info if isinstance(crd_info, dict) else {}

        def _vec(key, default=None):
            v = crd.get(key)
            if v is None:
                return (np.zeros(n, dtype=np.float32) if default is None
                        else np.asarray(default, dtype=np.float32))
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.shape[0] < n:
                arr = np.concatenate([arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
            return arr[:n]

        actual = _vec("actual_wind_w")
        # predicted defaults to actual → carbon(actual)==carbon(pred) → R_f=0.
        predicted = _vec("predicted_wind_w", default=actual)
        dt = float(crd.get("timestep_hours", 0.0) or 0.0)
        return {
            "crd_actual_green_w": actual,
            "crd_predicted_green_w": predicted,
            "crd_total_power_w": _vec("p_total_w"),
            "crd_green_factor": _vec("green_carbon_factor"),
            "crd_brown_factor": _vec("brown_carbon_factor"),
            "crd_timestep_hours": np.asarray([dt], dtype=np.float32),
        }

    def _hierarchical_to_flat_observations(
        self,
        hierarchical_obs: Dict[str, Any],
        crd_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert hierarchical observation format to flat agent dict with action masks.

        For RLlib action masking support, each observation is a dict with:
        - "observation": the actual observation array
        - "action_mask": binary mask of valid actions (1=valid, 0=invalid)

        Args:
            hierarchical_obs: {
                "global": {...},
                "local": {0: {...}, 1: {...}, ...}
            }

        Returns:
            {
                "global_agent": {"observation": ..., "action_mask": ...},
                "local_agent_0": {"observation": ..., "action_mask": ...},
                ...
            }
        """
        flat_obs = {}

        # EU-CRD: one shared snapshot for all agents (system-level wind/carbon).
        crd_aux = self._build_crd_aux(crd_info) if self.crd_enabled else None

        # Global agent observation with slot-level mask for MultiDiscrete routing.
        global_obs = hierarchical_obs["global"]
        try:
            global_action_mask = self.base_env.get_global_action_mask(global_obs)
            if global_action_mask.shape[0] != self.global_routing_batch_size:
                logger.warning(
                    "global_action_mask length %d != expected %d, padding/trimming accordingly",
                    global_action_mask.shape[0],
                    self.global_routing_batch_size,
                )
                fixed_mask = np.zeros(self.global_routing_batch_size, dtype=np.float32)
                copy_len = min(self.global_routing_batch_size, global_action_mask.shape[0])
                fixed_mask[:copy_len] = global_action_mask[:copy_len]
                global_action_mask = fixed_mask
            else:
                global_action_mask = global_action_mask.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to get global action mask: {e}")
            global_action_mask = np.ones(self.global_routing_batch_size, dtype=np.float32)

        flat_obs["global_agent"] = {
            "observation": global_obs,
            "action_mask": global_action_mask,
        }
        if crd_aux is not None:
            flat_obs["global_agent"]["crd_aux"] = {k: v.copy() for k, v in crd_aux.items()}

        # CTDE: pre-compute flattened global state once for all local agents
        if self.ctde_enabled:
            flat_global_state = self._flatten_global_obs(global_obs)

        # Local agents observations with action masks (UNIFIED padded format)
        for dc_id_raw, local_obs in hierarchical_obs["local"].items():
            # Ensure dc_id is Python int (Java may return Integer object)
            dc_id = int(dc_id_raw)
            agent_name = f"local_agent_{dc_id}"

            dc_vm_count = self.base_env._get_dc_vm_count(dc_id)
            dc_host_count = self.base_env._get_dc_host_count(dc_id)

            # Build valid_vm_mask (1 for real VMs, 0 for padding)
            valid_vm_mask = np.zeros(self.max_vms, dtype=np.float32)
            valid_vm_mask[:dc_vm_count] = 1.0

            # DC ID one-hot
            dc_id_onehot = np.zeros(self.num_datacenters, dtype=np.float32)
            if 0 <= dc_id < self.num_datacenters:
                dc_id_onehot[dc_id] = 1.0

            # Use padded observations directly from base_env (already length max_*)
            unified_obs = {
                "host_loads": local_obs["host_loads"],
                "host_ram_usage": local_obs["host_ram_usage"],
                "vm_loads": local_obs["vm_loads"],
                "vm_types": local_obs["vm_types"],
                "vm_available_pes": local_obs["vm_available_pes"],
                "waiting_cloudlets": local_obs["waiting_cloudlets"],
                "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
                "dc_id_onehot": dc_id_onehot,
                "valid_vm_mask": valid_vm_mask,
            }

            # Deferrable-batch lever: pass through the green-now + forecast features
            # the base env injected (dispatch_rate only; keys absent otherwise).
            for _k in ("green_now", "green_forecast_short", "green_forecast_long"):
                if _k in local_obs:
                    unified_obs[_k] = local_obs[_k]

            # CTDE: include global state for centralized critic
            if self.ctde_enabled:
                unified_obs["global_state"] = flat_global_state

            # Get action mask for this local agent from base env (already size max_vms+1)
            try:
                action_mask = self.base_env.get_local_action_masks(dc_id)
                # Ensure correct length and dtype
                if action_mask.shape[0] != self.max_actions:
                    logger.warning(
                        f"{agent_name}: base_env action_mask length {action_mask.shape[0]} "
                        f"!= expected {self.max_actions}, padding/trimming accordingly"
                    )
                    padded_mask = np.zeros(self.max_actions, dtype=bool)
                    copy_len = min(self.max_actions, action_mask.shape[0])
                    padded_mask[:copy_len] = action_mask[:copy_len]
                    action_mask = padded_mask
                action_mask = action_mask.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to get action mask for {agent_name}: {e}")
                # Fallback: allow all actions (with unified size)
                action_mask = np.ones(self.max_actions, dtype=np.float32)

            flat_obs[agent_name] = {
                "observation": unified_obs,
                "action_mask": action_mask,
            }
            if crd_aux is not None:
                flat_obs[agent_name]["crd_aux"] = {k: v.copy() for k, v in crd_aux.items()}

        if self._validate_obs_shapes:
            for agent, obs in flat_obs.items():
                self._check_obs_shapes(agent, obs)

        return flat_obs

    def _check_obs_shapes(self, agent: str, obs: Dict[str, Any]) -> None:
        """Validate observation shapes are consistent across timesteps."""
        def _shape_of(value):
            if isinstance(value, dict):
                return {k: _shape_of(v) for k, v in value.items()}
            arr = np.asarray(value)
            return arr.shape

        current = _shape_of(obs)
        if agent not in self._obs_shape_ref:
            self._obs_shape_ref[agent] = current
            logger.info("[ObsShapeRef] %s -> %s", agent, current)
            return

        if current != self._obs_shape_ref[agent]:
            logger.error(
                "[ObsShapeMismatch] %s expected=%s got=%s",
                agent,
                self._obs_shape_ref[agent],
                current,
            )
            raise ValueError(f"Observation shape mismatch for {agent}")

    def _flat_to_hierarchical_actions(
        self,
        flat_actions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert flat agent actions to hierarchical format.

        Args:
            flat_actions: {
                "global_agent": action,
                "local_agent_0": action,
                "local_agent_1": action,
                ...
            }

        Returns:
            {
                "global": action,
                "local": {0: action, 1: action, ...}
            }
        """
        # Extract global action
        global_action = flat_actions.get("global_agent")

        # Extract local actions with safety checks (avoid selecting padding-only VMs)
        # NOTE: in dispatch_rate mode the local action is a RELEASE COUNT (0..max_dispatch),
        # NOT a VM index, so the vm_count clamp below MUST NOT apply (it would corrupt the
        # temporal lever — e.g. release-61 wrongly clamped to 0/hold on a 53-VM DC).
        dispatch_rate_mode = str(
            self.base_env.config.get("local_dispatch_mode", "vm_placement")).strip() == "dispatch_rate"
        local_actions = {}
        for i in range(self.num_datacenters):
            agent_name = f"local_agent_{i}"
            if agent_name in flat_actions:
                raw_action = int(flat_actions[agent_name])

                if not dispatch_rate_mode:
                    dc_vm_count = self.base_env._get_dc_vm_count(i)
                    # Safety check (vm_placement only): action must be within [0, dc_vm_count]
                    if raw_action > dc_vm_count:
                        logger.warning(
                            f"Action {raw_action} for {agent_name} exceeds vm_count={dc_vm_count}, "
                            f"clamping to 0 (NoAssign)."
                        )
                        raw_action = 0  # Fallback to NoAssign

                local_actions[i] = raw_action

        hierarchical_actions = {
            "global": global_action,
            "local": local_actions,
        }

        return hierarchical_actions

    def _hierarchical_to_flat_rewards(
        self,
        hierarchical_rewards: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Convert hierarchical reward format to flat agent dict.

        Args:
            hierarchical_rewards: {
                "global": reward,
                "local": {0: reward, 1: reward, ...}
            }

        Returns:
            {
                "global_agent": reward,
                "local_agent_0": reward,
                "local_agent_1": reward,
                ...
            }
        """
        flat_rewards = {
            "global_agent": float(hierarchical_rewards["global"])
        }

        for dc_id, reward in hierarchical_rewards["local"].items():
            agent_name = f"local_agent_{dc_id}"
            flat_rewards[agent_name] = float(reward)

        return flat_rewards

    def get_action_mask(self, agent: str) -> Optional[np.ndarray]:
        """
        Get action mask for a specific agent.

        This method provides action masking information for invalid actions,
        which is useful for masked PPO algorithms.

        Args:
            agent: Agent name

        Returns:
            Boolean array where True = valid action, False = invalid action
            Returns None for global agent (no masking needed)
        """
        if agent == "global_agent":
            # Global agent doesn't need action masking
            return None

        if self._last_observations is None:
            logger.warning(f"Action mask requested for {agent} before first observation")
            return None

        # Extract DC ID from agent name
        if not agent.startswith("local_agent_"):
            logger.warning(f"Unknown agent for action masking: {agent}")
            return None

        try:
            dc_id = int(agent.split("_")[-1])
        except (ValueError, IndexError):
            logger.error(f"Invalid agent name format: {agent}")
            return None

        # Get mask from base environment
        try:
            mask = self.base_env.get_local_action_masks(dc_id)
            return mask
        except Exception as e:
            logger.error(f"Failed to get action mask for {agent}: {e}")
            return None

    def get_all_action_masks(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get action masks for all agents.

        Returns:
            Dict mapping agent_name -> action_mask
        """
        masks = {}

        for agent in self.agents:
            masks[agent] = self.get_action_mask(agent)

        return masks

    def render(self) -> Optional[Any]:
        """
        Render the environment.

        Returns:
            Render output (format depends on render_mode)
        """
        if self.render_mode is None:
            return None

        # Delegate to base environment
        return self.base_env.render()

    def close(self):
        """
        Close environment and cleanup resources.
        """
        logger.info("Closing PettingZoo environment...")
        if hasattr(self, 'base_env') and self.base_env is not None:
            self.base_env.close()
        logger.info("PettingZoo environment closed")

    def state(self) -> np.ndarray:
        """
        Get global state (optional method for centralized training).

        Returns:
            Global state array combining all observations
        """
        if self._last_observations is None:
            return np.array([])

        # Concatenate global and all local observations
        state_parts = []

        # Add global observation
        global_obs = self._last_observations.get("global_agent", {})
        state_parts.extend(self._flatten_observation(global_obs))

        # Add local observations
        for i in range(self.num_datacenters):
            local_obs = self._last_observations.get(f"local_agent_{i}", {})
            state_parts.extend(self._flatten_observation(local_obs))

        return np.array(state_parts, dtype=np.float32)

    def _flatten_observation(self, obs: Dict[str, Any]) -> List[float]:
        """
        Flatten a dictionary observation into a 1D list.

        Args:
            obs: Observation dictionary

        Returns:
            Flattened list of values
        """
        values = []

        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                values.extend(value.flatten().tolist())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))

        return values

    def _wrap_with_prediction_if_enabled(
        self,
        base_env: Any,
        config: Dict[str, Any]
    ) -> Any:
        """
        Wrap environment with wind prediction if enabled in config.

        Args:
            base_env: Base environment to wrap
            config: Configuration dictionary

        Returns:
            Wrapped or original environment
        """
        wind_pred_config = config.get('wind_prediction', {})

        if not wind_pred_config.get('enabled', False):
            logger.info("Wind prediction disabled (config: wind_prediction.enabled = false)")
            return base_env

        logger.info("Wrapping environment with wind power prediction...")

        # Import here to avoid circular dependency
        from gym_cloudsimplus.wrappers import WindPredictionWrapper

        # Parse turbine_csv_paths (REQUIRED - convert list of dicts to dict)
        csv_paths_config = wind_pred_config.get('turbine_csv_paths')
        if csv_paths_config is None:
            logger.error(
                "Wind prediction enabled but 'turbine_csv_paths' not configured! "
                "Please add turbine_csv_paths to config.yml under wind_prediction section."
            )
            raise ValueError("turbine_csv_paths is required when wind_prediction is enabled")

        if isinstance(csv_paths_config, dict):
            # Already a dict, ensure keys are ints
            turbine_csv_paths = {int(k): v for k, v in csv_paths_config.items()}
        elif isinstance(csv_paths_config, list):
            # List of {turbine_id: path} dicts, merge them
            turbine_csv_paths = {}
            for item in csv_paths_config:
                turbine_csv_paths.update({int(k): v for k, v in item.items()})
        else:
            raise ValueError(f"Invalid turbine_csv_paths format: {type(csv_paths_config)}")

        wrapped_env = WindPredictionWrapper(
            env=base_env,
            model_checkpoint=wind_pred_config.get('model_checkpoint'),
            scalers_path=wind_pred_config.get('scalers_path'),
            data_path=wind_pred_config.get('data_path'),
            turbine_ids=wind_pred_config.get('turbine_ids', [1, 57, 124]),
            turbine_csv_paths=turbine_csv_paths,
            prediction_horizon=wind_pred_config.get('horizon', 8),
            device=wind_pred_config.get('device', 'cpu'),
            enable_logging=wind_pred_config.get('enable_logging', True),
            csv_start_offset=wind_pred_config.get('csv_start_offset', 12)
        )

        logger.info(
            f"Wind prediction enabled for PettingZoo environment: "
            f"horizon={wind_pred_config.get('horizon', 8)}, "
            f"turbines={wind_pred_config.get('turbine_ids', [1, 57, 124])}, "
            f"mode=13-feature CSV"
        )

        return wrapped_env


# Convenience function for creating the environment
def make_env(config: Dict[str, Any], **kwargs) -> HierarchicalMultiDCParallelEnv:
    """
    Factory function to create PettingZoo environment.

    Args:
        config: Environment configuration dictionary
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        HierarchicalMultiDCParallelEnv instance
    """
    return HierarchicalMultiDCParallelEnv(config=config, **kwargs)
