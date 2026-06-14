"""
PettingZoo ParallelEnv wrapper for HierarchicalMultiDCEnvAblation.

Mirrors the HierarchicalMultiDCParallelEnvSimple pattern: re-uses the parent
wrapper's MARL plumbing (agent list, action spaces, action-mask logic) but
swaps the base env for the ablation env so ``forecast_mode`` controls the
global observation block.

Used by src/training/train_rlmodule_gtrxl.py via env_id matching:
    "Ablation" in env_id  →  this class
"""

import logging
from typing import Any, Dict, Optional

from .hierarchical_multidc_env_ablation import HierarchicalMultiDCEnvAblation
from .hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv

logger = logging.getLogger(__name__)


class HierarchicalMultiDCParallelEnvAblation(HierarchicalMultiDCParallelEnv):
    """
    PettingZoo wrapper that uses HierarchicalMultiDCEnvAblation as the base env.
    The ``forecast_mode`` config knob controls which future-block keys appear
    in the global observation.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "hierarchical_multidc_ablation_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        from pettingzoo import ParallelEnv

        # NOTE: We replicate the parent's __init__ body verbatim (with the only
        # change being the swap to HierarchicalMultiDCEnvAblation as base_env)
        # rather than calling super().__init__(). The parent's __init__
        # instantiates HierarchicalMultiDCEnv directly, so super() can't be
        # reused. Any new field the parent sets must be mirrored here.
        ParallelEnv.__init__(self)
        self.render_mode = render_mode
        self.config = config

        # Lagrangian state — see parent for full rationale.
        lagrangian_cfg = config.get("lagrangian", {}) if isinstance(config, dict) else {}
        self._lagrangian_cfg = lagrangian_cfg if isinstance(lagrangian_cfg, dict) else {}
        self._lagrangian_enabled = bool(self._lagrangian_cfg.get("enabled", False))
        self._ep_c_step_running_sum = 0.0
        self._ep_c_step_running_count = 0
        self._ep_lagrangian_penalty_sum = 0.0

        # CTDE flag — drives whether ``global_state`` is added to each local
        # agent's observation in ``_create_observation_spaces``.
        ctde_cfg = config.get("ctde", {})
        self.ctde_enabled = (
            bool(ctde_cfg.get("enabled", False)) if isinstance(ctde_cfg, dict) else bool(ctde_cfg)
        )

        # EU-CRD flag — mirrors the parent. `_create_observation_spaces` reads
        # `self.crd_enabled` to decide whether to add the `crd_aux` sibling
        # key; without setting it here the ablation wrapper raises
        # AttributeError before the obs spaces are built. Gated by crd.enabled
        # (False for the carbon v2 ablation runs), so non-CRD runs unaffected.
        crd_cfg = config.get("crd", {}) if isinstance(config, dict) else {}
        self.crd_enabled = bool(crd_cfg.get("enabled", False)) if isinstance(crd_cfg, dict) else False

        forecast_mode = str(config.get("forecast_mode", "full")).lower()
        logger.info(
            "Creating base HierarchicalMultiDCEnvAblation (forecast_mode=%s)...",
            forecast_mode,
        )
        base_env = HierarchicalMultiDCEnvAblation(config=config)
        base_env = self._wrap_with_prediction_if_enabled(base_env, config)
        self.base_env = base_env

        self.num_datacenters = self.base_env.num_datacenters
        self.global_routing_batch_size = self.base_env.global_routing_batch_size

        self.max_hosts = getattr(self.base_env, "max_hosts", None)
        self.max_vms = getattr(self.base_env, "max_vms", None)
        if self.max_hosts is None or self.max_vms is None:
            dc_host_counts = [
                self.base_env._get_dc_host_count(i) for i in range(self.num_datacenters)
            ]
            dc_vm_counts = [
                self.base_env._get_dc_vm_count(i) for i in range(self.num_datacenters)
            ]
            self.max_hosts = max(dc_host_counts) if dc_host_counts else 1
            self.max_vms = max(dc_vm_counts) if dc_vm_counts else 1

        # CTDE flattened-global-state dimension — must be set before
        # _create_observation_spaces() because that method reads it.
        self.global_state_dim = self._compute_global_state_dim()

        self.max_actions = getattr(self.base_env, "local_action_space", None)
        if self.max_actions is not None:
            self.max_actions = self.base_env.local_action_space.n
        else:
            self.max_actions = self.max_vms + 1

        self.possible_agents = self._create_agent_list()
        self.agents = self.possible_agents.copy()

        self._observation_spaces = self._create_observation_spaces()
        self._action_spaces = self._create_action_spaces()

        self._last_observations = None
        self._obs_shape_ref = {}
        self._validate_obs_shapes = bool(config.get("validate_obs_shapes", False))

        logger.info(
            "HierarchicalMultiDCParallelEnvAblation initialized: forecast_mode=%s | "
            "agents=%d | ctde=%s",
            forecast_mode,
            len(self.agents),
            self.ctde_enabled,
        )
