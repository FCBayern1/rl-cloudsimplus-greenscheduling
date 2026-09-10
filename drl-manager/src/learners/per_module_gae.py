"""Per-module GAE parameters.

RLlib's `GeneralAdvantageEstimation` connector stores one `gamma` / `lambda_` at construction
and applies them to every module in its loop, so `algorithm_config_overrides_per_module` never
reaches the advantage computation: a global policy configured with gamma 0.999 / lambda 0.98
silently ran with the algorithm-level 0.99 / 0.95 taken from the local policy
(reports/RL_POLICY_DEGRADATION_INVESTIGATION_2026_09_10.md).

`ModuleAwareGAE` looks the two values up per module through
`AlgorithmConfig.get_config_for_module(module_id)`, falling back to the algorithm-level values
when a module has no override. `install_module_aware_gae(pipeline, config)` swaps the stock
connector for it inside an already-built learner pipeline, so nothing else in the pipeline
changes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)

logger = logging.getLogger(__name__)


class ModuleAwareGAE(GeneralAdvantageEstimation):
    """GAE that honours per-module gamma / lambda overrides."""

    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._algo_config = config
        self._default = (self.gamma, self.lambda_)
        self._resolved: Dict[str, tuple] = {}

    def params_for_module(self, module_id: str) -> tuple:
        if module_id in self._resolved:
            return self._resolved[module_id]
        gamma, lambda_ = self._default
        cfg = self._algo_config
        if cfg is not None:
            try:
                mcfg = cfg.get_config_for_module(module_id)
                gamma = getattr(mcfg, "gamma", gamma)
                lambda_ = getattr(mcfg, "lambda_", lambda_)
            except Exception as e:  # a missing module config must not stop training
                logger.warning("[GAE] no per-module config for %r (%s); using %s",
                               module_id, e, self._default)
        self._resolved[module_id] = (gamma, lambda_)
        if (gamma, lambda_) != self._default:
            logger.info("[GAE] module %r uses gamma=%s lambda=%s (algorithm default %s)",
                        module_id, gamma, lambda_, self._default)
        return self._resolved[module_id]

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        # The stock implementation reads self.gamma / self.lambda_ inside its per-module loop.
        # Rather than copy that loop, run it once per module with the module's own values.
        module_ids = [mid for mid in rl_module.keys() if mid in batch]
        if len(module_ids) <= 1:
            if module_ids:
                self.gamma, self.lambda_ = self.params_for_module(module_ids[0])
            return super().__call__(rl_module=rl_module, batch=batch, episodes=episodes, **kwargs)

        saved = (self.gamma, self.lambda_)
        try:
            for mid in module_ids:
                self.gamma, self.lambda_ = self.params_for_module(mid)
                sub_batch = {mid: batch[mid]}
                super().__call__(rl_module=rl_module, batch=sub_batch, episodes=episodes, **kwargs)
                batch[mid] = sub_batch[mid]
        finally:
            self.gamma, self.lambda_ = saved
        return batch


def install_module_aware_gae(pipeline: Any, config: Optional[Any]) -> bool:
    """Replace a stock GAE connector inside `pipeline` with the module-aware one.

    Returns True when a connector was swapped. Idempotent: an already-installed
    ModuleAwareGAE is left alone.
    """
    conns = getattr(pipeline, "connectors", None)
    if conns is None:
        return False
    for i, c in enumerate(list(conns)):
        if isinstance(c, ModuleAwareGAE):
            return True
        if isinstance(c, GeneralAdvantageEstimation):
            new = ModuleAwareGAE(gamma=c.gamma, lambda_=c.lambda_, config=config)
            conns[i] = new
            logger.info("[GAE] module-aware GAE installed (defaults gamma=%s lambda=%s)",
                        c.gamma, c.lambda_)
            return True
    return False
