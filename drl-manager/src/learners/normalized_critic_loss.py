"""
P1 critic fix: PPO learner whose vf loss is normalized by Var(VALUE_TARGETS).

Why (run 20260526_055502 post-mortem): the global module's value targets sit
at a scale where the residuals² reach ~4000-10000, while `vf_clip_param`
clamps the squared error at a small constant (10 by default, 3000 in the
June-10 attempt). `torch.clamp`'s gradient is zero in the clipped region, so
the global critic received (near-)zero gradient for entire runs →
`vf_explained_var ≈ 0` → advantages were noise → the global actor never left
the uniform-random policy. The local module's residuals² (~1-7) never hit the
clamp, which is why the local critic learned fine on the same trace.

Fix (surgical, critic-side only — the new API stack does not standardize
advantages, so rescaling rewards or advantages would shrink actor gradients):

    var_G   ← EMA( Var(masked VALUE_TARGETS) )       # per-module, detached
    vf_loss = (V − G)² / max(var_G, eps)              # O(1) scale
    clamp(vf_loss, 0, vf_clip_param)                  # clip now in σ² units

Everything else in the PPO loss (surrogate, entropy, KL, GAE, value-head
architecture) is byte-identical to `PPOTorchLearner.compute_loss_for_module`.

Config, read from the module's `model_config["normalized_critic"]`:
    enabled:   bool = False   # per-module gate; when off, the vf term is
                              # bit-identical to base PPO (raw squared error,
                              # absolute-unit clip). Default-off so that
                              # subclasses (CRDPPOTorchLearner) and old
                              # configs keep their exact previous behavior.
    ema_decay: float = 0.99   # EMA decay for the running target variance
    var_eps:   float = 1e-8   # denominator floor (degenerate constant targets)

The trainer (`create_rlmodule_config`) injects this block from the
experiment-level `normalized_critic:` config — global module by default,
local module only when `normalized_critic.local: true` (the local critic is
healthy and serves as the reference signal during the P1 smoke).

Note on `vf_loss_coeff`: the global module's ×10 vf_coef was a hack to fight
the scale mismatch; with the loss normalized to O(1) it overshoots — revert
to 1.0 when enabling this learner.

Note on checkpoint restore: the EMA is intentionally NOT persisted in learner
state. It re-initializes from the first post-restore batch's variance (one
minibatch of warm-up), which is within the estimator's own noise.

`CRDPPOTorchLearner` inherits from this class, so the EU-CRD Q-head loss and
all CRD diagnostics run on top of the normalized critic automatically.
"""

import logging
from typing import Any, Dict, Optional

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import ModuleID, TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

_DEFAULT_EMA_DECAY = 0.99
_DEFAULT_VAR_EPS = 1e-8

# Extra per-module metrics so the absolute critic error stays visible after
# normalization (the normalized vf_loss alone can't distinguish "critic got
# better" from "targets got wider").
LEARNER_RESULTS_VF_TARGET_VAR_EMA_KEY = "vf_target_var_ema"
LEARNER_RESULTS_VF_LOSS_RAW_MSE_KEY = "vf_loss_raw_mse"


class NormalizedCriticPPOTorchLearner(PPOTorchLearner):
    """PPOTorchLearner with the vf term divided by EMA-Var(VALUE_TARGETS)."""

    def build(self) -> None:
        super().build()
        # Per-module running variance of the value targets. Plain python
        # floats: detached by construction, never part of the autograd graph.
        self._vf_target_var_ema: Dict[ModuleID, float] = {}

    # ------------------------------------------------------------------
    # Running-variance bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_var(
        targets: TensorType, mask: Optional[TensorType]
    ) -> float:
        """Population variance of the valid VALUE_TARGETS, as a float."""
        vals = targets[mask] if mask is not None else targets
        vals = vals.detach().float()
        if vals.numel() < 2:
            return 0.0
        return float(vals.var(unbiased=False).item())

    def _update_target_var_ema(
        self, module_id: ModuleID, var_now: float, decay: float
    ) -> float:
        """EMA update; the first observation seeds the EMA directly (an
        EMA started at 0 would under-normalize for hundreds of batches)."""
        prev = self._vf_target_var_ema.get(module_id)
        ema = var_now if prev is None else decay * prev + (1.0 - decay) * var_now
        self._vf_target_var_ema[module_id] = ema
        return ema

    def _read_module_norm_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `normalized_critic` from the module's model_config."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return mcfg.get("normalized_critic", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        # Copy of PPOTorchLearner.compute_loss_for_module (ray 2.4x new API
        # stack); the only change is the vf term — marked below.
        module = self.module[module_id].unwrapped()

        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data_):
                return torch.sum(data_[mask]) / num_valid

        else:
            mask = None
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()

        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )

        if config.use_kl_loss:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES]
            * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        if config.use_critic:
            value_fn_out = module.compute_values(
                batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
            )
            # --- CHANGED vs base PPO: when enabled for this module, normalize
            # --- the squared error by the running variance of the targets
            # --- before clipping, so the clamp operates in σ² units instead
            # --- of absolute units. Disabled (default) → denom=1 → the vf
            # --- term is bit-identical to base PPO.
            norm_cfg = self._read_module_norm_config(module_id)
            targets = batch[Postprocessing.VALUE_TARGETS]
            vf_err2 = torch.pow(value_fn_out - targets, 2.0)
            if norm_cfg.get("enabled", False):
                decay = float(norm_cfg.get("ema_decay", _DEFAULT_EMA_DECAY))
                var_eps = float(norm_cfg.get("var_eps", _DEFAULT_VAR_EPS))
                var_ema = self._update_target_var_ema(
                    module_id, self._masked_var(targets, mask), decay
                )
                denom = max(var_ema, var_eps)
            else:
                var_ema = 0.0
                denom = 1.0
            vf_loss = vf_err2 / denom
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
            mean_vf_raw_mse = possibly_masked_mean(vf_err2)
            # --- end change
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z
            mean_vf_raw_mse = z
            var_ema = 0.0

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
                * curr_entropy
            )
        )

        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
                LEARNER_RESULTS_VF_TARGET_VAR_EMA_KEY: var_ema,
                LEARNER_RESULTS_VF_LOSS_RAW_MSE_KEY: mean_vf_raw_mse,
            },
            key=module_id,
            window=1,
        )
        return total_loss
