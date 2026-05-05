"""
M1.2: PPO learner extension that trains the Q-head ensemble via off-policy
TD bootstrap, with per-sample bootstrap masking (Osband et al., 2016).

Design:
- Subclasses RLlib's `PPOTorchLearner`. The base PPO loss (policy + V-head
  + entropy + KL) is unchanged. We only ADD an extra term:
      total_loss = base_ppo_loss + q_loss_coef * q_loss
- The Q-head TD target is `batch[Postprocessing.VALUE_TARGETS]`, which is
  RLlib's GAE n-step return (≈ Q(s_t, a_t) for the actually-taken action).
  This avoids any need to manually compute `r + γ V(s')`: we simply reuse
  what GAE already produced.
- For each (sample, head), a Bernoulli(p=`bootstrap_p`) mask decides whether
  that head sees this transition. This keeps the K heads from collapsing to
  identical predictions.
- `q_train_every_n_iters` lets us skip Q-loss on certain calls if its cost
  becomes a bottleneck.

Modules that don't carry a `crd_q_ensemble` key in their forward output (e.g.,
non-ensemble RLModules) are treated as no-ops here — their loss is whatever
the base learner returned.
"""

from typing import Any, Dict, Optional

import torch

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.typing import ModuleID, TensorType

from src.models.rlmodule_gtrxl_ensemble import COL_Q_ENSEMBLE


_DEFAULT_Q_LOSS_COEF = 0.5
_DEFAULT_BOOTSTRAP_P = 0.7
_DEFAULT_TRAIN_EVERY = 1


class CRDPPOTorchLearner(PPOTorchLearner):
    """
    PPO learner with EU-CRD Q-head TD loss bolted on.

    Reads the per-module Q-loss config from `module.model_config["crd"]
    ["ensemble"]`; falls back to defaults if absent. Modules that don't
    expose `crd_q_ensemble` get only the base PPO loss.
    """

    def build(self) -> None:
        super().build()
        # Per-module call counter for `q_train_every_n_iters` gating.
        self._crd_call_counts: Dict[ModuleID, int] = {}

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        base_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        q_ensemble = fwd_out.get(COL_Q_ENSEMBLE)
        if q_ensemble is None:
            return base_loss  # Non-ensemble module — no Q-loss contribution.

        crd_cfg = self._read_module_crd_config(module_id)
        coef = float(crd_cfg.get("q_loss_coef", _DEFAULT_Q_LOSS_COEF))
        bootstrap_p = float(crd_cfg.get("bootstrap_p", _DEFAULT_BOOTSTRAP_P))
        train_every = max(1, int(crd_cfg.get("q_train_every_n_iters", _DEFAULT_TRAIN_EVERY)))

        # Skip-by-counter: keep call cadence sparse when knob > 1.
        n = self._crd_call_counts.get(module_id, 0)
        self._crd_call_counts[module_id] = n + 1
        if (n % train_every) != 0:
            return base_loss

        q_loss = self._compute_q_loss(q_ensemble, batch, bootstrap_p)
        return base_loss + coef * q_loss

    # ------------------------------------------------------------------ helpers

    def _read_module_crd_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.ensemble` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("ensemble", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def _compute_q_loss(
        q_ensemble: torch.Tensor,
        batch: Dict[str, Any],
        bootstrap_p: float,
    ) -> torch.Tensor:
        """
        TD-style MSE loss with per-sample bootstrap masking.

        Args:
            q_ensemble: from `_forward_train`; shape is one of
                - local agent (Discrete):       (B, T, K, A)
                - global agent (MultiDiscrete): (B, T, K, batch_size, num_dc)
            batch: must contain `Columns.ACTIONS` and `Postprocessing.VALUE_TARGETS`.
            bootstrap_p: Bernoulli probability for each (sample, head) inclusion.

        Returns:
            Scalar tensor (mean over included entries).
        """
        target = batch[Postprocessing.VALUE_TARGETS].detach()  # (B, T) or (B,)
        actions = batch[Columns.ACTIONS]                       # (B, T) or (B, T, batch_size)

        if q_ensemble.dim() == 4:
            # ----- Local agent: (B, T, K, A) -----
            B, T, K, A = q_ensemble.shape
            # Bring action to (B, T, K, 1) for gather along last dim.
            act = actions.long()
            if act.dim() == 1:
                act = act.unsqueeze(-1)  # (B, 1) — assume T==1
            idx = act.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
            q_chosen = q_ensemble.gather(-1, idx).squeeze(-1)  # (B, T, K)

        elif q_ensemble.dim() == 5:
            # ----- Global agent: (B, T, K, batch_size, num_dc) -----
            B, T, K, bs, nd = q_ensemble.shape
            act = actions.long()
            if act.dim() == 2:
                act = act.unsqueeze(1)  # (B, 1, batch_size) — assume T==1
            # Per-cloudlet gather → (B, T, K, batch_size)
            idx = act.unsqueeze(2).unsqueeze(-1).expand(-1, -1, K, -1, 1)
            q_per_cloudlet = q_ensemble.gather(-1, idx).squeeze(-1)
            # Aggregate per-cloudlet Q to a scalar Q for this routing decision
            # (mean keeps magnitude independent of batch_size, matching the
            # `compute_q_ensemble` API used by the M2 callback).
            q_chosen = q_per_cloudlet.mean(dim=-1)  # (B, T, K)

        else:
            raise RuntimeError(
                f"Unexpected q_ensemble dim {q_ensemble.dim()} (shape "
                f"{tuple(q_ensemble.shape)}); expected 4 (local) or 5 (global)."
            )

        # Align target shape to (B, T, 1) for broadcast against (B, T, K).
        if target.dim() == 1:
            target = target.unsqueeze(-1)  # (B, 1)
        target = target.unsqueeze(-1).expand_as(q_chosen)  # (B, T, K)

        # Per-sample bootstrap mask: shape (B, K), broadcast over T.
        B = q_chosen.shape[0]
        K = q_chosen.shape[-1]
        mask = (torch.rand(B, K, device=q_chosen.device) < bootstrap_p).float()
        # Ensure no division by zero: if a sample has zero mask everywhere,
        # force include head 0 for that sample.
        any_per_sample = (mask.sum(dim=1) > 0).float()  # (B,)
        if (any_per_sample < 1.0).any():
            mask[any_per_sample < 1.0, 0] = 1.0
        mask = mask.unsqueeze(1)  # (B, 1, K) → broadcast over T

        sq_err = (q_chosen - target).pow(2)               # (B, T, K)
        masked = sq_err * mask                             # (B, T, K)
        denom = mask.sum() * sq_err.shape[1]               # active heads × T
        return masked.sum() / (denom + 1e-8)
