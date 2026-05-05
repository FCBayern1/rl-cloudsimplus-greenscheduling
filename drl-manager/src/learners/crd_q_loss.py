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

import logging
from typing import Any, Dict, Optional

import torch

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.typing import ModuleID, TensorType

from src.crd.cf_math import forecast_cf_per_step
from src.models.rlmodule_gtrxl_ensemble import COL_Q_ENSEMBLE


logger = logging.getLogger(__name__)


_DEFAULT_Q_LOSS_COEF = 0.5
_DEFAULT_BOOTSTRAP_P = 0.7
_DEFAULT_TRAIN_EVERY = 1
# Defaults for forecast-CF coefficients chosen to match the env's default
# global reward formula r = α·L − β·Ĉ − γ·R_w (see SimulationSettings
# defaults in MultiDatacenterSimulationCore: getGlobalRewardBeta=0.5,
# getGlobalRewardGamma=0.3).
_DEFAULT_BETA_FORECAST = 0.5
_DEFAULT_GAMMA_FORECAST = 0.3

# Custom batch column where M2.2 writes the per-transition R_forecast.
COL_CRD_FORECAST = "crd_forecast"


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
        # M2.1: per-module first-hit flag for the "[CRD] hook reached" log.
        self._crd_hook_logged: Dict[ModuleID, bool] = {}
        # M2.2: per-module first-hit flag for the
        # "predicted_wind_w missing → R_forecast=0" warning. Avoids spamming
        # the log every minibatch when the prediction pathway isn't active
        # (e.g., timecap-as-godeye experiments without WindPredictionWrapper).
        self._crd_pred_missing_warned: Dict[ModuleID, bool] = {}

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        # M2.1: CRD hook — entry point for forecast/routing/scheduling
        # counterfactual computation. First version is a no-op that just logs
        # once per module to verify the hook is reachable under the New API
        # stack. M2.2-M2.5 will fill in the actual computation here.
        self._compute_crd_terms(module_id=module_id, batch=batch, fwd_out=fwd_out)

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

    def _compute_crd_terms(
        self,
        *,
        module_id: ModuleID,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> None:
        """
        EU-CRD counterfactual computation entry point. Each milestone fills in
        a piece:
          - M2.1: log once per module (verify hook is reachable).
          - M2.2: forecast CF → batch[COL_CRD_FORECAST]   ← THIS milestone
          - M2.3: baseline action → batch["crd_baseline_action"]
          - M2.4: ΔQ + σ² via compute_q_ensemble → batch["crd_dq"], ["crd_sigma2"]
          - M2.5: Δr proxy → batch["crd_dr"]

        Skips entirely for non-ensemble modules (no `crd_q_ensemble` in
        fwd_out): if there's no Q-head ensemble, the rest of EU-CRD has no
        machinery to consume CF terms, so don't waste cycles producing them.
        """
        # Gate: only run CF computation for ensemble modules.
        if not (isinstance(fwd_out, dict) and COL_Q_ENSEMBLE in fwd_out):
            return
        if not isinstance(batch, dict):
            return

        # M2.1: warn-once log on first hit per module.
        if not self._crd_hook_logged.get(module_id, False):
            self._crd_hook_logged[module_id] = True
            logger.info(
                f"[CRD] hook reached for module {module_id!r}; "
                f"batch_keys={sorted(batch.keys())}; "
                f"fwd_out_keys={sorted(fwd_out.keys())}; "
                f"has_q_ensemble=True"
            )

        # M2.2: forecast counterfactual.
        self._compute_forecast_cf(module_id=module_id, batch=batch)

    def _read_module_crd_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.ensemble` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("ensemble", {}) or {}
        except Exception:
            return {}

    def _read_module_forecast_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.forecast` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("forecast", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ M2.2

    def _compute_forecast_cf(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        Compute R_forecast per transition and write `batch[COL_CRD_FORECAST]`.

        Reads the per-DC snapshot from `batch[Columns.INFOS][t]["crd"]`
        (populated by `HierarchicalMultiDCEnv._collect_crd_info` in M0.2).
        Each transition's R_forecast is:
            R_forecast(t) = β·[Ĉ(W_actual) - Ĉ(W_predicted)]
                          + γ·[R_w(W_actual) - R_w(W_predicted)]

        If `predicted_wind_w` is missing for a transition (e.g., experiments
        using TimeCAP-as-godeye without WindPredictionWrapper), the value
        falls back to 0.0 — that transition contributes no forecast share to
        ρ, which is the correct attribution since we have no forecast signal
        to compare against.
        """
        infos = batch.get(Columns.INFOS)
        if infos is None:
            return  # nothing to compute against

        forecast_cfg = self._read_module_forecast_config(module_id)
        beta = float(forecast_cfg.get("beta", _DEFAULT_BETA_FORECAST))
        gamma = float(forecast_cfg.get("gamma", _DEFAULT_GAMMA_FORECAST))

        r_forecast_list, n_missing_pred = self._compute_forecast_cf_values(
            infos=infos, beta=beta, gamma=gamma
        )
        if not r_forecast_list:
            return

        # Warn once if most transitions in the first observed batch had no
        # predicted_wind_w (e.g., TimeCAP-godeye experiments). This signals to
        # the user that the forecast-CF pathway is currently silent.
        if (
            n_missing_pred > 0
            and not self._crd_pred_missing_warned.get(module_id, False)
        ):
            self._crd_pred_missing_warned[module_id] = True
            frac = n_missing_pred / len(r_forecast_list)
            logger.warning(
                f"[CRD] module {module_id!r}: predicted_wind_w missing in "
                f"{n_missing_pred}/{len(r_forecast_list)} transitions "
                f"({frac:.1%}); R_forecast = 0 for those steps. "
                "Hint: enable WindPredictionWrapper or add predicted_wind_w "
                "into info['crd'] from your prediction pathway."
            )

        # Reshape to match a sibling (B, T) tensor so that downstream
        # consumers (M5 advantage rewrite) can broadcast cleanly.
        forecast_tensor = torch.tensor(r_forecast_list, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if (
            isinstance(ref, torch.Tensor)
            and ref.numel() == forecast_tensor.numel()
        ):
            forecast_tensor = forecast_tensor.reshape(ref.shape).to(
                ref.device, dtype=ref.dtype
            )
        batch[COL_CRD_FORECAST] = forecast_tensor

    @staticmethod
    def _compute_forecast_cf_values(
        infos: Any, beta: float, gamma: float
    ) -> tuple[list[float], int]:
        """
        Static helper that converts a sequence of info-dicts into a flat list
        of R_forecast values (one per transition) plus a count of transitions
        where predicted_wind_w was missing.

        Tolerant of multiple `infos` layouts: numpy object array, plain list,
        or anything iterable whose elements are dict-like.
        """
        r_forecast_list: list[float] = []
        n_missing_pred = 0

        for info in CRDPPOTorchLearner._iter_infos(infos):
            r = 0.0
            if isinstance(info, dict):
                crd = info.get("crd")
                if isinstance(crd, dict):
                    pred = crd.get("predicted_wind_w")
                    if pred is None:
                        n_missing_pred += 1
                    else:
                        r = forecast_cf_per_step(crd, pred, beta=beta, gamma=gamma)
                else:
                    n_missing_pred += 1  # No CRD snapshot → treat as missing.
            r_forecast_list.append(r)

        return r_forecast_list, n_missing_pred

    @staticmethod
    def _iter_infos(infos: Any):
        """Iterate `infos` regardless of whether it's a list, ndarray, or other iterable."""
        try:
            iter(infos)
        except TypeError:
            return iter([])
        return iter(infos)

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
