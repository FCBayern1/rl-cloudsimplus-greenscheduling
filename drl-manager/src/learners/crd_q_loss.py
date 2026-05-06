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

from src.baselines.global_schedulers import GreenQueueBalancedGlobalScheduler
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
# Default green-weight for the GreenQueueBalanced baseline scheduler in M2.3.
# Mirrors `GreenQueueBalancedGlobalScheduler`'s own default; exposed as a
# config knob so future experiments can sweep heuristic-strength.
_DEFAULT_BASELINE_GREEN_WEIGHT = 0.6
# Default α weight on the M2.5 Δr proxy. Independent of the env's
# `global_reward_alpha` (which may be 0 in carbon-only experiments) — Δr
# only needs a meaningful non-zero magnitude for M3 soft-blending to work.
_DEFAULT_ALPHA_DR = 1.0

# Custom batch column where M2.2 writes the per-transition R_forecast.
COL_CRD_FORECAST = "crd_forecast"
# Custom batch column where M2.3 writes per-transition baseline action ã.
COL_CRD_BASELINE_ACTION = "crd_baseline_action"
# Custom batch columns where M2.4 writes ΔQ = μ(s,a) - μ(s,ã) and
# σ²_tot = σ²(s,a) + σ²(s,ã). Both have shape (B, T).
COL_CRD_DQ = "crd_dq"
COL_CRD_SIGMA2 = "crd_sigma2"
# Custom batch column where M2.5 writes the reward-level fallback Δr,
# proxied by the load-std difference between actual and baseline routing.
# Shape (B, T). Used by M3 soft-blending when σ²_tot is large (ΔQ untrusted).
COL_CRD_DR = "crd_dr"


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
        # M2.3: per-module cache of the heuristic baseline scheduler used to
        # produce ã. None entries mark modules where baseline action does not
        # apply (e.g., local agents — those will be addressed in M4).
        self._crd_baseline_schedulers: Dict[ModuleID, Optional[Any]] = {}
        # M2.3: warn-once flag for missing decision-time signals
        # (dc_green_ratio / dc_queue_sizes) in info["crd"].
        self._crd_baseline_signal_warned: Dict[ModuleID, bool] = {}

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

        # M2.3: baseline (heuristic) action ã for the routing CF.
        self._compute_baseline_action(module_id=module_id, batch=batch)

        # M2.4: ΔQ + σ²_tot via ensemble lookup (no extra trunk forward).
        self._compute_dq_and_sigma2(
            module_id=module_id, batch=batch, fwd_out=fwd_out
        )

        # M2.5: reward-level Δr fallback (load-std proxy).
        self._compute_dr(module_id=module_id, batch=batch)

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

    # ------------------------------------------------------------------ M2.3

    def _compute_baseline_action(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        Compute heuristic baseline action ã per transition and write
        `batch[COL_CRD_BASELINE_ACTION]`.

        For the global routing module this calls
        `GreenQueueBalancedGlobalScheduler.schedule(obs)` per transition,
        feeding it the decision-time `dc_green_ratio` and `dc_queue_sizes`
        signals stashed by the env into `info["crd"]` (M0 / M2.3 prep).

        Local-agent modules currently fall through (M4 will wire BestFit).
        Returns silently if no usable scheduler exists or info data is missing.
        """
        sched = self._get_or_build_baseline_scheduler(module_id)
        if sched is None:
            return

        infos = batch.get(Columns.INFOS)
        if infos is None:
            return

        actions_list, n_missing_signals = self._compute_baseline_action_values(
            infos=infos,
            num_dc=sched.num_datacenters,
            scheduler=sched,
        )
        if not actions_list:
            return

        # Warn once if many transitions lack the queue-size signal — this
        # signals the env didn't populate the new info["crd"] keys (probably
        # an outdated env version mid-rollout).
        if n_missing_signals > 0 and not self._crd_baseline_signal_warned.get(
            module_id, False
        ):
            self._crd_baseline_signal_warned[module_id] = True
            frac = n_missing_signals / len(actions_list)
            logger.warning(
                f"[CRD] module {module_id!r}: dc_queue_sizes/dc_green_ratio "
                f"missing in {n_missing_signals}/{len(actions_list)} "
                f"({frac:.1%}); baseline ã defaults to all-zeros for those steps."
            )

        baseline_tensor = torch.tensor(actions_list, dtype=torch.long)
        # Align shape with batch[ACTIONS] when possible — M2.4 will gather Q
        # at this action so shape parity matters.
        ref = batch.get(Columns.ACTIONS)
        if isinstance(ref, torch.Tensor):
            try:
                if ref.numel() == baseline_tensor.numel():
                    baseline_tensor = baseline_tensor.reshape(ref.shape).to(ref.device)
            except RuntimeError:
                # Shape mismatch (e.g., RLlib reshapes ACTIONS unexpectedly);
                # leave baseline_tensor as (N, batch_size) and let M2.4 cope.
                pass
        batch[COL_CRD_BASELINE_ACTION] = baseline_tensor

    @staticmethod
    def _compute_baseline_action_values(
        infos: Any, num_dc: int, scheduler: Any
    ) -> tuple[list[list[int]], int]:
        """
        Static helper: walk the infos sequence, run the scheduler against the
        decision-time signals stashed in info["crd"], return a flat list of
        per-transition actions (each itself a list of length scheduler.batch_size).

        `n_missing_signals` counts transitions where queue/green-ratio data
        was absent — those default to all-zeros (a deterministic fallback).
        """
        actions_list: list[list[int]] = []
        n_missing_signals = 0
        zeros = [0.0] * num_dc
        zero_action = [0] * scheduler.batch_size

        for info in CRDPPOTorchLearner._iter_infos(infos):
            crd = info.get("crd") if isinstance(info, dict) else None
            if not isinstance(crd, dict):
                actions_list.append(list(zero_action))
                n_missing_signals += 1
                continue
            queue_sizes = crd.get("dc_queue_sizes")
            green_ratio = crd.get("dc_green_ratio")
            if queue_sizes is None or green_ratio is None:
                actions_list.append(list(zero_action))
                n_missing_signals += 1
                continue
            try:
                actions = scheduler.schedule(
                    {"dc_green_ratio": green_ratio, "dc_queue_sizes": queue_sizes}
                )
                actions_list.append([int(a) for a in actions])
            except Exception:
                actions_list.append(list(zero_action))
                n_missing_signals += 1

        return actions_list, n_missing_signals

    def _get_or_build_baseline_scheduler(
        self, module_id: ModuleID
    ) -> Optional[Any]:
        """
        Lazily construct the heuristic baseline scheduler for the given module.
        Cached after first build. Returns None for modules that don't have a
        MultiDiscrete (= global routing) action space; those are skipped.
        """
        if module_id in self._crd_baseline_schedulers:
            return self._crd_baseline_schedulers[module_id]
        try:
            module = self.module[module_id].unwrapped()
            action_space = getattr(module, "action_space", None)
            from gymnasium import spaces  # local import; avoid hard top-level dep

            if not isinstance(action_space, spaces.MultiDiscrete):
                # Local-agent module (Discrete) — defer to M4.
                self._crd_baseline_schedulers[module_id] = None
                return None
            nvec = list(action_space.nvec)
            if not nvec or any(int(n) != int(nvec[0]) for n in nvec):
                # Heterogeneous nvec is unusual for routing; bail safely.
                logger.warning(
                    f"[CRD] {module_id!r} action_space.nvec={nvec} is not "
                    "uniform; baseline scheduler skipped."
                )
                self._crd_baseline_schedulers[module_id] = None
                return None
            num_dc = int(nvec[0])
            batch_size = len(nvec)
            crd_cfg = self._read_module_baseline_config(module_id)
            green_w = float(
                crd_cfg.get("green_weight", _DEFAULT_BASELINE_GREEN_WEIGHT)
            )
            sched = GreenQueueBalancedGlobalScheduler(
                num_datacenters=num_dc, batch_size=batch_size, green_weight=green_w
            )
            self._crd_baseline_schedulers[module_id] = sched
            logger.info(
                f"[CRD] built GreenQueueBalanced baseline for {module_id!r}: "
                f"num_dc={num_dc}, batch_size={batch_size}, green_weight={green_w}"
            )
            return sched
        except Exception as e:
            logger.warning(
                f"[CRD] failed to build baseline scheduler for {module_id!r}: {e}"
            )
            self._crd_baseline_schedulers[module_id] = None
            return None

    def _read_module_baseline_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.baseline` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("baseline", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ M2.4

    def _compute_dq_and_sigma2(
        self,
        *,
        module_id: ModuleID,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> None:
        """
        Compute ΔQ = μ(s,a) - μ(s,ã) and σ²_tot = σ²(s,a) + σ²(s,ã) per
        transition, where μ/σ² are the ensemble mean/variance from the K-head
        Q-ensemble in `fwd_out[COL_Q_ENSEMBLE]`.

        Reuses the forward-pass ensemble output rather than re-running the
        GTrXL trunk via `module.compute_q_ensemble(batch, action)`. Saves an
        entire trunk forward per call (PPO does multiple SGD epochs over the
        same minibatch, so this matters).

        Skips silently if either the Q-ensemble or the M2.3 baseline action
        is absent (e.g., local agents in M2; M4 will wire those up).
        """
        q_ensemble = fwd_out.get(COL_Q_ENSEMBLE)
        if not isinstance(q_ensemble, torch.Tensor):
            return
        actual_action = batch.get(Columns.ACTIONS)
        baseline_action = batch.get(COL_CRD_BASELINE_ACTION)
        if actual_action is None or baseline_action is None:
            return  # M2.3 didn't produce ã (non-routing module, etc.)

        if not isinstance(actual_action, torch.Tensor):
            actual_action = torch.as_tensor(actual_action, dtype=torch.long)
        else:
            actual_action = actual_action.long()
        if not isinstance(baseline_action, torch.Tensor):
            baseline_action = torch.as_tensor(baseline_action, dtype=torch.long)
        else:
            baseline_action = baseline_action.long()

        try:
            delta_q, sigma2_tot = self._compute_dq_and_sigma2_values(
                q_ensemble=q_ensemble,
                actual_action=actual_action,
                baseline_action=baseline_action,
            )
        except Exception as e:
            logger.warning(
                f"[CRD] dq/sigma2 compute failed for {module_id!r}: {e}"
            )
            return

        # Detach: M5 will broadcast these against advantages — they should
        # not flow gradient back through the q-heads (q_loss in M1.2 is the
        # canonical training signal for the ensemble).
        batch[COL_CRD_DQ] = delta_q.detach()
        batch[COL_CRD_SIGMA2] = sigma2_tot.detach()

    @staticmethod
    def _gather_q_chosen(
        q_ensemble: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather Q values across the ensemble for the given action.

        Layouts:
          - Local discrete   q (B, T, K, A);          action (B, T)
              → returns (B, T, K)
          - Global MultiDisc q (B, T, K, bs, num_dc); action (B, T, bs)
              → returns (B, T, K) — per-cloudlet mean over bs

        Mirrors the gather pattern in `_compute_q_loss` so M1.2 and M2.4
        agree on what "Q at action" means.
        """
        if q_ensemble.dim() == 4:
            # Local: (B, T, K, A)
            B, T, K, A = q_ensemble.shape
            act = action
            if act.dim() == 1:
                act = act.unsqueeze(-1)  # (B, 1) — assume T==1
            idx = act.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
            return q_ensemble.gather(-1, idx).squeeze(-1)  # (B, T, K)
        elif q_ensemble.dim() == 5:
            # Global: (B, T, K, batch_size, num_dc)
            B, T, K, bs, nd = q_ensemble.shape
            act = action
            if act.dim() == 2:
                act = act.unsqueeze(1)  # (B, 1, batch_size) — assume T==1
            idx = act.unsqueeze(2).unsqueeze(-1).expand(-1, -1, K, -1, 1)
            q_per_cloudlet = q_ensemble.gather(-1, idx).squeeze(-1)  # (B,T,K,bs)
            return q_per_cloudlet.mean(dim=-1)  # (B, T, K)
        else:
            raise RuntimeError(
                f"Unexpected q_ensemble dim {q_ensemble.dim()} (shape "
                f"{tuple(q_ensemble.shape)}); expected 4 (local) or 5 (global)."
            )

    @staticmethod
    def _compute_dq_and_sigma2_values(
        *,
        q_ensemble: torch.Tensor,
        actual_action: torch.Tensor,
        baseline_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_q:    (B, T)   = μ(s, a) - μ(s, ã)
            sigma2_tot: (B, T)   = σ²(s, a) + σ²(s, ã), always ≥ 0
        Both averages/variances are taken over the K ensemble dimension.
        """
        q_a = CRDPPOTorchLearner._gather_q_chosen(q_ensemble, actual_action)
        q_b = CRDPPOTorchLearner._gather_q_chosen(q_ensemble, baseline_action)
        # mean / var over K dim (last after gather)
        mu_a = q_a.mean(dim=-1)
        mu_b = q_b.mean(dim=-1)
        var_a = q_a.var(dim=-1, unbiased=False)
        var_b = q_b.var(dim=-1, unbiased=False)
        delta_q = mu_a - mu_b
        sigma2_tot = var_a + var_b
        return delta_q, sigma2_tot

    # ------------------------------------------------------------------ M2.5

    def _compute_dr(self, *, module_id: ModuleID, batch: Dict[str, Any]) -> None:
        """
        Reward-level Δr fallback (proxy: load-std difference under actual vs
        baseline routing).

        The exact baseline reward would require either re-simulating the env
        (no replay) or analytical reward decomposition that's costlier than
        the value taken-from-Q-ensemble path (M2.4) anyway. For M3's soft
        blending we only need Δr to satisfy:
          - sign matches "agent vs baseline routing quality"
          - magnitude on a comparable scale to ΔQ when σ² is large enough
            that we'd actually fall back to it
        We approximate the agent-controllable α·L term by the negative
        load-imbalance (std of per-DC pending queue under each routing):
            Δr ≈ α · [(-actual_load_std) - (-baseline_load_std)]
               = α · (baseline_load_std - actual_load_std)
        Other reward terms (β·Ĉ, γ·R_w) are assumed identical between actual
        and baseline (they're forecast-driven, not routing-driven on the
        same step), so they cancel in the difference.

        Skips silently when the M2.3 baseline action wasn't produced
        (non-routing module) or when the queue snapshot is missing.
        """
        baseline_action = batch.get(COL_CRD_BASELINE_ACTION)
        actual_action = batch.get(Columns.ACTIONS)
        infos = batch.get(Columns.INFOS)
        if baseline_action is None or actual_action is None or infos is None:
            return

        # Use the same source of truth M2.3 used (not the cache dict directly,
        # so test stubs that override the method without populating the cache
        # still see the right scheduler).
        sched = self._get_or_build_baseline_scheduler(module_id)
        if sched is None:
            return  # No routing scheduler available (e.g., local agent) → skip.
        num_dc = sched.num_datacenters

        cfg = self._read_module_dr_config(module_id)
        alpha = float(cfg.get("alpha", _DEFAULT_ALPHA_DR))

        try:
            dr_list = self._compute_dr_values(
                infos=infos,
                actual_actions=actual_action,
                baseline_actions=baseline_action,
                num_dc=num_dc,
                alpha=alpha,
            )
        except Exception as e:
            logger.warning(f"[CRD] Δr compute failed for {module_id!r}: {e}")
            return

        dr_tensor = torch.tensor(dr_list, dtype=torch.float32)
        # Align shape with batch[REWARDS] (B, T) when possible.
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor) and ref.numel() == dr_tensor.numel():
            dr_tensor = dr_tensor.reshape(ref.shape).to(
                ref.device, dtype=ref.dtype
            )
        batch[COL_CRD_DR] = dr_tensor

    def _read_module_dr_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.delta_r` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("delta_r", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def _compute_dr_values(
        *,
        infos: Any,
        actual_actions: Any,
        baseline_actions: Any,
        num_dc: int,
        alpha: float,
    ) -> list[float]:
        """
        Static helper: walk the (info, actual_routing, baseline_routing)
        triples and compute per-transition Δr.

        Per-transition logic:
            base_q = info["crd"]["dc_queue_sizes"]      # length num_dc
            actual_q[dc]   = base_q[dc] + count(actual_routing == dc)
            baseline_q[dc] = base_q[dc] + count(baseline_routing == dc)
            Δr = α · (std(baseline_q) - std(actual_q))

        Missing queue snapshot → Δr = 0 for that transition (deterministic
        fallback, matches M2.3's approach).
        """
        # Coerce action tensors into a flat (N, batch_size) numpy form.
        import numpy as np  # local import keeps heavy deps out of module top
        if isinstance(actual_actions, torch.Tensor):
            actual_np = actual_actions.detach().cpu().numpy()
        else:
            actual_np = np.asarray(actual_actions)
        if isinstance(baseline_actions, torch.Tensor):
            baseline_np = baseline_actions.detach().cpu().numpy()
        else:
            baseline_np = np.asarray(baseline_actions)
        actual_np = actual_np.reshape(-1, actual_np.shape[-1])
        baseline_np = baseline_np.reshape(-1, baseline_np.shape[-1])

        infos_list = list(CRDPPOTorchLearner._iter_infos(infos))
        n = min(len(infos_list), actual_np.shape[0], baseline_np.shape[0])

        dr_values: list[float] = []
        for i in range(n):
            info = infos_list[i]
            crd = info.get("crd") if isinstance(info, dict) else None
            if not isinstance(crd, dict):
                dr_values.append(0.0)
                continue
            base_q_raw = crd.get("dc_queue_sizes")
            if base_q_raw is None or len(base_q_raw) != num_dc:
                dr_values.append(0.0)
                continue
            base_q = np.asarray(base_q_raw, dtype=np.float64)

            # Increment queues by routed cloudlet counts.
            actual_inc = np.bincount(
                actual_np[i].astype(np.int64), minlength=num_dc
            )[:num_dc]
            baseline_inc = np.bincount(
                baseline_np[i].astype(np.int64), minlength=num_dc
            )[:num_dc]
            actual_q = base_q + actual_inc
            baseline_q = base_q + baseline_inc

            actual_std = float(actual_q.std())
            baseline_std = float(baseline_q.std())
            dr_values.append(alpha * (baseline_std - actual_std))

        return dr_values

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
