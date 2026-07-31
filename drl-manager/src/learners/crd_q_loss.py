"""
M1.2: PPO learner extension that trains the Q-head ensemble via off-policy
TD bootstrap, with per-sample bootstrap masking (Osband et al., 2016).

Design:
- Subclasses `NormalizedCriticPPOTorchLearner` (a PPOTorchLearner whose vf
  term can be variance-normalized per module via the `normalized_critic`
  model_config gate; identical to vanilla PPO when the gate is off). The
  base PPO loss (policy + V-head + entropy + KL) is otherwise unchanged.
  We only ADD an extra term:
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
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.typing import ModuleID, TensorType

from src.baselines.global_schedulers import GreenQueueBalancedGlobalScheduler
from src.baselines.local_schedulers import BestFitLocalScheduler
from src.crd.blender import CRDBlender
from src.crd.cf_math import forecast_cf_per_step
from src.learners.normalized_critic_loss import NormalizedCriticPPOTorchLearner
# CRD inherits PerSlotCredit (which itself inherits NormalizedCritic) so that when
# per_slot_credit.enabled=true (Tier-1 routing-slot masking, needed by the deferrable
# gdpd/C-regime arch) the masked surrogate is applied AFTER CRD reweights ADVANTAGES.
# When per_slot_credit is off (legacy 5dc_v2 CRD configs) PerSlotCredit is a pass-through
# to NormalizedCritic → bit-identical to before this change.
from src.learners.per_slot_credit_loss import PerSlotCreditPPOTorchLearner
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
# Default α weight on the M4 local-scheduling Δr proxy (PE-waste difference
# between the agent's VM choice and the BestFit baseline). Separate knob from
# the routing Δr above so the two layers can be scaled independently.
_DEFAULT_ALPHA_DR_LOCAL = 1.0
# Default M5 responsibility-weight floor. Per-transition ρ ∈ [ρ_min, 1]
# clamping prevents the policy gradient from vanishing on transitions
# where the agent's controllable share is dwarfed by exogenous forecast
# error. Plan §3.4 default.
_DEFAULT_RHO_MIN = 0.05

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
# Custom batch columns where M3 writes the soft-blended routing-responsibility
# signal R^routing_t = c(t)·ΔQ + (1-c(t))·Δr, plus diagnostics: c(t) per
# transition and the scalar τ used for this minibatch.
COL_CRD_R_ROUTING = "crd_r_routing"
COL_CRD_C_T = "crd_c_t"
COL_CRD_TAU = "crd_tau"
# M4 writes R_scheduling for local agents (soft-blended ΔQ_local/Δr_local).
# M5 reads this: in a local module's batch it is the agent-controllable share
# that reweights the local ADVANTAGES; in a global module's batch it is absent
# (each module only carries its own layer's signal).
COL_CRD_R_SCHEDULING = "crd_r_scheduling"
# M5 writes per-transition responsibility weights. ρ_routing /
# ρ_scheduling reach the policy gradient via the advantage rewrite;
# ρ_forecast is logging-only (never multiplied — exogenous error stays
# unattributed, which is the whole point of EU-CRD).
COL_CRD_RHO_FORECAST = "crd_rho_forecast"
COL_CRD_RHO_ROUTING = "crd_rho_routing"
COL_CRD_RHO_SCHEDULING = "crd_rho_scheduling"


class _PolicySelfBaselineMarker:
    """Sentinel for crd.baseline.kind='policy_self'. Carries the shape fields the
    baseline-gating code reads; the actual counterfactual action is drawn from the
    policy distribution in _compute_policy_self_baseline_action, not from a router."""

    def __init__(self, num_datacenters: int, batch_size: int):
        self.num_datacenters = int(num_datacenters)
        self.batch_size = int(batch_size)


class CRDPPOTorchLearner(PerSlotCreditPPOTorchLearner):
    """
    PPO learner with EU-CRD Q-head TD loss bolted on.

    Reads the per-module Q-loss config from `module.model_config["crd"]
    ["ensemble"]`; falls back to defaults if absent. Modules that don't
    expose `crd_q_ensemble` get only the base PPO loss.

    P1 critic fix (2026-06-11): the base class is now
    NormalizedCriticPPOTorchLearner, so CRD runs can normalize the vf loss
    via the same per-module `normalized_critic` model_config gate. The gate
    defaults to OFF — configs without the block keep pre-P1 behavior
    bit-for-bit.
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
        # warn-once flag for R_forecast failing to align to the (B, T) grid
        # (neither B*T nor loss_mask valid-count) — would silently zero the
        # forecast share, disabling forecast shielding.
        self._crd_forecast_align_warned: Dict[ModuleID, bool] = {}
        # M2.3: per-module cache of the heuristic baseline scheduler used to
        # produce ã. None entries mark modules where baseline action does not
        # apply (e.g., local agents — those will be addressed in M4).
        self._crd_baseline_schedulers: Dict[ModuleID, Optional[Any]] = {}
        # M4: per-module cache of the local BestFit baseline scheduler used to
        # produce ã_local for Discrete (local-scheduling) modules. None marks
        # modules where local baseline doesn't apply (e.g., the global router).
        self._crd_local_baseline_schedulers: Dict[ModuleID, Optional[Any]] = {}
        # M4: warn-once flag for the obs-based local baseline path being
        # unavailable (missing action_mask / vm_available_pes), which forces
        # ΔQ_local/σ²_local to be skipped for that module.
        self._crd_local_baseline_warned: Dict[ModuleID, bool] = {}
        # M2.3: warn-once flag for missing decision-time signals
        # (dc_green_ratio / dc_queue_sizes) in info["crd"].
        self._crd_baseline_signal_warned: Dict[ModuleID, bool] = {}
        # M3: per-module CRDBlender (stateful EMA of σ² + adaptive τ) that
        # combines ΔQ and Δr into R^routing.
        self._crd_blenders: Dict[ModuleID, CRDBlender] = {}
        # CCA-PG baseline: per-module hindsight net + its online optimizer,
        # lazily built on first use. Config-gated cross-comparison method.
        self._cca_nets: Dict[ModuleID, Any] = {}
        self._cca_opts: Dict[ModuleID, Any] = {}
        # v5: per-module per-source running-scale EMAs (share normalisation)
        # and the forecast-anomaly EMA (anomaly-gated quarantine). Both are
        # config-gated (responsibility.normalize_shares / .anomaly_gate),
        # default off — bit-identical to v4 unless enabled.
        self._crd_share_scale_ema: Dict[ModuleID, Dict[str, float]] = {}
        self._crd_forecast_anom_ema: Dict[ModuleID, Dict[str, float]] = {}
        # v5.2: training-iteration counter for the stable bootstrap mask.
        self._crd_train_iter: int = 0
        # M2.4: warn-once flag for the dq/sigma2 alignment failure path.
        # Layout mismatches between the padded q_ensemble and the
        # sequence-packed `infos`-derived baseline_action are still under
        # investigation; until M2.3 produces baseline_action in a layout
        # that always aligns, we want exactly one diagnostic line per
        # module instead of one per minibatch (PPO does ~150+ minibatches).
        self._crd_dq_align_warned: Dict[ModuleID, bool] = {}
        # M2.4 fix: warn-once flag for "obs-based baseline path unavailable",
        # which forces the fallback to the misaligned infos path. Helps
        # diagnose why ΔQ/σ² is still being skipped.
        self._crd_baseline_obs_warned: bool = False
        # M7: warn-once flag for diagnostics-logging failure.
        self._crd_diag_warned: bool = False

    def _warn_baseline_obs_unavailable_once(self, reason: str) -> None:
        if not self._crd_baseline_obs_warned:
            self._crd_baseline_obs_warned = True
            logger.warning(
                f"[CRD] obs-based baseline path unavailable ({reason}); "
                "falling back to infos-based path which may misalign with "
                "the padded (B,T) grid. ΔQ/σ² may be skipped as a result."
            )

    def before_gradient_based_update(self, *, timesteps=None, **kwargs) -> None:
        # v5.2: one tick per TRAINING ITERATION (not per minibatch call) — the
        # stable bootstrap mask is seeded from this so all epochs/minibatches
        # of one iteration share head-data assignments (per-call resampling let
        # every head see every transition, collapsing bootstrap diversity onto
        # the priors alone).
        self._crd_train_iter += 1
        return super().before_gradient_based_update(timesteps=timesteps, **kwargs)

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

        # Risk-averse baseline objectives (CVaR / risk-sensitive / mean-variance).
        # Config-gated cross-comparison methods: transform the advantage BEFORE the
        # base PPO loss so they act on the vanilla backbone (no ensemble/EU-CRD
        # machinery). Default kind="none" is bit-identical to standard PPO.
        self._apply_risk_objective(module_id=module_id, batch=batch)

        # CCA-PG baseline (Mesnard 2021): replace the advantage with
        # R_t - V^h(V(s_t), Phi_t), Phi = future realised green. Config-gated,
        # no-op unless crd.cca.enabled — acts on the vanilla backbone.
        self._apply_cca(module_id=module_id, batch=batch)

        base_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        q_ensemble = fwd_out.get(COL_Q_ENSEMBLE)
        if q_ensemble is None:
            return base_loss  # Non-ensemble module — no Q-loss contribution.

        # Ablation gate (crd.local_cf_enabled=false): a gated local module's
        # Q-ensemble has no consumer (no ΔQ_local), so skip its TD loss too —
        # the local layer is trained by the base PPO loss alone.
        if self._skip_local_cf(module_id):
            return base_loss

        crd_cfg = self._read_module_crd_config(module_id)
        coef = float(crd_cfg.get("q_loss_coef", _DEFAULT_Q_LOSS_COEF))
        bootstrap_p = float(crd_cfg.get("bootstrap_p", _DEFAULT_BOOTSTRAP_P))
        train_every = max(1, int(crd_cfg.get("q_train_every_n_iters", _DEFAULT_TRAIN_EVERY)))

        # Skip-by-counter: keep call cadence sparse when knob > 1.
        n = self._crd_call_counts.get(module_id, 0)
        self._crd_call_counts[module_id] = n + 1
        if (n % train_every) != 0:
            return base_loss

        # Normalize the Q-loss by the SAME running target-variance the critic
        # uses (set by NormalizedCriticPPOTorchLearner during the super() call
        # above). The Q-head's TD target IS Postprocessing.VALUE_TARGETS, so
        # without this the raw Q-loss carries the full return scale (residuals²
        # ~thousands) and, now that the base vf loss is O(1) in σ² units, would
        # swamp the total loss. `_vf_target_var_ema[module_id]` is populated
        # only when `normalized_critic.enabled` for this module; absent → 1.0
        # → raw Q-loss (bit-identical to pre-normalization behavior).
        var_ema = self._vf_target_var_ema.get(module_id)
        target_var = max(float(var_ema), 1e-8) if var_ema is not None else 1.0
        q_loss = self._compute_q_loss(
            q_ensemble, batch, bootstrap_p, target_var=target_var,
            mask_padding=self._read_crd_mask_padding(module_id),
            stable_seed=(self._crd_train_iter
                         if bool(crd_cfg.get("stable_bootstrap", False)) else None),
        )
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

        # Ablation gate (crd.local_cf_enabled, default True): when False, local
        # scheduler (Discrete-action) modules skip the ENTIRE counterfactual
        # pipeline — no forecast CF, no BestFit baseline, no R_scheduling, no
        # advantage reweight — reverting the local layer to conventional PPO.
        # Paired with the Q-loss skip in compute_loss_for_module. The global
        # router is unaffected.
        if self._skip_local_cf(module_id):
            if not hasattr(self, "_crd_local_cf_gate_logged"):
                self._crd_local_cf_gate_logged = {}
            if not self._crd_local_cf_gate_logged.get(module_id, False):
                self._crd_local_cf_gate_logged[module_id] = True
                logger.info(
                    f"[CRD] local_cf_enabled=false — module {module_id!r} "
                    "skips the local counterfactual pipeline (conventional PPO)."
                )
            return

        # M2.2: forecast counterfactual. Shared by both layers — every agent's
        # info["crd"] carries the same exogenous wind snapshot, so the forecast
        # share shields local scheduling agents from forecast error too.
        self._compute_forecast_cf(module_id=module_id, batch=batch)

        # Dispatch on action-space kind: the global router (MultiDiscrete) gets
        # the routing CF; local schedulers (Discrete) get the scheduling CF. A
        # module is exactly one of the two, so the shared columns
        # (baseline_action / crd_dq / crd_sigma2 / crd_dr) never collide within
        # a single batch.
        global_sched = self._get_or_build_baseline_scheduler(module_id)
        local_sched = (
            None
            if global_sched is not None
            else self._get_or_build_local_baseline_scheduler(module_id)
        )

        # ── Baseline action ã (layer-specific producer) ─────────────────────
        if global_sched is not None:
            # M2.3: heuristic GreenQueueBalanced routing baseline.
            self._compute_baseline_action(module_id=module_id, batch=batch)
        elif local_sched is not None:
            # M4.2a: BestFit baseline VM choice from the obs grid.
            self._compute_local_baseline_action(module_id=module_id, batch=batch)

        # M2.4 / M4.2b: ΔQ + σ²_tot via ensemble lookup (no extra trunk
        # forward). Layout-agnostic — gates internally on baseline_action being
        # present, so it is safe to call unconditionally (tests may pre-seed
        # baseline_action without a cached scheduler).
        self._compute_dq_and_sigma2(
            module_id=module_id, batch=batch, fwd_out=fwd_out
        )

        # ── Δr fallback + σ²-gated blend (layer-specific) ───────────────────
        if global_sched is not None:
            # M2.5 reward-level Δr (load-std proxy) → M3 blend into R^routing.
            self._compute_dr(module_id=module_id, batch=batch)
            self._compute_r_routing(module_id=module_id, batch=batch)
        elif local_sched is not None:
            # M4.2c reward-level Δr_local (PE-waste proxy) → M4.2d blend into
            # R^scheduling (reuses the per-module CRDBlender).
            self._compute_local_dr(module_id=module_id, batch=batch)
            self._compute_r_scheduling(module_id=module_id, batch=batch)

        # M5: per-transition responsibility weights ρ_k + reweight advantages.
        # Must run BEFORE super().compute_loss_for_module so PPO's surrogate
        # loss reads the reweighted ADVANTAGES tensor.
        self._compute_responsibilities(module_id=module_id, batch=batch)

        # M7: push CRD diagnostic scalars into the RLlib result dict. The
        # WandbLoggerCallback auto-forwards everything in result to wandb, so
        # no wandb-specific code is needed here — just log via self.metrics.
        self._log_crd_diagnostics(module_id=module_id, batch=batch)

    # ------------------------------------------------------------------ M7

    def _log_crd_diagnostics(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        Surface CRD internal signals as logged scalars so they appear under
        result["learners"][module_id]["crd/..."] and flow to wandb/TensorBoard.

        Logged (per minibatch, aggregated by the MetricsLogger over the iter):
          - crd/rho_forecast_mean    — exogenous (forecast) responsibility share
          - crd/rho_routing_mean     — routing responsibility share
          - crd/rho_scheduling_mean  — scheduling responsibility share
          - crd/sigma2_tot_mean      — ensemble epistemic uncertainty (↓ as critic matures)
          - crd/c_t_mean             — soft-blend confidence gate (↑ as σ² drops)
          - crd/tau                  — adaptive temperature
          - crd/r_forecast_abs_mean  — |R_forecast| magnitude (forecast-CF signal strength)
          - crd/dq_mean / crd/dr_mean — value-level vs reward-level CF magnitudes

        No-op if self.metrics isn't available (e.g., unit-test stubs that
        bypass Learner.build()).
        """
        metrics = getattr(self, "metrics", None)
        if metrics is None:
            return

        lm = batch.get(Columns.LOSS_MASK)

        def _scalar_mean(key: str):
            # Masked mean when the loss mask aligns: padded cells carry garbage
            # magnitudes (e.g. |dr| ~ 51 at pads vs ~0.5 valid) that otherwise
            # dominate the diagnostic and hide dead mechanisms. Logging-only.
            t = batch.get(key)
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                if isinstance(lm, torch.Tensor) and lm.shape == t.shape and bool(lm.any()):
                    t = t[lm.bool()]
                return t.detach().float().mean().item()
            return None

        def _scalar_spread(key: str, short: str) -> Dict[str, float]:
            """Within-batch dispersion of a ρ column, plus the reweight strength.

            The batch MEAN of ρ is stable by averaging and says nothing about
            whether credit is actually redistributed: the mean-preserving update
            multiplies each advantage by w = ρ_t / mean(ρ), so if ρ_t barely
            varies across transitions then w ≈ 1 and the reweighting is inert.
            std(w) is therefore the metric that tells a live mechanism from a
            no-op. Logging-only, computed on the detached column.
            """
            t = batch.get(key)
            if not (isinstance(t, torch.Tensor) and t.numel() > 1):
                return {}
            if isinstance(lm, torch.Tensor) and lm.shape == t.shape and bool(lm.any()):
                t = t[lm.bool()]
            t = t.detach().float().flatten()
            if t.numel() < 2:
                return {}
            mean = t.mean()
            out = {
                f"crd/{short}_std": t.std(unbiased=False).item(),
                f"crd/{short}_p10": torch.quantile(t, 0.1).item(),
                f"crd/{short}_p90": torch.quantile(t, 0.9).item(),
            }
            if torch.isfinite(mean) and float(mean) > 1e-6:
                w = t / mean
                out["crd/reweight_w_std"] = w.std(unbiased=False).item()
                out["crd/reweight_w_max"] = w.max().item()
            return out

        diag = {
            "crd/rho_forecast_mean": _scalar_mean(COL_CRD_RHO_FORECAST),
            "crd/rho_routing_mean": _scalar_mean(COL_CRD_RHO_ROUTING),
            "crd/rho_scheduling_mean": _scalar_mean(COL_CRD_RHO_SCHEDULING),
            "crd/sigma2_tot_mean": _scalar_mean(COL_CRD_SIGMA2),
            "crd/c_t_mean": _scalar_mean(COL_CRD_C_T),
            "crd/dq_mean": _scalar_mean(COL_CRD_DQ),
            "crd/dr_mean": _scalar_mean(COL_CRD_DR),
            "crd/r_routing_mean": _scalar_mean(COL_CRD_R_ROUTING),
            "crd/r_scheduling_mean": _scalar_mean(COL_CRD_R_SCHEDULING),
            "crd/reweight_applied": (
                1.0 if isinstance(batch.get("crd_reweight_applied"), torch.Tensor) else 0.0
            ),
        }
        # |R_forecast| magnitude (sign-agnostic — how strong the forecast-CF
        # signal is, which is what the attribution-fidelity experiment tracks).
        rf = batch.get(COL_CRD_FORECAST)
        if isinstance(rf, torch.Tensor) and rf.numel() > 0:
            diag["crd/r_forecast_abs_mean"] = rf.detach().float().abs().mean().item()
        # τ is a 0-d tensor written by M3.
        tau = batch.get(COL_CRD_TAU)
        if isinstance(tau, torch.Tensor) and tau.numel() == 1:
            diag["crd/tau"] = tau.detach().float().item()

        # Dispersion of the share this module is actually reweighted by, so the
        # curves can distinguish an active decomposition from an inert one.
        # Same rule the reweighting itself uses: a module is the global router
        # when R_routing is present, otherwise a local scheduler.
        if isinstance(batch.get(COL_CRD_R_ROUTING), torch.Tensor):
            own_rho_col, own_short = COL_CRD_RHO_ROUTING, "rho_routing"
        else:
            own_rho_col, own_short = COL_CRD_RHO_SCHEDULING, "rho_scheduling"
        try:
            diag.update(_scalar_spread(own_rho_col, own_short))
        except Exception:
            # Dispersion stats are a diagnostic; never let them stop training.
            pass

        diag = {k: v for k, v in diag.items() if v is not None}
        if not diag:
            return
        try:
            metrics.log_dict(diag, key=module_id, window=1)
        except Exception as e:
            # Diagnostics must never crash training.
            if not self._crd_diag_warned:
                self._crd_diag_warned = True
                logger.warning(f"[CRD] diagnostics logging failed: {e}")

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
        forecast_cfg = self._read_module_forecast_config(module_id)
        beta = float(forecast_cfg.get("beta", _DEFAULT_BETA_FORECAST))
        gamma = float(forecast_cfg.get("gamma", _DEFAULT_GAMMA_FORECAST))
        # v5.2 R_f repairs (both default False = legacy): carbon_norm makes the
        # beta term dimensionless (worst-case step-carbon normalisation) so it
        # is not numerically dead at 1 s timesteps; magnitude sums |terms| so
        # opposite-signed beta/gamma cannot cancel a real forecast error.
        carbon_norm = bool(forecast_cfg.get("carbon_norm", False))
        magnitude = bool(forecast_cfg.get("magnitude", False))

        # ── Preferred path: obs-based forecast (M-fix) ──────────────────────
        # The env attaches a per-step `crd_aux` snapshot to the obs (a sibling
        # of "observation", invisible to the policy). It rides the padded
        # (B, T) obs grid, so it stays aligned with ADVANTAGES/ΔQ under PPO
        # minibatching — unlike `infos`, whose length is unreliable
        # (observed 95 vs 2048; 1953 vs 1952). When present this is the
        # authoritative path and we skip infos entirely.
        obs_based = self._compute_forecast_cf_from_obs(
            batch, beta=beta, gamma=gamma,
            carbon_norm=carbon_norm, magnitude=magnitude,
        )
        if obs_based is not None:
            batch[COL_CRD_FORECAST] = obs_based
            return

        # ── Fallback: legacy infos-based path ───────────────────────────────
        infos = batch.get(Columns.INFOS)
        if infos is None:
            return  # nothing to compute against

        r_forecast_list, n_missing_pred = self._compute_forecast_cf_values(
            infos=infos, beta=beta, gamma=gamma,
            carbon_norm=carbon_norm, magnitude=magnitude,
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

        # Align to the (B, T) grid the other CRD signals (ΔQ/Δr/R_routing)
        # live on, so M5 actually folds |R_forecast| into the ρ denominator.
        # Three cases, mirroring the ΔQ/Δr loss-mask handling:
        #   1. len == B*T            → direct reshape (full padded grid).
        #   2. len == loss_mask.sum() → infos is sequence-packed to N_valid
        #      (RLlib's unpadded form under PPO minibatching); scatter the
        #      values into a zero-filled (B, T) grid at the valid positions.
        #   3. neither               → leave 1-D + warn once. Downstream M5
        #      will zero it (can't broadcast), so the forecast share would be
        #      silently dropped — surfacing the warning makes that visible.
        forecast_tensor = torch.tensor(r_forecast_list, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        loss_mask = batch.get(Columns.LOSS_MASK)
        aligned = self._align_forecast_to_bt(
            forecast_tensor, ref=ref, loss_mask=loss_mask
        )
        if aligned is not None:
            forecast_tensor = aligned
        elif (
            isinstance(ref, torch.Tensor)
            and not self._crd_forecast_align_warned.get(module_id, False)
        ):
            self._crd_forecast_align_warned[module_id] = True
            logger.warning(
                f"[CRD] module {module_id!r}: R_forecast length "
                f"{forecast_tensor.numel()} matches neither B*T="
                f"{ref.numel()} nor loss_mask valid-count="
                f"{None if not isinstance(loss_mask, torch.Tensor) else int(loss_mask.sum().item())}; "
                "leaving 1-D — ρ_forecast will be 0 (forecast shielding "
                "inactive) for this module until the layout is reconciled."
            )
        batch[COL_CRD_FORECAST] = forecast_tensor

    def _compute_forecast_cf_from_obs(
        self, batch: Dict[str, Any], *, beta: float, gamma: float,
        carbon_norm: bool = False, magnitude: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Compute R_forecast per (b, t) from the env's `crd_aux` obs channel,
        producing a (B, T) tensor aligned with ADVANTAGES/ΔQ.

        `crd_aux` is a sibling of obs["observation"] (see the PettingZoo
        wrapper) carrying the raw per-DC wind/power/carbon snapshot:
            crd_actual_green_w, crd_predicted_green_w, crd_total_power_w,
            crd_green_factor, crd_brown_factor   — each (B, T, num_dc)
            crd_timestep_hours                   — (B, T, 1)
        Each cell is fed to `forecast_cf_per_step`, the same helper the infos
        path uses, so values match what the infos path would have produced.

        Returns None if crd_aux is absent/malformed (caller falls back to the
        infos path). Missing predicted wind defaults to actual → R_forecast=0.
        """
        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            return None
        aux = obs.get("crd_aux")
        # crd_aux may itself be nested if a connector wrapped the obs; unwrap
        # one "observation" level defensively, though the wrapper places it at
        # the top level alongside "observation".
        if not isinstance(aux, dict) and isinstance(obs.get("observation"), dict):
            aux = obs["observation"].get("crd_aux")
        if not isinstance(aux, dict):
            return None
        req = (
            "crd_actual_green_w", "crd_predicted_green_w", "crd_total_power_w",
            "crd_green_factor", "crd_brown_factor", "crd_timestep_hours",
        )
        if not all(isinstance(aux.get(k), torch.Tensor) for k in req):
            return None

        def _grid(t: torch.Tensor) -> Optional[torch.Tensor]:
            # Coerce (B, T, F) or flat (N, F) → (B, T, F).
            if t.dim() == 3:
                return t
            if t.dim() == 2:
                return t.unsqueeze(1)
            return None

        actual = _grid(aux["crd_actual_green_w"])
        if actual is None:
            return None
        B, T, n = actual.shape
        pred = _grid(aux["crd_predicted_green_w"])
        total = _grid(aux["crd_total_power_w"])
        gf = _grid(aux["crd_green_factor"])
        bf = _grid(aux["crd_brown_factor"])
        dt = _grid(aux["crd_timestep_hours"])
        if any(x is None or x.shape[:2] != (B, T) for x in (pred, total, gf, bf, dt)):
            return None

        import numpy as np
        a_np = actual.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()
        tot_np = total.detach().cpu().numpy()
        gf_np = gf.detach().cpu().numpy()
        bf_np = bf.detach().cpu().numpy()
        dt_np = dt.detach().cpu().numpy()

        out = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                crd_info = {
                    "actual_wind_w": a_np[b, t],
                    "p_total_w": tot_np[b, t],
                    "timestep_hours": float(dt_np[b, t].flat[0]),
                    "green_carbon_factor": gf_np[b, t],
                    "brown_carbon_factor": bf_np[b, t],
                }
                out[b, t] = forecast_cf_per_step(
                    crd_info, p_np[b, t], beta=beta, gamma=gamma,
                    carbon_norm=carbon_norm, magnitude=magnitude,
                )

        tensor = torch.tensor(out, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor) and ref.shape == (B, T):
            tensor = tensor.to(ref.device, dtype=ref.dtype)
        elif isinstance(ref, torch.Tensor) and ref.numel() == B * T:
            tensor = tensor.reshape(ref.shape).to(ref.device, dtype=ref.dtype)
        return tensor

    @staticmethod
    def _align_forecast_to_bt(
        forecast_tensor: torch.Tensor,
        *,
        ref: Any,
        loss_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Coerce a flat per-transition R_forecast vector to the (B, T) grid.

        Returns the (B, T) tensor (on ref's device/dtype) or None if it cannot
        be aligned (caller decides how to surface that).
        """
        n = forecast_tensor.numel()
        # Case 1: already covers the full padded grid.
        if isinstance(ref, torch.Tensor) and ref.numel() == n:
            return forecast_tensor.reshape(ref.shape).to(ref.device, dtype=ref.dtype)
        # Case 2: sequence-packed to ~N_valid — scatter via loss_mask.
        #
        # `infos` under PPO minibatching is sequence-packed in the same
        # row-major valid order as the actions (the q-gather mask path relies
        # on the same ordering and works). It often carries a few EXTRA
        # trailing bootstrap-timestep entries, so len(infos) is N_valid or a
        # little more (observed: N_valid+1). We tolerate up to one extra per
        # sequence, trim from the front (= drop the trailing bootstrap rows),
        # and scatter the first N_valid values into the (B, T) grid. A 1-cell
        # misalignment over ~2000 transitions is negligible for the
        # per-cell ρ ratio and the σ²/τ EMAs.
        if isinstance(loss_mask, torch.Tensor):
            flat_mask = loss_mask.reshape(-1).bool()
            n_valid = int(flat_mask.sum().item())
            num_seqs = int(loss_mask.shape[0]) if loss_mask.dim() >= 1 else 1
            if n_valid <= n <= n_valid + max(1, num_seqs):
                device = ref.device if isinstance(ref, torch.Tensor) else forecast_tensor.device
                dtype = ref.dtype if isinstance(ref, torch.Tensor) else forecast_tensor.dtype
                vals = forecast_tensor[:n_valid].to(device, dtype=dtype)
                out = torch.zeros(flat_mask.numel(), dtype=dtype, device=device)
                out[flat_mask] = vals
                return out.reshape(loss_mask.shape)
        return None

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
        signals.

        Layout strategy (M2.4 fix):
          1. **OBS-based path (preferred)**: read `dc_green_ratio` /
             `dc_queue_sizes` from `batch[Columns.OBS]` which RLlib lays out
             as (B, T, num_dc) padded tensors — same shape as
             `batch[Columns.ACTIONS]`. The resulting baseline_action is
             (B, T, bs), aligning directly with actual_action for M2.4
             gather. This is robust to RLlib's bootstrap-timestep
             insertion in `Columns.INFOS`.
          2. **INFOS-based fallback**: legacy path used when obs is not in
             dict form or required keys are absent. Produces (N, bs) and
             M2.4 routes to its loss-mask path (which may still mismatch
             due to bootstrap timesteps; in that case ΔQ/σ² is skipped
             with a warn-once).

        Local-agent modules currently fall through (M4 will wire BestFit).
        """
        sched = self._get_or_build_baseline_scheduler(module_id)
        if sched is None:
            return
        if isinstance(sched, _PolicySelfBaselineMarker):
            self._compute_policy_self_baseline_action(module_id=module_id, batch=batch)
            return

        # ── Preferred path: build (B, T, bs) from padded obs ────────────
        baseline_from_obs = self._compute_baseline_from_obs(batch, sched)
        if baseline_from_obs is not None:
            batch[COL_CRD_BASELINE_ACTION] = baseline_from_obs
            return

        # ── Fallback: legacy infos-based path ────────────────────────────
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
        ref = batch.get(Columns.ACTIONS)
        if isinstance(ref, torch.Tensor):
            try:
                if ref.numel() == baseline_tensor.numel():
                    baseline_tensor = baseline_tensor.reshape(ref.shape).to(ref.device)
            except RuntimeError:
                pass
        batch[COL_CRD_BASELINE_ACTION] = baseline_tensor

    def _compute_baseline_from_obs(
        self, batch: Dict[str, Any], scheduler: Any
    ) -> Optional[torch.Tensor]:
        """
        Build per-cell baseline action by iterating the padded obs grid
        `(B, T, num_dc)`. Returns a `(B, T, bs)` tensor (long) aligned with
        actual_action; returns None if obs is not in usable dict form or
        required keys are missing.

        Cost: B*T scheduler.schedule() calls per minibatch. For B=4 T=128
        that's 512 calls — each is a few numpy ops, on the order of <1ms,
        so per-minibatch cost is sub-second on CPU.
        """
        obs = batch.get(Columns.OBS)
        # Unwrap RLlib's "observation" nesting (the Connector wrapper / action-
        # masking RLModules store the real obs dict under obs["observation"]).
        # GTrXL{Global,ScoreBased,...} RLModules all do this unwrap in their
        # _extract_obs_and_mask / _forward_pass; we must mirror it here or the
        # dc_* keys are invisible and we silently fall back to the misaligned
        # infos path.
        if isinstance(obs, dict) and isinstance(obs.get("observation"), dict):
            obs = obs["observation"]
        if not isinstance(obs, dict):
            self._warn_baseline_obs_unavailable_once("obs is not a dict")
            return None
        green_ratio = obs.get("dc_green_ratio")
        queue_sizes = obs.get("dc_queue_sizes")
        if not isinstance(green_ratio, torch.Tensor):
            self._warn_baseline_obs_unavailable_once(
                f"dc_green_ratio missing/non-tensor; obs keys={sorted(obs.keys())}"
            )
            return None
        if not isinstance(queue_sizes, torch.Tensor):
            self._warn_baseline_obs_unavailable_once(
                f"dc_queue_sizes missing/non-tensor; obs keys={sorted(obs.keys())}"
            )
            return None
        if green_ratio.shape != queue_sizes.shape:
            self._warn_baseline_obs_unavailable_once(
                f"dc_green_ratio {tuple(green_ratio.shape)} != dc_queue_sizes "
                f"{tuple(queue_sizes.shape)}"
            )
            return None
        # Accept padded (B, T, num_dc); also accept (N, num_dc) flat layouts
        # by reshaping to (N, 1, num_dc) so the same iteration works.
        if green_ratio.dim() == 3:
            B, T, num_dc = green_ratio.shape
        elif green_ratio.dim() == 2:
            N, num_dc = green_ratio.shape
            green_ratio = green_ratio.view(N, 1, num_dc)
            queue_sizes = queue_sizes.view(N, 1, num_dc)
            B, T = N, 1
        else:
            return None

        if num_dc != scheduler.num_datacenters:
            self._warn_baseline_obs_unavailable_once(
                f"obs num_dc={num_dc} != scheduler num_dc="
                f"{scheduler.num_datacenters} (DEFER-slot miscount? see "
                f"crd.baseline.infer_num_dc) — falling back to the infos layout"
            )
            return None

        bs = scheduler.batch_size
        # Move to CPU and to numpy/list for the scheduler call (sched expects
        # plain Python sequences). Keep grads disabled since baseline is
        # purely a target signal, never a function of policy params.
        green_np = green_ratio.detach().cpu().tolist()
        queue_np = queue_sizes.detach().cpu().tolist()

        out = [[ [0] * bs for _ in range(T) ] for _ in range(B)]
        zero_action = [0] * bs
        for b in range(B):
            for t in range(T):
                obs_dict = {
                    "dc_green_ratio": green_np[b][t],
                    "dc_queue_sizes": queue_np[b][t],
                }
                try:
                    action = scheduler.schedule(obs_dict)
                    out[b][t] = [int(a) for a in action]
                except Exception:
                    out[b][t] = list(zero_action)

        tensor = torch.tensor(out, dtype=torch.long)
        # Place on the same device as actual_action so M2.4 gather doesn't
        # need to cross-device.
        ref = batch.get(Columns.ACTIONS)
        if isinstance(ref, torch.Tensor):
            tensor = tensor.to(ref.device)
        return tensor

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
            # v5 fix (config-gated): nvec[0] counts ACTION CHOICES, which is
            # num_dc + 1 when the global DEFER slot is enabled — NOT the DC
            # count. Building the baseline with num_dc=6 on a 5-DC env made
            # every obs-grid alignment check fail silently (5 != 6): the
            # baseline fell back to the off-by-two infos layout (ΔQ/σ²
            # intermittently skipped) and Δr degraded to a constant 0 for the
            # router. Infer the true DC count from the observation space.
            if bool(crd_cfg.get("infer_num_dc", False)):
                inferred = self._infer_num_dc_from_obs_space(module)
                if inferred is None:
                    logger.warning(
                        f"[CRD] {module_id!r}: infer_num_dc=True but the "
                        f"observation space was not recognisable — keeping "
                        f"nvec[0]={num_dc}, which may include the DEFER slot"
                    )
                elif inferred != num_dc:
                    logger.info(
                        f"[CRD] {module_id!r}: baseline num_dc corrected "
                        f"{num_dc} -> {inferred} (observation-space inference; "
                        f"nvec[0] includes the DEFER slot)"
                    )
                    num_dc = inferred
            green_w = float(
                crd_cfg.get("green_weight", _DEFAULT_BASELINE_GREEN_WEIGHT)
            )
            # 2026-07-25: the counterfactual baseline is configurable via
            # crd.baseline.kind. Motivation: GreenQueueBalanced OUTPERFORMS the
            # learned router on some regimes (C-regime: GQB 0.136 vs RL 0.152),
            # which makes dQ = Q(action) - Q(baseline) systematically NEGATIVE
            # (observed dq_mean ~ -14) and degenerates the routing credit into a
            # constant penalty. A weaker, same-information baseline restores a
            # signed signal. Default keeps the historical GreenQueueBalanced.
            kind = str(crd_cfg.get("kind", "green_queue_balanced")).strip().lower()
            sched = self._build_baseline_by_kind(kind, num_dc, batch_size, green_w)
            self._crd_baseline_schedulers[module_id] = sched
            logger.info(
                f"[CRD] built {kind!r} baseline for {module_id!r}: "
                f"num_dc={num_dc}, batch_size={batch_size}, green_weight={green_w}"
            )
            return sched
        except Exception as e:
            logger.warning(
                f"[CRD] failed to build baseline scheduler for {module_id!r}: {e}"
            )
            self._crd_baseline_schedulers[module_id] = None
            return None

    def _compute_policy_self_baseline_action(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """COMA-style self-baseline: draw the counterfactual action from the
        policy's OWN per-slot distribution instead of a heuristic router.

        Rationale (2026-07-25): a heuristic baseline that outperforms the
        learned router makes dQ = Q(a) - Q(a~heuristic) systematically
        negative, so routing credit degenerates into a constant penalty. With
        a~pi the estimator is an unbiased single-sample version of the COMA
        counterfactual advantage Q(s,a) - E_{a'~pi}[Q(s,a')], which is
        centred at zero by construction and cannot be dominated by a stronger
        baseline. Writes COL_CRD_BASELINE_ACTION in the same layout as the
        heuristic path, so every downstream stage is unchanged.
        """
        logits = batch.get(Columns.ACTION_DIST_INPUTS)
        actions = batch.get(Columns.ACTIONS)
        if logits is None or actions is None:
            return
        if not isinstance(logits, torch.Tensor):
            return
        act = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        n_slots = int(act.shape[-1]) if act.dim() >= 1 else 0
        if n_slots <= 0 or logits.shape[-1] % n_slots != 0:
            if not self._crd_dq_align_warned.get(f"{module_id}:self", False):
                self._crd_dq_align_warned[f"{module_id}:self"] = True
                logger.warning(
                    f"[CRD] policy_self baseline: cannot split logits "
                    f"{tuple(logits.shape)} into {n_slots} slots for "
                    f"{module_id!r}; baseline action skipped."
                )
            return
        n_choices = logits.shape[-1] // n_slots
        per_slot = logits.reshape(*logits.shape[:-1], n_slots, n_choices)
        probs = torch.softmax(per_slot.float(), dim=-1)
        flat = probs.reshape(-1, n_choices)
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
        batch[COL_CRD_BASELINE_ACTION] = sampled.reshape(probs.shape[:-1]).long()

    @staticmethod
    def _build_baseline_by_kind(kind: str, num_dc: int, batch_size: int,
                                green_weight: float) -> Any:
        """Construct the counterfactual baseline router named by `kind`.

        Supported kinds (all read the SAME observation the policy sees, so the
        counterfactual stays information-fair):
          green_queue_balanced  green score + queue balancing (default, strong)
          green_aware           green score only (no queue term, weaker)
          min_queue             load-only routing, green-blind (weak)
          round_robin           fixed rotation, state-blind (weakest)
          policy_self           COMA-style: baseline drawn from the policy's own
                                distribution (no scheduler; returns a marker so
                                the caller's gating still fires)
        Unknown kinds fall back to green_queue_balanced with a warning.
        """
        if kind == "policy_self":
            return _PolicySelfBaselineMarker(num_dc, batch_size)
        from src.baselines.global_schedulers import (
            GreenAwareGlobalScheduler,
            MinQueueGlobalScheduler,
            RoundRobinGlobalScheduler,
        )

        if kind == "green_queue_balanced":
            return GreenQueueBalancedGlobalScheduler(
                num_datacenters=num_dc, batch_size=batch_size,
                green_weight=green_weight,
            )
        if kind == "green_aware":
            return GreenAwareGlobalScheduler(
                num_datacenters=num_dc, batch_size=batch_size)
        if kind == "min_queue":
            return MinQueueGlobalScheduler(
                num_datacenters=num_dc, batch_size=batch_size)
        if kind == "round_robin":
            return RoundRobinGlobalScheduler(
                num_datacenters=num_dc, batch_size=batch_size)
        logger.warning(
            f"[CRD] unknown crd.baseline.kind={kind!r}; "
            "falling back to green_queue_balanced"
        )
        return GreenQueueBalancedGlobalScheduler(
            num_datacenters=num_dc, batch_size=batch_size,
            green_weight=green_weight,
        )

    def _read_module_baseline_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.baseline` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("baseline", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ M4

    def _get_or_build_local_baseline_scheduler(
        self, module_id: ModuleID
    ) -> Optional[Any]:
        """
        Lazily construct the BestFit local-scheduling baseline for a Discrete
        (local-agent) module. Cached after first build. Returns None for
        modules with a MultiDiscrete action space (the global router), which
        are handled by `_get_or_build_baseline_scheduler` instead.
        """
        if module_id in self._crd_local_baseline_schedulers:
            return self._crd_local_baseline_schedulers[module_id]
        try:
            module = self.module[module_id].unwrapped()
            action_space = getattr(module, "action_space", None)
            from gymnasium import spaces  # local import; avoid hard top-level dep

            if not isinstance(action_space, spaces.Discrete):
                # Global router (MultiDiscrete) — not a local-scheduling module.
                self._crd_local_baseline_schedulers[module_id] = None
                return None
            # action_space.n = max_vms + 1 (index 0 = NoAssign, i = VM i-1).
            num_vms = max(1, int(action_space.n) - 1)
            sched = BestFitLocalScheduler(num_vms=num_vms)
            self._crd_local_baseline_schedulers[module_id] = sched
            logger.info(
                f"[CRD] built BestFit local baseline for {module_id!r}: "
                f"num_vms={num_vms} (action_dim={action_space.n})"
            )
            return sched
        except Exception as e:
            logger.warning(
                f"[CRD] failed to build local baseline scheduler for "
                f"{module_id!r}: {e}"
            )
            self._crd_local_baseline_schedulers[module_id] = None
            return None

    def _compute_local_baseline_action(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        M4.2a: compute the BestFit baseline VM choice ã_local per transition
        and write `batch[COL_CRD_BASELINE_ACTION]` as a (B, T) long tensor,
        aligned with the local agent's actual Discrete action.

        Reads from the padded obs grid (the same layout the GTrXL local module
        consumes): the inner observation's `vm_available_pes` /
        `next_cloudlet_pes`, plus the sibling `action_mask`. Obs is robust to
        PPO minibatching (unlike infos), so M2.4's ΔQ/σ² gather aligns directly.

        Skips silently (no baseline written) if the obs grid or action_mask is
        unavailable — in that case ΔQ_local/σ²_local and R^scheduling are not
        produced, and M5 falls back to ρ_scheduling = floor for this module.
        """
        sched = self._get_or_build_local_baseline_scheduler(module_id)
        if sched is None:
            return

        baseline = self._compute_local_baseline_from_obs(module_id, batch, sched)
        if baseline is not None:
            batch[COL_CRD_BASELINE_ACTION] = baseline

    def _compute_local_baseline_from_obs(
        self, module_id: ModuleID, batch: Dict[str, Any], scheduler: Any
    ) -> Optional[torch.Tensor]:
        """
        Build the (B, T) BestFit baseline action by iterating the padded obs
        grid. Returns None if the required obs fields are missing.

        Cost: B*T scheduler.schedule() calls — each is an O(num_vms) numpy
        scan, sub-millisecond, so per-minibatch cost is well under a second.
        """
        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            self._warn_local_baseline_once(module_id, "obs is not a dict")
            return None
        # action_mask sits as a sibling of "observation" (see GTrXLMaskedAction
        # RLModule._extract_obs_and_mask). The real obs fields are nested under
        # "observation" when the connector wrapper is active.
        action_mask = obs.get("action_mask")
        inner = obs.get("observation", obs)
        if not isinstance(inner, dict):
            inner = obs
        vm_avail = inner.get("vm_available_pes")
        next_pes = inner.get("next_cloudlet_pes")
        if not isinstance(action_mask, torch.Tensor):
            self._warn_local_baseline_once(module_id, "action_mask missing")
            return None
        if not isinstance(vm_avail, torch.Tensor) or not isinstance(
            next_pes, torch.Tensor
        ):
            self._warn_local_baseline_once(
                module_id,
                f"vm_available_pes/next_cloudlet_pes missing; "
                f"inner keys={sorted(inner.keys()) if isinstance(inner, dict) else '?'}",
            )
            return None

        # Coerce a leading (B, T, ...) or flat (N, ...) layout to (B, T, ...).
        def _to_bt(t: torch.Tensor) -> Optional[torch.Tensor]:
            if t.dim() >= 3:
                return t
            if t.dim() == 2:
                return t.unsqueeze(1)  # (N, F) → (N, 1, F)
            return None

        am = _to_bt(action_mask)
        va = _to_bt(vm_avail)
        if am is None or va is None:
            return None
        B, T = am.shape[0], am.shape[1]
        if va.shape[0] != B or va.shape[1] != T:
            return None
        # next_cloudlet_pes is (B, T, 1) (env stores a length-1 vector).
        np_ = next_pes
        if np_.dim() == 3:
            np_grid = np_
        elif np_.dim() == 2:
            np_grid = np_.unsqueeze(1)
        elif np_.dim() == 1:
            np_grid = np_.view(B, T, 1) if np_.numel() == B * T else None
        else:
            np_grid = None
        if np_grid is None:
            return None

        am_np = am.detach().cpu().numpy()
        va_np = va.detach().cpu().tolist()
        np_np = np_grid.detach().cpu().numpy()

        out = [[0] * T for _ in range(B)]
        for b in range(B):
            for t in range(T):
                local_obs = {
                    "vm_available_pes": va_np[b][t],
                    "next_cloudlet_pes": float(np_np[b, t].flat[0]),
                }
                try:
                    out[b][t] = int(scheduler.schedule(local_obs, am_np[b, t]))
                except Exception:
                    out[b][t] = 0  # NoAssign fallback

        tensor = torch.tensor(out, dtype=torch.long)
        ref = batch.get(Columns.ACTIONS)
        if isinstance(ref, torch.Tensor):
            tensor = tensor.to(ref.device)
        return tensor

    def _warn_local_baseline_once(self, module_id: ModuleID, reason: str) -> None:
        if not self._crd_local_baseline_warned.get(module_id, False):
            self._crd_local_baseline_warned[module_id] = True
            logger.warning(
                f"[CRD] local baseline obs path unavailable for {module_id!r} "
                f"({reason}); ΔQ_local/σ²_local skipped — ρ_scheduling falls "
                "back to the floor for this module."
            )

    def _compute_local_dr(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        M4.2c: reward-level Δr_local fallback for the local scheduling layer.

        Proxy: the PE-waste difference between the BestFit baseline VM choice
        and the agent's actual choice. Like the routing Δr, we use a
        "lower-is-better" cost (PE waste) and define
            Δr_local = α · (waste(ã_baseline) − waste(a_actual)),
        so Δr_local > 0 when the agent wastes fewer PEs than BestFit. Other
        reward terms are assumed common between the two VM choices on the same
        step and cancel in the difference.

        Skips silently when the baseline action or obs grid is unavailable.
        """
        baseline_action = batch.get(COL_CRD_BASELINE_ACTION)
        if not isinstance(baseline_action, torch.Tensor):
            return
        cfg = self._read_module_local_dr_config(module_id)
        alpha = float(cfg.get("alpha", _DEFAULT_ALPHA_DR_LOCAL))

        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            return
        inner = obs.get("observation", obs)
        if not isinstance(inner, dict):
            inner = obs
        vm_avail = inner.get("vm_available_pes")
        next_pes = inner.get("next_cloudlet_pes")
        if not isinstance(vm_avail, torch.Tensor) or not isinstance(
            next_pes, torch.Tensor
        ):
            return

        if vm_avail.dim() == 2:
            vm_avail = vm_avail.unsqueeze(1)
        if vm_avail.dim() != 3:
            return
        B, T, _V = vm_avail.shape
        if baseline_action.dim() != 2 or baseline_action.shape[:2] != (B, T):
            return

        actual_bt = self._align_local_action_to_bt(
            batch.get(Columns.ACTIONS), B, T, batch.get(Columns.LOSS_MASK)
        )
        if actual_bt is None:
            return

        if next_pes.dim() == 3:
            next_grid = next_pes
        elif next_pes.dim() == 2:
            next_grid = next_pes.unsqueeze(1)
        elif next_pes.dim() == 1 and next_pes.numel() == B * T:
            next_grid = next_pes.view(B, T, 1)
        else:
            return

        import numpy as np
        va_np = vm_avail.detach().cpu().numpy().astype(np.float64)
        np_np = next_grid.detach().cpu().numpy().astype(np.float64)
        act_np = actual_bt.detach().cpu().numpy().astype(np.int64)
        base_np = baseline_action.detach().cpu().numpy().astype(np.int64)

        def _waste(vm_row, demand, a):
            # a == 0 (NoAssign): the cloudlet is deferred → its full demand is
            # the opportunity "waste" (a deterministic, comparable penalty).
            if a <= 0:
                return float(demand)
            idx = a - 1
            if idx >= vm_row.shape[0]:
                return float(demand)
            w = float(vm_row[idx]) - float(demand)
            return w if w >= 0.0 else float(demand)

        dr = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                demand = float(np_np[b, t].flat[0])
                vm_row = va_np[b, t]
                w_actual = _waste(vm_row, demand, int(act_np[b, t]))
                w_base = _waste(vm_row, demand, int(base_np[b, t]))
                dr[b, t] = alpha * (w_base - w_actual)

        tensor = torch.tensor(dr, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor):
            tensor = tensor.to(ref.device, dtype=ref.dtype)
        batch[COL_CRD_DR] = tensor

    @staticmethod
    def _align_local_action_to_bt(
        action: Any, B: int, T: int, loss_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Coerce a Discrete local action tensor to padded (B, T); None if impossible."""
        if not isinstance(action, torch.Tensor):
            return None
        a = action.long()
        if a.dim() == 2 and a.shape[0] == B and a.shape[1] == T:
            return a
        if a.dim() == 1 and a.numel() == B * T:
            return a.reshape(B, T)
        # Sequence-packed (N_valid,): scatter via loss_mask.
        if a.dim() == 1 and isinstance(loss_mask, torch.Tensor):
            flat_mask = loss_mask.reshape(-1).bool()
            if int(flat_mask.sum().item()) != a.shape[0]:
                return None
            out = torch.zeros(B * T, dtype=torch.long, device=a.device)
            out[flat_mask] = a
            return out.reshape(B, T)
        return None

    def _read_module_local_dr_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.delta_r_local` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("delta_r_local", {}) or {}
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

        # Pull loss_mask if present — needed when actions are sequence-packed
        # to N_valid while q_ensemble is the padded (B, T, ...) form returned
        # by the GTrXL forward pass. RLlib's connector pipeline gives us
        # actions in unpadded form alongside `loss_mask` of shape (B, T).
        loss_mask = batch.get(Columns.LOSS_MASK)

        try:
            delta_q, sigma2_tot = self._compute_dq_and_sigma2_values(
                q_ensemble=q_ensemble,
                actual_action=actual_action,
                baseline_action=baseline_action,
                loss_mask=loss_mask,
            )
        except Exception as e:
            # Warn-once per module: PPO multi-epoch SGD calls compute_loss
            # ~150+ times per iter; spamming this log every minibatch buries
            # everything else. The first occurrence carries enough diagnostic
            # detail to debug the root cause separately (M2.3 baseline_action
            # layout mismatch, currently under investigation).
            if not self._crd_dq_align_warned.get(module_id, False):
                self._crd_dq_align_warned[module_id] = True
                # Probe a representative info entry to help diagnose layout
                # mismatch (we're still figuring out why baseline_action's
                # length disagrees with both B*T and N_valid in some calls).
                infos_probe = batch.get(Columns.INFOS)
                infos_len = (
                    len(infos_probe)
                    if infos_probe is not None and hasattr(infos_probe, "__len__")
                    else "?"
                )
                logger.warning(
                    f"[CRD] dq/sigma2 alignment failed for {module_id!r}: {e}; "
                    f"q_ensemble.shape={tuple(q_ensemble.shape)}, "
                    f"actual_action.shape={tuple(actual_action.shape)}, "
                    f"baseline_action.shape={tuple(baseline_action.shape)}, "
                    f"loss_mask.shape="
                    f"{None if loss_mask is None else tuple(loss_mask.shape)}, "
                    f"len(infos)={infos_len}. "
                    "ΔQ/σ² will be skipped for this module — M3 blending will "
                    "fall back to Δr-only path. Suppressing further alignment "
                    "warnings for this module."
                )
            return

        # Detach: M5 will broadcast these against advantages — they should
        # not flow gradient back through the q-heads (q_loss in M1.2 is the
        # canonical training signal for the ensemble).
        batch[COL_CRD_DQ] = delta_q.detach()
        batch[COL_CRD_SIGMA2] = sigma2_tot.detach()

    @staticmethod
    def _gather_q_chosen(
        q_ensemble: torch.Tensor,
        action: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gather Q values across the K ensemble for the given action.

        q_ensemble is always the padded (B, T, ...) tensor returned from the
        forward pass (shape 4 for local, shape 5 for global). The `action`
        tensor's layout depends on RLlib's connector pipeline — it can be
        any of:

          * Padded (B, T, ...) — already aligned with q_ensemble
          * Flat (B*T, ...) — needs reshape to (B, T, ...)
          * Sequence-packed (N_valid, ...) where N_valid < B*T — RLlib's
            "unpadded" form; we mask q_ensemble down to valid positions
            using `loss_mask`

        Output is always (B, T, K) — for the masked-input path we scatter
        valid positions back into a zero-filled (B, T, K) tensor so the
        downstream M5 reshape against `advantages` (also (B, T)) is trivial.
        """
        if q_ensemble.dim() == 4:
            return CRDPPOTorchLearner._gather_q_chosen_local(
                q_ensemble, action, loss_mask
            )
        if q_ensemble.dim() == 5:
            return CRDPPOTorchLearner._gather_q_chosen_global(
                q_ensemble, action, loss_mask
            )
        raise RuntimeError(
            f"Unexpected q_ensemble dim {q_ensemble.dim()} (shape "
            f"{tuple(q_ensemble.shape)}); expected 4 (local) or 5 (global)."
        )

    @staticmethod
    def _gather_q_chosen_local(
        q_ensemble: torch.Tensor,
        action: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Local agent (Discrete action). q (B, T, K, A) → returns (B, T, K)."""
        B, T, K, A = q_ensemble.shape
        act = action.long()

        # Case 1: already (B, T)
        if act.dim() == 2 and act.shape[0] == B and act.shape[1] == T:
            idx = act.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
            return q_ensemble.gather(-1, idx).squeeze(-1)

        # Case 2: flat (B*T,)
        if act.dim() == 1 and act.numel() == B * T:
            act_bt = act.reshape(B, T)
            idx = act_bt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
            return q_ensemble.gather(-1, idx).squeeze(-1)

        # Case 3: sequence-packed (N_valid,) — mask q to N_valid, gather, scatter back.
        if act.dim() == 1 and loss_mask is not None:
            return CRDPPOTorchLearner._gather_local_via_mask(
                q_ensemble, act, loss_mask
            )

        raise ValueError(
            f"local agent: cannot align action shape {tuple(action.shape)} with "
            f"q_ensemble shape {tuple(q_ensemble.shape)} "
            f"(loss_mask={'available' if loss_mask is not None else 'absent'})"
        )

    @staticmethod
    def _gather_q_chosen_global(
        q_ensemble: torch.Tensor,
        action: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Global agent (MultiDiscrete). q (B, T, K, bs, nd) → returns (B, T, K)."""
        B, T, K, bs, nd = q_ensemble.shape
        act = action.long()

        # Case 1: already (B, T, bs)
        if (
            act.dim() == 3
            and act.shape[0] == B
            and act.shape[1] == T
            and act.shape[2] == bs
        ):
            idx = act.unsqueeze(2).unsqueeze(-1).expand(-1, -1, K, -1, 1)
            q_pc = q_ensemble.gather(-1, idx).squeeze(-1)  # (B, T, K, bs)
            return q_pc.mean(dim=-1)  # (B, T, K)

        # Case 2: flat (B*T, bs)
        if act.dim() == 2 and act.shape[0] == B * T and act.shape[1] == bs:
            act_btbs = act.reshape(B, T, bs)
            idx = act_btbs.unsqueeze(2).unsqueeze(-1).expand(-1, -1, K, -1, 1)
            q_pc = q_ensemble.gather(-1, idx).squeeze(-1)
            return q_pc.mean(dim=-1)

        # Case 3: sequence-packed (N_valid, bs) — mask q, gather, scatter back.
        if act.dim() == 2 and act.shape[1] == bs and loss_mask is not None:
            return CRDPPOTorchLearner._gather_global_via_mask(
                q_ensemble, act, loss_mask
            )

        raise ValueError(
            f"global agent: cannot align action shape {tuple(action.shape)} with "
            f"q_ensemble shape {tuple(q_ensemble.shape)} "
            f"(loss_mask={'available' if loss_mask is not None else 'absent'})"
        )

    @staticmethod
    def _gather_local_via_mask(
        q_ensemble: torch.Tensor,  # (B, T, K, A)
        action: torch.Tensor,       # (N_valid,)
        loss_mask: torch.Tensor,    # (B, T) or compatible
    ) -> torch.Tensor:
        B, T, K, A = q_ensemble.shape
        flat_mask = loss_mask.reshape(-1).bool()
        n_valid = int(flat_mask.sum().item())
        if n_valid != action.shape[0]:
            raise ValueError(
                f"local mask path: loss_mask N_valid={n_valid} != action N={action.shape[0]}"
            )
        # Mask q_ensemble (B*T, K, A) → (N_valid, K, A) → gather → (N_valid, K)
        q_flat = q_ensemble.reshape(B * T, K, A)
        q_valid = q_flat[flat_mask]                               # (N_valid, K, A)
        idx = action.view(-1, 1, 1).expand(-1, K, 1)
        q_chosen_valid = q_valid.gather(-1, idx).squeeze(-1)     # (N_valid, K)
        # Scatter back to (B, T, K), zero at masked positions.
        out = torch.zeros(B * T, K, dtype=q_ensemble.dtype, device=q_ensemble.device)
        out[flat_mask] = q_chosen_valid
        return out.reshape(B, T, K)

    @staticmethod
    def _gather_global_via_mask(
        q_ensemble: torch.Tensor,  # (B, T, K, bs, nd)
        action: torch.Tensor,       # (N_valid, bs)
        loss_mask: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:
        B, T, K, bs, nd = q_ensemble.shape
        flat_mask = loss_mask.reshape(-1).bool()
        n_valid = int(flat_mask.sum().item())
        if n_valid != action.shape[0]:
            raise ValueError(
                f"global mask path: loss_mask N_valid={n_valid} != action N={action.shape[0]}"
            )
        # Mask q_ensemble (B*T, K, bs, nd) → (N_valid, K, bs, nd)
        q_flat = q_ensemble.reshape(B * T, K, bs, nd)
        q_valid = q_flat[flat_mask]                                  # (N_valid, K, bs, nd)
        # gather along last dim with action (N_valid, bs)
        idx = action.unsqueeze(1).unsqueeze(-1).expand(-1, K, -1, 1)  # (N_valid, K, bs, 1)
        q_pc = q_valid.gather(-1, idx).squeeze(-1)                    # (N_valid, K, bs)
        q_per_step = q_pc.mean(dim=-1)                                # (N_valid, K)
        # Scatter back to (B, T, K)
        out = torch.zeros(B * T, K, dtype=q_ensemble.dtype, device=q_ensemble.device)
        out[flat_mask] = q_per_step
        return out.reshape(B, T, K)

    @staticmethod
    def _compute_dq_and_sigma2_values(
        *,
        q_ensemble: torch.Tensor,
        actual_action: torch.Tensor,
        baseline_action: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_q:    (B, T)   = μ(s, a) - μ(s, ã)
            sigma2_tot: (B, T)   = σ²(s, a) + σ²(s, ã), always ≥ 0
        Both averages/variances are taken over the K ensemble dimension.
        Masked-out timesteps in the sequence-packed path are zero-filled in
        both outputs, which is benign because M5 reweights advantages via
        `adv * rho_routing` and advantages are also zero/masked at those slots.
        """
        q_a = CRDPPOTorchLearner._gather_q_chosen(q_ensemble, actual_action, loss_mask)
        q_b = CRDPPOTorchLearner._gather_q_chosen(q_ensemble, baseline_action, loss_mask)
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
        if baseline_action is None:
            return

        sched = self._get_or_build_baseline_scheduler(module_id)
        if sched is None:
            return  # No routing scheduler available (e.g., local agent) → skip.

        cfg = self._read_module_dr_config(module_id)
        alpha = float(cfg.get("alpha", _DEFAULT_ALPHA_DR))

        # v5.3 (crd.delta_r.mode="green"): reward-ALIGNED Δr for the gdpd
        # regime. The legacy load-std proxy tracks the α·L term of the OLD
        # reward formula; under gdpd the reward is per-action carbon +
        # completion (alpha=0 — load balance is NOT in the reward), so the
        # load-std Δr measures a quantity with no guaranteed relation to the
        # reward and systematically disagrees with ΔQ in sign (observed
        # dr=+19 vs dq=-25 on rwtight). Green mode scores each slot by the
        # decision-time green-fit of the DC it picks, agent minus baseline;
        # DEFER slots contribute 0 (immediate green-fit is undefined for
        # defer — long-horizon defer value is the ΔQ path's job).
        if str(cfg.get("mode", "load_std")) == "green":
            dr_tensor = self._compute_dr_green_from_obs(batch, sched, alpha)
            if dr_tensor is not None:
                batch[COL_CRD_DR] = dr_tensor
                return
            # ingredients unavailable → fall through to the legacy paths

        # ── Preferred path: obs-based, shape-aligned with ΔQ ──────────────
        # Reads dc_queue_sizes from the padded (B, T, num_dc) obs grid — the
        # same grid baseline_action lives on — so Δr comes out (B, T),
        # guaranteed to align with ΔQ/σ² in M3's blend. This avoids the
        # `infos`-derived path's failure when PPO drops infos from a
        # minibatch (len(infos)=0 → empty Δr → blend shape mismatch).
        dr_tensor = self._compute_dr_from_obs(batch, sched, alpha)
        if dr_tensor is not None:
            batch[COL_CRD_DR] = dr_tensor
            return

        # ── Fallback: legacy infos-based path ─────────────────────────────
        actual_action = batch.get(Columns.ACTIONS)
        infos = batch.get(Columns.INFOS)
        if actual_action is None or infos is None:
            return
        try:
            dr_list = self._compute_dr_values(
                infos=infos,
                actual_actions=actual_action,
                baseline_actions=baseline_action,
                num_dc=sched.num_datacenters,
                alpha=alpha,
            )
        except Exception as e:
            logger.warning(f"[CRD] Δr compute failed for {module_id!r}: {e}")
            return
        if not dr_list:
            return  # empty → don't write a (0,) tensor that breaks M3's blend
        dr_tensor = torch.tensor(dr_list, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor) and ref.numel() == dr_tensor.numel():
            dr_tensor = dr_tensor.reshape(ref.shape).to(ref.device, dtype=ref.dtype)
        batch[COL_CRD_DR] = dr_tensor

    def _compute_dr_from_obs(
        self, batch: Dict[str, Any], scheduler: Any, alpha: float
    ) -> Optional[torch.Tensor]:
        """
        Compute Δr per (b, t) cell from the padded obs grid, producing a
        (B, T) tensor aligned with ΔQ/σ².

        Δr_t = α · (std(baseline_q) − std(actual_q)), where each per-DC queue
        is the obs `dc_queue_sizes` plus the per-DC cloudlet counts from the
        respective routing. Returns None if obs / baseline_action layouts are
        not usable (caller falls back to the infos path).
        """
        obs = batch.get(Columns.OBS)
        if isinstance(obs, dict) and isinstance(obs.get("observation"), dict):
            obs = obs["observation"]
        if not isinstance(obs, dict):
            return None
        queue_sizes = obs.get("dc_queue_sizes")
        if not isinstance(queue_sizes, torch.Tensor) or queue_sizes.dim() != 3:
            return None
        B, T, num_dc = queue_sizes.shape
        if num_dc != scheduler.num_datacenters:
            return None

        baseline_action = batch.get(COL_CRD_BASELINE_ACTION)
        if (
            not isinstance(baseline_action, torch.Tensor)
            or baseline_action.dim() != 3
            or baseline_action.shape[:2] != (B, T)
        ):
            return None
        bs = baseline_action.shape[2]

        # Align actual routing to (B, T, bs). It may arrive padded (B, T, bs)
        # or sequence-packed (N, bs); in the packed case use loss_mask to
        # scatter valid rows back into the padded grid (zeros elsewhere — those
        # cells are masked out of the loss downstream anyway).
        actual_bt = self._align_routing_to_bt(
            batch.get(Columns.ACTIONS), B, T, bs, batch.get(Columns.LOSS_MASK)
        )
        if actual_bt is None:
            return None

        import numpy as np
        q_np = queue_sizes.detach().cpu().numpy()          # (B, T, num_dc)
        act_np = actual_bt.detach().cpu().numpy().astype(np.int64)
        base_np = baseline_action.detach().cpu().numpy().astype(np.int64)

        dr = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                base_q = q_np[b, t]
                actual_inc = np.bincount(act_np[b, t], minlength=num_dc)[:num_dc]
                baseline_inc = np.bincount(base_np[b, t], minlength=num_dc)[:num_dc]
                actual_std = float((base_q + actual_inc).std())
                baseline_std = float((base_q + baseline_inc).std())
                dr[b, t] = alpha * (baseline_std - actual_std)

        tensor = torch.tensor(dr, dtype=torch.float32)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor):
            tensor = tensor.to(ref.device, dtype=ref.dtype)
        return tensor

    def _compute_dr_green_from_obs(
        self, batch: Dict[str, Any], scheduler: Any, alpha: float
    ) -> Optional[torch.Tensor]:
        """
        v5.3 green-mode Δr: per-(b,t) mean over routing slots of the
        decision-time green-fit difference between the taken DC and the
        baseline DC,
            Δr = α · mean_slots[ g_ratio(dc_a) − g_ratio(dc_ã) ],
        with DEFER slots contributing 0. Positive ⇔ the agent routes to
        greener DCs than the heuristic — the same orientation ΔQ is meant
        to have, so the soft blend mixes two estimators of the SAME quantity.
        Returns None when the obs grid / baseline layout is unusable.
        """
        obs = batch.get(Columns.OBS)
        if isinstance(obs, dict) and isinstance(obs.get("observation"), dict):
            obs = obs["observation"]
        if not isinstance(obs, dict):
            return None
        green = obs.get("dc_green_ratio")
        if not isinstance(green, torch.Tensor):
            return None
        if green.dim() == 2:
            green = green.view(green.shape[0], 1, green.shape[1])
        if green.dim() != 3:
            return None
        B, T, num_dc = green.shape
        if num_dc != scheduler.num_datacenters:
            return None
        baseline_action = batch.get(COL_CRD_BASELINE_ACTION)
        if (
            not isinstance(baseline_action, torch.Tensor)
            or baseline_action.dim() != 3
            or baseline_action.shape[:2] != (B, T)
        ):
            return None
        bs = baseline_action.shape[2]
        actual_bt = self._align_routing_to_bt(
            batch.get(Columns.ACTIONS), B, T, bs, batch.get(Columns.LOSS_MASK)
        )
        if actual_bt is None:
            return None
        g = green.detach().float()
        # Extra zero column at index num_dc: the DEFER action gathers 0.
        gpad = torch.cat(
            [g, torch.zeros(B, T, 1, dtype=g.dtype, device=g.device)], dim=-1
        )
        a = actual_bt.long().clamp(0, num_dc).to(g.device)
        b = baseline_action.long().clamp(0, num_dc).to(g.device)
        dr = alpha * (torch.gather(gpad, 2, a) - torch.gather(gpad, 2, b)).mean(dim=2)
        ref = batch.get(Columns.REWARDS, batch.get(Postprocessing.ADVANTAGES))
        if isinstance(ref, torch.Tensor):
            dr = dr.to(ref.device, dtype=ref.dtype)
        return dr

    @staticmethod
    def _align_routing_to_bt(
        action: Any, B: int, T: int, bs: int, loss_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Coerce a routing-action tensor to padded (B, T, bs); None if impossible."""
        if not isinstance(action, torch.Tensor):
            return None
        a = action.long()
        if a.dim() == 3 and a.shape[0] == B and a.shape[1] == T and a.shape[2] == bs:
            return a
        if a.dim() == 2 and a.shape[0] == B * T and a.shape[1] == bs:
            return a.reshape(B, T, bs)
        # Sequence-packed (N_valid, bs): scatter via loss_mask.
        if a.dim() == 2 and a.shape[1] == bs and isinstance(loss_mask, torch.Tensor):
            flat_mask = loss_mask.reshape(-1).bool()
            if int(flat_mask.sum().item()) != a.shape[0]:
                return None
            out = torch.zeros(B * T, bs, dtype=torch.long, device=a.device)
            out[flat_mask] = a
            return out.reshape(B, T, bs)
        return None

    def _read_module_dr_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.delta_r` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("delta_r", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ M3

    def _compute_r_routing(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        Soft-blend ΔQ and Δr into the routing-responsibility signal R^routing
        used by M5's advantage rewrite (global routing layer).

            c(t)         = exp(-σ²_tot(t) / τ(t))
            R^routing_t  = c(t) · ΔQ_t + (1 - c(t)) · Δr_t

        τ is adaptive (driven by an EMA of σ²) so the c(t) curriculum
        smoothly transitions from "trust Δr" early in training (immature
        critic, high σ²) to "trust ΔQ" later (mature critic discriminates
        OOD actions). Per-module CRDBlender holds the EMA state.

        Skips silently if any of (ΔQ, Δr, σ²) is missing — those happen for
        non-routing modules where M2.4/M2.5 didn't produce outputs.
        """
        self._blend_dq_dr(module_id=module_id, batch=batch, out_col=COL_CRD_R_ROUTING)

    def _compute_r_scheduling(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        M4: soft-blend ΔQ_local and Δr_local into the scheduling-responsibility
        signal R^scheduling for a local-agent module. Identical blending math
        to :meth:`_compute_r_routing` — the only difference is the output column
        and that the per-module CRDBlender carries its own EMA state (keyed by
        the local module_id, so routing and scheduling never share a τ).
        """
        self._blend_dq_dr(
            module_id=module_id, batch=batch, out_col=COL_CRD_R_SCHEDULING
        )

    def _blend_dq_dr(
        self, *, module_id: ModuleID, batch: Dict[str, Any], out_col: str
    ) -> None:
        """
        Shared σ²-gated blend of (ΔQ, Δr) → out_col, plus c(t)/τ diagnostics.
        Used by both the routing (M3) and scheduling (M4) layers.
        """
        dq = batch.get(COL_CRD_DQ)
        dr = batch.get(COL_CRD_DR)
        sigma2 = batch.get(COL_CRD_SIGMA2)
        if not (
            isinstance(dq, torch.Tensor)
            and isinstance(dr, torch.Tensor)
            and isinstance(sigma2, torch.Tensor)
        ):
            return

        blender = self._get_or_build_blender(module_id)
        # v5.1 (config-gated crd.mask_padding): zero ΔQ/Δr/σ² at padded grid
        # cells and keep them out of the τ EMA. Padded cells carry garbage
        # magnitudes (all-zero pad actions → |Δr|≈51 vs ~0.5 valid) that
        # otherwise dominate the blend statistics and diagnostics.
        ema_mask = None
        if self._read_crd_mask_padding(module_id):
            m = self._bt_mask(batch, dq)
            if m is not None:
                dq = dq * m
                dr = dr.to(dq.dtype).to(dq.device) * m
                sigma2 = sigma2 * m
                ema_mask = m
        # v5.3 (config-gated crd.blender.sigma2_norm): divide σ² by the SAME
        # running value-target variance the critic/Q-loss normalisation uses,
        # so the gate compares RELATIVE epistemic uncertainty. Without this,
        # σ² grows with the (squared) return scale as training progresses,
        # the EMA-adaptive temperature τ = τ₀·exp(κ·σ̄²) explodes (observed
        # τ→7.9e4 on rwtight, gate saturated open from iter ~35, entropy
        # de-converged 1.74→4.19), and the epistemic gate is effectively dead.
        if bool(self._read_module_blender_config(module_id).get("sigma2_norm", False)):
            var_ema = self._vf_target_var_ema.get(module_id)
            if var_ema is not None and float(var_ema) > 1e-8:
                sigma2 = sigma2 / float(var_ema)
        try:
            r_blend, c_t, tau = blender.update_and_blend(
                dq=dq, dr=dr.to(dq.dtype).to(dq.device), sigma2=sigma2,
                ema_mask=ema_mask,
            )
        except Exception as e:
            logger.warning(f"[CRD] {out_col} blend failed for {module_id!r}: {e}")
            return

        # All three are detached: M5 will reweight advantages with this signal
        # but the gradient signal for the policy must come from PPO's GAE
        # path, not from σ²/Q-head. (ΔQ/σ² were already detached upstream
        # in M2.4; defensive .detach() here guards against future changes.)
        batch[out_col] = r_blend.detach()
        batch[COL_CRD_C_T] = c_t.detach()
        # τ is a scalar — store as a 0-d tensor for shape uniformity.
        batch[COL_CRD_TAU] = torch.tensor(tau, dtype=torch.float32, device=dq.device)

    def _get_or_build_blender(self, module_id: ModuleID) -> CRDBlender:
        """Lazily construct one CRDBlender per module_id (cached)."""
        if module_id in self._crd_blenders:
            return self._crd_blenders[module_id]
        cfg = self._read_module_blender_config(module_id)
        fixed_c = cfg.get("fixed_c")
        blender = CRDBlender(
            tau_0=float(cfg.get("tau_0", 1.0)),
            kappa=float(cfg.get("kappa", 0.5)),
            eta=float(cfg.get("eta", 0.05)),
            tau_mode=str(cfg.get("tau_mode", "exp")),  # v5.3: "linear" = scale-free τ
            ema_init=cfg.get("ema_init"),  # None ⇒ bootstrap on first batch
            fixed_c=None if fixed_c is None else float(fixed_c),  # ablation: EU off
        )
        self._crd_blenders[module_id] = blender
        logger.info(
            f"[CRD] built blender for {module_id!r}: "
            f"tau_0={blender.tau_0}, kappa={blender.kappa}, "
            f"eta={blender.eta}, tau_mode={blender.tau_mode}, "
            f"ema_init={blender.ema_init}, fixed_c={blender.fixed_c}"
        )
        return blender

    def _read_module_blender_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.blender` from the module's model_config; default to {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("blender", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ M5

    def _compute_responsibilities(
        self, *, module_id: ModuleID, batch: Dict[str, Any]
    ) -> None:
        """
        Per-transition responsibility weights ρ_k = |R_k| / Σ|R_k'|, with a
        floor at ρ_min (plan §3.4). Then reweight ADVANTAGES by ρ_routing so
        PPO's surrogate loss only updates the policy on the agent's
        controllable share of the credit.

        Outputs (each shape (B, T)):
          batch[crd_rho_forecast]    — logging only, NEVER multiplied
          batch[crd_rho_routing]     — multiplier on global ADVANTAGES
          batch[crd_rho_scheduling]  — multiplier on local ADVANTAGES

        Side effect (per-transition):
          global router  →  batch[Postprocessing.ADVANTAGES] *= ρ_routing
          local agent    →  batch[Postprocessing.ADVANTAGES] *= ρ_scheduling

        Each module carries only its own layer's signal: the global router's
        batch has R_routing (no R_scheduling), the local agent's batch has
        R_scheduling (no R_routing). The forecast share |R_forecast| is the
        common exogenous term in both denominators — it shields both layers
        from forecast error but is never itself applied to a gradient.

        Skips if neither R_routing nor R_scheduling is in the batch (the CF
        pipeline didn't run, e.g. a non-ensemble module).

        VALUE_TARGETS is intentionally untouched: the V-head should still
        learn the unbiased episode return so its bootstrapping for the
        Q-head TD target (M1.2) stays correct.
        """
        r_routing = batch.get(COL_CRD_R_ROUTING)
        r_scheduling = batch.get(COL_CRD_R_SCHEDULING)
        # Decide which responsibility share reweights *this* module's
        # advantage. A module is either the global router (R_routing) or a
        # local scheduler (R_scheduling); if both are somehow present we
        # prefer routing (preserves pre-M4 behaviour for the global module).
        if isinstance(r_routing, torch.Tensor):
            own_kind = "routing"
            ref = r_routing
        elif isinstance(r_scheduling, torch.Tensor):
            own_kind = "scheduling"
            ref = r_scheduling
        else:
            return

        cfg = self._read_module_responsibility_config(module_id)
        rho_min = float(cfg.get("rho_min", _DEFAULT_RHO_MIN))

        # Components — fall back to zeros when this module's layer doesn't
        # produce a given term (the global router has no R_scheduling and
        # vice-versa).
        r_f = batch.get(COL_CRD_FORECAST)
        zeros_like_r = torch.zeros_like(ref)
        abs_f = (r_f.abs() if isinstance(r_f, torch.Tensor) else zeros_like_r).to(
            ref.dtype
        )
        abs_r = (
            r_routing.abs() if isinstance(r_routing, torch.Tensor) else zeros_like_r
        ).to(ref.dtype)
        abs_s = (
            r_scheduling.abs()
            if isinstance(r_scheduling, torch.Tensor)
            else zeros_like_r
        ).to(ref.dtype)

        # Align shapes — forecast may have come in shaped (B, T) but also may
        # have been reshaped to match REWARDS in M2.2; coerce to the owning
        # signal's shape if possible.
        if abs_f.shape != ref.shape:
            try:
                abs_f = abs_f.reshape(ref.shape)
            except RuntimeError:
                abs_f = zeros_like_r
        if abs_s.shape != ref.shape:
            try:
                abs_s = abs_s.reshape(ref.shape)
            except RuntimeError:
                abs_s = zeros_like_r
        if abs_r.shape != ref.shape:
            try:
                abs_r = abs_r.reshape(ref.shape)
            except RuntimeError:
                abs_r = zeros_like_r

        # ── v5 (config-gated) share repairs ────────────────────────────────
        # normalize_shares: the three |R_k| live in incommensurate units —
        # R_routing is ΔQ-derived and grows with the critic's value scale
        # (observed |ΔQ| ~ 10–60) while R_forecast stays in reward-proxy
        # units (~0.1–0.2) — so raw ratios let one source swamp the rest
        # (observed ρ_routing→0.99, ρ_forecast→0.03: quarantine inert).
        # Divide each source by a running EMA of its own batch-mean magnitude
        # so the shares compare RELATIVE anomaly, not native units.
        # Padding-aware statistics (v5.1): all EMA statistics below exclude
        # padded grid cells when crd.mask_padding is on.
        v5_mask = self._bt_mask(batch, ref) if self._read_crd_mask_padding(module_id) else None
        # anomaly_gate FIRST, on the RAW |R_f|: quarantine only ANOMALOUS
        # forecast error. Ordinary residual error exists in every regime, and
        # taxing it suppresses learning exactly on the informative wind-ramp
        # transitions. Running it before scale-normalisation keeps the gate
        # sensitive to SUSTAINED corruption (the scale EMA would otherwise
        # absorb a persistent shift and blind the gate).
        if bool(cfg.get("anomaly_gate", False)):
            abs_f = self._forecast_anomaly_excess(
                module_id, abs_f,
                z=float(cfg.get("anomaly_z", 1.0)),
                decay=float(cfg.get("anomaly_decay", 0.99)),
                mask=v5_mask,
            )
        if bool(cfg.get("normalize_shares", False)):
            decay = float(cfg.get("share_scale_decay", 0.99))
            abs_f = self._scale_by_running_ema(module_id, "forecast", abs_f, decay, mask=v5_mask)
            abs_r = self._scale_by_running_ema(module_id, "routing", abs_r, decay, mask=v5_mask)
            abs_s = self._scale_by_running_ema(module_id, "scheduling", abs_s, decay, mask=v5_mask)

        eps = 1e-8
        total = abs_f + abs_r + abs_s + eps

        # Per-transition raw ratios. ρ_forecast left without floor — it's
        # logging-only and the floor would distort the diagnostic signal.
        rho_f = abs_f / total
        rho_r = (abs_r / total).clamp(min=rho_min)
        rho_s = (abs_s / total).clamp(min=rho_min)

        # Detach: ρ is a non-grad multiplier. The gradient signal for the
        # policy comes from PPO's surrogate via the (reweighted) ADVANTAGES
        # tensor, which carries no gradient itself either.
        batch[COL_CRD_RHO_FORECAST] = rho_f.detach()
        batch[COL_CRD_RHO_ROUTING] = rho_r.detach()
        batch[COL_CRD_RHO_SCHEDULING] = rho_s.detach()

        # ----------------- M5.2: actually reweight advantages -----------------
        adv = batch.get(Postprocessing.ADVANTAGES)
        if not isinstance(adv, torch.Tensor):
            return  # No GAE output in this batch (probably a unit-test minibatch).

        # Reweight by the share this module actually controls: ρ_routing for
        # the global router, ρ_scheduling for a local scheduler.
        rho_own = rho_r if own_kind == "routing" else rho_s
        rho_view = rho_own.to(adv.dtype).to(adv.device)
        if adv.shape != rho_view.shape:
            try:
                rho_view = rho_view.reshape(adv.shape)
            except RuntimeError:
                logger.warning(
                    f"[CRD] rho_{own_kind} shape {tuple(rho_view.shape)} != "
                    f"advantages shape {tuple(adv.shape)}; skipping reweight "
                    f"for module {module_id!r}."
                )
                return

        # Ablation gate (crd.responsibility.reweight_advantages, default True): when False,
        # the rho shares are still computed + logged above but NOT multiplied into the
        # advantage — isolates "does the responsibility-shrinking hurt the policy?" from the
        # ensemble-module/integration. Default preserves the standard EU-CRD reweighting.
        if bool(cfg.get("reweight_advantages", True)):
            # crd.responsibility.reweight_warmup_calls (default 0): leave the
            # advantages untouched for the first N loss calls of this module.
            # Root cause treated: at the start of training the Q-ensemble that
            # produces ρ is itself untrained, yet its (garbage) shares already
            # re-scale the policy gradient — amplified defer credit early on
            # tips seeds into the irrecoverable always-defer basin (observed
            # collapse rates: vanilla 1/10 vs v2 3/10). Warmup lets policy and
            # ensemble earn competence under vanilla PPO first; ρ is still
            # computed and logged above, so diagnostics are unaffected.
            warmup = int(cfg.get("reweight_warmup_calls", 0))
            if warmup > 0:
                if not hasattr(self, "_crd_reweight_calls"):
                    self._crd_reweight_calls = {}
                n = self._crd_reweight_calls.get(module_id, 0) + 1
                self._crd_reweight_calls[module_id] = n
                if n <= warmup:
                    return
            w = rho_view
            # crd.responsibility.normalize_rho (default False): mean-preserving
            # reweighting. Plain multiplication by ρ∈[ρ_min,1] shrinks the AVERAGE
            # advantage magnitude by mean(ρ) every batch — an implicit learning-rate
            # cut that leaves the policy under-optimized at equal steps (observed as
            # clean-carbon regret vs vanilla). Dividing by batch-mean(ρ) keeps the
            # total learning signal constant and only REDISTRIBUTES credit across
            # transitions (high-responsibility ones amplified, low ones damped).
            if bool(cfg.get("normalize_rho", False)):
                # v5.1: exclude padded cells from the mean-preserving denominator
                # (pads sit at ρ≈1 or the floor depending on layout — either way
                # they shift the divisor and turn "mean-preserving" into a
                # padding-fraction-dependent effective-LR change).
                denom_src = w
                if v5_mask is not None and bool(v5_mask.any()) and v5_mask.shape == w.shape:
                    denom_src = w[v5_mask.bool()]
                denom = denom_src.mean()
                if torch.isfinite(denom) and float(denom) > 1e-6:
                    w = w / denom
                    # crd.responsibility.normalize_rho_cap (default inf): uncapped
                    # normalization over-amplifies high-ρ (defer-credit) transitions
                    # ~1/mean(ρ)≈2-3x → observed defer saturation (rate 0.99, a
                    # degenerate always-defer policy riding the deadline backstop).
                    # Capping the amplification keeps the mean-preserving intent for
                    # the DOWN-weighting while bounding the up-weighting.
                    cap = float(cfg.get("normalize_rho_cap", float("inf")))
                    if cap < float("inf"):
                        w = w.clamp(max=cap)
            batch[Postprocessing.ADVANTAGES] = adv * w
            # v5.2 diagnostics honesty: a 0/1 signal that the multiply RAN
            # (warmup/bail paths leave it absent → logged 0 via metrics).
            batch["crd_reweight_applied"] = torch.ones(1, device=adv.device)

    def _read_module_responsibility_config(
        self, module_id: ModuleID
    ) -> Dict[str, Any]:
        """Pull `crd.responsibility` from the module's model_config; default {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("responsibility", {}) or {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ v5

    @staticmethod
    def _infer_num_dc_from_obs_space(module: Any) -> Optional[int]:
        """True datacenter count from the module's observation space.

        The global obs is Dict{action_mask, crd_aux, observation:{dc_*...}};
        every dc_* Box has shape (num_dc,). Returns None when the layout is
        not recognisable (caller keeps the nvec-derived count)."""
        try:
            space = getattr(module, "observation_space", None)
            inner = space
            if hasattr(space, "spaces") and "observation" in space.spaces:
                inner = space.spaces["observation"]
            if hasattr(inner, "spaces") and "dc_green_ratio" in inner.spaces:
                return int(inner.spaces["dc_green_ratio"].shape[0])
        except Exception:
            pass
        return None

    def _read_crd_mask_padding(self, module_id: ModuleID) -> bool:
        """Read the v5.1 `crd.mask_padding` gate (default False = v4 bit-identical)."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return bool((mcfg.get("crd", {}) or {}).get("mask_padding", False))
        except Exception:
            return False

    def _read_crd_local_cf_enabled(self, module_id: ModuleID) -> bool:
        """Read the `crd.local_cf_enabled` ablation gate (default True = M4 active)."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return bool((mcfg.get("crd", {}) or {}).get("local_cf_enabled", True))
        except Exception:
            return True

    def _is_local_action_module(self, module_id: ModuleID) -> bool:
        """
        True iff the module's action space is Discrete (a local scheduler).
        The global router is MultiDiscrete. Cached per module; defaults to
        False when the module or its action space can't be inspected, so the
        ablation gate can only ever fire on a positively identified local
        module.
        """
        if not hasattr(self, "_crd_is_local_module"):
            self._crd_is_local_module = {}
        cached = self._crd_is_local_module.get(module_id)
        if cached is not None:
            return cached
        try:
            from gymnasium import spaces  # local import; avoid hard top-level dep

            module = self.module[module_id].unwrapped()
            is_local = isinstance(
                getattr(module, "action_space", None), spaces.Discrete
            )
        except Exception:
            is_local = False
        self._crd_is_local_module[module_id] = is_local
        return is_local

    def _skip_local_cf(self, module_id: ModuleID) -> bool:
        """
        Ablation predicate (crd.local_cf_enabled=false): this module is a
        local scheduler whose counterfactual pipeline is switched off. Used by
        both _compute_crd_terms (skip CFs + reweight) and
        compute_loss_for_module (skip the Q-ensemble TD loss).
        """
        return (not self._read_crd_local_cf_enabled(module_id)) and (
            self._is_local_action_module(module_id)
        )

    @staticmethod
    def _bt_mask(batch: Dict[str, Any], ref: torch.Tensor) -> Optional[torch.Tensor]:
        """(B, T) float loss mask aligned to `ref`, or None when unavailable."""
        lm = batch.get(Columns.LOSS_MASK)
        if isinstance(lm, torch.Tensor) and lm.shape == ref.shape:
            return lm.to(ref.dtype).to(ref.device)
        return None

    def _scale_by_running_ema(
        self,
        module_id: ModuleID,
        key: str,
        x: torch.Tensor,
        decay: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Divide a source-magnitude tensor by the running EMA of its own
        batch-mean, so the three responsibility sources become unit-free
        before the share ratio.

        Padding-aware: `mask` (0/1, same shape) keeps padded cells out of the
        batch statistic. Zero-seed guard: while the EMA is ~0 (a dormant
        source, e.g. R_forecast before predictions start flowing) the tensor
        is returned UNSCALED — dividing by an epsilon would explode the share
        by ~1e8 the moment the source wakes up."""
        vals = x[mask.bool()] if (mask is not None and bool(mask.any())) else x
        m = float(vals.mean().detach())
        d = self._crd_share_scale_ema.setdefault(module_id, {})
        prev = d.get(key)
        ema = m if (prev is None or prev <= 1e-8) else decay * prev + (1.0 - decay) * m
        d[key] = ema
        if ema <= 1e-8:
            return x
        return x / ema

    def _forecast_anomaly_excess(
        self,
        module_id: ModuleID,
        x: torch.Tensor,
        z: float,
        decay: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Excess of |R_forecast| above a running mean + z*std baseline.

        Ordinary forecast error contributes ~0 (no quarantine tax on the
        informative ramp transitions); only anomalous spikes survive and
        claim forecast share. Padding-aware via `mask`. NOTE: this must run
        on the RAW |R_f| (before scale normalisation) — normalising first
        would let the scale EMA absorb a SUSTAINED corruption within ~1/(1-decay)
        calls and blind the gate to exactly the deployment threat it targets."""
        vals = x[mask.bool()] if (mask is not None and bool(mask.any())) else x
        m = float(vals.mean().detach())
        s = float(vals.std().detach())
        d = self._crd_forecast_anom_ema.setdefault(module_id, {})
        em = m if d.get("mean") is None else decay * d["mean"] + (1.0 - decay) * m
        es = s if d.get("std") is None else decay * d["std"] + (1.0 - decay) * s
        d["mean"], d["std"] = em, es
        return torch.clamp(x - (em + z * es), min=0.0)

    def _read_module_risk_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.risk` from the module's model_config; default {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("risk", {}) or {}
        except Exception:
            return {}

    def _apply_risk_objective(self, *, module_id: ModuleID, batch: Dict[str, Any]) -> None:
        """Config-gated risk-averse baseline: transform ADVANTAGES in place.
        No-op unless crd.risk.kind in {cvar, risk_sensitive, mean_variance}."""
        cfg = self._read_module_risk_config(module_id)
        if str(cfg.get("kind", "none")).strip().lower() in ("", "none"):
            return
        adv = batch.get(Postprocessing.ADVANTAGES)
        if not isinstance(adv, torch.Tensor):
            return
        ret = batch.get(Postprocessing.VALUE_TARGETS)  # return distribution (dist_cvar)
        from src.learners.risk_objectives import apply_risk_objective
        batch[Postprocessing.ADVANTAGES] = apply_risk_objective(adv, cfg, ret=ret)

    def _read_module_cca_config(self, module_id: ModuleID) -> Dict[str, Any]:
        """Pull `crd.cca` from the module's model_config; default {}."""
        try:
            module = self.module[module_id].unwrapped()
            mcfg = getattr(module, "model_config", {}) or {}
            return (mcfg.get("crd", {}) or {}).get("cca", {}) or {}
        except Exception:
            return {}

    @staticmethod
    def _extract_actual_green_bt_d(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Return realised per-DC green power as (B, T, D) from crd_aux, or None.
        Mirrors the unwrap logic in `_compute_forecast_cf_from_obs`."""
        obs = batch.get(Columns.OBS)
        if not isinstance(obs, dict):
            return None
        aux = obs.get("crd_aux")
        if not isinstance(aux, dict) and isinstance(obs.get("observation"), dict):
            aux = obs["observation"].get("crd_aux")
        if not isinstance(aux, dict):
            return None
        g = aux.get("crd_actual_green_w")
        if not isinstance(g, torch.Tensor):
            return None
        if g.dim() == 3:
            return g
        if g.dim() == 2:
            return g.unsqueeze(1)  # (N, D) → (N, 1, D)
        return None

    def _apply_cca(self, *, module_id: ModuleID, batch: Dict[str, Any]) -> None:
        """Config-gated CCA-PG baseline: replace ADVANTAGES with R_t - V^h.
        No-op unless crd.cca.enabled. Trains a per-module hindsight net online
        against the return, conditioned on V(s_t) and the future green Phi."""
        cfg = self._read_module_cca_config(module_id)
        if not bool(cfg.get("enabled", False)):
            return
        adv = batch.get(Postprocessing.ADVANTAGES)
        ret = batch.get(Postprocessing.VALUE_TARGETS)
        val = batch.get(Columns.VF_PREDS)
        if not (isinstance(adv, torch.Tensor) and isinstance(ret, torch.Tensor)):
            return
        green = self._extract_actual_green_bt_d(batch)  # (B, T, D)
        if green is None:
            return  # crd_aux absent → leave vanilla advantage untouched.

        from src.learners.cca import (
            HindsightBaseline, future_green_phi, cca_advantage,
        )
        horizon = int(cfg.get("horizon", 12))
        phi_bt_d = future_green_phi(green.detach().float(), horizon)  # (B, T, D)
        B, T, D = phi_bt_d.shape
        N = B * T
        phi = phi_bt_d.reshape(N, D).to(adv.device, dtype=adv.dtype)
        returns = ret.reshape(N).to(adv.device, dtype=adv.dtype)
        if isinstance(val, torch.Tensor) and val.numel() == N:
            value = val.detach().reshape(N).to(adv.device, dtype=adv.dtype)
        else:
            value = torch.zeros(N, device=adv.device, dtype=adv.dtype)

        net = self._cca_nets.get(module_id)
        if net is None:
            net = HindsightBaseline(phi_dim=D, hidden=int(cfg.get("hidden", 64)))
            net.to(adv.device, dtype=adv.dtype)
            self._cca_nets[module_id] = net
            self._cca_opts[module_id] = torch.optim.Adam(
                net.parameters(), lr=float(cfg.get("lr", 1e-3))
            )
        opt = self._cca_opts[module_id]
        new_adv, _loss = cca_advantage(
            returns, value, phi, net, opt,
            train_iters=int(cfg.get("train_iters", 1)),
        )
        batch[Postprocessing.ADVANTAGES] = new_adv.reshape(adv.shape)

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
        infos: Any, beta: float, gamma: float,
        carbon_norm: bool = False, magnitude: bool = False,
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
                        r = forecast_cf_per_step(
                            crd, pred, beta=beta, gamma=gamma,
                            carbon_norm=carbon_norm, magnitude=magnitude,
                        )
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
        target_var: float = 1.0,
        mask_padding: bool = False,
        stable_seed: "int | None" = None,
    ) -> torch.Tensor:
        """
        TD-style MSE loss with per-sample bootstrap masking.

        Args:
            q_ensemble: from `_forward_train`; shape is one of
                - local agent (Discrete):       (B, T, K, A)
                - global agent (MultiDiscrete): (B, T, K, batch_size, num_dc)
            target_var: running variance of VALUE_TARGETS used to normalize the
                squared TD error into σ² units (mirrors the critic vf-loss
                normalization; the Q-target IS VALUE_TARGETS). Default 1.0 =
                raw MSE (un-normalized, pre-2026-06-16 behavior).
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
        if stable_seed is not None:
            # v5.2 (gated ensemble.stable_bootstrap): deterministic within one
            # training iteration — epochs revisit the SAME head-data assignment.
            g = torch.Generator().manual_seed(int(stable_seed))
            mask = (torch.rand(B, K, generator=g) < bootstrap_p).float().to(q_chosen.device)
        else:
            mask = (torch.rand(B, K, device=q_chosen.device) < bootstrap_p).float()
        # Ensure no division by zero: if a sample has zero mask everywhere,
        # force include head 0 for that sample.
        any_per_sample = (mask.sum(dim=1) > 0).float()  # (B,)
        if (any_per_sample < 1.0).any():
            mask[any_per_sample < 1.0, 0] = 1.0
        mask = mask.unsqueeze(1)  # (B, 1, K) → broadcast over T

        # Normalize the squared TD error by the target variance (σ² units),
        # mirroring NormalizedCriticPPOTorchLearner's vf-loss normalization.
        sq_err = (q_chosen - target).pow(2) / target_var  # (B, T, K)
        masked = sq_err * mask                             # (B, T, K)
        # v5.1 (gated by mask_padding at the call site): keep padded timesteps
        # out of the TD loss — RLlib zero-pads VALUE_TARGETS, so without this
        # the heads are regressed toward Q(pad-features)=0 and the valid-cell
        # signal is diluted by the padding fraction.
        if mask_padding:
            lm = batch.get(Columns.LOSS_MASK)
            if isinstance(lm, torch.Tensor) and lm.shape == q_chosen.shape[:2]:
                lm_b = lm.to(sq_err.dtype).unsqueeze(-1)   # (B, T, 1)
                masked = masked * lm_b
                denom = (mask * lm_b).sum()
                return masked.sum() / (denom + 1e-8)
        denom = mask.sum() * sq_err.shape[1]               # active heads × T
        return masked.sum() / (denom + 1e-8)
