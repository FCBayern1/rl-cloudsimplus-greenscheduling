"""
Lagrangian SLA-constraint callback for carbon-aware hierarchical RL.

Formulation:
    max  E[Σ r_step]          r_step  = −β·Ĉ_per_mi + (+shaping terms)
    s.t. E[completion_rate] ≥ c*

    L(π, λ) = −E[Σ r_step] + λ · E[c_ep]
    r_train_global = r_step − λ · c_step      (applied in env wrapper)
    λ_{k+1} = max(0, λ_k + η_λ · (c_ep_avg − 0))

Signals (Java side, `global_energy_stats`):
  - sla_cost_step     : max(0, pending_ratio − d)       — dense, per-step
  - sla_cost_episode  : max(0, c* − completion_rate_mi) — sparse, evaluated per episode

Sync model:
  - λ and the per-episode accumulator both live inside the shared
    ``env_config["lagrangian"]`` dict, which the env holds by reference.
  - **Accumulation is driven by the env**, not by RLlib's ``on_episode_end``
    hook: under the new API stack (``enable_env_runner_and_connector_v2``),
    ``on_episode_end`` does not fire reliably on every registered callback,
    so the env appends (c_ep, c_step_mean, completion, pending) into
    ``lagrangian["_accum"]`` whenever a terminated/truncated step is seen.
  - The callback drains ``_accum`` in ``on_train_result``, runs dual ascent,
    then resets the accumulator.  This is robust across RLlib versions.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


LAGRANGIAN_CSV_HEADERS = [
    "iteration",
    "env_steps_lifetime",
    "lambda",
    "lambda_prev",
    "c_ep_mean",            # episode-level completion violation
    "c_step_mean",          # per-step pending violation (avg over iter)
    "completion_rate_mi",   # last observed episode completion
    "pending_ratio_mean",   # last observed per-step pending ratio
    "num_episodes_in_iter",
]


class LagrangianCallback(DefaultCallbacks):
    """Maintain Lagrangian multiplier λ for the SLA completion-rate constraint.

    Configuration is read from ``algorithm.config.env_config["lagrangian"]``:
        enabled      : bool         — master switch
        lambda_init  : float = 0.0  — starting λ
        lambda_lr    : float = 0.05 — step size for dual update
        lambda_max   : float = 20.0 — hard cap to stop λ blowing up
        c_ep_tolerance : float = 0.0 — deadband around 0 for λ updates
    """

    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir
        self._csv_path: Optional[str] = None
        self._csv_inited = False

        # Resolved once we see the first algorithm reference
        self._lagrangian_state: Optional[Dict[str, Any]] = None
        self._hyperparams: Dict[str, float] = {
            "lambda_lr": 0.05,
            "lambda_max": 20.0,
            "c_ep_tolerance": 0.0,
        }
        self._enabled: Optional[bool] = None

    # ---------- state resolution ----------
    def _resolve_state(self, algorithm) -> Optional[Dict[str, Any]]:
        """Locate the shared lagrangian dict on the algorithm's env_config.

        RLlib stores env_config under ``algorithm.config.env_config`` (a
        FrozenDict in newer releases, plain dict otherwise).  The nested
        "lagrangian" entry is a mutable dict regardless — we mutate in place.
        """
        if self._lagrangian_state is not None:
            return self._lagrangian_state

        env_cfg = None
        cfg = getattr(algorithm, "config", None)
        if cfg is not None:
            env_cfg = getattr(cfg, "env_config", None)
            if env_cfg is None and hasattr(cfg, "to_dict"):
                env_cfg = cfg.to_dict().get("env_config", {})
        if not isinstance(env_cfg, dict):
            return None

        lag = env_cfg.get("lagrangian")
        if not isinstance(lag, dict):
            return None

        # Ensure lambda key exists — otherwise envs can't read it.
        lag.setdefault("lambda", float(lag.get("lambda_init", 0.0)))
        self._lagrangian_state = lag
        self._enabled = bool(lag.get("enabled", False))
        self._hyperparams["lambda_lr"] = float(lag.get("lambda_lr", 0.05))
        self._hyperparams["lambda_max"] = float(lag.get("lambda_max", 20.0))
        self._hyperparams["c_ep_tolerance"] = float(lag.get("c_ep_tolerance", 0.0))
        logger.info(
            "[Lagrangian] resolved state: enabled=%s, lambda_init=%.4f, lr=%.4f, max=%.1f",
            self._enabled, lag["lambda"],
            self._hyperparams["lambda_lr"], self._hyperparams["lambda_max"],
        )
        return lag

    # ---------- dual update per training iteration ----------
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        state = self._resolve_state(algorithm)
        if state is None or not self._enabled:
            self._reset_accum(state)
            return

        iteration = int(result.get("training_iteration", 0))
        env_steps = int(
            result.get("num_env_steps_sampled_lifetime")
            or (result.get("env_runners") or {}).get("num_env_steps_sampled_lifetime")
            or 0
        )

        # Drain shared per-episode accumulator populated by the env.
        accum = state.setdefault("_accum", {
            "c_ep_sum": 0.0, "c_step_sum": 0.0,
            "completion_sum": 0.0, "pending_sum": 0.0, "ep_count": 0,
        })
        ep_count = int(accum.get("ep_count", 0))
        n = max(1, ep_count)
        c_ep_mean = float(accum.get("c_ep_sum", 0.0)) / n
        c_step_mean = float(accum.get("c_step_sum", 0.0)) / n
        compl_mean = float(accum.get("completion_sum", 0.0)) / n
        pending_mean = float(accum.get("pending_sum", 0.0)) / n

        lam_prev = float(state.get("lambda", 0.0))
        lr = self._hyperparams["lambda_lr"]
        lam_max = self._hyperparams["lambda_max"]
        tol = self._hyperparams["c_ep_tolerance"]

        # Dual ascent on the episode-level violation.  c_ep is already
        # max(0, c*-completion), so it's always ≥ 0 — when 0, we decay λ so
        # policy can relax once SLA is comfortably met.
        # Skip update when no episodes finished in this iteration (keeps λ stable).
        if ep_count > 0:
            violation = c_ep_mean - tol
            lam_new = lam_prev + lr * violation
            if violation <= 0.0:
                lam_new = max(0.0, lam_prev * 0.95)
            lam_new = float(max(0.0, min(lam_new, lam_max)))
        else:
            lam_new = lam_prev
        state["lambda"] = lam_new

        # Propagate to all envs (no-op for shared-dict workers but safe).
        try:
            def _push(env):
                inner = env
                if hasattr(inner, "par_env"):
                    inner = inner.par_env
                if hasattr(inner, "set_lagrangian_lambda"):
                    inner.set_lagrangian_lambda(lam_new)
            if hasattr(algorithm, "workers") and algorithm.workers is not None:
                algorithm.workers.foreach_env(_push)
            elif hasattr(algorithm, "env_runner_group") and algorithm.env_runner_group is not None:
                algorithm.env_runner_group.foreach_env(_push)
        except Exception as e:
            logger.debug("[Lagrangian] foreach_env push failed: %s", e)

        # Expose in result dict for TB/CLI.
        result.setdefault("custom_metrics", {})
        result["custom_metrics"]["lagrangian_lambda"] = lam_new
        result["custom_metrics"]["lagrangian_c_ep_mean"] = c_ep_mean
        result["custom_metrics"]["lagrangian_c_step_mean"] = c_step_mean
        result["custom_metrics"]["lagrangian_completion_mean"] = compl_mean

        logger.info(
            "[Lagrangian iter %d] λ: %.4f → %.4f  (c_ep=%.4f, c_step=%.4f, "
            "compl=%.3f, pending=%.3f, eps=%d)",
            iteration, lam_prev, lam_new, c_ep_mean, c_step_mean,
            compl_mean, pending_mean, ep_count,
        )

        self._append_csv_row([
            iteration, env_steps, lam_new, lam_prev,
            c_ep_mean, c_step_mean, compl_mean, pending_mean, ep_count,
        ])

        self._reset_accum(state)

    # ---------- helpers ----------
    def _reset_accum(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return
        accum = state.get("_accum")
        if isinstance(accum, dict):
            accum["c_ep_sum"] = 0.0
            accum["c_step_sum"] = 0.0
            accum["completion_sum"] = 0.0
            accum["pending_sum"] = 0.0
            accum["ep_count"] = 0
        self._ep_count = 0

    def _init_csv(self) -> None:
        if self._csv_inited:
            return
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass
        self._csv_path = os.path.join(self.log_dir, "lagrangian.csv")
        try:
            if not os.path.exists(self._csv_path):
                with open(self._csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(LAGRANGIAN_CSV_HEADERS)
        except Exception as e:
            logger.error("[Lagrangian] CSV init failed: %s", e)
            self._csv_path = None
        self._csv_inited = True

    def _append_csv_row(self, row) -> None:
        self._init_csv()
        if not self._csv_path:
            return
        try:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.error("[Lagrangian] CSV append failed: %s", e)
