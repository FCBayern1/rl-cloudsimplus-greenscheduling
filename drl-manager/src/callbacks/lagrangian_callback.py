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
  - λ lives inside the shared ``env_config["lagrangian"]`` dict.
  - With num_workers == 0 the env and callback share the same dict reference,
    so writing ``self.lagrangian_state["lambda"] = new_lam`` is immediately
    visible to the env's step().
  - For num_workers > 0, ``algorithm.workers.foreach_env(lambda e: ...)`` is
    used to push λ into each worker's env copies.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

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


def _get_from_info(info: Dict[str, Any], path: str, default: float = 0.0) -> float:
    """Dot-path lookup on nested info dict, returning float default on miss."""
    node: Any = info
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    try:
        return float(node)
    except (TypeError, ValueError):
        return default


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

        # Per-iteration accumulators
        self._ep_c_ep_sum = 0.0
        self._ep_c_step_sum = 0.0
        self._ep_completion_sum = 0.0
        self._ep_pending_sum = 0.0
        self._ep_count = 0

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

    # ---------- episode accounting ----------
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker = None,
        base_env: BaseEnv = None,
        policies: Dict[str, Policy] = None,
        episode: EpisodeV2 = None,
        env_index: Optional[int] = None,
        env_runner=None,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        """Accumulate per-episode SLA violation and mean per-step cost.

        Uses only the last step's info (episode-to-date cumulatives), which is
        sufficient for ``completion_rate_mi`` and ``pending_ratio`` since the
        Java side exposes them as running episode values.
        """
        last_info: Dict[str, Any] = {}

        # New API path
        if hasattr(episode, "get_infos"):
            try:
                all_infos = episode.get_infos()
                if isinstance(all_infos, dict):
                    for key in ("global_agent", *all_infos.keys()):
                        lst = all_infos.get(key)
                        if lst:
                            last_info = lst[-1] or {}
                            break
            except Exception:
                pass

        # Old API fallback
        if not last_info and hasattr(episode, "last_info_for"):
            try:
                last_info = episode.last_info_for("global_agent") or {}
            except Exception:
                last_info = {}

        if not isinstance(last_info, dict):
            return

        c_ep = _get_from_info(last_info, "global_energy_stats.sla_cost_episode", 0.0)
        c_step = _get_from_info(last_info, "global_energy_stats.sla_cost_step", 0.0)
        compl = _get_from_info(last_info, "global_energy_stats.completion_rate_mi", 0.0)
        pending = _get_from_info(last_info, "global_energy_stats.sla_pending_ratio", 0.0)

        self._ep_c_ep_sum += c_ep
        self._ep_c_step_sum += c_step
        self._ep_completion_sum += compl
        self._ep_pending_sum += pending
        self._ep_count += 1

    # ---------- dual update per training iteration ----------
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        state = self._resolve_state(algorithm)
        if state is None or not self._enabled:
            # No Lagrangian configured — reset accumulators and leave.
            self._reset_accum()
            return

        iteration = int(result.get("training_iteration", 0))
        env_steps = int(
            result.get("num_env_steps_sampled_lifetime")
            or (result.get("env_runners") or {}).get("num_env_steps_sampled_lifetime")
            or 0
        )

        n = max(1, self._ep_count)
        c_ep_mean = self._ep_c_ep_sum / n
        c_step_mean = self._ep_c_step_sum / n
        compl_mean = self._ep_completion_sum / n
        pending_mean = self._ep_pending_sum / n

        lam_prev = float(state.get("lambda", 0.0))
        lr = self._hyperparams["lambda_lr"]
        lam_max = self._hyperparams["lambda_max"]
        tol = self._hyperparams["c_ep_tolerance"]

        # Dual ascent on the episode-level violation.  c_ep is already
        # max(0, c*-completion), so it's always ≥ 0 — when 0, we decay λ so
        # policy can relax once SLA is comfortably met.
        violation = c_ep_mean - tol
        lam_new = lam_prev + lr * violation
        if violation <= 0.0:
            # decay toward 0 (but not below)
            lam_new = max(0.0, lam_prev * 0.95)
        lam_new = float(max(0.0, min(lam_new, lam_max)))
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
            compl_mean, pending_mean, self._ep_count,
        )

        # Persist to CSV for post-hoc analysis.
        self._append_csv_row([
            iteration, env_steps, lam_new, lam_prev,
            c_ep_mean, c_step_mean, compl_mean, pending_mean, self._ep_count,
        ])

        self._reset_accum()

    # ---------- helpers ----------
    def _reset_accum(self) -> None:
        self._ep_c_ep_sum = 0.0
        self._ep_c_step_sum = 0.0
        self._ep_completion_sum = 0.0
        self._ep_pending_sum = 0.0
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
