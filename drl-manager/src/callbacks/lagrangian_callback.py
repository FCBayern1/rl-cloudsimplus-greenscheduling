"""
Lagrangian SLA-constraint callback for carbon-aware hierarchical RL.

Formulation:
    max  E[Σ r_step]          r_step  = −β·Ĉ_per_mi + (+shaping terms)
    s.t. E[completion_rate] ≥ c*

    L(π, λ) = −E[Σ r_step] + λ · E[c_ep]
    r_train_global = r_step − λ · c_step      (applied in env wrapper)
    λ_{k+1} = max(0, λ_k + η_λ · (c_ep_avg − tol))

Signals (Java side, ``global_energy_stats``):
  - sla_cost_step     : max(0, pending_ratio − d)       — dense, per-step
  - sla_cost_episode  : max(0, c* − completion_rate_mi) — sparse, evaluated per episode
  - completion_rate_mi, sla_pending_ratio               — diagnostics

Data flow:
  - Env applies the λ·c_step penalty every step using ``lagrangian["lambda"]``.
  - Env exposes per-episode aggregates in the *terminal-step info dict*:
        global_energy_stats.sla_cost_episode,
        global_energy_stats.completion_rate_mi,
        global_energy_stats.sla_pending_ratio,
        info["lagrangian_c_step_mean_episode"].
  - This callback reads them in ``on_episode_end`` (the same hook
    ``GreenEnergyLoggerCallback`` uses, which works reliably under the new API
    stack) and accumulates them into ``self._iter_episodes``.
  - In ``on_train_result`` we drain that list, run dual ascent on the mean
    episode-level violation, update λ, and push the new value to every env via
    ``set_lagrangian_lambda`` (both the shared-dict ``env_config["lagrangian"]``
    and remote envs through ``foreach_env``).  CSV logging happens here too.

Why not mutate a shared dict directly from the env?  RLlib's env_config can be
deep-copied or pickled when constructing EnvRunners, so a per-step mutation
inside the env may land in a dict that the driver-side callback never sees.
Going through ``on_episode_end`` + ``foreach_env`` is the stable pattern.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional

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


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_last_info(episode) -> Dict[str, Any]:
    """Pull the terminal info dict from a MultiAgentEpisode / EpisodeV2.

    Mirrors the fallback ladder used in ``rllib_green_energy_logger`` so we
    stay compatible with both API stacks.  Prefers ``global_agent`` but falls
    back to any agent since all agents share the same info reference.
    """
    # New API: MultiAgentEpisode.get_infos() -> {agent_id: [info, info, ...]}
    if hasattr(episode, "get_infos"):
        try:
            all_infos = episode.get_infos()
            if isinstance(all_infos, dict) and all_infos:
                for agent_key in ("global_agent",):
                    lst = all_infos.get(agent_key)
                    if lst:
                        return lst[-1] or {}
                # Any agent
                for _, lst in all_infos.items():
                    if lst:
                        return lst[-1] or {}
        except Exception as e:
            logger.debug("[Lagrangian] get_infos() failed: %s", e)

    # Old API: episode.last_info_for(agent_id)
    if hasattr(episode, "last_info_for"):
        try:
            info = episode.last_info_for("global_agent")
            if info:
                return info
        except Exception:
            pass
        try:
            info = episode.last_info_for()
            if info:
                return info
        except Exception:
            pass

    # Last resort: agent_to_last_info cache
    if hasattr(episode, "agent_to_last_info"):
        try:
            agent_infos = episode.agent_to_last_info
            if agent_infos:
                first = next(iter(agent_infos.values()))
                return first or {}
        except Exception:
            pass

    return {}


def _foreach_env_safely(algorithm, fn) -> None:
    """Run ``fn(env)`` on every env across all API stacks.

    Tries (in order):
      - algorithm.env_runner_group (new API, includes local runner when
        num_env_runners=0 and create_env_on_local_worker=True)
      - algorithm.workers (old API)
      - algorithm.env_runner (new API, local-only fallback)
    """
    for attr in ("env_runner_group", "workers"):
        group = getattr(algorithm, attr, None)
        if group is not None and hasattr(group, "foreach_env"):
            try:
                group.foreach_env(fn)
                return
            except Exception as e:
                logger.debug("[Lagrangian] foreach_env via %s failed: %s", attr, e)
    runner = getattr(algorithm, "env_runner", None)
    if runner is not None and hasattr(runner, "foreach_env"):
        try:
            runner.foreach_env(fn)
        except Exception as e:
            logger.debug("[Lagrangian] foreach_env via env_runner failed: %s", e)


class LagrangianCallback(DefaultCallbacks):
    """Maintain Lagrangian multiplier λ for the SLA completion-rate constraint.

    Configuration is read from ``algorithm.config.env_config["lagrangian"]``:
        enabled        : bool         — master switch
        lambda_init    : float = 0.0  — starting λ
        lambda_lr      : float = 0.05 — step size for dual update
        lambda_max     : float = 20.0 — hard cap to stop λ blowing up
        c_ep_tolerance : float = 0.0  — deadband around 0 for λ updates
    """

    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir
        self._csv_path: Optional[str] = None
        self._csv_inited = False

        # Hyperparameters + λ — resolved once on first on_train_result.
        self._hyperparams_loaded = False
        self._enabled: bool = False
        self._lambda: float = 0.0
        self._lambda_init: float = 0.0
        self._lambda_lr: float = 0.05
        self._lambda_max: float = 20.0
        self._c_ep_tolerance: float = 0.0
        # Reference to algorithm.config.env_config["lagrangian"] so we can also
        # write λ back into the shared dict for num_workers=0 runs (local env
        # may or may not share the same dict — we try both channels).
        self._shared_state: Optional[Dict[str, Any]] = None

        # Per-iteration buffer of finished episodes.
        self._iter_episodes: List[Dict[str, float]] = []

    # ---------- hyperparameter resolution ----------
    def _load_hyperparams(self, algorithm) -> None:
        env_cfg = None
        cfg = getattr(algorithm, "config", None)
        if cfg is not None:
            env_cfg = getattr(cfg, "env_config", None)
            if env_cfg is None and hasattr(cfg, "to_dict"):
                env_cfg = cfg.to_dict().get("env_config", {})
        if not isinstance(env_cfg, dict):
            self._hyperparams_loaded = True
            return

        lag = env_cfg.get("lagrangian")
        if isinstance(lag, dict):
            self._enabled = bool(lag.get("enabled", False))
            self._lambda_init = _safe_float(lag.get("lambda_init"), 0.0)
            self._lambda_lr = _safe_float(lag.get("lambda_lr"), 0.05)
            self._lambda_max = _safe_float(lag.get("lambda_max"), 20.0)
            self._c_ep_tolerance = _safe_float(lag.get("c_ep_tolerance"), 0.0)
            self._lambda = _safe_float(lag.get("lambda", self._lambda_init), self._lambda_init)
            lag["lambda"] = self._lambda  # ensure the key exists
            self._shared_state = lag

        self._hyperparams_loaded = True
        logger.info(
            "[Lagrangian] hyperparams loaded: enabled=%s λ_init=%.4f lr=%.4f max=%.1f tol=%.4f",
            self._enabled, self._lambda_init, self._lambda_lr, self._lambda_max,
            self._c_ep_tolerance,
        )

    # ---------- episode-level data collection ----------
    def on_episode_end(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index: Optional[int] = None,
        env_runner=None,
        metrics_logger=None,
        **kwargs,
    ) -> None:
        if not self._hyperparams_loaded:
            # algorithm handle not available here in the new API — rely on
            # _load_hyperparams running in on_train_result.  Still record stats
            # so the first iteration isn't wasted.
            pass
        if episode is None:
            return

        last_info = _extract_last_info(episode)
        if not last_info:
            logger.debug("[Lagrangian] on_episode_end: no info extracted")
            return

        ges_raw = last_info.get("global_energy_stats", {}) or {}
        # Info may contain a Java Map — convert to dict defensively.
        if not isinstance(ges_raw, dict):
            try:
                from src.callbacks.rllib_green_energy_logger import safe_convert_to_dict
                ges = safe_convert_to_dict(ges_raw, "global_energy_stats")
            except Exception:
                ges = {}
        else:
            ges = ges_raw

        c_ep = _safe_float(ges.get("sla_cost_episode"))
        compl = _safe_float(ges.get("completion_rate_mi"))
        pending = _safe_float(ges.get("sla_pending_ratio"))
        c_step_mean = _safe_float(last_info.get("lagrangian_c_step_mean_episode"))

        self._iter_episodes.append({
            "c_ep": c_ep,
            "c_step_mean": c_step_mean,
            "completion": compl,
            "pending": pending,
        })
        logger.info(
            "[Lagrangian] episode done: c_ep=%.4f c_step_mean=%.4f compl=%.3f pending=%.3f",
            c_ep, c_step_mean, compl, pending,
        )

    # ---------- dual update per training iteration ----------
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if not self._hyperparams_loaded:
            self._load_hyperparams(algorithm)

        if not self._enabled:
            self._iter_episodes.clear()
            return

        iteration = int(result.get("training_iteration", 0))
        env_steps = int(
            result.get("num_env_steps_sampled_lifetime")
            or (result.get("env_runners") or {}).get("num_env_steps_sampled_lifetime")
            or 0
        )

        ep_count = len(self._iter_episodes)
        if ep_count > 0:
            c_ep_mean = sum(e["c_ep"] for e in self._iter_episodes) / ep_count
            c_step_mean = sum(e["c_step_mean"] for e in self._iter_episodes) / ep_count
            compl_mean = sum(e["completion"] for e in self._iter_episodes) / ep_count
            pending_mean = sum(e["pending"] for e in self._iter_episodes) / ep_count
        else:
            c_ep_mean = c_step_mean = compl_mean = pending_mean = 0.0

        lam_prev = float(self._lambda)
        if ep_count > 0:
            violation = c_ep_mean - self._c_ep_tolerance
            lam_new = lam_prev + self._lambda_lr * violation
            if violation <= 0.0:
                # Decay λ when constraint is comfortably satisfied so policy
                # can relax once SLA is met — keeps λ from latching high.
                lam_new = max(0.0, lam_prev * 0.95)
            lam_new = float(max(0.0, min(lam_new, self._lambda_max)))
        else:
            lam_new = lam_prev
        self._lambda = lam_new

        # Propagate λ to every env + the shared env_config dict (belt &
        # suspenders: one of these is guaranteed to reach the env actually
        # running step()).
        if isinstance(self._shared_state, dict):
            self._shared_state["lambda"] = lam_new

        def _push(env):
            inner = env
            if hasattr(inner, "par_env"):
                inner = inner.par_env
            if hasattr(inner, "set_lagrangian_lambda"):
                inner.set_lagrangian_lambda(lam_new)

        _foreach_env_safely(algorithm, _push)

        # Expose in result for TB / CLI.
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

        self._iter_episodes.clear()

    # ---------- CSV helpers ----------
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
