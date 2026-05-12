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

Bridging env_runner ↔ driver:
  Under the new RLlib API stack, callbacks are instantiated separately on the
  env_runner (where ``on_episode_end`` fires) and on the algorithm/driver
  (where ``on_train_result`` fires). They are different Python objects, so
  any state stashed on ``self`` from the env_runner side is invisible to the
  driver side. The only reliable cross-process channel is ``metrics_logger``
  (env_runner side) → ``result["env_runners"][...]`` (driver side).

  - Env applies the λ·c_step penalty every step using ``lagrangian["lambda"]``
    (kept in sync via ``set_lagrangian_lambda`` from the driver).
  - Env exposes per-episode aggregates in the *terminal-step info dict*
    (``global_energy_stats.sla_cost_episode/completion_rate_mi/sla_pending_ratio``
    plus ``lagrangian_c_step_mean_episode``).
  - This callback reads them in ``on_episode_end`` and pushes the four scalars
    to ``metrics_logger`` (mean reduction).
  - In ``on_train_result`` we read the aggregated means out of
    ``result["env_runners"]``, gate on ``num_episodes`` to detect empty iters,
    run dual ascent, push the new λ to every env via ``foreach_env``, and
    append a CSV row.
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
    "completion_rate_mi",   # mean episode completion this iter
    "pending_ratio_mean",   # mean per-step pending ratio
    "num_episodes_in_iter",
]

# Metric keys logged via metrics_logger on the env_runner side.  These keys
# show up under result["env_runners"][...] on the driver side after RLlib
# aggregates them.  Prefix is intentional so they can't clash with anything
# GreenEnergyLoggerCallback already publishes.
_KEY_C_EP = "lagrangian_c_ep"
_KEY_C_STEP = "lagrangian_c_step"
_KEY_COMPLETION = "lagrangian_completion_mi"
_KEY_PENDING = "lagrangian_pending_ratio"


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
    if hasattr(episode, "get_infos"):
        try:
            all_infos = episode.get_infos()
            if isinstance(all_infos, dict) and all_infos:
                lst = all_infos.get("global_agent")
                if lst:
                    return lst[-1] or {}
                for _, lst in all_infos.items():
                    if lst:
                        return lst[-1] or {}
        except Exception as e:
            logger.debug("[Lagrangian] get_infos() failed: %s", e)

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

    New API (``MultiAgentEnvRunner``) has no ``foreach_env`` method — only
    ``EnvRunnerGroup.foreach_env`` exists, and *it* tries to call
    ``w.foreach_env(...)`` on each runner, which silently AttributeErrors
    on the new stack.  So we route via ``foreach_worker`` (always available)
    and extract ``runner.env`` ourselves.  Old API ``RolloutWorker`` does
    expose ``foreach_env``; we keep that branch as a fallback.
    """
    def _runner_to_env_fn(runner):
        env = getattr(runner, "env", None)
        if env is None:
            return None
        # New API may wrap envs in a vector; unwrap if possible.
        sub_envs = getattr(env, "envs", None)
        if isinstance(sub_envs, (list, tuple)) and sub_envs:
            for sub in sub_envs:
                fn(sub)
            return None
        fn(env)
        return None

    # New API path.
    group = getattr(algorithm, "env_runner_group", None)
    if group is not None and hasattr(group, "foreach_worker"):
        try:
            group.foreach_worker(_runner_to_env_fn, local_env_runner=True)
            return
        except Exception as e:
            logger.debug("[Lagrangian] foreach_worker via env_runner_group failed: %s", e)

    # Old API path: RolloutWorker exposes foreach_env directly.
    workers = getattr(algorithm, "workers", None)
    if workers is not None and hasattr(workers, "foreach_env"):
        try:
            workers.foreach_env(fn)
            return
        except Exception as e:
            logger.debug("[Lagrangian] foreach_env via workers failed: %s", e)

    # Last resort: local-only single env_runner attribute.
    runner = getattr(algorithm, "env_runner", None)
    if runner is not None:
        env = getattr(runner, "env", None)
        if env is not None:
            try:
                fn(env)
            except Exception as e:
                logger.debug("[Lagrangian] direct env_runner.env push failed: %s", e)


def _read_env_runner_metric(result: dict, key: str, default: float = 0.0) -> float:
    """Look up an env_runners-scoped scalar from the train result dict."""
    er = result.get("env_runners")
    if isinstance(er, dict) and key in er:
        return _safe_float(er.get(key), default)
    # Some Ray versions also publish under top-level custom_metrics.
    cm = result.get("custom_metrics")
    if isinstance(cm, dict) and key in cm:
        return _safe_float(cm.get(key), default)
    return default


class LagrangianCallback(DefaultCallbacks):
    """Maintain Lagrangian multiplier λ for the SLA completion-rate constraint.

    Configuration is read from ``algorithm.config.env_config["lagrangian"]``:
        enabled        : bool         — master switch
        lambda_init    : float = 0.0  — starting λ
        lambda_lr      : float = 0.05 — step size for dual update
        lambda_max     : float = 20.0 — hard cap
        c_ep_tolerance : float = 0.0  — deadband around 0 for λ updates
    """

    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        self.log_dir = log_dir
        self._csv_path: Optional[str] = None
        self._csv_inited = False

        self._hyperparams_loaded = False
        self._enabled: bool = False
        self._lambda: float = 0.0
        self._lambda_init: float = 0.0
        self._lambda_lr: float = 0.05
        self._lambda_max: float = 20.0
        self._c_ep_tolerance: float = 0.0

        # 2026-05-12 Fix 4: per-iter c_ep tracking.  RLlib's
        # `env_runners/c_ep_mean` is a windowed mean over the last
        # `metrics_num_episodes_for_smoothing` episodes (default 100), which
        # is effectively a lifetime average for our training scale.  That
        # makes λ ramp linearly even when recent episodes have improved —
        # see 76-iter run 20260509_011407 (λ: 0.017 → 1.247 while completion
        # had already plateaued).  We now also record c_ep values that came
        # in during the *current* iter only, and feed THAT mean to the dual
        # update.  Falls back to the legacy windowed mean if no episode
        # finished in the iter.
        self._iter_c_ep_values: List[float] = []
        self._iter_completion_values: List[float] = []
        self._iter_c_step_values: List[float] = []
        self._iter_pending_values: List[float] = []

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

        self._hyperparams_loaded = True
        logger.info(
            "[Lagrangian] hyperparams loaded: enabled=%s λ_init=%.4f lr=%.4f max=%.1f tol=%.4f",
            self._enabled, self._lambda_init, self._lambda_lr, self._lambda_max,
            self._c_ep_tolerance,
        )

    # ---------- episode-level data collection (env_runner side) ----------
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
        if episode is None:
            return

        last_info = _extract_last_info(episode)
        if not last_info:
            logger.debug("[Lagrangian] on_episode_end: no info extracted")
            return

        ges_raw = last_info.get("global_energy_stats", {}) or {}
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

        # Bridge env_runner → driver: only metrics_logger is guaranteed to
        # cross the runner/algorithm boundary in the new API stack.
        if metrics_logger is not None:
            try:
                metrics_logger.log_value(_KEY_C_EP, c_ep, reduce="mean")
                metrics_logger.log_value(_KEY_C_STEP, c_step_mean, reduce="mean")
                metrics_logger.log_value(_KEY_COMPLETION, compl, reduce="mean")
                metrics_logger.log_value(_KEY_PENDING, pending, reduce="mean")
            except Exception as e:
                logger.debug("[Lagrangian] metrics_logger.log_value failed: %s", e)

        # 2026-05-12 Fix 4: also record this episode's values into the
        # callback's own per-iter buffer.  These are consumed (and cleared)
        # in on_train_result so the dual update sees ONLY episodes that
        # finished in the current iter — not a windowed lifetime average.
        # In the single-runner setup this code runs in the driver process;
        # for multi-runner deployments these buffers would need to be
        # bridged via metrics_logger too, but RLlib's reduce="mean" is
        # window-averaged so the per-iter requirement still wins here.
        self._iter_c_ep_values.append(c_ep)
        self._iter_completion_values.append(compl)
        self._iter_c_step_values.append(c_step_mean)
        self._iter_pending_values.append(pending)

        # Old API fallback: episode.custom_metrics is read by RLlib's old metrics
        # path and surfaces under result["custom_metrics"][f"{key}_mean"].
        if hasattr(episode, "custom_metrics"):
            try:
                episode.custom_metrics[_KEY_C_EP] = c_ep
                episode.custom_metrics[_KEY_C_STEP] = c_step_mean
                episode.custom_metrics[_KEY_COMPLETION] = compl
                episode.custom_metrics[_KEY_PENDING] = pending
            except Exception:
                pass

        logger.info(
            "[Lagrangian] episode done: c_ep=%.4f c_step_mean=%.4f compl=%.3f pending=%.3f",
            c_ep, c_step_mean, compl, pending,
        )

    # ---------- dual update per training iteration (driver side) ----------
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if not self._hyperparams_loaded:
            self._load_hyperparams(algorithm)

        if not self._enabled:
            return

        iteration = int(result.get("training_iteration", 0))
        env_steps = int(
            result.get("num_env_steps_sampled_lifetime")
            or (result.get("env_runners") or {}).get("num_env_steps_sampled_lifetime")
            or 0
        )

        # 2026-05-12 Fix 4: prefer the per-iter mean over the windowed
        # `env_runners/c_ep_mean` (which is a 100-episode rolling average ≈
        # lifetime mean at our training scale).  Without this, λ keeps
        # ramping even after recent episodes have improved.  We still fall
        # back to the windowed mean when no episode finished this iter.
        if self._iter_c_ep_values:
            c_ep_mean = sum(self._iter_c_ep_values) / len(self._iter_c_ep_values)
            c_step_mean = (sum(self._iter_c_step_values) / len(self._iter_c_step_values)
                           if self._iter_c_step_values else 0.0)
            compl_mean = (sum(self._iter_completion_values) / len(self._iter_completion_values)
                          if self._iter_completion_values else 0.0)
            pending_mean = (sum(self._iter_pending_values) / len(self._iter_pending_values)
                            if self._iter_pending_values else 0.0)
            n_iter_episodes = len(self._iter_c_ep_values)
            logger.info(
                "[Lagrangian] per-iter (Fix 4): n=%d c_ep=%.4f c_step=%.4f compl=%.4f",
                n_iter_episodes, c_ep_mean, c_step_mean, compl_mean,
            )
            # Clear buffers so next iter starts fresh.
            self._iter_c_ep_values.clear()
            self._iter_completion_values.clear()
            self._iter_c_step_values.clear()
            self._iter_pending_values.clear()
        else:
            # No episode finished in this iter — fall back to windowed mean.
            c_ep_mean = _read_env_runner_metric(result, _KEY_C_EP)
            c_step_mean = _read_env_runner_metric(result, _KEY_C_STEP)
            compl_mean = _read_env_runner_metric(result, _KEY_COMPLETION)
            pending_mean = _read_env_runner_metric(result, _KEY_PENDING)

        # Episode count this iteration — canonical RLlib metric.
        er = result.get("env_runners") or {}
        ep_count = int(er.get("num_episodes") or 0)

        lam_prev = float(self._lambda)
        if ep_count > 0:
            violation = c_ep_mean - self._c_ep_tolerance
            lam_new = lam_prev + self._lambda_lr * violation
            if violation <= 0.0:
                # Decay λ when constraint comfortably satisfied so the policy
                # can relax once SLA is met — keeps λ from latching high.
                lam_new = max(0.0, lam_prev * 0.95)
            lam_new = float(max(0.0, min(lam_new, self._lambda_max)))
        else:
            lam_new = lam_prev
        self._lambda = lam_new

        push_results: List[str] = []

        def _push(env):
            # The env handed in by RLlib is wrapped:
            #   gym.OrderEnforcing → RLModulePettingZooEnv → HierarchicalMultiDCParallelEnv
            # Walk every common unwrap attribute (.env, .par_env, .unwrapped,
            # .base_env) until we find set_lagrangian_lambda or run out.
            chain: List[str] = []
            current = env
            chain.append(type(current).__name__)
            target = None
            visited = {id(current)}
            for _ in range(8):  # bounded walk to avoid pathological loops
                if hasattr(current, "set_lagrangian_lambda"):
                    target = current
                    break
                # Try common wrapper attributes in priority order.
                next_env = None
                for attr in ("par_env", "env", "unwrapped", "base_env"):
                    cand = getattr(current, attr, None)
                    if cand is not None and id(cand) not in visited and cand is not current:
                        next_env = cand
                        break
                if next_env is None:
                    break
                current = next_env
                visited.add(id(current))
                chain.append(type(current).__name__)
            if target is not None:
                target.set_lagrangian_lambda(lam_new)
                cfg = getattr(target, "_lagrangian_cfg", None)
                read_back = cfg.get("lambda") if isinstance(cfg, dict) else "NO_CFG"
                push_results.append(
                    f"OK chain={'→'.join(chain)} set λ={lam_new:.4f} "
                    f"read_back={read_back}"
                )
            else:
                push_results.append(
                    f"NO_METHOD chain={'→'.join(chain)} "
                    "(set_lagrangian_lambda not found in unwrap path)"
                )

        _foreach_env_safely(algorithm, _push)
        if push_results:
            for r in push_results:
                logger.info("[Lagrangian push] %s", r)
        else:
            logger.warning("[Lagrangian push] _push was never called — foreach path failed silently")

        # Surface in custom_metrics so TB / progress.csv show λ trajectory.
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
