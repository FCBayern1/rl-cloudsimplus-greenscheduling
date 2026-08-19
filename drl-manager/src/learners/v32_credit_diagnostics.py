"""Read-only V3.2 temporal-credit diagnostics for the PPO learner.

The global policy has one scalar reward/value/advantage per environment step,
but several real routing slots in that step.  These helpers therefore report
*action-occurrence weighted* conditional statistics: a transition containing
two DEFER and one ROUTE decisions contributes its TD residual twice to the
DEFER bucket and once to the ROUTE bucket.  This is the most granular honest
attribution available without changing the reward or PPO loss.

All returned values are detached Python numbers.  The helper never mutates the
batch and is gated by the V3.2 observation keys, so legacy experiments are a
strict no-op.
"""

from __future__ import annotations

from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.framework import try_import_torch


torch, _ = try_import_torch()

WAIT_EDGES_SEC = (0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, float("inf"))
WAIT_LABELS = ("0-60", "60-300", "300-900", "900-1800", "1800-3600", ">3600")


def _as_float(value) -> float:
    return float(value.detach().float().item())


def _weighted_mean(values, weights) -> tuple[float, int]:
    weights = weights.detach().float()
    count = int(weights.sum().item())
    if count <= 0:
        return float("nan"), 0
    values = values.detach().float()
    return _as_float((values * weights).sum() / weights.sum()), count


def _leading_tensor(value, leading_shape):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    while tensor.ndim > len(leading_shape) and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    return tensor.reshape(leading_shape)


def compute_v32_credit_diagnostics(
    batch: Dict[str, Any],
    value_fn_out,
    *,
    gamma: float,
    num_slots: int,
    num_choices: int,
    wait_age_scale_sec: float,
) -> Dict[str, float]:
    """Return conditionally bucketed TD residual and advantage metrics.

    The one-step residual is ``r_t + gamma*V(s_{t+1}) - V(s_t)``.  Artificial
    sequence boundaries are excluded unless the current transition truly
    terminates; this avoids treating zero padding as a bootstrap value.
    """
    obs = batch.get(Columns.OBS)
    inner = obs.get("observation", obs) if isinstance(obs, dict) else None
    required = {
        "batch_cloudlet_mi",
        "batch_cloudlet_wait_age",
        "batch_cloudlet_forecast_gain",
    }
    if not isinstance(inner, dict) or not required.issubset(inner):
        return {}
    if num_slots <= 0 or num_choices < 2:
        return {}

    mi = inner["batch_cloudlet_mi"]
    mi = mi if torch.is_tensor(mi) else torch.as_tensor(mi)
    if mi.shape[-1] != num_slots:
        return {}
    leading_shape = tuple(mi.shape[:-1])
    actions = batch.get(Columns.ACTIONS)
    advantages = batch.get(Postprocessing.ADVANTAGES)
    rewards = batch.get(Columns.REWARDS)
    terminateds = batch.get(Columns.TERMINATEDS)
    truncateds = batch.get(Columns.TRUNCATEDS)
    if any(x is None for x in (actions, advantages, rewards, terminateds, truncateds)):
        return {}

    actions = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
    try:
        actions = actions.reshape(*leading_shape, num_slots)
        adv = _leading_tensor(advantages, leading_shape).detach().float()
        rew = _leading_tensor(rewards, leading_shape).detach().float()
        terminated = _leading_tensor(terminateds, leading_shape).detach().bool()
        truncated = _leading_tensor(truncateds, leading_shape).detach().bool()
        values = _leading_tensor(value_fn_out, leading_shape).detach().float()
    except (RuntimeError, ValueError):
        return {}

    real = (mi > 0).detach()
    defer = real & (actions == num_choices - 1)
    route = real & (actions >= 0) & (actions < num_choices - 1)
    defer_per_step = defer.sum(dim=-1).float()
    route_per_step = route.sum(dim=-1).float()

    loss_mask = batch.get(Columns.LOSS_MASK)
    if loss_mask is None:
        transition_mask = torch.ones(leading_shape, dtype=torch.bool, device=values.device)
    else:
        transition_mask = _leading_tensor(loss_mask, leading_shape).detach().bool()

    out: Dict[str, float] = {}

    # A true one-step TD residual needs a temporal dimension.  GTrXL batches
    # are (B,T); the defensive ndim check leaves other module layouts untouched.
    if values.ndim >= 2:
        next_values = torch.zeros_like(values)
        next_values[..., :-1] = values[..., 1:]
        next_is_real = torch.zeros_like(transition_mask)
        next_is_real[..., :-1] = transition_mask[..., 1:]
        td_valid = transition_mask & ~truncated & (terminated | next_is_real)
        td = rew + float(gamma) * (~terminated).float() * next_values - values
        td_abs = td.abs()
        td_defer, n_td_defer = _weighted_mean(
            td_abs, defer_per_step * td_valid.float())
        td_route, n_td_route = _weighted_mean(
            td_abs, route_per_step * td_valid.float())
        out.update({
            "v32_td_abs_defer": td_defer,
            "v32_td_abs_route": td_route,
            "v32_td_defer_count": float(n_td_defer),
            "v32_td_route_count": float(n_td_route),
        })

    adv_defer, n_adv_defer = _weighted_mean(
        adv, defer_per_step * transition_mask.float())
    adv_route, n_adv_route = _weighted_mean(
        adv, route_per_step * transition_mask.float())
    out.update({
        "v32_adv_defer": adv_defer,
        "v32_adv_route": adv_route,
        "v32_adv_defer_count": float(n_adv_defer),
        "v32_adv_route_count": float(n_adv_route),
    })

    wait = inner["batch_cloudlet_wait_age"]
    wait = wait if torch.is_tensor(wait) else torch.as_tensor(wait)
    wait_sec = wait.detach().float() * max(1.0, float(wait_age_scale_sec))
    adv_slots = adv.unsqueeze(-1).expand_as(wait_sec)
    valid_slots = real & transition_mask.unsqueeze(-1)
    for lo, hi, label in zip(WAIT_EDGES_SEC[:-1], WAIT_EDGES_SEC[1:], WAIT_LABELS):
        in_bin = valid_slots & (wait_sec >= lo) & (wait_sec < hi)
        d_mean, d_count = _weighted_mean(adv_slots, (in_bin & defer).float())
        r_mean, r_count = _weighted_mean(adv_slots, (in_bin & route).float())
        safe = label.replace(">", "gt").replace("-", "_")
        out[f"v32_adv_defer_wait_{safe}"] = d_mean
        out[f"v32_adv_defer_wait_{safe}_count"] = float(d_count)
        out[f"v32_adv_route_wait_{safe}"] = r_mean
        out[f"v32_adv_route_wait_{safe}_count"] = float(r_count)
    return out
