"""Frozen forecast-perturbation ladder for the Scheme-2 Stage A' gate.

The question the ladder answers: how does the value of a 144-row forecast decay as its
quality decays? Retraining a predictor gives one point on that curve; this module gives
the whole curve without training anything, by feeding the frozen planner an oracle view
corrupted in registered, deterministic ways.

Error model (tiers s05..s60 and timecap_cal):

    view[tau] = max(0, G[tau] + lead_scale(tau - t) * sigma_rel * scale_ref * eps[tau])

  - eps is ONE frozen AR(1) field per (site, tier, episode): the error pattern persists
    across planning steps, the way a real forecast's mistakes do. Resampling per step
    would let the planner average the noise away and overstate robustness.
  - lead_scale grows with lead time, so the near future is better known than the far
    future, and the error at a fixed row shrinks as the row approaches. Real forecasts
    behave this way; a lead-flat error would be unfair to the far rows and generous to
    the near ones at once.
  - scale_ref is the mean of the site's true trace over the episode, so sigma_rel is a
    dimensionless quality knob comparable across sites and divisors.

Negative controls:

    shuffle   one frozen permutation of the whole episode, applied to rows >= t:
              marginals kept, timing destroyed
    anti      the episode reversed in time, applied to rows >= t:
              marginals kept, phase inverted

Everything is a pure function of (series bytes, site, tier), so two processes, two
machines or two reruns see byte-identical corruption. No tier reads the scheduler's
carbon, and the settlement always uses the TRUE trace; only the planner's eyes change.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

# The registered ladder. timecap_cal takes its numbers from a calibration artifact
# measured on the existing TimeCAP checkpoint's validation residuals, so one rung of the
# ladder stands at the quality a real predictor is known to reach.
TIERS = {
    "godeye": {"kind": "noise", "sigma_rel": 0.0},
    "s05": {"kind": "noise", "sigma_rel": 0.05},
    "s15": {"kind": "noise", "sigma_rel": 0.15},
    "s30": {"kind": "noise", "sigma_rel": 0.30},
    "s60": {"kind": "noise", "sigma_rel": 0.60},
    "timecap_cal": {"kind": "noise", "sigma_rel": None},   # from the calibration artifact
    "shuffle": {"kind": "shuffle"},
    "anti": {"kind": "anti"},
}
AR1_RHO = 0.8
LEAD_ALPHA = 0.25          # error at lead 0 is this fraction of the full-lead error


def domain_seed(payload: str, domain: str) -> int:
    digest = hashlib.sha256(f"{payload}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def series_key(series: np.ndarray, site: int, tier: str) -> str:
    h = hashlib.sha256(np.ascontiguousarray(series, dtype=np.float64).tobytes())
    return f"{h.hexdigest()}:{site}:{tier}"


def ar1_field(key: str, n: int, rho: float = AR1_RHO) -> np.ndarray:
    """Unit-variance AR(1) noise, a pure function of the key."""
    rng = np.random.default_rng(domain_seed(key, "eps"))
    white = rng.standard_normal(n)
    eps = np.empty(n)
    eps[0] = white[0]
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + scale * white[i]
    return eps


def lead_scale(leads: np.ndarray, horizon: int, alpha: float = LEAD_ALPHA) -> np.ndarray:
    return alpha + (1.0 - alpha) * np.minimum(leads, horizon) / float(max(horizon, 1))


def perturbed_future(series: np.ndarray, t: int, horizon: int, site: int, tier: str,
                     calibration: dict | None = None) -> np.ndarray:
    """The corrupted view of rows [t, t+horizon), given the TRUE series.

    Rows before t and at or beyond t+horizon are the caller's business (measured past,
    shared tail); this function never returns them.
    """
    spec = TIERS[tier]
    lo, hi = t, min(len(series), t + horizon)
    truth = np.asarray(series[lo:hi], dtype=np.float64)
    if spec["kind"] == "shuffle":
        rng = np.random.default_rng(domain_seed(series_key(series, site, tier), "perm"))
        perm = rng.permutation(len(series))
        return np.maximum(0.0, np.asarray(series, dtype=np.float64)[perm][lo:hi])
    if spec["kind"] == "anti":
        return np.maximum(0.0, np.asarray(series, dtype=np.float64)[::-1][lo:hi])

    sigma = spec["sigma_rel"]
    rho, alpha = AR1_RHO, LEAD_ALPHA
    if tier == "timecap_cal":
        if not calibration:
            raise ValueError("tier timecap_cal needs the calibration artifact")
        sigma = float(calibration["sigma_rel"])
        rho = float(calibration.get("ar1_rho", AR1_RHO))
        alpha = float(calibration.get("lead_alpha", LEAD_ALPHA))
    if sigma == 0.0:
        return truth.copy()
    eps = ar1_field(series_key(series, site, tier), len(series), rho)[lo:hi]
    scale_ref = float(np.mean(np.abs(series))) or 1.0
    leads = np.arange(len(truth), dtype=np.float64)
    return np.maximum(0.0, truth + lead_scale(leads, horizon, alpha) * sigma
                      * scale_ref * eps)


TIERS_V2 = {
    "godeye": {"kind": "noise", "sigma_rel": 0.0},
    "s05": {"kind": "noise", "sigma_rel": 0.05},
    "s15": {"kind": "noise", "sigma_rel": 0.15},
    "s30": {"kind": "noise", "sigma_rel": 0.30},
    "s60": {"kind": "noise", "sigma_rel": 0.60},
    "checkpoint_residual_surrogate_v2": {"kind": "surrogate"},
    "shuffle": {"kind": "shuffle"},
    "anti": {"kind": "anti"},
}


def perturbed_future_v2(series, t, horizon, site, tier, calibration=None,
                        common_key=None):
    """The v2 view of rows [t, t+horizon): lead 0 is the measured present, always.

    Codex 2026-09-02 (ladder-v2, R3): the current row is an observation, not a forecast,
    so no tier may touch it — the old ladder scaled lead-0 error by alpha and let shuffle
    and anti rewrite the present, which mixed "the future is mispredicted" with "the
    sensor is broken". Only leads 1..horizon-1 are corrupted, and because the view is
    rebuilt at every planning step, each newly arrived row reverts to truth on its own.

    The surrogate tier (R2) draws its error as a one-factor field,

        eps_d = sqrt(c) * eps_common + sqrt(1-c) * eps_d_independent

    with c and the per-DC scales measured at DC level from 2020 residuals and never
    hand-rounded. eps_common comes from `common_key`, an episode-level key every site
    shares, so the common mode is genuinely common across sites.
    """
    spec = TIERS_V2[tier]
    lo, hi = t, min(len(series), t + horizon)
    truth = np.asarray(series[lo:hi], dtype=np.float64)
    if len(truth) == 0:
        return truth
    if spec["kind"] == "shuffle":
        rng = np.random.default_rng(domain_seed(series_key(series, site, tier), "perm"))
        perm = rng.permutation(len(series))
        out = np.maximum(0.0, np.asarray(series, dtype=np.float64)[perm][lo:hi])
    elif spec["kind"] == "anti":
        out = np.maximum(0.0, np.asarray(series, dtype=np.float64)[::-1][lo:hi])
    elif spec["kind"] == "surrogate":
        if not calibration:
            raise ValueError("the surrogate tier needs the DC-level calibration artifact")
        if common_key is None:
            raise ValueError("the surrogate tier needs an episode-level common_key")
        c = float(calibration["c"])
        rho = float(calibration["ar1_rho"])
        alpha = float(calibration["lead_alpha"])
        sigma = float(calibration["sigma_rel_dc"].get(str(site), 0.0))
        if sigma == 0.0:
            out = truth.copy()
        else:
            common = ar1_field(f"{common_key}:{tier}", len(series), rho)[lo:hi]
            indep = ar1_field(series_key(series, site, tier), len(series), rho)[lo:hi]
            eps = np.sqrt(c) * common + np.sqrt(1.0 - c) * indep
            scale_ref = float(np.mean(np.abs(series))) or 1.0
            leads = np.arange(len(truth), dtype=np.float64)
            out = np.maximum(0.0, truth + lead_scale(leads, horizon, alpha)
                             * sigma * scale_ref * eps)
    else:
        sigma = spec["sigma_rel"]
        if sigma == 0.0:
            out = truth.copy()
        else:
            eps = ar1_field(series_key(series, site, tier), len(series), AR1_RHO)[lo:hi]
            scale_ref = float(np.mean(np.abs(series))) or 1.0
            leads = np.arange(len(truth), dtype=np.float64)
            out = np.maximum(0.0, truth + lead_scale(leads, horizon, LEAD_ALPHA)
                             * sigma * scale_ref * eps)
    out[0] = truth[0]                      # lead 0 is an observation, never corrupted
    return out


def load_calibration(path: str) -> dict:
    cal = json.load(open(path))
    for field in ("sigma_rel", "ar1_rho", "lead_alpha", "source_checkpoint_sha"):
        if field not in cal:
            raise ValueError(f"calibration artifact is missing {field!r}")
    return cal
