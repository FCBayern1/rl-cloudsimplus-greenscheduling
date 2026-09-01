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

Everything a tier derives -- scale_ref, the AR(1) field, the shuffle permutation, the
reversal -- is confined to the episode plus one horizon (`span`). The planner hands this
module `self.G[d]`, which CurveInformedPlanner fills for all 20000 grid steps whatever the
episode's length, and the frozen scheme-2 windows sit only 8072 rows apart. Deriving the
dose from the whole grid would let a DISCOVERY window's noise amplitude be set partly by
CONFIRMATION weather, and would make `anti` return rows twenty thousand steps away rather
than the episode reversed. `span` defaults to the whole series, which is right only when
the caller passes exactly the episode.

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

import functools
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


def tier_params(tier: str, calibration: dict | None = None):
    """(sigma_rel, rho, alpha) for a noise tier, resolving timecap_cal from the artifact."""
    spec = TIERS[tier]
    sigma, rho, alpha = spec.get("sigma_rel"), AR1_RHO, LEAD_ALPHA
    if tier == "timecap_cal":
        if not calibration:
            raise ValueError("tier timecap_cal needs the calibration artifact")
        sigma = float(calibration["sigma_rel"])
        rho = float(calibration.get("ar1_rho", AR1_RHO))
        alpha = float(calibration.get("lead_alpha", LEAD_ALPHA))
    return sigma, rho, alpha


class FrozenField:
    """Everything about one (site, tier, episode) corruption that does not depend on t.

    Built once per episode and reused for every planning step. This is not an optimisation
    of convenience: `_costs_all` calls `_green_view` once per (job, site), and rebuilding
    the AR(1) field -- a Python loop over the series -- plus a sha256 of the series cost
    5.7 ms per call at the production shape, which puts hours of pure noise generation into
    a single episode. Rebuilt per call or built once, the numbers are identical; only the
    ladder's runnability changes.
    """

    def __init__(self, series, site: int, tier: str, horizon: int,
                 calibration: dict | None = None, span: int | None = None):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; registered: {sorted(TIERS)}")
        self.series = np.ascontiguousarray(series, dtype=np.float64)
        n = self.series.size
        self.span = n if span is None else max(1, min(int(span), n))
        # The episode, plus the one horizon the last decision step can still look into.
        self.extent = min(n, self.span + int(horizon))
        self.tier, self.site, self.horizon = tier, site, int(horizon)
        window = self.series[:self.extent]
        key = series_key(window, site, tier)
        kind = TIERS[tier]["kind"]
        self.shuffled = self.reversed = self.eps = None
        self.sigma, self.rho, self.alpha = tier_params(tier, calibration)
        if kind == "shuffle":
            rng = np.random.default_rng(domain_seed(key, "perm"))
            self.shuffled = window[rng.permutation(self.extent)]
        elif kind == "anti":
            self.reversed = window[::-1].copy()
        elif self.sigma:
            self.eps = ar1_field(key, self.extent, self.rho)
            # The site's own level DURING the episode, not over the whole planning grid.
            self.scale_ref = float(np.mean(np.abs(self.series[:self.span]))) or 1.0

    def view(self, t: int, horizon: int | None = None) -> np.ndarray:
        """The corrupted view of rows [t, t+horizon), given the TRUE series.

        Rows before t and at or beyond t+horizon are the caller's business (measured past,
        shared tail); this method never returns them.
        """
        horizon = self.horizon if horizon is None else int(horizon)
        lo, hi = t, min(self.series.size, t + horizon)
        inner = min(hi, self.extent)
        if self.shuffled is not None:
            head = np.maximum(0.0, self.shuffled[lo:inner])
        elif self.reversed is not None:
            head = np.maximum(0.0, self.reversed[lo:inner])
        elif not self.sigma:
            return self.series[lo:hi].copy()
        else:
            truth = self.series[lo:inner]
            leads = np.arange(truth.size, dtype=np.float64)
            head = np.maximum(0.0, truth + lead_scale(leads, horizon, self.alpha)
                              * self.sigma * self.scale_ref * self.eps[lo:inner])
        if inner >= hi:
            return head
        # Only reachable past the episode's own end, where nothing is ever settled; the
        # truth is what every other arm in the family sees there.
        return np.concatenate([head, self.series[inner:hi]])


def perturbed_future(series: np.ndarray, t: int, horizon: int, site: int, tier: str,
                     calibration: dict | None = None,
                     span: int | None = None) -> np.ndarray:
    """The corrupted view of rows [t, t+horizon), given the TRUE series.

    Pure and stateless at the call site; the frozen field behind it is memoised on the
    content of the episode window, so a caller that loops over t pays for it once.
    """
    return _field_for(series, site, tier, horizon, span,
                      None if calibration is None
                      else json.dumps(calibration, sort_keys=True)).view(t, horizon)


@functools.lru_cache(maxsize=64)
def _field_cached(payload: bytes, shape: int, site: int, tier: str, horizon: int,
                  span: int | None, calibration_json: str | None) -> FrozenField:
    return FrozenField(np.frombuffer(payload, dtype=np.float64), site, tier, horizon,
                       None if calibration_json is None else json.loads(calibration_json),
                       span)


def _field_for(series, site, tier, horizon, span, calibration_json):
    arr = np.ascontiguousarray(series, dtype=np.float64)
    return _field_cached(arr.tobytes(), arr.size, site, tier, int(horizon),
                         None if span is None else int(span), calibration_json)


def load_calibration(path: str) -> dict:
    cal = json.load(open(path))
    for field in ("sigma_rel", "ar1_rho", "lead_alpha", "source_checkpoint_sha"):
        if field not in cal:
            raise ValueError(f"calibration artifact is missing {field!r}")
    return cal
