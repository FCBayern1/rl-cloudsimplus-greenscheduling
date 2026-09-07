"""PerturbedGodEyeProvider — God's Eye truth pushed through the frozen perturbation ladder.

Why this exists. Until now the corruption ladder lived only on the planner side
(`src/baselines/forecast_perturb`), while the RL policy's forecast observation came from
either Java's God's Eye (true future) or the TimeCAP provider (a real prediction). So the
question "does a policy trained against a degraded forecast get fooled, or does it learn to
ignore the channel" had no instrument. This provider is that instrument: it reads the same
true future God's Eye reads, pushes it through the same tiers the planner arms use, and
emits the same four DC features the policy already consumes.

What was inventoried before writing it (the work order asked, and the answer matters):
`HierarchicalMultiDCEnv._perturb_forecast` already implements `blend / shuffle / anti /
panti / bias / pshuffle` behind `FORECAST_PERTURB_MODE`. **It is not reusable here, and the
names actively mislead.** That mechanism perturbs the four AGGREGATED FEATURES after they
are computed; the E-line perturbs the 144-row SERIES before the features exist. Two names
collide with opposite meanings:

    name       env._perturb_forecast (feature space)   forecast_perturb (series space)
    shuffle    reverse the per-DC axis                 permute the time axis
    anti       1 - feature, trend negated              reverse the time axis

An env-side `anti` mirrors a value; an E-line `anti` reverses time. Wiring one while
believing the other would silently answer a different question, so this provider goes
through `forecast_perturb` and the env-side path is left untouched.

Semantics, identical to the E line by construction because the same functions are called:

    lead 0 is the measured present and is never corrupted (enforced inside
        perturbed_future_v2 / audited_future)
    only leads 1..pred_len-1 are touched
    every tier is a deterministic function of (series bytes, site, tier) plus an
        episode-level common key, so two runs and two machines agree
    the world and the settlement always use the truth; only the observation changes

One deviation is registered rather than hidden. The planner perturbs a DC-aggregated
series with site = dc_id; this provider perturbs each TURBINE's series with site = dc_id,
because the four features are defined per turbine (peak_timing is the earliest peak ACROSS
the DC's turbines, not the peak of their sum, so aggregating first would change the feature
definition and break the byte-for-byte godeye equivalence this provider is tested on).
Consequence: on a two-turbine DC the independent component of the residual is drawn twice
and partially averages out in the DC sum, so DC0 and DC1 carry slightly less independent
noise than a DC-level draw would give; DC2 has one turbine and is exact. The common
(cross-site) component is unaffected. Flagged for whoever calibrates against DC-level
numbers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRLMANAGER = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DRLMANAGER not in sys.path:
    sys.path.insert(0, _DRLMANAGER)

from src.baselines import forecast_perturb as fp                       # noqa: E402
from src.prediction.csv_feature_loader import CSVFeatureLoader         # noqa: E402

NEUTRAL_FEATURES = np.array([0.5, 0.0, 0.5, 0.5], dtype=np.float32)

# Tiers this provider accepts. godeye is the identity and must reproduce Java's God's Eye
# bit for bit; the rest are the frozen ladder's.
SUPPORTED_TIERS = ("godeye", "s05", "s15", "s30", "s60",
                   # RL_V2: the ladder's amplitude-shrink rungs (view = m + lam * (truth - m), m the
                   # turbine's full-series mean; per-DC sums shrink around the DC's mean exactly as
                   # ladder_run.rung_curve does), tested equal to rung_curve on one window
                   "shrink75", "shrink50", "shrink25", "shrink0",
                   "shrink75", "shrink50", "shrink25", "shrink0",
                   "shuffle", "anti", "calibrated_shrink_v1")
DEFAULT_TIER = "godeye"


class PerturbedGodEyeProvider:
    """Per-DC forecast features computed from a deliberately degraded view of the truth.

    Parameters mirror TimeCAPGodEyeProvider where they mean the same thing, so the two can
    be swapped without touching anything downstream.
    """

    def __init__(
        self,
        dc_assignments: Dict[int, List[int]],
        turbine_csv_paths: Dict[int, str],
        perturb_tier: str = DEFAULT_TIER,
        pred_len: int = 144,
        short_term_steps: int = 3,
        long_term_steps: int = 144,
        error_params: Optional[dict] = None,
        feature_columns: Optional[Sequence[str]] = None,
        csv_start_offset: int = 0,
        dc_tz_offsets: Optional[Dict[int, int]] = None,
        simulation_warmup_rows: int = 0,
    ):
        if perturb_tier not in SUPPORTED_TIERS:
            raise ValueError(f"unknown perturb_tier {perturb_tier!r}; "
                             f"supported: {sorted(SUPPORTED_TIERS)}")
        if perturb_tier == "calibrated_shrink_v1" and not error_params:
            raise ValueError("perturb_tier='calibrated_shrink_v1' needs error_params "
                             "(the real-error audit's primary_error_params block)")
        self.dc_assignments = {int(d): [int(t) for t in ts]
                               for d, ts in dc_assignments.items()}
        self.perturb_tier = perturb_tier
        self.pred_len = int(pred_len)
        self.short_term_steps = max(1, int(short_term_steps))
        self.long_term_steps = max(1, int(long_term_steps))
        self.error_params = error_params
        self.dc_ids: List[int] = sorted(self.dc_assignments)
        # Row mapping, composed exactly as TimeCAPGodEyeProvider composes it: a per-DC
        # time zone plus the global warm-up when tz offsets are given, otherwise the
        # scalar fallback. Getting this wrong does not crash, it just serves a different
        # hour's weather than the simulator is burning, so it is mirrored rather than
        # reinvented.
        self.simulation_warmup_rows = max(0, int(simulation_warmup_rows))
        self.dc_tz_offsets = {int(k): int(v) for k, v in (dc_tz_offsets or {}).items()}
        self.row_offset: Dict[int, int] = {}
        for d, tids in self.dc_assignments.items():
            if self.dc_tz_offsets:
                off = self.dc_tz_offsets.get(d, 0) + self.simulation_warmup_rows
            else:
                off = int(csv_start_offset) + self.simulation_warmup_rows
            for t in tids:
                self.row_offset[t] = off

        self.loader = CSVFeatureLoader(
            turbine_csv_paths=turbine_csv_paths,
            feature_columns=list(feature_columns) if feature_columns else None,
        )
        # Patv is the only column this provider reads: it is the God's Eye quantity.
        self.truth: Dict[int, np.ndarray] = {}
        self.max_power_kw: Dict[int, float] = {}
        for tid in turbine_csv_paths:
            df = self.loader.turbine_data.get(int(tid))
            if df is None or "Patv" not in df.columns:
                raise ValueError(f"turbine {tid}: no Patv column; cannot serve God's Eye")
            series = df["Patv"].to_numpy(dtype=np.float64)
            self.truth[int(tid)] = np.nan_to_num(series, nan=0.0)
            mx = float(np.nanmax(series))
            self.max_power_kw[int(tid)] = mx if mx > 0 else 1.0

        self._episode_key = self._build_episode_key()
        self._cache: Dict[int, Dict[int, np.ndarray]] = {}
        self._last_per_t: Optional[Dict[int, np.ndarray]] = None
        logger.info("PerturbedGodEyeProvider ready: tier=%s dcs=%s pred_len=%d",
                    self.perturb_tier, sorted(self.dc_assignments), self.pred_len)

    # -- lifecycle ---------------------------------------------------------------------
    def _build_episode_key(self) -> str:
        """One key per (dataset, DC layout), shared by every site.

        Mirrors the planner arm, which hashes its whole truth grid: the common component of
        the correlated residual has to be the SAME draw for every site or the cross-site
        structure the audit measured is not reproduced.
        """
        h = hashlib.sha256()
        for d in sorted(self.dc_assignments):
            for tid in self.dc_assignments[d]:
                h.update(np.ascontiguousarray(self.truth[tid]).tobytes())
        return h.hexdigest()

    def reset(self) -> None:
        self._cache.clear()
        self._last_per_t = None

    def warmup(self, start_step: int = 0) -> None:
        """No history buffer to fill: the truth is read directly. Kept for interface
        parity with TimeCAPGodEyeProvider so callers need no branch."""
        return None

    def update(self, simulation_step: int) -> None:
        """Also a no-op, and also kept for parity. God's Eye has nothing to accumulate."""
        return None

    # -- the view ----------------------------------------------------------------------
    def _perturbed_series(self, turbine_id: int, dc_id: int, step: int) -> np.ndarray:
        """The degraded view of rows [step, step + pred_len) for one turbine."""
        series = self.truth[turbine_id]
        step = int(step) + self.row_offset.get(turbine_id, 0)
        if self.perturb_tier == "calibrated_shrink_v1":
            out = fp.perturbed_future_e(series, step, self.pred_len, dc_id,
                                        self.perturb_tier,
                                        eparams=self.error_params,
                                        common_key=self._episode_key)
        else:
            out = fp.perturbed_future_v2(series, step, self.pred_len, dc_id,
                                         self.perturb_tier,
                                         common_key=self._episode_key)
        return np.asarray(out, dtype=np.float64)

    def true_series(self, turbine_id: int, step: int) -> np.ndarray:
        s = self.truth[turbine_id]
        r = int(step) + self.row_offset.get(turbine_id, 0)
        return s[r:min(len(s), r + self.pred_len)]

    def _aggregate_dc(self, per_t: Dict[int, np.ndarray], turbine_ids: List[int]):
        """Java's aggregation, reproduced. Copied in behaviour from
        TimeCAPGodEyeProvider._aggregate_dc so the two providers are interchangeable:
        maxPower-weighted means for the three level/trend features, EARLIEST peak across
        the DC's turbines for peak timing."""
        max_powers = [self.max_power_kw.get(t, 1.0) for t in turbine_ids]
        total_mp = float(sum(max_powers))
        if total_mp <= 0.0:
            return NEUTRAL_FEATURES.copy()
        st = min(self.short_term_steps, self.pred_len)
        lt = min(self.long_term_steps, self.pred_len)
        w_sm = w_st = w_lm = 0.0
        earliest_peak = 1.0
        for tid, mp in zip(turbine_ids, max_powers):
            pred = per_t.get(tid)
            if pred is None or pred.size == 0 or mp <= 0.0:
                continue
            sm = float(np.clip(float(np.mean(pred[:st])) / mp, 0.0, 1.0))
            tr = float(pred[st - 1] - pred[0]) / mp if st >= 2 else 0.0
            tr = float(np.clip(tr, -1.0, 1.0))
            lm = float(np.clip(float(np.mean(pred[:lt])) / mp, 0.0, 1.0))
            pt = float(np.clip(float(int(np.argmax(pred[:lt]))) / max(lt - 1, 1), 0.0, 1.0))
            w_sm += sm * mp
            w_st += tr * mp
            w_lm += lm * mp
            if pt < earliest_peak:
                earliest_peak = pt
        return np.array([min(1.0, w_sm / total_mp),
                         max(-1.0, min(1.0, w_st / total_mp)),
                         min(1.0, w_lm / total_mp),
                         earliest_peak], dtype=np.float32)

    def get_features(self, simulation_step: int) -> Dict[int, np.ndarray]:
        step = int(simulation_step)
        if step in self._cache:
            return {d: v.copy() for d, v in self._cache[step].items()}
        out, last = {}, {}
        for d, tids in self.dc_assignments.items():
            per_t = {t: self._perturbed_series(t, d, step) for t in tids}
            last.update(per_t)
            out[d] = self._aggregate_dc(per_t, tids)
        self._last_per_t = last
        self._cache[step] = out
        return {d: v.copy() for d, v in out.items()}

    def step_and_get(self, simulation_step: int) -> Dict[int, np.ndarray]:
        self.update(simulation_step)
        return self.get_features(simulation_step)

    # -- the two extra channels the env reads off a provider ---------------------------
    def get_raw_forecast_per_dc(self, horizon: Optional[int] = None,
                                normalize: bool = True) -> Optional[Dict[int, np.ndarray]]:
        """Per-DC forecast trajectory, same aggregation rule as TimeCAPGodEyeProvider:
        max-power-weighted mean when normalized, summed kW converted to W when not."""
        if self._last_per_t is None:
            return None
        h = self.pred_len if horizon is None else int(horizon)
        if h < 1:
            raise ValueError(f"horizon must be >= 1, got {h}")
        h = min(h, self.pred_len)
        out = {}
        for d, tids in self.dc_assignments.items():
            total_mp = float(sum(self.max_power_kw.get(t, 1.0) for t in tids)) or 1.0
            acc = np.zeros(h, dtype=np.float64)
            for t in tids:
                pr = self._last_per_t.get(t)
                if pr is not None and pr.size:
                    acc[:min(h, pr.size)] += pr[:h]
            out[d] = (acc / total_mp) if normalize else (acc * 1000.0)
        return out

    def get_predicted_wind_w_per_dc(self, horizon: int = 0) -> Optional[List[float]]:
        """Per-DC predicted wind in W at one lead, ordered by dc_ids.

        None before the first get_features of an episode, matching the TimeCAP provider's
        contract: callers treat None as "no prediction yet" and omit the field.
        """
        if self._last_per_t is None:
            return None
        h = int(horizon)
        vals = []
        for d in self.dc_ids:
            tot = 0.0
            for t in self.dc_assignments[d]:
                pr = self._last_per_t.get(t)
                if pr is not None and pr.size > h:
                    tot += float(pr[h])
            vals.append(tot * 1000.0)
        return vals

    # -- introspection, for tests and run records --------------------------------------
    def describe(self) -> dict:
        return {
            "provider": "perturbed_godeye",
            "perturb_tier": self.perturb_tier,
            "pred_len": self.pred_len,
            "short_term_steps": self.short_term_steps,
            "long_term_steps": self.long_term_steps,
            "dc_assignments": {str(d): list(t) for d, t in self.dc_assignments.items()},
            "episode_key": self._episode_key,
            "max_power_kw": {str(k): v for k, v in self.max_power_kw.items()},
            "uses_error_params": self.error_params is not None,
            "row_offset": {str(k): v for k, v in self.row_offset.items()},
        }


def from_config(config: dict, dc_assignments: Dict[int, List[int]],
                turbine_csv_paths: Dict[int, str]) -> PerturbedGodEyeProvider:
    """Build the provider from an experiment block.

        green_oracle_mode: perturbed_godeye
        perturb_tier: shrink50           # default godeye, i.e. the identity
        perturb_error_params: <path>     # only for calibrated_shrink_v1

    The mode key is validated here so a typo fails at construction rather than silently
    falling back to the true forecast, which would make a contaminated arm quietly clean.
    """
    mode = str(config.get("green_oracle_mode", "")).strip().lower()
    if mode != "perturbed_godeye":
        raise ValueError(f"green_oracle_mode must be 'perturbed_godeye', got {mode!r}")
    tier = str(config.get("perturb_tier", DEFAULT_TIER)).strip()
    params = None
    p = config.get("perturb_error_params")
    if p:
        import json
        params = json.load(open(p))
        params = params.get("primary_error_params", params)
    return PerturbedGodEyeProvider(
        dc_assignments=dc_assignments,
        turbine_csv_paths=turbine_csv_paths,
        perturb_tier=tier,
        pred_len=int(config.get("timecap", {}).get("pred_len", 144)),
        short_term_steps=int(config.get("forecast_short_term_rows", 3)),
        long_term_steps=int(config.get("forecast_long_term_rows", 144)),
        error_params=params,
        csv_start_offset=int((config.get("timecap") or {}).get("csv_start_offset", 0)),
        dc_tz_offsets=config.get("dc_tz_offsets"),
        simulation_warmup_rows=int(config.get("simulation_warmup_rows", 0)),
    )
