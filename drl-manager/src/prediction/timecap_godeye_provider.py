"""
TimeCAPGodEyeProvider
=====================
Drop-in replacement for the Java-side God's Eye (which reads ground-truth
future Patv from CSVs). This provider runs the fine-tuned TimeCAP model on
each turbine's rolling history and aggregates the resulting forecasts to the
4 DC-level features that the RL policy already consumes:

    dc_future_short_mean       ∈ [0, 1]   mean(forecast[:short_term_steps]) / maxPower
    dc_future_short_trend      ∈ [-1, 1]  (forecast[short-1] - forecast[0]) / maxPower
    dc_future_long_mean        ∈ [0, 1]   mean(forecast[:long_term_steps])  / maxPower
    dc_future_long_peak_timing ∈ [0, 1]   argmax(forecast[:long_term_steps]) / (long-1)

The cross-turbine aggregation **mirrors Java's
GreenEnergyProvider.computeAggregatedFutureTrendFeatures()** byte-for-byte:

    short_mean_DC  = Σ short_mean_t  · maxPower_t  /  Σ maxPower_t
    short_trend_DC = Σ short_trend_t · maxPower_t  /  Σ maxPower_t
    long_mean_DC   = Σ long_mean_t   · maxPower_t  /  Σ maxPower_t
    peak_timing_DC = min   peak_timing_t      ← *earliest* peak across turbines, NOT a mean

This makes the provider a 1:1 functional replacement for Java's God's Eye —
swap-in / swap-out yields directly comparable RL observations (oracle vs.
forecast) with no other code changes downstream.

Architecture notes
------------------
* **Single shared model.** All turbines (across all DCs) live inside one
  TimeCAP_GreenPredictor, so the ~50M-parameter network is loaded into memory
  exactly once. Per-turbine state (history buffer, max_power_kw) is what the
  predictor already keeps as dicts keyed by turbine_id.
* **Caching.** The four features change slowly relative to a 10-min step
  (long_mean / peak_timing barely move within a few hours), so a configurable
  `forecast_every` parameter throttles how often the heavy forward pass runs.
  Between refreshes the last computed 4-tuple is returned. Set to 1 for "every
  step" (recommended on GPU).
* **Cold start.** `predict_per_turbine()` zero-pads the buffer so calls
  during the first `seq_len` (=96) steps still return something — but those
  forecasts are degraded. Optionally call :meth:`warmup` after :meth:`reset`
  to push the first 96 CSV rows of history into each buffer before any
  forecast is consumed.

Usage
-----
    provider = TimeCAPGodEyeProvider(
        dc_assignments    = {0: [1, 15], 1: [30, 60]},   # dc_id → list of turbine_ids
        turbine_csv_paths = {1: "...Turbine_1_2021.csv", 15: "...", ...},
        checkpoint_path   = "drl-manager/timecap_prediction/TimeCAP/model/"
                            "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth",
        feature_set       = "v1",      # v1 (baseline) or v2 (Phase 1)
        forecast_every    = 6,         # one TimeCAP forward per simulated hour
        device            = "cpu",
    )
    provider.reset()
    for step in range(N):
        feats = provider.step_and_get(step)         # dict {dc_id: np.ndarray (4,)}
        # feats[dc_id][0] = short_mean
        # feats[dc_id][1] = short_trend
        # feats[dc_id][2] = long_mean
        # feats[dc_id][3] = peak_timing
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Make `from timecap_prediction.predictor import ...` work regardless of cwd
_DRLMANAGER = Path(__file__).resolve().parents[2]
if str(_DRLMANAGER) not in sys.path:
    sys.path.insert(0, str(_DRLMANAGER))

from timecap_prediction.predictor import TimeCAP_GreenPredictor  # noqa: E402

logger = logging.getLogger(__name__)


# Defaults (mirror Java's RL-side feature definitions)
_NEUTRAL_FEATURES = np.array([0.5, 0.0, 0.5, 0.5], dtype=np.float32)


class TimeCAPGodEyeProvider:
    """
    Wraps a single TimeCAP_GreenPredictor and turns its per-turbine forecasts
    into the 4 DC-level God's Eye features the RL env expects.
    """

    NUM_FEATURES = 4

    def __init__(
        self,
        dc_assignments: Dict[int, List[int]],
        turbine_csv_paths: Dict[int, str],
        checkpoint_path: str,
        *,
        feature_set: str = "v1",
        auto_derive_features: Optional[bool] = None,
        forecast_every: int = 6,
        short_term_steps: int = 3,
        long_term_steps: int = 144,
        device: str = "cpu",
        seq_len: int = 96,
        pred_len: int = 144,
        csv_start_offset: int = 0,
        dc_tz_offsets: Optional[Dict[int, int]] = None,
        simulation_warmup_rows: int = 0,
    ):
        """
        Parameters
        ----------
        dc_assignments:
            ``{dc_id: [turbine_id, ...]}`` — which turbines belong to which DC.
            Aggregation is done per DC. dc_ids must be contiguous integers
            starting at 0 if the caller intends to vectorise into a numpy array.
        turbine_csv_paths:
            ``{turbine_id: csv_path}`` for every turbine appearing in
            ``dc_assignments``. Extra turbines are allowed and ignored.
        checkpoint_path:
            TimeCAP fine-tuned ``ckpt_best.pth``. ``model_args.json`` must sit
            next to it (training writes it automatically).
        feature_set:
            "v1" → baseline 13-feature schema (matches ckpt 4358062);
            "v2" → Phase 1 23-feature schema (engineered + cyclical time).
        auto_derive_features:
            Whether to auto-compute v2 engineered features at load time. If
            None, defaults to True for v2 and False for v1.
        forecast_every:
            Re-run the TimeCAP forward pass every N simulation steps; between
            refreshes the cached 4-tuple is returned. With 10-min steps,
            ``forecast_every=6`` means "one forecast per simulated hour".
            Set to 1 for "every step" (only sensible on GPU).
        short_term_steps / long_term_steps:
            Match Java's RL-side feature horizons (default 3 and 144).
        device:
            torch device string, e.g. "cpu", "cuda", "cuda:0".
        seq_len / pred_len:
            Forwarded to TimeCAP_GreenPredictor; defaults match the trained
            checkpoint.
        csv_start_offset:
            Fallback per-turbine row offset used when ``dc_tz_offsets`` is None
            (single-time-zone setups). Ignored when dc_tz_offsets is provided
            because per-turbine offsets are then derived from
            ``tz_offset[dc] + simulation_warmup_rows``.
        dc_tz_offsets:
            ``{dc_id: time_zone_offset_rows}`` — mirrors Java's per-DC
            ``time_zone_offset_rows`` config. Required for multi-DC setups
            with non-zero geographic time-zone offsets, otherwise the TimeCAP
            forecasts will be misaligned with Java's oracle reference.
        simulation_warmup_rows:
            Global row offset added on top of every dc's tz_offset. Must match
            the Java side's ``simulation_warmup_rows`` config. Recommended 96
            (= seq_len) so the buffer at sim_step=0 is filled with real CSV
            rows rather than zero-padded — eliminates the cold-start period.
        """
        if not dc_assignments:
            raise ValueError("dc_assignments must be non-empty.")

        # Validate that every turbine in dc_assignments has a CSV path
        all_turbines: List[int] = []
        for dc_id, tids in dc_assignments.items():
            if not tids:
                raise ValueError(f"DC {dc_id} has no turbines.")
            all_turbines.extend(tids)
        missing_paths = [t for t in all_turbines if t not in turbine_csv_paths]
        if missing_paths:
            raise ValueError(
                f"Missing CSV path for turbine_id(s) {missing_paths}; "
                f"got paths for {sorted(turbine_csv_paths.keys())}"
            )

        self.dc_assignments: Dict[int, List[int]] = {
            int(dc_id): list(tids) for dc_id, tids in dc_assignments.items()
        }
        self.dc_ids: List[int] = sorted(self.dc_assignments.keys())
        self.short_term_steps = max(1, int(short_term_steps))
        self.long_term_steps = max(1, int(long_term_steps))
        self.forecast_every = max(1, int(forecast_every))

        if auto_derive_features is None:
            auto_derive_features = (feature_set == "v2")

        # Filter csv_paths down to just the turbines we actually need
        used_csv_paths = {tid: turbine_csv_paths[tid] for tid in all_turbines}

        # Build per-turbine row offset to mirror Java's GreenEnergyProvider:
        #   csv_row(tid, sim_step) = sim_step + tz_offset_dc(tid) + warmup
        # When dc_tz_offsets is None, every turbine uses the scalar fallback.
        self.simulation_warmup_rows = max(0, int(simulation_warmup_rows))
        self.dc_tz_offsets: Dict[int, int] = {
            int(k): int(v) for k, v in (dc_tz_offsets or {}).items()
        }
        if self.dc_tz_offsets:
            per_turbine_offset: Dict[int, int] = {}
            for dc_id, tids in self.dc_assignments.items():
                tz = self.dc_tz_offsets.get(dc_id, 0)
                for tid in tids:
                    per_turbine_offset[tid] = tz + self.simulation_warmup_rows
            effective_offset = per_turbine_offset
        else:
            effective_offset = csv_start_offset + self.simulation_warmup_rows

        self.predictor = TimeCAP_GreenPredictor(
            checkpoint_path=checkpoint_path,
            turbine_csv_paths=used_csv_paths,
            seq_len=seq_len,
            pred_len=pred_len,
            short_term_steps=short_term_steps,
            long_term_steps=long_term_steps,
            device=device,
            csv_start_offset=effective_offset,
            feature_set=feature_set,
            auto_derive_features=auto_derive_features,
        )
        self.seq_len = self.predictor.seq_len
        self.pred_len = self.predictor.pred_len

        # Per-DC cache (last computed 4-tuple + last forecast step)
        self._last_features: Dict[int, np.ndarray] = {
            dc_id: _NEUTRAL_FEATURES.copy() for dc_id in self.dc_ids
        }
        self._last_forecast_step: Dict[int, int] = {dc_id: -10**9 for dc_id in self.dc_ids}
        # Updates that have been pushed into the buffers but not yet inferenced
        self._dirty_steps: int = 0

        logger.info(
            "TimeCAPGodEyeProvider ready: dcs=%s, turbines=%s, feature_set=%s, "
            "forecast_every=%d, device=%s, warmup_rows=%d, dc_tz_offsets=%s",
            self.dc_ids,
            sorted(self.predictor.turbine_ids),
            feature_set,
            self.forecast_every,
            device,
            self.simulation_warmup_rows,
            self.dc_tz_offsets if self.dc_tz_offsets else "(scalar fallback)",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all rolling buffers and reset the per-DC cache. Call inside env.reset()."""
        self.predictor.reset()
        for dc_id in self.dc_ids:
            self._last_features[dc_id] = _NEUTRAL_FEATURES.copy()
            self._last_forecast_step[dc_id] = -10**9
        self._dirty_steps = 0

    def warmup(self, start_step: int = 0) -> None:
        """
        Push ``seq_len`` real CSV rows into every turbine buffer before the
        first forecast is consumed. Use immediately after :meth:`reset` if
        you'd rather not see degraded zero-padded forecasts during the first
        16 hours of simulation.
        """
        for s in range(start_step, start_step + self.seq_len):
            self.predictor.update(s)
        self._dirty_steps = self.seq_len

    def update(self, simulation_step: int) -> None:
        """Push the row at ``simulation_step`` into every turbine's history buffer."""
        self.predictor.update(simulation_step)
        self._dirty_steps += 1

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step_and_get(self, simulation_step: int) -> Dict[int, np.ndarray]:
        """
        Convenience: ``update(step)`` then return the (cached or freshly
        computed) per-DC 4-tuple. Returns ``{dc_id: np.ndarray shape (4,)}``.
        """
        self.update(simulation_step)
        return self.get_features(simulation_step)

    def get_features(self, simulation_step: int) -> Dict[int, np.ndarray]:
        """
        Return per-DC features for the given step. Internally re-runs the
        TimeCAP forward only if at least one DC has gone ``forecast_every``
        steps without a refresh.
        """
        # Decide which DCs need a refresh this call
        stale_dcs = [
            dc_id for dc_id in self.dc_ids
            if simulation_step - self._last_forecast_step[dc_id] >= self.forecast_every
        ]
        if not stale_dcs:
            return self._features_snapshot()

        per_t_pred = self.predictor.predict_per_turbine()
        if per_t_pred is None:
            logger.warning(
                "predict_per_turbine() returned None at step %d — keeping previous "
                "cached features.",
                simulation_step,
            )
            return self._features_snapshot()

        for dc_id in stale_dcs:
            self._last_features[dc_id] = self._aggregate_dc(
                per_t_pred, self.dc_assignments[dc_id]
            )
            self._last_forecast_step[dc_id] = simulation_step

        return self._features_snapshot()

    def get_features_array(self, simulation_step: int) -> np.ndarray:
        """
        Same as :meth:`get_features` but stacked into a ``(num_dcs, 4)`` numpy
        array indexed by ``self.dc_ids`` order. Convenient for direct
        assignment into the env's obs dict.
        """
        feats = self.get_features(simulation_step)
        return np.stack([feats[dc_id] for dc_id in self.dc_ids], axis=0)

    # ------------------------------------------------------------------
    # Aggregation (mirrors Java)
    # ------------------------------------------------------------------

    def _aggregate_dc(
        self,
        per_t_pred: Dict[int, np.ndarray],
        turbine_ids: List[int],
    ) -> np.ndarray:
        """
        Java-style aggregation across the turbines of one DC. Matches
        GreenEnergyProvider.computeAggregatedFutureTrendFeatures():

            short_mean / short_trend / long_mean : maxPower-weighted mean
            peak_timing                          : earliest (min) peak

        ``per_t_pred[t]`` is an unscaled kW forecast of shape (pred_len,).
        """
        max_powers = [self.predictor.max_power_kw.get(t, 1.0) for t in turbine_ids]
        total_mp = float(sum(max_powers))
        if total_mp <= 0.0:
            return _NEUTRAL_FEATURES.copy()

        st = min(self.short_term_steps, self.pred_len)
        lt = min(self.long_term_steps, self.pred_len)

        weighted_short_mean = 0.0
        weighted_short_trend = 0.0
        weighted_long_mean = 0.0
        earliest_peak = 1.0  # initial bound; Java uses 1.0 too

        for tid, mp in zip(turbine_ids, max_powers):
            pred = per_t_pred.get(tid)
            if pred is None or pred.size == 0 or mp <= 0.0:
                continue

            short_mean_t = float(np.mean(pred[:st])) / mp
            short_mean_t = float(np.clip(short_mean_t, 0.0, 1.0))

            if st >= 2:
                short_trend_t = float(pred[st - 1] - pred[0]) / mp
            else:
                short_trend_t = 0.0
            short_trend_t = float(np.clip(short_trend_t, -1.0, 1.0))

            long_mean_t = float(np.mean(pred[:lt])) / mp
            long_mean_t = float(np.clip(long_mean_t, 0.0, 1.0))

            peak_idx = int(np.argmax(pred[:lt]))
            peak_timing_t = float(peak_idx) / max(lt - 1, 1)
            peak_timing_t = float(np.clip(peak_timing_t, 0.0, 1.0))

            weighted_short_mean += short_mean_t * mp
            weighted_short_trend += short_trend_t * mp
            weighted_long_mean += long_mean_t * mp

            if peak_timing_t < earliest_peak:
                earliest_peak = peak_timing_t

        return np.array(
            [
                min(1.0, weighted_short_mean / total_mp),
                max(-1.0, min(1.0, weighted_short_trend / total_mp)),
                min(1.0, weighted_long_mean / total_mp),
                earliest_peak,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _features_snapshot(self) -> Dict[int, np.ndarray]:
        """Return a copy-protected snapshot of the per-DC cache."""
        return {dc_id: self._last_features[dc_id].copy() for dc_id in self.dc_ids}
