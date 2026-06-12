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
        forecast_every: int = 6,
        short_term_steps: int = 3,
        long_term_steps: int = 144,
        device: str = "cpu",
        seq_len: int = 96,
        pred_len: int = 144,
        csv_start_offset: int = 0,
        dc_tz_offsets: Optional[Dict[int, int]] = None,
        simulation_warmup_rows: int = 0,
        forecast_shift: Optional[Dict] = None,
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
        forecast_shift:
            Optional synthetic forecast-error injection for the EU-CRD
            conservative-collapse experiment. Applied to the raw per-turbine
            forecast right after the TimeCAP forward, BEFORE both the
            DC-feature aggregation (what the agent sees) and the CRD
            predicted_wind_w accessor — so the agent is misled by exactly the
            forecast that R_forecast measures. Schema::

                {enabled: bool, mode: "scale"|"bias_kw",
                 factor: float,           # scale mode: pred *= factor
                 bias_kw: float,          # bias mode:  pred += bias_kw
                 start_step: int,         # in-episode window [start, end]
                 end_step: int}           # -1 = until episode end

            The realized wind is untouched: this is pure forecast error.
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
        )
        self.seq_len = self.predictor.seq_len
        self.pred_len = self.predictor.pred_len

        # Per-DC cache (last computed 4-tuple + last forecast step)
        self._last_features: Dict[int, np.ndarray] = {
            dc_id: _NEUTRAL_FEATURES.copy() for dc_id in self.dc_ids
        }
        self._last_forecast_step: Dict[int, int] = {dc_id: -10**9 for dc_id in self.dc_ids}
        # CRD M2.2 plumbing: cache the last raw per-turbine forecast so that
        # `_collect_crd_info` can extract a per-DC predicted_wind_w in W. We
        # cache `per_t_pred` (kW arrays of shape (pred_len,)) rather than
        # re-running the TimeCAP forward on demand. None until the first
        # forward has fired.
        self._last_per_t_pred: Optional[Dict[int, np.ndarray]] = None
        # Updates that have been pushed into the buffers but not yet inferenced
        self._dirty_steps: int = 0

        # Synthetic forecast-error injection (conservative-collapse experiment).
        self._shift_cfg = self._resolve_shift_cfg(forecast_shift)

        logger.info(
            "TimeCAPGodEyeProvider ready: dcs=%s, turbines=%s, "
            "forecast_every=%d, device=%s, warmup_rows=%d, dc_tz_offsets=%s, "
            "forecast_shift=%s",
            self.dc_ids,
            sorted(self.predictor.turbine_ids),
            self.forecast_every,
            device,
            self.simulation_warmup_rows,
            self.dc_tz_offsets if self.dc_tz_offsets else "(scalar fallback)",
            self._shift_cfg if self._shift_cfg else "off",
        )

    @staticmethod
    def _resolve_shift_cfg(raw: Optional[Dict]) -> Optional[Dict]:
        """Parse/validate the forecast_shift config; None when disabled."""
        if not isinstance(raw, dict) or not raw.get("enabled", False):
            return None
        mode = str(raw.get("mode", "scale")).lower()
        if mode not in ("scale", "bias_kw"):
            raise ValueError(
                f"forecast_shift.mode must be 'scale' or 'bias_kw', got {mode!r}"
            )
        return {
            "mode": mode,
            "factor": float(raw.get("factor", 1.0)),
            "bias_kw": float(raw.get("bias_kw", 0.0)),
            "start_step": int(raw.get("start_step", 0)),
            "end_step": int(raw.get("end_step", -1)),
        }

    @staticmethod
    def _apply_forecast_shift(
        per_t_pred: Dict[int, np.ndarray],
        simulation_step: int,
        cfg: Optional[Dict],
    ) -> Dict[int, np.ndarray]:
        """
        Apply the synthetic shift to a raw per-turbine forecast dict.

        Pure function (returns new arrays; input not mutated). Outside the
        in-episode [start_step, end_step] window — or with cfg=None — the
        input is returned unchanged. Shifted forecasts are clipped at >= 0
        (a negative wind forecast is physically meaningless and would leak
        the injection's magnitude into the waste-ratio formula's max(0,·)).
        """
        if cfg is None or per_t_pred is None:
            return per_t_pred
        start, end = cfg["start_step"], cfg["end_step"]
        if simulation_step < start or (end >= 0 and simulation_step > end):
            return per_t_pred
        out: Dict[int, np.ndarray] = {}
        for tid, pred in per_t_pred.items():
            if pred is None:
                out[tid] = pred
                continue
            if cfg["mode"] == "scale":
                shifted = pred * cfg["factor"]
            else:  # bias_kw
                shifted = pred + cfg["bias_kw"]
            out[tid] = np.clip(shifted, 0.0, None).astype(pred.dtype, copy=False)
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all rolling buffers and reset the per-DC cache. Call inside env.reset()."""
        self.predictor.reset()
        for dc_id in self.dc_ids:
            self._last_features[dc_id] = _NEUTRAL_FEATURES.copy()
            self._last_forecast_step[dc_id] = -10**9
        # Drop the cached raw forecast — a stale prediction across episodes
        # would be a worse signal than `predicted_wind_w` simply being absent.
        self._last_per_t_pred = None
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

        # Synthetic forecast-error injection (conservative-collapse experiment).
        # Applied BEFORE aggregation and BEFORE stashing, so every consumer —
        # the agent's dc_future_* features, CRD's predicted_wind_w, and the A1
        # raw-forecast ablation — sees the same shifted forecast.
        per_t_pred = self._apply_forecast_shift(
            per_t_pred, simulation_step, self._shift_cfg
        )

        for dc_id in stale_dcs:
            self._last_features[dc_id] = self._aggregate_dc(
                per_t_pred, self.dc_assignments[dc_id]
            )
            self._last_forecast_step[dc_id] = simulation_step

        # Stash the raw per-turbine forecast for the CRD framework's
        # forecast counterfactual. M2.2 reads horizon-0 predictions from
        # `info["crd"]["predicted_wind_w"]`; we expose those via
        # :meth:`get_predicted_wind_w_per_dc` below.
        self._last_per_t_pred = per_t_pred

        return self._features_snapshot()

    def get_features_array(self, simulation_step: int) -> np.ndarray:
        """
        Same as :meth:`get_features` but stacked into a ``(num_dcs, 4)`` numpy
        array indexed by ``self.dc_ids`` order. Convenient for direct
        assignment into the env's obs dict.
        """
        feats = self.get_features(simulation_step)
        return np.stack([feats[dc_id] for dc_id in self.dc_ids], axis=0)

    def get_predicted_wind_w_per_dc(
        self, horizon: int = 0
    ) -> Optional[List[float]]:
        """
        Return per-DC predicted wind power (W) at the given forecast horizon.

        Used by the CRD framework's forecast counterfactual (M2.2) so that
        ``info["crd"]["predicted_wind_w"]`` carries Ŵ_t alongside the M0
        snapshot fields. The list is ordered by ``self.dc_ids`` to match the
        rest of the per-DC arrays the env writes.

        Returns ``None`` when no forecast has been computed yet (e.g., before
        the first :meth:`update`/:meth:`get_features` call after reset, during
        warmup). Callers should treat None as "no prediction available" and
        leave the field absent from info.

        Args:
            horizon: which step into ``predict_per_turbine`` output to read
                from. ``0`` means "the next step after now", which matches
                the CRD plan's notion of Ŵ_t — the forecast that fed the
                agent's decision at this step.
        """
        if self._last_per_t_pred is None:
            return None
        out: List[float] = []
        for dc_id in self.dc_ids:
            total_kw = 0.0
            for tid in self.dc_assignments.get(dc_id, []):
                pred = self._last_per_t_pred.get(tid)
                if pred is None or pred.size <= horizon:
                    continue
                total_kw += float(pred[horizon])
            out.append(total_kw * 1000.0)  # kW → W (matches actual_wind_w units)
        return out

    def get_raw_forecast_per_dc(
        self,
        horizon: Optional[int] = None,
        normalize: bool = True,
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Return per-DC raw forecast trajectory over ``horizon`` future steps.

        Used by the A1 "HiGreen-Raw" ablation env, which exposes raw multi-step
        forecasts to the policy *instead of* the 4 compressed features
        (μ^short, τ^short, μ^long, φ^peak). The whole point of that ablation is
        to test whether semantic state compression is necessary — see paper §IV
        and the Contribution 3 claim about "representational entanglement".

        Aggregation rule (mirrors :meth:`_aggregate_dc` for the means):

            raw_DC[h] = (Σ_t pred_t[h]) / (Σ_t maxPower_t)        if normalize
            raw_DC[h] = (Σ_t pred_t[h]) * 1000                    if not normalize
                                                                  (kW → W)

        i.e. when ``normalize=True`` we max-power-weighted-average across the
        DC's turbines (output ∈ [0, 1], same dimensional regime as the
        compressed features); when ``normalize=False`` we sum to a total kW
        signal and convert to Watts (matches :meth:`get_predicted_wind_w_per_dc`).

        Args:
            horizon: number of future steps to return per DC. Defaults to the
                provider's ``pred_len`` (the full TimeCAP forecast horizon).
                Clamped to ``pred_len`` if larger; values <1 raise ValueError.
            normalize: see formula above.

        Returns:
            ``{dc_id: np.ndarray shape (horizon,)}`` or ``None`` if no forecast
            has been computed yet (e.g., before the first :meth:`step_and_get`
            call after :meth:`reset`).
        """
        if self._last_per_t_pred is None:
            return None
        if horizon is None:
            horizon = self.pred_len
        horizon = min(int(horizon), self.pred_len)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        out: Dict[int, np.ndarray] = {}
        for dc_id in self.dc_ids:
            turbine_ids = self.dc_assignments.get(dc_id, [])
            max_powers = [self.predictor.max_power_kw.get(t, 1.0) for t in turbine_ids]
            total_mp = float(sum(max_powers))

            agg_kw = np.zeros(horizon, dtype=np.float32)
            for tid, mp in zip(turbine_ids, max_powers):
                pred = self._last_per_t_pred.get(tid)
                if pred is None or pred.size == 0 or mp <= 0.0:
                    continue
                h = min(horizon, int(pred.size))
                agg_kw[:h] += pred[:h].astype(np.float32)

            if normalize:
                if total_mp <= 0.0:
                    out[dc_id] = np.full(horizon, 0.5, dtype=np.float32)
                else:
                    out[dc_id] = np.clip(agg_kw / total_mp, 0.0, 1.0).astype(np.float32)
            else:
                out[dc_id] = (agg_kw * 1000.0).astype(np.float32)  # kW → W

        return out

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
