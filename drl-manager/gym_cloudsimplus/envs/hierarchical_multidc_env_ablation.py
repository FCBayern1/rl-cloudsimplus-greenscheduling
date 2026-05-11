"""
HierarchicalMultiDCEnvAblation
==============================

Ablation env for the A1 study (Semantic State Compression).

Paper Contribution 3 claims that compressing the raw multi-step TimeCAP forecast
into 4 structural priors (μ^short, τ^short, μ^long, φ^peak) avoids
representational entanglement and noise propagation. The existing baselines
(PPO-MLP / gMLP / ResMLP) only test the function approximator; they all consume
the same 4 compressed features. To actually isolate the compression itself we
need ablation variants that vary *what enters the global obs*, keeping
everything else (env, policy, training) fixed.

This env adds one knob ``forecast_mode`` and inherits the rest from
``HierarchicalMultiDCEnv``. Supported values:

    forecast_mode        | global obs future block
    ---------------------+------------------------------------------------
    "full"               | all 4 compressed features  (= current HiGreen)
    "none"               | (drop all 4)               (= no-forecast bsl)
    "short_only"         | μ^short, τ^short
    "long_only"          | μ^long, φ^peak
    "no_peak"            | μ^short, τ^short, μ^long   (drop φ^peak)
    "raw"                | dc_future_raw (num_dc, horizon) — raw TimeCAP
                         | forecast trajectory, max-power-normalized so
                         | each value is in [0, 1] (same regime as μ^short).
                         | Requires ``green_oracle_mode="timecap"``.

Usage:
    >>> config = {..., "forecast_mode": "raw",
    ...           "forecast_raw_horizon": 144,
    ...           "green_oracle_mode": "timecap", ...}
    >>> env = HierarchicalMultiDCEnvAblation(config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from .hierarchical_multidc_env import HierarchicalMultiDCEnv

logger = logging.getLogger(__name__)


COMPRESSED_FORECAST_KEYS: List[str] = [
    "dc_future_short_mean",
    "dc_future_short_trend",
    "dc_future_long_mean",
    "dc_future_long_peak_timing",
]

RAW_FORECAST_KEY = "dc_future_raw"

VALID_FORECAST_MODES = {"full", "none", "short_only", "long_only", "no_peak", "raw"}


def _kept_keys_for_mode(mode: str) -> List[str]:
    """Which of the 4 compressed keys survive under each mode."""
    if mode == "full":
        return list(COMPRESSED_FORECAST_KEYS)
    if mode == "none" or mode == "raw":
        return []
    if mode == "short_only":
        return ["dc_future_short_mean", "dc_future_short_trend"]
    if mode == "long_only":
        return ["dc_future_long_mean", "dc_future_long_peak_timing"]
    if mode == "no_peak":
        return [
            "dc_future_short_mean",
            "dc_future_short_trend",
            "dc_future_long_mean",
        ]
    raise ValueError(f"Unknown forecast_mode={mode!r}; expected one of {sorted(VALID_FORECAST_MODES)}")


class HierarchicalMultiDCEnvAblation(HierarchicalMultiDCEnv):
    """
    Hierarchical multi-DC env with a configurable forecast feature block.

    See module docstring for the mode → obs-keys table. Everything else
    (action spaces, Java gateway, local obs, reward, step logic) is
    inherited unchanged from ``HierarchicalMultiDCEnv``.
    """

    def __init__(self, config: Dict[str, Any]):
        self.forecast_mode = str(config.get("forecast_mode", "full")).lower()
        if self.forecast_mode not in VALID_FORECAST_MODES:
            raise ValueError(
                f"config['forecast_mode']={self.forecast_mode!r}; "
                f"expected one of {sorted(VALID_FORECAST_MODES)}"
            )

        # Raw mode horizon defaults to TimeCAP's pred_len (144 = 24h of 10-min steps)
        self._forecast_raw_horizon = int(config.get("forecast_raw_horizon", 144))
        if self._forecast_raw_horizon < 1:
            raise ValueError(
                f"config['forecast_raw_horizon']={self._forecast_raw_horizon}; must be >=1"
            )

        super().__init__(config=config)

        if self.forecast_mode == "raw" and self.green_oracle_mode != "timecap":
            raise ValueError(
                "forecast_mode='raw' requires green_oracle_mode='timecap' "
                "(Java oracle only exposes the 4 compressed features, not raw "
                "multi-step trajectories)."
            )

        self._kept_compressed_keys = _kept_keys_for_mode(self.forecast_mode)
        self._emit_raw_forecast = self.forecast_mode == "raw"

        self._apply_forecast_mode_to_obs_space()

        logger.info(
            "HierarchicalMultiDCEnvAblation: forecast_mode=%s | kept_compressed=%s | "
            "raw=%s (horizon=%d)",
            self.forecast_mode,
            self._kept_compressed_keys,
            self._emit_raw_forecast,
            self._forecast_raw_horizon if self._emit_raw_forecast else 0,
        )

    # ------------------------------------------------------------------
    # Observation space surgery
    # ------------------------------------------------------------------

    def _apply_forecast_mode_to_obs_space(self) -> None:
        """
        Rebuild ``self.global_observation_space`` so it reflects the chosen
        ``forecast_mode``: drop compressed keys not in ``_kept_compressed_keys``,
        and insert ``dc_future_raw`` if raw mode is on.
        """
        parent_space = self.global_observation_space
        new_spaces: Dict[str, spaces.Space] = {}
        for key, sp in parent_space.spaces.items():
            if key in COMPRESSED_FORECAST_KEYS and key not in self._kept_compressed_keys:
                continue
            new_spaces[key] = sp

        if self._emit_raw_forecast:
            new_spaces[RAW_FORECAST_KEY] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_datacenters, self._forecast_raw_horizon),
                dtype=np.float32,
            )

        self.global_observation_space = spaces.Dict(new_spaces)

        logger.info(
            "  global_observation_space: %d keys (removed %d compressed%s)",
            len(new_spaces),
            len(COMPRESSED_FORECAST_KEYS) - len(self._kept_compressed_keys),
            f", added '{RAW_FORECAST_KEY}'" if self._emit_raw_forecast else "",
        )

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _convert_global_observation(self, global_obs_java) -> Dict[str, Any]:
        """
        Drop compressed keys per ``forecast_mode`` and (in raw mode) inject the
        per-DC raw forecast trajectory.
        """
        full_obs = super()._convert_global_observation(global_obs_java)

        out: Dict[str, Any] = {}
        for key, value in full_obs.items():
            if key in COMPRESSED_FORECAST_KEYS and key not in self._kept_compressed_keys:
                continue
            out[key] = value

        if self._emit_raw_forecast:
            out[RAW_FORECAST_KEY] = self._build_raw_forecast_obs(global_obs_java)

        return out

    def _build_raw_forecast_obs(self, global_obs_java) -> np.ndarray:
        """
        Build the ``dc_future_raw`` array of shape (num_datacenters, horizon).

        Pulls the raw trajectory from the TimeCAP provider; falls back to a
        neutral 0.5-fill when no forecast has been computed yet (e.g. before
        the first step after reset).
        """
        out = np.full(
            (self.num_datacenters, self._forecast_raw_horizon),
            0.5,
            dtype=np.float32,
        )

        provider = getattr(self, "timecap_provider", None)
        if provider is None:
            return out

        raw_per_dc: Optional[Dict[int, np.ndarray]] = provider.get_raw_forecast_per_dc(
            horizon=self._forecast_raw_horizon,
            normalize=True,
        )
        if raw_per_dc is None:
            return out

        for i, dc_id in enumerate(self.dc_ids):
            arr = raw_per_dc.get(dc_id)
            if arr is None or arr.size == 0:
                continue
            h = min(self._forecast_raw_horizon, int(arr.size))
            out[i, :h] = arr[:h]

        return out
