"""
TimeCAP_GreenPredictor
======================
Replaces the "God's Eye" future green-energy features in HierarchicalMultiDCEnv
with real forecasts produced by the TimeCAP model.

God's Eye (Java) computes four features from ground-truth future CSV rows:
    dc_future_short_mean      – mean(next 3 rows Patv) / maxPower      → [0, 1]
    dc_future_short_trend     – (row+2 − row+0) Patv  / maxPower       → [−1, 1]
    dc_future_long_mean       – mean(next 144 rows Patv) / maxPower     → [0, 1]
    dc_future_long_peak_timing– argmax(next 144 rows) / 143             → [0, 1]

This class reproduces those four numbers from TimeCAP predictions instead.

Usage (one instance per datacenter)
------------------------------------
predictor = TimeCAP_GreenPredictor(
    checkpoint_path = "timecap_prediction/checkpoints/ckpt_best.pth",
    turbine_csv_paths = {1: "/path/to/Turbine_1_2021.csv"},
    # model_config: loaded automatically from  <checkpoint_dir>/model_args.json
    #               or pass explicitly as a dict
)

# In env.reset():
predictor.reset()

# In env.step(), after receiving obs from Java:
predictor.update(current_simulation_step)
pred = predictor.predict()                     # np.ndarray (pred_len,) kW, or None
if pred is not None:
    feats = predictor.compute_god_eye_features(pred)
    # feats["short_mean"], feats["short_trend"],
    # feats["long_mean"],  feats["peak_timing"]

Checkpoint / args convention
------------------------------
The training script (Code/run_turbine_timecap.py) should save:
    torch.save(model.state_dict(), "<dir>/ckpt_best.pth")
    json.dump(vars(args), open("<dir>/model_args.json", "w"))

If model_args.json is absent, pass model_config=<dict> to the constructor.
"""

import json
import logging
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap so we can import TimeCAP from drl-manager/Code/.
# Code/ must come BEFORE src/ — otherwise `from models.TimeCAP import Model`
# resolves to src/models (an unrelated RL-side package) and fails.
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_TIMECAP_CODE_DIR = str(Path(__file__).resolve().parent.parent / "Code")
if _TIMECAP_CODE_DIR not in sys.path:
    sys.path.insert(0, _TIMECAP_CODE_DIR)

from prediction.csv_feature_loader import CSVFeatureLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Default TimeCAP model config matching the turbine fine-tune setup
# (seq_len=96, pred_len=144, enc_in=13, 10-min wind data)
# Override any field via model_config dict passed to the constructor.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_CONFIG: Dict = {
    # Task
    "task_name": "finetune",
    "downstream_task": "forecasting",
    # Sequence lengths
    "seq_len": 96,
    "label_len": 0,
    "pred_len": 144,
    "pretrain_pred_len": 16,
    # Input
    "enc_in": 13,
    "features": "M",
    # Multi-scale blocks
    "depth": 2,
    "patch_len": [96, 24],
    "stride_time": [96, 24],
    "window_size": [3, 3],
    "stride_channel": [1, 1],
    # Transformer
    "d_model": 736,
    "d_ff": 992,
    "e_layers": 2,
    "n_heads": 8,
    "dropout": 0.0,
    "activation": "gelu",
    # Heads
    "use_ar_head": True,
    "use_os_head": True,
    # OS / AR fusion sigmoid params (from TimeCAP paper)
    "alpha": 1.0,
    "beta": 0.3326362081926146,
    # Misc
    "output_attention": False,
    "flash_attention": False,
    "covariate": False,
    "scope": 0,
}

# v1 — 13 raw feature columns matching the SDWPF split CSVs
_V1_FEATURE_COLUMNS: List[str] = [
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
    "Pab1", "Prtv", "T2m",
    "Sp", "RelH", "Wspd_w", "Wdir_w",
    "Patv",
]

# v2 — 23 engineered feature columns matching turbines_all134_2021_v2.csv
# (column order MUST match drl-manager/timecap_prediction/engineer_features.py)
_V2_FEATURE_COLUMNS: List[str] = [
    "Wspd", "Wspd_cubed",
    "Etmp", "Itmp", "Pab1", "Prtv", "T2m", "Sp", "RelH", "Wspd_w",
    "Wdir_sin", "Wdir_cos",
    "Ndir_sin", "Ndir_cos",
    "Wdir_w_sin", "Wdir_w_cos",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "Patv",
]

# Backwards compat alias
_DEFAULT_FEATURE_COLUMNS: List[str] = _V1_FEATURE_COLUMNS

# Candidate timestamp column names found in raw turbine CSVs
_TIME_COL_CANDIDATES = ("date", "Tmstamp", "timestamp", "time")


def _derive_v2_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a raw SDWPF-style turbine DataFrame (with the v1 13 columns plus a
    timestamp column) and return a DataFrame containing the 23 v2 columns in
    the canonical order. Mirrors drl-manager/timecap_prediction/engineer_features.py.
    """
    df = raw_df.copy()

    time_col = next((c for c in _TIME_COL_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(
            f"v2 feature derivation needs a timestamp column "
            f"(one of {_TIME_COL_CANDIDATES}); found columns={list(df.columns)}"
        )

    # Wspd^3 — physics prior (P ~ V^3)
    if "Wspd" not in df.columns:
        raise ValueError("v2 derivation requires 'Wspd' column.")
    df["Wspd_cubed"] = (df["Wspd"].astype(np.float64) ** 3).astype(np.float32)

    # sin/cos for direction columns (degrees)
    for c in ("Wdir", "Ndir", "Wdir_w"):
        if c not in df.columns:
            raise ValueError(f"v2 derivation requires angle column '{c}'.")
        rad = np.deg2rad(df[c].astype(np.float64).values)
        df[f"{c}_sin"] = np.sin(rad).astype(np.float32)
        df[f"{c}_cos"] = np.cos(rad).astype(np.float32)

    # Cyclical time features (must use the SAME normalisation as engineer_features.py:
    #   hour ÷ 24, doy ÷ 366, dow ÷ 7)
    dt = pd.to_datetime(df[time_col])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    doy = dt.dt.dayofyear.astype(np.float64)
    dow = dt.dt.dayofweek.astype(np.float64)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366.0).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366.0).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

    missing = [c for c in _V2_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"v2 derivation produced incomplete frame; missing {missing}")

    return df[_V2_FEATURE_COLUMNS].copy()


def _dict_to_namespace(d: Dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for attribute-style access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


class TimeCAP_GreenPredictor:
    """
    Per-datacenter green energy predictor backed by a fine-tuned TimeCAP model.

    One instance is created for each datacenter. Call update() every step to
    push the latest features into the rolling buffer, then call predict() to
    get a (pred_len,) array of forecasted Patv values (kW). Pass that to
    compute_god_eye_features() to obtain the four observation features.
    """

    # Default 13-feature columns in the SDWPF turbine CSVs (v1)
    DEFAULT_FEATURE_COLUMNS: List[str] = _V1_FEATURE_COLUMNS
    V1_FEATURE_COLUMNS: List[str] = _V1_FEATURE_COLUMNS
    V2_FEATURE_COLUMNS: List[str] = _V2_FEATURE_COLUMNS

    def __init__(
        self,
        checkpoint_path: str,
        turbine_csv_paths: Dict[int, str],
        model_config: Optional[Dict] = None,
        seq_len: int = 96,
        pred_len: int = 144,
        short_term_steps: int = 3,
        long_term_steps: int = 144,
        device: str = "cpu",
        csv_start_offset: int = 12,
        feature_columns: Optional[List[str]] = None,
        feature_set: str = "v1",
        auto_derive_features: bool = True,
    ):
        """
        Parameters
        ----------
        checkpoint_path:
            Path to the fine-tuned TimeCAP state-dict (.pth).
            A sibling file ``model_args.json`` is loaded automatically when
            model_config is not provided.
        turbine_csv_paths:
            Mapping turbine_id → absolute path to its 13-feature split CSV.
            e.g. {1: ".../windProduction/split/Turbine_1_2021.csv"}
        model_config:
            Override dict for TimeCAP hyper-parameters. Missing keys fall back
            to _DEFAULT_MODEL_CONFIG. If None, tries to load model_args.json
            from the same directory as the checkpoint.
        seq_len:
            History window length fed to TimeCAP (must satisfy seq_len % patch_len == 0).
        pred_len:
            Forecast horizon in time-steps (1 step = 10 min). 144 → 24 h.
        short_term_steps:
            Number of forecast steps used for short-term God's Eye features (default 3 → 30 min).
        long_term_steps:
            Number of forecast steps used for long-term God's Eye features (default 144 → 24 h).
        device:
            Torch device string ("cpu" or "cuda").
        csv_start_offset:
            Row offset applied by Java (default 12). simulation_step=0 → CSV row 12.
        feature_columns:
            Explicit column names to feed the model. If provided, takes precedence
            over feature_set. Otherwise resolved from feature_set.
        feature_set:
            Either "v1" (13 raw SDWPF features — matches the baseline 4358062
            checkpoint) or "v2" (23 engineered features incl. Wspd^3, sin/cos
            angle pairs, and hour/doy/dow cyclic encodings — matches the
            Phase 1 retrain).
        auto_derive_features:
            When feature_set="v2" and the raw turbine CSVs only contain the v1
            schema, set True to compute the engineered columns in memory at
            construction time (mirrors engineer_features.py). The CSV must
            contain a timestamp column named one of: 'date', 'Tmstamp',
            'timestamp', 'time'.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.short_term_steps = min(short_term_steps, pred_len)
        self.long_term_steps = min(long_term_steps, pred_len)
        self.device = torch.device(device)
        self.turbine_ids = list(turbine_csv_paths.keys())

        # Resolve feature column list: explicit > feature_set
        if feature_set not in ("v1", "v2"):
            raise ValueError(f"feature_set must be 'v1' or 'v2', got {feature_set!r}")
        self.feature_set = feature_set
        self.auto_derive_features = auto_derive_features

        if feature_columns is not None:
            self.feature_columns = feature_columns
        elif feature_set == "v2":
            self.feature_columns = _V2_FEATURE_COLUMNS
        else:
            self.feature_columns = _V1_FEATURE_COLUMNS
        self.num_features = len(self.feature_columns)
        # Patv must be present; it is used as the prediction target
        if "Patv" not in self.feature_columns:
            raise ValueError("'Patv' must be in feature_columns.")
        self.patv_idx = self.feature_columns.index("Patv")

        # Build model config (merge defaults ← file ← constructor arg)
        resolved_config = self._resolve_model_config(checkpoint_path, model_config)
        # Ensure enc_in matches actual feature count
        resolved_config["enc_in"] = self.num_features
        resolved_config["seq_len"] = seq_len
        resolved_config["pred_len"] = pred_len

        # Build TimeCAP and load weights
        self.model = self._build_model(resolved_config, checkpoint_path)
        self.model_args = _dict_to_namespace(resolved_config)

        # CSV feature loader (reuses existing infrastructure).
        # In v2+auto_derive mode we first load with the raw v1 column list (so
        # angle/raw columns survive), then replace the per-turbine DataFrames
        # with the engineered v2 frames computed from the original CSVs.
        needs_derive = (feature_set == "v2") and auto_derive_features
        loader_columns = (
            list(_V1_FEATURE_COLUMNS) if needs_derive else self.feature_columns
        )
        self.feature_loader = CSVFeatureLoader(
            turbine_csv_paths=turbine_csv_paths,
            csv_timestep_seconds=600,
            feature_columns=loader_columns,
            csv_start_offset=csv_start_offset,
        )

        if needs_derive:
            self._derive_v2_into_loader(turbine_csv_paths)
            # Keep the loader's metadata in sync with the actual frames it now holds
            self.feature_loader.feature_columns = list(self.feature_columns)

        # Per-turbine max Patv (kW) – mirrors Java's maxPowerKw computation
        self.max_power_kw: Dict[int, float] = {}
        for tid in self.turbine_ids:
            df = self.feature_loader.turbine_data.get(tid)
            if df is not None and "Patv" in df.columns:
                max_val = float(df["Patv"].max())
                self.max_power_kw[tid] = max_val if max_val > 0 else 1.0
            else:
                self.max_power_kw[tid] = 1.0
                logger.warning(f"Turbine {tid}: could not compute max Patv, defaulting to 1.0 kW")

        # Per-turbine rolling history buffer: deque of (num_features,) arrays
        self._history: Dict[int, deque] = {
            tid: deque(maxlen=seq_len) for tid in self.turbine_ids
        }

        logger.info(
            f"TimeCAP_GreenPredictor ready: "
            f"turbines={self.turbine_ids}, seq_len={seq_len}, pred_len={pred_len}, "
            f"device={device}, max_power_kw={self.max_power_kw}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all history buffers. Call once inside env.reset()."""
        for tid in self.turbine_ids:
            self._history[tid].clear()

    def update(self, simulation_step: int) -> None:
        """
        Read the 13-feature row at *simulation_step* from the turbine CSVs
        and push it into each turbine's history buffer.

        Call once per environment step, right after receiving the Java observation.

        Parameters
        ----------
        simulation_step:
            The env's current_step counter (0-indexed).
        """
        for tid in self.turbine_ids:
            row = self.feature_loader.get_feature_at_time(tid, float(simulation_step))
            if row is not None:
                self._history[tid].append(row.astype(np.float32))
            else:
                # Out-of-range step: push a zero row so the buffer keeps moving
                self._history[tid].append(np.zeros(self.num_features, dtype=np.float32))
                logger.debug(f"Turbine {tid} step {simulation_step}: no CSV row, pushing zeros")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self) -> Optional[np.ndarray]:
        """
        Run TimeCAP one-shot head to forecast Patv for the next *pred_len* steps.

        The history buffer is zero-padded at the front during the cold-start
        period (before seq_len steps have been observed), so predictions are
        available from step 0.

        Returns
        -------
        np.ndarray of shape (pred_len,), Patv in kW, averaged across turbines.
        Returns None only if the model raises an unexpected exception.
        """
        turbine_preds: List[np.ndarray] = []

        for tid in self.turbine_ids:
            hist = list(self._history[tid])

            # Zero-pad at the front if we haven't seen seq_len steps yet
            if len(hist) < self.seq_len:
                pad_rows = self.seq_len - len(hist)
                hist = [np.zeros(self.num_features, dtype=np.float32)] * pad_rows + hist

            # Stack → (seq_len, num_features) → batch → (1, seq_len, num_features)
            x = np.stack(hist, axis=0)
            x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(self.device)

            try:
                with torch.no_grad():
                    # Returns (dec_out_AR, dec_out_OS, attns)
                    # dec_out_OS shape: (1, pred_len, enc_in)  — one-shot head
                    _, dec_out_OS, _ = self.model(x_tensor, activate_os_head=True)

                if dec_out_OS is None:
                    logger.warning(
                        f"Turbine {tid}: OS head returned None. "
                        "Check that task_name='finetune' in model config."
                    )
                    turbine_preds.append(np.zeros(self.pred_len, dtype=np.float32))
                    continue

                # Extract Patv channel and clip negatives (power ≥ 0)
                pred_patv = dec_out_OS[0, :, self.patv_idx].cpu().numpy()  # (pred_len,)
                pred_patv = np.clip(pred_patv, 0.0, None)
                turbine_preds.append(pred_patv.astype(np.float32))

            except Exception as exc:
                logger.error(f"Turbine {tid}: TimeCAP inference failed: {exc}", exc_info=True)
                return None

        if not turbine_preds:
            return None

        # Simple average across turbines
        # (Could weight by max_power_kw to match Java's weighted aggregation)
        return np.mean(turbine_preds, axis=0)  # (pred_len,)

    # ------------------------------------------------------------------
    # Feature computation  (mirrors Java GreenEnergyProvider)
    # ------------------------------------------------------------------

    def compute_god_eye_features(
        self,
        predicted_patv_kw: np.ndarray,
        max_power_kw: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute the four God's Eye observation features from a Patv forecast.

        Semantics match Java's GreenEnergyProvider.computeAggregatedFutureTrendFeatures():
            short_mean   = mean(pred[:short_term_steps]) / maxPower   → clipped [0, 1]
            short_trend  = (pred[short-1] − pred[0])    / maxPower   → clipped [−1, 1]
            long_mean    = mean(pred[:long_term_steps])  / maxPower   → clipped [0, 1]
            peak_timing  = argmax(pred[:long_term_steps]) / (n − 1)   → [0, 1]

        Parameters
        ----------
        predicted_patv_kw:
            Forecast returned by predict(), shape (pred_len,), unit kW.
        max_power_kw:
            Normalisation constant. If None, the average max across all turbines
            in this DC is used (same approach as Java's weighted aggregation).

        Returns
        -------
        dict with keys: "short_mean", "short_trend", "long_mean", "peak_timing"
        """
        if max_power_kw is None:
            vals = list(self.max_power_kw.values())
            max_power_kw = float(np.mean(vals)) if vals else 1.0
        if max_power_kw <= 0.0:
            max_power_kw = 1.0

        pred = predicted_patv_kw
        st = min(self.short_term_steps, len(pred))
        lt = min(self.long_term_steps, len(pred))

        # Short-term mean
        short_mean = float(np.mean(pred[:st])) / max_power_kw
        short_mean = float(np.clip(short_mean, 0.0, 1.0))

        # Short-term trend: (last − first) of the short window, normalised
        if st >= 2:
            short_trend = float(pred[st - 1] - pred[0]) / max_power_kw
        else:
            short_trend = 0.0
        short_trend = float(np.clip(short_trend, -1.0, 1.0))

        # Long-term mean
        long_mean = float(np.mean(pred[:lt])) / max_power_kw
        long_mean = float(np.clip(long_mean, 0.0, 1.0))

        # Peak timing: normalised position of the maximum within the long window
        peak_idx = int(np.argmax(pred[:lt]))
        peak_timing = float(peak_idx) / max(lt - 1, 1)
        peak_timing = float(np.clip(peak_timing, 0.0, 1.0))

        return {
            "short_mean": short_mean,
            "short_trend": short_trend,
            "long_mean": long_mean,
            "peak_timing": peak_timing,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _derive_v2_into_loader(self, turbine_csv_paths: Dict[int, str]) -> None:
        """
        Re-read each raw turbine CSV, run _derive_v2_features on it, and replace
        the DataFrame stored inside self.feature_loader.turbine_data. After this
        call get_feature_at_time(tid, step) returns a 23-element v2 row.
        """
        for tid, path in turbine_csv_paths.items():
            try:
                raw_df = pd.read_csv(path)
            except Exception as exc:
                logger.error(f"Turbine {tid}: failed to re-read raw CSV {path}: {exc}")
                continue

            try:
                v2_df = _derive_v2_features(raw_df)
            except Exception as exc:
                logger.error(
                    f"Turbine {tid}: v2 feature derivation failed: {exc}. "
                    "Falling back to v1 frame zero-padded to v2 width."
                )
                v1_df = self.feature_loader.turbine_data.get(tid)
                if v1_df is None:
                    continue
                v2_df = pd.DataFrame(
                    np.zeros((len(v1_df), len(_V2_FEATURE_COLUMNS)), dtype=np.float32),
                    columns=_V2_FEATURE_COLUMNS,
                )
                # Carry over Patv at minimum so the AR head has a target signal
                if "Patv" in v1_df.columns:
                    v2_df["Patv"] = v1_df["Patv"].values

            v2_df.fillna(0.0, inplace=True)
            self.feature_loader.turbine_data[tid] = v2_df
            logger.info(
                f"Turbine {tid}: v2 feature derivation complete "
                f"({len(v2_df)} rows, {v2_df.shape[1]} columns)"
            )

    @staticmethod
    def _resolve_model_config(checkpoint_path: str, override: Optional[Dict]) -> Dict:
        """
        Merge model config from three sources (lowest → highest priority):
          1. _DEFAULT_MODEL_CONFIG
          2. model_args.json next to the checkpoint (if it exists)
          3. override dict passed by the caller
        """
        config = dict(_DEFAULT_MODEL_CONFIG)

        # Try to load model_args.json from checkpoint directory
        ckpt_path = Path(checkpoint_path)
        json_path = ckpt_path.parent / "model_args.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    file_cfg = json.load(f)
                config.update(file_cfg)
                logger.info(f"Loaded model config from {json_path}")
            except Exception as exc:
                logger.warning(f"Could not load {json_path}: {exc}. Using defaults.")
        else:
            logger.info(
                f"model_args.json not found at {json_path}. "
                "Using _DEFAULT_MODEL_CONFIG. Pass model_config= to override."
            )

        if override:
            config.update(override)

        return config

    @staticmethod
    def _build_model(config: Dict, checkpoint_path: str):
        """Instantiate TimeCAP.Model and load the fine-tuned state dict."""
        try:
            from models.TimeCAP import Model as TimeCAP_Model  # from Code/
        except ImportError as exc:
            raise ImportError(
                "Could not import TimeCAP Model. "
                f"Ensure drl-manager/Code is in sys.path. Original error: {exc}"
            )

        args = _dict_to_namespace(config)
        model = TimeCAP_Model(args).float()

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TimeCAP checkpoint not found: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")

        # Support both plain state-dicts and dicts with 'model_state_dict' key
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        device = torch.device("cpu")
        model = model.to(device).eval()
        logger.info(f"TimeCAP model loaded from {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params)")
        return model
