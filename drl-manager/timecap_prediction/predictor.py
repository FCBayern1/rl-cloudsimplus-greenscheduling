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
from typing import Dict, List, Optional, Union

import numpy as np
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

# 13 raw feature columns matching the SDWPF split CSVs
_DEFAULT_FEATURE_COLUMNS: List[str] = [
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
    "Pab1", "Prtv", "T2m",
    "Sp", "RelH", "Wspd_w", "Wdir_w",
    "Patv",
]


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

    # Default 13-feature columns in the SDWPF turbine CSVs
    DEFAULT_FEATURE_COLUMNS: List[str] = _DEFAULT_FEATURE_COLUMNS

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
        csv_start_offset: Union[int, Dict[int, int]] = 0,
        feature_columns: Optional[List[str]] = None,
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
            Row offset added to sim_step when reading the CSV. Two forms:
              * int — single offset shared by all turbines. simulation_step=0
                reads CSV row ``csv_start_offset``. Default 0.
              * Dict[turbine_id, int] — per-turbine offset. Use this in
                multi-DC setups where each DC has a different
                time_zone_offset_rows; the value should be
                ``tz_offset_rows[dc_of_turbine] + simulation_warmup_rows``.
        feature_columns:
            Explicit column names to feed the model. Defaults to the 13-feature
            SDWPF schema.
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.short_term_steps = min(short_term_steps, pred_len)
        self.long_term_steps = min(long_term_steps, pred_len)
        self.device = torch.device(device)
        self.turbine_ids = list(turbine_csv_paths.keys())

        self.feature_columns = (
            list(feature_columns) if feature_columns is not None
            else list(_DEFAULT_FEATURE_COLUMNS)
        )
        self.num_features = len(self.feature_columns)
        if "Patv" not in self.feature_columns:
            raise ValueError("'Patv' must be in feature_columns.")
        self.patv_idx = self.feature_columns.index("Patv")

        # Build model config (merge defaults ← file ← constructor arg)
        resolved_config = self._resolve_model_config(checkpoint_path, model_config)
        resolved_config["enc_in"] = self.num_features
        resolved_config["seq_len"] = seq_len
        resolved_config["pred_len"] = pred_len

        # Build TimeCAP and load weights, then move to the requested device.
        self.model = self._build_model(resolved_config, checkpoint_path)
        self.model = self.model.to(self.device)
        self.model_args = _dict_to_namespace(resolved_config)

        self.feature_loader = CSVFeatureLoader(
            turbine_csv_paths=turbine_csv_paths,
            csv_timestep_seconds=600,
            feature_columns=self.feature_columns,
            csv_start_offset=csv_start_offset,
        )

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

    def _forward_one_turbine(self, turbine_id: int) -> Optional[np.ndarray]:
        """
        Run TimeCAP's one-shot head for a single turbine using its current
        history buffer. Zero-pads at the front when the buffer is not yet full.

        Returns
        -------
        np.ndarray of shape (pred_len,), Patv in kW, clipped to ≥ 0.
        None on inference failure.
        """
        hist = list(self._history[turbine_id])

        # Zero-pad at the front if we haven't seen seq_len steps yet
        if len(hist) < self.seq_len:
            pad_rows = self.seq_len - len(hist)
            hist = [np.zeros(self.num_features, dtype=np.float32)] * pad_rows + hist

        x = np.stack(hist, axis=0)                                         # (seq_len, num_features)
        x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(self.device)  # (1, seq_len, num_features)

        try:
            with torch.no_grad():
                # Returns (dec_out_AR, dec_out_OS, attns); OS shape: (1, pred_len, enc_in)
                _, dec_out_OS, _ = self.model(x_tensor, activate_os_head=True)
        except Exception as exc:
            logger.error(f"Turbine {turbine_id}: TimeCAP inference failed: {exc}", exc_info=True)
            return None

        if dec_out_OS is None:
            logger.warning(
                f"Turbine {turbine_id}: OS head returned None. "
                "Check that task_name='finetune' in model config."
            )
            return np.zeros(self.pred_len, dtype=np.float32)

        pred_patv = dec_out_OS[0, :, self.patv_idx].cpu().numpy()  # (pred_len,)
        return np.clip(pred_patv, 0.0, None).astype(np.float32)

    def predict(self) -> Optional[np.ndarray]:
        """
        Run TimeCAP one-shot head to forecast Patv for the next *pred_len* steps,
        averaged (simple arithmetic mean) across all turbines.

        For God's-Eye-style DC aggregation that needs maxPower-weighted means,
        use :meth:`predict_per_turbine` and aggregate externally instead.

        Returns
        -------
        np.ndarray of shape (pred_len,), Patv in kW.
        Returns None on inference failure.
        """
        preds: List[np.ndarray] = []
        for tid in self.turbine_ids:
            p = self._forward_one_turbine(tid)
            if p is None:
                return None
            preds.append(p)
        if not preds:
            return None
        return np.mean(preds, axis=0)

    def predict_per_turbine(self) -> Optional[Dict[int, np.ndarray]]:
        """
        Same as :meth:`predict` but returns a dict mapping turbine_id → (pred_len,)
        forecast — without the cross-turbine averaging step. Useful when an
        external aggregator (e.g. Java-style maxPower-weighted mean per DC)
        needs per-turbine values.

        Returns
        -------
        dict {turbine_id: np.ndarray of shape (pred_len,)}, or None on failure.
        """
        out: Dict[int, np.ndarray] = {}
        for tid in self.turbine_ids:
            p = self._forward_one_turbine(tid)
            if p is None:
                return None
            out[tid] = p
        return out

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

        # Loaded onto CPU here; the caller (__init__) moves it to self.device
        # afterwards.  Don't hard-code CPU here or downstream forward passes
        # will device-mismatch when self.device == cuda.
        model = model.eval()
        logger.info(f"TimeCAP model loaded from {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params)")
        return model
