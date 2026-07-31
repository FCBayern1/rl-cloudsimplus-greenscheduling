"""
Deployment-time forecast-trust sentinel (EU-CRD Phase-D).

Reuses the Q-head ensemble that EU-CRD trains for credit assignment as an
epistemic-uncertainty SENSOR at inference time: high ensemble disagreement on
the current trunk features means the state (which embeds the forecast
features) is off the training manifold — i.e. the forecast can no longer be
trusted. In "gate" mode the sentinel then suppresses the global DEFER logit,
so the policy degrades to reactive (run-now) behaviour with its learned
spatial routing intact, instead of waiting on a false green promise.

This is inference-only: no retraining, no change to the training pipeline.
Vanilla (non-EU-CRD) checkpoints carry no Q-ensemble and cannot use it — the
sentinel raises a clear error rather than producing garbage.

Env-var protocol (mirrors FORECAST_PERTURB_MODE):
    TRUST_GATE_MODE    ""/unset = off · "log" = record sigma² only ·
                       "gate" = log + suppress DEFER when sigma² > threshold
    TRUST_GATE_THRESH  float, required in gate mode (calibrate from a clean
                       log-mode run, e.g. its p95)
    TRUST_GATE_LOG     CSV path for per-decision records (required in log
                       mode, optional in gate mode)
    TRUST_GATE_SIGNAL  which sigma² column drives the gate: "defer" (default,
                       variance of the DEFER column) or "all" (mean over all
                       action columns)

Weight loading: the env-runner module restored by Algorithm.from_checkpoint
may be inference-only, in which case its `q_heads` attribute exists (created
in setup()) but holds RANDOM init — silently meaningless. We therefore ALWAYS
load the trained ensemble weights from the learner-side module_state file:
    <ckpt>/learner_group/learner/rl_module/<policy_id>/module_state.pt
(pickled OrderedDict, not a torch.save archive).
"""

import logging
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch

from src.models.rlmodule_gtrxl_ensemble import EnsembleQHeads

logger = logging.getLogger(__name__)

Q_PREFIX = "q_heads."


def _learner_module_state_path(checkpoint_path: str, policy_id: str) -> Path:
    return (
        Path(checkpoint_path)
        / "learner_group"
        / "learner"
        / "rl_module"
        / policy_id
        / "module_state.pt"
    )


def load_q_heads_from_checkpoint(
    checkpoint_path: str,
    policy_id: str = "global_policy",
    prior_lambda: Optional[float] = None,
) -> EnsembleQHeads:
    """
    Rebuild the trained EnsembleQHeads from the learner-side module state.

    Shapes are inferred from the state dict itself so this works for any
    K / hidden_dim / action_dim without needing the original model_config.
    Raises ValueError if the checkpoint has no Q-ensemble (vanilla policy).
    """
    state_path = _learner_module_state_path(checkpoint_path, policy_id)
    if not state_path.exists():
        raise FileNotFoundError(f"learner module state not found: {state_path}")
    with open(state_path, "rb") as f:
        full_state = pickle.load(f)

    q_state = OrderedDict(
        (k[len(Q_PREFIX):], torch.as_tensor(v))
        for k, v in full_state.items()
        if k.startswith(Q_PREFIX)
    )
    if not q_state:
        raise ValueError(
            f"checkpoint {checkpoint_path} has no '{Q_PREFIX}*' keys — this is "
            f"a vanilla (non-EU-CRD) policy; the trust sentinel needs the "
            f"EU-CRD Q-ensemble."
        )

    head_ids = {
        int(m.group(1))
        for k in q_state
        if (m := re.match(r"q_heads\.(\d+)\.", k))
    }
    K = max(head_ids) + 1
    w_in = q_state["q_heads.0.0.weight"]     # (hidden_dim, d_model)
    w_out = q_state["q_heads.0.2.weight"]    # (action_dim, hidden_dim)
    hidden_dim, d_model = w_in.shape
    action_dim = w_out.shape[0]

    heads = EnsembleQHeads(
        d_model=d_model,
        action_dim=action_dim,
        K=K,
        prior_lambda=3.0 if prior_lambda is None else float(prior_lambda),
        hidden_dim=hidden_dim,
    )
    heads.load_state_dict(q_state, strict=True)
    heads.eval()
    logger.info(
        f"[TrustSentinel] loaded Q-ensemble from {state_path}: "
        f"K={K}, d_model={d_model}, hidden={hidden_dim}, action_dim={action_dim}"
    )
    return heads


class TrustSentinel:
    """
    Per-decision epistemic-uncertainty monitor + optional DEFER gate.

    Usage (inside RLlibNewAPIGlobalScheduler.schedule):
        sigma = sentinel.measure(module._captured_features)   # after forward
        logits = sentinel.maybe_gate(logits)                  # before decode
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_slots: int,
        policy_id: str = "global_policy",
        mode: str = "log",
        threshold: Optional[float] = None,
        log_path: Optional[str] = None,
        signal: str = "defer",
        prior_lambda: Optional[float] = None,
    ):
        if mode not in ("log", "gate"):
            raise ValueError(f"TRUST_GATE_MODE must be 'log' or 'gate', got {mode!r}")
        if mode == "gate" and threshold is None:
            raise ValueError("gate mode requires TRUST_GATE_THRESH")
        if mode == "log" and not log_path:
            raise ValueError("log mode requires TRUST_GATE_LOG (output CSV path)")
        if signal not in ("defer", "all"):
            raise ValueError(f"TRUST_GATE_SIGNAL must be 'defer' or 'all', got {signal!r}")

        self.mode = mode
        self.threshold = threshold
        self.signal = signal
        self.num_slots = int(num_slots)
        self.q_heads = load_q_heads_from_checkpoint(
            checkpoint_path, policy_id=policy_id, prior_lambda=prior_lambda
        )
        if self.q_heads.action_dim % self.num_slots != 0:
            raise ValueError(
                f"Q-ensemble action_dim {self.q_heads.action_dim} not divisible "
                f"by num_slots {self.num_slots}"
            )
        self.num_choices = self.q_heads.action_dim // self.num_slots

        self._log_file = None
        if log_path:
            self._log_file = open(log_path, "a", buffering=1)
            if self._log_file.tell() == 0:
                self._log_file.write("step,sigma2_all,sigma2_defer,gated\n")

        self._step = 0
        self._n_gated = 0
        self._last_sigma2_all = float("nan")
        self._last_sigma2_defer = float("nan")

    @classmethod
    def from_env(cls, checkpoint_path: str, num_slots: int,
                 policy_id: str = "global_policy") -> Optional["TrustSentinel"]:
        """Build from TRUST_GATE_* env vars; returns None when the gate is off."""
        mode = os.environ.get("TRUST_GATE_MODE", "").strip().lower()
        if not mode:
            return None
        thresh_s = os.environ.get("TRUST_GATE_THRESH", "").strip()
        return cls(
            checkpoint_path=checkpoint_path,
            num_slots=num_slots,
            policy_id=policy_id,
            mode=mode,
            threshold=float(thresh_s) if thresh_s else None,
            log_path=os.environ.get("TRUST_GATE_LOG", "").strip() or None,
            signal=os.environ.get("TRUST_GATE_SIGNAL", "defer").strip().lower(),
        )

    def measure(self, trunk_features: torch.Tensor) -> float:
        """
        Compute epistemic uncertainty on the CURRENT decision state.

        Args:
            trunk_features: (B, T, d_model) captured GTrXL trunk output; the
                last timestep is the current state.
        Returns:
            the gating sigma² (per TRUST_GATE_SIGNAL) as a float.
        """
        if trunk_features is None:
            raise RuntimeError(
                "trunk features not captured — is this an EU-CRD ensemble "
                "module with the feature-capture hook installed?"
            )
        with torch.no_grad():
            state = trunk_features[:, -1, :].detach().float().cpu()  # (B, d)
            q = self.q_heads(state)                                  # (B, K, A)
            q = q.view(-1, self.q_heads.K, self.num_slots, self.num_choices)
            var = q.var(dim=1, unbiased=False)                       # (B, slots, choices)
            self._last_sigma2_all = float(var.mean())
            self._last_sigma2_defer = float(var[..., -1].mean())
        return (
            self._last_sigma2_defer if self.signal == "defer" else self._last_sigma2_all
        )

    def maybe_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Suppress the DEFER column (last choice index) when the last measured
        sigma² exceeds the threshold. `logits` shape: (B, num_slots, num_choices).
        Also writes the per-decision log line (both modes).
        """
        sigma = (
            self._last_sigma2_defer if self.signal == "defer" else self._last_sigma2_all
        )
        gated = (
            self.mode == "gate"
            and self.threshold is not None
            and sigma > self.threshold
        )
        if gated:
            logits = logits.clone()
            logits[..., -1] = torch.finfo(logits.dtype).min
            self._n_gated += 1
        if self._log_file is not None:
            self._log_file.write(
                f"{self._step},{self._last_sigma2_all:.6g},"
                f"{self._last_sigma2_defer:.6g},{int(gated)}\n"
            )
        self._step += 1
        return logits

    def summary(self) -> str:
        return (
            f"TrustSentinel[{self.mode}/{self.signal}]: steps={self._step}, "
            f"gated={self._n_gated} "
            f"({(100.0 * self._n_gated / max(1, self._step)):.1f}%)"
        )

    def close(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class ForecastResidualMonitor:
    """
    Deployment-time forecast VERIFICATION sentinel (Phase-1b).

    Phase-1 finding: Q-ensemble disagreement does NOT separate corrupted from
    clean forecasts — every anti-forecast frame is pointwise plausible; what's
    wrong is its CORRELATION with reality, which pointwise OOD detection is
    blind to. So this monitor audits the forecast against what actually
    happens: rolling per-DC Pearson correlation between the short-horizon
    green forecast (dc_future_short_mean, normalized) and the realized green
    power (dc_current_green_power_w). Clean → strongly positive; anti → sign
    flips negative; blend (neutralized, zero-variance) → 0 by convention (a
    forecast that never moves carries no information). Gate: suppress DEFER
    when correlation < threshold (note: BELOW, opposite of the qvar sentinel).

    Needs no Q-ensemble — works on vanilla checkpoints too (report this
    honestly: it is an architecture-agnostic trust manager).

    Env protocol (with TRUST_GATE_SOURCE=resid):
        TRUST_GATE_MODE    "log" | "gate"
        TRUST_GATE_THRESH  gate when corr < this (default 0.2 in gate mode)
        TRUST_GATE_LOG     CSV path (required in log mode)
        TRUST_GATE_WINDOW  rolling window in decisions (default 600)
        TRUST_GATE_WARMUP  min samples before gating (default 120; trusts by
                           default during warmup)
    """

    def __init__(
        self,
        num_slots: int,
        mode: str = "log",
        threshold: Optional[float] = None,
        log_path: Optional[str] = None,
        window: int = 600,
        warmup: int = 120,
        repair_thresh: float = 0.5,
    ):
        if mode not in ("log", "gate", "repair", "soft", "maskdc", "persist"):
            raise ValueError(
                "TRUST_GATE_MODE must be one of "
                f"log/gate/repair/soft/maskdc/persist, got {mode!r}"
            )
        if mode == "log" and not log_path:
            raise ValueError("log mode requires TRUST_GATE_LOG (output CSV path)")
        self.mode = mode
        needs_thresh = mode in ("gate", "soft", "maskdc", "persist")
        self.threshold = 0.2 if (needs_thresh and threshold is None) else threshold
        # soft mode: graded DEFER penalty instead of a hard -inf ban.
        # penalty = soft_lambda * severity, severity = clip(thresh - corr, 0, 2)/2.
        self.soft_lambda = float(os.environ.get("TRUST_GATE_SOFT_LAMBDA", "6.0"))
        self.num_slots = int(num_slots)
        self.window = int(window)
        self.warmup = max(2, int(warmup))
        # repair mode: a STRONGLY NEGATIVE per-DC correlation means the forecast
        # is a consistent liar — an information-PRESERVING corruption (unlike
        # blend, which destroys information and is unrepairable). Inverting the
        # lying DC's forecast features restores their meaning, so the policy can
        # keep its full temporal lever instead of falling back to reactive.
        self.repair_thresh = float(repair_thresh)

        self._fore = None   # (window, num_dc) ring buffers, lazily sized
        self._real = None
        self._n = 0
        self._last_corr = float("nan")
        self._last_corrs = None   # per-DC rolling corr (NaN where undefined)
        self._step = 0
        self._n_gated = 0
        self._n_repaired = 0

        self._log_file = None
        if log_path:
            self._log_file = open(log_path, "a", buffering=1)
            if self._log_file.tell() == 0:
                self._log_file.write("step,corr,n_samples,gated\n")

    @classmethod
    def from_env(cls, num_slots: int) -> "ForecastResidualMonitor":
        mode = os.environ.get("TRUST_GATE_MODE", "").strip().lower()
        thresh_s = os.environ.get("TRUST_GATE_THRESH", "").strip()
        return cls(
            num_slots=num_slots,
            mode=mode,
            threshold=float(thresh_s) if thresh_s else None,
            log_path=os.environ.get("TRUST_GATE_LOG", "").strip() or None,
            window=int(os.environ.get("TRUST_GATE_WINDOW", "600")),
            warmup=int(os.environ.get("TRUST_GATE_WARMUP", "120")),
            repair_thresh=float(os.environ.get("TRUST_GATE_REPAIR_THRESH", "0.5")),
        )

    def measure_obs(self, global_obs) -> float:
        """Feed this decision's (forecast, realized-green) pair; returns the
        rolling mean correlation over DCs whose realized green actually varies."""
        import numpy as np

        f = np.asarray(global_obs.get("dc_future_short_mean", []), dtype=np.float64).ravel()
        g = np.asarray(global_obs.get("dc_current_green_power_w", []), dtype=np.float64).ravel()
        if f.size == 0 or g.size == 0 or f.size != g.size:
            self._last_corr = float("nan")
            return self._last_corr
        if self._fore is None:
            self._fore = np.zeros((self.window, f.size))
            self._real = np.zeros((self.window, f.size))
        i = self._n % self.window
        self._fore[i] = f
        self._real[i] = g
        self._n += 1

        m = min(self._n, self.window)
        if m >= self.warmup:
            F, G = self._fore[:m], self._real[:m]
            fs = F.std(axis=0)
            gs = G.std(axis=0)
            live = gs > 1e-9          # DCs with varying green (green-enabled)
            per_dc = np.full(f.size, np.nan)
            corrs = []
            for d in range(f.size):
                if not live[d]:
                    continue
                if fs[d] <= 1e-9:
                    per_dc[d] = 0.0    # frozen forecast = zero information
                    corrs.append(0.0)
                    continue
                fc = F[:, d] - F[:, d].mean()
                gc = G[:, d] - G[:, d].mean()
                per_dc[d] = float((fc * gc).mean() / (fs[d] * gs[d]))
                corrs.append(per_dc[d])
            self._last_corrs = per_dc
            self._last_corr = float(sum(corrs) / len(corrs)) if corrs else float("nan")
        else:
            self._last_corrs = None
            self._last_corr = float("nan")   # warmup: undecided, never gates/repairs
        return self._last_corr

    def repair(self, global_obs):
        """
        In repair mode: for each DC whose rolling correlation is strongly
        NEGATIVE (< -repair_thresh), invert that DC's forecast features so
        their meaning is restored (anti-style corruption is a bijection:
        mean-features → 1-x, trend → -x). Returns a shallow-copied obs dict
        with repaired arrays; the ring buffer keeps the RAW forecast, so the
        trust estimate stays anchored to the incoming stream and cannot
        oscillate. No-op (returns obs unchanged) in other modes, during
        warmup, or when no DC crosses the threshold.
        """
        import numpy as np

        if self.mode not in ("repair", "maskdc", "persist") or self._last_corrs is None:
            return global_obs
        corrs = np.nan_to_num(self._last_corrs, nan=0.0)
        if self.mode == "repair":
            bad = np.where(corrs < -self.repair_thresh)[0]
        else:
            thresh = self.threshold if self.threshold is not None else 0.2
            bad = np.where(corrs < thresh)[0]
        if bad.size == 0:
            return global_obs

        obs = dict(global_obs)
        if self.mode == "repair":
            # inversion gamble: only pays off when corruption really is anti.
            transforms = (
                ("dc_future_short_mean",       lambda x, d: 1.0 - x),
                ("dc_future_short_trend",      lambda x, d: -x),
                ("dc_future_long_mean",        lambda x, d: 1.0 - x),
                ("dc_future_long_peak_timing", lambda x, d: 1.0 - x),
            )
        elif self.mode == "maskdc":
            # per-DC neutralization: destroy the lying DC's forecast information
            # (uninformative constants) while honest DCs keep theirs.
            transforms = (
                ("dc_future_short_mean",       lambda x, d: np.full_like(x, 0.5)),
                ("dc_future_short_trend",      lambda x, d: np.zeros_like(x)),
                ("dc_future_long_mean",        lambda x, d: np.full_like(x, 0.5)),
                ("dc_future_long_peak_timing", lambda x, d: np.full_like(x, 0.5)),
            )
        else:  # persist
            # substitute a boring-but-honest forecast: persistence of the DC's
            # realized green, normalized by its rolling max (ring buffer).
            m = min(self._n, self.window)
            recent = self._real[:m] if (self._real is not None and m > 0) else None
            def _gnorm(d):
                if recent is None or d >= recent.shape[1]:
                    return 0.5
                peak = float(recent[:, d].max())
                if peak <= 1e-9:
                    return 0.0
                return float(np.clip(recent[-1 if m < self.window else (self._n - 1) % self.window, d] / peak, 0.0, 1.0))
            transforms = (
                ("dc_future_short_mean",       lambda x, d: np.full_like(x, _gnorm(d))),
                ("dc_future_short_trend",      lambda x, d: np.zeros_like(x)),
                ("dc_future_long_mean",        lambda x, d: np.full_like(x, _gnorm(d))),
                ("dc_future_long_peak_timing", lambda x, d: np.full_like(x, 0.5)),
            )
        for key, fn in transforms:
            if key not in obs:
                continue
            arr = np.array(obs[key], dtype=np.float32).ravel().copy()
            for d in bad[bad < arr.size]:
                arr[d] = fn(arr[d:d + 1], int(d))[0]
            obs[key] = arr
        self._n_repaired += 1
        return obs

    def maybe_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Suppress DEFER (last choice column) when trust is broken:
        corr < threshold. NaN (warmup) never gates."""
        corr = self._last_corr
        gated = (
            self.mode == "gate"
            and self.threshold is not None
            and corr == corr            # not NaN
            and corr < self.threshold
        )
        if gated:
            logits = logits.clone()
            logits[..., -1] = torch.finfo(logits.dtype).min
            self._n_gated += 1
        elif (
            self.mode == "soft"
            and self.threshold is not None
            and corr == corr
            and corr < self.threshold
        ):
            # graded distrust: shave the DEFER logit in proportion to how far
            # below threshold the correlation fell, instead of a hard ban.
            severity = min(max(self.threshold - corr, 0.0), 2.0) / 2.0
            logits = logits.clone()
            logits[..., -1] = logits[..., -1] - self.soft_lambda * severity
            self._n_gated += 1
            gated = True
        if self._log_file is not None:
            self._log_file.write(
                f"{self._step},{corr:.6g},{min(self._n, self.window)},{int(gated)}\n"
            )
        self._step += 1
        return logits

    def summary(self) -> str:
        return (
            f"ForecastResidualMonitor[{self.mode}]: steps={self._step}, "
            f"gated={self._n_gated} "
            f"({(100.0 * self._n_gated / max(1, self._step)):.1f}%), "
            f"repaired={self._n_repaired}, "
            f"last_corr={self._last_corr:.4f}"
        )

    def close(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
