"""
M3 — Stateful soft-blender for the EU-CRD routing counterfactual.

Combines the value-level signal (ΔQ from M2.4) and the reward-level fallback
(Δr from M2.5) into a single per-transition routing-responsibility signal:

    R^routing_t = c(t) · ΔQ_t + (1 - c(t)) · Δr_t

where the confidence weight c(t) ∈ [0, 1] is driven by the per-transition
ensemble-disagreement σ²_tot from M2.4:

    c(t) = exp(-σ²_tot(t) / τ(t))

The temperature τ adapts to training maturity via an EMA of σ²:

    bar_σ²(t) ← (1-η)·bar_σ²(t-1) + η·mean(σ²_tot)
    τ(t)      = τ_0 · exp(κ · bar_σ²(t))

Larger bar_σ² (immature critic) → larger τ → c(t) less aggressive about
penalising high-σ² transitions; as critic matures and bar_σ² shrinks, τ
contracts and c(t) becomes a discriminative OOD signal — yielding an
implicit curriculum from reward-level to value-level CFs without an
externally tuned schedule (Osband et al., 2018; plan §3.3).

Pure-Python/torch and free of any RLlib coupling: testable in isolation
and reusable for future variants of the blending function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class CRDBlender:
    """
    Stateful EMA-driven blender for one (module_id) policy.

    Attributes:
        tau_0: base temperature τ_0; sets the floor for τ.
        kappa: how aggressively τ stretches as bar_σ² grows.
        eta:   EMA rate for bar_σ² (larger = more reactive, less stable).
        ema_init: optional initial bar_σ². If None (default), the first
            batch bootstraps the EMA with its own σ²-mean — avoids cold-start
            bias toward an arbitrary constant.

    Internal state:
        bar_sigma2: running EMA of σ²_tot mean across all batches seen.
    """

    tau_0: float = 1.0
    kappa: float = 0.5
    eta: float = 0.05
    # v5.3 (tau_mode="linear"): τ = τ₀ · bar_σ². The exponential form
    # τ = τ₀·exp(κ·bar_σ²) assumes bar_σ² stays O(1); with raw-return-scale
    # Q-heads, σ² grows with the squared return scale and τ explodes
    # (observed 7.9e4 on rwtight — gate saturated open, entropy de-converged).
    # Linear τ is scale-free: c = exp(−σ²/(τ₀·bar_σ²)) depends only on the
    # RATIO σ²/bar_σ², so a typical transition always gets c = e^(−1/τ₀) and
    # a λ-times-typical outlier gets c_typ^λ, at any value scale.
    tau_mode: str = "exp"
    ema_init: Optional[float] = None
    # Ablation knob for the "EU" (epistemic-uncertainty) component: when set,
    # the confidence gate c(t) is FIXED to this constant instead of the adaptive
    # c(t)=exp(-σ²/τ). fixed_c=1.0 disables epistemic calibration entirely —
    # the blend becomes R = ΔQ (always trust the Q-signal, ignore ensemble
    # disagreement). This isolates what the epistemic-uncertainty gating buys.
    fixed_c: Optional[float] = None

    bar_sigma2: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.ema_init is not None:
            self.bar_sigma2 = float(self.ema_init)
        if self.tau_mode not in ("exp", "linear"):
            raise ValueError(f"tau_mode must be 'exp' or 'linear', got {self.tau_mode!r}")
        if self.tau_0 <= 0.0:
            raise ValueError(f"tau_0 must be > 0, got {self.tau_0}")
        if not 0.0 < self.eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {self.eta}")
        if self.fixed_c is not None and not 0.0 <= self.fixed_c <= 1.0:
            raise ValueError(f"fixed_c must be in [0, 1], got {self.fixed_c}")

    # Cap on κ·bar_σ² to avoid float overflow when σ² explodes. e^60 ≈ 1e26
    # is comfortably in float32 range; well past anything a healthy K-head
    # ensemble should produce. If you hit this clamp regularly the q-heads
    # have other problems.
    _EXP_ARG_CLAMP = 60.0

    def temperature(self) -> float:
        """Current adaptive τ. Falls back to τ_0 if EMA hasn't bootstrapped."""
        if self.bar_sigma2 is None:
            return self.tau_0
        if self.tau_mode == "linear":
            # Scale-free: τ tracks the running σ² level directly. Gate value
            # depends only on σ²/bar_σ², immune to value-scale growth.
            return self.tau_0 * max(self.bar_sigma2, 1e-12)
        arg = self.kappa * self.bar_sigma2
        if arg > self._EXP_ARG_CLAMP:
            arg = self._EXP_ARG_CLAMP
        # exp(κ·bar_σ²) ≥ 1, so τ ≥ τ_0 always — c(t) can't run away to ∞.
        return self.tau_0 * math.exp(arg)

    def update_ema(self, sigma2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """
        Fold this minibatch's σ²-mean into the EMA. Bootstraps on first call
        if `ema_init` was None.

        Args:
            mask: optional bool/0-1 tensor, same shape as sigma2. When given,
                only masked-in cells contribute — padded grid cells carry
                garbage σ² that would otherwise bias τ. Non-finite batch means
                are skipped so one NaN batch cannot poison the EMA.

        Returns the post-update bar_σ² for diagnostic logging.
        """
        vals = sigma2
        if mask is not None and bool(mask.any()):
            vals = sigma2[mask.bool()]
        batch_mean = float(vals.detach().mean().item())
        if not math.isfinite(batch_mean):
            return self.bar_sigma2 if self.bar_sigma2 is not None else 0.0
        if self.bar_sigma2 is None:
            self.bar_sigma2 = batch_mean
        else:
            self.bar_sigma2 = (1.0 - self.eta) * self.bar_sigma2 + self.eta * batch_mean
        return self.bar_sigma2

    def blend(
        self,
        dq: torch.Tensor,
        dr: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Pure (no-EMA-update) blend at the *current* τ. Use `update_and_blend`
        in normal training; this method is exposed for tests and ablations.

        Args:
            dq, dr, sigma2: tensors of identical shape (B, T) (or any same shape).
        Returns:
            r_routing: c·dq + (1-c)·dr
            c_t:       per-transition gate ∈ [0, 1]
            tau:       scalar temperature actually used
        """
        if not (dq.shape == dr.shape == sigma2.shape):
            raise ValueError(
                f"shape mismatch: dq={tuple(dq.shape)}, dr={tuple(dr.shape)}, "
                f"sigma2={tuple(sigma2.shape)}"
            )
        tau = max(self.temperature(), 1e-8)
        if self.fixed_c is not None:
            # Ablation: bypass epistemic-uncertainty gating. c is a constant, so
            # the blend ignores σ² entirely (fixed_c=1.0 → R = ΔQ).
            c_t = torch.full_like(dq, float(self.fixed_c))
        else:
            # σ²_tot is non-negative by construction, so c_t ∈ (0, 1].
            c_t = torch.exp(-sigma2 / tau)
        r_routing = c_t * dq + (1.0 - c_t) * dr
        return r_routing, c_t, float(tau)

    def update_and_blend(
        self,
        dq: torch.Tensor,
        dr: torch.Tensor,
        sigma2: torch.Tensor,
        ema_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Idiomatic per-batch entry point: update the EMA with this batch's
        σ²-mean *before* computing the blend, so τ reflects the freshly
        observed uncertainty. `ema_mask` (optional) keeps padded cells out
        of the EMA; the per-cell blend itself is unaffected.

        Returns same triple as `blend`.
        """
        self.update_ema(sigma2, mask=ema_mask)
        return self.blend(dq, dr, sigma2)
