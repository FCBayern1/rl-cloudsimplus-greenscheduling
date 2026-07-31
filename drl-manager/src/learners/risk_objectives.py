"""Risk-averse RL objectives as advantage transforms — cross-comparison baselines.

These are the standard risk-averse policy-gradient objectives a reviewer expects us to
compare against. Each is implemented as a *mean-preserving-ish* transform of the PPO
advantage, so it plugs onto the SAME actor/critic backbone as EU-CRD — the only thing
that changes is the objective, making the comparison fair.

The shared blind spot (our paper's point): all three operate on the RETURN/advantage
distribution WITHOUT distinguishing WHY a return is low — policy-controllable error vs
exogenous forecast error. They therefore make the policy globally conservative but cannot
target the forecast-specific risk that EU-CRD isolates.

Config (model_config['crd']['risk']):
    kind:  "cvar" | "risk_sensitive" | "mean_variance" | "none"
    cvar:  alpha (tail fraction, default 0.2), lam (mean-CVaR mix, default 0.5)
    risk_sensitive: beta (risk aversion, default 1.0), wmax (weight clamp, default 5.0)
    mean_variance:  lam (variance penalty, default 0.5)

References: Rockafellar & Uryasev (2000, CVaR); Tamar et al. (2012, variance PG);
Howard & Matheson (1972, exponential utility / risk-sensitive control).
"""
from typing import Any, Dict

import torch


def apply_risk_objective(
    adv: torch.Tensor,
    cfg: Dict[str, Any],
    ret: torch.Tensor = None,
) -> torch.Tensor:
    """Transform PPO advantages per the configured risk objective.

    Args:
        adv: advantage tensor (any shape; reductions are over all elements).
        cfg: the `crd.risk` config dict.
        ret: optional return tensor (VALUE_TARGETS) — the *return distribution*.
            Used by the distributional-CVaR objective ("dist_cvar"); other
            objectives operate on the advantage distribution alone.
    Returns:
        transformed advantage tensor, same shape.
    """
    kind = str(cfg.get("kind", "none")).strip().lower()
    if kind in ("", "none"):
        return adv
    if not isinstance(adv, torch.Tensor) or adv.numel() == 0:
        return adv

    flat = adv.reshape(-1)
    a = flat.detach()

    if kind == "dist_cvar":
        # Distributional CVaR: model the RETURN distribution (not the advantage)
        # and focus learning on its lower tail (downside outcomes). This is the
        # distributional-RL flavor of CVaR — CVaR is taken over the return
        # distribution R, and transitions whose return falls in the worst-alpha
        # tail have their advantage upweighted by 1/alpha (mean-CVaR mixed).
        # Falls back to the advantage distribution if returns are unavailable.
        alpha = min(max(float(cfg.get("alpha", 0.1)), 1e-3), 0.999)
        lam = float(cfg.get("lam", 0.7))
        if isinstance(ret, torch.Tensor) and ret.numel() == flat.numel():
            r = ret.reshape(-1).detach()
        else:
            r = a  # graceful fallback: use the advantage as the return proxy
        var_thresh = torch.quantile(r, alpha)          # VaR_alpha on RETURNS
        tail = (r <= var_thresh).to(flat.dtype)         # downside-return tail
        w = (1.0 - lam) + lam * tail / alpha            # upweight downside outcomes
        w = w / w.mean().clamp_min(1e-6)                # mean-preserving
        return (flat * w).reshape(adv.shape)

    if kind == "cvar":
        # Mean-CVaR: (1-lam)*mean objective + lam*CVaR_alpha objective. The CVaR
        # component upweights the worst-alpha tail (most negative advantages =
        # worst outcomes) by 1/alpha so the policy learns hardest from bad cases.
        alpha = float(cfg.get("alpha", 0.2))
        lam = float(cfg.get("lam", 0.5))
        alpha = min(max(alpha, 1e-3), 0.999)
        var_thresh = torch.quantile(a, alpha)          # VaR_alpha (low tail)
        tail = (flat <= var_thresh).to(flat.dtype)     # 1 on the bad tail
        # Mean-normalized weight form (matches the other transforms): without
        # the w.mean() division the 1/alpha tail upweight inflated the total
        # gradient magnitude ~ (1+lam*(1/alpha-1)) -- an effective-LR change
        # confounded with the risk objective.
        w = (1.0 - lam) + lam * tail / alpha
        w = w / w.mean().clamp_min(1e-6)
        return (flat * w).reshape(adv.shape)

    if kind == "risk_sensitive":
        # Exponential utility U(x) = -exp(-beta*x): risk-averse for beta>0. The
        # policy-gradient weight is exp(-beta*z), z = standardized advantage, so
        # negative (bad) advantages are UP-weighted. Mean-normalized so the total
        # learning signal is preserved (only redistributed toward bad outcomes).
        beta = float(cfg.get("beta", 1.0))
        wmax = float(cfg.get("wmax", 5.0))
        std = a.std().clamp_min(1e-6)
        z = (flat - a.mean()) / std
        w = torch.exp(-beta * z).clamp(max=wmax)
        w = w / w.detach().mean().clamp_min(1e-6)
        return (flat * w).reshape(adv.shape)

    if kind == "mean_variance":
        # Variance-averse reweighting (mean-variance objective E[R]-lam*Var[R]):
        # downweight high-deviation transitions (the variance contributors), so
        # the policy is pulled toward low-variance behavior. Mean-normalized.
        lam = float(cfg.get("lam", 0.5))
        lam = min(max(lam, 0.0), 1.0)
        std = a.std().clamp_min(1e-6)
        z = (a - a.mean()) / std
        # Tamar et al. (2012) mean-variance policy gradient: linear ASYMMETRIC
        # weight 1 - 2*lam*z (downweight above-mean returns, upweight below-mean)
        # clamped at 0; mean-normalized. The previous symmetric 1/(1+lam*z^2)
        # suppressed BOTH tails (outlier-robustness, not risk aversion).
        w = (1.0 - 2.0 * lam * z).clamp_min(0.0)
        w = w / w.mean().clamp_min(1e-6)
        return (flat * w).reshape(adv.shape)

    raise ValueError(f"unknown risk objective kind: {kind!r}")
