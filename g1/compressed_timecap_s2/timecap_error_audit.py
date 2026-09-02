#!/usr/bin/env python3
"""Real-error audit of the deployed TimeCAP checkpoint. Read-only probe: trains nothing,
edits nothing, and reads no scheduling or carbon artifact of any kind.

Codex work order §3. The shrink pilot (PILOT_SHRINK_REPORT.md) showed that in this testbed
the damage from a flattened forecast does not come from an arm that stops waiting -- the
shrink arms still deferred 11-17 steps per job -- but from an arm that waits into the WRONG
slot: a view flattened toward the mean is systematically optimistic during lean hours
(view = mean > truth), so work gets committed to future slots that are actually brown. That
is a mixture of amplitude loss and lean-time optimism, and it is the classic pathology of a
regression-to-the-mean predictor.

This probe asks whether the real checkpoint has the same disease, and measures it densely
enough to parameterise a contaminator from the measurement rather than from a guess.

The model fitted, per DC and per lead, with mu frozen as that DC's full-year 2020 truth mean:

    pred - mu = lambda * (truth - mu) + b + eps

Everything below is measured on 2020 only, through the deployed inference path
(update -> _forward_one_turbine), single-threaded, at the deployed DC aggregation.
No parameter here may be selected for what it does to a carbon result; the block named
`primary_error_params` is emitted whole and is meant to be consumed whole.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SPL = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/split")
CK = os.path.join(REPO, "drl-manager/timecap_prediction/TimeCAP/model/"
                        "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth")
DC_TURBINES = {0: (12, 36), 1: (95, 91), 2: (96,)}
YEAR = 2020
STRIDE = 240              # denser than the calibration's 480: this is an audit
LABEL_OFFSET = 0          # k=0 audit: the deployed window starts at the current row
SEQ, PRED = 96, 144
PEAK_Q = 75               # within-window percentile defining "peak" and "predicted high"
MAX_LAG = 48              # cross-correlation search range for the phase probe
RANDOM_ARGMAX = 1.0 / 3.0


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _truth(path):
    import csv
    with open(path) as f:
        return np.array([float(r["Patv"] or 0.0) for r in csv.DictReader(f)])


def fit_lambda_b(pred, truth, mu):
    """Per-lead lambda and the two intercepts, on (anchors x leads) matrices.

    Two intercepts are returned on purpose. The work order defines b as the mean bias,
    mean(pred - truth); that is a diagnostic, not the intercept of the fitted line, and the
    two coincide only when lambda == 1. Using the mean bias as a generative intercept
    leaves the residual with a non-zero mean and inflates its variance, which would
    mis-calibrate any contaminator built from it. Both are emitted, each with the residual
    variance that belongs to it, so the consumer picks knowingly rather than by accident.
    """
    p, t = pred - mu, truth - mu
    tm = t.mean(axis=0)
    pm = p.mean(axis=0)
    var_t = t.var(axis=0)
    cov = ((p - pm) * (t - tm)).mean(axis=0)
    lam = np.where(var_t > 1e-12, cov / np.where(var_t > 1e-12, var_t, 1.0), 0.0)
    b_mean_bias = (pred - truth).mean(axis=0)          # the work order's b
    b_ols = pm - lam * tm                              # the intercept of the fitted line
    eps_spec = p - lam * t - b_mean_bias
    eps_ols = p - lam * t - b_ols
    return {
        "lambda_lead": lam,
        "b_mean_bias_lead": b_mean_bias,
        "b_ols_lead": b_ols,
        "resid_var_lead_with_mean_bias": eps_spec.var(axis=0),
        "resid_mean_lead_with_mean_bias": eps_spec.mean(axis=0),
        "resid_var_lead_with_ols_intercept": eps_ols.var(axis=0),
        "eps_ols": eps_ols,
    }


def ar1_along_lead(eps):
    """Lag-1 autocorrelation of the residual along the lead axis, anchors pooled."""
    c = eps - eps.mean(axis=0, keepdims=True)
    num = float(np.sum(c[:, 1:] * c[:, :-1]))
    den = float(np.sum(c[:, :-1] ** 2))
    return num / max(den, 1e-12)


def lean_time_optimism(pred, truth, mu):
    """The pilot's mechanism, measured directly: when the truth is below the DC's own
    level, how often is the forecast above the truth, and by how much?"""
    lean = truth < mu
    n = int(lean.sum())
    if n == 0:
        return {"n_lean_cells": 0, "p_over_given_lean": None,
                "mean_overestimate_given_lean": None,
                "mean_signed_error_given_lean": None, "lean_fraction": 0.0}
    over = (pred > truth) & lean
    d = (pred - truth)[lean]
    return {
        "n_lean_cells": n,
        "lean_fraction": float(lean.mean()),
        "p_over_given_lean": float(over.sum() / n),
        # Mean size of the overestimate, counted only where it overestimates.
        "mean_overestimate_given_lean": float(d[d > 0].mean()) if (d > 0).any() else 0.0,
        # Mean signed error over all lean cells: the net optimism the planner sees.
        "mean_signed_error_given_lean": float(d.mean()),
    }


def peak_rates(pred, truth):
    """Within each anchor's own window, call the top quartile a peak.

    Thresholding each series by its own P75 makes the two comparable even when the forecast
    lives on a different scale, which matters here: a near-flat forecast still has a top
    quartile, and asking whether it lands on the real one is the question.
    """
    t_hi = truth >= np.percentile(truth, PEAK_Q, axis=1, keepdims=True)
    p_hi = pred >= np.percentile(pred, PEAK_Q, axis=1, keepdims=True)
    n_true, n_pred = int(t_hi.sum()), int(p_hi.sum())
    return {
        "true_peak_fraction": float(t_hi.mean()),
        "pred_high_fraction": float(p_hi.mean()),
        # P(not flagged high | truly a peak)
        "miss_rate": float((t_hi & ~p_hi).sum() / n_true) if n_true else None,
        # P(not truly a peak | flagged high)
        "false_peak_rate": float((p_hi & ~t_hi).sum() / n_pred) if n_pred else None,
        # A forecast with no information flags a random quartile, so its false-peak rate
        # equals the fraction of cells that are not peaks.
        "base_rate_not_true_peak": float((~t_hi).mean()),
        "hit_rate": float((t_hi & p_hi).sum() / n_pred) if n_pred else None,
    }


def phase_lag(pred, truth):
    """Best cross-correlation lag per anchor, on mean-removed windows.

    A forecast that barely varies has no phase to speak of; those anchors are counted as
    undefined rather than being assigned the argmax of numerical noise.
    """
    lags, undefined = [], 0
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean()
        t = truth[i] - truth[i].mean()
        if p.std() < 1e-9 or t.std() < 1e-9:
            undefined += 1
            continue
        best, best_r = 0, -np.inf
        for lag in range(-MAX_LAG, MAX_LAG + 1):
            if lag < 0:
                a, b = p[-lag:], t[:len(t) + lag]
            elif lag > 0:
                a, b = p[:len(p) - lag], t[lag:]
            else:
                a, b = p, t
            if a.size < 16 or a.std() < 1e-9 or b.std() < 1e-9:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best_r:
                best_r, best = r, lag
        lags.append(best)
    if not lags:
        return {"n_defined": 0, "undefined_fraction": 1.0, "median_lag": None,
                "iqr_lag": None, "max_lag_searched": MAX_LAG}
    a = np.array(lags, dtype=float)
    return {
        "n_defined": len(lags),
        "undefined_fraction": float(undefined / pred.shape[0]),
        "median_lag": float(np.median(a)),
        "iqr_lag": [float(np.percentile(a, 25)), float(np.percentile(a, 75))],
        "at_search_edge_fraction": float(np.mean(np.abs(a) >= MAX_LAG)),
        "max_lag_searched": MAX_LAG,
    }


def kendall_tau_3(x, y):
    """Kendall tau over three items. With n = 3 it takes values in {-1, -1/3, 1/3, 1}."""
    pairs = ((0, 1), (0, 2), (1, 2))
    conc = sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in pairs)
    return float(conc / len(pairs))


def _rank_agreement(P, T):
    hits, taus = 0, []
    for i in range(P.shape[1]):
        for l in range(P.shape[2]):
            p, t = P[:, i, l], T[:, i, l]
            hits += int(np.argmax(p) == np.argmax(t))
            taus.append(kendall_tau_3(p, t))
    n = P.shape[1] * P.shape[2]
    return float(hits / n), float(np.mean(taus)), int(n)


def ranking_stats(pred_dc, truth_dc, dcs, mu):
    """Do the DCs get ranked the way the truth ranks them, at each anchor and lead?

    Uniform random is the wrong reference here and would flatter the model. The three DCs
    have very different mean levels (their frozen mu differ by more than a factor of two),
    so a forecaster that emits each DC's own mean and nothing else already ranks them
    correctly whenever the truth happens to agree with the long-run order. That constant-mu
    ranker is the reference that isolates whether the model knows anything TIME-VARYING
    about which site is greener right now; 1/3 only tells you it is not shuffling.
    """
    P = np.stack([pred_dc[d] for d in dcs])          # (3, anchors, leads)
    T = np.stack([truth_dc[d] for d in dcs])
    C = np.stack([np.full_like(T[0], mu[d]) for d in dcs])
    hit, tau, n = _rank_agreement(P, T)
    c_hit, c_tau, _ = _rank_agreement(C, T)
    return {
        "n_comparisons": n,
        "argmax_hit_rate": hit,
        "mean_kendall_tau": tau,
        "random_argmax_baseline": RANDOM_ARGMAX,
        "kendall_tau_zero_baseline": 0.0,
        # Emits each DC's frozen full-year mean and nothing else.
        "constant_mu_argmax_hit_rate": c_hit,
        "constant_mu_mean_kendall_tau": c_tau,
        "argmax_lift_over_constant_mu": hit - c_hit,
        "kendall_lift_over_constant_mu": tau - c_tau,
    }


def run(stride=STRIDE, device="cpu"):
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, os.path.join(REPO, "drl-manager"))
    from timecap_prediction.predictor import TimeCAP_GreenPredictor

    paths = {t: os.path.join(SPL, f"Turbine_{t}_{YEAR}.csv")
             for ts in DC_TURBINES.values() for t in ts}
    truths = {t: _truth(p) for t, p in paths.items()}
    n = min(len(v) for v in truths.values())
    anchors = list(range(SEQ, n - PRED, stride))

    per_turbine = {}
    for t, p in paths.items():
        pr = TimeCAP_GreenPredictor(checkpoint_path=CK,
                                    turbine_csv_paths={t: os.path.abspath(p)},
                                    device=device)
        pr.reset()
        step, rows = 0, {}
        for a in anchors:
            while step <= a:
                pr.update(step)
                step += 1
            f = pr._forward_one_turbine(t)
            if f is None:
                raise SystemExit(f"inference failed for turbine {t} at anchor {a}")
            rows[a] = np.asarray(f, dtype=np.float64)
        per_turbine[t] = rows

    dcs = sorted(DC_TURBINES)
    pred_dc, truth_dc, mu = {}, {}, {}
    for d in dcs:
        ts = DC_TURBINES[d]
        pred_dc[d] = np.stack([sum(per_turbine[t][a] for t in ts) for a in anchors])
        truth_dc[d] = np.stack([sum(truths[t][a + LABEL_OFFSET:a + LABEL_OFFSET + PRED]
                                    for t in ts) for a in anchors])
        # Frozen definition: the DC's full-year 2020 truth mean, not a per-window level.
        mu[d] = float(np.mean(sum(truths[t][:n] for t in ts)))

    per_dc, eps_by_dc = {}, {}
    for d in dcs:
        P, T = pred_dc[d], truth_dc[d]
        fit = fit_lambda_b(P, T, mu[d])
        eps_by_dc[d] = fit["eps_ols"]
        p_flat, t_flat = (P - mu[d]).ravel(), (T - mu[d]).ravel()
        var_t = float(t_flat.var())
        lam_pool = float(np.cov(p_flat, t_flat, bias=True)[0, 1] / var_t) if var_t > 1e-12 else 0.0
        per_dc[str(d)] = {
            "mu": mu[d],
            "turbines": list(DC_TURBINES[d]),
            "truth_mean_abs": float(np.mean(np.abs(T))),
            "pred_std_within_window_mean": float(np.mean(P.std(axis=1))),
            "truth_std_within_window_mean": float(np.mean(T.std(axis=1))),
            "lambda_lead": fit["lambda_lead"].tolist(),
            "b_mean_bias_lead": fit["b_mean_bias_lead"].tolist(),
            "b_ols_lead": fit["b_ols_lead"].tolist(),
            "resid_var_lead_with_mean_bias": fit["resid_var_lead_with_mean_bias"].tolist(),
            "resid_mean_lead_with_mean_bias": fit["resid_mean_lead_with_mean_bias"].tolist(),
            "resid_var_lead_with_ols_intercept": fit["resid_var_lead_with_ols_intercept"].tolist(),
            "lambda_pooled": lam_pool,
            "lambda_lead_median": float(np.median(fit["lambda_lead"])),
            "b_mean_bias_pooled": float((P - T).mean()),
            "b_ols_pooled": float(p_flat.mean() - lam_pool * t_flat.mean()),
            "resid_ar1_along_lead": ar1_along_lead(fit["eps_ols"]),
            "lean_time_optimism": lean_time_optimism(P, T, mu[d]),
            "peak": peak_rates(P, T),
            "phase": phase_lag(P, T),
        }

    z = np.stack([(eps_by_dc[d] / max(eps_by_dc[d].std(), 1e-12)).ravel() for d in dcs])
    corr = np.corrcoef(z)
    off = [float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2])]

    rank = ranking_stats(pred_dc, truth_dc, dcs, mu)

    lam_all = [per_dc[str(d)]["lambda_pooled"] for d in dcs]
    fp = [per_dc[str(d)]["peak"]["false_peak_rate"] for d in dcs]
    br = [per_dc[str(d)]["peak"]["base_rate_not_true_peak"] for d in dcs]
    verdicts = {
        "q1_regression_to_mean": {
            "question": "is there a lambda < 1 shrinkage toward the DC mean?",
            "answer": bool(all(l < 1.0 for l in lam_all)),
            "lambda_pooled_per_dc": {str(d): per_dc[str(d)]["lambda_pooled"] for d in dcs},
            "lambda_lead_median_per_dc": {str(d): per_dc[str(d)]["lambda_lead_median"]
                                          for d in dcs},
            "max_lambda_pooled": float(max(lam_all)),
        },
        "q2_systematic_false_peaks": {
            "question": "are predicted peaks systematically not real peaks?",
            "answer": bool(all(f is not None and b is not None and f >= b
                               for f, b in zip(fp, br))),
            "false_peak_rate_per_dc": {str(d): per_dc[str(d)]["peak"]["false_peak_rate"]
                                       for d in dcs},
            "base_rate_not_true_peak_per_dc": {
                str(d): per_dc[str(d)]["peak"]["base_rate_not_true_peak"] for d in dcs},
            "note": "at or above the base rate means the flagged peaks carry no more "
                    "information than a random quartile",
        },
        "q3_spatial_ranking_error": {
            "question": "is the cross-DC ranking wrong more than chance would allow?",
            # Answered against the constant-mu ranker, not against uniform random: the
            # three DCs' levels differ by more than 2x, so emitting the long-run order
            # alone already beats 1/3 without knowing anything about the weather.
            "answer": bool(rank["argmax_hit_rate"] <= rank["constant_mu_argmax_hit_rate"]),
            "argmax_hit_rate": rank["argmax_hit_rate"],
            "constant_mu_argmax_hit_rate": rank["constant_mu_argmax_hit_rate"],
            "argmax_lift_over_constant_mu": rank["argmax_lift_over_constant_mu"],
            "random_baseline": RANDOM_ARGMAX,
            "mean_kendall_tau": rank["mean_kendall_tau"],
            "constant_mu_mean_kendall_tau": rank["constant_mu_mean_kendall_tau"],
            "kendall_lift_over_constant_mu": rank["kendall_lift_over_constant_mu"],
        },
    }

    primary = {
        "_consumption_note": "emitted whole and meant to be consumed whole; no parameter "
                             "here was selected for its effect on any carbon result",
        "mu_per_dc": {str(d): mu[d] for d in dcs},
        "lambda_lead_per_dc": {str(d): per_dc[str(d)]["lambda_lead"] for d in dcs},
        "lambda_pooled_per_dc": {str(d): per_dc[str(d)]["lambda_pooled"] for d in dcs},
        "b_mean_bias_lead_per_dc": {str(d): per_dc[str(d)]["b_mean_bias_lead"] for d in dcs},
        "b_ols_lead_per_dc": {str(d): per_dc[str(d)]["b_ols_lead"] for d in dcs},
        "intercept_pairing": {
            "with_b_mean_bias": "resid_var_lead_with_mean_bias",
            "with_b_ols": "resid_var_lead_with_ols_intercept",
            "recommended_for_generation": "b_ols",
            "why": "the mean-bias intercept leaves a non-zero residual mean, so pairing it "
                   "with the OLS residual variance double-counts the offset",
        },
        "resid_var_lead_per_dc": {
            str(d): per_dc[str(d)]["resid_var_lead_with_ols_intercept"] for d in dcs},
        "resid_ar1_along_lead_per_dc": {
            str(d): per_dc[str(d)]["resid_ar1_along_lead"] for d in dcs},
        "resid_corr_matrix": corr.tolist(),
        "resid_corr_off_diagonals": off,
        "resid_corr_median_off_diagonal": float(np.median(off)),
        "lean_time_optimism_per_dc": {
            str(d): per_dc[str(d)]["lean_time_optimism"] for d in dcs},
    }

    return {
        "identity": "real-error audit of the deployed checkpoint; no carbon artifact read",
        "checkpoint": os.path.relpath(CK, REPO),
        "source_checkpoint_sha": _sha(CK),
        "val_csv_shas": {os.path.basename(p): _sha(p) for p in paths.values()},
        "year": YEAR, "stride": stride, "label_offset": LABEL_OFFSET,
        "seq_len": SEQ, "pred_len": PRED,
        "dc_turbines": {str(k): list(v) for k, v in DC_TURBINES.items()},
        "n_anchors": len(anchors), "device": device, "torch_num_threads": 1,
        "peak_percentile": PEAK_Q, "max_lag_searched": MAX_LAG,
        "per_dc": per_dc,
        "resid_corr_matrix": corr.tolist(),
        "resid_corr_off_diagonals": off,
        "ranking": rank,
        "verdicts": verdicts,
        "primary_error_params": primary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "timecap_error_audit.json"))
    a = ap.parse_args()
    out = run(stride=a.stride, device=a.device)
    tmp = a.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, a.out)

    print(f"anchors {out['n_anchors']}  stride {out['stride']}  threads 1")
    print(f"\n{'DC':>3} {'lambda_pool':>12} {'b_meanbias':>12} {'b_ols':>10} "
          f"{'AR1':>8} {'pred_sd':>9} {'truth_sd':>9}")
    for d in ("0", "1", "2"):
        p = out["per_dc"][d]
        print(f"{d:>3} {p['lambda_pooled']:>12.6f} {p['b_mean_bias_pooled']:>12.3f} "
              f"{p['b_ols_pooled']:>10.3f} {p['resid_ar1_along_lead']:>8.4f} "
              f"{p['pred_std_within_window_mean']:>9.2f} "
              f"{p['truth_std_within_window_mean']:>9.2f}")
    print(f"\n{'DC':>3} {'P(over|lean)':>13} {'mean over':>11} {'net signed':>11} "
          f"{'miss':>7} {'falsepk':>8} {'baserate':>9} {'medlag':>7} {'undef':>7}")
    for d in ("0", "1", "2"):
        p = out["per_dc"][d]
        lt, pk, ph = p["lean_time_optimism"], p["peak"], p["phase"]
        print(f"{d:>3} {lt['p_over_given_lean']:>13.4f} "
              f"{lt['mean_overestimate_given_lean']:>11.2f} "
              f"{lt['mean_signed_error_given_lean']:>11.2f} "
              f"{pk['miss_rate']:>7.4f} {pk['false_peak_rate']:>8.4f} "
              f"{pk['base_rate_not_true_peak']:>9.4f} "
              f"{str(ph['median_lag']):>7} {ph['undefined_fraction']:>7.3f}")
    r = out["ranking"]
    print(f"\nranking  argmax hit {r['argmax_hit_rate']:.4f}   "
          f"constant-mu {r['constant_mu_argmax_hit_rate']:.4f}   "
          f"random {RANDOM_ARGMAX:.4f}   lift {r['argmax_lift_over_constant_mu']:+.4f}")
    print(f"         Kendall tau {r['mean_kendall_tau']:.4f}   "
          f"constant-mu {r['constant_mu_mean_kendall_tau']:.4f}   "
          f"lift {r['kendall_lift_over_constant_mu']:+.4f}")
    print(f"resid corr off-diagonals {[round(x, 4) for x in out['resid_corr_off_diagonals']]}")
    print("\nverdicts")
    for k, v in out["verdicts"].items():
        print(f"  {k}: {v['answer']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
