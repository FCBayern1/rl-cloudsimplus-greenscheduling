"""DC-level residual calibration for checkpoint_residual_surrogate_v2 (ladder-v2, R2).

The planner never sees single turbines; it sees per-DC green. So the surrogate must be
calibrated on the residuals of the DC aggregates the deployment actually produces:
DC0 = T12+T36, DC1 = T95+T91, DC2 = T96, predictions summed per DC through the deployed
inference path, anchors synchronized across all five turbines, 2020 data only.

Outputs, all measured and none hand-rounded:

    sigma_rel_dc     per-DC residual std over the DC's mean absolute truth
    ar1_rho          median over DCs of the lag-1 residual autocorrelation along leads
    lead_alpha       median over DCs of lead-0 std over lead-143 std
    corr_matrix      3x3 correlation of standardized DC residuals at same anchor+lead
    c                median of the three off-diagonals

The prereg freezes the single-factor reproduction tolerance max|r_ij - c| <= 0.10; if the
measured matrix violates it the surrogate is not force-fitted and the fallback is a
deterministic residual replay under a further addendum.
"""
from __future__ import annotations

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
STRIDE = 480
LABEL_OFFSET = 0          # k=0 audit: the provider's window starts at the current row
SEQ, PRED = 96, 144
TOLERANCE = 0.10


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


def _pin_threads():
    """Byte-reproducibility across machines requires a fixed BLAS reduction order.

    The 3060 replication showed the fitted scalars move by ~1e-7 with the thread count
    (provenance fields byte-stable, same-thread reruns byte-identical), so inference runs
    single-threaded and the artifact records it. Existing artifacts predate this pin and
    stay as committed: the v1 ladder is closed and the v2 sweep froze its input in
    flight; their cross-machine claim is provenance-bytes plus fitted values to 1e-6.
    """
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def main():
    _pin_threads()
    sys.path.insert(0, os.path.join(REPO, "drl-manager"))
    from timecap_prediction.predictor import TimeCAP_GreenPredictor

    paths = {t: os.path.join(SPL, f"Turbine_{t}_{YEAR}.csv")
             for ts in DC_TURBINES.values() for t in ts}
    truths = {t: _truth(p) for t, p in paths.items()}
    n = min(len(v) for v in truths.values())
    anchors = list(range(SEQ, n - PRED, STRIDE))

    per_turbine_pred = {}
    for t, p in paths.items():
        pred = TimeCAP_GreenPredictor(checkpoint_path=CK,
                                      turbine_csv_paths={t: os.path.abspath(p)},
                                      device="cpu")
        pred.reset()
        step, rows = 0, {}
        for a in anchors:
            while step <= a:
                pred.update(step)
                step += 1
            f = pred._forward_one_turbine(t)
            if f is None:
                raise SystemExit(f"inference failed for turbine {t} at anchor {a}")
            rows[a] = np.asarray(f, dtype=np.float64)
        per_turbine_pred[t] = rows

    res, scale = {}, {}
    for d, ts in DC_TURBINES.items():
        R = []
        for a in anchors:
            pred = sum(per_turbine_pred[t][a] for t in ts)
            lo = a + LABEL_OFFSET
            true = sum(truths[t][lo:lo + PRED] for t in ts)
            R.append(pred - true)
        R = np.stack(R)
        res[d] = R
        scale[d] = float(np.mean(np.abs(
            np.stack([sum(truths[t][a:a + PRED] for t in ts) for a in anchors]))))

    sigma_rel_dc, ar1, alpha = {}, {}, {}
    for d, R in res.items():
        sigma_rel_dc[str(d)] = float(np.std(R) / max(scale[d], 1e-9))
        centred = R - R.mean(axis=0, keepdims=True)
        num = float(np.sum(centred[:, 1:] * centred[:, :-1]))
        den = float(np.sum(centred[:, :-1] ** 2))
        ar1[d] = num / max(den, 1e-12)
        std = R.std(axis=0)
        alpha[d] = float(std[0] / max(std[-1], 1e-12))

    z = {d: (res[d] / max(res[d].std(), 1e-12)).ravel() for d in res}
    corr = np.corrcoef(np.stack([z[d] for d in sorted(z)]))
    off = [corr[0, 1], corr[0, 2], corr[1, 2]]
    c = float(np.median(off))
    reproduction_ok = bool(max(abs(x - c) for x in off) <= TOLERANCE)

    out = {
        "c": c, "corr_matrix": corr.tolist(),
        "off_diagonals": [float(x) for x in off],
        "single_factor_tolerance": TOLERANCE,
        "single_factor_reproduction_ok": reproduction_ok,
        "sigma_rel_dc": sigma_rel_dc,
        "ar1_rho": float(np.median(list(ar1.values()))),
        "ar1_rho_per_dc": {str(d): float(v) for d, v in ar1.items()},
        "lead_alpha": float(np.clip(np.median(list(alpha.values())), 0.05, 1.0)),
        "lead_alpha_per_dc": {str(d): float(v) for d, v in alpha.items()},
        "anchors": len(anchors), "stride": STRIDE, "label_offset": LABEL_OFFSET,
        "year": YEAR, "dc_turbines": {str(k): list(v) for k, v in DC_TURBINES.items()},
        "torch_num_threads": 1,
        "source_checkpoint_sha": _sha(CK),
        "val_csv_shas": {os.path.basename(p): _sha(p) for p in paths.values()},
    }
    path = os.path.join(HERE, "dc_residual_cal.json")
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    print(json.dumps({k: out[k] for k in
                      ("c", "off_diagonals", "single_factor_reproduction_ok",
                       "sigma_rel_dc", "ar1_rho", "lead_alpha", "anchors")},
                     indent=2, sort_keys=True))
    if not reproduction_ok:
        print("SINGLE FACTOR INSUFFICIENT: fall back to deterministic residual replay "
              "(needs a further addendum before running)")
        sys.exit(1)


if __name__ == "__main__":
    main()
