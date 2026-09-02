#!/usr/bin/env python3
"""The persistence forecaster measured on exactly dc_residual_calibration.py's protocol.

Persistence is not a model anyone proposes; it is the floor. It costs nothing, needs no
checkpoint, and reads only the row the simulator is already standing on. Putting it through
the same calibration the surrogate went through gives two things a retrain cannot be judged
without:

  1. the target a retrained predictor has to clear. "Better than the old checkpoint" is not
     a bar worth clearing if the old checkpoint is itself below the free baseline.
  2. the quantitative form of "this is not the ceiling" if a surrogate tier fails. A failed
     forecast tier means something quite different when the free baseline would also have
     failed than when the free baseline would have passed.

Identical to dc_residual_calibration.py in every respect that could move a number: the same
DC map (DC0 = T12+T36, DC1 = T95+T91, DC2 = T96), the same 2020 split files, the same
anchors (SEQ..n-PRED step 480, synchronized across all five turbines), the same
label_offset = 0, the same per-DC aggregation of predictions before the residual is taken,
and the same output field names so the two artifacts can be diffed field by field.

The one thing that differs is what plays the part of the model:

    prediction[lead] = truth[anchor]        for every lead in 0..143

i.e. hold the anchor row. Note what label_offset = 0 then implies: the truth window also
starts at the anchor row, so the persistence residual at lead 0 is exactly zero for every
anchor, by construction. That makes lead_alpha degenerate (std[0] = 0, so the ratio is 0
and the shared clip floors it at 0.05) and it means sigma_rel is carried entirely by leads
>= 1. This is reported rather than smoothed over: it is a real property of the baseline
under the deployed convention, not an artefact to be fitted around.

With --compare-checkpoint the same pipeline is also run with a real checkpoint, and the
per-lead residuals of both are written into a comparison block. dc_residual_cal.json is
never read or written by this script.
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
DC_TURBINES = {0: (12, 36), 1: (95, 91), 2: (96,)}
YEAR = 2020
STRIDE = 480
LABEL_OFFSET = 0          # k=0 audit: the provider's window starts at the current row
SEQ, PRED = 96, 144
TOLERANCE = 0.10
# The near field the scheduler can actually act on. The retrain acceptance gate is stated
# over this band in the retrain prereg draft; it is recorded here so the artifact carries
# the band it will be judged on.
NEAR_LEADS = list(range(1, 24))


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


def _stats(res, scale):
    """The dc_residual_calibration.py reductions, applied to whatever residuals come in."""
    sigma_rel_dc, ar1, alpha, per_lead = {}, {}, {}, {}
    for d, R in res.items():
        sigma_rel_dc[str(d)] = float(np.std(R) / max(scale[d], 1e-9))
        centred = R - R.mean(axis=0, keepdims=True)
        num = float(np.sum(centred[:, 1:] * centred[:, :-1]))
        den = float(np.sum(centred[:, :-1] ** 2))
        ar1[d] = num / max(den, 1e-12)
        std = R.std(axis=0)
        alpha[d] = float(std[0] / max(std[-1], 1e-12))
        per_lead[str(d)] = {
            "rmse": [float(x) for x in np.sqrt(np.mean(R ** 2, axis=0))],
            "std_rel": [float(x / max(scale[d], 1e-9)) for x in std],
        }
    z = {d: (res[d] / max(res[d].std(), 1e-12)).ravel() for d in res}
    corr = np.corrcoef(np.stack([z[d] for d in sorted(z)]))
    off = [corr[0, 1], corr[0, 2], corr[1, 2]]
    c = float(np.median(off))
    return {
        "sigma_rel_dc": sigma_rel_dc,
        "ar1_rho": float(np.median(list(ar1.values()))),
        "ar1_rho_per_dc": {str(d): float(v) for d, v in ar1.items()},
        "lead_alpha": float(np.clip(np.median(list(alpha.values())), 0.05, 1.0)),
        "lead_alpha_per_dc": {str(d): float(v) for d, v in alpha.items()},
        "lead_alpha_raw_per_dc": {str(d): float(v) for d, v in alpha.items()},
        "corr_matrix": corr.tolist(),
        "off_diagonals": [float(x) for x in off],
        "c": c,
        "single_factor_tolerance": TOLERANCE,
        "single_factor_reproduction_ok": bool(max(abs(x - c) for x in off) <= TOLERANCE),
        "per_lead_dc": per_lead,
    }


def _dc_residuals(pred_by_turbine, truths, anchors):
    """Aggregate to DC level first, then take the residual. Order matters and this is the
    order the deployment uses: the planner sees a DC's summed green, not a turbine's."""
    res, scale = {}, {}
    for d, ts in DC_TURBINES.items():
        R = []
        for a in anchors:
            pred = sum(pred_by_turbine[t][a] for t in ts)
            lo = a + LABEL_OFFSET
            true = sum(truths[t][lo:lo + PRED] for t in ts)
            R.append(pred - true)
        res[d] = np.stack(R)
        scale[d] = float(np.mean(np.abs(
            np.stack([sum(truths[t][a:a + PRED] for t in ts) for a in anchors]))))
    return res, scale


def _checkpoint_predictions(ckpt, paths, anchors):
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, os.path.join(REPO, "drl-manager"))
    from timecap_prediction.predictor import TimeCAP_GreenPredictor
    out = {}
    for t, p in paths.items():
        pr = TimeCAP_GreenPredictor(checkpoint_path=ckpt,
                                    turbine_csv_paths={t: os.path.abspath(p)},
                                    device="cpu")
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
        out[t] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-checkpoint", default="",
                    help="also run this checkpoint through the identical protocol and "
                         "emit a per-lead head-to-head block")
    ap.add_argument("--out", default=os.path.join(HERE, "persistence_baseline_cal.json"))
    a = ap.parse_args()

    paths = {t: os.path.join(SPL, f"Turbine_{t}_{YEAR}.csv")
             for ts in DC_TURBINES.values() for t in ts}
    truths = {t: _truth(p) for t, p in paths.items()}
    n = min(len(v) for v in truths.values())
    anchors = list(range(SEQ, n - PRED, STRIDE))

    # The virtual model: hold the anchor row for the whole horizon.
    persistence = {t: {a: np.full(PRED, truths[t][a], dtype=np.float64) for a in anchors}
                   for t in paths}
    res_p, scale = _dc_residuals(persistence, truths, anchors)
    stats_p = _stats(res_p, scale)

    out = dict(stats_p)
    out.update({
        "model": "persistence",
        "model_definition": "prediction[lead] = truth[anchor] for every lead",
        "source_checkpoint_sha": None,
        "requires_inference": False,
        "deterministic_without_thread_pin": True,
        "lead0_residual_is_zero_by_construction": True,
        "scale_ref_dc": {str(d): v for d, v in scale.items()},
        "near_leads": [NEAR_LEADS[0], NEAR_LEADS[-1]],
        "anchors": len(anchors), "stride": STRIDE, "label_offset": LABEL_OFFSET,
        "year": YEAR, "dc_turbines": {str(k): list(v) for k, v in DC_TURBINES.items()},
        "val_csv_shas": {os.path.basename(p): _sha(p) for p in paths.values()},
        "protocol_source": "g1/compressed_timecap_s2/dc_residual_calibration.py",
    })

    if a.compare_checkpoint:
        ck = os.path.abspath(a.compare_checkpoint)
        res_m, _ = _dc_residuals(_checkpoint_predictions(ck, paths, anchors),
                                 truths, anchors)
        stats_m = _stats(res_m, scale)
        beats = {}
        for d in sorted(DC_TURBINES):
            pm = stats_m["per_lead_dc"][str(d)]["rmse"]
            pp = stats_p["per_lead_dc"][str(d)]["rmse"]
            beats[str(d)] = {
                "near_leads_model_better": int(sum(pm[l] < pp[l] for l in NEAR_LEADS)),
                "near_leads_total": len(NEAR_LEADS),
                "all_leads_model_better": int(sum(m < p for m, p in zip(pm, pp))),
                "all_leads_total": len(pm),
            }
        out["comparison"] = {
            "checkpoint": os.path.relpath(ck, REPO),
            "checkpoint_sha": _sha(ck),
            "device": "cpu", "torch_num_threads": 1,
            "model_stats": stats_m,
            "model_beats_persistence": beats,
        }

    tmp = a.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, a.out)

    print(f"anchors {len(anchors)}   stride {STRIDE}   label_offset {LABEL_OFFSET}")
    print(f"persistence  sigma_rel_dc {stats_p['sigma_rel_dc']}")
    print(f"persistence  ar1 {stats_p['ar1_rho']:.6f}  lead_alpha {stats_p['lead_alpha']:.6f}"
          f"  c {stats_p['c']:.6f}  single_factor_ok {stats_p['single_factor_reproduction_ok']}")
    if a.compare_checkpoint:
        sm = out["comparison"]["model_stats"]
        print(f"checkpoint   sigma_rel_dc {sm['sigma_rel_dc']}")
        print(f"checkpoint   ar1 {sm['ar1_rho']:.6f}  lead_alpha {sm['lead_alpha']:.6f}"
              f"  c {sm['c']:.6f}")
        print("\nper-DC RMSE, model vs persistence")
        print(f"{'lead':>5} " + " ".join(f"{'DC'+str(d)+' model':>13} {'persist':>10}"
                                         for d in sorted(DC_TURBINES)))
        for lead in (0, 1, 2, 5, 11, 23, 47, 95, 143):
            cells = []
            for d in sorted(DC_TURBINES):
                m = sm["per_lead_dc"][str(d)]["rmse"][lead]
                p = stats_p["per_lead_dc"][str(d)]["rmse"][lead]
                cells.append(f"{m:>13.2f} {p:>10.2f}")
            print(f"{lead:>5} " + " ".join(cells))
        print("\nmodel better than persistence:")
        for d, b in out["comparison"]["model_beats_persistence"].items():
            print(f"  DC{d}  near leads 1-23: {b['near_leads_model_better']}/"
                  f"{b['near_leads_total']}   all 144 leads: "
                  f"{b['all_leads_model_better']}/{b['all_leads_total']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
