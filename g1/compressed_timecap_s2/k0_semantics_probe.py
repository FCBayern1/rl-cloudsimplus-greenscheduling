#!/usr/bin/env python3
"""Empirical probe for the k=0 alignment question. Read-only: trains nothing, edits nothing.

Two conventions can disagree by one row and the disagreement is invisible in any loss curve:

    training      Dataset_Custom / SingleDataset build seq_y from r_begin = s_end - label_len
                  with label_len = 0, so the label window starts at the row AFTER the last
                  history row. pred[0] means row h+1, where h is the last row fed in.
    deployment    GreenEnergyProvider.computeFutureTrendFeatures sums series[i] for
                  i in [currentIdx, currentIdx + shortTermRows), i.e. the window INCLUDES
                  the row the simulation is standing on. Its drop-in replacement,
                  TimeCAPGodEyeProvider, averages forecast[:short] after update(step) has
                  pushed row `step`, so its window starts at row step+1.

This probe does not argue from the code. It sweeps the label offset over the existing
checkpoint's own validation year and reports, per offset, how well pred[i] tracks
truth[anchor + offset + i]. The offset that minimises the error is the row the network
actually learned to emit first.

Wind at ten minutes a row is strongly autocorrelated, so correlations at adjacent offsets
sit very close together. The decisive statistic is therefore RMSE, and the lead-0 slice is
reported separately: a one-row skew shows up most sharply at the shortest lead, where the
true next row is nearly the current one and being off by one costs the most in relative
terms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
SPLIT = os.path.join(_REPO, "cloudsimplus-gateway/src/main/resources/windProduction/split")


def true_patv(csv_path):
    import csv as _csv
    with open(csv_path) as f:
        return np.asarray([float(r["Patv"] or 0.0) for r in _csv.DictReader(f)],
                          dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--turbine-id", action="append", type=int, required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--stride", type=int, default=480)
    ap.add_argument("--offsets", default="-2,-1,0,1,2,3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # Same single-thread pin the calibration scripts took at 7a40df7, so this probe's
    # numbers are reproducible for a fixed checkpoint and input set.
    torch.set_num_threads(1)
    offsets = [int(x) for x in a.offsets.split(",")]

    sys.path.insert(0, os.path.join(_REPO, "drl-manager"))
    from timecap_prediction.predictor import TimeCAP_GreenPredictor

    preds, truths, anchors_used = [], [], 0
    for tid in a.turbine_id:
        csv_path = os.path.join(SPLIT, f"Turbine_{tid}_{a.year}.csv")
        p = TimeCAP_GreenPredictor(checkpoint_path=a.checkpoint,
                                   turbine_csv_paths={tid: csv_path}, device=a.device)
        seq, H = p.seq_len, p.pred_len
        truth = true_patv(csv_path)
        p.reset()
        step = 0
        pad = max(offsets) + H
        for anchor in range(seq, len(truth) - pad, a.stride):
            while step <= anchor:            # feed true history up to and including `anchor`
                p.update(step)
                step += 1
            out = p._forward_one_turbine(tid)
            if out is None:
                raise SystemExit(f"inference failed at anchor {anchor} ({csv_path})")
            preds.append(np.asarray(out, dtype=np.float64))
            truths.append(truth)
            anchors_used += 1
        # keep the anchor list per turbine so the offset slice is exact
        if not preds:
            raise SystemExit("no anchors produced")

    # Rebuild the anchor list identically to the loop above, per turbine.
    P = np.stack(preds)
    H = P.shape[1]
    anchor_list = []
    for tid in a.turbine_id:
        csv_path = os.path.join(SPLIT, f"Turbine_{tid}_{a.year}.csv")
        n = len(true_patv(csv_path))
        anchor_list += [(tid, anc) for anc in range(96, n - (max(offsets) + H), a.stride)]
    assert len(anchor_list) == len(P), (len(anchor_list), len(P))

    series = {tid: true_patv(os.path.join(SPLIT, f"Turbine_{tid}_{a.year}.csv"))
              for tid in a.turbine_id}

    rows = {}
    for off in offsets:
        T = np.stack([series[tid][anc + off:anc + off + H]
                      for tid, anc in anchor_list])
        resid = P - T
        rows[off] = {
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "mae": float(np.mean(np.abs(resid))),
            "corr": float(np.corrcoef(P.ravel(), T.ravel())[0, 1]),
            "rmse_lead0": float(np.sqrt(np.mean((P[:, 0] - T[:, 0]) ** 2))),
            "corr_lead0": float(np.corrcoef(P[:, 0], T[:, 0])[0, 1]),
        }

    # Per-lead argmin. The pooled RMSE is not an alignment statistic: it is dominated by
    # the long leads, where the model reverts toward a level and shifting the target window
    # earlier simply moves it closer to the recent past, so it drifts monotonically with
    # the offset instead of showing a minimum. Lead 0 alone cannot settle it either, since
    # a model that has learned near-persistence at short lead would favour the offset that
    # lines pred[0] up with the last observed row whatever the label convention was. The
    # discriminating question is whether the SAME offset wins at leads where persistence
    # has decayed. If one offset wins across short, medium and long leads, the emission is
    # genuinely shifted; if the winner drifts with lead, it is a persistence artefact.
    probe_leads = [l for l in (0, 1, 2, 5, 11, 23, 47, 95, 143) if l < H]
    per_lead = {}
    for lead in probe_leads:
        r = {}
        for off in offsets:
            T = np.stack([series[tid][anc + off + lead] for tid, anc in anchor_list])
            r[off] = float(np.sqrt(np.mean((P[:, lead] - T) ** 2)))
        per_lead[str(lead)] = {"rmse_by_offset": {str(k): v for k, v in r.items()},
                               "argmin_offset": min(r, key=r.get)}
    # Persistence reference at each lead: predict the last observed row and hold it.
    last_obs = np.stack([series[tid][anc] for tid, anc in anchor_list])
    persistence = {}
    for lead in probe_leads:
        T = np.stack([series[tid][anc + lead] for tid, anc in anchor_list])
        persistence[str(lead)] = float(np.sqrt(np.mean((last_obs - T) ** 2)))

    best_rmse = min(rows, key=lambda o: rows[o]["rmse"])
    best_lead0 = min(rows, key=lambda o: rows[o]["rmse_lead0"])
    out = {
        "checkpoint": os.path.relpath(a.checkpoint, _REPO),
        "year": a.year, "stride": a.stride, "turbine_ids": a.turbine_id,
        "n_anchors": len(P), "pred_len": H,
        "offsets": {str(k): v for k, v in rows.items()},
        "per_lead": per_lead,
        "persistence_rmse": persistence,
        "argmin_rmse_offset": best_rmse,
        "argmin_rmse_lead0_offset": best_lead0,
        "skew_vs_deployment_offset0": best_rmse - 0,
        "torch_num_threads": torch.get_num_threads(),
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(f"{'offset':>7} {'RMSE':>12} {'MAE':>12} {'corr':>10} {'RMSE@lead0':>12} {'corr@lead0':>11}")
    for off in offsets:
        r = rows[off]
        mark = "  <-- min" if off == best_rmse else ""
        print(f"{off:>7} {r['rmse']:>12.4f} {r['mae']:>12.4f} {r['corr']:>10.6f} "
              f"{r['rmse_lead0']:>12.4f} {r['corr_lead0']:>11.6f}{mark}")
    print(f"\nanchors {len(P)}   argmin RMSE offset {best_rmse}   "
          f"argmin RMSE@lead0 offset {best_lead0}")
    print(f"\nper-lead RMSE by offset (persistence = hold the last observed row)")
    print(f"{'lead':>5} " + " ".join(f"{o:>10}" for o in offsets)
          + f" {'argmin':>7} {'persist':>10}")
    for lead in probe_leads:
        r = per_lead[str(lead)]["rmse_by_offset"]
        print(f"{lead:>5} " + " ".join(f"{r[str(o)]:>10.3f}" for o in offsets)
              + f" {per_lead[str(lead)]['argmin_offset']:>7} "
              + f"{persistence[str(lead)]:>10.3f}")


if __name__ == "__main__":
    main()
