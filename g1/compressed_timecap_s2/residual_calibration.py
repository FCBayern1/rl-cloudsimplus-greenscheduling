"""Measure the timecap_cal tier from the EXISTING TimeCAP checkpoint. GPU-side script.

No training happens here. The script runs the already-trained predictor over its own
validation split, collects the residuals of the 144-row prediction, and freezes three
numbers plus provenance into an artifact the perturbation ladder reads:

    sigma_rel    std of the residual, divided by the mean absolute target level
    ar1_rho      lag-1 autocorrelation of the residual along the lead axis
    lead_alpha   fitted ratio of the lead-0 residual std to the lead-143 residual std

That artifact is what makes one rung of the ladder stand at the quality the real
predictor is KNOWN to reach, instead of at a guessed noise level.

It drives the DEPLOYED prediction path, `TimeCAP_GreenPredictor.update()` then
`_forward_one_turbine()`, so whatever scaling or padding the deployment does is inside
the measured residual. Anchors step through the split with a stride; at each anchor the
buffer holds the true history and the 144-row prediction is compared with the true next
144 Patv rows read from the same CSV.

Run on the GPU server (paths from the frozen manifest, not hard-coded):

    .venv/bin/python g1/compressed_timecap_s2/residual_calibration.py \
        --checkpoint <path/to/checkpoint.pth> \
        --val-csv <turbine 13-feature split CSV, validation year> \
        --turbine-id <id> \
        --out g1/compressed_timecap_s2/timecap_cal.json

Repeat --val-csv/--turbine-id pairs to pool several turbines. The validation data must
be the registered predictor split (2020 under the recommended isolation) and must not
overlap any scheduler DISCOVERY or CONFIRMATION window. The artifact records the SHA of
everything it read so that claim is checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _true_patv(csv_path):
    import csv as _csv

    import numpy as np
    with open(csv_path) as f:
        rows = [float(r["Patv"] or 0.0) for r in _csv.DictReader(f)]
    return np.asarray(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-csv", action="append", required=True)
    ap.add_argument("--turbine-id", action="append", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=48,
                    help="rows between anchors; 48 keeps anchors well separated")
    ap.add_argument("--label-offset", type=int, default=1,
                    help="row the first predicted value refers to, relative to the "
                         "anchor. AUDIT POINT: confirm against the provider's k=0 "
                         "semantics before trusting the calibration (work order s6)")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine; this is inference over one split")
    args = ap.parse_args()
    if len(args.val_csv) != len(args.turbine_id):
        raise SystemExit("--val-csv and --turbine-id must be paired")

    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "drl-manager"))
    from timecap_prediction.predictor import TimeCAP_GreenPredictor  # noqa: E402

    residuals, targets = [], []
    for csv_path, tid in zip(args.val_csv, args.turbine_id):
        predictor = TimeCAP_GreenPredictor(
            checkpoint_path=args.checkpoint,
            turbine_csv_paths={tid: os.path.abspath(csv_path)},
            device=args.device)
        seq, pred_len = predictor.seq_len, predictor.pred_len
        truth = _true_patv(csv_path)
        predictor.reset()
        step = 0
        for anchor in range(seq, len(truth) - pred_len, args.stride):
            while step <= anchor:                    # feed true history up to the anchor
                predictor.update(step)
                step += 1
            pred = predictor._forward_one_turbine(tid)
            if pred is None:
                raise SystemExit(f"inference failed at anchor {anchor} ({csv_path})")
            lo = anchor + args.label_offset
            target = truth[lo:lo + pred_len]
            residuals.append(np.asarray(pred, dtype=np.float64) - target)
            targets.append(target)
    if not residuals:
        raise SystemExit("no anchors produced; check the split and paths")

    R = np.stack(residuals)                         # (n_windows, 144)
    T = np.stack(targets)
    scale_ref = float(np.mean(np.abs(T)))
    sigma_rel = float(np.std(R) / max(scale_ref, 1e-9))

    centred = R - R.mean(axis=0, keepdims=True)
    num = float(np.sum(centred[:, 1:] * centred[:, :-1]))
    den = float(np.sum(centred[:, :-1] ** 2))
    ar1_rho = num / max(den, 1e-12)

    per_lead_std = R.std(axis=0)
    lead_alpha = float(per_lead_std[0] / max(per_lead_std[-1], 1e-12))

    out = {
        "sigma_rel": sigma_rel,
        "ar1_rho": float(np.clip(ar1_rho, 0.0, 0.99)),
        "lead_alpha": float(np.clip(lead_alpha, 0.05, 1.0)),
        "per_lead_std_rel": [float(x / max(scale_ref, 1e-9)) for x in per_lead_std],
        "n_windows": int(R.shape[0]),
        "scale_ref": scale_ref,
        "stride": args.stride,
        "label_offset": args.label_offset,
        "source_checkpoint_sha": _sha(args.checkpoint),
        "val_csv_shas": {os.path.basename(p): _sha(p) for p in args.val_csv},
        "turbine_ids": args.turbine_id,
    }
    tmp = args.out + ".partial"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, args.out)
    print(json.dumps({k: out[k] for k in ("sigma_rel", "ar1_rho", "lead_alpha",
                                          "n_windows")}, indent=2))


if __name__ == "__main__":
    main()
