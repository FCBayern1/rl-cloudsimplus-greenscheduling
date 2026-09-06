import csv, json, os, sys
sys.path.insert(0, "/home/joshua/rl-cloudsimplus-greenscheduling/g1/compressed_timecap_s2")
import ladder_run as lr, f_fits_run as ff
lad = json.load(open(ff.LADDER_VERDICT)); flat = {int(k): w["shrink_0"]["C_sim"] for k, w in lad["windows"].items()}
dev = lr._dev(); out = {}
for F in ("F2", "F3"):
    cfg, cell = ff.twin(F); d = os.path.join(ff.OUT, "cover_diag", F); os.makedirs(d, exist_ok=True); num = den = 0.0; out[F] = {}
    for k, off in enumerate(dev):
        p = os.path.join(d, f"k{k}.csv")
        ok = lr._evaluate(cfg, cell, k, off, "cover_argmax", p, {"OFFSET_GRID_DENSE": "1"})
        row = list(csv.DictReader(open(p)))[-1] if ok else {}
        ref = list(csv.DictReader(open(os.path.join(ff.CAUSAL, f"causal_truth_k{k}.csv"))))[-1]
        c, ce = float(row.get("total_carbon_kg", "nan")), float(ref["total_carbon_kg"]); head = flat[k] - ce
        cap = (flat[k] - c) / head
        assert str(row.get("cover_missing")) == "0", f"arm did not see cand_green_cover: {row.get(chr(99)+chr(111)+chr(118)+chr(101)+chr(114)+chr(95)+chr(109)+chr(105)+chr(115)+chr(115)+chr(105)+chr(110)+chr(103))}"
        out[F][k] = {"C": c, "capture": cap, "ontime": row.get("ontime_mi_share"), "forced": row.get("deadline_forced_count"), "missing": row.get("cover_missing")}
        num += flat[k] - c; den += head
        print(f"cover_argmax {F} k{k}: carbon {c:.6f} capture {cap:.3f} ontime {row.get('ontime_mi_share')} forced {row.get('deadline_forced_count')}", flush=True)
    out[F]["pooled_capture"] = num / den; print(f"cover_argmax {F} pooled capture {num/den:.3f}", flush=True)
json.dump(out, open(os.path.join(ff.OUT, "cover_diag.json"), "w"), indent=1)
