"""Gate 4, small-sample learnability (reports/OPTION_ACTION_DESIGN.md §6 gate 4, Addenda A4, A7).

Corpus: oracle_opt's decisions on the six development windows (`hz_opt_corpus`): one
decision per job at its first sighting, label = the option action (ROUTE_NOW(d) or
HOLD_FOR_GREEN(d)), with the global observation of every step. k0-k3 train, k4-k5 held out.

Fit: the D' score-based global module in option mode, built from the option config exactly
as training builds it (same spaces, same gtrxl block), trained by per-slot cross-entropy on
the labelled slots, recurrent state carried through each window in time order (detached
between steps), seed 20260905, 200 epochs, no early stopping, no hyper-parameter search.

Score (held out): recurrent pass; P(hold) = the mass on the HOLD columns of the job's slot at
its decision step, RAW (mask lifted) for the classification gate and DEPLOYED for reference;
lift >= 0.10 and balanced AUC >= 0.60 over one decision per job. Fewer than 60 held-out jobs
or fewer than 15 of either class -> INVALID_CORPUS (A7). Site agreement with the oracle is
reported descriptively.

Usage: python option_bc.py fit | score | all
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRL = os.path.join(REPO, "drl-manager")
sys.path.insert(0, HERE)
sys.path.insert(0, DRL)
OUT = os.path.join(HERE, "stage_a_out")
CORPUS = os.path.join(OUT, "option_corpus")
OUTD = os.path.join(OUT, "option_bc")
CONFIG = os.path.join(HERE, "config_stage_d_dprime_option.yml")
BLOCK = "sd_V_s2_r48_w72_c3_n35"
TRAIN_K, HELD_K = (0, 1, 2, 3), (4, 5)
SEED, EPOCHS, LR = 20260905, 200, 1e-3
LIFT_MIN, AUC_MIN = 0.10, 0.60
MIN_HELD_JOBS, MIN_PER_CLASS = 60, 15


# ── pure pieces ───────────────────────────────────────────────────────────────────────
def first_decisions(rows, num_dcs):
    """One (step, slot, action, is_hold, site) per job: its first sighting. Pure."""
    by = {}
    for r in rows:
        cid = int(float(r.get("cloudlet_id", -1) or -1))
        if cid < 0:
            continue
        step, slot, a = int(float(r["step"])), int(float(r["slot"])), int(float(r["action"]))
        if cid not in by or (step, slot) < (by[cid]["step"], by[cid]["slot"]):
            by[cid] = {"id": cid, "step": step, "slot": slot, "action": a,
                       "is_hold": int(a >= num_dcs), "site": a - num_dcs if a >= num_dcs else a}
    return by


def corpus_valid(labels):
    n = len(labels)
    pos = int(sum(labels))
    ok = n >= MIN_HELD_JOBS and pos >= MIN_PER_CLASS and (n - pos) >= MIN_PER_CLASS
    return {"n_jobs": n, "n_hold": pos, "n_route": n - pos, "valid": ok}


def model_config_from_block(cfg):
    from src.baselines.option_bc_module import model_config_from_block as _m
    return _m(cfg)


# ── module (shared with the executed arm, src/baselines/option_bc_module.py) ─────────
def load_block(config=CONFIG, block=BLOCK):
    from src.baselines.option_bc_module import load_block as _l
    return _l(config, block)


def build_module(cfg, seed=SEED):
    from src.baselines.option_bc_module import build_option_module
    return build_option_module(cfg, seed)


def load_window(k, corpus=CORPUS, cell=BLOCK):
    dec = os.path.join(corpus, f"{cell}_k{k}_decisions.csv")
    obs = dec.replace("_decisions.csv", "_decisions_obs.npz")
    rows = list(csv.DictReader(open(dec)))
    z = np.load(obs)
    return rows, {key: z[key] for key in z.keys()}


def _obs_t(z, t, nb):
    import torch
    obs = {k: torch.as_tensor(np.asarray(z[k][t])[None, ...]) for k in z}
    return {"obs": {"observation": obs, "action_mask": torch.ones(1, nb)}}


def _state_in(state):
    import torch
    if isinstance(state, dict):
        return {k: torch.as_tensor(v) for k, v in state.items()}
    return {"gtrxl_mem": torch.as_tensor(state)}


def _next_state(out, prev):
    import torch
    so = out.get("state_out")
    if so is None:
        return prev
    if isinstance(so, dict):
        return {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 4 else v).detach() for k, v in so.items()}
    return {"gtrxl_mem": (so[0] if so.dim() == 4 else so).detach()}


def fit(out_dir=OUTD, corpus=CORPUS, epochs=EPOCHS, seed=SEED, lr=LR, train_k=TRAIN_K):
    import torch
    import torch.nn.functional as F
    cfg = load_block()
    mod, obs_space, act_space = build_module(cfg, seed)
    nvec = [int(x) for x in act_space.nvec]
    nb, nchoice = len(nvec), nvec[0]
    n = nchoice // 2
    windows = []
    for k in train_k:
        rows, z = load_window(k, corpus)
        dec = first_decisions(rows, n)
        targets = {}
        for d in dec.values():
            targets.setdefault(d["step"], []).append((d["slot"], d["action"]))
        n_steps = int(z[next(iter(z))].shape[0])
        windows.append((k, z, targets, n_steps))
    n_samples = sum(len(v) for _k, _z, t, _n in windows for v in t.values())
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    mod.train()
    hist = []
    for ep in range(epochs):
        tot, cnt, correct = 0.0, 0, 0
        for k, z, targets, n_steps in windows:
            state = mod.get_initial_state()
            opt.zero_grad()
            loss_sum, n_t = 0.0, 0
            for t in range(n_steps):
                b = _obs_t(z, t, nb)
                b["state_in"] = _state_in(state)
                out = mod._forward_train(b)
                if t in targets:
                    lg = out["action_dist_inputs"].reshape(-1, nb, nchoice)[-1]
                    slots = torch.tensor([s for s, _a in targets[t]])
                    acts = torch.tensor([a for _s, a in targets[t]])
                    loss_sum = loss_sum + F.cross_entropy(lg[slots], acts, reduction="sum")
                    n_t += len(slots)
                    correct += int((lg[slots].argmax(-1) == acts).sum())
                state = _next_state(out, state)
            if n_t:
                loss = loss_sum / n_t
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * n_t
                cnt += n_t
        hist.append({"epoch": ep + 1, "loss": tot / max(1, cnt), "top1": correct / max(1, cnt)})
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{epochs} loss {hist[-1]['loss']:.4f} top1 {hist[-1]['top1']:.3f}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(mod.state_dict(), os.path.join(out_dir, "model.pt"))
    meta = {"seed": seed, "epochs": epochs, "lr": lr, "train_windows": list(train_k), "n_samples": n_samples,
            "model_config": model_config_from_block(cfg), "history": hist, "decode": "argmax",
            "corpus": corpus, "config": CONFIG, "block": BLOCK}
    with open(os.path.join(out_dir, "fit.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("written", os.path.join(out_dir, "model.pt"))
    return meta


def load_fitted(out_dir=OUTD):
    import torch
    cfg = load_block()
    mod, obs_space, act_space = build_module(cfg)
    mod.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu"))
    mod.eval()
    return mod, act_space


def score(out_dir=OUTD, corpus=CORPUS, held_k=HELD_K):
    import torch
    from timing_selectivity import lift_and_auc
    mod, act_space = load_fitted(out_dir)
    nvec = [int(x) for x in act_space.nvec]
    nb, nchoice = len(nvec), nvec[0]
    n = nchoice // 2
    P = {"raw": [], "dep": []}
    Y, agree, per_window = [], [], {}
    with torch.no_grad():
        for k in held_k:
            rows, z = load_window(k, corpus)
            dec = first_decisions(rows, n)
            want = {}
            for d in dec.values():
                want.setdefault(d["step"], []).append(d)
            n_steps = int(z[next(iter(z))].shape[0])
            state = mod.get_initial_state()
            wp, wy = {"raw": [], "dep": []}, []
            for t in range(n_steps):
                b = _obs_t(z, t, nb)
                b["state_in"] = _state_in(state)
                mod._audit_skip_defer_mask = False
                out = mod.forward_inference(b)
                if t in want:
                    mod._audit_skip_defer_mask = True
                    out_raw = mod.forward_inference(b)
                    mod._audit_skip_defer_mask = False
                    pd_ = torch.softmax(out["action_dist_inputs"].reshape(-1, nb, nchoice)[-1], -1)
                    pr_ = torch.softmax(out_raw["action_dist_inputs"].reshape(-1, nb, nchoice)[-1], -1)
                    for d in want[t]:
                        s = d["slot"]
                        wp["raw"].append(float(pr_[s, n:].sum())); wp["dep"].append(float(pd_[s, n:].sum()))
                        wy.append(d["is_hold"])
                        a = int(pd_[s].argmax())
                        agree.append(int((a >= n) == bool(d["is_hold"]) and (a % n) == d["site"]))
                state = _next_state(out, state)
            per_window[k] = {"raw": lift_and_auc(wp["raw"], wy), "n": len(wy)}
            P["raw"] += wp["raw"]; P["dep"] += wp["dep"]; Y += wy
    val = corpus_valid(Y)
    res = {"held_windows": list(held_k), "corpus_check": val,
           "main_gate_raw": lift_and_auc(P["raw"], Y), "deployed": lift_and_auc(P["dep"], Y),
           "site_and_branch_agreement": float(np.mean(agree)) if agree else None,
           "per_window": per_window, "decode": "argmax"}
    if not val["valid"]:
        res["verdict"] = "INVALID_CORPUS"
    else:
        g = res["main_gate_raw"]
        res["verdict"] = "PASS_CLASSIFICATION" if (g["lift"] is not None and g["lift"] >= LIFT_MIN and g["auc"] >= AUC_MIN) else "FAIL_CLASSIFICATION"
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("corpus_check", "main_gate_raw", "deployed", "site_and_branch_agreement", "verdict")}, indent=1))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fit", "all"):
        fit()
    if what in ("score", "all"):
        score()
