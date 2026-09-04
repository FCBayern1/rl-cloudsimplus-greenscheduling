"""M5: action-conditioned credit audit of EU-CRD on archived checkpoints (STAGE_D_PRIME_DESIGN §5).

Post-verdict diagnostic. For one checkpoint of one line: restore the algorithm (its own env
runner and policy), sample fresh episodes on the training scene, build a LOCAL learner from
the frozen config with the checkpoint's module weights, disable the reweighting warm-up,
burn in the learner-side EMA state on a few batches, then run the EU-CRD term computation
exactly as the learner does and capture, per transition of the global module:

    rho_routing (after normalisation), w (the mean-preserving weight actually applied),
    advantage before and after reweighting, dQ, dr, c_t, tau, and the DEFER share of the
    valid slots at that transition.

Transitions are split into DEFER-dominated (share >= 0.5) and ROUTE-dominated. Reported per
class: distribution summaries, upper-tail amplification P(w > 1), lower-tail suppression
P(w < 0.2), advantage sign, and the DEFER-minus-ROUTE difference of mean w. Nothing here
changes the verdict; nothing here tunes anything.

Usage: python stage_d_credit_audit.py <line> <checkpoint_dir> [--steps 8000] [--burnin 5]
       [--out results/stage_d_credit_audit/<line>_<ckpt>.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DRL = os.path.join(REPO, "drl-manager")
sys.path.insert(0, DRL)

GLOBAL = "global_policy"
CAPTURE_COLS = ("crd_rho_routing", "crd_dq", "crd_dr", "crd_c_t", "crd_tau", "crd_r_routing", "crd_forecast")


# ------------------------------------------------------------------ pure analysis
def defer_share(actions, defer_index, mask=None):
    """actions: (N, slots) ints; mask: (N, slots) 0/1 valid slots or None -> (N,) share."""
    a = np.asarray(actions)
    is_defer = (a == defer_index).astype(float)
    if mask is None:
        return is_defer.mean(axis=1)
    m = np.asarray(mask).astype(float)
    denom = np.maximum(m.sum(axis=1), 1.0)
    return (is_defer * m).sum(axis=1) / denom


def summarize(rec):
    """rec: dict of 1-D arrays (same length): rho, w, adv_pre, adv_post, dq, dr, c_t, tau, share.
    Returns the per-class report."""
    share = np.asarray(rec["share"])
    out = {"n": int(share.size)}
    classes = {"DEFER": share >= 0.5, "ROUTE": share < 0.5}
    for name, sel in classes.items():
        n = int(sel.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        c = {"n": n}
        for k in ("rho", "w", "adv_pre", "adv_post", "dq", "dr", "c_t"):
            v = np.asarray(rec[k])[sel]
            c[k] = {"mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
                    "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90))}
        w = np.asarray(rec["w"])[sel]
        c["upper_tail_amplification"] = float((w > 1.0).mean())
        c["lower_tail_suppression"] = float((w < 0.2).mean())
        adv = np.asarray(rec["adv_pre"])[sel]
        c["adv_positive_frac"] = float((adv > 0).mean())
        c["adv_abs_mean_pre"] = float(np.abs(adv).mean())
        c["adv_abs_mean_post"] = float(np.abs(np.asarray(rec["adv_post"])[sel]).mean())
        out[name] = c
    if out["DEFER"]["n"] and out["ROUTE"]["n"]:
        out["w_defer_minus_route"] = out["DEFER"]["w"]["mean"] - out["ROUTE"]["w"]["mean"]
        out["rho_defer_minus_route"] = out["DEFER"]["rho"]["mean"] - out["ROUTE"]["rho"]["mean"]
        out["dq_defer_minus_route"] = out["DEFER"]["dq"]["mean"] - out["ROUTE"]["dq"]["mean"]
    out["tau"] = float(np.asarray(rec["tau"]).mean()) if len(rec["tau"]) else None
    out["defer_share_mean"] = float(share.mean())
    return out


# ------------------------------------------------------------------ capture
class Capture:
    """Wraps the learner's _compute_responsibilities to snapshot the batch around it."""

    def __init__(self, learner, defer_index):
        self.learner = learner
        self.defer_index = defer_index
        self.records = []
        self.enabled = False
        self._orig = learner._compute_responsibilities

    def install(self):
        cap = self

        def wrapped(*, module_id, batch):
            if not cap.enabled or module_id != GLOBAL:
                return cap._orig(module_id=module_id, batch=batch)
            import torch
            from ray.rllib.core.columns import Columns
            from ray.rllib.utils.postprocessing.value_predictions import Postprocessing  # noqa
            adv_pre = batch[Postprocessing.ADVANTAGES].detach().clone()
            cap._orig(module_id=module_id, batch=batch)
            adv_post = batch[Postprocessing.ADVANTAGES].detach()
            rho = batch.get("crd_rho_routing")
            if rho is None:
                return
            w = (adv_post / torch.where(adv_pre.abs() > 1e-12, adv_pre, torch.ones_like(adv_pre))).detach()
            w = torch.where(adv_pre.abs() > 1e-12, w, torch.ones_like(w))
            acts = batch[Columns.ACTIONS]
            mask = batch.get(Columns.LOSS_MASK)
            n = adv_pre.numel()
            acts_np = acts.detach().cpu().numpy().reshape(n, -1) if acts.numel() % n == 0 else None
            mask_np = None
            share = defer_share(acts_np, cap.defer_index) if acts_np is not None else np.full(n, np.nan)
            rec = {"rho": rho.detach().cpu().numpy().reshape(-1), "w": w.cpu().numpy().reshape(-1),
                   "adv_pre": adv_pre.cpu().numpy().reshape(-1), "adv_post": adv_post.cpu().numpy().reshape(-1),
                   "share": share}
            for col, key in (("crd_dq", "dq"), ("crd_dr", "dr"), ("crd_c_t", "c_t")):
                v = batch.get(col)
                rec[key] = (v.detach().cpu().numpy().reshape(-1) if v is not None and v.numel() == n
                            else np.full(n, np.nan))
            tau = batch.get("crd_tau")
            rec["tau"] = np.asarray([float(tau)]) if tau is not None else np.asarray([])
            if mask is not None and mask.numel() == n:
                keep = mask.detach().cpu().numpy().reshape(-1).astype(bool)
                rec = {k: (v[keep] if v.shape[0] == n else v) for k, v in rec.items()}
            cap.records.append(rec)

        self.learner._compute_responsibilities = wrapped

    def merged(self):
        keys = ("rho", "w", "adv_pre", "adv_post", "dq", "dr", "c_t", "share", "tau")
        return {k: np.concatenate([r[k] for r in self.records]) if self.records else np.array([]) for k in keys}


def build_local_learner(algo, ckpt):
    from ray.rllib.core.rl_module.rl_module import RLModule
    spaces = None
    try:
        spaces = algo.env_runner.get_spaces()
    except Exception:
        pass
    cfg = algo.config.copy(copy_frozen=False)
    cfg.learners(num_learners=0, num_gpus_per_learner=0)
    learner = cfg.build_learner(spaces=spaces) if spaces else cfg.build_learner(env=algo.env_runner.env)
    mods = RLModule.from_checkpoint(os.path.join(ckpt, "learner_group", "learner", "rl_module"))
    learner.module.set_state({mid: mods[mid].get_state() for mid in mods if mid in learner.module})
    # the audit measures the reweighting, so the warm-up gate must be open
    learner._crd_reweight_calls = {GLOBAL: 10 ** 9, "shared_local_policy": 10 ** 9}
    return learner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("checkpoint")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--burnin", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ckpt = os.path.abspath(a.checkpoint)
    out = a.out or os.path.join(DRL, "results", "stage_d_credit_audit", f"{a.line}_{os.path.basename(ckpt)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    os.chdir(DRL)
    from src.baselines.global_schedulers import load_rllib_algorithm
    t0 = time.time()
    algo = load_rllib_algorithm(ckpt, py4j_port_override=None)
    env = algo.env_runner.env
    try:
        num_dc = int(getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "num_datacenters", 5))
    except Exception:
        num_dc = 5
    defer_index = num_dc
    learner = build_local_learner(algo, ckpt)
    cap = Capture(learner, defer_index)
    cap.install()

    def one_batch(record):
        eps = algo.env_runner.sample(num_timesteps=a.steps, random_actions=False)
        cap.enabled = record
        learner.update_from_episodes(eps, minibatch_size=a.minibatch, num_epochs=1)
        cap.enabled = False

    first = None
    for i in range(a.burnin):
        one_batch(record=(i == 0))
        if i == 0 and cap.records:
            first = summarize(cap.merged())
            cap.records.clear()
    one_batch(record=True)
    warmed = summarize(cap.merged()) if cap.records else None
    res = {"line": a.line, "checkpoint": ckpt, "defer_index": defer_index, "steps_per_batch": a.steps,
           "burnin_batches": a.burnin, "first_batch": first, "warmed": warmed,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("line", "checkpoint", "elapsed_s")}))
    if warmed:
        for cls in ("DEFER", "ROUTE"):
            c = warmed[cls]
            if c["n"]:
                print(f"{cls:5s} n={c['n']:5d} rho={c['rho']['mean']:.3f} w={c['w']['mean']:.3f} "
                      f"P(w>1)={c['upper_tail_amplification']:.2f} P(w<0.2)={c['lower_tail_suppression']:.2f} "
                      f"adv+={c['adv_positive_frac']:.2f} |adv|pre={c['adv_abs_mean_pre']:.3f} post={c['adv_abs_mean_post']:.3f} "
                      f"dq={c['dq']['mean']:+.3f} dr={c['dr']['mean']:+.4f} c_t={c['c_t']['mean']:.2f}")
        if "w_defer_minus_route" in warmed:
            print(f"w(DEFER)-w(ROUTE)={warmed['w_defer_minus_route']:+.3f}  rho diff={warmed['rho_defer_minus_route']:+.3f}  "
                  f"dq diff={warmed['dq_defer_minus_route']:+.3f}  tau={warmed['tau']}  defer share={warmed['defer_share_mean']:.3f}")
    try:
        algo.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
