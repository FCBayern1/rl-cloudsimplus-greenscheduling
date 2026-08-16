#!/usr/bin/env python3
"""V3.2B step 2: behaviour-clone the slack-aware teacher into the global module.

Loads the teacher dataset written by teacher_reward_audit.py --dataset-dir
(one npz per episode: obs_* arrays [T, ...], actions [T, 128] in 0..8 with
8 = defer, real_mask [T, 128]), fine-tunes the FULL global RLModule (trunk +
factorized temporal gate + route scores) with per-slot cross-entropy on real
slots only, and writes a probe-loadable checkpoint copy.

Design notes:
  - Source module = v32_g2_s1 checkpoint_000005 (300k): the gate exists and is
    wired, encoders carry 300k steps of representation - BC refines toward the
    teacher's hold/route mapping instead of starting from a random trunk.
  - Steps are treated independently with zero recurrent state (the dataset has
    no state chains). GTrXL memory therefore sees a stateless approximation
    during BC; the PPO fine-tune stage restores the recurrent regime. Recorded
    as a known limitation, decision doc §5.
  - Output keeps RLlib's checkpoint layout under
    <out>/learner_group/learner/rl_module/global_policy so
    probe_forecast_sensitivity.load_module() and later warm-starts read it
    directly; params.json is copied beside it for configure_dims().

Usage:
    .venv/bin/python v32b_bc_train.py --dataset-dir ../local_eval_rt/v32b_teacher_data \
        --checkpoint logs/v32_g2_s1/.../checkpoint_000005 --out ../local_eval_rt/v32b_bc_ck \
        [--epochs 2] [--batch-steps 8] [--lr 1e-4] [--val-frac 0.1]
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def load_dataset(dataset_dir: pathlib.Path):
    """Return (steps, actions, mask): steps = list of per-step obs dicts."""
    files = sorted(dataset_dir.glob("teacher_ep*.npz"))
    if not files:
        sys.exit(f"no teacher_ep*.npz under {dataset_dir}")
    steps, acts, masks = [], [], []
    for f in files:
        d = np.load(f)
        keys = [k for k in d.files if k.startswith("obs_")]
        # Decompress each key ONCE. NpzFile re-inflates the whole array on
        # every d[k] access, and a row view pins its freshly-inflated parent
        # alive - the per-step loop below then holds T copies of every
        # [T, ...] array (~25GB for one episode; rc=137 on the first run).
        arrs = {k: d[k] for k in keys}
        T = d["actions"].shape[0]
        for t in range(T):
            steps.append({k[4:]: arrs[k][t] for k in keys})
        acts.append(d["actions"])
        masks.append(d["real_mask"])
    actions = np.concatenate(acts, axis=0)
    mask = np.concatenate(masks, axis=0)
    assert len(steps) == actions.shape[0] == mask.shape[0]
    return steps, actions, mask


def build_label_weights(steps, actions, mask, defer_idx,
                        hold_refresh_every: int = 32):
    """Label hygiene for the teacher dataset (Codex review, 2026-08-16).

    A job that waits N steps emits N HOLD labels but only one ROUTE label -
    the raw dataset is 99.5% defer and BC collapses to always-defer. Without
    per-slot job ids we use the observation's own defer_count channel:
      - every ROUTE label is kept;
      - a HOLD label is kept only at the FIRST defer decision
        (defer_count == 0) or as a sparse refresher every
        `hold_refresh_every` further holds;
      - padding slots stay excluded (weight 0);
      - after filtering, classes are reweighted to equal total mass so the
        temporal head sees a balanced defer/route objective (weighted CE).
    Returns float32 weights, same shape as actions.
    """
    W = np.zeros_like(actions, dtype=np.float32)
    for i, s in enumerate(steps):
        dc = np.asarray(s.get("batch_cloudlet_defer_count", np.zeros(actions.shape[1])))
        m = mask[i]
        a = actions[i]
        is_hold = (a == defer_idx) & m
        is_route = (a != defer_idx) & m
        keep_hold = is_hold & ((dc < 0.5) | (np.rint(dc) % hold_refresh_every == 0))
        W[i][is_route] = 1.0
        W[i][keep_hold] = 1.0
    hold_mass = float(W[(actions == defer_idx)].sum())
    route_mass = float(W[(actions != defer_idx)].sum())
    if hold_mass > 0 and route_mass > 0:
        W[(actions == defer_idx)] *= route_mass / hold_mass
    return W


def collate(steps, idx):
    """Stack a list of per-step obs dicts into a batched tensor dict."""
    keys = steps[idx[0]].keys()
    return {k: torch.as_tensor(
        np.stack([np.asarray(steps[i][k]) for i in idx]).astype(np.float32))
        for k in keys}


def masked_slot_ce(logits: torch.Tensor, actions: torch.Tensor,
                   mask: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Per-slot CE weighted by `mask` (bool mask or float label weights)."""
    B = logits.shape[0]
    n_opt = logits.shape[-1] // n_slots
    z = logits.reshape(B, n_slots, n_opt)
    ce = torch.nn.functional.cross_entropy(
        z.reshape(-1, n_opt), actions.reshape(-1).long(), reduction="none")
    m = mask.reshape(-1).float()
    return (ce * m).sum() / m.sum().clamp_min(1e-6)


def defer_metrics(logits, actions, mask, n_slots, defer_idx):
    """Accuracy overall + precision/recall on the defer class (real slots)."""
    B = logits.shape[0]
    n_opt = logits.shape[-1] // n_slots
    pred = logits.reshape(B, n_slots, n_opt).argmax(-1)
    m = mask.bool()
    if int(m.sum()) == 0:
        return float("nan"), float("nan"), float("nan")
    acc = (pred[m] == actions[m]).float().mean().item()
    t_def = (actions[m] == defer_idx)
    p_def = (pred[m] == defer_idx)
    tp = (t_def & p_def).sum().item()
    prec = tp / max(1, p_def.sum().item())
    rec = tp / max(1, t_def.sum().item())
    return acc, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="source RLlib checkpoint dir (checkpoint_00000N)")
    ap.add_argument("--out", required=True, help="output checkpoint copy dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics-out", default=None)
    args = ap.parse_args()

    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.core.columns import Columns

    src = pathlib.Path(args.checkpoint)
    mod_rel = pathlib.Path("learner_group/learner/rl_module/global_policy")
    module = RLModule.from_checkpoint(src / mod_rel)
    module.train()
    n_slots = int(getattr(module, "num_batch_slots", 128))
    n_dc = int(getattr(module, "num_dcs", 8))
    defer_idx = n_dc

    steps, actions, mask = load_dataset(pathlib.Path(args.dataset_dir))
    # drop all-padding steps (empty batches produced NaN val metrics)
    keep = mask.any(axis=1)
    steps = [s for s, k in zip(steps, keep) if k]
    actions, mask = actions[keep], mask[keep]
    weights = build_label_weights(steps, actions, mask, defer_idx)
    kept_holds = int((weights > 0)[(actions == defer_idx)].sum())
    kept_routes = int((weights > 0)[(actions != defer_idx)].sum())
    print(f"label hygiene: kept {kept_holds} hold + {kept_routes} route labels "
          f"(raw hold share was {(actions[mask] == defer_idx).mean():.3f}); "
          f"class mass balanced")
    N = len(steps)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    print(f"dataset: {N} steps ({len(train_idx)} train / {n_val} val), "
          f"defer share of real slots: "
          f"{(actions[mask] == defer_idx).mean():.3f}")

    opt = torch.optim.Adam(module.parameters(), lr=args.lr)
    actions_t = torch.as_tensor(actions)
    mask_t = torch.as_tensor(mask)
    weights_t = torch.as_tensor(weights)
    history = []
    for ep in range(args.epochs):
        rng.shuffle(train_idx)
        run_loss, nb = 0.0, 0
        for s in range(0, len(train_idx), args.batch_steps):
            idx = train_idx[s:s + args.batch_steps]
            batch = {Columns.OBS: collate(steps, idx)}
            out = module.forward_train(batch)
            logits = out[Columns.ACTION_DIST_INPUTS]
            loss = masked_slot_ce(logits, actions_t[idx], weights_t[idx], n_slots)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss); nb += 1
            if nb % 500 == 0:
                print(f"  ep{ep} batch {nb} loss {run_loss/nb:.4f}", flush=True)
        # validation
        module.eval()
        with torch.no_grad():
            v_loss, v_acc, v_prec, v_rec, vb = 0.0, 0.0, 0.0, 0.0, 0
            for s in range(0, len(val_idx), args.batch_steps):
                idx = val_idx[s:s + args.batch_steps]
                out = module.forward_train({Columns.OBS: collate(steps, idx)})
                logits = out[Columns.ACTION_DIST_INPUTS]
                v_loss += float(masked_slot_ce(
                    logits, actions_t[idx], weights_t[idx], n_slots))
                a, p, r = defer_metrics(
                    logits, actions_t[idx], mask_t[idx], n_slots, defer_idx)
                v_acc += a; v_prec += p; v_rec += r; vb += 1
        module.train()
        rec = {"epoch": ep, "train_loss": run_loss / max(1, nb),
               "val_loss": v_loss / vb, "val_acc": v_acc / vb,
               "val_defer_precision": v_prec / vb, "val_defer_recall": v_rec / vb}
        history.append(rec)
        print(f"[epoch {ep}] {json.dumps(rec)}", flush=True)

    # write a probe-loadable checkpoint copy
    out = pathlib.Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    # copy the WHOLE checkpoint (algorithm_state.pkl included): evaluate.py
    # goes through get_checkpoint_info() and rejects a bare rl_module tree
    # (2026-08-16 08:48). Only global_policy's weights are then overwritten.
    shutil.copytree(src, out)
    # RLlib's restore_from_path reads module_state.pt with pickle.load, NOT
    # torch.load - a torch.save archive there crashes every downstream loader
    # (probe + evaluate, 2026-08-16 04:00). Save through RLlib's own protocol.
    module.save_to_path(out / mod_rel)
    for parent in (src, *src.parents):
        pj = parent / "params.json"
        if pj.is_file():
            shutil.copy(pj, out / "params.json")
            break
    if args.metrics_out:
        pathlib.Path(args.metrics_out).write_text(json.dumps(history, indent=1))
    print(f"BC checkpoint -> {out}")


if __name__ == "__main__":
    main()
