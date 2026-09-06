"""F_FITS_V2 §4: the candidate-shared scorer  score(j, d, kappa) = cover + residual_theta(x).

Pure pieces (numpy / torch, no env): per-candidate features, the decode with lexicographic ties,
the set loss, fit with validation selection over the frozen grid, save / load. The executed arm
(`cover_residual` in global_schedulers) and the offline fit call the same feature function.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

FEATURE_NAMES = ["cover", "kappa_norm", "site_onehot", "site_green_now", "site_future_short", "site_future_long",
                 "site_util", "job_pes", "job_mi", "job_ttd", "legal"]
SEED = 20260907
GRID_EPOCHS = (50, 100, 200)
GRID_WD = (0.0, 1e-4)
LR = 1e-3
HIDDEN = (64, 64)
SCALE = {"green_w": 1000.0, "pes": 32.0, "mi": 2_000_000.0, "ttd": 1000.0}


def candidate_features(cover_row, mask_row, site_green_now, site_future_short, site_future_long, site_util,
                       job_pes, job_mi, job_ttd, num_dcs, K):
    """(n*K, F) float32 features of every candidate a = d*K + i (kappa = i on the dense grid).
    Site summaries are (n,) arrays as the env publishes them; the job scalars are raw values."""
    n = int(num_dcs)
    cover = np.asarray(cover_row, dtype=np.float64).reshape(n * K)
    legal = np.ones(n * K) if mask_row is None else (np.asarray(mask_row, dtype=np.float64).reshape(n * K) >= 0.5).astype(np.float64)
    d_idx = np.repeat(np.arange(n), K)
    kappa = np.tile(np.arange(K), n) / max(1, K - 1)
    onehot = np.eye(n)[d_idx]
    g = np.asarray(site_green_now, dtype=np.float64).reshape(n)[d_idx] / SCALE["green_w"]
    fs = np.asarray(site_future_short, dtype=np.float64).reshape(n)[d_idx]
    fl = np.asarray(site_future_long, dtype=np.float64).reshape(n)[d_idx]
    u = np.asarray(site_util, dtype=np.float64).reshape(n)[d_idx]
    jp = np.full(n * K, float(job_pes) / SCALE["pes"]); jm = np.full(n * K, float(job_mi) / SCALE["mi"])
    jt = np.full(n * K, min(float(job_ttd), 10 * SCALE["ttd"]) / SCALE["ttd"])
    X = np.column_stack([cover, kappa, onehot, g, fs, fl, u, jp, jm, jt, legal]).astype(np.float32)
    return X, cover, legal


def features_from_obs(obs, planner, slot, num_dcs, K):
    """The online / offline feature function from an observation dict (arm view or dump) and the
    planner channel; identical code path in both uses."""
    cover_all = obs.get("cand_green_cover"); mask_all = obs.get("batch_cloudlet_offset_allowed")
    cover_row = np.zeros(num_dcs * K) if cover_all is None else np.asarray(cover_all)[slot]
    mask_row = None if mask_all is None else np.asarray(mask_all)[slot]
    z = np.zeros(num_dcs)
    return candidate_features(cover_row, mask_row, obs.get("dc_current_green_power_w", z), obs.get("dc_future_short_mean", z),
                              obs.get("dc_future_long_mean", z), obs.get("dc_utilizations", z),
                              float(np.asarray(planner["batch_cloudlet_pes"])[slot]), float(np.asarray(planner["batch_cloudlet_mi"])[slot]),
                              float(np.asarray(planner["batch_cloudlet_time_to_deadline"])[slot]), num_dcs, K)


def decode(score, legal, K):
    """argmax over legal candidates; exact ties broken by smaller kappa, then smaller site."""
    s = np.where(legal >= 0.5, np.asarray(score, dtype=np.float64), -np.inf)
    if not np.isfinite(s).any():
        return 0
    best = s.max()
    cands = np.where(s == best)[0]
    return int(min(cands, key=lambda a: (a % K, a // K)))


class Residual:
    """Candidate-shared MLP; the output layer starts at zero so score == cover at init."""

    def __init__(self, n_features, hidden=HIDDEN, seed=SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        out = nn.Linear(d, 1)
        nn.init.zeros_(out.weight); nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def scores(self, X):
        import torch
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return Xt[..., 0] + self.net(Xt).squeeze(-1)                 # cover + residual

    def scores_np(self, X):
        import torch
        with torch.no_grad():
            return self.scores(X).cpu().numpy()

    def state(self):
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}


def set_loss(scores, legal, target):
    """-log sum_{a in target} softmax(scores over legal)_a, per decision (torch)."""
    import torch
    s = torch.where(torch.as_tensor(legal) >= 0.5, scores, torch.full_like(scores, -1e9))
    logp = torch.log_softmax(s, dim=-1)
    tgt = torch.as_tensor(target) >= 0.5
    return -torch.logsumexp(torch.where(tgt, logp, torch.full_like(logp, -1e9)), dim=-1)


def fit(X_train, legal_train, tgt_train, X_val, legal_val, tgt_val, epochs, weight_decay, seed=SEED, lr=LR):
    """Full-batch Adam on the set loss; returns (model, train_loss, val_loss)."""
    import torch
    n_feat = X_train.shape[-1]
    m = Residual(n_feat, seed=seed)
    opt = torch.optim.Adam(m.net.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = set_loss(m.scores(Xt), legal_train, tgt_train).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tl = float(set_loss(m.scores(Xt), legal_train, tgt_train).mean())
        vl = float(set_loss(m.scores(torch.as_tensor(np.asarray(X_val, dtype=np.float32))), legal_val, tgt_val).mean())
    return m, tl, vl


def select(X_train, legal_train, tgt_train, X_val, legal_val, tgt_val, grid_epochs=GRID_EPOCHS, grid_wd=GRID_WD, seed=SEED):
    """The frozen grid; the model with the lowest validation set loss. Returns (model, table)."""
    best, table = None, []
    for ep in grid_epochs:
        for wd in grid_wd:
            m, tl, vl = fit(X_train, legal_train, tgt_train, X_val, legal_val, tgt_val, ep, wd, seed=seed)
            table.append({"epochs": ep, "weight_decay": wd, "train_loss": tl, "val_loss": vl})
            if best is None or vl < best[0]:
                best = (vl, m, ep, wd)
    return best[1], {"grid": table, "selected": {"epochs": best[2], "weight_decay": best[3], "val_loss": best[0]}}


def save(model, path, meta=None):
    import torch
    os.makedirs(path, exist_ok=True)
    torch.save(model.net.state_dict(), os.path.join(path, "residual.pt"))
    json.dump({"n_features": int(model.net[0].in_features), "hidden": list(HIDDEN), "features": FEATURE_NAMES,
               "scale": SCALE, "seed": SEED, **(meta or {})}, open(os.path.join(path, "meta.json"), "w"), indent=1)


def load(path):
    import torch
    meta = json.load(open(os.path.join(path, "meta.json")))
    m = Residual(int(meta["n_features"]), hidden=tuple(meta.get("hidden", HIDDEN)))
    m.net.load_state_dict(torch.load(os.path.join(path, "residual.pt"), map_location="cpu"))
    m.net.eval()
    return m, meta
