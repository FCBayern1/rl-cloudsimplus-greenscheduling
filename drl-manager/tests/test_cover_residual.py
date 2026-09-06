"""F_FITS_V2 §4: candidate-shared scorer. Zero residual decodes exactly as cover_argmax; the set
loss trains; save / load round-trips; the feature function is one code path."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.baselines.cover_residual import Residual, candidate_features, decode, set_loss, fit, select, save, load, features_from_obs  # noqa: E402
from src.baselines.global_schedulers import cover_argmax_action  # noqa: E402

N, K = 3, 9


def _obs(rng, nb=4):
    return {"cand_green_cover": rng.random((nb, N * K)).astype(np.float32), "batch_cloudlet_offset_allowed": (rng.random((nb, N * K)) > 0.3).astype(np.float32),
            "dc_current_green_power_w": rng.random(N) * 500, "dc_future_short_mean": rng.random(N), "dc_future_long_mean": rng.random(N),
            "dc_utilizations": rng.random(N)}, {"batch_cloudlet_pes": np.full(nb, 32), "batch_cloudlet_mi": np.full(nb, 1920000.0), "batch_cloudlet_time_to_deadline": rng.random(nb) * 800}


def test_zero_residual_decodes_bit_for_bit_like_cover_argmax():
    rng = np.random.default_rng(1)
    obs, pl = _obs(rng)
    X, cov, legal = features_from_obs(obs, pl, 0, N, K)
    m = Residual(X.shape[1])
    assert np.allclose(m.scores_np(X), cov)                                             # score == cover at init
    for slot in range(4):
        X, cov, legal = features_from_obs(obs, pl, slot, N, K)
        assert decode(m.scores_np(X), legal, K) == cover_argmax_action(np.asarray(obs["cand_green_cover"][slot], dtype=np.float64),
                                                                        np.asarray(obs["batch_cloudlet_offset_allowed"][slot], dtype=np.float64), N)
    # exact ties -> smaller kappa then smaller site, outside the argmax
    cov2 = np.zeros(N * K); cov2[[1 * K + 3, 0 * K + 5, 2 * K + 3]] = 0.7
    assert decode(cov2, np.ones(N * K), K) == 1 * K + 3


def test_set_loss_and_fit_reduce_loss_and_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    Xs, Ls, Ts = [], [], []
    for _ in range(40):
        obs, pl = _obs(rng, nb=1)
        X, cov, legal = features_from_obs(obs, pl, 0, N, K)
        # a synthetic target: the legal candidate with the largest kappa_norm * cover (not the cover argmax)
        score_true = cov * X[:, 1] * legal
        tgt = (score_true >= score_true.max() - 1e-12).astype(np.float64) * legal
        Xs.append(X); Ls.append(legal); Ts.append(tgt)
    X = np.stack(Xs); L = np.stack(Ls); T = np.stack(Ts)
    m0 = Residual(X.shape[-1])
    import torch
    l0 = float(set_loss(m0.scores(torch.as_tensor(X)), L, T).mean())
    m, tl, vl = fit(X[:30], L[:30], T[:30], X[30:], L[30:], T[30:], epochs=200, weight_decay=0.0)
    assert tl < l0
    best, table = select(X[:30], L[:30], T[:30], X[30:], L[30:], T[30:], grid_epochs=(20, 60), grid_wd=(0.0,))
    assert len(table["grid"]) == 2 and table["selected"]["epochs"] in (20, 60)
    save(m, str(tmp_path / "m"), meta={"note": "test"})
    m2, meta = load(str(tmp_path / "m"))
    assert np.allclose(m2.scores_np(X[0]), m.scores_np(X[0])) and meta["note"] == "test"
