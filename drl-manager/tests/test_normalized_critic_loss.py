"""
P1 critic fix: NormalizedCriticPPOTorchLearner.

The global critic's VALUE_TARGETS have variance ~1e3-1e4, so the raw PPO vf
loss (V−G)² sits at 4000-10000 while vf_clip_param=10 clamps it — clamp's
gradient is zero, so the critic received NO gradient all run (the 20260526
post-mortem). The fix normalizes the vf term by a per-module EMA of
Var(VALUE_TARGETS):

    vf_loss_norm = (V − G)² / max(EMA-Var(G), eps)    # O(1) scale
    clamp(vf_loss_norm, 0, vf_clip_param)             # clip now in σ² units

These tests exercise the REAL compute_loss_for_module via a stub learner
(fake module / metrics / config), the same bypass pattern as
test_crd_compute_loss.py.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_normalized_critic_loss.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from src.learners.normalized_critic_loss import NormalizedCriticPPOTorchLearner


# ---------------------------------------------------------------------------
# Stub plumbing (bypass the heavy Learner build, keep the real loss math)
# ---------------------------------------------------------------------------


class _RecorderMetrics:
    def __init__(self):
        self.logged = {}

    def log_dict(self, d, key=None, window=None):
        self.logged.setdefault(key, {}).update(d)

    def log_value(self, *a, **k):
        pass


class _ConstSchedule:
    def __init__(self, v):
        self._v = v

    def get_current_value(self):
        return self._v


class _FakeModule:
    """Just enough RLModule surface for PPOTorchLearner.compute_loss_for_module."""

    def __init__(self, values, model_config=None):
        self._values = values
        self.model_config = model_config or {}

    def unwrapped(self):
        return self

    def get_train_action_dist_cls(self):
        return TorchCategorical

    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    def compute_values(self, batch, embeddings=None):
        return self._values


class _StubLearner(NormalizedCriticPPOTorchLearner):
    """Skips Learner.__init__; mirrors only the attrs the loss path touches."""

    def __init__(self, values, model_config=None, entropy_coeff=0.0):
        if model_config is None:
            # Most tests exercise the normalized path; the gate itself is
            # covered by the dedicated default-off tests below.
            model_config = {"normalized_critic": {"enabled": True}}
        self._vf_target_var_ema = {}
        self._fake_modules = {"m0": _FakeModule(values, model_config)}
        self.entropy_coeff_schedulers_per_module = {
            "m0": _ConstSchedule(entropy_coeff)
        }
        self.curr_kl_coeffs_per_module = {}
        self.metrics = _RecorderMetrics()

    @property
    def module(self):  # Learner.module is a read-only property
        return self._fake_modules


class _StubBaseLearner(PPOTorchLearner):
    """Same stub but with the UNmodified RLlib loss, for equivalence tests."""

    def __init__(self, values, entropy_coeff=0.0):
        self._fake_modules = {"m0": _FakeModule(values)}
        self.entropy_coeff_schedulers_per_module = {
            "m0": _ConstSchedule(entropy_coeff)
        }
        self.curr_kl_coeffs_per_module = {}
        self.metrics = _RecorderMetrics()

    @property
    def module(self):
        return self._fake_modules


def _cfg(vf_clip=10.0, vf_coeff=1.0):
    return SimpleNamespace(
        use_kl_loss=False,
        clip_param=0.3,
        use_critic=True,
        vf_clip_param=vf_clip,
        vf_loss_coeff=vf_coeff,
    )


def _make_batch(targets, advantages=None, mask=None, n_actions=3):
    """Batch where logp_ratio ≡ 1 (same logits in batch and fwd_out), so the
    surrogate term reduces to the advantages — keeps hand-calc simple."""
    n = len(targets)
    g = torch.Generator().manual_seed(42)
    logits = torch.randn(n, n_actions, generator=g)
    actions = torch.arange(n, dtype=torch.int64) % n_actions
    logp = TorchCategorical.from_logits(logits).logp(actions)
    batch = {
        Columns.ACTION_DIST_INPUTS: logits,
        Columns.ACTIONS: actions,
        Columns.ACTION_LOGP: logp,
        Postprocessing.ADVANTAGES: (
            torch.zeros(n) if advantages is None else torch.tensor(advantages)
        ),
        Postprocessing.VALUE_TARGETS: torch.tensor(targets),
    }
    if mask is not None:
        batch[Columns.LOSS_MASK] = torch.tensor(mask, dtype=torch.bool)
    fwd_out = {Columns.ACTION_DIST_INPUTS: logits}
    return batch, fwd_out


def _loss(learner, batch, fwd_out, cfg=None):
    return learner.compute_loss_for_module(
        module_id="m0", config=cfg or _cfg(), batch=batch, fwd_out=fwd_out
    )


# ---------------------------------------------------------------------------
# _masked_var / EMA bookkeeping
# ---------------------------------------------------------------------------


def test_masked_var_matches_population_variance():
    t = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert NormalizedCriticPPOTorchLearner._masked_var(t, None) == pytest.approx(
        125.0
    )


def test_masked_var_excludes_masked_entries():
    t = torch.tensor([10.0, 1e6, 20.0, 30.0, 40.0])
    m = torch.tensor([True, False, True, True, True])
    assert NormalizedCriticPPOTorchLearner._masked_var(t, m) == pytest.approx(125.0)


def test_ema_initializes_to_first_batch_var():
    learner = _StubLearner(values=torch.zeros(4))
    ema = learner._update_target_var_ema("m0", 125.0, decay=0.99)
    assert ema == pytest.approx(125.0)  # NOT 0.99*0 + 0.01*125


def test_ema_update_formula():
    learner = _StubLearner(values=torch.zeros(4))
    learner._update_target_var_ema("m0", 100.0, decay=0.9)
    ema = learner._update_target_var_ema("m0", 200.0, decay=0.9)
    assert ema == pytest.approx(0.9 * 100.0 + 0.1 * 200.0)


def test_ema_isolated_per_module():
    learner = _StubLearner(values=torch.zeros(4))
    learner._update_target_var_ema("global", 100.0, decay=0.9)
    learner._update_target_var_ema("local", 1.0, decay=0.9)
    assert learner._vf_target_var_ema["global"] == pytest.approx(100.0)
    assert learner._vf_target_var_ema["local"] == pytest.approx(1.0)


def test_ema_state_is_plain_float():
    """Detached running stat — must never carry autograd graph."""
    learner = _StubLearner(values=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    batch, fwd_out = _make_batch([10.0, 20.0, 30.0, 40.0])
    _loss(learner, batch, fwd_out)
    assert isinstance(learner._vf_target_var_ema["m0"], float)


# ---------------------------------------------------------------------------
# Normalized vf loss inside the full PPO loss
# ---------------------------------------------------------------------------


def test_loss_matches_hand_computed():
    targets = [10.0, 20.0, 30.0, 40.0]   # population var = 125
    values = [12.0, 18.0, 33.0, 35.0]    # err² = [4, 4, 9, 25]
    adv = [1.0, -2.0, 0.5, 3.0]
    learner = _StubLearner(values=torch.tensor(values))
    batch, fwd_out = _make_batch(targets, advantages=adv)

    total = _loss(learner, batch, fwd_out, _cfg(vf_clip=10.0, vf_coeff=1.0))

    err2 = torch.tensor([4.0, 4.0, 9.0, 25.0])
    expected = (-torch.tensor(adv)).mean() + (err2 / 125.0).mean()
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)


def test_huge_residuals_no_longer_clipped_gradient_flows():
    """THE failure mode being fixed: residuals² ~4000-10000 with vf_clip=10.
    Raw PPO clamps every sample → zero critic gradient. Normalized loss is
    O(1) → clip doesn't bite → gradient flows."""
    targets = [0.0, 100.0, 200.0, 300.0]          # var = 12500
    values = torch.tensor([80.0, 20.0, 120.0, 220.0], requires_grad=True)
    learner = _StubLearner(values=values)
    batch, fwd_out = _make_batch(targets)

    total = _loss(learner, batch, fwd_out, _cfg(vf_clip=10.0, vf_coeff=1.0))
    total.backward()

    vf_logged = learner.metrics.logged["m0"]["vf_loss"]
    assert vf_logged.item() < 10.0                # well inside the clip
    assert values.grad is not None
    assert torch.all(values.grad != 0.0)          # every sample contributes


def test_clamp_bites_in_sigma_units():
    """With EMA seeded to 1 (decay=1.0 keeps it), a residual² of 10000 is
    10000 σ² — the σ²-unit clamp must cap it at vf_clip and kill its grad."""
    targets = [0.0, 0.0, 0.0, 0.0]
    values = torch.tensor([100.0, 100.0, 100.0, 100.0], requires_grad=True)
    learner = _StubLearner(
        values=values,
        model_config={"normalized_critic": {"enabled": True, "ema_decay": 1.0}},
    )
    learner._vf_target_var_ema["m0"] = 1.0
    batch, fwd_out = _make_batch(targets)

    total = _loss(learner, batch, fwd_out, _cfg(vf_clip=10.0, vf_coeff=1.0))
    total.backward()

    assert learner.metrics.logged["m0"]["vf_loss"].item() == pytest.approx(10.0)
    assert torch.all(values.grad == 0.0)          # fully clipped


def test_mask_excludes_padded_entries_everywhere():
    """Garbage in masked-out slots must touch neither the mean nor the EMA."""
    targets = [10.0, 20.0, 30.0, 40.0, 1e6]
    values = [12.0, 18.0, 33.0, 35.0, 1e6]
    adv = [1.0, -2.0, 0.5, 3.0, 1e6]
    mask = [True, True, True, True, False]
    learner = _StubLearner(values=torch.tensor(values))
    batch, fwd_out = _make_batch(targets, advantages=adv, mask=mask)

    total = _loss(learner, batch, fwd_out, _cfg(vf_clip=10.0, vf_coeff=1.0))

    assert learner._vf_target_var_ema["m0"] == pytest.approx(125.0)
    err2 = torch.tensor([4.0, 4.0, 9.0, 25.0])
    expected = (-torch.tensor(adv[:4])).mean() + (err2 / 125.0).mean()
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)


def test_eps_floor_keeps_loss_finite_on_constant_targets():
    targets = [5.0, 5.0, 5.0, 5.0]                # var = 0 → denom = eps
    learner = _StubLearner(values=torch.tensor([6.0, 6.0, 6.0, 6.0]))
    batch, fwd_out = _make_batch(targets)

    total = _loss(learner, batch, fwd_out)

    assert torch.isfinite(total)


def test_equivalent_to_base_ppo_when_variance_is_one():
    """With EMA pinned to 1.0 the normalization is the identity — the loss
    must equal RLlib's unmodified PPOTorchLearner loss bit-for-bit."""
    targets = [1.0, 2.0, 3.0, 4.0]
    values = [1.5, 1.5, 3.5, 3.5]
    adv = [1.0, -1.0, 2.0, 0.5]
    norm = _StubLearner(
        values=torch.tensor(values),
        model_config={"normalized_critic": {"enabled": True, "ema_decay": 1.0}},
    )
    norm._vf_target_var_ema["m0"] = 1.0
    base = _StubBaseLearner(values=torch.tensor(values))

    batch_n, fwd_n = _make_batch(targets, advantages=adv)
    batch_b, fwd_b = _make_batch(targets, advantages=adv)
    cfg = _cfg(vf_clip=10.0, vf_coeff=1.0)

    t_norm = norm.compute_loss_for_module(
        module_id="m0", config=cfg, batch=batch_n, fwd_out=fwd_n
    )
    t_base = base.compute_loss_for_module(
        module_id="m0", config=cfg, batch=batch_b, fwd_out=fwd_b
    )
    assert t_norm.item() == pytest.approx(t_base.item(), rel=1e-6)


def test_default_off_matches_base_ppo_exactly():
    """Without a `normalized_critic.enabled: true` in model_config the vf
    term must be bit-identical to base PPO — including the pathological
    huge-residual clipping. This is what keeps CRDPPOTorchLearner (which now
    inherits this class) backward-compatible with pre-P1 configs."""
    targets = [0.0, 100.0, 200.0, 300.0]
    values = [80.0, 20.0, 120.0, 220.0]   # err² = 6400 each — clip bites
    adv = [1.0, -1.0, 2.0, 0.5]
    off = _StubLearner(values=torch.tensor(values), model_config={})
    base = _StubBaseLearner(values=torch.tensor(values))

    batch_o, fwd_o = _make_batch(targets, advantages=adv)
    batch_b, fwd_b = _make_batch(targets, advantages=adv)
    cfg = _cfg(vf_clip=10.0, vf_coeff=1.0)

    t_off = off.compute_loss_for_module(
        module_id="m0", config=cfg, batch=batch_o, fwd_out=fwd_o
    )
    t_base = base.compute_loss_for_module(
        module_id="m0", config=cfg, batch=batch_b, fwd_out=fwd_b
    )
    assert t_off.item() == pytest.approx(t_base.item(), rel=1e-6)
    assert off._vf_target_var_ema == {}   # EMA untouched when gated off
    # Gate-off must not emit the normalization diagnostics — a 0.0 there
    # would read as "target variance collapsed" on dashboards.
    assert "vf_target_var_ema" not in off.metrics.logged["m0"]


def test_metrics_expose_var_ema_and_raw_mse():
    """Normalization must not hide the absolute critic error: the raw MSE and
    the EMA denominator are logged alongside the (normalized) vf_loss."""
    targets = [10.0, 20.0, 30.0, 40.0]
    learner = _StubLearner(values=torch.tensor([12.0, 18.0, 33.0, 35.0]))
    batch, fwd_out = _make_batch(targets)

    _loss(learner, batch, fwd_out)

    logged = learner.metrics.logged["m0"]
    assert logged["vf_target_var_ema"] == pytest.approx(125.0)
    assert float(logged["vf_loss_raw_mse"]) == pytest.approx(10.5)  # (4+4+9+25)/4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
