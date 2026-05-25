"""
Regression test for the 2026-05-19 plotter fix.

Bug: `_plot_global_reward_breakdown` only knew about the 5 legacy reward
terms (local_sum, carbon_sum, waste_sum, throughput_sum, completion_mi_sum).
In v2 experiments those are all zeroed (alpha/beta/gamma=0), and the actual
signal comes from `global_term_per_action_sum` introduced by Stage 1.  Result:
the plot had 5 flat-at-zero lines and looked broken.

Fix: include `global_term_per_action_sum` in the available columns AND drop
identically-zero columns so the legend isn't cluttered.

Run from drl-manager/:
    .venv/bin/python -m pytest tests/test_auto_plot_reward_breakdown.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.auto_plot import _plot_global_reward_breakdown


def _make_df(per_action_signal=True, legacy_signal=False, n=50):
    """Build a fake monitor.csv-style DataFrame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "episode": np.arange(1, n + 1),
        "global_term_local_sum":        rng.normal(0, 10, n) if legacy_signal else np.zeros(n),
        "global_term_carbon_sum":       rng.normal(0, 10, n) if legacy_signal else np.zeros(n),
        "global_term_waste_sum":        rng.normal(0, 10, n) if legacy_signal else np.zeros(n),
        "global_term_throughput_sum":   rng.normal(0, 10, n) if legacy_signal else np.zeros(n),
        "global_term_completion_mi_sum":rng.normal(0, 10, n) if legacy_signal else np.zeros(n),
        "global_term_per_action_sum":   rng.normal(50, 30, n) if per_action_signal else np.zeros(n),
    })


def test_per_action_column_is_plotted_when_legacy_terms_are_zero(tmp_path):
    """v2 mode: only per-action has signal — plot must include it."""
    df = _make_df(per_action_signal=True, legacy_signal=False)
    out = tmp_path / "plot.png"

    # Spy on ax.plot to see which columns get rendered.
    plotted_labels = []
    real_plot = None

    class SpyAxes:
        def __init__(self, real_ax):
            self._real = real_ax
        def plot(self, *args, **kwargs):
            if "label" in kwargs:
                plotted_labels.append(kwargs["label"])
            return self._real.plot(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._real, name)

    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    def patched_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        return fig, SpyAxes(ax)

    with patch("matplotlib.pyplot.subplots", side_effect=patched_subplots):
        _plot_global_reward_breakdown(df, out)

    assert out.exists(), "plot file was not generated"
    # Must include the per-action term label.
    assert any("per-action" in lbl for lbl in plotted_labels), (
        f"plot is missing per-action term; got labels: {plotted_labels}"
    )
    # Must NOT plot the all-zero legacy terms (clutter prevention).
    legacy_terms = ["α·L̄", "−β·Ĉ", "−γ·R_w", "+k_T·log1pΔMI", "+k_C·Δcompl"]
    plotted_legacy = [t for t in legacy_terms if t in plotted_labels]
    assert not plotted_legacy, (
        f"all-zero legacy terms must be skipped, but got: {plotted_legacy}"
    )


def test_legacy_terms_plotted_when_they_have_signal(tmp_path):
    """v1-style config: legacy terms have signal, per-action might also; both should plot."""
    df = _make_df(per_action_signal=True, legacy_signal=True)
    out = tmp_path / "plot.png"

    plotted_labels = []
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    class SpyAxes:
        def __init__(self, real_ax):
            self._real = real_ax
        def plot(self, *args, **kwargs):
            if "label" in kwargs:
                plotted_labels.append(kwargs["label"])
            return self._real.plot(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._real, name)

    def patched_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        return fig, SpyAxes(ax)

    with patch("matplotlib.pyplot.subplots", side_effect=patched_subplots):
        _plot_global_reward_breakdown(df, out)

    assert out.exists()
    # Both legacy and per-action should appear.
    assert any("α·L̄" in lbl for lbl in plotted_labels), (
        f"legacy local term missing; got {plotted_labels}"
    )
    assert any("per-action" in lbl for lbl in plotted_labels), (
        f"per-action term missing; got {plotted_labels}"
    )


def test_all_zero_produces_no_file(tmp_path):
    """If every term is zero, plotter should bail without writing a stub PNG."""
    df = _make_df(per_action_signal=False, legacy_signal=False)
    out = tmp_path / "plot.png"
    _plot_global_reward_breakdown(df, out)
    assert not out.exists(), "should not emit a plot when every term is zero"


# ---------------------------------------------------------------------------
# 2026-05-23 regression: rewards.png had two bugs
#   1. Looked for column "reward", actual column is "episode_reward".
#   2. Plotted global + local on shared y-axis, so after 2026-05-20 reward
#      redesign (global magnitudes 100× local) local looked flat at zero.
# ---------------------------------------------------------------------------

def test_plot_rewards_uses_correct_column_names(tmp_path):
    """Plotter must reference `episode_reward`, NOT a phantom `reward` column."""
    from src.evaluation.auto_plot import _plot_rewards
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "episode": np.arange(1, n + 1),
        "episode_reward":          rng.normal(-5000, 1000, n),
        "global_agent_reward":     rng.normal(-3000, 800, n),
        "local_agents_avg_reward": rng.normal(-200, 30, n),
    })
    out = tmp_path / "rewards.png"
    plotted_labels = []
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    class SpyAxes:
        def __init__(self, ax): self._a = ax
        def plot(self, *a, **kw):
            if "label" in kw:
                plotted_labels.append(kw["label"])
            return self._a.plot(*a, **kw)
        def __getattr__(self, n):
            return getattr(self._a, n)

    def patched(*a, **kw):
        fig, axes = real_subplots(*a, **kw)
        if hasattr(axes, "__iter__"):
            axes = [SpyAxes(ax) for ax in axes]
        else:
            axes = SpyAxes(axes)
        return fig, axes

    from unittest.mock import patch
    with patch("matplotlib.pyplot.subplots", side_effect=patched):
        _plot_rewards(df, out)

    assert out.exists(), "rewards.png should be written"
    # episode_reward MUST be plotted now (was the silently-dropped column).
    assert any("episode_reward" in lbl for lbl in plotted_labels), (
        f"episode_reward not plotted; got labels: {plotted_labels}"
    )
    # All three should appear since all are present in df.
    assert any("global_agent" in lbl for lbl in plotted_labels)
    assert any("local_avg" in lbl for lbl in plotted_labels)


def test_plot_rewards_uses_separate_subplots_so_scales_dont_crush_each_other(tmp_path):
    """
    The fix uses one subplot per series so each gets its own y-range.
    With global ~ -30k and local ~ -200, a shared axis hid the local line.
    Verify the plotter creates `len(available_series)` axes.
    """
    from src.evaluation.auto_plot import _plot_rewards
    n = 30
    df = pd.DataFrame({
        "episode": np.arange(1, n + 1),
        "episode_reward":          [-5000] * n,
        "global_agent_reward":     [-3000] * n,
        "local_agents_avg_reward": [-200]  * n,
    })
    out = tmp_path / "rewards.png"

    captured = {"nrows": None}
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots
    def patched(nrows=1, *a, **kw):
        captured["nrows"] = nrows
        return real_subplots(nrows, *a, **kw)
    from unittest.mock import patch
    with patch("matplotlib.pyplot.subplots", side_effect=patched):
        _plot_rewards(df, out)

    assert captured["nrows"] == 3, (
        f"expected 3 subplots (one per series), got {captured['nrows']}; "
        "shared-axis plotting is the bug we're guarding against"
    )
