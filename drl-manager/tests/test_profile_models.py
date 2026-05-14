"""Tests for the static-cost profiler.

Loading real RLlib checkpoints in unit tests is too heavy (each load takes
~30-60s + GPU). Instead, test the small pure-Python helpers and the
heuristic path; the RL load path is exercised end-to-end by running the
script itself (manual integration).
"""

import importlib.util
import sys
from pathlib import Path

import torch

_DRL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DRL / "scripts" / "rl"))
sys.path.insert(0, str(_DRL))

spec = importlib.util.spec_from_file_location(
    "profile_models", _DRL / "scripts" / "rl" / "profile_models.py"
)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def test_profile_heuristic_returns_zero_costs():
    row = pm.profile_heuristic()
    assert row["params"] == 0
    assert row["peak_gpu_mb"] == 0.0
    assert row["rss_delta_mb"] == 0.0


def test_count_new_api_params_walks_multi_rl_module():
    """Synthetic MultiRLModule-like container with two sub-nets."""
    class _Sub(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.lin = torch.nn.Linear(in_f, out_f)

    class _MARL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._rl_modules = {
                "global_policy": _Sub(16, 8),
                "shared_local_policy": _Sub(32, 16),
            }
            # Expose children so torch sees them
            for name, m in self._rl_modules.items():
                self.add_module(name, m)

    class _EnvRunner:
        def __init__(self, module):
            self.module = module

    class _Algo:
        def __init__(self):
            self.env_runner = _EnvRunner(_MARL())

    algo = _Algo()
    n = pm._count_new_api_params(algo)
    # 16*8 + 8 (bias) + 32*16 + 16 (bias) = 128+8+512+16 = 664
    assert n == 664


def test_count_old_api_params_walks_policy_map():
    class _Model(torch.nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.lin = torch.nn.Linear(in_f, out_f)

    class _Policy:
        def __init__(self, in_f, out_f):
            self.model = _Model(in_f, out_f)

    class _LocalWorker:
        def __init__(self):
            self.policy_map = {
                "global_policy": _Policy(10, 5),
                "local_policy_0": _Policy(20, 8),
            }

    class _Workers:
        def local_worker(self):
            return _LocalWorker()

    class _Algo:
        def __init__(self):
            self.workers = _Workers()

    algo = _Algo()
    n = pm._count_old_api_params(algo)
    # 10*5+5 + 20*8+8 = 55 + 168 = 223
    assert n == 223


def test_count_old_api_params_falls_back_to_get_weights():
    """When pol.model is None (common for some old-API policies), use
    pol.get_weights() to count parameters."""
    import numpy as np

    class _Policy:
        def __init__(self, sizes):
            self.model = None  # the first probe path returns 0
            self._weights = {f"w{i}": np.zeros(s, dtype=np.float32)
                              for i, s in enumerate(sizes)}

        def get_weights(self):
            return self._weights

    class _LW:
        policy_map = {"p": _Policy([(10, 5), (20,)])}

    class _W:
        def local_worker(self):
            return _LW()

    class _Algo:
        workers = _W()

    n = pm._count_old_api_params(_Algo())
    # 10*5 + 20 = 70
    assert n == 70


def test_count_algo_params_falls_back_when_wrong_flag_passed():
    """If we wrongly say new_api=True on an old-API algo, we should still
    recover the right count via the fallback path."""
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

    class _Pol:
        def __init__(self):
            self.model = _Model()

    class _LW:
        policy_map = {"p": _Pol()}

    class _W:
        def local_worker(self):
            return _LW()

    class _Algo:
        workers = _W()
        env_runner = None

    n = pm.count_algo_params(_Algo(), use_new_api=True)
    assert n == 4 * 4 + 4  # 20


def test_peak_gpu_zero_when_no_cuda(monkeypatch):
    """When CUDA is unavailable, peak GPU must be 0.0, not an error."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pm._peak_gpu_mb() == 0.0
    pm._reset_gpu_peak()  # must be a no-op
