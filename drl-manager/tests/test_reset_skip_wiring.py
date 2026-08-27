"""--reset-skip 必须对**启发式**路径生效(2026-08-27 T3 发现的静默失效)。

症状:三个注册窗口 low/mid/high 跑出**逐位相同**的结果,因为 run_evaluation
从未消费 reset_skip;该参数当时只接在 run_rllib_evaluation 上。
窗口映射:offset_rows = (1009*k) mod green_episode_offset_range。
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.baselines import evaluate as ev


def test_run_evaluation_accepts_reset_skip():
    assert "reset_skip" in inspect.signature(ev.run_evaluation).parameters


def test_run_evaluation_defaults_to_zero_skip():
    assert inspect.signature(ev.run_evaluation).parameters["reset_skip"].default == 0


def test_run_evaluation_body_consumes_reset_skip():
    src = inspect.getsource(ev.run_evaluation)
    assert "reset_skip" in src and "advanced reset counter" in src


def test_cli_passes_reset_skip_to_run_evaluation():
    src = pathlib.Path(ev.__file__).read_text()
    i = src.index("        run_evaluation(")
    assert "reset_skip=" in src[i:i + 800], "CLI 未把 reset_skip 传给 run_evaluation"


def test_registered_window_offsets_match_formula():
    import json
    p = pathlib.Path(__file__).resolve().parents[1] / "calib/p0c_green_windows.json"
    d = json.loads(p.read_text())
    rng = d["green_episode_offset_range"]
    for w in d["windows"]:
        assert (1009 * w["episode_index_k"]) % rng == w["offset_rows"], w["stratum"]


class _StubEnv:
    """最小存根:数 reset 次数,step 立刻终止。"""
    instances = []

    def __init__(self, config):
        self.reset_calls = 0
        self.num_datacenters = 1
        self.global_routing_batch_size = 4
        self.max_vms = 4
        _StubEnv.instances.append(self)

    def reset(self, seed=None, **kw):
        self.reset_calls += 1
        return ({"global": {}, "local": {}}, {})

    def step(self, action):
        return ({"global": {}, "local": {}}, {"global": 0.0, "local": {}}, True, False,
                {"global_energy_stats": {}, "datacenter_energy_metrics": {}})

    def get_local_action_masks(self, dc_id):
        import numpy as np
        return np.ones(4, dtype=bool)

    def close(self):
        pass


def test_reset_skip_actually_advances_the_counter(monkeypatch):
    """T3 静默失效的直接回归:k 次跳过必须产生 k+1 次 reset。
    修复前 run_evaluation 完全忽略 reset_skip,三个窗口跑出逐位相同的结果。"""
    _StubEnv.instances.clear()
    monkeypatch.setattr(ev, "HierarchicalMultiDCEnv", _StubEnv)
    monkeypatch.setitem(ev.GLOBAL_SCHEDULERS, "_stub", _StubGlobal)
    monkeypatch.setitem(ev.LOCAL_SCHEDULERS, "_stub", _StubLocal)
    for skip in (0, 3, 19):
        _StubEnv.instances.clear()
        ev.run_evaluation("_stub", "_stub", {"datacenters": [{}]}, num_episodes=1,
                          seed=1, verbose=False, reset_skip=skip)
        env = _StubEnv.instances[-1]
        assert env.reset_calls == skip + 1, (
            f"reset_skip={skip} 应产生 {skip+1} 次 reset,实际 {env.reset_calls}")


class _StubGlobal:
    def __init__(self, num_dcs, batch_size):
        pass

    def reset(self):
        pass

    def schedule(self, *a, **kw):
        return [0] * 4

    def select_datacenters(self, *a, **kw):
        return [0] * 4


class _StubLocal:
    def __init__(self, *a, **kw):
        pass

    def reset(self):
        pass

    def schedule(self, *a, **kw):
        return 0

    def select_vm(self, *a, **kw):
        return 0
