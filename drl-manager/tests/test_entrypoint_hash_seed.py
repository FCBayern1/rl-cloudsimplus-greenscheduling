"""The GTrXL entrypoint must not overwrite a protocol-frozen hash seed."""
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "entrypoint_rlmodule_gtrxl.py"
    spec = importlib.util.spec_from_file_location("entrypoint_rlmodule_gtrxl_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_hash_seed_is_preserved_for_ray_children(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _module().set_seed(20260915)
    assert __import__("os").environ["PYTHONHASHSEED"] == "0"


def test_absent_hash_seed_keeps_historical_run_seed_default(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    _module().set_seed(20260915)
    assert __import__("os").environ["PYTHONHASHSEED"] == "20260915"
