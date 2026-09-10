"""Per-module gamma/lambda must reach the advantage computation.

RLlib's GeneralAdvantageEstimation keeps one gamma/lambda for every module, so a global policy
configured with 0.999/0.98 silently ran with the algorithm-level 0.99/0.95.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from src.learners.per_module_gae import ModuleAwareGAE, install_module_aware_gae


class _Cfg:
    """Stands in for AlgorithmConfig.get_config_for_module."""

    def __init__(self, per_module):
        self._per_module = per_module

    def get_config_for_module(self, module_id):
        if module_id not in self._per_module:
            raise KeyError(module_id)
        g, l = self._per_module[module_id]
        return types.SimpleNamespace(gamma=g, lambda_=l)


def test_per_module_values_are_resolved_and_others_fall_back():
    gae = ModuleAwareGAE(gamma=0.99, lambda_=0.95,
                         config=_Cfg({"global_policy": (0.999, 0.98)}))
    assert gae.params_for_module("global_policy") == (0.999, 0.98)
    assert gae.params_for_module("shared_local_policy") == (0.99, 0.95)   # no override
    assert gae.params_for_module("global_policy") == (0.999, 0.98)        # cached, stable


def test_install_replaces_the_stock_connector_and_is_idempotent():
    class _Pipe:
        def __init__(self, conns):
            self.connectors = conns

    stock = GeneralAdvantageEstimation(gamma=0.99, lambda_=0.95)
    pipe = _Pipe([object(), stock, object()])
    assert install_module_aware_gae(pipe, _Cfg({"global_policy": (0.999, 0.98)})) is True
    assert isinstance(pipe.connectors[1], ModuleAwareGAE)
    assert pipe.connectors[1].params_for_module("global_policy") == (0.999, 0.98)
    # running it again must not stack a second replacement
    before = pipe.connectors[1]
    assert install_module_aware_gae(pipe, None) is True
    assert pipe.connectors[1] is before


def test_install_reports_when_there_is_nothing_to_replace():
    class _Pipe:
        def __init__(self):
            self.connectors = [object()]

    assert install_module_aware_gae(_Pipe(), None) is False
    assert install_module_aware_gae(object(), None) is False


def test_a_broken_module_config_falls_back_instead_of_raising():
    class _Bad:
        def get_config_for_module(self, module_id):
            raise RuntimeError("boom")

    gae = ModuleAwareGAE(gamma=0.97, lambda_=0.9, config=_Bad())
    assert gae.params_for_module("anything") == (0.97, 0.9)
