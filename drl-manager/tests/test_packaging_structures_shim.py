"""Verifies the unpickle shim that lets old RLlib checkpoints (pickled when
packaging<22 was installed) load against the modern packaging package which
no longer ships a `_structures` module.
"""

import pickle
import sys


def test_shim_provides_module_and_singletons():
    # Import side-effect installs the shim; importing twice should be a no-op.
    from src.baselines import load_rllib_model  # noqa: F401
    import packaging._structures as ps  # noqa: F401

    assert hasattr(ps, "Infinity")
    assert hasattr(ps, "NegativeInfinity")
    assert hasattr(ps, "InfinityType")
    assert hasattr(ps, "NegativeInfinityType")


def test_ordering_semantics_match_originals():
    from src.baselines import load_rllib_model  # noqa: F401
    from packaging._structures import Infinity, NegativeInfinity

    # Infinity is greater than any concrete value.
    assert Infinity > 10**12
    assert Infinity >= 10**12
    assert not (Infinity < 0)
    # NegativeInfinity is less than any concrete value.
    assert NegativeInfinity < 0
    assert NegativeInfinity < -(10**12)
    assert -Infinity is NegativeInfinity or repr(-Infinity) == repr(NegativeInfinity)


def test_pickle_roundtrip_through_shim():
    """Simulate the failing path: pickle a payload that references the
    pre-22 _structures module, then unpickle it after our shim is installed."""
    # Construct a synthetic pickle stream that references
    # packaging._structures.Infinity, mirroring what an old checkpoint stored.
    from src.baselines import load_rllib_model  # noqa: F401
    payload = pickle.dumps({"sentinel": sys.modules["packaging._structures"].Infinity})
    restored = pickle.loads(payload)
    # Equality (via isinstance check on the original) is what callers rely on;
    # singleton-identity is not required to load the checkpoint.
    assert restored["sentinel"] == sys.modules["packaging._structures"].Infinity
    assert isinstance(restored["sentinel"],
                      sys.modules["packaging._structures"].InfinityType)


if __name__ == "__main__":
    test_shim_provides_module_and_singletons()
    test_ordering_semantics_match_originals()
    test_pickle_roundtrip_through_shim()
    print("OK")
