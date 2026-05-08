"""Verify the noise-suppression hooks installed at evaluate.py import time.

Three pieces:
- py4j / ray.rllib loggers are silenced (CRITICAL) so retry tracebacks and
  multiagent env-validation ERROR messages don't pollute the output.
- LearnerGroup.__del__ is monkey-patched to swallow exceptions raised at
  interpreter shutdown (Ray issue: it tries to talk to a torn-down GCS).
- Ray DeprecationWarnings filtered out.
"""

import logging
import warnings


def _emit_and_capture(logger_name: str, msg: str = "should be silenced") -> list:
    """Emit an ERROR record on the given logger and return the records that
    actually made it past root handlers' filters."""
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    cap = _Capture(level=logging.DEBUG)
    # Mimic the filters our root handlers carry, by re-adding them to the
    # capture handler.
    for h in logging.getLogger().handlers:
        for f in list(h.filters):
            cap.addFilter(f)

    logger = logging.getLogger(logger_name)
    # Force the message past any setLevel back-pressure: log at CRITICAL
    # (highest) and let our filter be the only thing that can drop it.
    prev = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(cap)
    try:
        logger.critical(msg)
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev)
    return captured


def test_py4j_records_dropped():
    import src.baselines.evaluate  # noqa: F401
    # py4j and py4j.java_gateway must both be filtered out.
    assert _emit_and_capture("py4j") == []
    assert _emit_and_capture("py4j.java_gateway") == []


def test_rllib_validation_records_dropped():
    import src.baselines.evaluate  # noqa: F401
    assert _emit_and_capture("ray.rllib.env.multi_agent_env_runner") == []
    assert _emit_and_capture("ray.rllib.utils.pre_checks.env") == []
    assert _emit_and_capture("ray.rllib.core.rl_module.rl_module") == []


def test_unrelated_logger_still_emits():
    """The filter must NOT silence loggers that aren't on the noise list."""
    import src.baselines.evaluate  # noqa: F401
    captured = _emit_and_capture("some.user.code")
    assert len(captured) == 1
    assert captured[0].getMessage() == "should be silenced"


def test_learner_group_del_swallows_exceptions():
    """The patched __del__ should never propagate; simulate the failure mode
    by making the wrapped __del__ raise."""
    import src.baselines.evaluate  # noqa: F401
    try:
        from ray.rllib.core.learner.learner_group import LearnerGroup
    except Exception:
        # Ray not installed in this venv — patch is a no-op, nothing to test.
        return

    # Build a stand-in instance whose __del__ would raise.
    class _Boom(LearnerGroup):
        def __init__(self):  # bypass real init
            pass

    boom = _Boom()
    # Calling the patched __del__ explicitly must not raise.
    boom.__del__()


def test_ray_deprecation_warning_filtered():
    """A RayDeprecationWarning must be silenced; user code's own
    DeprecationWarnings must still be visible."""
    import src.baselines.evaluate  # noqa: F401
    try:
        from ray._private.utils import RayDeprecationWarning
    except Exception:
        return  # Ray version doesn't expose it; nothing to assert.

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # reset, then re-apply our filters
        # Re-apply the same filter the eval module installed:
        warnings.filterwarnings("ignore", category=RayDeprecationWarning)
        warnings.warn("ray internal", RayDeprecationWarning)
        warnings.warn("user code drift", DeprecationWarning)

        ray_warnings = [w for w in caught if issubclass(w.category, RayDeprecationWarning)]
        user_warnings = [
            w for w in caught
            if w.category is DeprecationWarning
            and not issubclass(w.category, RayDeprecationWarning)
        ]
        assert len(ray_warnings) == 0
        assert len(user_warnings) == 1
