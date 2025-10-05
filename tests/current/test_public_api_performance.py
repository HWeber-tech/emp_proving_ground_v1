import importlib
import logging
import sys

import pytest


def test_performance_public_api_no_side_effects_and_lazy_vectorized(monkeypatch, caplog):
    # Ensure vectorized module is not pre-imported
    vec_mod = "src.performance.vectorized_indicators"
    if vec_mod in sys.modules:
        del sys.modules[vec_mod]

    caplog.set_level(logging.INFO)
    perf = importlib.import_module("src.performance")

    # No import-time side-effect logs
    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    # Public API surface is explicit
    assert hasattr(perf, "__all__")
    expected = {"MarketDataCache", "VectorizedIndicators", "get_global_cache"}
    assert expected.issubset(set(perf.__all__))

    # Lazy import check: module not loaded until attribute access
    assert vec_mod not in sys.modules
    _ = perf.VectorizedIndicators  # triggers lazy import
    assert vec_mod in sys.modules

    # Removed shims should surface clear errors
    with pytest.raises(AttributeError):
        getattr(perf, "GlobalCache")


def test_get_global_cache_smoke():
    perf = importlib.import_module("src.performance")
    cache = perf.get_global_cache()
    assert cache is not None
