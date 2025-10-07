import importlib
import logging
import sys


def test_core_performance_public_api(monkeypatch, caplog):
    vec_mod = "src.core.performance.vectorized_indicators"
    sys.modules.pop(vec_mod, None)

    caplog.set_level(logging.INFO)
    perf = importlib.import_module("src.core.performance")

    joined = "\n".join(record.message for record in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    assert hasattr(perf, "__all__")
    expected = {"MarketDataCache", "VectorizedIndicators", "get_global_cache"}
    assert expected.issubset(set(perf.__all__))

    # Vectorized indicators import on package import; confirm binding matches module symbol
    assert vec_mod in sys.modules
    vector_module = sys.modules[vec_mod]
    assert perf.VectorizedIndicators is vector_module.VectorizedIndicators


def test_get_global_cache_smoke():
    perf = importlib.import_module("src.core.performance")
    cache = perf.get_global_cache()
    assert cache is not None
