import importlib
import logging
import sys


def test_intelligence_public_api_no_side_effects_and_lazy(caplog):
    caplog.set_level(logging.INFO)

    intel = importlib.import_module("src.intelligence")

    # No import-time side-effect logs
    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    # Public API surface is explicit
    assert hasattr(intel, "__all__")
    expected = {
        "SentientAdaptationEngine",
        "PredictiveMarketModeler",
        "MarketGAN",
        "AdversarialTrainer",
        "RedTeamAI",
        "SpecializedPredatorEvolution",
        "PortfolioEvolutionEngine",
        "CompetitiveIntelligenceSystem",
        "Phase3IntelligenceOrchestrator",
    }
    assert expected.issubset(set(intel.__all__))

    # dir should include public names
    dir_set = set(dir(intel))
    assert expected.issubset(dir_set)

    # Accessing a symbol should work (lazy load)
    _ = intel.PredictiveMarketModeler  # noqa: F841
    _ = intel.RedTeamAI  # noqa: F841


def test_intelligence_facade_modules_import_without_heavy_deps(caplog, monkeypatch):
    # Block heavy libs at import-time to ensure modules don't try to import them on import
    blocked = {"enabled": True}
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if blocked["enabled"] and (name.startswith("sklearn") or name.startswith("torch")):
            raise ImportError("Blocked heavy import at module import-time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    caplog.set_level(logging.INFO)
    # Import façade modules; should succeed without importing sklearn/torch
    rt_mod = importlib.import_module("src.intelligence.red_team_ai")
    ci_mod = importlib.import_module("src.intelligence.competitive_intelligence")

    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    # Accessing legacy classes should not trigger heavy imports until constructed
    assert hasattr(rt_mod, "StrategyAnalyzerLegacy")
    assert hasattr(ci_mod, "AlgorithmFingerprinterLegacy")

    # Constructing them while blocked should raise ImportError due to localized heavy deps
    try:
        _ = rt_mod.StrategyAnalyzerLegacy()
        raised = False
    except ImportError:
        raised = True
    assert raised, "Expected ImportError due to blocked sklearn during construction"

    try:
        _ = ci_mod.AlgorithmFingerprinterLegacy()
        raised = False
    except ImportError:
        raised = True
    assert raised, "Expected ImportError due to blocked sklearn during construction"

    # Now allow heavy imports with simple stubs to prove the path is executed lazily
    import types

    blocked["enabled"] = False  # stop blocking
    # Provide minimal sklearn stubs
    sklearn = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    ensemble = types.ModuleType("sklearn.ensemble")
    preprocessing = types.ModuleType("sklearn.preprocessing")

    class _DBSCAN:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, X):
            return [0] * len(X)

    class _IsolationForest:
        def __init__(self, *args, **kwargs):
            pass

        def decision_function(self, X):
            return [0.0]

    class _StandardScaler:
        def fit_transform(self, X):
            return X

    cluster.DBSCAN = _DBSCAN
    ensemble.IsolationForest = _IsolationForest
    preprocessing.StandardScaler = _StandardScaler

    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.cluster"] = cluster
    sys.modules["sklearn.ensemble"] = ensemble
    sys.modules["sklearn.preprocessing"] = preprocessing

    # Construct again; should succeed with stubs
    _ = rt_mod.StrategyAnalyzerLegacy()
    _ = ci_mod.AlgorithmFingerprinterLegacy()