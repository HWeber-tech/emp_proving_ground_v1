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
    # Block heavy libs at import-time to ensure façade modules stay lightweight
    blocked = {"enabled": True}
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if blocked["enabled"] and (name.startswith("sklearn") or name.startswith("torch")):
            raise ImportError("Blocked heavy import at module import-time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    caplog.set_level(logging.INFO)
    rt_mod = importlib.import_module("src.intelligence.red_team_ai")
    ci_mod = importlib.import_module("src.intelligence.competitive_intelligence")

    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    # Facades should behave as pure re-exports with no legacy shims
    assert not hasattr(rt_mod, "StrategyAnalyzerLegacy")
    assert not hasattr(ci_mod, "AlgorithmFingerprinterLegacy")

    assert rt_mod.StrategyAnalyzer.__module__ == "src.thinking.adversarial.red_team_ai"
    assert rt_mod.RedTeamAI.__module__ == "src.thinking.adversarial.red_team_ai"
    assert ci_mod.CompetitiveIntelligenceSystem.__module__ == (
        "src.thinking.competitive.competitive_intelligence_system"
    )
    assert ci_mod.AlgorithmFingerprinter.__module__ == (
        "src.thinking.competitive.competitive_intelligence_system"
    )

    # Unblock heavy imports and ensure objects are instantiable for sanity
    blocked["enabled"] = False
    _ = rt_mod.StrategyAnalyzer()
    _ = ci_mod.AlgorithmFingerprinter()

    # Explicitly clean up patched imports
    for key in ["sklearn", "sklearn.cluster", "sklearn.ensemble", "torch"]:
        sys.modules.pop(key, None)
