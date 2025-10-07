import importlib
import logging

import pytest


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


def test_intelligence_facade_modules_reexport_canonical_symbols(caplog):
    caplog.set_level(logging.INFO)

    rt_mod = importlib.import_module("src.intelligence.red_team_ai")
    ci_mod = importlib.import_module("src.intelligence.competitive_intelligence")

    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    red_team_symbols = {
        "StrategyAnalyzer": "src.thinking.adversarial.red_team_ai",
        "WeaknessDetector": "src.thinking.adversarial.red_team_ai",
        "AttackGenerator": "src.thinking.adversarial.red_team_ai",
        "ExploitDeveloper": "src.thinking.adversarial.red_team_ai",
        "RedTeamAI": "src.thinking.adversarial.red_team_ai",
    }
    for name, module_name in red_team_symbols.items():
        attr = getattr(rt_mod, name)
        assert attr.__module__ == module_name

    with pytest.raises(AttributeError, match="removed"):
        getattr(rt_mod, "StrategyAnalyzerLegacy")

    comp_symbols = {
        "AlgorithmFingerprinter": "src.thinking.competitive.competitive_intelligence_system",
        "BehaviorAnalyzer": "src.thinking.competitive.competitive_intelligence_system",
        "CompetitiveIntelligenceSystem": "src.thinking.competitive.competitive_intelligence_system",
        "CounterStrategyDeveloper": "src.thinking.competitive.competitive_intelligence_system",
        "MarketShareTracker": "src.thinking.competitive.competitive_intelligence_system",
    }
    for name, module_name in comp_symbols.items():
        attr = getattr(ci_mod, name)
        assert attr.__module__ == module_name

    with pytest.raises(AttributeError, match="removed"):
        getattr(ci_mod, "AlgorithmFingerprinterLegacy")
