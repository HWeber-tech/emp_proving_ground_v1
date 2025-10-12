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


def test_intelligence_facade_resolves_canonical_symbols(caplog):
    caplog.set_level(logging.INFO)

    intel = importlib.import_module("src.intelligence")

    joined = "\n".join(r.message for r in caplog.records)
    assert "Starting" not in joined
    assert "Configured logging" not in joined

    canonical_map = {
        "SentientAdaptationEngine": "src.sentient.adaptation.sentient_adaptation_engine",
        "RedTeamAI": "src.thinking.adversarial.red_team_ai",
        "AdversarialTrainer": "src.thinking.adversarial.adversarial_trainer",
        "MarketGAN": "src.thinking.adversarial.market_gan",
        "PredictiveMarketModeler": "src.thinking.prediction.predictive_market_modeler",
        "CompetitiveIntelligenceSystem": "src.thinking.competitive.competitive_intelligence_system",
        "SpecializedPredatorEvolution": "src.ecosystem.evolution.specialized_predator_evolution",
    }

    for name, module_name in canonical_map.items():
        attr = getattr(intel, name)
        assert getattr(attr, "__module__", module_name) == module_name

    portfolio_engine = getattr(intel, "PortfolioEvolutionEngine")
    assert portfolio_engine.__module__ == "src.intelligence"

    legacy_modules = [
        (
            "src.intelligence.red_team_ai",
            "src.thinking.adversarial.red_team_ai",
        ),
        (
            "src.intelligence.competitive_intelligence",
            "src.thinking.competitive.competitive_intelligence_system",
        ),
        (
            "src.intelligence.predictive_modeling",
            "src.thinking.prediction.predictive_market_modeler",
        ),
        (
            "src.intelligence.specialized_predators",
            "src.ecosystem",
        ),
    ]

    for module_name, expected_fragment in legacy_modules:
        with pytest.raises(ModuleNotFoundError) as excinfo:
            importlib.import_module(module_name)

        assert module_name in str(excinfo.value)
        assert expected_fragment in str(excinfo.value)
