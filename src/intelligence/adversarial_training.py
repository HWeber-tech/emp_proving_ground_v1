"""Canonical re-export façade for adversarial market training components."""

from __future__ import annotations

from dataclasses import dataclass

from src.thinking.adversarial.adversarial_trainer import AdversarialTrainer
from src.thinking.adversarial.market_gan import MarketGAN, ScenarioValidator
from src.thinking.prediction.market_data_generator import MarketDataGenerator
from src.thinking.prediction.predictive_market_modeler import MarketScenario
from src.trading.strategy_engine.testing.strategy_tester import StrategyTester

__all__ = [
    "AdversarialTrainer",
    "MarketGAN",
    "ScenarioValidator",
    "MarketDataGenerator",
    "StrategyTester",
    "MarketScenario",
    "SurvivalResult",
]


@dataclass(slots=True)
class SurvivalResult:
    """Outcome snapshot for adversarial survival evaluations."""

    strategy_id: str
    survived: bool
    performance_score: float
    stress_endurance: float
    adaptation_score: float
    failure_reason: str | None = None
