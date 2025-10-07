"""Canonical re-export façade for predictive market modelling components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.thinking.prediction.predictive_market_modeler import (
    BayesianProbabilityEngine,
    ConfidenceCalibrator,
    MarketScenario,
    MarketScenarioGenerator,
    OutcomePredictor,
    PredictiveMarketModeler,
)

__all__ = [
    "PredictiveMarketModeler",
    "MarketScenario",
    "MarketScenarioGenerator",
    "BayesianProbabilityEngine",
    "ConfidenceCalibrator",
    "OutcomePredictor",
    "ScenarioOutcome",
]


@dataclass(slots=True)
class ScenarioOutcome:
    """Represents the predicted outcome for a scenario."""

    expected_return: float
    risk_level: float
    probability: float
    confidence: float
    scenario: Any
