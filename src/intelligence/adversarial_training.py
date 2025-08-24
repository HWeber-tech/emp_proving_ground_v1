#!/usr/bin/env python3
"""
ADVERSARIAL-30: Generative Adversarial Markets
=============================================

Declarative façade that exposes canonical adversarial training APIs while
keeping import-time light. Heavy dependencies (if any) are loaded lazily
on first attribute access to preserve legacy public paths.

This façade preserves:
- AdversarialTrainer
- MarketGAN
- ScenarioValidator
- MarketDataGenerator
- StrategyTester
- MarketScenario

Example/CLI code removed to avoid import-time side effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

# __all__ is computed dynamically from lazy exports and local classes.

_LAZY_EXPORTS: dict[str, str] = {
    "AdversarialTrainer": "src.thinking.adversarial.adversarial_trainer:AdversarialTrainer",
    "MarketGAN": "src.thinking.adversarial.market_gan:MarketGAN",
    "ScenarioValidator": "src.thinking.adversarial.market_gan:ScenarioValidator",
    "MarketDataGenerator": "src.thinking.prediction.market_data_generator:MarketDataGenerator",
    "StrategyTester": "src.trading.strategy_engine.testing.strategy_tester:StrategyTester",
    "MarketScenario": "src.thinking.prediction.predictive_market_modeler:MarketScenario",
}
# __all__ derived from lazy exports and local classes to satisfy Ruff F822 for lazy names.
__all__ = list(_LAZY_EXPORTS.keys()) + ["SurvivalResult"]


def __getattr__(name: str) -> Any:
    # Lazy import to reduce import-time cost; preserves legacy public path.
    target = _LAZY_EXPORTS.get(name)
    if target:
        mod_path, attr = target.split(":")
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


@dataclass
class SurvivalResult:
    """Represents strategy survival results."""

    strategy_id: str
    survived: bool
    performance_score: float
    stress_endurance: float
    adaptation_score: float
    failure_reason: str | None = None
