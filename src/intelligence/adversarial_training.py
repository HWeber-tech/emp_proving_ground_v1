#!/usr/bin/env python3
"""
ADVERSARIAL-30: Generative Adversarial Markets
=============================================

Train strategies against intelligent, evolving adversaries using GAN architecture.
Implements intelligent market simulation, adaptive difficulty, stress testing,
and realistic validation for creating robust trading strategies.

This module creates a competitive training environment where strategies
must survive increasingly challenging market conditions.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from src.trading.strategy_engine.testing.strategy_tester import (
    StrategyTester as StrategyTester,
)

logger = logging.getLogger(__name__)


from src.thinking.prediction.predictive_market_modeler import (
    MarketScenario as MarketScenario,
)


@dataclass
class SurvivalResult:
    """Represents strategy survival results."""
    strategy_id: str
    survived: bool
    performance_score: float
    stress_endurance: float
    adaptation_score: float
    failure_reason: Optional[str] = None


# Legacy neural MarketDataGenerator removed; using canonical thinking.prediction.market_data_generator.MarketDataGenerator


# Legacy StrategyTester (nn.Module) removed; using canonical trading.strategy_engine.testing.strategy_tester.StrategyTester


# Legacy StrategyTester implementation removed; using canonical trading.strategy_engine.testing.strategy_tester.StrategyTester
from src.thinking.adversarial.adversarial_trainer import (
    AdversarialTrainer as AdversarialTrainer,
)

# Canonical import for MarketGAN (structural unification)
from src.thinking.adversarial.market_gan import MarketGAN as MarketGAN  # type: ignore

# Canonical import for ScenarioValidator (structural unification)
from src.thinking.adversarial.market_gan import (
    ScenarioValidator as ScenarioValidator,  # type: ignore
)
from src.thinking.prediction.market_data_generator import (
    MarketDataGenerator as MarketDataGenerator,
)


# Example usage and testing
async def test_adversarial_training():
    """Test the adversarial training system."""
    gan = MarketGAN(num_epochs=5)
    
    # Create test strategy population
    strategies = [
        {'id': f'strategy_{i}', 'type': 'momentum', 'risk_tolerance': 0.02}
        for i in range(10)
    ]
    
    # Train adversarially
    trained_strategies = await gan.train_adversarial_strategies(strategies)
    
    print(f"Trained {len(trained_strategies)} strategies")
    stats = gan.get_training_stats()
    print(f"Training stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_adversarial_training())
