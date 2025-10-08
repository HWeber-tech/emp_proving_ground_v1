"""
Market GAN System
Generative Adversarial Network for creating challenging market scenarios.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import json
import logging
from datetime import datetime
from typing import Any, List, cast

import numpy as np

from src.core.state_store import StateStore
from src.thinking.adversarial.adversarial_trainer import AdversarialTrainer as AdversarialTrainer
from src.thinking.models.normalizers import normalize_survival_result
from src.thinking.prediction.market_data_generator import MarketDataGenerator as MarketDataGenerator
from src.thinking.prediction.predictive_market_modeler import MarketScenario as MarketScenario

logger = logging.getLogger(__name__)


def _get_strategy_tester() -> Any:
    mod = importlib.import_module("src.trading.strategy_engine.testing.strategy_tester")
    return getattr(mod, "StrategyTester")


class ScenarioValidator:
    """Validates that synthetic scenarios are realistic."""

    def __init__(self) -> None:
        self.validation_rules: dict[str, object] = {
            "volatility_range": (0.001, 0.2),
            "price_range": (0.1, 10.0),
            "trend_consistency": 0.8,
        }

    async def validate_realism(
        self, synthetic_scenarios: List[MarketScenario], real_market_data: dict[str, object]
    ) -> dict[str, object]:
        """Validate that synthetic scenarios are realistic."""
        try:
            valid: int = 0
            invalid: int = 0
            total: int = len(synthetic_scenarios)
            validation_score: float = 0.0

            for scenario in synthetic_scenarios:
                is_valid = await self._validate_scenario(scenario, real_market_data)
                if is_valid:
                    valid += 1
                else:
                    invalid += 1

            if total > 0:
                validation_score = valid / total

            logger.info(f"Scenario validation: {valid}/{total} valid ({validation_score:.2f})")

            return {
                "valid": valid,
                "invalid": invalid,
                "total": total,
                "validation_score": validation_score,
            }

        except Exception as e:
            logger.error(f"Error validating scenarios: {e}")
            return {"valid": 0, "invalid": 0, "total": 0, "validation_score": 0.0}

    async def _validate_scenario(
        self, scenario: MarketScenario, real_market_data: dict[str, object]
    ) -> bool:
        """Validate a single scenario."""
        try:
            # Check volatility range
            min_vol, max_vol = cast(tuple[float, float], self.validation_rules["volatility_range"])
            if not (min_vol <= scenario.volatility <= max_vol):
                return False

            # Check price path validity
            if not scenario.price_path or len(scenario.price_path) < 2:
                return False

            # Check for reasonable price movements
            price_changes = [
                abs(scenario.price_path[i] - scenario.price_path[i - 1])
                / scenario.price_path[i - 1]
                for i in range(1, len(scenario.price_path))
            ]

            max_change = max(price_changes) if price_changes else 0
            if max_change > 0.2:  # More than 20% daily change is unrealistic
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating scenario: {e}")
            return False


class MarketGAN:
    """
    Generative Adversarial Network for training strategies against challenging scenarios.

    Features:
    - Generator creates challenging market scenarios
    - Discriminator tests strategies against scenarios
    - Adversarial training improves both components
    - Realistic scenario validation
    """

    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.generator = MarketDataGenerator()
        StrategyTester = _get_strategy_tester()
        self.discriminator = StrategyTester()
        self.adversarial_trainer = AdversarialTrainer()
        self.scenario_validator = ScenarioValidator()

        self._training_history_key = "emp:gan_training_history"

    async def initialize(self) -> bool:
        return True

    async def stop(self) -> bool:
        return True

    async def train_adversarial_strategies(
        self, strategy_population: List[str], num_epochs: int = 10
    ) -> List[str]:
        """
        Train strategies using adversarial training.

        Args:
            strategy_population: List of strategy IDs to train
            num_epochs: Number of training epochs

        Returns:
            List of improved strategies
        """
        try:
            improved_strategies = []

            for epoch in range(num_epochs):
                logger.info(f"GAN Training Epoch {epoch + 1}/{num_epochs}")

                # Step 1: Generator creates challenging scenarios
                difficulty = self._get_difficulty_for_epoch(epoch, num_epochs)
                synthetic_scenarios = await self.generator.generate_scenarios(
                    difficulty_level=difficulty, num_scenarios=100
                )

                if not synthetic_scenarios:
                    continue

                # Step 2: Discriminator tests strategies
                strategy_payload: List[dict[str, object]] = [{"id": s} for s in strategy_population]
                survival_results = await self.discriminator.test_strategies(
                    strategy_payload, cast(List[object], synthetic_scenarios)
                )

                # Step 3: Train generator to create more challenging scenarios
                norm_sr = [normalize_survival_result(r) for r in survival_results]
                await self.adversarial_trainer.train_generator(
                    self.generator, cast(List[object], norm_sr), target_failure_rate=0.3
                )

                # Step 4: Train discriminator (strategies) to survive
                improved = await self.adversarial_trainer.train_discriminator(
                    cast(List[object], strategy_population),
                    cast(List[object], synthetic_scenarios),
                    cast(List[object], norm_sr),
                )

                improved_strategies.extend([str(x) for x in improved])

                # Step 5: Validate scenarios are realistic
                real_market_data = await self._get_real_market_data()
                validation = await self.scenario_validator.validate_realism(
                    synthetic_scenarios, real_market_data
                )

                # Store training results
                await self._store_training_results(epoch, survival_results, validation, improved)

            logger.info(f"GAN training complete: {len(improved_strategies)} strategies improved")

            return improved_strategies

        except Exception as e:
            logger.error(f"Error in adversarial training: {e}")
            return []

    def _get_difficulty_for_epoch(self, epoch: int, total_epochs: int) -> str:
        """Get difficulty level for training epoch."""
        try:
            progress = epoch / total_epochs

            if progress < 0.25:
                return "easy"
            elif progress < 0.5:
                return "medium"
            elif progress < 0.75:
                return "hard"
            else:
                return "extreme"

        except Exception as e:
            logger.error(f"Error getting difficulty: {e}")
            return "medium"

    async def _get_real_market_data(self) -> dict[str, object]:
        """Get real market data for validation."""
        try:
            # This would be enhanced with actual market data
            return {"volatility": 0.02, "price_range": (90, 110), "trend_consistency": 0.7}
        except Exception as e:
            logger.error(f"Error getting real market data: {e}")
            return {}

    async def _store_training_results(
        self,
        epoch: int,
        survival_results: Sequence[object],
        validation: dict[str, object],
        improved_strategies: Sequence[object],
    ) -> None:
        """Store training results for analysis."""
        try:
            if not survival_results:
                return

            avg_survival = (
                np.mean([normalize_survival_result(r)["survival_rate"] for r in survival_results])
                if survival_results
                else 0.0
            )

            result = {
                "epoch": epoch,
                "timestamp": datetime.utcnow().isoformat(),
                "strategies_tested": len(survival_results),
                "average_survival_rate": avg_survival,
                "validation_score": validation.get("validation_score", 0),
                "strategies_improved": len(improved_strategies),
            }

            key = f"{self._training_history_key}:{epoch}"
            payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
            await self.state_store.set(key, payload, expire=86400 * 30)  # 30 days

        except Exception as e:
            logger.error(f"Error storing training results: {e}")

    async def get_training_stats(self) -> dict[str, object]:
        """Get GAN training statistics."""
        try:
            keys = await self.state_store.keys(f"{self._training_history_key}:*")

            epochs = []
            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        logger.warning("Discarding invalid GAN training history payload", exc_info=exc)
                        continue

                    if not isinstance(parsed, dict):
                        logger.warning(
                            "GAN training history payload must be JSON object; got %s", type(parsed)
                        )
                        continue

                    epochs.append(parsed)

            if not epochs:
                return {
                    "total_epochs": 0,
                    "average_survival_rate": 0.0,
                    "total_strategies_improved": 0,
                    "last_training": None,
                }

            avg_survival = np.mean([e["average_survival_rate"] for e in epochs])
            total_improved = sum([e["strategies_improved"] for e in epochs])

            return {
                "total_epochs": len(epochs),
                "average_survival_rate": avg_survival,
                "total_strategies_improved": total_improved,
                "last_training": max([e["timestamp"] for e in epochs]),
            }

        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return {
                "total_epochs": 0,
                "average_survival_rate": 0.0,
                "total_strategies_improved": 0,
                "last_training": None,
            }
