#!/usr/bin/env python3
"""
Canonical MarketDataGenerator for scenario creation (thinking/prediction).

This generator produces canonical MarketScenario instances defined in:
- src.thinking.prediction.predictive_market_modeler.MarketScenario

It preserves a simple signature compatible with prior usages:
  generate_scenarios(difficulty_level, target_strategies, num_scenarios=100)
Where:
- difficulty_level can be a string: {"easy","medium","hard","extreme"} or a float in [0,1]
- target_strategies is accepted for backward-compat but not required by the canonical shape
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import numpy as np

from src.thinking.prediction.predictive_market_modeler import (
    MarketScenario as MarketScenario,
)


class MarketDataGenerator:
    """
    Canonical scenario generator that emits MarketScenario objects with fields:
      scenario_id, timestamp, scenario_type, probability, price_path,
      volatility, direction_bias, confidence
    """

    def __init__(self) -> None:
        # String difficulty presets mapped to (volatility, noise)
        self._presets: Dict[str, Tuple[float, float]] = {
            "easy": (0.01, 0.02),
            "medium": (0.02, 0.04),
            "hard": (0.04, 0.08),
            "extreme": (0.08, 0.15),
        }

    async def generate_scenarios(
        self,
        difficulty_level: Any,
        num_scenarios: int = 100,
    ) -> List[MarketScenario]:
        """
        Generate canonical market scenarios.

        Args:
          difficulty_level: str {"easy","medium","hard","extreme"} or float in [0,1]
          target_strategies: retained for compatibility (not used by canonical type)
          num_scenarios: number of scenarios to generate

        Returns:
          List[MarketScenario] (canonical)
        """
        vol, noise = self._difficulty_params(difficulty_level)

        scenarios: List[MarketScenario] = []
        for _ in range(max(0, num_scenarios)):
            price_path = self._generate_price_path(
                base_price=100.0,
                volatility=vol,
                noise_level=noise,
                days=30,
            )

            scenario = MarketScenario(
                scenario_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                scenario_type="generated",
                probability=1.0,  # canonical simple prior
                price_path=price_path,
                volatility=vol,
                direction_bias=0.0,
                confidence=Decimal("0.8"),
            )
            scenarios.append(scenario)

        return scenarios

    def _difficulty_params(self, difficulty_level: Any) -> Tuple[float, float]:
        """
        Resolve difficulty into (volatility, noise) tuple.
        Accepts preset strings or float in [0,1].
        """
        if isinstance(difficulty_level, str):
            key = difficulty_level.strip().lower()
            if key in self._presets:
                return self._presets[key]
            # Fallback unknown strings to medium
            return self._presets["medium"]

        # Numeric difficulty in [0,1] → interpolate between easy and extreme
        try:
            x = float(difficulty_level)
        except Exception:
            x = 0.5
        x = max(0.0, min(1.0, x))

        easy_vol, easy_noise = self._presets["easy"]
        extreme_vol, extreme_noise = self._presets["extreme"]

        vol = easy_vol + (extreme_vol - easy_vol) * x
        noise = easy_noise + (extreme_noise - easy_noise) * x
        return vol, noise

    def _generate_price_path(
        self,
        base_price: float,
        volatility: float,
        noise_level: float,
        days: int = 30,
    ) -> List[float]:
        """
        Generate a price path with regime shifts and noise to stress strategies.
        """
        prices = [base_price]
        for day in range(days):
            # Regime shift every ~week
            if day % 7 == 0:
                regime_change = np.random.normal(0.0, volatility * 3.0)
            else:
                regime_change = 0.0

            noise = np.random.normal(0.0, noise_level)
            drift = regime_change + noise
            new_price = prices[-1] * (1.0 + drift)
            new_price = max(new_price, base_price * 0.5)  # keep positive
            prices.append(float(new_price))
        return prices