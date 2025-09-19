from __future__ import annotations

import random
from math import tanh
from typing import Any, Mapping

from src.core.base import DimensionalReading, MarketRegime
from src.sensory.enhanced._shared import (
    ReadingAdapter,
    build_legacy_payload,
    clamp,
    ensure_market_data,
    safe_timestamp,
)

__all__ = ["InstitutionalIntelligenceEngine"]


class InstitutionalIntelligenceEngine:
    def analyze_institutional_intelligence(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Estimate institutional participation and liquidity dynamics."""

        market_data = ensure_market_data(data)
        spread = float(getattr(market_data, "spread", 0.0))
        volume = float(getattr(market_data, "volume", 0.0))
        volatility = float(getattr(market_data, "volatility", 0.0))
        depth = float(getattr(market_data, "depth", 0.0))
        imbalance = float(getattr(market_data, "order_imbalance", 0.0))
        quality_hint = float(getattr(market_data, "data_quality", 0.75) or 0.75)

        liquidity = clamp(1.0 - tanh(max(0.0, spread) * 15000.0), 0.0, 1.0)
        participation = tanh(volume / 1200.0 + depth / 5000.0)
        imbalance_score = tanh(imbalance)
        volatility_drag = tanh(max(0.0, volatility) * 4.0)

        institutional_bias = (
            0.5 * participation
            + 0.25 * imbalance_score
            + 0.2 * liquidity
            - 0.15 * volatility_drag
        )
        signal_strength = clamp(institutional_bias, -1.0, 1.0)

        confidence = clamp(
            0.3
            + 0.4 * liquidity
            + 0.2 * abs(imbalance_score)
            + 0.1 * clamp(quality_hint, 0.0, 1.0),
            0.0,
            1.0,
        )

        regime = MarketRegime.UNKNOWN
        if signal_strength > 0.4 and liquidity > 0.6:
            regime = MarketRegime.TRENDING_STRONG
        elif signal_strength > 0.15:
            regime = MarketRegime.TRENDING_WEAK
        elif signal_strength < -0.35:
            regime = MarketRegime.REVERSAL
        elif liquidity < 0.3:
            regime = MarketRegime.CONSOLIDATING

        context: dict[str, Any] = {
            "source": "sensory.how",
            "liquidity": float(liquidity),
            "participation": float(participation),
            "imbalance": float(imbalance_score),
            "volatility_drag": float(volatility_drag),
        }

        # Introduce slight stochasticity to avoid lock-step behaviour when
        # upstream data lacks variation.
        if confidence > 0.6:
            jitter = 0.02 * random.random()
            signal_strength = clamp(signal_strength + jitter, -1.0, 1.0)

        reading = DimensionalReading(
            dimension="HOW",
            signal_strength=float(signal_strength),
            confidence=float(confidence),
            regime=regime,
            context=context,
            data_quality=clamp(quality_hint, 0.0, 1.0),
            processing_time_ms=0.0,
            timestamp=safe_timestamp(market_data),
        )
        extras = {
            "liquidity": float(liquidity),
            "participation": float(participation),
            "imbalance": float(imbalance_score),
            "volatility_drag": float(volatility_drag),
        }
        return build_legacy_payload(reading, source="sensory.how", extras=extras)
