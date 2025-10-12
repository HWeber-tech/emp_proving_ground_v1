from __future__ import annotations

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

__all__ = [
    "EnhancedFundamentalUnderstandingEngine",
    "EnhancedFundamentalIntelligenceEngine",
]


class EnhancedFundamentalUnderstandingEngine:
    def analyze_fundamental_understanding(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Generate a fundamentals-oriented reading for the WHY dimension.

        The heuristic intentionally blends momentum, volume participation and
        volatility drag to provide a stable yet reactive score.
        """

        market_data = ensure_market_data(data)
        price_base = abs(float(getattr(market_data, "close", 0.0))) or 1.0
        open_price = float(getattr(market_data, "open", price_base))
        close_price = float(getattr(market_data, "close", open_price))
        volume = float(getattr(market_data, "volume", 0.0))
        volatility = float(getattr(market_data, "volatility", 0.0))
        macro_bias = float(getattr(market_data, "macro_bias", 0.0) or 0.0)
        quality_hint = float(getattr(market_data, "data_quality", 0.8) or 0.8)

        momentum = (close_price - open_price) / price_base
        # Participation saturates using tanh to keep values in [-1, 1]
        participation = tanh(volume / max(1.0, price_base * 750.0))
        volatility_penalty = tanh(max(0.0, volatility) * 6.0)

        raw_signal = (
            0.55 * tanh(momentum * 3.0)
            + 0.30 * participation
            + 0.20 * tanh(macro_bias)
            - 0.25 * volatility_penalty
        )
        signal_strength = clamp(raw_signal, -1.0, 1.0)

        confidence = clamp(
            0.35
            + 0.35 * abs(tanh(momentum * 2.5))
            + 0.20 * (1.0 - volatility_penalty)
            + 0.10 * clamp(quality_hint, 0.0, 1.0),
            0.0,
            1.0,
        )

        regime = MarketRegime.UNKNOWN
        if signal_strength > 0.45:
            regime = MarketRegime.TRENDING_STRONG
        elif signal_strength > 0.15:
            regime = MarketRegime.TRENDING_WEAK
        elif signal_strength < -0.45:
            regime = MarketRegime.REVERSAL
        elif signal_strength < -0.15:
            regime = MarketRegime.EXHAUSTED

        context: dict[str, Any] = {
            "source": "sensory.why",
            "momentum": float(momentum),
            "participation": float(participation),
            "volatility_penalty": float(volatility_penalty),
            "macro_bias": float(macro_bias),
            "quality_hint": float(quality_hint),
        }

        reading = DimensionalReading(
            dimension="WHY",
            signal_strength=float(signal_strength),
            confidence=float(confidence),
            regime=regime,
            context=context,
            data_quality=clamp(quality_hint, 0.0, 1.0),
            processing_time_ms=0.0,
            timestamp=safe_timestamp(market_data),
        )
        extras = {
            "volume": float(volume),
            "momentum": float(momentum),
            "participation": float(participation),
            "volatility_penalty": float(volatility_penalty),
            "macro_bias": float(macro_bias),
        }
        return build_legacy_payload(reading, source="sensory.why", extras=extras)

    # ------------------------------------------------------------------
    # Backwards compatible legacy surface
    # ------------------------------------------------------------------
    def analyze_fundamental_intelligence(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Legacy alias maintained for callers using the intelligence surface."""

        return self.analyze_fundamental_understanding(data)


# Preserve the legacy class name for import stability while the
# understanding-first nomenclature becomes the canonical surface.
EnhancedFundamentalIntelligenceEngine = EnhancedFundamentalUnderstandingEngine
