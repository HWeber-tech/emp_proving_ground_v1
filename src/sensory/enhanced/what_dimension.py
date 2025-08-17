from __future__ import annotations

from typing import Any, Dict

from src.core.base import DimensionalReading, MarketData
from src.sensory.organs.dimensions.base_organ import MarketRegime

__all__ = ["TechnicalRealityEngine"]


class TechnicalRealityEngine:
    async def analyze_technical_reality(self, data: MarketData) -> DimensionalReading:
        close = float(getattr(data, "close", 0.0))
        open_ = float(getattr(data, "open", close))
        high = float(getattr(data, "high", max(open_, close)))
        low = float(getattr(data, "low", min(open_, close)))
        bid = float(getattr(data, "bid", 0.0))
        ask = float(getattr(data, "ask", 0.0))
        spread = float(ask - bid)

        base = abs(close) if abs(close) > 1e-12 else 1.0
        directional = (close - open_) / base
        signal_strength = max(-1.0, min(1.0, directional))

        # Minimal confidence heuristic using candle range
        rng = max(0.0, high - low)
        confidence = max(0.0, min(1.0, 0.3 + 0.5 * (rng / base)))

        context: Dict[str, Any] = {
            "source": "sensory.what",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "spread": spread,
            "directional": directional,
        }

        return DimensionalReading(
            dimension="WHAT",
            signal_strength=float(signal_strength),
            confidence=float(confidence),
            regime=MarketRegime.UNKNOWN,
            context=context,
            data_quality=1.0,
            processing_time_ms=0.0,
        )