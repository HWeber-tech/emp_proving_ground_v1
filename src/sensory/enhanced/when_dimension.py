from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.core.base import DimensionalReading, MarketData
from src.sensory.organs.dimensions.base_organ import MarketRegime

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["ChronalIntelligenceEngine"]


class ChronalIntelligenceEngine:
    def analyze_temporal_intelligence(self, data: MarketData) -> DimensionalReading:
        ts = getattr(data, "timestamp", None)
        minute_of_day = 0
        is_weekend = False
        if ts is not None:
            try:
                minute_of_day = int(getattr(ts, "hour", 0)) * 60 + int(getattr(ts, "minute", 0))
                # weekday(): Monday=0 .. Sunday=6
                is_weekend = int(getattr(ts, "weekday", lambda: 0)()) >= 5
            except Exception:
                minute_of_day = 0
                is_weekend = False

        # Normalize minutes since midnight to [-1, 1]
        signal = (minute_of_day / 1440.0) * 2.0 - 1.0 if minute_of_day else 0.0
        confidence = 0.5

        context: Dict[str, Any] = {
            "source": "sensory.when",
            "meta": {"source": "sensory.when"},
            "minute_of_day": float(minute_of_day),
            "is_weekend": bool(is_weekend),
        }

        return DimensionalReading(
            dimension="WHEN",
            signal_strength=float(max(-1.0, min(1.0, signal))),
            confidence=float(max(0.0, min(1.0, confidence))),
            regime=MarketRegime.UNKNOWN,
            context=context,
            data_quality=1.0,
            processing_time_ms=0.0,
        )