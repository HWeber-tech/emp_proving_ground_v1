from __future__ import annotations

from typing import Any, Mapping

from src.core.base import DimensionalReading, MarketRegime
from src.sensory.enhanced._shared import (
    ReadingAdapter,
    build_legacy_payload,
    clamp,
    ensure_market_data,
    safe_timestamp,
)

__all__ = ["ChronalUnderstandingEngine", "ChronalIntelligenceEngine"]


class ChronalUnderstandingEngine:
    def analyze_temporal_understanding(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Assess the temporal context of the market snapshot."""

        market_data = ensure_market_data(data)
        ts = safe_timestamp(market_data)
        minute_of_day = ts.hour * 60 + ts.minute
        weekday = ts.weekday()
        is_weekend = weekday >= 5

        # Model liquidity sessions (rough UTC approximations)
        session_bias = 0.0
        if 7 * 60 <= minute_of_day <= 10 * 60:
            session = "asia-europe-overlap"
            session_bias = 0.2
        elif 12 * 60 <= minute_of_day <= 16 * 60:
            session = "london-newyork-overlap"
            session_bias = 0.35
        elif 16 * 60 < minute_of_day <= 20 * 60:
            session = "us-session"
            session_bias = 0.15
        else:
            session = "off-peak"
            session_bias = -0.05

        weekend_penalty = 0.4 if is_weekend else 0.0
        circadian = clamp((minute_of_day / 720.0) - 1.0, -1.0, 1.0)
        signal_strength = clamp(session_bias + 0.5 * circadian - weekend_penalty, -1.0, 1.0)

        confidence = clamp(
            0.25
            + 0.45 * (1.0 - weekend_penalty)
            + 0.2 * (0.5 + 0.5 * abs(circadian))
            - (0.15 if session == "off-peak" else 0.0),
            0.0,
            1.0,
        )

        regime = MarketRegime.UNKNOWN
        if session == "london-newyork-overlap" and signal_strength > 0.2:
            regime = MarketRegime.BREAKOUT
        elif session_bias > 0.1:
            regime = MarketRegime.TRENDING_WEAK
        elif session == "off-peak":
            regime = MarketRegime.CONSOLIDATING

        context: dict[str, Any] = {
            "source": "sensory.when",
            "minute_of_day": float(minute_of_day),
            "session": session,
            "weekday": weekday,
            "is_weekend": is_weekend,
        }

        reading = DimensionalReading(
            dimension="WHEN",
            signal_strength=float(signal_strength),
            confidence=float(confidence),
            regime=regime,
            context=context,
            data_quality=1.0,
            processing_time_ms=0.0,
            timestamp=ts,
        )
        extras = {"session": session, "minute_of_day": float(minute_of_day)}
        return build_legacy_payload(reading, source="sensory.when", extras=extras)

    def analyze_temporal_intelligence(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Legacy alias maintained for backwards compatibility."""

        return self.analyze_temporal_understanding(data)


ChronalIntelligenceEngine = ChronalUnderstandingEngine
