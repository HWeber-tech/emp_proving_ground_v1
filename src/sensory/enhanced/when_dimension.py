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
from src.sensory.when.session_analytics import (
    SessionAnalytics,
    extract_session_event_flags,
)

__all__ = ["ChronalUnderstandingEngine"]


class ChronalUnderstandingEngine:
    def __init__(self) -> None:
        self._session_analytics = SessionAnalytics()

    def analyze_temporal_understanding(
        self, data: Mapping[str, Any] | Any | None = None
    ) -> ReadingAdapter:
        """Assess the temporal context of the market snapshot."""

        market_data = ensure_market_data(data)
        halted_flag, _ = extract_session_event_flags(dict(vars(market_data)))
        ts = safe_timestamp(market_data)
        minute_of_day = ts.hour * 60 + ts.minute
        weekday = ts.weekday()
        is_weekend = weekday >= 5

        session_snapshot = self._session_analytics.analyse(ts)
        session_token = session_snapshot.session_token
        active_count = len(session_snapshot.active_sessions)

        # Model liquidity sessions (rough UTC approximations)
        base_bias = {
            "Asia": 0.12,
            "London": 0.3,
            "NY": 0.18,
            "auction_open": 0.32,
            "auction_close": 0.24,
            "halt/resume": -0.4,
        }
        session_bias = base_bias.get(session_token, -0.05)
        if active_count >= 2:
            session_bias = max(session_bias, 0.35)

        weekend_penalty = 0.4 if is_weekend else 0.0
        circadian = clamp((minute_of_day / 720.0) - 1.0, -1.0, 1.0)
        signal_strength = clamp(session_bias + 0.5 * circadian - weekend_penalty, -1.0, 1.0)

        confidence = clamp(
            0.25
            + 0.45 * (1.0 - weekend_penalty)
            + 0.2 * (0.5 + 0.5 * abs(circadian))
            - (0.15 if active_count == 0 and session_token != "auction_open" else 0.0),
            0.0,
            1.0,
        )

        regime = MarketRegime.UNKNOWN
        if active_count >= 2 and signal_strength > 0.2:
            regime = MarketRegime.BREAKOUT
        elif session_bias > 0.1 and not halted_flag:
            regime = MarketRegime.TRENDING_WEAK
        elif active_count == 0 and session_token != "auction_open":
            regime = MarketRegime.CONSOLIDATING

        context: dict[str, Any] = {
            "source": "sensory.when",
            "minute_of_day": float(minute_of_day),
            "session": session_token,
            "weekday": weekday,
            "is_weekend": is_weekend,
        }
        if session_snapshot.minutes_to_next_session is not None:
            context["minutes_to_next_session"] = session_snapshot.minutes_to_next_session
        if session_snapshot.minutes_to_session_close is not None:
            context["minutes_to_session_close"] = session_snapshot.minutes_to_session_close

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
        extras = {"session": session_token, "minute_of_day": float(minute_of_day)}
        return build_legacy_payload(reading, source="sensory.when", extras=extras)
