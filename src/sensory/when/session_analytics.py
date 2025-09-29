"""Session analytics for the WHEN sensory dimension."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Iterable

import pandas as pd

__all__ = ["TradingSession", "SessionAnalyticsConfig", "SessionSnapshot", "SessionAnalytics"]


@dataclass(slots=True, frozen=True)
class TradingSession:
    """Represents a trading session window in UTC time."""

    name: str
    start: time
    end: time

    def contains(self, timestamp: datetime) -> bool:
        ts_time = timestamp.time()
        if self.start <= self.end:
            return self.start <= ts_time < self.end
        # Overnight session (e.g., Asia)
        return ts_time >= self.start or ts_time < self.end


@dataclass(slots=True)
class SessionAnalyticsConfig:
    sessions: tuple[TradingSession, ...] = (
        TradingSession("asia", time(22, 0), time(7, 0)),
        TradingSession("london", time(7, 0), time(16, 0)),
        TradingSession("new_york", time(12, 0), time(21, 0)),
    )

    overlap_bonus: float = 0.3
    near_open_minutes: int = 45
    near_close_minutes: int = 45


@dataclass(slots=True)
class SessionSnapshot:
    """Output of the session analytics transformer."""

    intensity: float
    active_sessions: tuple[str, ...]
    upcoming_session: str | None
    minutes_to_session_close: float | None
    minutes_to_next_session: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "intensity": float(self.intensity),
            "active_sessions": list(self.active_sessions),
            "upcoming_session": self.upcoming_session,
            "minutes_to_session_close": self.minutes_to_session_close,
            "minutes_to_next_session": self.minutes_to_next_session,
        }


class SessionAnalytics:
    """Derive structured information about global trading sessions."""

    def __init__(self, config: SessionAnalyticsConfig | None = None) -> None:
        self._config = config or SessionAnalyticsConfig()

    def analyse(self, timestamp: datetime | pd.Timestamp) -> SessionSnapshot:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)

        active: list[TradingSession] = []
        for session in self._config.sessions:
            if session.contains(ts.to_pydatetime()) and session.name not in {s.name for s in active}:
                active.append(session)

        intensity = 0.1  # Base activity for electronic venues
        if active:
            intensity = 0.6 if len(active) == 1 else 0.85
            if len(active) >= 2:
                intensity = min(1.0, intensity + self._config.overlap_bonus)

        minutes_to_close = self._minutes_to_close(ts, active)

        upcoming_session, minutes_to_next = self._next_session(ts)
        if not active and upcoming_session is not None and minutes_to_next is not None:
            # Build anticipation as we approach the next opening bell.
            anticipation = max(0.0, 1.0 - minutes_to_next / max(self._config.near_open_minutes, 1))
            intensity = max(intensity, 0.3 * anticipation)

        if active and minutes_to_close is not None:
            proximity = max(0.0, 1.0 - minutes_to_close / max(self._config.near_close_minutes, 1))
            intensity = max(intensity, 0.6 + 0.3 * proximity)

        active_names = tuple(session.name for session in active)
        return SessionSnapshot(
            intensity=float(min(1.0, intensity)),
            active_sessions=active_names,
            upcoming_session=upcoming_session,
            minutes_to_session_close=minutes_to_close,
            minutes_to_next_session=minutes_to_next,
        )

    def _minutes_to_close(
        self, timestamp: pd.Timestamp, active: Iterable[TradingSession]
    ) -> float | None:
        if not active:
            return None

        soonest_close: float | None = None
        for session in active:
            end = self._session_end_timestamp(session, timestamp)
            delta = end - timestamp
            minutes = delta.total_seconds() / 60.0
            if minutes < 0:
                minutes = 0.0
            if soonest_close is None or minutes < soonest_close:
                soonest_close = minutes
        return soonest_close

    def _next_session(self, timestamp: pd.Timestamp) -> tuple[str | None, float | None]:
        soonest: tuple[str | None, float | None] = (None, None)
        for session in self._config.sessions:
            start = self._session_start_timestamp(session, timestamp)
            delta = start - timestamp
            minutes = delta.total_seconds() / 60.0
            if minutes < 0:
                # If already started, look at next day's session
                start = self._session_start_timestamp(session, timestamp + pd.Timedelta(days=1))
                delta = start - timestamp
                minutes = delta.total_seconds() / 60.0
            if minutes < 0:
                continue
            if soonest[1] is None or minutes < (soonest[1] or float("inf")):
                soonest = (session.name, minutes)
        return soonest

    def _session_start_timestamp(
        self, session: TradingSession, timestamp: pd.Timestamp
    ) -> pd.Timestamp:
        date = timestamp.tz_convert(timezone.utc).date()
        dt = datetime.combine(date, session.start, tzinfo=timezone.utc)
        return pd.Timestamp(dt)

    def _session_end_timestamp(
        self, session: TradingSession, timestamp: pd.Timestamp
    ) -> pd.Timestamp:
        start = self._session_start_timestamp(session, timestamp)
        end_time = session.end
        if session.start <= session.end:
            dt = datetime.combine(start.date(), end_time, tzinfo=timezone.utc)
        else:
            dt = datetime.combine((start + pd.Timedelta(days=1)).date(), end_time, tzinfo=timezone.utc)
        return pd.Timestamp(dt)

