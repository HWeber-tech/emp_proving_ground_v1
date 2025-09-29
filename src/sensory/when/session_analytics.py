"""Session analytics for the WHEN sensory dimension."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["SessionWindow", "SessionAnalytics", "analyse_session", "DEFAULT_CALENDAR"]


@dataclass(frozen=True)
class SessionWindow:
    """Represents a trading session window in UTC."""

    name: str
    start_hour: int
    end_hour: int

    def contains(self, hour: int) -> bool:
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        # Overnight session (wraps around midnight)
        return hour >= self.start_hour or hour < self.end_hour


DEFAULT_CALENDAR: tuple[SessionWindow, ...] = (
    SessionWindow(name="Asia", start_hour=22, end_hour=7),
    SessionWindow(name="London", start_hour=7, end_hour=16),
    SessionWindow(name="NewYork", start_hour=12, end_hour=21),
)


@dataclass(slots=True)
class SessionAnalytics:
    """Computed analytics for a given timestamp relative to session windows."""

    timestamp: pd.Timestamp
    active_sessions: tuple[str, ...]
    next_sessions: tuple[str, ...]
    minutes_to_close: float
    minutes_to_next_open: float
    intensity: float

    def as_metadata(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_sessions": list(self.active_sessions),
            "next_sessions": list(self.next_sessions),
            "minutes_to_close": self.minutes_to_close,
            "minutes_to_next_open": self.minutes_to_next_open,
            "intensity": self.intensity,
        }


def _iter_sessions(calendar: Iterable[SessionWindow]) -> list[SessionWindow]:
    return list(calendar)


def _hours_until(hour: int, target: int) -> int:
    delta = (target - hour) % 24
    return delta


def analyse_session(
    timestamp: datetime | pd.Timestamp,
    calendar: Sequence[SessionWindow] | None = None,
) -> SessionAnalytics:
    """Analyse trading sessions for a given timestamp."""

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)

    calendar_windows = _iter_sessions(calendar or DEFAULT_CALENDAR)
    hour = ts.hour

    active: list[SessionWindow] = [window for window in calendar_windows if window.contains(hour)]
    active_names = tuple(window.name for window in active)

    if len(active_names) >= 2:
        intensity = 1.0
    elif active_names:
        if active_names[0] in {"London", "NewYork"}:
            intensity = 0.75
        else:
            intensity = 0.45
    else:
        intensity = 0.2

    minutes_to_close = float("inf")
    minutes_to_next_open = float("inf")
    distances: list[int] = []

    if active:
        soonest_close = min(_hours_until(hour, window.end_hour) for window in active)
        minutes_to_close = float(soonest_close * 60)

    if active:
        minutes_to_next_open = minutes_to_close
    else:
        distances = [_hours_until(hour, window.start_hour) for window in calendar_windows]
        soonest_open = min(distances) if distances else 0
        minutes_to_next_open = float(soonest_open * 60)

    next_sessions: list[str] = []
    if active:
        # Determine the next session(s) that will be active once current windows close.
        future_hour = (hour + int(minutes_to_close // 60)) % 24
        next_sessions = [window.name for window in calendar_windows if window.contains(future_hour)]
    else:
        if distances:
            soonest = min(distances)
            next_sessions = [
                window.name
                for window, distance in zip(calendar_windows, distances)
                if distance == soonest
            ]

    return SessionAnalytics(
        timestamp=ts,
        active_sessions=active_names,
        next_sessions=tuple(next_sessions),
        minutes_to_close=minutes_to_close,
        minutes_to_next_open=minutes_to_next_open,
        intensity=float(max(0.0, min(1.0, intensity))),
    )
