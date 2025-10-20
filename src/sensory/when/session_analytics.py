"""Session analytics for the WHEN sensory dimension."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from math import isfinite
from numbers import Real
from typing import Iterable, Mapping

import pandas as pd

__all__ = [
    "TradingSession",
    "SessionAnalyticsConfig",
    "SessionSnapshot",
    "SessionAnalytics",
    "normalise_session_tokens",
    "primary_session_token",
    "extract_session_event_flags",
]


_SESSION_NAME_TOKENS = {
    "asia": "Asia",
    "london": "London",
    "new_york": "NY",
}

_SESSION_TOKEN_ORDER: tuple[str, ...] = (
    "Asia",
    "London",
    "NY",
    "auction_open",
    "auction_close",
    "halt",
    "resume",
)

_PRIMARY_SESSION_PRIORITY: tuple[str, ...] = ("London", "NY", "Asia")

_HALT_FLAG_KEYS: tuple[str, ...] = (
    "halted",
    "halt",
    "halt_active",
    "trading_halt",
    "market_halt",
)

_RESUME_FLAG_KEYS: tuple[str, ...] = (
    "halt_resumed",
    "resume",
    "resumed",
    "trading_resume",
)

_HALT_KEYWORDS: frozenset[str] = frozenset(
    {"halt", "halted", "trade_halt", "trading_halt", "halt_active", "market_halt"}
)
_RESUME_KEYWORDS: frozenset[str] = frozenset(
    {"resume", "resumed", "trading_resume", "halt_resumed"}
)


def normalise_session_tokens(candidates: Iterable[str]) -> tuple[str, ...]:
    """Return session tokens in canonical order without duplicates."""

    seen: set[str] = set()
    ordered: list[str] = []

    for token in candidates:
        canonical = str(token)
        if not canonical:
            continue
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)

    prioritised: list[str] = []
    for token in _SESSION_TOKEN_ORDER:
        if token in seen:
            prioritised.append(token)
            seen.remove(token)

    for token in ordered:
        if token in seen:
            prioritised.append(token)
            seen.remove(token)

    return tuple(prioritised)


def primary_session_token(tokens: Iterable[str]) -> str | None:
    """Return the dominant session token when available."""

    collected = tuple(tokens)
    for candidate in _PRIMARY_SESSION_PRIORITY:
        if candidate in collected:
            return candidate
    return None


def _flag_truthy(value: object, *, keywords: frozenset[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, Real):
        numeric = float(value)
        if not isfinite(numeric):
            return False
        return numeric != 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered:
            return False
        if lowered in keywords:
            return True
        return lowered in {"1", "true", "yes", "y", "on"}
    return False


def extract_session_event_flags(payload: Mapping[str, object]) -> tuple[bool, bool]:
    """Detect halt/resume flags from heterogeneous payloads."""

    halted = any(
        _flag_truthy(payload.get(key), keywords=_HALT_KEYWORDS)
        for key in _HALT_FLAG_KEYS
    )
    resumed = any(
        _flag_truthy(payload.get(key), keywords=_RESUME_KEYWORDS)
        for key in _RESUME_FLAG_KEYS
    )
    return halted, resumed


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
        TradingSession("Asia", time(22, 0), time(7, 0)),
        TradingSession("London", time(7, 0), time(16, 0)),
        TradingSession("NY", time(12, 0), time(21, 0)),
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
    session_token: str

    def as_dict(self) -> dict[str, object]:
        return {
            "intensity": float(self.intensity),
            "active_sessions": list(self.active_sessions),
            "upcoming_session": self.upcoming_session,
            "minutes_to_session_close": self.minutes_to_session_close,
            "minutes_to_next_session": self.minutes_to_next_session,
            "session_token": self.session_token,
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
        session_token = self._derive_session_token(
            active_sessions=active_names,
            minutes_to_close=minutes_to_close,
            upcoming_session=upcoming_session,
            minutes_to_next=minutes_to_next,
        )
        return SessionSnapshot(
            intensity=float(min(1.0, intensity)),
            active_sessions=active_names,
            upcoming_session=upcoming_session,
            minutes_to_session_close=minutes_to_close,
            minutes_to_next_session=minutes_to_next,
            session_token=session_token,
        )

    def _derive_session_token(
        self,
        *,
        active_sessions: tuple[str, ...],
        minutes_to_close: float | None,
        upcoming_session: str | None,
        minutes_to_next: float | None,
    ) -> str:
        if active_sessions:
            if (
                minutes_to_close is not None
                and minutes_to_close <= max(self._config.near_close_minutes, 0)
            ):
                return "auction_close"
            return active_sessions[-1]

        if (
            upcoming_session
            and minutes_to_next is not None
            and minutes_to_next <= max(self._config.near_open_minutes, 0)
        ):
            return "auction_open"

        return upcoming_session or "Asia"

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
        ts = timestamp.tz_convert(timezone.utc)
        date = ts.date()
        dt = datetime.combine(date, session.start, tzinfo=timezone.utc)
        start = pd.Timestamp(dt)
        if session.start > session.end and ts.time() < session.start:
            start -= pd.Timedelta(days=1)
        return start

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
