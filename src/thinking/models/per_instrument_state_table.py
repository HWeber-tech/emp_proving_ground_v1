from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping

__all__ = [
    "InstrumentStateReset",
    "PerInstrumentStateTable",
]


@dataclass(frozen=True, slots=True)
class InstrumentStateReset:
    """Audit information captured whenever an instrument state is cleared."""

    instrument: str
    reason: str
    at: datetime
    details: Dict[str, object]


@dataclass(slots=True)
class _Entry:
    """Internal container for pinned state and associated metadata."""

    state: object
    expires_at: datetime | None
    session: str | None
    pinned: bool
    last_updated: datetime
    last_event_at: datetime | None

    def expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now


class PerInstrumentStateTable:
    """Manage streaming model state scoped to individual instruments.

    The table keeps on-heap pinned state objects so callers can mutate them in
    place.  Entries honour an optional TTL and automatically reset when trading
    session boundaries, data gaps, or halts are observed.
    """

    DEFAULT_TTL = timedelta(minutes=15)
    DEFAULT_GAP_RESET_THRESHOLD = timedelta(seconds=10)

    def __init__(
        self,
        *,
        default_ttl: timedelta | float | int | None = DEFAULT_TTL,
        gap_reset_threshold: timedelta | float | int | None = DEFAULT_GAP_RESET_THRESHOLD,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or self._utc_now
        self._default_ttl = self._coerce_timedelta(default_ttl)
        self._gap_reset_threshold = self._coerce_timedelta(gap_reset_threshold)
        self._entries: Dict[str, _Entry] = {}
        self._last_reset: Dict[str, InstrumentStateReset] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pin(
        self,
        instrument: str,
        *,
        session: str | None = None,
        ttl: timedelta | float | int | None = None,
        factory: Callable[[], object] | None = None,
    ) -> object:
        """Return a pinned state object for ``instrument`` creating it if needed."""

        now = self._now()
        entry = self._get_entry(instrument, now)
        if entry is not None:
            if session is not None:
                entry.session = session
            entry.last_updated = now
            entry.last_event_at = now
            entry.expires_at = self._compute_expiry(now, ttl)
            return entry.state

        producer = factory or dict
        state = producer()
        new_entry = _Entry(
            state=state,
            expires_at=self._compute_expiry(now, ttl),
            session=session,
            pinned=True,
            last_updated=now,
            last_event_at=now,
        )
        self._entries[instrument] = new_entry
        return state

    def set(
        self,
        instrument: str,
        state: object,
        *,
        session: str | None = None,
        ttl: timedelta | float | int | None = None,
    ) -> None:
        """Store ``state`` for ``instrument`` overwriting any existing entry."""

        now = self._now()
        self._entries[instrument] = _Entry(
            state=state,
            expires_at=self._compute_expiry(now, ttl),
            session=session,
            pinned=False,
            last_updated=now,
            last_event_at=now,
        )

    def get(self, instrument: str) -> object | None:
        """Return the pinned state for ``instrument`` if it exists and is fresh."""

        now = self._now()
        entry = self._get_entry(instrument, now)
        return entry.state if entry is not None else None

    def apply_market_event(
        self,
        instrument: str,
        *,
        session: str | None = None,
        gap_seconds: timedelta | float | int | None = None,
        halted: bool = False,
        gap_reset_threshold: timedelta | float | int | None = None,
        ttl: timedelta | float | int | None = None,
        event_time: datetime | None = None,
    ) -> bool:
        """Refresh metadata and enforce reset rules for the provided event.

        Returns ``True`` when the state was cleared due to the event.
        """

        now = event_time or self._now()
        entry = self._get_entry(instrument, now)
        if entry is None:
            return False

        if halted:
            return self.reset(
                instrument,
                reason="halt",
                when=now,
                details=self._details_with_session(entry),
            )

        duration = self._coerce_timedelta(gap_seconds)
        threshold = (
            self._coerce_timedelta(gap_reset_threshold)
            if gap_reset_threshold is not None
            else self._gap_reset_threshold
        )
        if duration is not None and (threshold is None or duration >= threshold):
            details: Dict[str, object] = {
                "gap_seconds": duration.total_seconds(),
            }
            if threshold is not None:
                details["threshold_seconds"] = threshold.total_seconds()
            return self.reset(instrument, reason="gap", when=now, details=details)

        if session is not None:
            if entry.session is not None and entry.session != session:
                details = {
                    "previous_session": entry.session,
                    "session": session,
                }
                return self.reset(
                    instrument,
                    reason="session_boundary",
                    when=now,
                    details=details,
                )
            entry.session = session

        entry.last_event_at = now
        entry.last_updated = now
        entry.expires_at = self._compute_expiry(now, ttl)
        return False

    def reset(
        self,
        instrument: str,
        *,
        reason: str,
        when: datetime | None = None,
        details: Mapping[str, object] | None = None,
    ) -> bool:
        """Remove state for ``instrument`` recording the provided ``reason``."""

        entry = self._entries.pop(instrument, None)
        if entry is None:
            return False
        timestamp = when or self._now()
        self._record_reset(instrument, reason, timestamp, details)
        return True

    def purge_expired(self, *, now: datetime | None = None) -> int:
        """Remove all entries whose TTL has elapsed."""

        current = now or self._now()
        removed = 0
        for instrument, entry in list(self._entries.items()):
            if entry.expired(current):
                self._expire_entry(instrument, entry, current)
                removed += 1
        return removed

    def last_reset(self, instrument: str) -> InstrumentStateReset | None:
        """Return the most recent reset audit for ``instrument`` if available."""

        return self._last_reset.get(instrument)

    def clear(self) -> None:
        """Clear the entire table without recording individual resets."""

        self._entries.clear()
        self._last_reset.clear()

    def __contains__(self, instrument: str) -> bool:  # pragma: no cover - simple proxy
        return instrument in self._entries and self.get(instrument) is not None

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_states())

    def iter_states(self):
        """Yield ``(instrument, state)`` pairs for non-expired entries."""

        now = self._now()
        for instrument in list(self._entries.keys()):
            entry = self._get_entry(instrument, now)
            if entry is not None:
                yield instrument, entry.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(tz=timezone.utc)

    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _coerce_timedelta(value: timedelta | float | int | None) -> timedelta | None:
        if value is None:
            return None
        if isinstance(value, timedelta):
            return value if value > timedelta(0) else None
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("Timedelta value must be numeric") from exc
        if seconds <= 0:
            return None
        return timedelta(seconds=seconds)

    def _compute_expiry(
        self,
        now: datetime,
        ttl: timedelta | float | int | None,
    ) -> datetime | None:
        delta = self._coerce_timedelta(ttl)
        if delta is None:
            delta = self._default_ttl
        if delta is None:
            return None
        return now + delta

    def _get_entry(self, instrument: str, now: datetime) -> _Entry | None:
        entry = self._entries.get(instrument)
        if entry is None:
            return None
        if entry.expired(now):
            self._expire_entry(instrument, entry, now)
            return None
        return entry

    def _expire_entry(self, instrument: str, entry: _Entry, now: datetime) -> None:
        current = self._entries.get(instrument)
        if current is entry:
            del self._entries[instrument]
        else:  # pragma: no cover - defensive path
            self._entries.pop(instrument, None)
        details: Dict[str, object] = {}
        if entry.expires_at is not None:
            details["expired_at"] = entry.expires_at.isoformat()
        self._record_reset(instrument, "ttl_expired", now, details)

    def _record_reset(
        self,
        instrument: str,
        reason: str,
        when: datetime,
        details: Mapping[str, object] | None,
    ) -> None:
        payload: Dict[str, object] = {}
        if details:
            payload.update({k: v for k, v in details.items() if v is not None})
        self._last_reset[instrument] = InstrumentStateReset(
            instrument=instrument,
            reason=str(reason),
            at=when,
            details=payload,
        )

    @staticmethod
    def _details_with_session(entry: _Entry) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if entry.session is not None:
            payload["session"] = entry.session
        return payload


InstrumentStateTable = PerInstrumentStateTable
"""Alias retained for backwards compatibility during migration."""
