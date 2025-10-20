"""Per-instrument runtime state management helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, MutableMapping


@dataclass(frozen=True, slots=True)
class InstrumentStateEvent:
    """Normalised event metadata used to decide state resets."""

    timestamp: float | None
    session_id: str | None
    gap_seconds: float | None
    halted: bool


@dataclass(slots=True)
class _StateEntry:
    state: Any
    pinned: bool
    expires_at: float | None
    last_session: str | None
    last_timestamp: float | None
    last_update: float


class InstrumentStateTable:
    """Track pinned per-instrument state with TTL and reset semantics."""

    def __init__(
        self,
        *,
        default_ttl_seconds: float | int | None = 900,
        gap_reset_seconds: float | int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if default_ttl_seconds is not None and default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive when provided")
        if gap_reset_seconds is not None and gap_reset_seconds <= 0:
            raise ValueError("gap_reset_seconds must be positive when provided")

        self._default_ttl_seconds = float(default_ttl_seconds) if default_ttl_seconds is not None else None
        self._gap_reset_seconds = float(gap_reset_seconds) if gap_reset_seconds is not None else None
        self._clock = clock or time.monotonic
        self._entries: MutableMapping[str, _StateEntry] = {}

    def get(
        self,
        symbol: str,
        *,
        event: InstrumentStateEvent | Mapping[str, object] | None = None,
    ) -> Any | None:
        entry = self._entries.get(symbol)
        if entry is None:
            return None
        now = self._clock()
        if self._is_expired(entry, now):
            self._entries.pop(symbol, None)
            return None

        candidate_event = self._coerce_event(event)
        if candidate_event and self._should_reset(entry, candidate_event):
            self._entries.pop(symbol, None)
            return None

        self._refresh_metadata(entry, candidate_event, now, refresh_ttl=False)
        return entry.state

    def put(
        self,
        symbol: str,
        state: Any,
        *,
        event: InstrumentStateEvent | Mapping[str, object] | None = None,
        ttl_seconds: float | int | None = None,
        pinned: bool = True,
    ) -> Any:
        now = self._clock()
        candidate_event = self._coerce_event(event)

        entry = self._entries.get(symbol)
        if entry is not None and (self._is_expired(entry, now) or (candidate_event and self._should_reset(entry, candidate_event))):
            self._entries.pop(symbol, None)
            entry = None

        if entry is None:
            entry = _StateEntry(
                state=state,
                pinned=bool(pinned),
                expires_at=None,
                last_session=None,
                last_timestamp=None,
                last_update=now,
            )
            self._entries[symbol] = entry
        else:
            entry.state = state
            entry.pinned = entry.pinned or bool(pinned)

        self._refresh_metadata(entry, candidate_event, now, ttl_override=ttl_seconds, refresh_ttl=True)
        return entry.state

    def pin(
        self,
        symbol: str,
        factory: Callable[[], Any],
        *,
        event: InstrumentStateEvent | Mapping[str, object] | None = None,
        ttl_seconds: float | int | None = None,
    ) -> Any:
        state = self.get(symbol, event=event)
        if state is not None:
            return state
        return self.put(symbol, factory(), event=event, ttl_seconds=ttl_seconds, pinned=True)

    def reset(self, symbol: str) -> bool:
        return self._entries.pop(symbol, None) is not None

    def reset_all(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    def prune_expired(self) -> int:
        now = self._clock()
        removed = 0
        for key in list(self._entries.keys()):
            entry = self._entries[key]
            if self._is_expired(entry, now):
                self._entries.pop(key, None)
                removed += 1
        return removed

    def snapshot(self) -> dict[str, dict[str, Any]]:
        now = self._clock()
        summary: dict[str, dict[str, Any]] = {}
        for symbol, entry in self._entries.items():
            ttl_remaining = entry.expires_at - now if entry.expires_at is not None else None
            summary[symbol] = {
                "pinned": entry.pinned,
                "expires_at": entry.expires_at,
                "ttl_remaining_seconds": ttl_remaining,
                "last_session": entry.last_session,
                "last_timestamp": entry.last_timestamp,
            }
        return summary

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        entry = self._entries.get(symbol)
        if entry is None:
            return False
        if self._is_expired(entry, self._clock()):
            return False
        return True

    def _resolve_ttl(self, ttl_override: float | int | None) -> float | None:
        ttl = ttl_override if ttl_override is not None else self._default_ttl_seconds
        if ttl is None:
            return None
        ttl_float = float(ttl)
        if ttl_float <= 0:
            raise ValueError("ttl_seconds must be positive when provided")
        return ttl_float

    def _is_expired(self, entry: _StateEntry, now: float) -> bool:
        return entry.expires_at is not None and entry.expires_at <= now

    def _should_reset(self, entry: _StateEntry, event: InstrumentStateEvent) -> bool:
        if event.halted:
            return True
        if event.session_id is not None and entry.last_session is not None and event.session_id != entry.last_session:
            return True

        if event.gap_seconds is not None:
            threshold = self._gap_reset_seconds if self._gap_reset_seconds is not None else 0.0
            if event.gap_seconds > 0 and event.gap_seconds >= threshold:
                return True

        if self._gap_reset_seconds is not None and event.timestamp is not None and entry.last_timestamp is not None:
            if event.timestamp - entry.last_timestamp >= self._gap_reset_seconds:
                return True

        return False

    def _refresh_metadata(
        self,
        entry: _StateEntry,
        event: InstrumentStateEvent | None,
        now: float,
        *,
        ttl_override: float | int | None = None,
        refresh_ttl: bool,
    ) -> None:
        if event is not None:
            if event.session_id is not None:
                entry.last_session = event.session_id
            if event.timestamp is not None:
                entry.last_timestamp = event.timestamp

        entry.last_update = now

        if refresh_ttl:
            ttl_seconds = self._resolve_ttl(ttl_override)
            if ttl_seconds is None:
                entry.expires_at = None
            else:
                entry.expires_at = now + ttl_seconds

    def _coerce_event(
        self, event: InstrumentStateEvent | Mapping[str, object] | None
    ) -> InstrumentStateEvent | None:
        if event is None:
            return None
        if isinstance(event, InstrumentStateEvent):
            return event
        if isinstance(event, Mapping):
            timestamp = self._coerce_timestamp(
                event.get("timestamp")
                or event.get("ts")
                or event.get("time")
                or event.get("observed_at")
            )
            session_raw = event.get("session_id") or event.get("session") or event.get("session_name")
            session_id = str(session_raw) if session_raw is not None else None
            gap_raw = event.get("gap_seconds") or event.get("gap") or event.get("gap_secs")
            gap_seconds = self._coerce_float(gap_raw)
            halted = bool(
                event.get("halted")
                or event.get("halt")
                or event.get("halt_active")
                or event.get("trading_halt")
            )
            return InstrumentStateEvent(timestamp=timestamp, session_id=session_id, gap_seconds=gap_seconds, halted=halted)
        raise TypeError("event must be an InstrumentStateEvent, mapping, or None")

    def _coerce_timestamp(self, value: object | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            candidate = value
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=timezone.utc)
            return candidate.timestamp()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(stripped)
                except ValueError:
                    return None
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
        raise TypeError("timestamp must be datetime, float, int, str, or None")

    def _coerce_float(self, value: object | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        raise TypeError("gap_seconds must be numeric or None")


__all__ = ["InstrumentStateTable", "InstrumentStateEvent"]

