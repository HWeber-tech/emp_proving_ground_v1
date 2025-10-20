from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Mapping
import copy
from collections.abc import Mapping as MappingABC

try:  # Optional runtime dependency for tensor states
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:  # Optional runtime dependency for tensor states
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

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
    model_version: str | None

    def expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now


def _clone_state_value(value: object) -> object:
    """Return a detached clone suitable for planner what-if simulations."""

    if value is None:
        return None
    if isinstance(value, (int, float, bool, str, bytes, complex)):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if np is not None and isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    clone_method = getattr(value, "clone", None)
    if callable(clone_method):
        try:
            candidate = clone_method()  # type: ignore[call-arg]
        except TypeError:
            candidate = None
        else:
            if candidate is not value:
                return candidate
    if isinstance(value, MappingABC):
        try:
            return value.__class__((key, _clone_state_value(item)) for key, item in value.items())
        except TypeError:
            return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        items = tuple(_clone_state_value(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items
    if isinstance(value, set):
        return {_clone_state_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_clone_state_value(item) for item in value)
    if is_dataclass(value):
        return replace(value)
    copy_method = getattr(value, "copy", None)
    if callable(copy_method):
        try:
            duplicated = copy_method()  # type: ignore[call-arg]
        except TypeError:
            duplicated = None
        else:
            return duplicated
    try:
        return copy.deepcopy(value)
    except Exception as exc:  # pragma: no cover - defensive path for exotic types
        raise TypeError(f"Unsupported state value type for cloning: {type(value)!r}") from exc


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
        state_version: str | None = None,
    ) -> None:
        self._clock = clock or self._utc_now
        self._default_ttl = self._coerce_timedelta(default_ttl)
        self._gap_reset_threshold = self._coerce_timedelta(gap_reset_threshold)
        self._entries: Dict[str, _Entry] = {}
        self._last_reset: Dict[str, InstrumentStateReset] = {}
        self._state_version = self._normalise_model_hash(state_version)

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
        model_hash: str | None = None,
    ) -> object:
        """Return a pinned state object for ``instrument`` creating it if needed."""

        now = self._now()
        target_version = self._resolve_model_hash(model_hash)
        entry = self._get_entry(instrument, now, target_version)
        if entry is not None:
            if session is not None:
                entry.session = session
            entry.last_updated = now
            entry.last_event_at = now
            entry.expires_at = self._compute_expiry(now, ttl)
            entry.model_version = target_version
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
            model_version=target_version,
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
        model_hash: str | None = None,
    ) -> None:
        """Store ``state`` for ``instrument`` overwriting any existing entry."""

        now = self._now()
        target_version = self._resolve_model_hash(model_hash)
        self._entries[instrument] = _Entry(
            state=state,
            expires_at=self._compute_expiry(now, ttl),
            session=session,
            pinned=False,
            last_updated=now,
            last_event_at=now,
            model_version=target_version,
        )

    def get(self, instrument: str, *, model_hash: str | None = None) -> object | None:
        """Return the pinned state for ``instrument`` if it exists and is fresh."""

        now = self._now()
        entry = self._get_entry(instrument, now, self._resolve_model_hash(model_hash))
        return entry.state if entry is not None else None

    def clone_state(
        self,
        instrument: str,
        *,
        model_hash: str | None = None,
        cloner: Callable[[object], object] | None = None,
    ) -> object | None:
        """Return a detached clone of the state for ``instrument``."""

        now = self._now()
        entry = self._get_entry(instrument, now, self._resolve_model_hash(model_hash))
        if entry is None:
            return None
        state = entry.state
        if state is None:
            return None
        clone_fn = cloner or _clone_state_value
        return clone_fn(state)

    def clone_states(
        self,
        instruments: Iterable[str] | None = None,
        *,
        model_hash: str | None = None,
        cloner: Callable[[object], object] | None = None,
    ) -> dict[str, object]:
        """Return detached clones for the provided instrument subset."""

        now = self._now()
        expected_version = self._resolve_model_hash(model_hash)
        clone_fn = cloner or _clone_state_value
        result: dict[str, object] = {}
        if instruments is None:
            candidates = list(self._entries.keys())
        else:
            candidates = list(instruments)
        for instrument in candidates:
            entry = self._get_entry(instrument, now, expected_version)
            if entry is None:
                continue
            state = entry.state
            if state is None:
                continue
            result[instrument] = clone_fn(state)
        return result

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
        model_hash: str | None = None,
    ) -> bool:
        """Refresh metadata and enforce reset rules for the provided event.

        Returns ``True`` when the state was cleared due to the event.
        """

        now = event_time or self._now()
        entry = self._get_entry(instrument, now, self._resolve_model_hash(model_hash))
        if entry is None:
            return False

        if halted:
            details: Dict[str, object] = {"session": "halt/resume"}
            if entry.session is not None:
                details["previous_session"] = entry.session
            return self.reset(
                instrument,
                reason="halt",
                when=now,
                details=details,
            )

        duration = self._coerce_timedelta(gap_seconds)
        threshold = (
            self._coerce_timedelta(gap_reset_threshold)
            if gap_reset_threshold is not None
            else self._gap_reset_threshold
        )
        if duration is not None and (threshold is None or duration >= threshold):
            details: Dict[str, object] = {"gap_seconds": duration.total_seconds()}
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
        entry.model_version = self._resolve_model_hash(model_hash)
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
        payload = self._build_reset_details(entry, base=details)
        self._record_reset(instrument, reason, timestamp, payload)
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

    @property
    def model_hash(self) -> str | None:
        """Return the active model hash used for versioned state tracking."""

        return self._state_version

    def set_model_hash(
        self,
        model_hash: str | None,
        *,
        invalidate: bool = True,
        when: datetime | None = None,
    ) -> int:
        """Update the table-wide model hash, optionally invalidating existing state."""

        new_hash = self._normalise_model_hash(model_hash)
        if new_hash == self._state_version:
            return 0
        cleared = 0
        if invalidate:
            cleared = self.invalidate_for_hot_reload(new_model_hash=new_hash, when=when)
        else:
            self._state_version = new_hash
        return cleared

    def invalidate_for_hot_reload(
        self,
        *,
        new_model_hash: str | None = None,
        when: datetime | None = None,
        reason: str = "hot_reload",
        details: Mapping[str, object] | None = None,
    ) -> int:
        """Invalidate all pinned state due to a hot-reload or model change."""

        now = when or self._now()
        cleared = 0
        next_version = self._normalise_model_hash(new_model_hash)
        for instrument, entry in list(self._entries.items()):
            current = self._entries.pop(instrument, None)
            if current is None:
                continue
            reset_details = self._build_reset_details(current, base=details, next_version=next_version)
            self._record_reset(instrument, reason, now, reset_details)
            cleared += 1
        self._state_version = next_version
        return cleared

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

    def _get_entry(self, instrument: str, now: datetime, expected_version: str | None) -> _Entry | None:
        entry = self._entries.get(instrument)
        if entry is None:
            return None
        if entry.expired(now):
            self._expire_entry(instrument, entry, now)
            return None
        if expected_version is not None and entry.model_version not in (expected_version, None):
            self._reset_for_version_mismatch(instrument, entry, now, expected_version)
            return None
        if expected_version is not None and entry.model_version != expected_version:
            entry.model_version = expected_version
        return entry

    def _expire_entry(self, instrument: str, entry: _Entry, now: datetime) -> None:
        current = self._entries.get(instrument)
        if current is entry:
            del self._entries[instrument]
        else:  # pragma: no cover - defensive path
            self._entries.pop(instrument, None)
        details: Dict[str, object] = {
            "expired_at": entry.expires_at.isoformat() if entry.expires_at is not None else None,
        }
        payload = self._build_reset_details(entry, base=details)
        self._record_reset(instrument, "ttl_expired", now, payload)

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

    @staticmethod
    def _normalise_model_hash(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _resolve_model_hash(self, override: str | None) -> str | None:
        candidate = self._normalise_model_hash(override)
        return self._state_version if candidate is None else candidate

    def _reset_for_version_mismatch(
        self,
        instrument: str,
        entry: _Entry,
        now: datetime,
        expected_version: str | None,
    ) -> None:
        current = self._entries.get(instrument)
        if current is entry:
            del self._entries[instrument]
        else:  # pragma: no cover - defensive path
            self._entries.pop(instrument, None)
        payload = self._build_reset_details(entry, next_version=expected_version)
        self._record_reset(instrument, "version_mismatch", now, payload)

    def _build_reset_details(
        self,
        entry: _Entry,
        *,
        base: Mapping[str, object] | None = None,
        next_version: str | None = None,
    ) -> Dict[str, object]:
        payload = self._details_with_session(entry)
        if base:
            payload.update({k: v for k, v in base.items() if v is not None})
        if entry.model_version is not None:
            payload.setdefault("previous_version", entry.model_version)
        if next_version is not None:
            payload["next_version"] = next_version
        return payload


InstrumentStateTable = PerInstrumentStateTable
"""Alias retained for backwards compatibility during migration."""
