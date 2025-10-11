"""Utilities for tracking event loop backlog and lag health."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, MutableMapping

UTC = timezone.utc


@dataclass(frozen=True)
class BacklogBreach:
    """Represents a backlog breach event."""

    timestamp: datetime
    lag_ms: float


class EventBacklogTracker:
    """Tracks lag between ingestion and processing to detect backlogs."""

    def __init__(self, *, threshold_ms: float = 250.0, window: int = 256) -> None:
        if threshold_ms <= 0:
            raise ValueError("threshold_ms must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self._threshold_ms = float(threshold_ms)
        self._window = int(window)
        self._samples: list[float] = []
        self._breaches: list[BacklogBreach] = []

    @property
    def threshold_ms(self) -> float:
        """Return the configured backlog threshold."""

        return self._threshold_ms

    def record(self, *, lag_ms: float | None, timestamp: datetime | None = None) -> None:
        """Record an observed lag measurement."""

        if lag_ms is None:
            return
        lag_value = max(float(lag_ms), 0.0)
        if timestamp is None:
            timestamp = datetime.now(tz=UTC)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)

        self._samples.append(lag_value)
        if len(self._samples) > self._window:
            self._samples = self._samples[-self._window :]

        if lag_value > self._threshold_ms:
            breach = BacklogBreach(timestamp=timestamp, lag_ms=lag_value)
            self._breaches.append(breach)
            if len(self._breaches) > self._window:
                self._breaches = self._breaches[-self._window :]

    def snapshot(self) -> Mapping[str, object | None]:
        """Return a snapshot describing backlog posture."""

        if not self._samples:
            return {
                "samples": 0,
                "threshold_ms": self._threshold_ms,
                "max_lag_ms": None,
                "avg_lag_ms": None,
                "breaches": 0,
                "healthy": True,
                "last_breach_at": None,
            }

        max_lag = max(self._samples)
        avg_lag = sum(self._samples) / len(self._samples)
        breaches = len(self._breaches)
        last_breach_at = (
            self._breaches[-1].timestamp.astimezone(UTC).isoformat()
            if self._breaches
            else None
        )
        healthy = max_lag <= self._threshold_ms
        snapshot: MutableMapping[str, object | None] = {
            "samples": len(self._samples),
            "threshold_ms": self._threshold_ms,
            "max_lag_ms": max_lag,
            "avg_lag_ms": avg_lag,
            "breaches": breaches,
            "healthy": healthy,
            "last_breach_at": last_breach_at,
        }
        if breaches:
            snapshot["worst_breach_ms"] = max(b.lag_ms for b in self._breaches)
        return snapshot

    def reset(self) -> None:
        """Clear tracked samples and breaches."""

        self._samples.clear()
        self._breaches.clear()
