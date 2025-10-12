"""Utilities for tracking event loop backlog and lag health."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Iterable, Mapping, MutableMapping

UTC = timezone.utc


@dataclass(frozen=True)
class BacklogBreach:
    """Represents a backlog breach event."""

    timestamp: datetime
    lag_ms: float


@dataclass(frozen=True)
class BacklogObservation:
    """Structured result returned after recording a backlog sample."""

    timestamp: datetime
    lag_ms: float
    threshold_ms: float
    breach: bool


class EventBacklogTracker:
    """Tracks lag between ingestion and processing to detect backlogs."""

    def __init__(self, *, threshold_ms: float = 250.0, window: int = 256) -> None:
        if threshold_ms <= 0:
            raise ValueError("threshold_ms must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self._threshold_ms = float(threshold_ms)
        self._window = int(window)
        self._samples: Deque[float] = deque()
        self._breaches: Deque[BacklogBreach] = deque()
        self._breach_flags: Deque[bool] = deque()
        self._sample_sum: float = 0.0
        self._max_lag: float = 0.0
        self._worst_breach: float = 0.0
        self._latest_lag: float | None = None

    @property
    def threshold_ms(self) -> float:
        """Return the configured backlog threshold."""

        return self._threshold_ms

    @property
    def window(self) -> int:
        """Return the maximum number of lag samples retained."""

        return self._window

    def record(
        self, *, lag_ms: float | None, timestamp: datetime | None = None
    ) -> BacklogObservation | None:
        """Record an observed lag measurement and return the observation."""

        if lag_ms is None:
            return None

        lag_value = max(float(lag_ms), 0.0)
        if timestamp is None:
            timestamp = datetime.now(tz=UTC)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)

        breach_flag = lag_value > self._threshold_ms

        if len(self._samples) >= self._window:
            removed_sample = self._samples.popleft()
            removed_flag = self._breach_flags.popleft()
            self._sample_sum -= removed_sample
            if removed_sample >= self._max_lag:
                self._max_lag = max(self._samples, default=0.0)
            if removed_flag and not any(self._breach_flags):
                self._worst_breach = max(
                    (entry.lag_ms for entry in self._breaches), default=0.0
                )

        was_empty = not self._samples
        self._samples.append(lag_value)
        self._breach_flags.append(breach_flag)
        self._sample_sum += lag_value
        if was_empty or lag_value > self._max_lag:
            self._max_lag = lag_value
        self._latest_lag = lag_value

        breach_event: BacklogBreach | None = None
        if breach_flag:
            breach_event = BacklogBreach(timestamp=timestamp, lag_ms=lag_value)
            if len(self._breaches) >= self._window:
                removed_breach = self._breaches.popleft()
                if removed_breach.lag_ms >= self._worst_breach:
                    self._worst_breach = max(
                        (entry.lag_ms for entry in self._breaches), default=0.0
                    )
            self._breaches.append(breach_event)
            if lag_value > self._worst_breach:
                self._worst_breach = lag_value

        return BacklogObservation(
            timestamp=timestamp,
            lag_ms=lag_value,
            threshold_ms=self._threshold_ms,
            breach=breach_flag,
        )

    def snapshot(self) -> Mapping[str, object | None]:
        """Return a snapshot describing backlog posture."""

        if not self._samples:
            return {
                "samples": 0,
                "threshold_ms": self._threshold_ms,
                "max_lag_ms": None,
                "avg_lag_ms": None,
                "p95_lag_ms": None,
                "breaches": 0,
                "breach_rate": 0.0,
                "max_breach_streak": 0,
                "latest_lag_ms": None,
                "healthy": True,
                "last_breach_at": None,
            }

        samples = len(self._samples)
        max_lag = self._max_lag if samples else 0.0
        avg_lag = (self._sample_sum / samples) if samples else 0.0
        breach_samples = sum(1 for flag in self._breach_flags if flag)
        breach_rate = (breach_samples / samples) if samples else 0.0
        p95_lag = _percentile(self._samples, 95.0)
        max_streak = _max_streak(self._breach_flags)
        latest_lag = self._latest_lag
        breaches = len(self._breaches)
        last_breach_at = (
            self._breaches[-1].timestamp.astimezone(UTC).isoformat()
            if self._breaches
            else None
        )
        healthy = max_lag <= self._threshold_ms
        snapshot: MutableMapping[str, object | None] = {
            "samples": samples,
            "threshold_ms": self._threshold_ms,
            "max_lag_ms": max_lag,
            "avg_lag_ms": avg_lag,
            "p95_lag_ms": p95_lag,
            "breaches": breaches,
            "breach_rate": breach_rate,
            "max_breach_streak": max_streak,
            "latest_lag_ms": latest_lag,
            "healthy": healthy,
            "last_breach_at": last_breach_at,
        }
        if breaches:
            worst = self._worst_breach
            if worst <= 0.0:
                worst = max((b.lag_ms for b in self._breaches), default=0.0)
                self._worst_breach = worst
            snapshot["worst_breach_ms"] = worst
        return snapshot

    def reset(self) -> None:
        """Clear tracked samples and breaches."""

        self._samples.clear()
        self._breaches.clear()
        self._breach_flags.clear()
        self._sample_sum = 0.0
        self._max_lag = 0.0
        self._worst_breach = 0.0
        self._latest_lag = None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if percentile <= 0.0:
        return ordered[0]
    if percentile >= 100.0:
        return ordered[-1]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _max_streak(flags: Iterable[bool]) -> int:
    max_streak = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak
