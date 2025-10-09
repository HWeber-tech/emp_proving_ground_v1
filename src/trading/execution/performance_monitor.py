"""Monitoring utilities for trading execution throughput and latency."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Deque, Iterable, Mapping

UTC = timezone.utc


@dataclass(frozen=True)
class ThroughputSample:
    """Represents a single execution processing measurement."""

    started_at: datetime
    finished_at: datetime
    duration_ms: float
    lag_ms: float | None


class ThroughputMonitor:
    """Maintains rolling throughput metrics for trade execution."""

    def __init__(self, *, window: int = 256) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._window = window
        self._samples: Deque[ThroughputSample] = deque(maxlen=window)

    def record(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        ingested_at: datetime | None = None,
    ) -> None:
        """Record a processing measurement.

        Args:
            started_at: Wall-clock timestamp when handling began (UTC aware).
            finished_at: Wall-clock timestamp when handling completed (UTC aware).
            ingested_at: Optional timestamp for when the intent entered the system.
        """

        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        else:
            started_at = started_at.astimezone(UTC)

        if finished_at.tzinfo is None:
            finished_at = finished_at.replace(tzinfo=UTC)
        else:
            finished_at = finished_at.astimezone(UTC)

        if finished_at < started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        duration_ms = (finished_at - started_at).total_seconds() * 1000.0

        lag_ms: float | None = None
        if ingested_at is not None:
            if ingested_at.tzinfo is None:
                ingested_at = ingested_at.replace(tzinfo=UTC)
            else:
                ingested_at = ingested_at.astimezone(UTC)
            lag_ms = max((started_at - ingested_at).total_seconds() * 1000.0, 0.0)

        self._samples.append(
            ThroughputSample(
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                lag_ms=lag_ms,
            )
        )

    def snapshot(self) -> Mapping[str, float | int | None]:
        """Return an aggregate snapshot for recent throughput metrics."""

        samples = list(self._samples)
        if not samples:
            return {
                "samples": 0,
                "avg_processing_ms": None,
                "p95_processing_ms": None,
                "max_processing_ms": None,
                "avg_lag_ms": None,
                "max_lag_ms": None,
                "throughput_per_min": None,
            }

        durations = [sample.duration_ms for sample in samples]
        lag_values = [sample.lag_ms for sample in samples if sample.lag_ms is not None]

        throughput_per_min = self._calculate_throughput_per_minute(samples)

        return {
            "samples": len(samples),
            "avg_processing_ms": mean(durations),
            "p95_processing_ms": self._percentile(durations, 95),
            "max_processing_ms": max(durations),
            "avg_lag_ms": mean(lag_values) if lag_values else None,
            "max_lag_ms": max(lag_values) if lag_values else None,
            "throughput_per_min": throughput_per_min,
        }

    def reset(self) -> None:
        """Clear all recorded samples."""

        self._samples.clear()

    def _calculate_throughput_per_minute(
        self, samples: Iterable[ThroughputSample]
    ) -> float | None:
        ordered = list(samples)
        if len(ordered) < 2:
            return None

        start = ordered[0].started_at
        end = ordered[-1].finished_at
        elapsed_seconds = (end - start).total_seconds()
        if elapsed_seconds <= 0:
            return None
        return len(ordered) / (elapsed_seconds / 60.0)

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        values_list = sorted(values)
        if not values_list:
            return 0.0
        if percentile <= 0:
            return values_list[0]
        if percentile >= 100:
            return values_list[-1]
        rank = (percentile / 100.0) * (len(values_list) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(values_list) - 1)
        weight = rank - lower
        return values_list[lower] * (1.0 - weight) + values_list[upper] * weight

