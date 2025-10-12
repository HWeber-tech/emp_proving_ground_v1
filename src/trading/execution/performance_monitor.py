"""Monitoring utilities for trading execution throughput and latency."""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Deque, Iterable, Mapping, Sequence

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
        self._window = int(window)
        self._samples: Deque[ThroughputSample] = deque()
        self._durations_sorted: list[float] = []
        self._duration_sum: float = 0.0
        self._lag_values_sorted: list[float] = []
        self._lag_sum: float = 0.0
        self._lag_count: int = 0
        self._last_snapshot: dict[str, float | int | None] | None = None
        self._dirty = True

    @property
    def window(self) -> int:
        """Return the configured rolling window length."""

        return self._window

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

        sample = ThroughputSample(
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            lag_ms=lag_ms,
        )

        if len(self._samples) == self._window:
            removed = self._samples.popleft()
            self._remove_sample_stats(removed)

        self._samples.append(sample)
        self._add_sample_stats(sample)
        self._dirty = True

    def snapshot(self) -> Mapping[str, float | int | None]:
        """Return an aggregate snapshot for recent throughput metrics."""

        if not self._dirty and self._last_snapshot is not None:
            return dict(self._last_snapshot)

        snapshot = self._compute_snapshot()
        self._last_snapshot = snapshot
        self._dirty = False
        return dict(snapshot)

    def reset(self) -> None:
        """Clear all recorded samples."""

        self._samples.clear()
        self._durations_sorted.clear()
        self._duration_sum = 0.0
        self._lag_values_sorted.clear()
        self._lag_sum = 0.0
        self._lag_count = 0
        self._last_snapshot = None
        self._dirty = True

    def _compute_snapshot(self) -> dict[str, float | int | None]:
        count = len(self._samples)
        if count == 0:
            return {
                "samples": 0,
                "avg_processing_ms": None,
                "p95_processing_ms": None,
                "max_processing_ms": None,
                "avg_lag_ms": None,
                "max_lag_ms": None,
                "throughput_per_min": None,
            }

        avg_processing = self._duration_sum / count
        p95_processing = self._percentile_sorted(self._durations_sorted, 95.0)
        max_processing = self._durations_sorted[-1]

        if self._lag_count:
            avg_lag = self._lag_sum / self._lag_count
            max_lag = self._lag_values_sorted[-1]
        else:
            avg_lag = None
            max_lag = None

        throughput_per_min = self._calculate_throughput_per_minute()

        return {
            "samples": count,
            "avg_processing_ms": avg_processing,
            "p95_processing_ms": p95_processing,
            "max_processing_ms": max_processing,
            "avg_lag_ms": avg_lag,
            "max_lag_ms": max_lag,
            "throughput_per_min": throughput_per_min,
        }

    def _calculate_throughput_per_minute(self) -> float | None:
        if len(self._samples) < 2:
            return None

        start = self._samples[0].started_at
        end = self._samples[-1].finished_at
        elapsed_seconds = (end - start).total_seconds()
        if elapsed_seconds <= 0:
            return None
        return len(self._samples) / (elapsed_seconds / 60.0)

    def _add_sample_stats(self, sample: ThroughputSample) -> None:
        duration = float(sample.duration_ms)
        insort(self._durations_sorted, duration)
        self._duration_sum += duration

        if sample.lag_ms is None:
            return

        lag = float(sample.lag_ms)
        insort(self._lag_values_sorted, lag)
        self._lag_sum += lag
        self._lag_count += 1

    def _remove_sample_stats(self, sample: ThroughputSample) -> None:
        duration = float(sample.duration_ms)
        self._remove_sorted_value(self._durations_sorted, duration)
        self._duration_sum -= duration

        if sample.lag_ms is None:
            return

        lag = float(sample.lag_ms)
        self._remove_sorted_value(self._lag_values_sorted, lag)
        self._lag_sum -= lag
        self._lag_count = max(self._lag_count - 1, 0)

    @staticmethod
    def _remove_sorted_value(container: list[float], value: float) -> None:
        index = bisect_left(container, value)
        length = len(container)
        tolerance = 1e-9
        while index < length:
            candidate = container[index]
            if math.isclose(candidate, value, rel_tol=tolerance, abs_tol=tolerance):
                container.pop(index)
                return
            if candidate > value and not math.isclose(candidate, value, rel_tol=tolerance, abs_tol=tolerance):
                break
            index += 1

        for idx, candidate in enumerate(container):
            if math.isclose(candidate, value, rel_tol=tolerance, abs_tol=tolerance):
                container.pop(idx)
                return

    @staticmethod
    def _percentile_sorted(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        if percentile <= 0.0:
            return values[0]
        if percentile >= 100.0:
            return values[-1]
        rank = (percentile / 100.0) * (len(values) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        weight = rank - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight
