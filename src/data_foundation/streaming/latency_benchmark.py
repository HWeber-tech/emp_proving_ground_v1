"""Streaming latency benchmarking utilities.

Phase 3 of the high-impact roadmap asks for streaming ingestion adapters with
latency benchmarks so operators can validate end-to-end freshness targets.  The
helpers in this module provide an allocation-free recording surface and summary
statistics suitable for CI and dashboard publication without introducing
runtime dependencies on Kafka clients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Callable, Iterable, Mapping, MutableSequence, Sequence

__all__ = [
    "LatencySample",
    "LatencySummary",
    "LatencyBenchmarkReport",
    "StreamingLatencyBenchmark",
]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _normalise_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class LatencySample:
    """Single latency observation for a streaming payload."""

    dimension: str
    producer_ts: datetime
    consumer_ts: datetime
    latency_ms: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "producer_ts": self.producer_ts.isoformat(),
            "consumer_ts": self.consumer_ts.isoformat(),
            "latency_ms": self.latency_ms,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class LatencySummary:
    """Summary statistics for a collection of latency samples."""

    dimension: str | None
    count: int
    min_ms: float | None
    max_ms: float | None
    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "count": self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }
        return payload


@dataclass(frozen=True)
class LatencyBenchmarkReport:
    """Structured view of latency statistics across dimensions."""

    generated_at: datetime
    overall: LatencySummary
    per_dimension: tuple[LatencySummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall": self.overall.as_dict(),
            "per_dimension": [summary.as_dict() for summary in self.per_dimension],
        }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def _summarise(dimension: str | None, latencies: Sequence[float]) -> LatencySummary:
    if not latencies:
        return LatencySummary(
            dimension=dimension,
            count=0,
            min_ms=None,
            max_ms=None,
            avg_ms=None,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
        )

    count = len(latencies)
    minimum = min(latencies)
    maximum = max(latencies)
    average = sum(latencies) / count
    return LatencySummary(
        dimension=dimension,
        count=count,
        min_ms=minimum,
        max_ms=maximum,
        avg_ms=average,
        p50_ms=_percentile(latencies, 50.0),
        p95_ms=_percentile(latencies, 95.0),
        p99_ms=_percentile(latencies, 99.0),
    )


class StreamingLatencyBenchmark:
    """Collect and summarise streaming latency observations."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._samples: MutableSequence[LatencySample] = []

    def record(
        self,
        producer_ts: datetime,
        *,
        dimension: str,
        metadata: Mapping[str, object] | None = None,
    ) -> LatencySample:
        consumer_ts = _normalise_timestamp(self._clock())
        producer_ts_norm = _normalise_timestamp(producer_ts)
        latency = max((consumer_ts - producer_ts_norm).total_seconds() * 1000.0, 0.0)
        sample = LatencySample(
            dimension=dimension,
            producer_ts=producer_ts_norm,
            consumer_ts=consumer_ts,
            latency_ms=latency,
            metadata=dict(metadata or {}),
        )
        self._samples.append(sample)
        return sample

    def extend(self, samples: Iterable[LatencySample]) -> None:
        for sample in samples:
            if not isinstance(sample, LatencySample):
                raise TypeError("extend expects LatencySample instances")
            self._samples.append(sample)

    def clear(self) -> None:
        self._samples.clear()

    def samples(self) -> tuple[LatencySample, ...]:
        return tuple(self._samples)

    def summarise(self) -> LatencyBenchmarkReport:
        if not self._samples:
            now = _normalise_timestamp(self._clock())
            return LatencyBenchmarkReport(
                generated_at=now,
                overall=_summarise(None, []),
                per_dimension=(),
            )

        latencies = [sample.latency_ms for sample in self._samples]
        overall = _summarise(None, latencies)

        buckets: dict[str, list[float]] = {}
        for sample in self._samples:
            buckets.setdefault(sample.dimension, []).append(sample.latency_ms)

        per_dimension = tuple(
            _summarise(dimension, values)
            for dimension, values in sorted(buckets.items())
        )

        generated_at = _normalise_timestamp(self._clock())
        return LatencyBenchmarkReport(
            generated_at=generated_at,
            overall=overall,
            per_dimension=per_dimension,
        )
