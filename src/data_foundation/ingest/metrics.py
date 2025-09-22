"""Summaries and metrics for Timescale ingest runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable, Mapping

from ..persist.timescale import TimescaleIngestResult


@dataclass(frozen=True)
class IngestDimensionMetrics:
    """Metric summary for a single ingest dimension."""

    dimension: str
    rows: int
    symbols: tuple[str, ...]
    ingest_duration_seconds: float
    freshness_seconds: float | None
    source: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "rows": self.rows,
            "symbols": list(self.symbols),
            "ingest_duration_seconds": self.ingest_duration_seconds,
            "freshness_seconds": self.freshness_seconds,
            "source": self.source,
        }


@dataclass(frozen=True)
class IngestMetricsSnapshot:
    """Aggregate metrics describing a Timescale ingest execution."""

    generated_at: datetime
    dimensions: tuple[IngestDimensionMetrics, ...]

    def total_rows(self) -> int:
        return sum(metric.rows for metric in self.dimensions)

    def active_dimensions(self) -> tuple[str, ...]:
        return tuple(metric.dimension for metric in self.dimensions if metric.rows > 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_rows": self.total_rows(),
            "dimensions": [metric.as_dict() for metric in self.dimensions],
        }

    def __bool__(self) -> bool:  # pragma: no cover - convenience for callers
        return bool(self.dimensions)


def _to_metric(result: TimescaleIngestResult) -> IngestDimensionMetrics:
    return IngestDimensionMetrics(
        dimension=result.dimension,
        rows=result.rows_written,
        symbols=result.symbols,
        ingest_duration_seconds=result.ingest_duration_seconds,
        freshness_seconds=result.freshness_seconds,
        source=result.source,
    )


def summarise_ingest_metrics(
    results: Mapping[str, TimescaleIngestResult] | Iterable[TimescaleIngestResult],
) -> IngestMetricsSnapshot:
    """Convert ingest results into a metrics snapshot for telemetry."""

    if isinstance(results, Mapping):
        ordered = [results[key] for key in sorted(results)]
    else:
        ordered = list(results)

    metrics = tuple(_to_metric(result) for result in ordered)
    generated_at = datetime.now(tz=UTC)
    return IngestMetricsSnapshot(generated_at=generated_at, dimensions=metrics)


__all__ = [
    "IngestDimensionMetrics",
    "IngestMetricsSnapshot",
    "summarise_ingest_metrics",
]
