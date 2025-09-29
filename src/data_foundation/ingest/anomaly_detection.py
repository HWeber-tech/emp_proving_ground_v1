"""Anomaly detection utilities for market data ingest streams.

The Phase 3 roadmap calls for proactive monitoring of data feed breaks and
false ticks so the institutional data backbone can quarantine corrupted
observations before they reach trading systems.  This module provides
light‑weight statistical heuristics that operate on the artefacts already
produced by the ingest layer.  The implementation favours deterministic,
well-tested logic over heavyweight third-party dependencies so it can run in
CI and constrained runtime environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from enum import Enum
from statistics import median
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.data_foundation.persist.timescale import TimescaleIngestResult

__all__ = [
    "FeedAnomalySeverity",
    "FeedAnomaly",
    "FalseTickSeverity",
    "FalseTickAnomaly",
    "detect_feed_anomalies",
    "detect_false_ticks",
]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class FeedAnomalySeverity(str, Enum):
    """Severity grade for ingest feed anomalies."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class FeedAnomaly:
    """Represents a detected anomaly for an ingest dimension."""

    dimension: str
    severity: FeedAnomalySeverity
    code: str
    message: str
    detected_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class FalseTickSeverity(str, Enum):
    """Severity grade for outlier tick detections."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class FalseTickAnomaly:
    """Description of a suspected false tick in a price series."""

    index: int
    price: float
    baseline: float
    z_score: float
    severity: FalseTickSeverity
    timestamp: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "price": self.price,
            "baseline": self.baseline,
            "z_score": self.z_score,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


_DEFAULT_STALE_THRESHOLDS: Mapping[str, float] = {
    "daily_bars": 6 * 60 * 60,  # 6 hours
    "intraday_trades": 15 * 60,  # 15 minutes
    "macro_events": 24 * 60 * 60,  # 1 day
}


def _normalise_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _staleness_thresholds(
    overrides: Mapping[str, float] | None,
) -> MutableMapping[str, float]:
    thresholds: MutableMapping[str, float] = dict(_DEFAULT_STALE_THRESHOLDS)
    if overrides:
        for key, value in overrides.items():
            try:
                thresholds[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return thresholds


def detect_feed_anomalies(
    results: Mapping[str, TimescaleIngestResult | None],
    *,
    observed_dimensions: Iterable[str] | None = None,
    stale_thresholds: Mapping[str, float] | None = None,
    min_rows: int = 1,
    now: datetime | None = None,
) -> tuple[FeedAnomaly, ...]:
    """Detect stalled feeds or suspiciously stale ingest snapshots."""

    clock = now or _utc_now()
    thresholds = _staleness_thresholds(stale_thresholds)

    dimensions = set(observed_dimensions or results.keys())
    dimensions.update(results.keys())

    anomalies: list[FeedAnomaly] = []

    for dimension in sorted(dimensions):
        result = results.get(dimension)
        rows_written = result.rows_written if result else 0
        freshness = result.freshness_seconds if result else None
        end_ts = _normalise_timestamp(result.end_ts if result else None)
        severity = None
        code = ""
        message = ""
        metadata: dict[str, object] = {
            "rows_written": rows_written,
        }
        if freshness is not None:
            metadata["freshness_seconds"] = freshness
        if end_ts is not None:
            metadata["end_ts"] = end_ts.isoformat()
        if result is not None and result.source:
            metadata["source"] = result.source

        if rows_written < min_rows:
            severity = FeedAnomalySeverity.CRITICAL
            code = "feed_break"
            message = (
                f"{dimension} ingest produced {rows_written} rows; feed likely stalled"
            )
        else:
            threshold = thresholds.get(dimension)
            if threshold is None:
                threshold = thresholds.get("default")
            effective_freshness = None
            if freshness is not None:
                effective_freshness = freshness
            elif end_ts is not None:
                effective_freshness = max((clock - end_ts).total_seconds(), 0.0)

            if effective_freshness is None:
                severity = FeedAnomalySeverity.WARNING
                code = "missing_freshness"
                message = (
                    f"{dimension} ingest missing freshness telemetry; unable to evaluate staleness"
                )
            elif threshold is not None and effective_freshness > threshold:
                metadata["stale_seconds"] = effective_freshness
                severity = (
                    FeedAnomalySeverity.CRITICAL
                    if effective_freshness > threshold * 2
                    else FeedAnomalySeverity.WARNING
                )
                code = "stale_feed"
                message = (
                    f"{dimension} ingest freshness {effective_freshness:.0f}s exceeds"
                    f" SLA {threshold:.0f}s"
                )

        if severity is None:
            continue

        anomalies.append(
            FeedAnomaly(
                dimension=dimension,
                severity=severity,
                code=code,
                message=message,
                detected_at=clock,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    return tuple(anomalies)


def _median_absolute_deviation(values: Sequence[float]) -> float:
    centre = median(values)
    deviations = [abs(value - centre) for value in values]
    mad = median(deviations)
    return mad


def _coerce_timestamp_sequence(
    timestamps: Sequence[datetime] | None,
    length: int,
) -> list[datetime]:
    if timestamps is None:
        base = _utc_now()
        return [base for _ in range(length)]
    if len(timestamps) != length:
        raise ValueError("timestamps length must match prices length")
    normalised: list[datetime] = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        else:
            ts = ts.astimezone(UTC)
        normalised.append(ts)
    return normalised


def detect_false_ticks(
    prices: Sequence[float],
    *,
    timestamps: Sequence[datetime] | None = None,
    window: int = 20,
    z_threshold: float = 6.0,
    min_relative_jump: float = 0.02,
) -> tuple[FalseTickAnomaly, ...]:
    """Detect price outliers that likely represent erroneous ticks."""

    if not prices:
        return ()
    if window < 5:
        raise ValueError("window must be at least 5")
    series = [float(price) for price in prices]
    timestamps_norm = _coerce_timestamp_sequence(timestamps, len(series))

    anomalies: list[FalseTickAnomaly] = []

    for idx, price in enumerate(series):
        start = max(0, idx - window)
        reference_window = series[start:idx]
        if len(reference_window) < max(5, window // 2):
            continue

        baseline = median(reference_window)
        dispersion = _median_absolute_deviation(reference_window)
        if dispersion == 0:
            if price == baseline:
                continue
            z_score = math.copysign(float("inf"), price - baseline)
        else:
            z_score = 0.67448975 * (price - baseline) / dispersion
        relative_jump_denominator = max(abs(baseline), 1e-9)
        relative_jump = abs(price - baseline) / relative_jump_denominator

        if abs(z_score) < z_threshold or relative_jump < min_relative_jump:
            continue

        severity = (
            FalseTickSeverity.CRITICAL
            if abs(z_score) >= z_threshold * 1.5 or relative_jump >= min_relative_jump * 3
            else FalseTickSeverity.WARNING
        )

        anomalies.append(
            FalseTickAnomaly(
                index=idx,
                price=price,
                baseline=baseline,
                z_score=z_score,
                severity=severity,
                timestamp=timestamps_norm[idx],
                metadata={"relative_jump": relative_jump},
            )
        )

    return tuple(anomalies)

