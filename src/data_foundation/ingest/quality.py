"""Data quality evaluation for Timescale ingest runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from ..persist.timescale import TimescaleIngestResult
from .health import DEFAULT_FRESHNESS_SLA_SECONDS
from .timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)


class IngestQualityStatus(StrEnum):
    """Overall grading for ingest data quality."""

    ok = "ok"
    warn = "warn"
    error = "error"


@dataclass(frozen=True)
class IngestQualityCheck:
    """Quality evaluation for a single ingest dimension."""

    dimension: str
    status: IngestQualityStatus
    score: float
    observed_rows: int
    expected_rows: int | None = None
    coverage_ratio: float | None = None
    messages: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "status": self.status.value,
            "score": self.score,
            "observed_rows": self.observed_rows,
        }
        if self.expected_rows is not None:
            payload["expected_rows"] = self.expected_rows
        if self.coverage_ratio is not None:
            payload["coverage_ratio"] = self.coverage_ratio
        if self.messages:
            payload["messages"] = list(self.messages)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class IngestQualityReport:
    """Aggregated quality findings across ingest dimensions."""

    status: IngestQualityStatus
    score: float
    generated_at: datetime
    checks: tuple[IngestQualityCheck, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "score": self.score,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }


_WARN_THRESHOLD_DEFAULT = 0.85
_ERROR_THRESHOLD_DEFAULT = 0.6


def _planned_dimensions(plan: TimescaleBackbonePlan | None) -> set[str]:
    if plan is None:
        return set()
    planned: set[str] = set()
    if plan.daily is not None:
        planned.add("daily_bars")
    if plan.intraday is not None:
        planned.add("intraday_trades")
    if plan.macro is not None:
        planned.add("macro_events")
    return planned


def _expected_symbols(plan: TimescaleBackbonePlan | None) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    if plan is None:
        return mapping
    if plan.daily is not None:
        mapping["daily_bars"] = tuple(plan.daily.normalised_symbols())
    if plan.intraday is not None:
        mapping["intraday_trades"] = tuple(plan.intraday.normalised_symbols())
    if plan.macro is not None and plan.macro.events:
        names = []
        for event in plan.macro.events:
            name = None
            if hasattr(event, "event_name"):
                name = getattr(event, "event_name")
            elif isinstance(event, Mapping):
                raw = event.get("event_name") or event.get("name")
                if raw:
                    name = str(raw)
            if name:
                names.append(name)
        if names:
            mapping["macro_events"] = tuple(names)
    return mapping


def _expected_daily_rows(plan: DailyBarIngestPlan | None) -> int | None:
    if plan is None:
        return None
    symbols = plan.normalised_symbols()
    if not symbols:
        return None
    lookback = max(int(plan.lookback_days), 0)
    if lookback <= 0:
        return None
    return len(symbols) * lookback


def _parse_intraday_interval(interval: str) -> tuple[int, str]:
    if not interval:
        return 1, "m"
    digits = ""
    unit = "m"
    for char in interval:
        if char.isdigit():
            digits += char
        else:
            unit = char.lower()
    value = int(digits or 1)
    return value or 1, unit


def _expected_intraday_rows(plan: IntradayTradeIngestPlan | None) -> int | None:
    if plan is None:
        return None
    symbols = plan.normalised_symbols()
    if not symbols:
        return None
    lookback = max(int(plan.lookback_days), 0)
    if lookback <= 0:
        return None
    value, unit = _parse_intraday_interval(plan.interval)
    samples_per_day: int
    if unit == "m":
        minutes = max(value, 1)
        trading_minutes = 390  # assume 6.5 hour trading session
        samples_per_day = max(trading_minutes // minutes, 1)
    elif unit == "h":
        hours = max(value, 1)
        trading_hours = 7  # approximate full session window
        samples_per_day = max(trading_hours // hours, 1)
    elif unit == "d":
        samples_per_day = 1
    else:
        samples_per_day = max(24 * 60 // max(value, 1), 1)
    return len(symbols) * lookback * samples_per_day


def _expected_macro_rows(plan: MacroEventIngestPlan | None) -> int | None:
    if plan is None:
        return None
    if plan.events:
        return len(tuple(plan.events))
    if plan.has_window():
        return 1  # window requested but no explicit expectations
    return None


def _expected_rows_for_dimension(
    dimension: str,
    plan: TimescaleBackbonePlan | None,
) -> int | None:
    if plan is None:
        return None
    if dimension == "daily_bars":
        return _expected_daily_rows(plan.daily)
    if dimension == "intraday_trades":
        return _expected_intraday_rows(plan.intraday)
    if dimension == "macro_events":
        return _expected_macro_rows(plan.macro)
    return None


def _quality_status(
    score: float, warn_threshold: float, error_threshold: float
) -> IngestQualityStatus:
    if score < error_threshold:
        return IngestQualityStatus.error
    if score < warn_threshold:
        return IngestQualityStatus.warn
    return IngestQualityStatus.ok


def _escalate(
    current: IngestQualityStatus,
    candidate: IngestQualityStatus,
) -> IngestQualityStatus:
    order = {
        IngestQualityStatus.ok: 0,
        IngestQualityStatus.warn: 1,
        IngestQualityStatus.error: 2,
    }
    if order[candidate] > order[current]:
        return candidate
    return current


def evaluate_ingest_quality(
    results: Mapping[str, TimescaleIngestResult],
    *,
    plan: TimescaleBackbonePlan | None = None,
    warn_threshold: float = _WARN_THRESHOLD_DEFAULT,
    error_threshold: float = _ERROR_THRESHOLD_DEFAULT,
    freshness_sla: Mapping[str, float] | None = None,
    metadata: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> IngestQualityReport:
    """Grade ingest results using coverage heuristics and freshness drift."""

    planned = _planned_dimensions(plan)
    expected_symbols = _expected_symbols(plan)

    slas = dict(DEFAULT_FRESHNESS_SLA_SECONDS)
    if freshness_sla:
        slas.update({key: float(value) for key, value in freshness_sla.items()})

    observed_dimensions = set(results.keys()) | planned
    checks: list[IngestQualityCheck] = []
    overall_status = IngestQualityStatus.ok
    overall_score = 1.0

    for dimension in sorted(observed_dimensions):
        result = results.get(dimension)
        if result is None:
            result = TimescaleIngestResult.empty(dimension=dimension)

        expected_rows = _expected_rows_for_dimension(dimension, plan)
        coverage_ratio: float | None = None
        score = 1.0
        messages: list[str] = []
        check_metadata: dict[str, object] = {
            "rows_written": result.rows_written,
            "freshness_seconds": result.freshness_seconds,
        }
        if result.start_ts is not None:
            check_metadata["start_ts"] = result.start_ts.isoformat()
        if result.end_ts is not None:
            check_metadata["end_ts"] = result.end_ts.isoformat()
        if result.source:
            check_metadata["source"] = result.source
        if expected_symbols.get(dimension):
            check_metadata["expected_symbols"] = list(expected_symbols[dimension])
        if result.symbols:
            check_metadata["observed_symbols"] = list(result.symbols)

        if result.rows_written <= 0:
            score = 0.0
            messages.append("No rows ingested for dimension")
        elif expected_rows:
            ratio = max(0.0, min(result.rows_written / expected_rows, 1.0))
            coverage_ratio = ratio
            score = min(score, ratio)
            messages.append(
                f"Coverage {ratio * 100:.1f}% ({result.rows_written}/{expected_rows} rows)"
            )
        else:
            messages.append(f"Observed rows: {result.rows_written}")

        target = slas.get(dimension)
        if target is not None:
            check_metadata["freshness_sla_seconds"] = target
        freshness = result.freshness_seconds
        if freshness is None:
            score = min(score, 0.75)
            messages.append("Freshness metric unavailable")
        elif target is not None and freshness > target:
            overshoot = max(freshness - target, 0.0)
            penalty = max(0.0, 1.0 - (overshoot / max(target, 1.0)))
            score = min(score, penalty)
            messages.append(f"Freshness {freshness:.0f}s exceeds SLA {target:.0f}s")

        status = _quality_status(score, warn_threshold, error_threshold)
        check = IngestQualityCheck(
            dimension=dimension,
            status=status,
            score=round(score, 4),
            observed_rows=result.rows_written,
            expected_rows=expected_rows,
            coverage_ratio=None if coverage_ratio is None else round(coverage_ratio, 4),
            messages=tuple(messages),
            metadata={
                key: value for key, value in check_metadata.items() if value not in (None, [], {})
            },
        )
        checks.append(check)
        overall_status = _escalate(overall_status, status)
        overall_score = min(overall_score, score)

    report_metadata: dict[str, object] = {
        "warn_threshold": warn_threshold,
        "error_threshold": error_threshold,
    }
    if metadata:
        report_metadata.update({str(key): value for key, value in metadata.items()})

    timestamp = generated_at or datetime.now(tz=UTC)
    report = IngestQualityReport(
        status=overall_status,
        score=round(overall_score, 4),
        generated_at=timestamp,
        checks=tuple(checks),
        metadata=report_metadata,
    )
    return report


__all__ = [
    "IngestQualityCheck",
    "IngestQualityReport",
    "IngestQualityStatus",
    "evaluate_ingest_quality",
]
