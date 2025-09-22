"""Ingest health evaluation aligned with the institutional data backbone roadmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from ..persist.timescale import TimescaleIngestResult
from .timescale_pipeline import (
    TimescaleBackbonePlan,
)


class IngestHealthStatus(StrEnum):
    """Severity levels for ingest health checks."""

    ok = "ok"
    warn = "warn"
    error = "error"


DEFAULT_FRESHNESS_SLA_SECONDS: dict[str, float] = {
    "daily_bars": 24 * 60 * 60.0,  # 24h freshness target
    "intraday_trades": 15 * 60.0,  # 15m freshness target
    "macro_events": 7 * 24 * 60 * 60.0,  # 7d freshness target
}

DEFAULT_MIN_ROWS: dict[str, int] = {
    "daily_bars": 1,
    "intraday_trades": 1,
    "macro_events": 0,
}


_STATUS_ORDER: dict[IngestHealthStatus, int] = {
    IngestHealthStatus.ok: 0,
    IngestHealthStatus.warn: 1,
    IngestHealthStatus.error: 2,
}


def _escalate(current: IngestHealthStatus, candidate: IngestHealthStatus) -> IngestHealthStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _expected_symbols(plan: TimescaleBackbonePlan | None) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = {}
    if plan is None:
        return mapping

    if plan.daily:
        mapping["daily_bars"] = tuple(plan.daily.normalised_symbols())
    if plan.intraday:
        mapping["intraday_trades"] = tuple(plan.intraday.normalised_symbols())
    if plan.macro and plan.macro.events:
        names: list[str] = []
        for event in plan.macro.events:
            name = None
            if hasattr(event, "event_name"):
                name = getattr(event, "event_name")
            elif isinstance(event, Mapping):
                raw_name = event.get("event_name") or event.get("name")
                if raw_name:
                    name = str(raw_name)
            if name:
                names.append(str(name))
        if names:
            mapping["macro_events"] = tuple(names)
    return mapping


def _planned_dimensions(plan: TimescaleBackbonePlan | None) -> set[str]:
    if plan is None:
        return set()
    planned: set[str] = set()
    if plan.daily:
        planned.add("daily_bars")
    if plan.intraday:
        planned.add("intraday_trades")
    if plan.macro:
        planned.add("macro_events")
    return planned


@dataclass(frozen=True)
class IngestHealthCheck:
    """Result of evaluating a single ingest dimension."""

    dimension: str
    status: IngestHealthStatus
    message: str
    rows_written: int
    freshness_seconds: float | None
    expected_symbols: tuple[str, ...] = field(default_factory=tuple)
    observed_symbols: tuple[str, ...] = field(default_factory=tuple)
    missing_symbols: tuple[str, ...] = field(default_factory=tuple)
    ingest_duration_seconds: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "status": self.status.value,
            "message": self.message,
            "rows_written": self.rows_written,
            "freshness_seconds": self.freshness_seconds,
            "observed_symbols": list(self.observed_symbols),
        }
        if self.expected_symbols:
            payload["expected_symbols"] = list(self.expected_symbols)
        if self.missing_symbols:
            payload["missing_symbols"] = list(self.missing_symbols)
        if self.ingest_duration_seconds is not None:
            payload["ingest_duration_seconds"] = self.ingest_duration_seconds
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class IngestHealthReport:
    """Aggregated ingest health findings across dimensions."""

    status: IngestHealthStatus
    generated_at: datetime
    checks: tuple[IngestHealthCheck, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }


def evaluate_ingest_health(
    results: Mapping[str, TimescaleIngestResult],
    *,
    plan: TimescaleBackbonePlan | None = None,
    freshness_sla: Mapping[str, float] | None = None,
    min_rows: Mapping[str, int] | None = None,
    metadata: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> IngestHealthReport:
    """Evaluate ingest outcomes against freshness and completeness thresholds."""

    planned = _planned_dimensions(plan)
    expected = _expected_symbols(plan)
    slas = dict(DEFAULT_FRESHNESS_SLA_SECONDS)
    if freshness_sla:
        slas.update({k: float(v) for k, v in freshness_sla.items()})
    row_thresholds = dict(DEFAULT_MIN_ROWS)
    if min_rows:
        row_thresholds.update({k: int(v) for k, v in min_rows.items()})

    observed_dimensions = set(results.keys()) | planned
    checks: list[IngestHealthCheck] = []
    overall = IngestHealthStatus.ok

    for dimension in sorted(observed_dimensions):
        result = results.get(dimension)
        if result is None:
            result = TimescaleIngestResult.empty(dimension=dimension)

        expected_symbols = expected.get(dimension, tuple())
        observed_symbols = tuple(result.symbols)
        missing_symbols: tuple[str, ...] = tuple(
            symbol for symbol in expected_symbols if symbol not in observed_symbols
        )

        status = IngestHealthStatus.ok
        messages: list[str] = []
        min_required = row_thresholds.get(dimension, 0)
        freshness_target = slas.get(dimension)
        is_planned = dimension in planned

        if is_planned and result.rows_written == 0:
            severity = IngestHealthStatus.error if expected_symbols else IngestHealthStatus.warn
            status = _escalate(status, severity)
            messages.append("No rows ingested for planned slice")
        elif min_required and result.rows_written < min_required:
            status = _escalate(status, IngestHealthStatus.warn)
            messages.append(f"Rows below threshold ({result.rows_written} < {min_required})")

        if expected_symbols:
            if missing_symbols and len(missing_symbols) == len(expected_symbols):
                status = _escalate(status, IngestHealthStatus.error)
                messages.append("All expected symbols missing from ingest result")
            elif missing_symbols:
                status = _escalate(status, IngestHealthStatus.warn)
                formatted = ", ".join(missing_symbols)
                messages.append(f"Missing symbols: {formatted}")

        freshness = result.freshness_seconds
        if freshness is None:
            if is_planned:
                status = _escalate(status, IngestHealthStatus.warn)
                messages.append("Freshness metric unavailable")
        elif freshness_target is not None and freshness > freshness_target:
            status = _escalate(status, IngestHealthStatus.warn)
            messages.append(f"Freshness {freshness:.0f}s exceeds SLA {freshness_target:.0f}s")

        if not messages:
            messages.append("Ingest healthy")

        check_metadata: dict[str, object] = {}
        if freshness_target is not None:
            check_metadata["freshness_sla_seconds"] = freshness_target
        if min_required:
            check_metadata["min_rows_required"] = min_required
        if result.source:
            check_metadata["source"] = result.source

        check = IngestHealthCheck(
            dimension=dimension,
            status=status,
            message="; ".join(messages),
            rows_written=result.rows_written,
            freshness_seconds=result.freshness_seconds,
            expected_symbols=expected_symbols,
            observed_symbols=observed_symbols,
            missing_symbols=missing_symbols,
            ingest_duration_seconds=result.ingest_duration_seconds,
            metadata=check_metadata,
        )
        checks.append(check)
        overall = _escalate(overall, status)

    report_metadata = {
        "planned_dimensions": sorted(planned),
        "observed_dimensions": sorted(observed_dimensions),
    }
    if metadata:
        report_metadata.update({str(k): v for k, v in metadata.items()})

    generated = generated_at or datetime.now(tz=UTC)
    return IngestHealthReport(
        status=overall,
        generated_at=generated,
        checks=tuple(checks),
        metadata=report_metadata,
    )


__all__ = [
    "DEFAULT_FRESHNESS_SLA_SECONDS",
    "DEFAULT_MIN_ROWS",
    "IngestHealthCheck",
    "IngestHealthReport",
    "IngestHealthStatus",
    "evaluate_ingest_health",
]
