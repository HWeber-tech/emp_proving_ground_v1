"""Failover decision engine for the institutional Timescale ingest path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .health import IngestHealthCheck, IngestHealthReport, IngestHealthStatus
from .timescale_pipeline import TimescaleBackbonePlan


def _planned_dimensions(plan: TimescaleBackbonePlan | None) -> tuple[str, ...]:
    if plan is None:
        return tuple()

    dimensions: list[str] = []
    if plan.daily is not None:
        dimensions.append("daily_bars")
    if plan.intraday is not None:
        dimensions.append("intraday_trades")
    if plan.macro is not None:
        dimensions.append("macro_events")
    return tuple(dimensions)


@dataclass(frozen=True)
class IngestFailoverPolicy:
    """Policy that determines when to fall back to the bootstrap ingest path."""

    trigger_statuses: tuple[IngestHealthStatus, ...] = (IngestHealthStatus.error,)
    require_rows: bool = True
    fail_on_missing_symbols: bool = True
    optional_dimensions: tuple[str, ...] = ("macro_events",)

    def is_optional(self, dimension: str) -> bool:
        return dimension in self.optional_dimensions


@dataclass(frozen=True)
class IngestFailoverDecision:
    """Outcome of evaluating ingest health against a failover policy."""

    should_failover: bool
    status: IngestHealthStatus
    reason: str | None
    generated_at: datetime
    triggered_dimensions: tuple[str, ...]
    optional_triggers: tuple[str, ...]
    planned_dimensions: tuple[str, ...]
    metadata: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "should_failover": self.should_failover,
            "status": self.status.value,
            "reason": self.reason,
            "generated_at": self.generated_at.isoformat(),
            "triggered_dimensions": list(self.triggered_dimensions),
            "optional_triggers": list(self.optional_triggers),
            "planned_dimensions": list(self.planned_dimensions),
            "metadata": dict(self.metadata),
        }


def decide_ingest_failover(
    report: IngestHealthReport,
    *,
    plan: TimescaleBackbonePlan | None = None,
    policy: IngestFailoverPolicy | None = None,
) -> IngestFailoverDecision:
    """Determine whether to fall back to the bootstrap ingest path."""

    policy = policy or IngestFailoverPolicy()
    planned = _planned_dimensions(plan)

    checks_by_dimension: Mapping[str, IngestHealthCheck] = {
        check.dimension: check for check in report.checks
    }

    triggered_required: list[str] = []
    triggered_optional: list[str] = []
    reasons: list[str] = []

    for dimension in planned:
        check = checks_by_dimension.get(dimension)
        is_optional = policy.is_optional(dimension)
        bucket = triggered_optional if is_optional else triggered_required
        if check is None:
            bucket.append(dimension)
            reasons.append(f"missing health check for {dimension}")
            continue

        # Treat zero rows or missing symbols as failures for required dimensions.
        if policy.require_rows and check.rows_written == 0:
            if dimension not in bucket:
                bucket.append(dimension)
            reasons.append(f"{dimension} produced zero rows")

        if policy.fail_on_missing_symbols and check.missing_symbols:
            targets = ", ".join(check.missing_symbols)
            if dimension not in bucket:
                bucket.append(dimension)
            reasons.append(f"{dimension} missing symbols: {targets}")

        if check.status in policy.trigger_statuses:
            if dimension not in bucket:
                bucket.append(dimension)
            reasons.append(f"{dimension} status={check.status.value}")

    should_failover = bool(triggered_required)

    if not should_failover and report.status in policy.trigger_statuses:
        if triggered_optional and not planned:
            # No required dimensions planned and only optional ones triggered.
            should_failover = False
        elif not triggered_optional:
            should_failover = True
            reasons.append(f"overall status={report.status.value}")

    reason = "; ".join(dict.fromkeys(reasons)) if should_failover else None

    return IngestFailoverDecision(
        should_failover=should_failover,
        status=report.status,
        reason=reason,
        generated_at=report.generated_at,
        triggered_dimensions=tuple(triggered_required),
        optional_triggers=tuple(triggered_optional),
        planned_dimensions=planned,
        metadata=report.metadata,
    )
