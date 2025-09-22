from __future__ import annotations

from datetime import UTC, datetime

from src.data_foundation.ingest.failover import (
    IngestFailoverPolicy,
    decide_ingest_failover,
)
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)


def _report(status: IngestHealthStatus, *checks: IngestHealthCheck) -> IngestHealthReport:
    return IngestHealthReport(
        status=status,
        generated_at=datetime.now(tz=UTC),
        checks=tuple(checks),
        metadata={"source": "test"},
    )


def _check(
    dimension: str,
    *,
    status: IngestHealthStatus,
    rows: int,
    expected: tuple[str, ...] = (),
    observed: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
) -> IngestHealthCheck:
    return IngestHealthCheck(
        dimension=dimension,
        status=status,
        message="test",
        rows_written=rows,
        freshness_seconds=0.0,
        expected_symbols=expected,
        observed_symbols=observed,
        missing_symbols=missing,
        ingest_duration_seconds=0.1,
        metadata={"dimension": dimension},
    )


def test_failover_triggers_on_required_dimension_error() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
    )
    report = _report(
        IngestHealthStatus.error,
        _check(
            "daily_bars",
            status=IngestHealthStatus.error,
            rows=0,
            expected=("EURUSD",),
            missing=("EURUSD",),
        ),
    )

    decision = decide_ingest_failover(report, plan=plan)

    assert decision.should_failover
    assert decision.triggered_dimensions == ("daily_bars",)
    assert decision.reason and "status=error" in decision.reason


def test_no_failover_when_all_required_dimensions_pass() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
    )
    report = _report(
        IngestHealthStatus.ok,
        _check(
            "daily_bars",
            status=IngestHealthStatus.ok,
            rows=42,
            expected=("EURUSD",),
            observed=("EURUSD",),
        ),
    )

    decision = decide_ingest_failover(report, plan=plan)

    assert not decision.should_failover
    assert decision.reason is None
    assert decision.triggered_dimensions == ()


def test_optional_macro_failure_does_not_trigger_failover() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
        macro=MacroEventIngestPlan(events=[]),
    )
    report = _report(
        IngestHealthStatus.error,
        _check(
            "daily_bars",
            status=IngestHealthStatus.ok,
            rows=10,
            expected=("EURUSD",),
            observed=("EURUSD",),
        ),
        _check(
            "macro_events",
            status=IngestHealthStatus.error,
            rows=0,
            missing=("CPI",),
        ),
    )

    decision = decide_ingest_failover(report, plan=plan)

    assert not decision.should_failover
    assert decision.optional_triggers == ("macro_events",)
    assert decision.triggered_dimensions == ()


def test_policy_can_ignore_missing_symbols() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"]),
    )
    report = _report(
        IngestHealthStatus.warn,
        _check(
            "daily_bars",
            status=IngestHealthStatus.warn,
            rows=1,
            expected=("EURUSD", "GBPUSD"),
            observed=("EURUSD",),
            missing=("GBPUSD",),
        ),
    )

    strict_decision = decide_ingest_failover(report, plan=plan)
    assert strict_decision.should_failover

    relaxed_policy = IngestFailoverPolicy(fail_on_missing_symbols=False)
    relaxed_decision = decide_ingest_failover(report, plan=plan, policy=relaxed_policy)

    assert not relaxed_decision.should_failover
    assert relaxed_decision.reason is None
