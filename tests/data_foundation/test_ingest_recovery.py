from datetime import UTC, datetime

import pytest

from src.data_foundation.ingest.configuration import TimescaleIngestRecoverySettings
from src.data_foundation.ingest.health import IngestHealthStatus, evaluate_ingest_health
from src.data_foundation.ingest.recovery import plan_ingest_recovery
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result_for(
    *,
    dimension: str,
    rows: int,
    symbols: tuple[str, ...],
    freshness: float | None,
) -> TimescaleIngestResult:
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=symbols,
        start_ts=datetime(2024, 1, 1, tzinfo=UTC),
        end_ts=datetime(2024, 1, 2, tzinfo=UTC),
        ingest_duration_seconds=1.0,
        freshness_seconds=freshness,
        dimension=dimension,
        source="yahoo",
    )


def test_recovery_targets_missing_symbols_and_extends_lookback() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"], lookback_days=30),
    )
    results = {
        "daily_bars": _result_for(
            dimension="daily_bars",
            rows=5,
            symbols=("EURUSD",),
            freshness=120.0,
        )
    }

    report = evaluate_ingest_health(results, plan=plan)
    assert report.status is IngestHealthStatus.warn

    recommendation = plan_ingest_recovery(
        report,
        original_plan=plan,
        results=results,
        settings=TimescaleIngestRecoverySettings(
            enabled=True,
            max_attempts=1,
            lookback_multiplier=1.5,
        ),
        attempt=1,
    )

    assert recommendation.is_empty() is False
    assert recommendation.reasons["daily_bars"].startswith("Missing symbols")
    assert recommendation.plan.daily is not None
    assert recommendation.plan.daily.symbols == ["GBPUSD"]
    assert recommendation.plan.daily.lookback_days == 45
    summary = recommendation.summary()
    assert summary["plan"]["daily_bars"]["symbols"] == ["GBPUSD"]


def test_recovery_handles_macro_events() -> None:
    macro_plan = MacroEventIngestPlan(
        events=[{"event_name": "CPI"}, {"event_name": "GDP"}],
        source="inline",
    )
    plan = TimescaleBackbonePlan(macro=macro_plan)
    results = {
        "macro_events": _result_for(
            dimension="macro_events",
            rows=0,
            symbols=tuple(),
            freshness=None,
        )
    }

    report = evaluate_ingest_health(results, plan=plan)
    assert report.status is IngestHealthStatus.error

    recommendation = plan_ingest_recovery(
        report,
        original_plan=plan,
        results=results,
        settings=TimescaleIngestRecoverySettings(enabled=True, max_attempts=1),
        attempt=1,
    )

    assert recommendation.plan.macro is not None
    assert recommendation.missing_symbols["macro_events"] == ("CPI", "GDP")
    events = recommendation.plan.macro.events
    assert events is not None and len(events) == 2


@pytest.mark.parametrize("attempt,expected", [(1, 60), (2, 120)])
def test_recovery_multiplier_scales_with_attempt(attempt: int, expected: int) -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=30),
    )
    results = {
        "daily_bars": _result_for(
            dimension="daily_bars",
            rows=0,
            symbols=tuple(),
            freshness=None,
        )
    }
    report = evaluate_ingest_health(results, plan=plan)
    settings = TimescaleIngestRecoverySettings(
        enabled=True,
        max_attempts=3,
        lookback_multiplier=2.0,
    )

    recommendation = plan_ingest_recovery(
        report,
        original_plan=plan,
        results=results,
        settings=settings,
        attempt=attempt,
    )

    assert recommendation.plan.daily is not None
    assert recommendation.plan.daily.lookback_days == expected
