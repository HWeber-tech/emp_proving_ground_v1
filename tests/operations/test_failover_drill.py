from datetime import UTC, datetime

import pytest

from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.operations.failover_drill import (
    FailoverDrillStatus,
    execute_failover_drill,
)


@pytest.mark.asyncio()
async def test_failover_drill_triggers_failover() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
    )
    results = {
        "daily_bars": TimescaleIngestResult(
            12,
            ("EURUSD",),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 23, 0, tzinfo=UTC),
            2.0,
            30.0,
            "daily_bars",
            "yahoo",
        )
    }
    fallback_calls: list[str] = []

    async def _fallback() -> None:
        fallback_calls.append("called")

    snapshot = await execute_failover_drill(
        plan=plan,
        results=results,
        fail_dimensions=("daily_bars",),
        scenario="required-daily",
        fallback=_fallback,
    )

    assert snapshot.status is FailoverDrillStatus.ok
    assert snapshot.failover_decision.should_failover is True
    assert snapshot.metadata["fallback"]["executed"] is True
    assert fallback_calls == ["called"]


@pytest.mark.asyncio()
async def test_failover_drill_warns_when_optional_dimension() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
        macro=MacroEventIngestPlan(events=[{"event_name": "CPI"}]),
    )
    results = {
        "daily_bars": TimescaleIngestResult(
            8,
            ("EURUSD",),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 22, 0, tzinfo=UTC),
            1.5,
            40.0,
            "daily_bars",
            "yahoo",
        ),
        "macro_events": TimescaleIngestResult(
            1,
            ("CPI",),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            0.1,
            None,
            "macro_events",
            "fred",
        ),
    }

    snapshot = await execute_failover_drill(
        plan=plan,
        results=results,
        fail_dimensions=("macro_events",),
        scenario="optional-macro",
        fallback=None,
    )

    assert snapshot.status is FailoverDrillStatus.warn
    assert snapshot.failover_decision.should_failover is False
    assert snapshot.failover_decision.optional_triggers == ("macro_events",)


@pytest.mark.asyncio()
async def test_failover_drill_records_fallback_error() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
    )
    results = {
        "daily_bars": TimescaleIngestResult(
            5,
            ("EURUSD",),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
            1.2,
            50.0,
            "daily_bars",
            "yahoo",
        )
    }

    async def _failing_fallback() -> None:
        raise RuntimeError("fallback failed")

    snapshot = await execute_failover_drill(
        plan=plan,
        results=results,
        fail_dimensions=("daily_bars",),
        scenario="failing-fallback",
        fallback=_failing_fallback,
    )

    assert snapshot.status is FailoverDrillStatus.fail
    fallback_meta = snapshot.metadata["fallback"]
    assert fallback_meta["executed"] is True
    assert "fallback failed" in str(fallback_meta["error"])
