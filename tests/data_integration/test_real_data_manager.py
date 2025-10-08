from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.runtime.task_supervisor import TaskSupervisor

from src.data_integration.real_data_integration import RealDataManager


class DummyPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, int]] = []

    def publish(self, result, *, metadata=None) -> None:  # pragma: no cover - simple container
        self.published.append((result.dimension, result.rows_written))


def _daily_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": base - timedelta(days=1),
                "symbol": "EURUSD",
                "open": 1.10,
                "high": 1.12,
                "low": 1.08,
                "close": 1.11,
                "adj_close": 1.105,
                "volume": 1200,
            },
            {
                "date": base,
                "symbol": "EURUSD",
                "open": 1.11,
                "high": 1.13,
                "low": 1.09,
                "close": 1.125,
                "adj_close": 1.12,
                "volume": 1500,
            },
        ]
    )


def _intraday_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": base - timedelta(minutes=2),
                "symbol": "EURUSD",
                "price": 1.121,
                "size": 640,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": base - timedelta(minutes=1),
                "symbol": "EURUSD",
                "price": 1.124,
                "size": 720,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )


def test_real_data_manager_ingest_fetch_and_cache(tmp_path):
    url = f"sqlite:///{tmp_path / 'manager_timescale.db'}"
    settings = TimescaleConnectionSettings(url=url)
    publisher = DummyPublisher()

    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
    )

    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=2),
        intraday=IntradayTradeIngestPlan(symbols=["EURUSD"], lookback_days=1, interval="1m"),
    )

    manager.run_ingest_plan(
        plan,
        fetch_daily=lambda symbols, lookback: _daily_frame(base),
        fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
    )

    assert publisher.published, "ingest should publish Kafka telemetry when configured"

    manager.cache_metrics(reset=True)
    intraday = manager.fetch_data("EURUSD", interval="1m")
    assert not intraday.empty
    assert list(intraday["price"]) == [1.121, 1.124]

    metrics = manager.cache_metrics()
    assert metrics["misses"] >= 1

    intraday_cached = manager.fetch_data("EURUSD", interval="1m")
    metrics_after = manager.cache_metrics()
    assert metrics_after["hits"] >= 1
    pdt.assert_frame_equal(intraday_cached, intraday)

    daily = manager.fetch_data("EURUSD", interval="1d", period="2d")
    assert len(daily) == 2
    assert daily.iloc[-1]["close"] == pytest.approx(1.125)

    manager.close()


@pytest.mark.asyncio()
async def test_real_data_manager_scheduler_uses_supervisor(tmp_path):
    url = f"sqlite:///{tmp_path / 'scheduler_timescale.db'}"
    settings = TimescaleConnectionSettings(url=url)
    supervisor = TaskSupervisor(namespace="test-data-manager")
    manager = RealDataManager(
        timescale_settings=settings,
        task_supervisor=supervisor,
    )

    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    plan_factory = lambda: TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=1)
    )

    call_count = 0

    def fetch_daily(symbols: list[str], lookback: int) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return _daily_frame(base + timedelta(minutes=call_count))

    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3)
    manager.start_ingest_scheduler(
        plan_factory,
        schedule,
        fetch_daily=fetch_daily,
    )

    await asyncio.sleep(0.2)
    await manager.stop_ingest_scheduler()

    assert call_count >= 2
    assert supervisor.active_count == 0

    await manager.shutdown()
