from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data_foundation.cache.redis_cache import ManagedRedisCache, RedisCachePolicy
from src.data_foundation.ingest import timescale_pipeline as ingest_pipeline
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.schemas import MacroEvent
from src.data_foundation.streaming.kafka_stream import KafkaIngestEventPublisher
from src.runtime.task_supervisor import TaskSupervisor

import src.data_integration.real_data_integration as real_data_module
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


def test_real_data_manager_ingest_market_slice_defaults(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'market_slice.db'}"
    settings = TimescaleConnectionSettings(url=url)
    publisher = DummyPublisher()

    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
    )

    base = datetime(2024, 3, 4, tzinfo=timezone.utc)

    def fake_fetch_daily(symbols: list[str], lookback: int) -> pd.DataFrame:
        assert symbols == ["EURUSD"]
        assert lookback == 2
        return _daily_frame(base)

    def fake_fetch_intraday(symbols: list[str], lookback: int, interval: str) -> pd.DataFrame:
        assert symbols == ["EURUSD"]
        assert lookback == 1
        assert interval == "1m"
        return _intraday_frame(base)

    macro_event = MacroEvent(
        timestamp=base,
        calendar="ECB",
        event="Rate Decision",
        currency="EUR",
        actual=4.0,
        forecast=4.0,
        previous=3.5,
        importance="high",
        source="fred",
    )

    def fake_fetch_macro(start: str, end: str):
        assert start == "2024-03-01"
        assert end == "2024-03-05"
        return [macro_event]

    monkeypatch.setattr(ingest_pipeline, "fetch_daily_bars", fake_fetch_daily)
    monkeypatch.setattr(ingest_pipeline, "fetch_intraday_trades", fake_fetch_intraday)
    monkeypatch.setattr(real_data_module, "fetch_fred_calendar", fake_fetch_macro)

    results = manager.ingest_market_slice(
        symbols=["eurusd"],
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
        macro_start="2024-03-01",
        macro_end="2024-03-05",
    )

    assert set(results) == {"daily_bars", "intraday_trades", "macro_events"}
    assert all(result.rows_written >= 0 for result in results.values())
    assert len(publisher.published) == 3

    manager.cache_metrics(reset=True)

    macro_frame = manager.fetch_macro_events(
        calendars=("ECB",),
        start="2024-03-01",
        end="2024-03-05",
    )
    assert not macro_frame.empty
    assert set(macro_frame["calendar"].unique()) == {"ECB"}
    assert "event_name" in macro_frame.columns

    intraday_first = manager.fetch_data("EURUSD", interval="1m")
    assert not intraday_first.empty
    intraday_second = manager.fetch_data("EURUSD", interval="1m")
    metrics = manager.cache_metrics(reset=True)
    assert metrics["misses"] >= 1
    assert metrics["hits"] >= 1
    pdt.assert_frame_equal(intraday_first, intraday_second)

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


class _HealthyRedis:
    def ping(self) -> bool:
        return True


class _HealthyKafkaProducer:
    def produce(self, topic, value, key=None) -> None:  # pragma: no cover - noop
        return None

    def flush(self, timeout=None) -> None:  # pragma: no cover - noop
        return None


def test_real_data_manager_connectivity_report_defaults(tmp_path):
    url = f"sqlite:///{tmp_path / 'connectivity_default.db'}"
    settings = TimescaleConnectionSettings(url=url)

    manager = RealDataManager(timescale_settings=settings)

    report = manager.connectivity_report()
    assert report.timescale is True
    assert report.redis is False
    assert report.kafka is False

    manager.close()


def test_real_data_manager_connectivity_report_full_stack(tmp_path):
    url = f"sqlite:///{tmp_path / 'connectivity_full.db'}"
    settings = TimescaleConnectionSettings(url=url)

    redis_policy = RedisCachePolicy.institutional_defaults()
    redis_cache = ManagedRedisCache(_HealthyRedis(), redis_policy)
    publisher = KafkaIngestEventPublisher(
        _HealthyKafkaProducer(),
        topic_map={"daily_bars": "telemetry.ingest"},
    )

    manager = RealDataManager(
        timescale_settings=settings,
        managed_cache=redis_cache,
        ingest_publisher=publisher,
    )

    report = manager.connectivity_report()
    assert report.timescale is True
    assert report.redis is True
    assert report.kafka is True

    manager.close()
