from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data_foundation.cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    TimescaleQueryCache,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings, TimescaleMigrator
from src.data_foundation.persist.timescale_reader import TimescaleQueryResult


@pytest.fixture()
def timescale_engine(tmp_path):
    db_path = tmp_path / "cache_timescale.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    try:
        yield engine
    finally:
        engine.dispose()


class FakeReader:
    def __init__(self) -> None:
        self.daily_calls = 0

    def fetch_daily_bars(
        self, *, symbols, start=None, end=None, limit=None
    ) -> TimescaleQueryResult:
        self.daily_calls += 1
        frame = pd.DataFrame(
            [
                {
                    "ts": datetime(2024, 1, 2, tzinfo=UTC),
                    "symbol": symbols[0],
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.18,
                    "adj_close": 1.17,
                    "volume": 1200,
                    "source": "test",
                    "ingested_at": datetime(2024, 1, 2, 12, tzinfo=UTC),
                }
            ]
        )
        return TimescaleQueryResult(
            dimension="daily_bars",
            frame=frame,
            symbols=tuple(symbols),
            start_ts=datetime(2024, 1, 2, tzinfo=UTC),
            end_ts=datetime(2024, 1, 2, tzinfo=UTC),
            max_ingested_at=datetime(2024, 1, 2, 12, tzinfo=UTC),
        )


def _managed_cache(ttl_seconds: int, *, time_fn=None) -> ManagedRedisCache:
    policy = RedisCachePolicy(ttl_seconds=ttl_seconds, max_keys=128, namespace="emp:test")
    return ManagedRedisCache(InMemoryRedis(), policy, time_fn=time_fn or (lambda: 0.0))


def test_timescale_query_cache_round_trip() -> None:
    reader = FakeReader()
    cache = _managed_cache(600)
    query_cache = TimescaleQueryCache(reader, cache)

    first = query_cache.fetch_daily_bars(symbols=["EURUSD"], end=None, limit=None)
    second = query_cache.fetch_daily_bars(symbols=["EURUSD"], end=None, limit=None)

    assert reader.daily_calls == 1
    pdt.assert_frame_equal(first.frame, second.frame, check_dtype=False)
    metrics = cache.metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1


def test_timescale_query_cache_respects_ttl() -> None:
    reader = FakeReader()
    time_state = {"now": 1_000.0}

    def fake_time() -> float:
        return time_state["now"]

    cache = _managed_cache(30, time_fn=fake_time)
    query_cache = TimescaleQueryCache(reader, cache)

    query_cache.fetch_daily_bars(symbols=["EURUSD"], end=None, limit=None)
    cache.metrics(reset=True)

    time_state["now"] += 60.0
    query_cache.fetch_daily_bars(symbols=["EURUSD"], end=None, limit=None)

    assert reader.daily_calls == 2
    metrics = cache.metrics()
    assert metrics["misses"] == 1
    assert metrics["expirations"] == 1


@pytest.mark.asyncio()
async def test_timescale_connector_uses_cache(timescale_engine) -> None:
    from datetime import timedelta

    from src.data_foundation.persist.timescale import TimescaleIngestor
    from src.data_foundation.persist.timescale_reader import TimescaleReader
    from src.data_foundation.fabric.timescale_connector import TimescaleDailyBarConnector

    ingestor = TimescaleIngestor(timescale_engine)
    base = datetime(2024, 1, 2, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {
                "date": base - timedelta(days=1),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "adj_close": 1.04,
                "volume": 1000,
                "symbol": "EURUSD",
            },
            {
                "date": base,
                "open": 1.05,
                "high": 1.2,
                "low": 1.0,
                "close": 1.18,
                "adj_close": 1.17,
                "volume": 1500,
                "symbol": "EURUSD",
            },
        ]
    )
    ingestor.upsert_daily_bars(frame)

    reader = TimescaleReader(timescale_engine)
    cache = _managed_cache(900)
    query_cache = TimescaleQueryCache(reader, cache)
    connector = TimescaleDailyBarConnector(reader, cache=query_cache)

    first = await connector.fetch("EURUSD")
    assert first is not None
    metrics = cache.metrics()
    assert metrics["misses"] >= 1

    second = await connector.fetch("EURUSD")
    assert second is not None
    metrics = cache.metrics()
    assert metrics["hits"] >= 1
    assert pytest.approx(first.close, rel=1e-6) == pytest.approx(second.close, rel=1e-6)
