from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import MetaData, Table, Column, DateTime, Float, Integer, String, create_engine

from src.data_foundation.storage.timescale_queries import TimescaleQueryInterface


@pytest.fixture()
def sqlite_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ticks.db'}")
    metadata = MetaData()

    Table(
        "market_data_ticks",
        metadata,
        Column("ts", DateTime(timezone=True), nullable=False),
        Column("symbol", String, nullable=False),
        Column("price", Float, nullable=False),
        Column("size", Integer, nullable=False),
        Column("ingested_at", DateTime(timezone=True), nullable=False),
    )

    Table(
        "market_data_quotes",
        metadata,
        Column("ts", DateTime(timezone=True), nullable=False),
        Column("symbol", String, nullable=False),
        Column("bid_price", Float, nullable=False),
        Column("bid_size", Integer, nullable=False),
        Column("ask_price", Float, nullable=False),
        Column("ask_size", Integer, nullable=False),
        Column("ingested_at", DateTime(timezone=True), nullable=False),
    )

    metadata.create_all(engine)

    start = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    rows = []
    for minute in range(3):
        ts = start + timedelta(minutes=minute)
        rows.append(
            {
                "ts": ts,
                "symbol": "AAPL",
                "price": 100.0 + minute,
                "size": 10 + minute,
                "ingested_at": ts + timedelta(minutes=1),
            }
        )
        rows.append(
            {
                "ts": ts,
                "symbol": "MSFT",
                "price": 200.0 + minute,
                "size": 20 + minute,
                "ingested_at": ts + timedelta(minutes=1),
            }
        )

    quote_rows = []
    for minute in range(3):
        ts = start + timedelta(minutes=minute)
        quote_rows.append(
            {
                "ts": ts,
                "symbol": "AAPL",
                "bid_price": 99.5 + minute,
                "bid_size": 50 + minute,
                "ask_price": 100.5 + minute,
                "ask_size": 60 + minute,
                "ingested_at": ts + timedelta(minutes=1),
            }
        )

    pd.DataFrame(rows).to_sql("market_data_ticks", engine, if_exists="append", index=False)
    pd.DataFrame(quote_rows).to_sql("market_data_quotes", engine, if_exists="append", index=False)

    try:
        yield engine
    finally:
        engine.dispose()


def test_fetch_trades_filters_and_orders(sqlite_engine):
    interface = TimescaleQueryInterface(sqlite_engine)
    start = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    result = interface.fetch_trades(symbols=["AAPL"], start=start, end=end)

    assert result.dimension == "ticks"
    assert result.symbols == ("AAPL",)
    assert result.rowcount == 2
    assert result.window_start == start
    assert result.window_end == end
    assert result.max_ingested_at == datetime(2024, 1, 1, 10, 2, tzinfo=UTC)
    assert list(result.frame["symbol"].unique()) == ["AAPL"]
    assert result.frame["ts"].is_monotonic_increasing


def test_fetch_trades_resample(sqlite_engine):
    interface = TimescaleQueryInterface(sqlite_engine)
    start = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    end = start + timedelta(minutes=2)

    result = interface.fetch_trades(symbols=["AAPL"], start=start, end=end, resample="1T")

    assert result.dimension == "ticks"
    assert result.rowcount == 3
    assert {"price", "size"}.issubset(result.frame.columns)
    assert pytest.approx(result.frame.loc[0, "price"], 0.0001) == 100.0
    assert result.frame.loc[0, "size"] == 10


def test_cache_hits_exposed(sqlite_engine):
    interface = TimescaleQueryInterface(sqlite_engine)
    now = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)

    interface.fetch_trades(symbols=["AAPL"], start=now, end=now + timedelta(minutes=2))
    stats_after_miss = interface.cache_stats()
    assert stats_after_miss.misses == 1
    assert stats_after_miss.hits == 0

    interface.fetch_trades(symbols=["AAPL"], start=now, end=now + timedelta(minutes=2))
    stats_after_hit = interface.cache_stats()
    assert stats_after_hit.hits == 1
    assert stats_after_hit.misses == 1

    # Cached results should be copies; mutating a frame from a cached result must not
    # affect future fetches.
    cached_result = interface.fetch_trades(symbols=["AAPL"], start=now, end=now + timedelta(minutes=2))
    cached_result.frame.loc[:, "price"] = 0
    fresh = interface.fetch_trades(symbols=["AAPL"], start=now, end=now + timedelta(minutes=2))
    assert fresh.frame["price"].iloc[0] != 0
