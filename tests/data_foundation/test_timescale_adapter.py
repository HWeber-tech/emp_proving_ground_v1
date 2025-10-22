from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestor,
    TimescaleMigrator,
)
from src.data_foundation.persist.timescale_reader import TimescaleReader
from src.data_foundation.storage.timescale_adapter import TimescaleAdapter


@pytest.fixture()
def timescale_engine(tmp_path):
    db_path = tmp_path / "adapter_timescale.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    try:
        yield engine
    finally:
        engine.dispose()


class RecordingIngestor(TimescaleIngestor):
    def __init__(self, engine, *, chunk_size: int = 500) -> None:
        super().__init__(engine, chunk_size=chunk_size)
        self.batch_sizes: list[int] = []

    def upsert_intraday_trades(self, df: pd.DataFrame, *, source: str = "yahoo"):
        self.batch_sizes.append(len(df))
        return super().upsert_intraday_trades(df, source=source)


class FaultyIngestor(TimescaleIngestor):
    def __init__(self, engine, *, chunk_size: int = 500) -> None:
        super().__init__(engine, chunk_size=chunk_size)
        self.calls = 0

    def upsert_intraday_trades(self, df: pd.DataFrame, *, source: str = "yahoo"):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return super().upsert_intraday_trades(df, source=source)


@pytest.mark.asyncio()
async def test_adapter_batches_intraday_trades(timescale_engine) -> None:
    adapter = TimescaleAdapter(
        timescale_engine,
        batch_size=1000,
        chunk_size=256,
        default_source="live",
        ingestor_factory=lambda engine, chunk: RecordingIngestor(engine, chunk_size=chunk),
    )

    base = datetime(2024, 1, 2, tzinfo=timezone.utc)
    rows = []
    for index in range(2500):
        rows.append(
            {
                "timestamp": base + timedelta(seconds=index),
                "symbol": "EURUSD" if index % 2 == 0 else "USDJPY",
                "price": 1.10 + index * 0.0001,
                "size": 100 + index,
                "exchange": "SIM",
                "conditions": "TEST",
            }
        )
    frame = pd.DataFrame(rows)

    result = await adapter.ingest_intraday_trades(frame, source="feed")

    assert result.ok
    assert result.dimension == "intraday_trades"
    assert result.batches == 3
    assert result.rows_written == len(frame)
    assert set(result.symbols) == {"EURUSD", "USDJPY"}
    assert not result.errors

    recording = adapter.ingestor
    assert isinstance(recording, RecordingIngestor)
    assert recording.batch_sizes == [1000, 1000, 500]

    reader = TimescaleReader(timescale_engine)
    trade_result = reader.fetch_intraday_trades(symbols=["EURUSD"])
    assert trade_result.rowcount == 1250


@pytest.mark.asyncio()
async def test_adapter_records_errors_and_continues(timescale_engine) -> None:
    adapter = TimescaleAdapter(
        timescale_engine,
        batch_size=100,
        chunk_size=64,
        default_source="live",
        ingestor_factory=lambda engine, chunk: FaultyIngestor(engine, chunk_size=chunk),
    )

    base = datetime(2024, 1, 3, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "timestamp": base + timedelta(seconds=index),
                "symbol": "GBPUSD",
                "price": 1.20 + index * 0.0001,
                "size": 50 + index,
            }
            for index in range(150)
        ]
    )

    result = await adapter.ingest_intraday_trades(frame)

    assert not result.ok
    assert result.batches == 2
    assert result.rows_written == 50
    assert any("RuntimeError" in message for message in result.errors)

    reader = TimescaleReader(timescale_engine)
    trade_result = reader.fetch_intraday_trades(symbols=["GBPUSD"])
    assert trade_result.rowcount == 50


@pytest.mark.asyncio()
async def test_adapter_accepts_iterable_payload(timescale_engine) -> None:
    adapter = TimescaleAdapter(timescale_engine, batch_size=10, chunk_size=32)

    base = datetime(2024, 1, 4, tzinfo=timezone.utc)
    payload = [
        {
            "timestamp": base,
            "symbol": "AUDUSD",
            "price": 0.70,
            "size": 250,
        },
        {
            "timestamp": base + timedelta(seconds=1),
            "symbol": "AUDUSD",
            "price": 0.701,
            "size": 260,
        },
    ]

    result = await adapter.ingest_intraday_trades(payload)

    assert result.ok
    assert result.rows_written == 2
    reader = TimescaleReader(timescale_engine)
    trade_result = reader.fetch_intraday_trades(symbols=["AUDUSD"])
    assert trade_result.rowcount == 2
