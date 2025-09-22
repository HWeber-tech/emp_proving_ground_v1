from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data_foundation.fabric.timescale_connector import (
    TimescaleDailyBarConnector,
    TimescaleIntradayTradeConnector,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestor,
    TimescaleMigrator,
)
from src.data_foundation.persist.timescale_reader import TimescaleReader
from src.data_foundation.services.macro_events import TimescaleMacroEventService


@pytest.fixture()
def timescale_engine(tmp_path):
    db_path = tmp_path / "connector_timescale.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.mark.asyncio()
async def test_daily_connector_fetches_latest_bar(timescale_engine) -> None:
    ingestor = TimescaleIngestor(timescale_engine)
    now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "date": now - timedelta(days=1),
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "adj_close": 1.04,
                "volume": 1000,
                "symbol": "EURUSD",
            },
            {
                "date": now,
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
    connector = TimescaleDailyBarConnector(reader)

    result = await connector.fetch("EURUSD")
    assert result is not None
    assert pytest.approx(result.close, rel=1e-6) == 1.18

    as_of = now - timedelta(hours=6)
    prior = await connector.fetch("EURUSD", as_of=as_of)
    assert prior is not None
    assert pytest.approx(prior.close, rel=1e-6) == 1.05


@pytest.mark.asyncio()
async def test_intraday_connector_prefers_latest_trade(timescale_engine) -> None:
    ingestor = TimescaleIngestor(timescale_engine)
    base = datetime(2024, 1, 3, 12, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "timestamp": base,
                "symbol": "EURUSD",
                "price": 1.12,
                "size": 900,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": base + timedelta(minutes=1),
                "symbol": "EURUSD",
                "price": 1.125,
                "size": 1100,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )
    ingestor.upsert_intraday_trades(trades)

    reader = TimescaleReader(timescale_engine)
    connector = TimescaleIntradayTradeConnector(reader)

    latest = await connector.fetch("EURUSD")
    assert latest is not None
    assert pytest.approx(latest.close, rel=1e-6) == 1.125
    assert pytest.approx(latest.volume, rel=1e-6) == 1100

    earlier = await connector.fetch("EURUSD", as_of=base)
    assert earlier is not None
    assert pytest.approx(earlier.close, rel=1e-6) == 1.12


@pytest.mark.asyncio()
async def test_connectors_return_none_when_no_data(timescale_engine) -> None:
    reader = TimescaleReader(timescale_engine)
    daily = TimescaleDailyBarConnector(reader)
    intraday = TimescaleIntradayTradeConnector(reader)

    assert await daily.fetch("MISSING") is None
    assert await intraday.fetch("MISSING") is None


@pytest.mark.asyncio()
async def test_daily_connector_enriches_macro_bias(timescale_engine) -> None:
    ingestor = TimescaleIngestor(timescale_engine)
    now = datetime(2024, 1, 5, tzinfo=timezone.utc)

    bars = pd.DataFrame(
        [
            {
                "date": now,
                "open": 1.10,
                "high": 1.16,
                "low": 1.08,
                "close": 1.14,
                "adj_close": 1.13,
                "volume": 1800,
                "symbol": "EURUSD",
            }
        ]
    )
    ingestor.upsert_daily_bars(bars)

    macro_events = pd.DataFrame(
        [
            {
                "timestamp": now - timedelta(hours=4),
                "calendar": "ECB",
                "event": "Rate Decision",
                "currency": "EUR",
                "actual": 3.75,
                "forecast": 3.50,
                "previous": 3.25,
                "importance": "high",
            },
            {
                "timestamp": now - timedelta(hours=2),
                "calendar": "FOMC",
                "event": "Rate Decision",
                "currency": "USD",
                "actual": 5.00,
                "forecast": 5.25,
                "previous": 5.00,
                "importance": "high",
            },
        ]
    )
    ingestor.upsert_macro_events(macro_events)

    reader = TimescaleReader(timescale_engine)
    macro_service = TimescaleMacroEventService(reader)
    connector = TimescaleDailyBarConnector(reader, macro_service=macro_service)

    enriched = await connector.fetch("EURUSD", as_of=now)
    assert enriched is not None
    assert hasattr(enriched, "macro_bias")
    assert enriched.macro_bias > 0.05
    assert hasattr(enriched, "macro_confidence")
    assert enriched.macro_confidence >= 0.5
    events = getattr(enriched, "macro_events", None)
    assert isinstance(events, list) and len(events) >= 1
    metadata = getattr(enriched, "macro_metadata", None)
    assert isinstance(metadata, dict)
