from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from src.data_foundation.ingest.yahoo_gateway import YahooMarketDataGateway


def test_fetch_data_logs_value_errors(monkeypatch, caplog) -> None:
    gateway = YahooMarketDataGateway()

    def fake_fetch(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise ValueError("bad range")

    monkeypatch.setattr(
        "src.data_foundation.ingest.yahoo_gateway.fetch_price_history",
        fake_fetch,
    )

    with caplog.at_level(logging.WARNING):
        result = gateway.fetch_data("AAPL")

    assert result is None
    assert "bad range" in caplog.text


def test_fetch_data_reraises_unexpected_errors(monkeypatch) -> None:
    gateway = YahooMarketDataGateway()

    def fake_fetch(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.data_foundation.ingest.yahoo_gateway.fetch_price_history",
        fake_fetch,
    )

    with pytest.raises(RuntimeError):
        gateway.fetch_data("AAPL")


def test_fetch_data_returns_frame(monkeypatch) -> None:
    gateway = YahooMarketDataGateway()
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "close": [1.0]})

    monkeypatch.setattr(
        "src.data_foundation.ingest.yahoo_gateway.fetch_price_history",
        lambda *a, **k: frame,
    )

    result = gateway.fetch_data("AAPL")
    assert result is frame


@pytest.mark.asyncio
async def test_get_market_data_delegates_to_thread(monkeypatch) -> None:
    gateway = YahooMarketDataGateway()
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "close": [1.0]})

    monkeypatch.setattr(
        "src.data_foundation.ingest.yahoo_gateway.fetch_price_history",
        lambda *a, **k: frame,
    )

    result = await gateway.get_market_data("AAPL")
    assert result is frame
