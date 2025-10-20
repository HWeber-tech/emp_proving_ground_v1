from __future__ import annotations

from datetime import datetime, timezone

from src.core.base import MarketData


def test_market_data_accepts_epoch_timestamp() -> None:
    data = MarketData(symbol="XYZ", timestamp=1_600_000_000, price=101.0)

    assert data.timestamp == datetime.utcfromtimestamp(1_600_000_000)


def test_market_data_accepts_tz_without_colon() -> None:
    iso = "2023-01-02T03:04:05+0000"

    data = MarketData(symbol="XYZ", timestamp=iso, price=101.0)

    assert data.timestamp == datetime(2023, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
