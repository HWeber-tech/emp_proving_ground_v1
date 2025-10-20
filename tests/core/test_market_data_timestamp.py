from __future__ import annotations

from datetime import datetime

from src.core.base import MarketData


def test_market_data_accepts_epoch_timestamp() -> None:
    data = MarketData(symbol="XYZ", timestamp=1_600_000_000, price=101.0)

    assert data.timestamp == datetime.utcfromtimestamp(1_600_000_000)
