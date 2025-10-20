from __future__ import annotations

from src.core.base import MarketData

def test_mid_price_ignores_non_finite_quotes() -> None:
    data = MarketData(symbol="XYZ", bid=float("nan"), ask=101.0, close=99.0)
    assert data.mid_price == 101.0


def test_mid_price_falls_back_to_close_when_quotes_invalid() -> None:
    data = MarketData(symbol="XYZ", bid=float("nan"), ask=float("nan"), close=99.0)
    assert data.mid_price == 99.0
