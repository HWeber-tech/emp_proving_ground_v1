from __future__ import annotations

import pytest

from src.core.base import MarketData

def test_mid_price_ignores_non_finite_quotes() -> None:
    data = MarketData(symbol="XYZ", bid=float("nan"), ask=101.0, close=99.0)
    assert data.mid_price == 101.0


def test_mid_price_falls_back_to_close_when_quotes_invalid() -> None:
    data = MarketData(symbol="XYZ", bid=float("nan"), ask=float("nan"), close=99.0)
    assert data.mid_price == 99.0


def test_market_data_coerces_localized_numeric_strings() -> None:
    data = MarketData(symbol="XYZ", bid="1,234.5", ask="1,235.5", close="1,233.0")

    assert data.bid == pytest.approx(1234.5)
    assert data.ask == pytest.approx(1235.5)
    assert data.close == pytest.approx(1233.0)
    assert data.mid_price == pytest.approx(1235.0)


def test_market_data_ignores_boolean_inputs_for_quotes() -> None:
    data = MarketData(symbol="XYZ", bid=True, ask=False, close=12.5)

    assert data.bid == pytest.approx(0.0)
    assert data.ask == pytest.approx(0.0)
    assert data.close == pytest.approx(12.5)
