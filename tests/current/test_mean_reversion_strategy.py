from __future__ import annotations

import pytest

from src.core.strategy.templates.mean_reversion import MeanReversionStrategy


@pytest.mark.asyncio
async def test_buy_signal_when_price_breaks_lower_band() -> None:
    strategy = MeanReversionStrategy("mean-revert", ["EURUSD"], {"period": 5, "num_std": 2})
    market_data = {"EURUSD": [100, 100, 100, 100, 80]}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal["signal"] == "BUY"
    assert signal["reason"] == "price_below_lower_band"
    assert pytest.approx(signal["z_score"], rel=1e-6) == -2.0
    assert signal["lower_band"] <= signal["price"]


@pytest.mark.asyncio
async def test_sell_signal_when_price_breaks_upper_band() -> None:
    strategy = MeanReversionStrategy("mean-revert", ["EURUSD"], {"period": 5, "num_std": 2})
    market_data = {"EURUSD": [100, 100, 100, 100, 120]}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal["signal"] == "SELL"
    assert signal["reason"] == "price_above_upper_band"
    assert pytest.approx(signal["z_score"], rel=1e-6) == 2.0


@pytest.mark.asyncio
async def test_hold_when_history_insufficient() -> None:
    strategy = MeanReversionStrategy("mean-revert", ["EURUSD"], {"period": 5})
    market_data = {"EURUSD": [101, 102]}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal == {
        "symbol": "EURUSD",
        "signal": "HOLD",
        "reason": "insufficient_history",
        "observations": 2,
        "required_history": 5,
    }


@pytest.mark.asyncio
async def test_extracts_close_from_dict_records() -> None:
    strategy = MeanReversionStrategy("mean-revert", ["EURUSD"], {"period": 5, "num_std": 2})
    market_data = {
        "EURUSD": [
            {"close": 100},
            {"close": 100},
            {"close": 100},
            {"close": 100},
            {"close": 80},
        ]
    }

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal["signal"] == "BUY"
    assert signal["reason"] == "price_below_lower_band"
    assert signal["observations"] == 5
