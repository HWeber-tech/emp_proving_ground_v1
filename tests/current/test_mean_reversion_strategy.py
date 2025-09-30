from __future__ import annotations

import pytest

from src.trading.strategies.mean_reversion import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
)


@pytest.mark.asyncio
async def test_buy_signal_when_price_breaks_lower_band() -> None:
    config = MeanReversionStrategyConfig(lookback=5, zscore_entry=1.0)
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=1_000_000.0,
        config=config,
    )
    market_data = {"EURUSD": {"close": [100, 100, 100, 95, 90]}}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal.action == "BUY"
    assert signal.confidence > 0.0
    assert signal.notional > 0.0
    metadata = signal.metadata
    assert metadata["last_price"] == pytest.approx(90.0)
    assert metadata["zscore"] <= -config.zscore_entry
    assert "mean_price" in metadata
    assert "price_std" in metadata


@pytest.mark.asyncio
async def test_sell_signal_when_price_breaks_upper_band() -> None:
    config = MeanReversionStrategyConfig(lookback=5, zscore_entry=1.0)
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=500_000.0,
        config=config,
    )
    market_data = {"EURUSD": {"close": [100, 100, 100, 105, 110]}}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal.action == "SELL"
    assert signal.confidence > 0.0
    assert signal.notional < 0.0
    metadata = signal.metadata
    assert metadata["last_price"] == pytest.approx(110.0)
    assert metadata["zscore"] >= config.zscore_entry


@pytest.mark.asyncio
async def test_returns_flat_signal_when_data_missing() -> None:
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=250_000.0,
    )
    market_data = {"EURUSD": {"close": [101.0]}}

    signal = await strategy.generate_signal(market_data, "EURUSD")

    assert signal.action == "FLAT"
    assert signal.confidence == 0.0
    assert signal.notional == 0.0
    assert signal.metadata["reason"] == "insufficient_data"
    assert "error" in signal.metadata
