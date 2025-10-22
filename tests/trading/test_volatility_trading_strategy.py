import math

import pytest

from src.trading.strategies import (
    StrategySignal,
    VolatilityTradingConfig,
    VolatilityTradingStrategy,
)


@pytest.mark.asyncio
async def test_volatility_trading_long_when_realised_exceeds_implied() -> None:
    config = VolatilityTradingConfig(
        realised_lookback=10,
        vol_spread_entry=0.03,
        target_volatility=0.15,
    )
    strategy = VolatilityTradingStrategy(
        "vol_long",
        ["ES"],
        capital=1_000_000,
        config=config,
    )

    closes = [100.0, 103.0, 97.0, 106.0, 94.0, 108.0, 92.0, 111.0, 90.0, 115.0, 88.0]
    market = {"ES": {"close": closes, "implied_volatility": 0.12}}

    signal = await strategy.generate_signal(market, "ES")

    assert isinstance(signal, StrategySignal)
    assert signal.action == "BUY"
    assert signal.notional > 0.0
    assert signal.metadata["volatility_spread"] > 0.0
    assert signal.confidence > 0.0
    assert "gamma_scalping" in signal.metadata


@pytest.mark.asyncio
async def test_volatility_trading_short_when_implied_dominates() -> None:
    config = VolatilityTradingConfig(realised_lookback=8, vol_spread_entry=0.02)
    strategy = VolatilityTradingStrategy(
        "vol_short",
        ["ES"],
        capital=750_000,
        config=config,
    )

    closes = [100.0, 100.5, 99.8, 100.2, 100.1, 100.4, 99.9, 100.3, 100.2]
    market = {"ES": {"close": closes, "implied_volatility": 0.35}}

    signal = await strategy.generate_signal(market, "ES")

    assert signal.action == "SELL"
    assert signal.notional < 0.0
    assert signal.metadata["volatility_spread"] < 0.0


@pytest.mark.asyncio
async def test_volatility_trading_handles_missing_implied_vol() -> None:
    strategy = VolatilityTradingStrategy(
        "vol_missing",
        ["ES"],
        capital=500_000,
    )

    closes = [100.0, 101.0, 102.0]
    market = {"ES": {"close": closes}}

    signal = await strategy.generate_signal(market, "ES")

    assert signal.action == "FLAT"
    assert signal.metadata["reason"] == "missing_implied_vol"
    assert signal.notional == 0.0
    assert math.isclose(signal.confidence, 0.0)
