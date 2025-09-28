from __future__ import annotations

import numpy as np
import pytest

from src.trading.strategies import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    MomentumStrategy,
    MomentumStrategyConfig,
    StrategySignal,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)


def _market(close: list[float]) -> dict[str, dict[str, list[float]]]:
    return {"EURUSD": {"close": close}}


@pytest.mark.asyncio
async def test_momentum_strategy_generates_buy_signal() -> None:
    config = MomentumStrategyConfig(lookback=5, entry_threshold=0.5)
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=1_000_000, config=config)

    closes = [1.0, 1.01, 1.015, 1.02, 1.03, 1.05]
    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert isinstance(signal, StrategySignal)
    assert signal.action == "BUY"
    assert signal.confidence > 0.0
    assert signal.notional > 0.0
    assert signal.metadata["momentum_score"] > 0.0


@pytest.mark.asyncio
async def test_momentum_strategy_generates_sell_signal() -> None:
    config = MomentumStrategyConfig(lookback=5, entry_threshold=0.5)
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=1_000_000, config=config)

    closes = [1.05, 1.04, 1.03, 1.02, 1.01, 0.995]
    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "SELL"
    assert signal.notional < 0.0
    assert signal.confidence > 0.0


@pytest.mark.asyncio
async def test_mean_reversion_strategy_prefers_reversion() -> None:
    config = MeanReversionStrategyConfig(lookback=10, zscore_entry=0.8)
    strategy = MeanReversionStrategy("mr", ["EURUSD"], capital=750_000, config=config)

    closes = [100.0] * 9 + [103.0]
    sell_signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert sell_signal.action == "SELL"
    assert sell_signal.notional < 0.0

    closes = [100.0] * 9 + [97.0]
    buy_signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert buy_signal.action == "BUY"
    assert buy_signal.notional > 0.0


@pytest.mark.asyncio
async def test_volatility_breakout_detects_price_channel_breach() -> None:
    config = VolatilityBreakoutConfig(
        breakout_lookback=5,
        baseline_lookback=20,
        price_channel_lookback=5,
        volatility_multiplier=1.1,
    )
    strategy = VolatilityBreakoutStrategy("vol", ["EURUSD"], capital=500_000, config=config)

    base = [1.0] * 20
    closes = base + [1.0, 1.05, 1.12, 1.2, 1.28]

    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "BUY"
    assert signal.notional > 0.0
    assert signal.confidence > 0.0


@pytest.mark.asyncio
async def test_strategies_return_flat_when_data_missing() -> None:
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=1_000_000)
    signal = await strategy.generate_signal({"EURUSD": {"close": [1.0]}}, "EURUSD")
    assert signal.action == "FLAT"
    assert signal.notional == pytest.approx(0.0)
