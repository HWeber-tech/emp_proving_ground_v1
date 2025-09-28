from __future__ import annotations

import pytest

from src.trading.strategies import (
    DonchianATRBreakoutConfig,
    DonchianATRBreakoutStrategy,
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    MomentumStrategy,
    MomentumStrategyConfig,
    MultiTimeframeMomentumConfig,
    MultiTimeframeMomentumStrategy,
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
async def test_multi_timeframe_momentum_requires_alignment() -> None:
    config = MultiTimeframeMomentumConfig(
        lookbacks=(5, 10, 20),
        entry_threshold=0.25,
        min_alignment=2,
    )
    strategy = MultiTimeframeMomentumStrategy(
        "mtm",
        ["EURUSD"],
        capital=1_500_000,
        config=config,
    )

    closes = [1.0 + 0.003 * idx for idx in range(25)]
    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "BUY"
    assert signal.confidence > 0.0
    assert signal.notional > 0.0
    assert signal.metadata["alignment"]["positive"] >= 2


@pytest.mark.asyncio
async def test_multi_timeframe_momentum_flats_on_misalignment() -> None:
    config = MultiTimeframeMomentumConfig(lookbacks=(3, 6, 12), entry_threshold=0.2, min_alignment=3)
    strategy = MultiTimeframeMomentumStrategy(
        "mtm",
        ["EURUSD"],
        capital=500_000,
        config=config,
    )

    closes = [
        1.20,
        1.18,
        1.16,
        1.14,
        1.12,
        1.10,
        1.08,
        1.06,
        1.05,
        1.04,
        1.045,
        1.05,
        1.055,
        1.06,
        1.065,
        1.07,
    ]
    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "FLAT"
    assert signal.notional == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_donchian_breakout_produces_trailing_stop() -> None:
    config = DonchianATRBreakoutConfig(
        channel_lookback=5,
        atr_lookback=5,
        breakout_buffer_atr=0.2,
        trailing_stop_atr=1.5,
    )
    strategy = DonchianATRBreakoutStrategy(
        "donchian",
        ["EURUSD"],
        capital=2_000_000,
        config=config,
    )

    closes = [1.0, 1.01, 1.012, 1.015, 1.017, 1.02, 1.025, 1.06]
    market = {"EURUSD": {"close": closes}}
    signal = await strategy.generate_signal(market, "EURUSD")

    assert signal.action == "BUY"
    assert signal.metadata["trailing_stop"] is not None
    assert signal.metadata["trailing_stop"] < closes[-1]


@pytest.mark.asyncio
async def test_donchian_breakout_handles_missing_data() -> None:
    strategy = DonchianATRBreakoutStrategy("donchian", ["EURUSD"], capital=1_000_000)
    signal = await strategy.generate_signal({"EURUSD": {"close": [1.0, 1.01]}}, "EURUSD")

    assert signal.action == "FLAT"
    assert signal.metadata["reason"] == "insufficient_data"


@pytest.mark.asyncio
async def test_strategies_return_flat_when_data_missing() -> None:
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=1_000_000)
    signal = await strategy.generate_signal({"EURUSD": {"close": [1.0]}}, "EURUSD")
    assert signal.action == "FLAT"
    assert signal.notional == pytest.approx(0.0)
