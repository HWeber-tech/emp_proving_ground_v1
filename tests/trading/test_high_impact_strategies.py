from __future__ import annotations

import numpy as np
import pytest

from src.trading.strategies import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    MomentumStrategy,
    MomentumStrategyConfig,
    MultiTimeframeMomentumConfig,
    MultiTimeframeMomentumStrategy,
    TimeframeMomentumLegConfig,
    StrategySignal,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)
from src.trading.strategies.capacity import (
    DEFAULT_L1_CAPACITY_RATIO,
    resolve_l1_depth_cap,
)


def _market(close: list[float]) -> dict[str, dict[str, list[float]]]:
    return {"EURUSD": {"close": close}}


def _mtf_market(
    *,
    daily: list[float],
    hourly: list[float],
    fifteen_min: list[float],
) -> dict[str, dict[str, object]]:
    return {
        "EURUSD": {
            "close": daily,
            "timeframes": {
                "1h": {"close": hourly},
                "15m": {"close": fifteen_min},
            },
        }
    }


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
async def test_multi_timeframe_momentum_confirms_buy_signal() -> None:
    config = MultiTimeframeMomentumConfig(
        timeframes=(
            TimeframeMomentumLegConfig("15m", lookback=4, weight=0.3, minimum_observations=6),
            TimeframeMomentumLegConfig("1h", lookback=4, weight=0.3, minimum_observations=6),
            TimeframeMomentumLegConfig("1d", lookback=4, weight=0.4, minimum_observations=6),
        ),
        entry_threshold=0.4,
        confirmation_ratio=0.66,
        target_volatility=0.12,
        max_leverage=2.0,
        volatility_timeframe="1d",
        volatility_lookback=5,
    )
    strategy = MultiTimeframeMomentumStrategy("mtf", ["EURUSD"], capital=1_250_000, config=config)

    market = _mtf_market(
        daily=[1.00, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035],
        hourly=[1.000, 1.002, 1.004, 1.007, 1.010, 1.013, 1.016],
        fifteen_min=[1.0000, 1.0006, 1.0012, 1.0020, 1.0027, 1.0035, 1.0044],
    )

    signal = await strategy.generate_signal(market, "EURUSD")

    assert signal.action == "BUY"
    assert signal.notional > 0.0
    assert signal.confidence > 0.0
    assert signal.metadata["support_ratio"] >= 2 / 3


@pytest.mark.asyncio
async def test_multi_timeframe_momentum_supports_sell_direction() -> None:
    config = MultiTimeframeMomentumConfig(
        timeframes=(
            TimeframeMomentumLegConfig("15m", lookback=4, weight=0.2, minimum_observations=6),
            TimeframeMomentumLegConfig("1h", lookback=4, weight=0.4, minimum_observations=6),
            TimeframeMomentumLegConfig("1d", lookback=4, weight=0.4, minimum_observations=6),
        ),
        entry_threshold=0.35,
        confirmation_ratio=0.6,
        target_volatility=0.10,
        max_leverage=1.75,
    )
    strategy = MultiTimeframeMomentumStrategy("mtf", ["EURUSD"], capital=900_000, config=config)

    market = _mtf_market(
        daily=[1.08, 1.07, 1.065, 1.06, 1.055, 1.05, 1.045],
        hourly=[1.080, 1.078, 1.075, 1.072, 1.069, 1.066, 1.063],
        fifteen_min=[1.0800, 1.0795, 1.0780, 1.0760, 1.0740, 1.0720, 1.0705],
    )

    signal = await strategy.generate_signal(market, "EURUSD")

    assert signal.action == "SELL"
    assert signal.notional < 0.0
    assert signal.confidence > 0.0
    assert signal.metadata["aggregate_score"] < 0.0


@pytest.mark.asyncio
async def test_multi_timeframe_momentum_handles_missing_timeframe() -> None:
    config = MultiTimeframeMomentumConfig(
        timeframes=(
            TimeframeMomentumLegConfig("15m", lookback=4, weight=0.4, minimum_observations=6),
            TimeframeMomentumLegConfig("1h", lookback=4, weight=0.3, minimum_observations=6),
            TimeframeMomentumLegConfig("1d", lookback=4, weight=0.3, minimum_observations=6),
        ),
        entry_threshold=0.4,
        confirmation_ratio=0.6,
    )
    strategy = MultiTimeframeMomentumStrategy("mtf", ["EURUSD"], capital=750_000, config=config)

    market = {
        "EURUSD": {
            "close": [1.0, 1.01, 1.015, 1.02, 1.025, 1.03],
            "timeframes": {
                "1h": {"close": [1.0, 1.001, 1.002, 1.003, 1.004, 1.005]},
            },
        }
    }

    signal = await strategy.generate_signal(market, "EURUSD")

    assert "15m" in " ".join(signal.metadata["issues"])
    assert signal.metadata["issues"]


@pytest.mark.asyncio
async def test_strategies_return_flat_when_data_missing() -> None:
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=1_000_000)
    signal = await strategy.generate_signal({"EURUSD": {"close": [1.0]}}, "EURUSD")
    assert signal.action == "FLAT"
    assert signal.notional == pytest.approx(0.0)


def test_resolve_l1_depth_cap_nested_percentiles() -> None:
    market = {
        "EURUSD": {
            "liquidity": {
                "l1": {
                    "depth_percentiles": {
                        "p50": 250_000.0,
                    }
                }
            }
        }
    }

    cap, metadata = resolve_l1_depth_cap(market, "EURUSD")
    assert cap == pytest.approx(250_000.0 * DEFAULT_L1_CAPACITY_RATIO)
    assert metadata["basis_path"].endswith("depth_percentiles.p50")
    assert metadata["cap_ratio"] == DEFAULT_L1_CAPACITY_RATIO


@pytest.mark.asyncio
async def test_momentum_strategy_respects_l1_depth_capacity() -> None:
    config = MomentumStrategyConfig(lookback=5, entry_threshold=0.4)
    strategy = MomentumStrategy("mom", ["EURUSD"], capital=5_000_000, config=config)

    closes = [1.0, 1.01, 1.02, 1.03, 1.05, 1.08, 1.12]
    market = {
        "EURUSD": {
            "close": closes,
            "liquidity": {
                "l1": {
                    "depth_percentiles": {
                        "p50": 50_000.0,
                    }
                }
            },
        }
    }

    signal = await strategy.generate_signal(market, "EURUSD")

    limit = 50_000.0 * DEFAULT_L1_CAPACITY_RATIO
    assert signal.action == "BUY"
    assert abs(signal.notional) <= limit + 1e-6
    capacity = signal.metadata.get("liquidity_capacity")
    assert capacity is not None
    assert capacity["cap"] == pytest.approx(limit)
    assert capacity["cap_applied"] is True
