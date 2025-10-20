from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.trading.strategies import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    StrategySignal,
)
from src.trading.strategies.signals.ict_microstructure import (
    ICTMicrostructureAnalyzer,
)


@pytest.fixture()
def strategy_config() -> MeanReversionStrategyConfig:
    return MeanReversionStrategyConfig(lookback=5, zscore_entry=1.0)


@pytest.fixture()
def analyzer() -> ICTMicrostructureAnalyzer:
    return ICTMicrostructureAnalyzer()


@pytest.fixture()
def strategy(strategy_config: MeanReversionStrategyConfig) -> MeanReversionStrategy:
    return MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=1_000_000,
        config=strategy_config,
    )


def _market(prices: list[float]) -> dict[str, dict[str, list[float]]]:
    return {"EURUSD": {"close": prices}}


@pytest.mark.asyncio
async def test_buy_signal_when_price_breaks_lower_band(
    strategy: MeanReversionStrategy,
) -> None:
    closes = [100.0, 100.0, 100.0, 100.0, 95.0, 90.0]

    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert isinstance(signal, StrategySignal)
    assert signal.action == "BUY"
    assert signal.notional > 0.0
    assert signal.metadata["zscore"] <= -1.0


@pytest.mark.asyncio
async def test_sell_signal_when_price_breaks_upper_band(
    strategy: MeanReversionStrategy,
) -> None:
    closes = [100.0, 100.0, 100.0, 100.0, 105.0, 110.0]

    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "SELL"
    assert signal.notional < 0.0
    assert signal.metadata["zscore"] >= 1.0


@pytest.mark.asyncio
async def test_hold_signal_when_data_missing(
    strategy: MeanReversionStrategy,
) -> None:
    closes = [100.0]

    signal = await strategy.generate_signal(_market(closes), "EURUSD")

    assert signal.action == "FLAT"
    assert signal.notional == pytest.approx(0.0)
    assert signal.metadata["reason"] == "insufficient_data"


@pytest.mark.asyncio
async def test_microstructure_alignment_adjusts_confidence(
    strategy_config: MeanReversionStrategyConfig,
    analyzer: ICTMicrostructureAnalyzer,
) -> None:
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=500_000,
        config=strategy_config,
        microstructure_analyzer=analyzer,
    )
    upward_prices = np.linspace(1.0, 1.05, num=10).tolist()
    market = {
        "EURUSD": {
            "close": upward_prices,
            "orderbook": {
                "bids": np.linspace(0.5, 0.6, num=10).tolist(),
                "asks": np.linspace(0.61, 0.7, num=10).tolist(),
            },
        }
    }

    signal = await strategy.generate_signal(market, "EURUSD")

    alignment = signal.metadata["microstructure"]["alignment"]
    assert -1.0 <= alignment["score"] <= 1.0
    assert alignment["breakdown"]
    assert 0.0 <= signal.confidence <= 1.0


@pytest.mark.asyncio
async def test_inventory_state_pushes_towards_flat() -> None:
    config = MeanReversionStrategyConfig(
        lookback=5,
        zscore_entry=1.0,
        inventory_half_life_minutes=1e-4,
        turnover_cap_per_minute=1.0,
        turnover_cap_per_hour=1.0,
    )
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=1_000_000,
        config=config,
    )

    buy_prices = [100.0, 100.0, 100.0, 100.0, 95.0, 90.0]
    first_signal = await strategy.generate_signal(_market(buy_prices), "EURUSD")
    assert first_signal.action == "BUY"
    assert first_signal.notional > 0.0

    state = strategy._inventory["EURUSD"]
    state.last_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    state.minute_turnover.clear()
    state.hour_turnover.clear()
    state.minute_total = 0.0
    state.hour_total = 0.0

    flat_prices = [100.0] * (config.lookback + 1)
    second_signal = await strategy.generate_signal(_market(flat_prices), "EURUSD")

    assert second_signal.action == "SELL"
    assert second_signal.notional < 0.0
    inventory_meta = second_signal.metadata["inventory_state"]
    assert inventory_meta["prior_position"] > inventory_meta["net_position"]
    assert inventory_meta["turnover_limited"] is False


@pytest.mark.asyncio
async def test_inventory_turnover_caps_limit_delta() -> None:
    config = MeanReversionStrategyConfig(
        lookback=5,
        zscore_entry=1.0,
        inventory_half_life_minutes=1e-4,
        turnover_cap_per_minute=0.01,
        turnover_cap_per_hour=0.05,
    )
    strategy = MeanReversionStrategy(
        "mean-revert",
        ["EURUSD"],
        capital=1_000_000,
        config=config,
    )

    seed_prices = [100.0, 100.0, 100.0, 100.0, 95.0, 90.0]
    await strategy.generate_signal(_market(seed_prices), "EURUSD")

    state = strategy._inventory["EURUSD"]
    state.net_position = 125_000.0
    state.last_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
    state.minute_turnover.clear()
    state.hour_turnover.clear()
    state.minute_total = 0.0
    state.hour_total = 0.0

    neutral_prices = [100.0] * (config.lookback + 1)
    signal = await strategy.generate_signal(_market(neutral_prices), "EURUSD")

    assert signal.action == "SELL"
    assert abs(signal.notional) == pytest.approx(strategy._minute_turnover_cap, rel=1e-6)
    inventory_meta = signal.metadata["inventory_state"]
    assert inventory_meta["turnover_limited"] is True
    assert "minute" in inventory_meta["turnover_limited_by"]
    assert inventory_meta["net_position"] == pytest.approx(
        125_000.0 - strategy._minute_turnover_cap
    )
