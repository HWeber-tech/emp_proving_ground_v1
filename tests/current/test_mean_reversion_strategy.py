from __future__ import annotations

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
