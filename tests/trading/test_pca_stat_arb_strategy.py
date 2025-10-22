import math

import numpy as np
import pytest

from src.trading.strategies import (
    PCAStatArbConfig,
    PCAStatArbStrategy,
    StrategySignal,
)


@pytest.mark.asyncio
async def test_pca_stat_arb_generates_market_neutral_positions() -> None:
    config = PCAStatArbConfig(lookback=40, n_components=2, entry_zscore=0.6, exit_zscore=0.25)
    strategy = PCAStatArbStrategy(
        "stat_arb_test",
        ["AAA", "BBB", "CCC", "DDD"],
        capital=2_000_000,
        config=config,
    )

    base = np.linspace(100.0, 105.0, 60)
    prices = {
        "AAA": base.copy(),
        "BBB": base.copy(),
        "CCC": base.copy(),
        "DDD": base.copy(),
    }

    # Introduce a strong cross-sectional divergence toward the end of the window.
    prices["AAA"][-3:] *= np.array([1.02, 1.05, 1.20])
    prices["BBB"][-3:] *= np.array([0.98, 0.95, 0.80])
    prices["CCC"][-3:] *= np.array([1.00, 1.00, 0.99])
    prices["DDD"][-3:] *= np.array([1.00, 1.02, 1.04])

    market = {symbol: {"close": series} for symbol, series in prices.items()}

    signals = {
        symbol: await strategy.generate_signal(market, symbol)
        for symbol in strategy.symbols
    }

    for signal in signals.values():
        assert isinstance(signal, StrategySignal)
        assert signal.metadata["components"] >= 0.0
        assert signal.metadata["variance_explained"] >= 0.0

    gross = sum(abs(signal.notional) for signal in signals.values())
    assert gross > 0.0
    assert math.isclose(sum(signal.notional for signal in signals.values()), 0.0, abs_tol=1e-6)

    active = [signal for signal in signals.values() if signal.action != "FLAT"]
    assert len(active) >= 2

    long_notional = sum(signal.notional for signal in active if signal.notional > 0)
    short_notional = sum(signal.notional for signal in active if signal.notional < 0)
    assert long_notional > 0.0
    assert short_notional < 0.0


@pytest.mark.asyncio
async def test_pca_stat_arb_handles_insufficient_history() -> None:
    strategy = PCAStatArbStrategy(
        "stat_arb_short_history",
        ["AAA", "BBB", "CCC"],
        capital=500_000,
    )

    market = {
        "AAA": {"close": [100.0, 101.0]},
        "BBB": {"close": [99.0, 100.0]},
        "CCC": {"close": [98.0]},
    }

    signal = await strategy.generate_signal(market, "AAA")

    assert signal.action == "FLAT"
    assert signal.notional == 0.0
    assert signal.metadata["reason"].startswith("insufficient_history")
