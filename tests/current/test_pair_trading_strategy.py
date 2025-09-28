from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.trading.strategies.pairs import PairTradingConfig, PairTradingStrategy


@pytest.fixture()
def base_market_data() -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0, 0.4, size=220)) + 100.0
    hedge = base * 0.85 + rng.normal(0, 0.15, size=220)
    return {
        "ALPHA": {"close": base.tolist()},
        "BETA": {"close": hedge.tolist()},
    }


def _make_strategy(**overrides: object) -> PairTradingStrategy:
    cfg = PairTradingConfig(
        lookback=160,
        zscore_entry=1.2,
        zscore_exit=0.3,
        adf_stat_threshold=float(overrides.pop("adf_stat_threshold", -1.0)),
        min_half_life=0.05,
        max_half_life=200.0,
        max_notional_fraction=0.05,
    )
    return PairTradingStrategy(
        strategy_id="pair-alpha-beta",
        symbols=["ALPHA", "BETA"],
        capital=float(overrides.pop("capital", 1_000_000.0)),
        config=cfg,
    )


@pytest.mark.asyncio
async def test_pair_trading_emits_long_spread_signals(base_market_data: dict[str, dict[str, list[float]]]) -> None:
    market_data = deepcopy(base_market_data)
    market_data["ALPHA"]["close"][-1] -= 2.5  # depress spread for long signal

    strategy = _make_strategy()

    primary_signal = await strategy.generate_signal(market_data, "ALPHA")
    hedge_signal = await strategy.generate_signal(market_data, "BETA")

    assert primary_signal.action == "BUY"
    assert hedge_signal.action == "SELL"
    assert primary_signal.notional > 0
    assert hedge_signal.notional < 0
    expected_hedge = abs(primary_signal.notional) * abs(primary_signal.metadata["hedge_ratio"])
    assert pytest.approx(abs(hedge_signal.notional), rel=0.25) == expected_hedge
    assert primary_signal.confidence > 0
    assert "reason" not in primary_signal.metadata


@pytest.mark.asyncio
async def test_pair_trading_emits_short_spread_signals(base_market_data: dict[str, dict[str, list[float]]]) -> None:
    market_data = deepcopy(base_market_data)
    market_data["ALPHA"]["close"][-1] += 3.0  # elevate spread for short signal

    strategy = _make_strategy()

    primary_signal = await strategy.generate_signal(market_data, "ALPHA")
    hedge_signal = await strategy.generate_signal(market_data, "BETA")

    assert primary_signal.action == "SELL"
    assert hedge_signal.action == "BUY"
    assert primary_signal.notional < 0
    assert hedge_signal.notional > 0
    expected_hedge = abs(primary_signal.notional) * abs(primary_signal.metadata["hedge_ratio"])
    assert pytest.approx(abs(hedge_signal.notional), rel=0.25) == expected_hedge
    assert primary_signal.confidence > 0


@pytest.mark.asyncio
async def test_pair_trading_blocks_when_cointegration_fails(base_market_data: dict[str, dict[str, list[float]]]) -> None:
    rng = np.random.default_rng(7)
    independent = np.cumsum(rng.normal(0, 1.0, size=220)) + 50.0
    market_data = deepcopy(base_market_data)
    market_data["BETA"]["close"] = independent.tolist()

    strategy = _make_strategy(adf_stat_threshold=-3.5)

    signal = await strategy.generate_signal(market_data, "ALPHA")
    assert signal.action == "FLAT"
    assert signal.notional == 0
    assert signal.metadata["reason"] == "failed_cointegration_test"
