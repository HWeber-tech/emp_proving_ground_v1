from datetime import datetime, timedelta, timezone

import pytest

from src.core.base import MarketData
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber


@pytest.mark.asyncio()
async def test_depth_prober_generates_positive_liquidity() -> None:
    prober = DepthAwareLiquidityProber(max_history=5)
    now = datetime.now(timezone.utc)
    market_data = MarketData(
        symbol="EURUSD",
        timestamp=now,
        close=1.1010,
        volume=5000,
        depth=4200,
        spread=0.00005,
        order_imbalance=0.25,
        data_quality=0.92,
    )
    prober.record_snapshot("EURUSD", market_data)

    levels = [1.1010, 1.1012]
    results = await prober.probe_liquidity("EURUSD", levels, "BUY")

    assert results[levels[0]] > 0
    assert results[levels[0]] >= results[levels[1]]

    score = prober.calculate_liquidity_confidence_score(results, intended_volume=2000)
    assert 0.0 < score <= 1.0

    summary = prober.get_probe_summary(results)
    assert summary["evaluated_levels"] == len(levels)
    assert summary["total_liquidity"] >= results[levels[0]]
    assert summary["symbol"] == "EURUSD"


@pytest.mark.asyncio()
async def test_depth_prober_detects_liquidity_shortfall() -> None:
    prober = DepthAwareLiquidityProber(max_history=3)
    now = datetime.now(timezone.utc)
    for depth in (80, 65, 55):
        md = MarketData(
            symbol="EURUSD",
            timestamp=now,
            close=1.1005,
            volume=150.0,
            depth=depth,
            spread=0.0002,
            order_imbalance=-0.45,
            data_quality=0.6,
        )
        prober.record_snapshot("EURUSD", md)
        now += timedelta(seconds=1)

    results = await prober.probe_liquidity("EURUSD", [1.1003, 1.1005], "SELL")
    score = prober.calculate_liquidity_confidence_score(results, intended_volume=800)

    assert score < 0.4

    summary = prober.get_probe_summary(results)
    assert summary["observations_used"] >= 1
    assert summary["total_liquidity"] == pytest.approx(sum(results.values()))
