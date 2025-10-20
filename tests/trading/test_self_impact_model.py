from datetime import datetime

import numpy as np
import pytest

from src.trading.models.trade import Trade
from src.trading.order_management.order_book.snapshot import OrderBookLevel, OrderBookSnapshot
from src.trading.performance.analytics import SelfImpactModel


def _snapshot(symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
        asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
    )


def test_self_impact_metrics_buy_trade() -> None:
    model = SelfImpactModel()

    trade = Trade(
        trade_id="T1",
        order_id="O1",
        symbol="XYZ",
        side="BUY",
        quantity=100.0,
        price=100.04,
        timestamp=datetime(2024, 1, 1, 12, 0, 1),
    )

    pre_snapshot = _snapshot(
        "XYZ",
        bids=[(99.99, 600.0), (99.98, 400.0)],
        asks=[(100.01, 500.0), (100.02, 300.0)],
    )

    post_snapshot = _snapshot(
        "XYZ",
        bids=[(100.01, 550.0), (99.99, 400.0)],
        asks=[(100.04, 450.0), (100.05, 250.0)],
    )

    pre_prices = [100.0, 100.05, 99.95, 100.02]
    post_prices = [100.025, 100.06, 100.10, 100.08]

    metrics = model.evaluate_trade(
        trade,
        pre_trade_snapshot=pre_snapshot,
        post_trade_snapshot=post_snapshot,
        pre_trade_mid_prices=pre_prices,
        post_trade_mid_prices=post_prices,
    )

    mid_before = pre_snapshot.mid_price
    mid_after = post_snapshot.mid_price

    expected_price_impact = ((mid_after - mid_before) / mid_before) * 10000.0
    expected_spread_change = (
        (post_snapshot.spread - pre_snapshot.spread) / mid_before
    ) * 10000.0
    expected_depth_consumed = (pre_snapshot.total_ask_volume - post_snapshot.total_ask_volume)
    expected_imbalance_before = (pre_snapshot.total_bid_volume - pre_snapshot.total_ask_volume) / (
        pre_snapshot.total_bid_volume + pre_snapshot.total_ask_volume
    )
    expected_imbalance_after = (post_snapshot.total_bid_volume - post_snapshot.total_ask_volume) / (
        post_snapshot.total_bid_volume + post_snapshot.total_ask_volume
    )
    expected_imbalance_change = expected_imbalance_after - expected_imbalance_before

    pre_returns = np.diff(pre_prices) / np.array(pre_prices[:-1])
    post_returns = np.diff(post_prices) / np.array(post_prices[:-1])
    expected_vol_before = np.std(pre_returns, ddof=1)
    expected_vol_after = np.std(post_returns, ddof=1)

    assert metrics.trade_id == "T1"
    assert metrics.symbol == "XYZ"
    assert metrics.price_impact_bps == pytest.approx(expected_price_impact, rel=1e-6)
    assert metrics.spread_change_bps == pytest.approx(expected_spread_change, rel=1e-6)
    assert metrics.depth_consumed == pytest.approx(expected_depth_consumed, rel=1e-6)
    assert metrics.imbalance_change == pytest.approx(expected_imbalance_change, rel=1e-6)
    assert metrics.volatility_before == pytest.approx(expected_vol_before, rel=1e-6)
    assert metrics.volatility_after == pytest.approx(expected_vol_after, rel=1e-6)
    assert metrics.volatility_change == pytest.approx(expected_vol_after - expected_vol_before, rel=1e-6)
    volatility_floor = 1e-9
    assert metrics.metadata["volatility_multiplier"] == pytest.approx(
        expected_vol_after / max(expected_vol_before, volatility_floor) if expected_vol_after > 0 else 0,
        rel=1e-6,
    )


def test_self_impact_sell_trade_with_volatility_floor() -> None:
    model = SelfImpactModel(volatility_floor=1e-3)

    trade = Trade(
        trade_id="T2",
        order_id="O2",
        symbol="ABC",
        side="SELL",
        quantity=75.0,
        price=50.00,
        timestamp=datetime(2024, 1, 1, 13, 0, 0),
    )

    pre_snapshot = _snapshot(
        "ABC",
        bids=[(50.00, 400.0), (49.98, 300.0)],
        asks=[(50.05, 500.0), (50.06, 250.0)],
    )

    post_snapshot = _snapshot(
        "ABC",
        bids=[(49.95, 250.0), (49.94, 300.0)],
        asks=[(50.05, 500.0), (50.06, 250.0)],
    )

    metrics = model.evaluate_trade(
        trade,
        pre_trade_snapshot=pre_snapshot,
        post_trade_snapshot=post_snapshot,
        pre_trade_mid_prices=[pre_snapshot.mid_price],
        post_trade_mid_prices=[post_snapshot.mid_price, post_snapshot.mid_price * 0.998, post_snapshot.mid_price * 1.002],
    )

    expected_depth_consumed = pre_snapshot.total_bid_volume - post_snapshot.total_bid_volume
    assert metrics.depth_consumed == pytest.approx(expected_depth_consumed, rel=1e-6)
    assert metrics.volatility_before == 0.0
    assert metrics.volatility_after > 0.0
    assert metrics.volatility_change == pytest.approx(metrics.volatility_after, rel=1e-6)
    assert metrics.metadata["volatility_multiplier"] == pytest.approx(
        metrics.volatility_after / 1e-3, rel=1e-6
    )


def test_trade_symbol_mismatch_raises() -> None:
    model = SelfImpactModel()

    trade = Trade(
        trade_id="T3",
        order_id="O3",
        symbol="XYZ",
        side="BUY",
        quantity=10.0,
        price=10.0,
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
    )

    pre_snapshot = _snapshot("XYZ", bids=[(10.0, 100.0)], asks=[(10.1, 100.0)])
    post_snapshot = _snapshot("ABC", bids=[(10.0, 100.0)], asks=[(10.1, 100.0)])

    with pytest.raises(ValueError):
        model.evaluate_trade(
            trade,
            pre_trade_snapshot=pre_snapshot,
            post_trade_snapshot=post_snapshot,
        )
