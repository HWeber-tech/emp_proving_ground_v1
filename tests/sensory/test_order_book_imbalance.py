from datetime import datetime, timezone

import pandas as pd

from src.sensory.how.how_sensor import HowSensor
from src.sensory.how.order_book_imbalance import compute_order_book_imbalance
from src.trading.order_management.order_book.snapshot import (
    OrderBookLevel,
    OrderBookSnapshot,
)


def _build_snapshot() -> OrderBookSnapshot:
    timestamp = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    bids = [
        OrderBookLevel(price=1.1000, volume=2_000),
        OrderBookLevel(price=1.0995, volume=1_200),
        OrderBookLevel(price=1.0990, volume=800),
    ]
    asks = [
        OrderBookLevel(price=1.1005, volume=1_100),
        OrderBookLevel(price=1.1010, volume=900),
        OrderBookLevel(price=1.1015, volume=700),
    ]
    return OrderBookSnapshot(symbol="EURUSD", timestamp=timestamp, bids=bids, asks=asks)


def test_compute_order_book_imbalance_returns_expected_ratio() -> None:
    snapshot = _build_snapshot()

    metrics = compute_order_book_imbalance(snapshot)

    assert metrics.has_volume
    assert metrics.total_volume == snapshot.total_bid_volume + snapshot.total_ask_volume
    expected_imbalance = (snapshot.total_bid_volume - snapshot.total_ask_volume) / metrics.total_volume
    assert abs(metrics.imbalance - expected_imbalance) < 1e-9


def test_how_sensor_derives_order_imbalance_from_snapshot() -> None:
    snapshot = _build_snapshot()

    rows = [
        {
            "timestamp": snapshot.timestamp,
            "symbol": snapshot.symbol,
            "open": snapshot.mid_price,
            "high": snapshot.mid_price + 0.0006,
            "low": snapshot.mid_price - 0.0005,
            "close": snapshot.mid_price,
            "volume": snapshot.total_bid_volume + snapshot.total_ask_volume,
            "volatility": 0.0005,
            "spread": snapshot.spread,
            "depth": snapshot.total_bid_volume + snapshot.total_ask_volume,
            "data_quality": 0.9,
            "order_book_snapshot": snapshot,
        }
    ]
    frame = pd.DataFrame(rows)

    sensor = HowSensor()
    signals = sensor.process(frame)

    assert len(signals) == 1
    value = signals[0].value
    assert "book_buy_volume" in value
    assert "book_sell_volume" in value
    assert value["book_total_volume"] == snapshot.total_bid_volume + snapshot.total_ask_volume
    assert value["imbalance"] == value["imbalance"]  # sanity check for NaN
