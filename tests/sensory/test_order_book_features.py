from datetime import datetime, timezone

import pytest

from src.sensory.how.order_book_features import compute_order_book_metrics
from src.trading.order_management.order_book.snapshot import (
    OrderBookLevel,
    OrderBookSnapshot,
)


def _build_snapshot() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="EURUSD",
        timestamp=datetime.now(tz=timezone.utc),
        bids=[
            OrderBookLevel(price=1.0999, volume=4_800),
            OrderBookLevel(price=1.0998, volume=3_500),
            OrderBookLevel(price=1.0997, volume=2_000),
        ],
        asks=[
            OrderBookLevel(price=1.1001, volume=5_200),
            OrderBookLevel(price=1.1002, volume=3_900),
            OrderBookLevel(price=1.1003, volume=2_500),
        ],
    )


def test_compute_order_book_metrics_from_snapshot() -> None:
    snapshot = _build_snapshot()

    metrics = compute_order_book_metrics(snapshot)

    assert metrics.total_bid_volume == pytest.approx(10_300)
    assert metrics.total_ask_volume == pytest.approx(11_600)
    assert -1.0 <= metrics.imbalance <= 1.0
    assert metrics.spread > 0
    assert metrics.mid_price == pytest.approx((1.0999 + 1.1001) / 2)
    assert len(metrics.volume_profile) == 6
    assert metrics.as_payload()["volume_profile"]


def test_compute_order_book_metrics_from_mapping() -> None:
    book = {
        "bids": [(100.0, 50.0), (99.5, 40.0)],
        "asks": [(100.5, 30.0), (101.0, 25.0)],
    }

    metrics = compute_order_book_metrics(book, depth_normaliser=200.0)

    assert metrics.top_of_book_liquidity == pytest.approx(80.0)
    assert 0.0 < metrics.depth_liquidity <= 1.0
    payload = metrics.as_payload()
    assert payload["order_imbalance"] == pytest.approx(metrics.imbalance)
