from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.sensory.how.order_book_analytics import (
    OrderBookAnalytics,
    OrderBookAnalyticsConfig,
)
from src.sensory.how.how_sensor import HowSensor


def _sample_order_book() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bid_price": [1.0998, 1.0996, 1.0994, 1.0992, 1.0990],
            "ask_price": [1.1000, 1.1002, 1.1004, 1.1006, 1.1008],
            "bid_size": [3.2, 2.8, 2.4, 2.0, 1.8],
            "ask_size": [2.4, 2.2, 1.9, 1.8, 1.6],
        }
    )


def test_order_book_analytics_reports_value_area() -> None:
    config = OrderBookAnalyticsConfig(depth_levels=3, depth_embedding_dim=12)
    analytics = OrderBookAnalytics(config)
    snapshot = analytics.describe(_sample_order_book())

    assert snapshot is not None
    assert 0.0 <= snapshot.imbalance <= 1.0
    assert snapshot.total_bid_volume > snapshot.total_ask_volume
    assert snapshot.value_area_low < snapshot.value_area_high
    assert snapshot.as_dict()["participation_ratio"] > 0.5
    assert len(snapshot.depth_embedding) == config.depth_embedding_dim
    assert any(abs(value) > 0 for value in snapshot.depth_embedding)


def test_how_sensor_incorporates_order_book_metrics() -> None:
    sensor = HowSensor()
    market_frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)],
            "symbol": ["EURUSD"],
            "open": [1.10],
            "high": [1.1005],
            "low": [1.0995],
            "close": [1.10],
            "volume": [2500],
            "volatility": [0.0004],
            "spread": [0.0002],
            "depth": [12000],
            "order_imbalance": [0.12],
            "data_quality": [0.9],
        }
    )

    signals = sensor.process(market_frame, order_book=_sample_order_book())

    assert len(signals) == 1
    signal = signals[0]
    metadata = signal.metadata or {}
    order_book_meta = metadata.get("order_book")
    assert isinstance(order_book_meta, dict)
    assert {"imbalance", "value_area_low", "value_area_high"} <= set(order_book_meta)
    assert order_book_meta.get("has_depth") is True
    depth_keys = [key for key in sensor.order_book_metric_names if "depth_embedding_" in key]
    assert isinstance(order_book_meta.get("depth_embedding"), list)
    assert len(order_book_meta["depth_embedding"]) == len(depth_keys)
    assert signal.value.get("order_book_participation_ratio") is not None
    assert "participation_ratio" not in signal.value
    assert signal.value.get("has_depth") == 1.0
    depth_keys = [key for key in sensor.order_book_metric_names if "depth_embedding_" in key]
    for key in depth_keys:
        assert key in signal.value


def test_how_sensor_zero_masks_order_book_metrics_without_depth() -> None:
    sensor = HowSensor()
    market_frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)],
            "symbol": ["EURUSD"],
            "open": [1.10],
            "high": [1.1005],
            "low": [1.0995],
            "close": [1.10],
            "volume": [2500],
            "volatility": [0.0004],
            "spread": [0.0002],
            "depth": [12000],
            "order_imbalance": [0.12],
            "data_quality": [0.9],
        }
    )

    signal = sensor.process(market_frame)[0]
    assert signal.value.get("has_depth") == 0.0

    expected_keys = set(sensor.order_book_metric_names)
    for key in expected_keys:
        assert key in signal.value
        assert signal.value[key] == 0.0

    metadata = signal.metadata or {}
    assert "order_book" not in metadata
