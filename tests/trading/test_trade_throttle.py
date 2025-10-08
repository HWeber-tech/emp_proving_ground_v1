from datetime import datetime, timedelta, timezone

import pytest

from src.trading.monitoring.trade_throttle import TradeThrottle, TradeThrottleConfig


def test_trade_throttle_blocks_when_window_exceeded() -> None:
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    throttle = TradeThrottle(TradeThrottleConfig(max_trades=2, window_seconds=60.0))

    first = throttle.evaluate(now=start)
    second = throttle.evaluate(now=start + timedelta(seconds=10))
    third = throttle.evaluate(now=start + timedelta(seconds=20))

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.retry_after_seconds == pytest.approx(40.0)
    assert third.snapshot is not None
    assert third.snapshot.active is True
    assert third.snapshot.state == "throttled"


def test_trade_throttle_allows_after_window_rolls() -> None:
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    throttle = TradeThrottle(TradeThrottleConfig(max_trades=1, window_seconds=30.0))

    throttle.evaluate(now=start)
    blocked = throttle.evaluate(now=start + timedelta(seconds=5))
    assert blocked.allowed is False

    allowed = throttle.evaluate(now=start + timedelta(seconds=35))
    assert allowed.allowed is True
    assert allowed.snapshot is not None
    assert allowed.snapshot.active is False


def test_trade_throttle_enforces_minimum_interval() -> None:
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    throttle = TradeThrottle(
        TradeThrottleConfig(
            max_trades=5,
            window_seconds=120.0,
            min_interval_seconds=30.0,
        )
    )

    first = throttle.evaluate(now=start)
    second = throttle.evaluate(now=start + timedelta(seconds=10))

    assert first.allowed is True
    assert second.allowed is False
    assert second.retry_after_seconds == pytest.approx(20.0)
    assert second.snapshot is not None
    assert second.snapshot.metadata["min_interval_seconds"] == pytest.approx(30.0)
