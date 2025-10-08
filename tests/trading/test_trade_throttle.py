from datetime import datetime, timedelta, timezone

import pytest

from src.trading.throttle.trade_throttle import TradeThrottle


def test_trade_throttle_blocks_excess_trades() -> None:
    throttle = TradeThrottle(
        name="unit_test",
        max_trades_per_window=1,
        window_seconds=60.0,
    )

    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    decision = throttle.evaluate(start)
    assert decision.allowed
    throttle.register_trade(start)

    blocked = throttle.evaluate(start + timedelta(seconds=10))
    assert not blocked.allowed
    assert blocked.reason == "rate_limit"
    assert blocked.wait_seconds is not None and blocked.wait_seconds > 0
    assert blocked.snapshot.active is True
    assert blocked.snapshot.state == "rate_limited"
    assert blocked.snapshot.metadata["trades_in_window"] == 1

    allowed = throttle.evaluate(start + timedelta(seconds=120))
    assert allowed.allowed


def test_trade_throttle_enforces_min_interval() -> None:
    throttle = TradeThrottle(
        name="cooldown",
        max_trades_per_window=5,
        window_seconds=120.0,
        min_interval_seconds=30.0,
    )

    first = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    throttle.register_trade(first)

    cooldown = throttle.evaluate(first + timedelta(seconds=5))
    assert not cooldown.allowed
    assert cooldown.reason == "minimum_interval"
    assert pytest.approx(cooldown.wait_seconds or 0.0, rel=1e-3) == 25.0
    assert cooldown.snapshot.active is True
    assert cooldown.snapshot.state == "cooldown"
    assert cooldown.snapshot.metadata["cooldown_seconds"] == pytest.approx(25.0)

    allowed = throttle.evaluate(first + timedelta(seconds=45))
    assert allowed.allowed
