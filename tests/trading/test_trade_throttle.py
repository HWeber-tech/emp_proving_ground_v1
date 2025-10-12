"""Unit tests for the TradeThrottle rate limiting utility."""

from datetime import datetime, timedelta, timezone

import pytest

from src.trading.execution.trade_throttle import TradeThrottle, TradeThrottleConfig


@pytest.mark.parametrize(
    "cooldown_seconds",
    [0.0, 30.0],
)
def test_trade_throttle_enforces_window_and_retry(cooldown_seconds: float) -> None:
    config = TradeThrottleConfig(
        name="governance",
        max_trades=1,
        window_seconds=60.0,
        cooldown_seconds=cooldown_seconds,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    first = throttle.evaluate(now=base, metadata={"symbol": "EURUSD"})
    assert first.allowed is True
    assert first.snapshot["state"] == "open"
    assert first.retry_at is None
    first_meta = first.snapshot.get("metadata", {})
    assert first_meta.get("remaining_trades") == 0
    assert first_meta.get("retry_in_seconds") is None

    second_time = base + timedelta(seconds=5)
    second = throttle.evaluate(now=second_time, metadata={"symbol": "EURUSD"})
    assert second.allowed is False
    assert second.snapshot["state"] == "rate_limited"
    assert second.reason == "max_1_trades_per_60s"
    retry_expected = (
        second_time + timedelta(seconds=cooldown_seconds)
        if cooldown_seconds
        else base + timedelta(seconds=60)
    )
    assert second.retry_at == retry_expected
    second_metadata = second.snapshot.get("metadata", {})
    assert second_metadata.get("recent_trades") == 1
    expected_retry_seconds = (
        cooldown_seconds if cooldown_seconds else 55.0
    )
    assert second_metadata.get("retry_in_seconds") == pytest.approx(
        expected_retry_seconds
    )
    assert second_metadata.get("remaining_trades") == 0
    context = second_metadata.get("context", {})
    assert context.get("symbol") == "EURUSD"

    third_time = second_time + timedelta(seconds=10)
    third = throttle.evaluate(now=third_time, metadata={"symbol": "EURUSD"})
    if cooldown_seconds:
        assert third.allowed is False
        assert third.reason == "cooldown_active"
        assert third.retry_at == second.retry_at
        assert third.snapshot["state"] == "cooldown"
        assert third.retry_at is not None
        remaining = max((third.retry_at - third_time).total_seconds(), 0.0)
        third_meta = third.snapshot.get("metadata", {})
        assert third_meta.get("retry_in_seconds") == pytest.approx(remaining)
    else:
        # Without cooldown, the retry time should remain anchored to the rolling window.
        assert third.allowed is False
        assert third.reason == "max_1_trades_per_60s"
        assert third.retry_at == base + timedelta(seconds=60)
        assert third.snapshot["state"] == "rate_limited"
        third_meta = third.snapshot.get("metadata", {})
        assert third_meta.get("retry_in_seconds") == pytest.approx(45.0)

    resume_time = base + timedelta(seconds=75)
    fourth = throttle.evaluate(now=resume_time, metadata={"symbol": "EURUSD"})
    assert fourth.allowed is True
    assert fourth.snapshot["state"] == "open"
    assert fourth.retry_at is None


def test_trade_throttle_scopes_by_metadata_field() -> None:
    config = TradeThrottleConfig(
        name="scoped",
        max_trades=1,
        window_seconds=60.0,
        scope_fields=("strategy_id",),
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    alpha_first = throttle.evaluate(
        now=base, metadata={"strategy_id": "alpha", "symbol": "EURUSD"}
    )
    assert alpha_first.allowed is True
    alpha_meta = alpha_first.snapshot.get("metadata", {})
    assert alpha_meta.get("remaining_trades") == 0

    beta_first = throttle.evaluate(
        now=base + timedelta(seconds=1),
        metadata={"strategy_id": "beta", "symbol": "EURUSD"},
    )
    assert beta_first.allowed is True, "independent scope should allow different strategy"

    alpha_second = throttle.evaluate(
        now=base + timedelta(seconds=2),
        metadata={"strategy_id": "alpha", "symbol": "EURUSD"},
    )
    assert alpha_second.allowed is False
    assert alpha_second.reason == "max_1_trades_per_60s"
    alpha_scope = alpha_second.snapshot.get("metadata", {}).get("scope", {})
    assert alpha_scope == {"strategy_id": "alpha"}
    alpha_key = alpha_second.snapshot.get("scope_key")
    assert alpha_key is not None
    assert isinstance(alpha_key, list) and alpha_key[0].endswith("'alpha'")

    missing_scope_first = throttle.evaluate(
        now=base + timedelta(seconds=3), metadata={"symbol": "EURUSD"}
    )
    assert missing_scope_first.allowed is True

    missing_scope_second = throttle.evaluate(
        now=base + timedelta(seconds=4), metadata={"symbol": "EURUSD"}
    )
    assert missing_scope_second.allowed is False
    missing_scope = missing_scope_second.snapshot.get("metadata", {}).get("scope", {})
    assert missing_scope == {"strategy_id": None}


def test_trade_throttle_enforces_minimum_spacing() -> None:
    config = TradeThrottleConfig(
        name="spacing",
        max_trades=10,
        window_seconds=300.0,
        min_spacing_seconds=30.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

    first = throttle.evaluate(now=base, metadata={"strategy_id": "alpha"})
    assert first.allowed is True
    assert first.snapshot["state"] == "open"

    second = throttle.evaluate(
        now=base + timedelta(seconds=10), metadata={"strategy_id": "alpha"}
    )
    assert second.allowed is False
    assert second.snapshot["state"] == "min_interval"
    assert second.reason == "min_interval_30s"
    assert second.retry_at == base + timedelta(seconds=30)
    assert second.snapshot.get("message", "").startswith(
        "Throttled: minimum interval of"
    )
    metadata = second.snapshot.get("metadata", {})
    assert metadata.get("min_spacing_seconds") == 30.0
    assert metadata.get("recent_trades") == 1
    assert metadata.get("retry_in_seconds") == pytest.approx(20.0)

    third = throttle.evaluate(
        now=base + timedelta(seconds=35), metadata={"strategy_id": "alpha"}
    )
    assert third.allowed is True
    assert third.snapshot["state"] == "open"
