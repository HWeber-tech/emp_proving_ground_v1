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
    assert first_meta.get("window_utilisation") == pytest.approx(1.0)
    assert first_meta.get("window_reset_in_seconds") == pytest.approx(60.0)
    reset_at = first_meta.get("window_reset_at")
    assert isinstance(reset_at, str)
    assert datetime.fromisoformat(reset_at) == base + timedelta(seconds=60)

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
    assert second.retry_in_seconds == pytest.approx(expected_retry_seconds)
    assert second_metadata.get("remaining_trades") == 0
    assert second_metadata.get("window_utilisation") == pytest.approx(1.0)
    expected_reset = base + timedelta(seconds=60)
    reset_at_second = second_metadata.get("window_reset_at")
    assert isinstance(reset_at_second, str)
    assert datetime.fromisoformat(reset_at_second) == expected_reset
    assert second_metadata.get("window_reset_in_seconds") == pytest.approx(
        max((expected_reset - second_time).total_seconds(), 0.0)
    )
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
        assert third.retry_in_seconds == pytest.approx(remaining)
        reset_at_third = third_meta.get("window_reset_at")
        assert isinstance(reset_at_third, str)
        reset_dt_third = datetime.fromisoformat(reset_at_third)
        assert third_meta.get("window_reset_in_seconds") == pytest.approx(
            max((reset_dt_third - third_time).total_seconds(), 0.0)
        )
    else:
        # Without cooldown, the retry time should remain anchored to the rolling window.
        assert third.allowed is False
        assert third.reason == "max_1_trades_per_60s"
        assert third.retry_at == base + timedelta(seconds=60)
        assert third.snapshot["state"] == "rate_limited"
        third_meta = third.snapshot.get("metadata", {})
        assert third_meta.get("retry_in_seconds") == pytest.approx(45.0)
        assert third.retry_in_seconds == pytest.approx(45.0)
        reset_at_third = third_meta.get("window_reset_at")
        assert isinstance(reset_at_third, str)
        reset_dt_third = datetime.fromisoformat(reset_at_third)
        assert third_meta.get("window_reset_in_seconds") == pytest.approx(
            max((reset_dt_third - third_time).total_seconds(), 0.0)
        )

    resume_time = base + timedelta(seconds=75)
    fourth = throttle.evaluate(now=resume_time, metadata={"symbol": "EURUSD"})
    assert fourth.allowed is True
    assert fourth.snapshot["state"] == "open"
    assert fourth.retry_at is None
    fourth_meta = fourth.snapshot.get("metadata", {})
    assert fourth_meta.get("window_utilisation") == pytest.approx(1.0)


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
    assert alpha_meta.get("window_utilisation") == pytest.approx(1.0)
    alpha_reset = alpha_meta.get("window_reset_at")
    assert isinstance(alpha_reset, str)
    assert datetime.fromisoformat(alpha_reset) == base + timedelta(seconds=60)
    assert alpha_meta.get("window_reset_in_seconds") == pytest.approx(60.0)

    beta_first = throttle.evaluate(
        now=base + timedelta(seconds=1),
        metadata={"strategy_id": "beta", "symbol": "EURUSD"},
    )
    assert beta_first.allowed is True, "independent scope should allow different strategy"
    beta_meta = beta_first.snapshot.get("metadata", {})
    assert beta_meta.get("window_utilisation") == pytest.approx(1.0)

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
    missing_scope_meta = missing_scope_first.snapshot.get("metadata", {})
    assert missing_scope_meta.get("window_utilisation") == pytest.approx(1.0)

    missing_scope_second = throttle.evaluate(
        now=base + timedelta(seconds=4), metadata={"symbol": "EURUSD"}
    )
    assert missing_scope_second.allowed is False
    missing_scope = missing_scope_second.snapshot.get("metadata", {}).get("scope", {})
    assert missing_scope == {"strategy_id": None}


def test_trade_throttle_scope_snapshots_expose_per_scope_state() -> None:
    config = TradeThrottleConfig(
        name="scoped-snapshot",
        max_trades=1,
        window_seconds=120.0,
        scope_fields=("strategy_id", "symbol"),
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

    throttle.evaluate(
        now=base,
        metadata={"strategy_id": "alpha", "symbol": "EURUSD"},
    )
    throttle.evaluate(
        now=base + timedelta(seconds=5),
        metadata={"strategy_id": "beta", "symbol": "GBPUSD"},
    )

    snapshots = throttle.scope_snapshots()
    assert isinstance(snapshots, tuple)
    assert len(snapshots) == 2

    scopes = {
        tuple(sorted(snapshot.get("metadata", {}).get("scope", {}).items()))
        for snapshot in snapshots
    }
    assert ("strategy_id", "alpha") in {item for scope in scopes for item in scope}
    assert ("symbol", "EURUSD") in {item for scope in scopes for item in scope}
    assert ("strategy_id", "beta") in {item for scope in scopes for item in scope}
    assert ("symbol", "GBPUSD") in {item for scope in scopes for item in scope}

    key_tokens = {
        token
        for snapshot in snapshots
        for token in snapshot.get("scope_key", [])
    }
    assert any(token.endswith("'alpha'") for token in key_tokens)
    assert any(token.endswith("'beta'") for token in key_tokens)


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
    first_meta = first.snapshot.get("metadata", {})
    assert first_meta.get("window_utilisation") == pytest.approx(0.1)

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
    assert second.retry_in_seconds == pytest.approx(20.0)
    assert metadata.get("window_utilisation") == pytest.approx(0.1)
    reset_at = metadata.get("window_reset_at")
    assert isinstance(reset_at, str)
    reset_dt = datetime.fromisoformat(reset_at)
    assert metadata.get("window_reset_in_seconds") == pytest.approx(
        max((reset_dt - (base + timedelta(seconds=10))).total_seconds(), 0.0)
    )

    third = throttle.evaluate(
        now=base + timedelta(seconds=35), metadata={"strategy_id": "alpha"}
    )
    assert third.allowed is True
    assert third.snapshot["state"] == "open"


def test_trade_throttle_prunes_stale_scoped_states() -> None:
    config = TradeThrottleConfig(
        name="scoped-prune",
        max_trades=1,
        window_seconds=60.0,
        scope_fields=("strategy_id",),
        min_spacing_seconds=45.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    for idx in range(5):
        throttle.evaluate(
            now=base + timedelta(seconds=idx),
            metadata={"strategy_id": f"alpha-{idx}"},
        )

    assert len(throttle._states) == 5  # type: ignore[attr-defined]  # intentional: verify scoped state count

    later = base + timedelta(seconds=180)
    throttle.evaluate(
        now=later,
        metadata={"strategy_id": "omega"},
    )

    state_keys = list(throttle._states.keys())  # type: ignore[attr-defined]  # intentional: ensure stale scopes pruned
    assert len(state_keys) == 1
    assert any("omega" in part for part in state_keys[0])


def test_trade_throttle_decision_surfaces_multiplier() -> None:
    config = TradeThrottleConfig(
        name="scoped",
        max_trades=2,
        window_seconds=120.0,
        multiplier=0.65,
    )
    throttle = TradeThrottle(config)

    moment = datetime(2024, 4, 10, 9, 30, tzinfo=timezone.utc)
    decision = throttle.evaluate(now=moment, metadata={"strategy_id": "alpha"})

    assert decision.allowed is True
    assert decision.multiplier == pytest.approx(0.65)

    snapshot = decision.as_dict()
    assert snapshot.get("multiplier") == pytest.approx(0.65)
    assert snapshot.get("active") is False
    metadata = snapshot.get("metadata", {})
    assert metadata.get("remaining_trades") == 1
