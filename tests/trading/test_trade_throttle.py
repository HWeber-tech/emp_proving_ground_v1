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


def test_trade_throttle_enforces_notional_limit() -> None:
    config = TradeThrottleConfig(
        name="notional",
        max_trades=5,
        window_seconds=60.0,
        max_notional=1_000.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

    first = throttle.evaluate(
        now=base,
        metadata={"symbol": "EURUSD", "strategy_id": "alpha", "notional": 600.0},
    )
    assert first.allowed is True
    assert first.applied_notional == pytest.approx(600.0)
    meta_first = first.snapshot.get("metadata", {})
    assert meta_first.get("consumed_notional") == pytest.approx(600.0)
    assert meta_first.get("remaining_notional") == pytest.approx(400.0)
    assert meta_first.get("notional_utilisation") == pytest.approx(0.6)

    second = throttle.evaluate(
        now=base + timedelta(seconds=5),
        metadata={"symbol": "EURUSD", "strategy_id": "alpha", "notional": 500.0},
    )
    assert second.allowed is False
    assert second.reason == "max_notional_1000_per_60s"
    assert second.snapshot["state"] == "notional_limit"
    message = second.snapshot.get("message")
    assert isinstance(message, str)
    assert "notional" in message.lower()
    meta_second = second.snapshot.get("metadata", {})
    assert meta_second.get("remaining_notional") == pytest.approx(400.0)
    assert meta_second.get("notional_utilisation") == pytest.approx(0.6)
    assert meta_second.get("attempted_notional") == pytest.approx(500.0)

    resume = throttle.evaluate(
        now=base + timedelta(seconds=65),
        metadata={"symbol": "EURUSD", "strategy_id": "alpha", "notional": 500.0},
    )
    assert resume.allowed is True
    meta_resume = resume.snapshot.get("metadata", {})
    assert meta_resume.get("consumed_notional") == pytest.approx(500.0)
    assert meta_resume.get("remaining_notional") == pytest.approx(500.0)


def test_trade_throttle_rollback_restores_notional_budget() -> None:
    config = TradeThrottleConfig(
        name="rollback-notional",
        max_trades=3,
        window_seconds=120.0,
        max_notional=2_000.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 2, 1, 14, 0, 0, tzinfo=timezone.utc)

    decision = throttle.evaluate(
        now=base,
        metadata={"symbol": "GBPUSD", "notional": 1_250.0},
    )
    assert decision.allowed is True
    meta_before = throttle.snapshot().get("metadata", {})
    assert meta_before.get("consumed_notional") == pytest.approx(1_250.0)
    assert meta_before.get("remaining_notional") == pytest.approx(750.0)

    throttle.rollback(decision)

    snapshot_after = throttle.snapshot()
    meta_after = snapshot_after.get("metadata", {})
    assert meta_after.get("consumed_notional") == pytest.approx(0.0)
    assert meta_after.get("remaining_notional") == pytest.approx(2_000.0)
    assert meta_after.get("notional_utilisation") == pytest.approx(0.0)


def test_trade_throttle_handles_fractional_window() -> None:
    config = TradeThrottleConfig(
        name="fractional",
        max_trades=1,
        window_seconds=2.5,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    first = throttle.evaluate(now=base, metadata={"symbol": "EURUSD"})
    assert first.allowed is True

    second_time = base + timedelta(seconds=1.0)
    blocked = throttle.evaluate(now=second_time, metadata={"symbol": "EURUSD"})
    assert blocked.allowed is False
    assert blocked.reason == "max_1_trades_per_2.5s"

    message = blocked.snapshot.get("message")
    assert isinstance(message, str)
    assert "Throttled: too many trades in short time" in message
    assert "2.5 seconds" in message

    retry_seconds = blocked.snapshot.get("metadata", {}).get("retry_in_seconds")
    assert retry_seconds == pytest.approx(1.5)
    assert blocked.retry_in_seconds == pytest.approx(1.5)


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


def test_trade_throttle_rollback_restores_capacity() -> None:
    config = TradeThrottleConfig(
        name="rollback",
        max_trades=1,
        window_seconds=30.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    metadata = {"symbol": "EURUSD"}

    first = throttle.evaluate(now=base, metadata=metadata)
    assert first.allowed is True
    assert first.scope_key == ("__global__",)

    blocked = throttle.evaluate(now=base + timedelta(seconds=5), metadata=metadata)
    assert blocked.allowed is False
    assert blocked.snapshot["state"] == "rate_limited"

    snapshot = throttle.rollback(first)
    assert snapshot is not None
    assert snapshot["state"] == "open"
    snapshot_meta = snapshot.get("metadata", {})
    assert snapshot_meta.get("recent_trades") == 0
    assert snapshot_meta.get("remaining_trades") == 1

    reopened = throttle.evaluate(now=base + timedelta(seconds=6), metadata=metadata)
    assert reopened.allowed is True
    assert reopened.snapshot["state"] == "open"


def test_trade_throttle_external_cooldown_blocks_and_expires() -> None:
    config = TradeThrottleConfig(
        name="backlog_guard",
        max_trades=5,
        window_seconds=120.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert throttle.evaluate(now=base).allowed is True

    cooldown_snapshot = throttle.apply_external_cooldown(
        30.0,
        reason="backlog_cooldown",
        message="Throttled: backlog cooldown active",
        metadata={"lag_ms": 1250.0, "threshold_ms": 250.0},
        now=base + timedelta(seconds=2),
    )
    assert cooldown_snapshot["state"] == "cooldown"
    cooldown_meta = cooldown_snapshot.get("metadata", {})
    assert cooldown_meta.get("cooldown_reason") == "backlog_cooldown"
    context = cooldown_meta.get("cooldown_context", {})
    assert context.get("lag_ms") == pytest.approx(1250.0)

    blocked = throttle.evaluate(now=base + timedelta(seconds=3))
    assert blocked.allowed is False
    assert blocked.reason == "backlog_cooldown"
    blocked_meta = blocked.snapshot.get("metadata", {})
    assert blocked_meta.get("cooldown_reason") == "backlog_cooldown"

    resumed = throttle.evaluate(now=base + timedelta(seconds=40))
    assert resumed.allowed is True


def test_trade_throttle_auto_multiplier_override() -> None:
    config = TradeThrottleConfig(
        name="auto",
        max_trades=3,
        window_seconds=30.0,
    )
    throttle = TradeThrottle(config)

    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    initial = throttle.evaluate(now=base, metadata={"symbol": "EURUSD"})
    assert initial.allowed is True
    assert initial.multiplier is None

    throttle.set_auto_multiplier(0.0, reason="backpressure", metadata={"source": "test"})

    second = throttle.evaluate(now=base + timedelta(seconds=1), metadata={"symbol": "EURUSD"})
    assert second.allowed is True
    assert second.multiplier == 0.0
    meta = second.snapshot.get("metadata", {})
    assert meta.get("auto_multiplier") == 0.0
    assert meta.get("auto_multiplier_reason") == "backpressure"
    context = meta.get("auto_multiplier_context")
    assert isinstance(context, dict)
    assert context.get("source") == "test"

    throttle.set_auto_multiplier(None)

    third = throttle.evaluate(now=base + timedelta(seconds=2), metadata={"symbol": "EURUSD"})
    assert third.allowed is True
    assert third.multiplier is None
