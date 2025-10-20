from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.trading.execution.manipulation_sentinel import ManipulationSentinel


def _ts(offset_seconds: float = 0.0) -> datetime:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def test_manipulation_sentinel_flags_spoof_pattern_and_cooldown() -> None:
    sentinel = ManipulationSentinel(
        window_seconds=90.0,
        min_large_notional=10_000.0,
        min_pattern_orders=3,
        small_execution_ratio=0.25,
        cooldown_seconds=30.0,
        min_small_notional=500.0,
    )

    # Three large buy orders that fail
    for idx, offset in enumerate((0.0, 10.0, 20.0), start=1):
        event_id = f"L{idx}"
        sentinel.observe_submission(
            event_id=event_id,
            symbol="EURUSD",
            side="BUY",
            notional=15_000.0 + idx * 500.0,
            timestamp=_ts(offset),
            strategy_id="alpha",
        )
        sentinel.mark_outcome(
            event_id,
            executed=False,
            timestamp=_ts(offset + 1.0),
            reason="cancelled",
        )

    # Small executed sell
    sentinel.observe_submission(
        event_id="S1",
        symbol="EURUSD",
        side="SELL",
        notional=1_500.0,
        timestamp=_ts(25.0),
        strategy_id="alpha",
    )
    sentinel.mark_outcome(
        "S1",
        executed=True,
        timestamp=_ts(26.0),
        fill_notional=1_500.0,
    )

    block = sentinel.should_block(
        symbol="EURUSD",
        side="BUY",
        notional=12_000.0,
        timestamp=_ts(30.0),
        event_id="B1",
        strategy_id="alpha",
    )
    assert block is not None
    assert block["reason"] == "manipulation.spoof_pattern"
    assert block["pattern"] == "spoof"
    assert block["spoof_order_count"] == 3
    assert block["executed_order_side"] == "SELL"

    sentinel.record_block(
        event_id="B1",
        symbol="EURUSD",
        side="BUY",
        notional=12_000.0,
        timestamp=_ts(30.0),
        reason=block["reason"],
        metadata=block,
    )

    cooldown = sentinel.should_block(
        symbol="EURUSD",
        side="BUY",
        notional=11_000.0,
        timestamp=_ts(40.0),
        event_id="B2",
        strategy_id="alpha",
    )
    assert cooldown is not None
    assert cooldown["reason"] == "manipulation.cooldown"
    assert cooldown["pattern"] == "spoof"

    unblock = sentinel.should_block(
        symbol="EURUSD",
        side="BUY",
        notional=11_000.0,
        timestamp=_ts(65.0),
        event_id="B3",
        strategy_id="alpha",
    )
    assert unblock is None


def test_manipulation_sentinel_requires_new_activity_after_marker() -> None:
    sentinel = ManipulationSentinel(
        window_seconds=120.0,
        min_large_notional=8_000.0,
        min_pattern_orders=2,
        small_execution_ratio=0.3,
        cooldown_seconds=15.0,
        min_small_notional=400.0,
    )

    sentinel.observe_submission(
        event_id="X1",
        symbol="GBPUSD",
        side="SELL",
        notional=9_000.0,
        timestamp=_ts(0.0),
    )
    sentinel.mark_outcome("X1", executed=False, timestamp=_ts(1.0))
    sentinel.observe_submission(
        event_id="X2",
        symbol="GBPUSD",
        side="SELL",
        notional=10_000.0,
        timestamp=_ts(12.0),
    )
    sentinel.mark_outcome("X2", executed=False, timestamp=_ts(13.0))
    sentinel.observe_submission(
        event_id="X3",
        symbol="GBPUSD",
        side="BUY",
        notional=1_000.0,
        timestamp=_ts(16.0),
    )
    sentinel.mark_outcome("X3", executed=True, timestamp=_ts(17.0), fill_notional=1_000.0)

    block = sentinel.should_block(
        symbol="GBPUSD",
        side="SELL",
        notional=9_500.0,
        timestamp=_ts(20.0),
        event_id="Y1",
    )
    assert block is not None
    sentinel.record_block(
        event_id="Y1",
        symbol="GBPUSD",
        side="SELL",
        notional=9_500.0,
        timestamp=_ts(20.0),
        reason=block["reason"],
        metadata=block,
    )

    # Without new large activity, sentinel should stand down after cooldown expiry
    post_cooldown = sentinel.should_block(
        symbol="GBPUSD",
        side="SELL",
        notional=9_500.0,
        timestamp=_ts(50.0),
        event_id="Y2",
    )
    assert post_cooldown is None

    # Fresh cancellation reactivates the sentinel
    sentinel.observe_submission(
        event_id="Z1",
        symbol="GBPUSD",
        side="SELL",
        notional=8_500.0,
        timestamp=_ts(60.0),
    )
    sentinel.mark_outcome("Z1", executed=False, timestamp=_ts(61.0))

    sentinel.observe_submission(
        event_id="Z2",
        symbol="GBPUSD",
        side="BUY",
        notional=800.0,
        timestamp=_ts(65.0),
    )
    sentinel.mark_outcome(
        "Z2",
        executed=True,
        timestamp=_ts(66.0),
        fill_notional=800.0,
    )

    rebound = sentinel.should_block(
        symbol="GBPUSD",
        side="SELL",
        notional=8_000.0,
        timestamp=_ts(70.0),
        event_id="Z3",
    )
    assert rebound is not None
    assert rebound["reason"] == "manipulation.spoof_pattern"
