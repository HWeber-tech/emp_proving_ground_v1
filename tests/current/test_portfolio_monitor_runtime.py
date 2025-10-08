from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.event_bus import EventBus, get_global_bus
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor


def test_portfolio_monitor_in_memory_state_tracks_positions() -> None:
    monitor = PortfolioMonitor(EventBus(), InMemoryRedis())

    state = monitor.get_state()
    assert state["open_positions_count"] == 0
    assert state["equity"] == monitor.get_total_value()

    monitor.reserve_position("EURUSD", 1.5, 1.1)
    state = monitor.get_state()
    assert state["open_positions_count"] == 1
    assert "EURUSD" in state["open_positions"]

    monitor.release_position("EURUSD", 1.5)
    state = monitor.get_state()
    assert state["open_positions_count"] == 0
    assert "EURUSD" not in state["open_positions"]


@pytest.mark.asyncio()
async def test_portfolio_monitor_updates_pnl_and_drawdown() -> None:
    monitor = PortfolioMonitor(EventBus(), InMemoryRedis())

    initial_state = monitor.get_state()
    assert initial_state["equity"] == pytest.approx(100000.0)
    assert initial_state["peak_equity"] == pytest.approx(100000.0)

    buy_report = SimpleNamespace(symbol="EURUSD", side="BUY", quantity=50, price=100.0)
    await monitor.on_execution_report(buy_report)

    monitor.portfolio["open_positions"]["EURUSD"]["last_price"] = 90.0
    state = monitor.get_state()
    assert state["unrealized_pnl"] == pytest.approx(-500.0)
    assert state["daily_pnl"] == pytest.approx(-500.0)
    assert state["current_daily_drawdown"] == pytest.approx(0.005, rel=1e-6)

    sell_partial = SimpleNamespace(symbol="EURUSD", side="SELL", quantity=20, price=120.0)
    await monitor.on_execution_report(sell_partial)
    state = monitor.get_state()
    assert state["realized_pnl"] == pytest.approx(400.0)
    assert state["unrealized_pnl"] == pytest.approx(600.0)
    assert state["total_pnl"] == pytest.approx(1000.0)
    assert state["daily_pnl"] == pytest.approx(1000.0)
    assert state["peak_equity"] == pytest.approx(101000.0)

    monitor.portfolio["open_positions"]["EURUSD"]["last_price"] = 85.0
    state = monitor.get_state()
    assert state["unrealized_pnl"] == pytest.approx(-450.0)
    assert state["total_pnl"] == pytest.approx(-50.0)
    assert state["current_daily_drawdown"] == pytest.approx(0.0005, rel=1e-6)

    sell_final = SimpleNamespace(symbol="EURUSD", side="SELL", quantity=30, price=80.0)
    await monitor.on_execution_report(sell_final)
    state = monitor.get_state()
    assert state["realized_pnl"] == pytest.approx(-200.0)
    assert state["unrealized_pnl"] == pytest.approx(0.0)
    assert state["total_pnl"] == pytest.approx(-200.0)
    assert state["equity"] == pytest.approx(99800.0)
    assert state["daily_pnl"] == pytest.approx(-200.0)
    assert state["current_daily_drawdown"] == pytest.approx(0.002, rel=1e-6)
    assert state["peak_equity"] == pytest.approx(101000.0)


def test_portfolio_monitor_emits_cache_metrics_events() -> None:
    global_bus = get_global_bus()
    captured: list[dict[str, object]] = []

    handle = global_bus.subscribe_topic(
        "telemetry.cache", lambda _topic, payload: captured.append(dict(payload))
    )
    try:
        monitor = PortfolioMonitor(EventBus(), InMemoryRedis())
        monitor._save_state_to_redis()
    finally:
        global_bus.unsubscribe(handle)

    assert captured, "Expected cache telemetry to be published"
    latest = captured[-1]
    assert latest["cache_key"] == "emp:portfolio_state"
    assert "hits" in latest and "misses" in latest
    assert latest.get("sets", 0) >= 1
    assert latest.get("keys", 0) >= 0
    policy = latest.get("policy")
    assert isinstance(policy, dict)
    assert "ttl_seconds" in policy and "max_keys" in policy
