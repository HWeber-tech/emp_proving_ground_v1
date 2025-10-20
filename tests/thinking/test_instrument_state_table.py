from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.thinking.models import InstrumentStateEvent, InstrumentStateTable


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_pin_returns_pinned_state() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None)

    first = table.pin("EURUSD", lambda: {"value": 1})
    first["value"] = 42

    second = table.pin("EURUSD", lambda: {"value": 2})

    assert first is second
    assert second["value"] == 42


def test_ttl_expiry_removes_state() -> None:
    clock = _FakeClock()
    table = InstrumentStateTable(default_ttl_seconds=5, clock=clock.now)

    table.put("GBPUSD", {"state": 1})

    clock.advance(3)
    assert table.get("GBPUSD") == {"state": 1}

    clock.advance(3)
    assert table.get("GBPUSD") is None
    assert "GBPUSD" not in table


def test_session_boundary_resets_state() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None)

    session_a = InstrumentStateEvent(timestamp=None, session_id="asia", gap_seconds=None, halted=False)
    state = table.pin("USDJPY", lambda: {"seq": 1}, event=session_a)
    assert table.get("USDJPY", event=session_a) is state

    session_b = InstrumentStateEvent(timestamp=None, session_id="new_york", gap_seconds=None, halted=False)
    assert table.get("USDJPY", event=session_b) is None

    replaced = table.pin("USDJPY", lambda: {"seq": 2}, event=session_b)
    assert replaced is not state


def test_gap_threshold_resets_state() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None, gap_reset_seconds=30)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    event_initial = InstrumentStateEvent(timestamp=start.timestamp(), session_id=None, gap_seconds=None, halted=False)
    state = table.pin("AUDUSD", lambda: {"seq": 0}, event=event_initial)

    within_gap = InstrumentStateEvent(
        timestamp=(start + timedelta(seconds=20)).timestamp(),
        session_id=None,
        gap_seconds=None,
        halted=False,
    )
    assert table.get("AUDUSD", event=within_gap) is state

    over_gap = InstrumentStateEvent(
        timestamp=(start + timedelta(seconds=65)).timestamp(),
        session_id=None,
        gap_seconds=None,
        halted=False,
    )
    assert table.get("AUDUSD", event=over_gap) is None


def test_explicit_gap_event_forces_reset() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None)
    table.put("MSFT", {"v": 1})

    assert table.get("MSFT", event={"gap_seconds": 15}) is None


def test_market_halt_resets_state() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None)
    table.put("AAPL", {"h": 1})

    assert table.get("AAPL", event={"halted": True}) is None
    table.put("AAPL", {"h": 2}, event={"halted": True})
    assert table.get("AAPL") == {"h": 2}


def test_prune_expired_returns_count() -> None:
    clock = _FakeClock()
    table = InstrumentStateTable(default_ttl_seconds=2, clock=clock.now)
    table.put("BTCUSD", {"state": 1})
    table.put("ETHUSD", {"state": 2})

    clock.advance(1)
    table.put("BTCUSD", {"state": 3})  # refresh TTL

    clock.advance(1.5)
    removed = table.prune_expired()

    assert removed == 1
    assert "BTCUSD" in table
    assert "ETHUSD" not in table


def test_invalid_ttl_rejected() -> None:
    table = InstrumentStateTable(default_ttl_seconds=None)
    with pytest.raises(ValueError):
        table.put("XAUUSD", {"state": 1}, ttl_seconds=0)
