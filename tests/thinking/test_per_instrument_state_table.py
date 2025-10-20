from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.thinking.models.per_instrument_state_table import PerInstrumentStateTable


class _FrozenClock:
    """Simple controllable clock for deterministic TTL tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


@pytest.fixture()
def clock() -> _FrozenClock:
    return _FrozenClock()


def test_pin_returns_pinned_state(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(clock=clock.now)
    state = table.pin("AAPL", session="us")
    state["hidden"] = [1, 2, 3]

    cached = table.get("AAPL")
    assert cached is state
    assert cached["hidden"] == [1, 2, 3]

    alt = table.pin("AAPL", factory=list)
    assert alt is state  # factory ignored because entry already exists


def test_ttl_expiration_and_refresh(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(default_ttl=timedelta(seconds=5), clock=clock.now)
    table.pin("AAPL")

    clock.advance(4)
    assert table.get("AAPL") is not None

    clock.advance(2)
    assert table.get("AAPL") is None
    reset = table.last_reset("AAPL")
    assert reset is not None
    assert reset.reason == "ttl_expired"

    # Fresh entry and TTL refresh when pinning again before expiry
    state = table.pin("AAPL")
    clock.advance(4)
    assert table.pin("AAPL") is state
    clock.advance(4)
    assert table.get("AAPL") is state


def test_session_boundary_resets_state(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(clock=clock.now)
    state = table.pin("AAPL", session="asia")

    # Same session keeps the state
    assert table.apply_market_event("AAPL", session="asia") is False
    assert table.get("AAPL") is state

    # Different session resets
    assert table.apply_market_event("AAPL", session="london") is True
    assert table.get("AAPL") is None

    reset = table.last_reset("AAPL")
    assert reset is not None
    assert reset.reason == "session_boundary"
    assert reset.details["previous_session"] == "asia"
    assert reset.details["session"] == "london"


def test_gap_reset_threshold(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(
        clock=clock.now,
        gap_reset_threshold=timedelta(seconds=10),
    )
    table.pin("MSFT")

    assert table.apply_market_event("MSFT", gap_seconds=5) is False
    assert table.get("MSFT") is not None

    assert table.apply_market_event("MSFT", gap_seconds=15) is True
    assert table.get("MSFT") is None

    reset = table.last_reset("MSFT")
    assert reset is not None
    assert reset.reason == "gap"
    assert pytest.approx(reset.details["gap_seconds"]) == 15.0
    assert pytest.approx(reset.details["threshold_seconds"]) == 10.0


def test_halt_resets_state(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(clock=clock.now)
    table.pin("TSLA", session="ny")

    assert table.apply_market_event("TSLA", halted=True) is True
    assert table.get("TSLA") is None

    reset = table.last_reset("TSLA")
    assert reset is not None
    assert reset.reason == "halt"
    assert reset.details["session"] == "ny"


def test_purge_expired_bulk(clock: _FrozenClock) -> None:
    table = PerInstrumentStateTable(default_ttl=timedelta(seconds=10), clock=clock.now)
    table.pin("AAPL")
    table.pin("MSFT")
    clock.advance(11)

    removed = table.purge_expired()
    assert removed == 2
    assert len(list(table.iter_states())) == 0
