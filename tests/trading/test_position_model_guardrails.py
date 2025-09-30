"""Regression tests for the lightweight trading position model.

These tests harden the coverage guardrails by validating the minimal API
exposed by :class:`src.trading.models.position.Position` and ensuring that
state transitions recompute derived values and timestamps.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from src.trading.models.position import Position


class _ControlledDatetime(datetime):
    """Deterministic datetime replacement used to assert timestamp updates."""

    _values: deque[datetime] = deque()

    @classmethod
    def prime(cls, *values: datetime) -> None:
        cls._values = deque(values)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if not cls._values:
            raise RuntimeError("_ControlledDatetime.now() called without primed values")
        value = cls._values.popleft()
        if tz is not None:
            return value.astimezone(tz)
        return value


@pytest.fixture(autouse=True)
def _restore_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the module-level datetime with the deterministic variant."""

    monkeypatch.setattr("src.trading.models.position.datetime", _ControlledDatetime)


def test_position_minimal_api_recomputes_aliases() -> None:
    base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    _ControlledDatetime.prime(base_time, base_time + timedelta(seconds=5))

    position = Position(symbol="AAPL", size=5, entry_price=100.0, current_price=102.0)

    assert position.size == 5.0
    assert position.quantity == 5.0
    assert position.entry_price == pytest.approx(100.0)
    assert position.current_price == pytest.approx(102.0)
    assert position.value == pytest.approx(510.0)
    assert position.unrealized_pnl == pytest.approx(10.0)
    assert position.last_updated == base_time

    position.size = 7
    assert position.quantity == 7.0
    assert position.unrealized_pnl == pytest.approx((102.0 - 100.0) * 7)
    assert position.last_updated == base_time  # setter does not touch the timestamp

    position.update_price(105.0)
    assert position.market_price == pytest.approx(105.0)
    assert position.current_price == pytest.approx(105.0)
    assert position.unrealized_pnl == pytest.approx((105.0 - 100.0) * 7)
    assert position.last_updated == base_time + timedelta(seconds=5)
    assert position.is_long
    assert not position.is_short
    assert not position.is_flat


def test_position_quantity_and_close_flow() -> None:
    start = datetime(2024, 3, 20, 9, 30, tzinfo=timezone.utc)
    _ControlledDatetime.prime(
        start,
        start + timedelta(minutes=1),
        start + timedelta(minutes=2),
        start + timedelta(minutes=3),
        start + timedelta(minutes=4),
    )

    position = Position(symbol="BTCUSD", size=2, entry_price=25000.0, current_price=26000.0)
    assert position.last_updated == start

    position.update_quantity(1, new_average_price=25500.0)
    assert position.size == pytest.approx(1.0)
    assert position.entry_price == pytest.approx(25500.0)
    assert position.unrealized_pnl == pytest.approx((26000.0 - 25500.0) * 1)
    assert position.last_updated == start + timedelta(minutes=1)

    position.add_realized_pnl(1500.0)
    assert position.realized_pnl == pytest.approx(1500.0)
    assert position.total_pnl == pytest.approx(1500.0 + position.unrealized_pnl)
    assert position.last_updated == start + timedelta(minutes=2)

    position.close(24000.0)
    assert position.current_price == pytest.approx(24000.0)
    assert position.unrealized_pnl == pytest.approx((24000.0 - 25500.0) * 1)
    assert position.market_value == pytest.approx(24000.0)
    assert position.total_pnl == pytest.approx(position.realized_pnl + position.unrealized_pnl)
    assert position.last_updated == start + timedelta(minutes=3)
    assert position.exit_time == start + timedelta(minutes=4)
    assert position.is_long  # quantity remains positive until explicit flatting
    assert not position.is_flat

