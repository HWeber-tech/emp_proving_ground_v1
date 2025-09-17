from __future__ import annotations

from datetime import datetime

import pytest

from src.trading.models.position import Position


def freeze_datetime(monkeypatch: pytest.MonkeyPatch, when: datetime) -> None:
    class Frozen(datetime):  # type: ignore[misc]
        @classmethod
        def now(cls) -> datetime:  # pragma: no cover - deterministic helper
            return when

    monkeypatch.setattr("src.trading.models.position.datetime", Frozen, raising=False)


def test_initialization_normalizes_aliases() -> None:
    pos = Position(symbol="EURUSD", size=2.5, entry_price=1.1)

    assert pos.quantity == pytest.approx(2.5)
    assert pos.average_price == pytest.approx(1.1)
    assert pos.current_price == pytest.approx(1.1)
    assert pos.value == pytest.approx(2.75)
    assert pos.unrealized_pnl == pytest.approx(0.0)


def test_update_price_recomputes_unrealized_and_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    pos = Position(symbol="AAPL", size=10, entry_price=100.0)
    freeze_datetime(monkeypatch, datetime(2024, 1, 1, 12, 0, 0))

    pos.update_price(104.5)

    assert pos.current_price == pytest.approx(104.5)
    assert pos.unrealized_pnl == pytest.approx(45.0)
    assert pos.last_updated == datetime(2024, 1, 1, 12, 0, 0)


def test_size_and_entry_price_setters_keep_derived_in_sync() -> None:
    pos = Position(symbol="ES", size=2, entry_price=100.0, current_price=110.0)

    pos.size = 3
    assert pos.unrealized_pnl == pytest.approx((110.0 - 100.0) * 3)

    pos.entry_price = 105.0
    assert pos.unrealized_pnl == pytest.approx((110.0 - 105.0) * 3)


def test_update_quantity_updates_average_and_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze_datetime(monkeypatch, datetime(2024, 2, 1, 8, 30, 0))
    pos = Position(symbol="ES", size=1, entry_price=100.0, current_price=110.0)

    freeze_datetime(monkeypatch, datetime(2024, 2, 1, 8, 45, 0))
    pos.update_quantity(3, new_average_price=105.0)

    assert pos.quantity == pytest.approx(3.0)
    assert pos.average_price == pytest.approx(105.0)
    assert pos.unrealized_pnl == pytest.approx((110.0 - 105.0) * 3)
    assert pos.last_updated == datetime(2024, 2, 1, 8, 45, 0)


def test_close_marks_exit_time_and_updates_price(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze_datetime(monkeypatch, datetime(2024, 3, 1, 9, 0, 0))
    pos = Position(symbol="ES", size=1, entry_price=100.0)

    freeze_datetime(monkeypatch, datetime(2024, 3, 1, 9, 5, 0))
    pos.close(95.0)

    assert pos.exit_time == datetime(2024, 3, 1, 9, 5, 0)
    assert pos.current_price == pytest.approx(95.0)
    assert pos.unrealized_pnl == pytest.approx((95.0 - pos.entry_price) * pos.size)
