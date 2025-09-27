from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trading.order_management import PositionTracker


def _ts(hour: int) -> datetime:
    return datetime(2025, 1, 1, hour, tzinfo=timezone.utc)


def test_fifo_tracker_realized_and_unrealized_pnl() -> None:
    tracker = PositionTracker(pnl_mode="fifo")

    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))
    tracker.record_fill("AAPL", 5, 110.0, timestamp=_ts(10))
    tracker.update_mark_price("AAPL", 115.0, timestamp=_ts(11))
    tracker.record_fill("AAPL", -8, 120.0, timestamp=_ts(12))

    snapshot = tracker.get_position_snapshot("AAPL")

    assert snapshot.net_quantity == pytest.approx(7.0)
    assert snapshot.realized_pnl == pytest.approx(160.0)
    assert snapshot.unrealized_pnl == pytest.approx(55.0)
    assert snapshot.exposure == pytest.approx(805.0)
    assert snapshot.market_value == pytest.approx(805.0)


def test_lifo_tracker_realized_pnl_changes() -> None:
    tracker = PositionTracker(pnl_mode="lifo")

    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))
    tracker.record_fill("AAPL", 5, 110.0, timestamp=_ts(10))
    tracker.record_fill("AAPL", -8, 120.0, timestamp=_ts(11))

    snapshot = tracker.get_position_snapshot("AAPL")

    assert snapshot.net_quantity == pytest.approx(7.0)
    assert snapshot.realized_pnl == pytest.approx(110.0)
    assert snapshot.market_price == pytest.approx(120.0)
    assert snapshot.unrealized_pnl == pytest.approx(140.0)


def test_short_position_accounting_and_unrealized_pnl() -> None:
    tracker = PositionTracker()

    tracker.record_fill("MSFT", -5, 50.0, timestamp=_ts(9))
    tracker.update_mark_price("MSFT", 45.0, timestamp=_ts(10))
    tracker.record_fill("MSFT", 3, 40.0, timestamp=_ts(11))

    snapshot = tracker.get_position_snapshot("MSFT")

    assert snapshot.net_quantity == pytest.approx(-2.0)
    assert snapshot.realized_pnl == pytest.approx(30.0)
    assert snapshot.unrealized_pnl == pytest.approx(10.0)
    assert snapshot.exposure == pytest.approx(90.0)


def test_reconciliation_report_highlights_differences() -> None:
    tracker = PositionTracker()
    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))

    report = tracker.generate_reconciliation_report({"AAPL": 8.0, "MSFT": 1.0})

    assert report.has_discrepancies() is True
    assert {(diff.symbol, round(diff.difference, 6)) for diff in report.differences} == {
        ("AAPL", 2.0),
        ("MSFT", -1.0),
    }


def test_accounts_and_reset() -> None:
    tracker = PositionTracker()
    tracker.record_fill("AAPL", 1, 100.0, account="alpha", timestamp=_ts(9))
    tracker.record_fill("MSFT", 1, 50.0, account="beta", timestamp=_ts(10))

    assert tracker.accounts() == ("alpha", "beta")
    assert tracker.total_exposure() > 0

    tracker.reset()

    assert tracker.accounts() == ()
    assert tracker.total_exposure() == 0.0
