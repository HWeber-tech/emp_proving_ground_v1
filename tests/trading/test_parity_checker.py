from __future__ import annotations

import logging
from typing import Any, Mapping

import pytest

from src.trading.monitoring.parity_checker import ParityChecker


class _BrokenOrder:
    order_id = "order-1"

    @property
    def status(self) -> str:
        raise ValueError("unavailable status")


class _DummyFixManager:
    def get_all_orders(self) -> Mapping[str, Any]:
        return {"order-1": _BrokenOrder()}


def test_check_orders_logs_and_counts_failures(caplog: pytest.LogCaptureFixture) -> None:
    checker = ParityChecker(_DummyFixManager())

    caplog.set_level(logging.WARNING)
    mismatches = checker.check_orders({"order-1": {"order_id": "order-1", "status": "OPEN"}})

    assert mismatches == 1
    assert any("Order parity comparison failed" in record.message for record in caplog.records)


def test_compare_order_fields_logs_and_sets_error(caplog: pytest.LogCaptureFixture) -> None:
    checker = ParityChecker(_DummyFixManager())
    caplog.set_level(logging.WARNING)

    diffs = checker.compare_order_fields(_BrokenOrder(), {"status": "OPEN"})

    assert diffs["error"] == "compare_failed"
    assert any("Order field comparison failed" in record.message for record in caplog.records)


def test_check_positions_logs_tracker_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.WARNING)

    class ExplodingTracker:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.trading.monitoring.portfolio_tracker.PortfolioTracker", ExplodingTracker
    )

    checker = ParityChecker(_DummyFixManager())
    mismatches = checker.check_positions({})

    assert mismatches == 0
    assert any("Failed to load local positions" in record.message for record in caplog.records)
