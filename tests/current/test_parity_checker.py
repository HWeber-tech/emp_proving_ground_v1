"""Regression coverage for the FIX parity checker utilities."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from src.core import telemetry
from src.trading.monitoring.parity_checker import ParityChecker


class _RecordingSink:
    """Collects gauge updates so tests can assert emissions."""

    def __init__(self) -> None:
        self.gauges: list[tuple[str, float, Dict[str, str] | None]] = []

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] | None = None) -> None:
        self.gauges.append((name, value, labels))

    def inc_counter(
        self, name: str, amount: float = 1.0, labels: Dict[str, str] | None = None
    ) -> None:
        return

    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: list[float] | None = None,
        labels: Dict[str, str] | None = None,
    ) -> None:
        return


@pytest.fixture
def metrics_sink() -> _RecordingSink:
    """Install a recording metrics sink for the duration of the test."""

    previous_has_sink = telemetry.has_metrics_sink()
    previous_sink = telemetry.get_metrics_sink()
    sink = _RecordingSink()
    telemetry.set_metrics_sink(sink)
    try:
        yield sink
    finally:
        if previous_has_sink:
            telemetry.set_metrics_sink(previous_sink)  # type: ignore[arg-type]
        else:
            setattr(telemetry, "_SINK", None)


class _Status(Enum):
    NEW = "new"
    FILLED = "filled"


class _StubOrder:
    def __init__(
        self,
        order_id: str,
        status: _Status,
        leaves_qty: float = 0.0,
        cum_qty: float = 0.0,
        avg_px: float = 0.0,
    ):
        self.order_id = order_id
        self.status = status
        self.leaves_qty = leaves_qty
        self.cum_qty = cum_qty
        self.avg_px = avg_px


class _StubFixManager:
    def __init__(self, orders: dict[str, _StubOrder]):
        self._orders = orders

    def get_all_orders(self) -> dict[str, _StubOrder]:
        return self._orders


def test_check_orders_counts_mismatches_and_emits_metric(metrics_sink: _RecordingSink) -> None:
    orders = {
        "alpha": _StubOrder(order_id="A1", status=_Status.NEW),
        "beta": _StubOrder(order_id="B1", status=_Status.FILLED),
        "gamma": _StubOrder(order_id="C1", status=_Status.NEW),
    }
    checker = ParityChecker(_StubFixManager(orders))

    broker_orders: dict[str, Dict[str, Any]] = {
        "alpha": {"order_id": "A1", "status": "new"},
        "beta": {"order_id": "WRONG", "status": "filled"},
    }

    mismatches = checker.check_orders(broker_orders)

    assert mismatches == 2
    assert ("fix_parity_mismatched_orders", 2.0, None) in metrics_sink.gauges


def test_compare_order_fields_reports_differences() -> None:
    checker = ParityChecker(_StubFixManager({}))
    local = _StubOrder(order_id="123", status=_Status.NEW, leaves_qty=10, cum_qty=4, avg_px=1.2)
    broker = {
        "order_id": "999",
        "status": "cancelled",
        "leaves_qty": "10",
        "cum_qty": "5",
        "avg_px": 2,
    }

    diffs = checker.compare_order_fields(local, broker)

    assert diffs == {
        "status": {"local": "new", "broker": "cancelled"},
        "cum_qty": {"local": 4, "broker": "5"},
        "avg_px": {"local": 1.2, "broker": 2},
        "order_id": {"local": "123", "broker": "999"},
    }


def test_check_positions_detects_missing_and_mismatched(
    monkeypatch: pytest.MonkeyPatch, metrics_sink: _RecordingSink
) -> None:
    from src.trading.monitoring import parity_checker as parity_module
    from src.trading.monitoring import portfolio_tracker as tracker_module

    class _StubPortfolioTracker:
        def __init__(self) -> None:
            self.positions = {
                "EURUSD": SimpleNamespace(quantity=1.0),
                "GBPUSD": SimpleNamespace(quantity=0.5),
                "AUDUSD": SimpleNamespace(quantity=0.25),
            }

    monkeypatch.setattr(tracker_module, "PortfolioTracker", _StubPortfolioTracker)

    checker = ParityChecker(_StubFixManager({}))
    broker_positions = {
        "EURUSD": {"quantity": 1.0},
        "GBPUSD": {"quantity": 0.1},
    }

    mismatches = checker.check_positions(broker_positions)

    assert mismatches == 2
    assert ("fix_parity_mismatched_positions", 2.0, None) in metrics_sink.gauges
