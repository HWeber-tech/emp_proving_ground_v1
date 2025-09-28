from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pytest

from src.trading.order_management.monitoring.pnl_metrics import PositionMetricsPublisher
from src.trading.order_management.position_tracker import PositionTracker


class _LabeledGauge:
    def __init__(self, store: Dict[Tuple[Tuple[str, str], ...], float], labels: Dict[str, str]) -> None:
        self._store = store
        self._labels = tuple(sorted(labels.items()))

    def set(self, value: float) -> None:
        self._store[self._labels] = float(value)


class _Gauge:
    def __init__(self) -> None:
        self.values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def labels(self, **labels: str) -> _LabeledGauge:
        return _LabeledGauge(self.values, labels)


@dataclass
class _Registry:
    gauges: Dict[str, _Gauge]

    def get_gauge(self, name: str, description: str, labelnames=None):  # type: ignore[override]
        gauge = self.gauges.setdefault(name, _Gauge())
        return gauge


def _metric_value(registry: _Registry, metric: str, **labels: str) -> float:
    gauge = registry.gauges[metric]
    key = tuple(sorted(labels.items()))
    return gauge.values[key]


def test_position_tracker_publishes_metrics() -> None:
    registry = _Registry(gauges={})
    publisher = PositionMetricsPublisher(registry=registry)
    tracker = PositionTracker(metrics_publisher=publisher)

    tracker.record_fill("EURUSD", 1.0, 1.1, account="ACC1")
    tracker.record_fill("EURUSD", -0.4, 1.2, account="ACC1")

    assert pytest.approx(
        _metric_value(registry, "emp_position_net_quantity", account="ACC1", symbol="EURUSD")
    ) == 0.6
    assert pytest.approx(
        _metric_value(registry, "emp_position_gross_long", account="ACC1", symbol="EURUSD")
    ) == 0.6
    assert pytest.approx(
        _metric_value(registry, "emp_position_gross_short", account="ACC1", symbol="EURUSD")
    ) == 0.0

    realized = _metric_value(registry, "emp_position_realized_pnl", account="ACC1", symbol="EURUSD")
    assert pytest.approx(realized, rel=1e-6) == pytest.approx(0.04, rel=1e-6)

    tracker.update_mark_price("EURUSD", 1.25)

    exposure = _metric_value(registry, "emp_position_notional_exposure", account="ACC1", symbol="EURUSD")
    assert pytest.approx(exposure, rel=1e-6) == pytest.approx(0.75, rel=1e-6)

    unrealized = _metric_value(registry, "emp_position_unrealized_pnl", account="ACC1", symbol="EURUSD")
    assert pytest.approx(unrealized, rel=1e-6) == pytest.approx(0.09, rel=1e-6)

    total_exposure = _metric_value(registry, "emp_account_total_exposure", account="ACC1")
    assert pytest.approx(total_exposure, rel=1e-6) == pytest.approx(0.75, rel=1e-6)

    total_realized = _metric_value(registry, "emp_account_total_realized_pnl", account="ACC1")
    assert pytest.approx(total_realized, rel=1e-6) == pytest.approx(0.04, rel=1e-6)

    total_unrealized = _metric_value(registry, "emp_account_total_unrealized_pnl", account="ACC1")
    assert pytest.approx(total_unrealized, rel=1e-6) == pytest.approx(0.09, rel=1e-6)

