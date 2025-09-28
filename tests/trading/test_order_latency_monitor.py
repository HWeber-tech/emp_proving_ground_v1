from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.trading.order_management import OrderExecutionEvent, OrderMetadata, OrderStateMachine
from src.trading.order_management.monitoring import OrderLatencyMonitor


class FakeHistogram:
    def __init__(self, name: str) -> None:
        self.name = name
        self.labels_called: list[dict[str, str]] = []
        self.observations: list[float] = []

    def labels(self, **labels: str) -> "FakeHistogram":
        self.labels_called.append(labels)
        return self

    def observe(self, value: float) -> None:
        self.observations.append(float(value))


class FakeRegistry:
    def __init__(self) -> None:
        self.histograms: dict[str, FakeHistogram] = {}

    def get_histogram(self, name: str, description: str, buckets, labelnames) -> FakeHistogram:  # type: ignore[override]
        hist = FakeHistogram(name)
        self.histograms[name] = hist
        return hist


def test_order_latency_monitor_records_all_latencies() -> None:
    registry = FakeRegistry()
    monitor = OrderLatencyMonitor(registry=registry)

    machine = OrderStateMachine()
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    metadata = OrderMetadata(
        "ORD-1",
        "EURUSD",
        "BUY",
        10,
        account="SIM",
        created_at=created_at,
        venue="ICM",
    )
    machine.register_order(metadata)

    def _ts(seconds: int) -> datetime:
        return created_at + timedelta(seconds=seconds)

    events = [
        OrderExecutionEvent("ORD-1", "acknowledged", "0", timestamp=_ts(1)),
        OrderExecutionEvent(
            "ORD-1",
            "partial_fill",
            "1",
            last_quantity=4,
            last_price=1.0,
            cumulative_quantity=4,
            timestamp=_ts(2),
        ),
        OrderExecutionEvent(
            "ORD-1",
            "filled",
            "2",
            last_quantity=6,
            cumulative_quantity=10,
            last_price=1.01,
            timestamp=_ts(3),
        ),
    ]

    for event in events:
        state = machine.apply_event(event)
        snapshot = machine.snapshot("ORD-1")
        monitor.record_transition(state, event, snapshot)

    assert registry.histograms["order_ack_latency_seconds"].observations == [1.0]
    assert registry.histograms["order_first_fill_latency_seconds"].observations == [2.0]
    assert registry.histograms["order_final_fill_latency_seconds"].observations == [3.0]
    assert registry.histograms["order_cancel_latency_seconds"].observations == []
    assert registry.histograms["order_reject_latency_seconds"].observations == []

    # Ensure venue label applied consistently
    assert registry.histograms["order_ack_latency_seconds"].labels_called == [{"venue": "ICM"}]
