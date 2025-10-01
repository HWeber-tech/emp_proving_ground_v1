from datetime import datetime, timezone

import pytest

from src.operations.roi import RoiStatus, RoiTelemetrySnapshot
from src.operations.event_bus_failover import EventPublishError
from src.operations.strategy_performance import (
    StrategyPerformanceStatus,
    evaluate_strategy_performance,
    format_strategy_performance_markdown,
    publish_strategy_performance_snapshot,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:
        self.events.append(event)
        return event

    def is_running(self) -> bool:
        return True


class FallbackStubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:
        raise RuntimeError("runtime bus unavailable")

    def is_running(self) -> bool:
        return True


class StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, str]] = []

    def publish_sync(self, event_type: str, payload: object, *, source: str) -> None:
        self.events.append((event_type, payload, source))


def _roi_snapshot() -> RoiTelemetrySnapshot:
    return RoiTelemetrySnapshot(
        status=RoiStatus.tracking,
        generated_at=datetime.now(timezone.utc),
        initial_capital=100_000.0,
        current_equity=105_000.0,
        gross_pnl=6_000.0,
        net_pnl=5_000.0,
        infrastructure_cost=500.0,
        fees=500.0,
        days_active=10.0,
        executed_trades=12,
        total_notional=250_000.0,
        roi=0.05,
        annualised_roi=0.18,
        gross_roi=0.06,
        gross_annualised_roi=0.22,
        breakeven_daily_return=0.001,
        target_annual_roi=0.2,
    )


def test_evaluate_strategy_performance_groups_events() -> None:
    now = datetime.now(timezone.utc)
    events = [
        {
            "strategy_id": "alpha",
            "status": "executed",
            "confidence": 0.8,
            "notional": 10_000.0,
            "timestamp": now.isoformat(),
        },
        {
            "strategy_id": "alpha",
            "status": "rejected",
            "metadata": {"reason": "risk_limit"},
            "timestamp": now.isoformat(),
        },
        {
            "strategy_id": "alpha",
            "status": "failed",
            "metadata": {"error": "engine_timeout"},
            "timestamp": (now.replace(microsecond=0)).isoformat(),
        },
        {
            "strategy_id": "beta",
            "status": "executed",
            "confidence": 0.55,
            "notional": 5_000.0,
            "timestamp": now.isoformat(),
        },
    ]

    snapshot = evaluate_strategy_performance(
        events,
        roi_snapshot=_roi_snapshot(),
        metadata={"window": "paper"},
    )

    assert snapshot.status is StrategyPerformanceStatus.warn
    assert snapshot.totals.total_events == 4
    assert snapshot.totals.roi_status == RoiStatus.tracking.value
    assert snapshot.totals.net_pnl == pytest.approx(5_000.0)

    assert snapshot.top_rejection_reasons["risk_limit"] == 1

    alpha_metrics = next(metric for metric in snapshot.strategies if metric.strategy_id == "alpha")
    assert alpha_metrics.rejected == 1
    assert alpha_metrics.failed == 1
    assert alpha_metrics.status is StrategyPerformanceStatus.warn

    markdown = format_strategy_performance_markdown(snapshot)
    assert "Strategy alpha" in markdown
    assert "ROI status" in markdown


def test_publish_strategy_performance_snapshot_emits_event() -> None:
    snapshot = evaluate_strategy_performance(
        [
            {
                "strategy_id": "alpha",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
    )

    bus = StubBus()
    publish_strategy_performance_snapshot(bus, snapshot, source="test")

    assert bus.events, "expected strategy performance telemetry event"
    event = bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.strategy.performance"
    assert getattr(event, "source", "") == "test"
    payload = getattr(event, "payload", {})
    assert payload.get("status") in {status.value for status in StrategyPerformanceStatus}


def test_publish_strategy_performance_snapshot_falls_back_to_global_bus() -> None:
    snapshot = evaluate_strategy_performance(
        [
            {
                "strategy_id": "alpha",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
    )

    bus = FallbackStubBus()
    topic_bus = StubTopicBus()

    publish_strategy_performance_snapshot(
        bus,
        snapshot,
        source="test",
        global_bus_factory=lambda: topic_bus,
    )

    assert topic_bus.events, "expected strategy performance telemetry event on global bus"
    event_type, payload, source = topic_bus.events[-1]
    assert event_type == "telemetry.strategy.performance"
    assert source == "test"
    assert payload.get("status") in {status.value for status in StrategyPerformanceStatus}


def test_publish_strategy_performance_snapshot_raises_on_unexpected_error() -> None:
    snapshot = evaluate_strategy_performance(
        [
            {
                "strategy_id": "alpha",
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
    )

    class BrokenBus(StubBus):
        def publish_from_sync(self, event: object) -> None:  # type: ignore[override]
            raise ValueError("unexpected failure")

    bus = BrokenBus()

    with pytest.raises(EventPublishError) as excinfo:
        publish_strategy_performance_snapshot(bus, snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.strategy.performance"
