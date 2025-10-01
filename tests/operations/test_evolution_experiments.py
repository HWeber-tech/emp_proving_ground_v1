from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.operations.evolution_experiments import (
    ExperimentStatus,
    EvolutionExperimentSnapshot,
    evaluate_evolution_experiments,
    format_evolution_experiment_markdown,
    publish_evolution_experiment_snapshot,
)
from src.operations.event_bus_failover import EventPublishError
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot


def _build_roi_snapshot(status: RoiStatus = RoiStatus.tracking) -> RoiTelemetrySnapshot:
    return RoiTelemetrySnapshot(
        status=status,
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        initial_capital=100_000.0,
        current_equity=102_500.0,
        gross_pnl=2_800.0,
        net_pnl=2_500.0,
        infrastructure_cost=100.0,
        fees=200.0,
        days_active=10.0,
        executed_trades=8,
        total_notional=250_000.0,
        roi=0.025,
        annualised_roi=0.09125,
        gross_roi=0.028,
        gross_annualised_roi=0.1022,
        breakeven_daily_return=0.0005,
        target_annual_roi=0.25,
    )


def test_evaluate_evolution_experiments_generates_metrics() -> None:
    events = [
        {
            "event_id": "1",
            "status": "executed",
            "confidence": 0.82,
            "notional": 12_500.0,
        },
        {
            "event_id": "2",
            "status": "rejected",
            "confidence": 0.22,
            "metadata": {"reason": "low_confidence"},
        },
        {
            "event_id": "3",
            "status": "failed",
            "metadata": {"error": "engine"},
        },
    ]

    roi_snapshot = _build_roi_snapshot(RoiStatus.tracking)
    snapshot = evaluate_evolution_experiments(
        events,
        roi_snapshot=roi_snapshot,
        metadata={"ingest_success": True},
    )

    assert snapshot.status is ExperimentStatus.warn
    assert snapshot.metrics.total_events == 3
    assert snapshot.metrics.executed == 1
    assert snapshot.metrics.rejected == 1
    assert snapshot.metrics.failure_rate == pytest.approx(1 / 3)
    assert snapshot.metrics.metadata["ingest_success"] is True
    assert snapshot.rejection_reasons["low_confidence"] == 1

    markdown = format_evolution_experiment_markdown(snapshot)
    assert "Execution rate" in markdown
    assert "low_confidence" in markdown


def test_evaluate_evolution_experiments_alert_on_poor_execution() -> None:
    roi_snapshot = _build_roi_snapshot(RoiStatus.at_risk)
    alert_snapshot = evaluate_evolution_experiments(
        [],
        roi_snapshot=roi_snapshot,
    )

    assert alert_snapshot.status is ExperimentStatus.alert
    assert alert_snapshot.metrics.total_events == 0


def test_publish_evolution_experiment_snapshot_uses_event_bus() -> None:
    class StubBus:
        def __init__(self) -> None:
            self.events: list[object] = []

        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event) -> int:  # pragma: no cover - trivial
            self.events.append(event)
            return 1

    bus = StubBus()
    snapshot = evaluate_evolution_experiments(
        [
            {"event_id": "1", "status": "executed", "confidence": 0.9},
            {"event_id": "2", "status": "rejected", "metadata": {"reason": "risk"}},
        ],
        roi_snapshot=_build_roi_snapshot(),
    )

    publish_evolution_experiment_snapshot(bus, snapshot)

    assert bus.events, "expected snapshot to publish via stub bus"
    published = bus.events[0]
    assert published.payload["status"] == snapshot.status.value


def test_publish_evolution_experiment_snapshot_falls_back_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class PrimaryBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: object) -> None:
            raise RuntimeError("loop stopped")

    published: list[tuple[str, dict[str, object], str | None]] = []

    class GlobalBus:
        def publish_sync(
            self, event_type: str, payload: dict[str, object], source: str | None = None
        ) -> None:
            published.append((event_type, payload, source))

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", lambda: GlobalBus()
    )

    snapshot = evaluate_evolution_experiments(
        [
            {"event_id": "1", "status": "executed", "confidence": 0.9},
            {"event_id": "2", "status": "rejected", "metadata": {"reason": "risk"}},
        ],
        roi_snapshot=_build_roi_snapshot(),
    )

    caplog.set_level("WARNING", logger="src.operations.evolution_experiments")
    publish_evolution_experiment_snapshot(PrimaryBus(), snapshot)

    assert published, "expected snapshot to publish via global bus fallback"
    assert any(
        "falling back to global bus" in message for message in caplog.messages
    ), "expected warning when falling back to global bus"


def test_publish_evolution_experiment_snapshot_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    class PrimaryBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: object) -> None:
            raise ValueError("boom")

    def fail_global_bus() -> object:
        raise AssertionError("global bus should not be used when primary raises")

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", fail_global_bus
    )

    snapshot = evaluate_evolution_experiments(
        [
            {"event_id": "1", "status": "executed", "confidence": 0.9},
            {"event_id": "2", "status": "rejected", "metadata": {"reason": "risk"}},
        ],
        roi_snapshot=_build_roi_snapshot(),
    )

    with pytest.raises(EventPublishError) as excinfo:
        publish_evolution_experiment_snapshot(PrimaryBus(), snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.evolution.experiments"


def test_evaluate_evolution_experiments_accepts_roi_mapping() -> None:
    events = [
        {"event_id": "1", "status": "executed", "confidence": 0.9},
        {"event_id": "2", "status": "rejected", "metadata": {"reason": "risk"}},
    ]
    roi_mapping = {"status": RoiStatus.tracking.value, "roi": "0.1", "net_pnl": "1500"}

    snapshot = evaluate_evolution_experiments(events, roi_snapshot=roi_mapping)

    assert snapshot.metrics.roi == pytest.approx(0.1)
    assert snapshot.metrics.net_pnl == pytest.approx(1500.0)
    assert snapshot.metrics.roi_status == RoiStatus.tracking.value


def test_evaluate_evolution_experiments_filters_invalid_events() -> None:
    events = [
        {"event_id": "1", "status": "executed"},
        "not-a-mapping",  # type: ignore[list-item]
        42,  # type: ignore[list-item]
        {"event_id": "2", "status": "rejected", "metadata": {"reason": "risk"}},
    ]

    snapshot = evaluate_evolution_experiments(events, lookback=10)

    assert snapshot.metrics.total_events == 2
    assert snapshot.rejection_reasons["risk"] == 1
