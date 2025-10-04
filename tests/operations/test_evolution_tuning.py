import logging
from datetime import UTC, datetime

import pytest

from src.operations.evolution_experiments import (
    ExperimentMetrics,
    ExperimentStatus,
    EvolutionExperimentSnapshot,
)
from src.operations.evolution_tuning import (
    EvolutionTuningStatus,
    EvolutionTuningSnapshot,
    evaluate_evolution_tuning,
    format_evolution_tuning_markdown,
    publish_evolution_tuning_snapshot,
)
from src.operations.strategy_performance import (
    StrategyPerformanceMetrics,
    StrategyPerformanceSnapshot,
    StrategyPerformanceStatus,
    StrategyPerformanceTotals,
)
from src.operations.event_bus_failover import EventPublishError


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return True


def _make_experiment(
    *,
    status: ExperimentStatus,
    execution_rate: float,
    roi: float,
    roi_status: str,
) -> EvolutionExperimentSnapshot:
    metrics = ExperimentMetrics(
        total_events=10,
        executed=int(round(execution_rate * 10)),
        rejected=2,
        failed=1,
        execution_rate=execution_rate,
        rejection_rate=0.2,
        failure_rate=0.1,
        avg_confidence=0.6,
        avg_notional=25_000.0,
        roi_status=roi_status,
        roi=roi,
        net_pnl=roi * 50_000,
        metadata={"window": 10},
    )
    return EvolutionExperimentSnapshot(
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        status=status,
        metrics=metrics,
        rejection_reasons={"risk": 1},
        metadata={"ingest_success": True},
    )


def _make_performance(
    metric: StrategyPerformanceMetrics,
    *,
    status: StrategyPerformanceStatus,
    roi: float,
    roi_status: str,
) -> StrategyPerformanceSnapshot:
    totals = StrategyPerformanceTotals(
        total_events=metric.total_events,
        executed=metric.executed,
        rejected=metric.rejected,
        failed=metric.failed,
        other=metric.other,
        execution_rate=metric.execution_rate,
        rejection_rate=metric.rejection_rate,
        failure_rate=metric.failure_rate,
        roi_status=roi_status,
        roi=roi,
        net_pnl=roi * 50_000,
    )
    return StrategyPerformanceSnapshot(
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        status=status,
        strategies=(metric,),
        totals=totals,
        lookback=metric.total_events,
        top_rejection_reasons={"risk": metric.rejected},
        metadata={"ingest_success": True},
    )


def test_evaluate_evolution_tuning_flags_failure_and_formats_markdown() -> None:
    metric = StrategyPerformanceMetrics(
        strategy_id="trend_follow",
        status=StrategyPerformanceStatus.alert,
        total_events=10,
        executed=2,
        rejected=3,
        failed=5,
        other=0,
        execution_rate=0.2,
        rejection_rate=0.3,
        failure_rate=0.5,
        avg_confidence=0.35,
        avg_notional=15_000.0,
        last_event_at=datetime(2024, 1, 4, tzinfo=UTC),
        rejection_reasons={"risk": 3},
        metadata={"policy": "aggressive"},
    )
    performance = _make_performance(
        metric,
        status=StrategyPerformanceStatus.alert,
        roi=-0.12,
        roi_status="at_risk",
    )
    experiment = _make_experiment(
        status=ExperimentStatus.alert,
        execution_rate=0.2,
        roi=-0.12,
        roi_status="at_risk",
    )

    snapshot = evaluate_evolution_tuning(
        experiment,
        performance,
        metadata={"ingest_success": True},
    )

    assert snapshot.status is EvolutionTuningStatus.alert
    assert snapshot.summary.total_recommendations == 1
    assert snapshot.summary.action_counts.get("disable_strategy") == 1
    recommendation = snapshot.recommendations[0]
    assert recommendation.strategy_id == "trend_follow"
    assert recommendation.action == "disable_strategy"
    assert "Failure rate" in recommendation.rationale

    markdown = format_evolution_tuning_markdown(snapshot)
    assert "Recommendations" in markdown
    assert "disable_strategy" in markdown
    assert "ROI status" in markdown


def test_evaluate_evolution_tuning_scales_successful_strategy() -> None:
    metric = StrategyPerformanceMetrics(
        strategy_id="carry",
        status=StrategyPerformanceStatus.normal,
        total_events=12,
        executed=10,
        rejected=1,
        failed=1,
        other=0,
        execution_rate=0.83,
        rejection_rate=0.08,
        failure_rate=0.08,
        avg_confidence=0.72,
        avg_notional=40_000.0,
        last_event_at=datetime(2024, 1, 5, tzinfo=UTC),
        rejection_reasons={},
        metadata={"notional_limit": 50_000},
    )
    performance = _make_performance(
        metric,
        status=StrategyPerformanceStatus.normal,
        roi=0.18,
        roi_status="ahead",
    )
    experiment = _make_experiment(
        status=ExperimentStatus.normal,
        execution_rate=0.85,
        roi=0.18,
        roi_status="ahead",
    )

    snapshot = evaluate_evolution_tuning(experiment, performance)

    assert snapshot.status is EvolutionTuningStatus.normal
    assert snapshot.summary.total_recommendations == 1
    rec = snapshot.recommendations[0]
    assert rec.action == "scale_successful_strategy"
    assert "consider scaling" in rec.rationale
    assert rec.confidence > 0.5


def _make_snapshot() -> EvolutionTuningSnapshot:
    metric = StrategyPerformanceMetrics(
        strategy_id="momentum",
        status=StrategyPerformanceStatus.normal,
        total_events=5,
        executed=4,
        rejected=1,
        failed=0,
        other=0,
        execution_rate=0.8,
        rejection_rate=0.2,
        failure_rate=0.0,
        avg_confidence=0.6,
        avg_notional=10_000.0,
        last_event_at=datetime(2024, 1, 7, tzinfo=UTC),
        rejection_reasons={"risk": 1},
        metadata={},
    )
    performance = _make_performance(
        metric,
        status=StrategyPerformanceStatus.normal,
        roi=0.05,
        roi_status="steady",
    )
    experiment = _make_experiment(
        status=ExperimentStatus.normal,
        execution_rate=0.9,
        roi=0.05,
        roi_status="steady",
    )
    return evaluate_evolution_tuning(experiment, performance)


def test_publish_evolution_tuning_snapshot_prefers_runtime_bus() -> None:
    bus = _StubEventBus()
    snapshot = _make_snapshot()

    publish_evolution_tuning_snapshot(bus, snapshot)

    assert bus.events, "expected evolution tuning event"
    event = bus.events[0]
    assert event.type == "telemetry.evolution.tuning"
    assert event.payload["status"] == snapshot.status.value


def test_publish_evolution_tuning_snapshot_falls_back_to_global_bus(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    snapshot = _make_snapshot()

    class _FailingRuntimeBus(_StubEventBus):
        def publish_from_sync(self, event) -> int:  # type: ignore[override]
            raise RuntimeError("offline")

    class _TopicBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, object, str | None]] = []

        def publish_sync(self, topic: str, payload: object, *, source: str | None = None) -> int:
            self.published.append((topic, payload, source))
            return 1

    captured = _TopicBus()

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: captured,
    )

    with caplog.at_level(logging.WARNING):
        publish_evolution_tuning_snapshot(_FailingRuntimeBus(), snapshot)

    assert captured.published
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "falling back to global bus" in messages


def test_publish_evolution_tuning_snapshot_raises_on_unexpected_runtime_error() -> None:
    snapshot = _make_snapshot()

    class _UnexpectedRuntimeBus(_StubEventBus):
        def publish_from_sync(self, event) -> int:  # type: ignore[override]
            raise ValueError("boom")

    with pytest.raises(EventPublishError) as excinfo:
        publish_evolution_tuning_snapshot(_UnexpectedRuntimeBus(), snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.evolution.tuning"
