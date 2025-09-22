from datetime import UTC, datetime

from src.operations.evolution_experiments import (
    ExperimentMetrics,
    ExperimentStatus,
    EvolutionExperimentSnapshot,
)
from src.operations.evolution_tuning import (
    EvolutionTuningStatus,
    evaluate_evolution_tuning,
    format_evolution_tuning_markdown,
)
from src.operations.strategy_performance import (
    StrategyPerformanceMetrics,
    StrategyPerformanceSnapshot,
    StrategyPerformanceStatus,
    StrategyPerformanceTotals,
)


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
