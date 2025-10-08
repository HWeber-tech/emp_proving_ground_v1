from datetime import UTC, datetime, timedelta

import pytest

from src.operations import (
    LoopKpiMetrics,
    StrategyPerformanceReport,
    StrategyPerformanceTracker,
)


def _timestamp(minutes: int) -> datetime:
    return datetime.now(tz=UTC) + timedelta(minutes=minutes)


def test_tracker_computes_per_strategy_kpis() -> None:
    tracker = StrategyPerformanceTracker(initial_capital=100_000.0)

    tracker.record_trade(
        "alpha",
        pnl=500.0,
        notional=10_000.0,
        return_pct=0.05,
        timestamp=_timestamp(0),
        fast_weights_enabled=True,
        regime="balanced",
    )
    tracker.record_trade(
        "alpha",
        pnl=-200.0,
        notional=8_000.0,
        return_pct=-0.025,
        timestamp=_timestamp(1),
        fast_weights_enabled=False,
        regime="balanced",
    )
    tracker.record_trade(
        "beta",
        pnl=300.0,
        notional=5_000.0,
        return_pct=0.06,
        timestamp=_timestamp(2),
        fast_weights_enabled=False,
        regime="storm",
    )

    report = tracker.generate_report(as_of=_timestamp(5))

    assert isinstance(report, StrategyPerformanceReport)
    alpha = next(strategy for strategy in report.strategies if strategy.strategy_id == "alpha")
    assert alpha.trades == 2
    assert alpha.wins == 1
    assert alpha.losses == 1
    assert alpha.total_pnl == pytest.approx(300.0)
    assert alpha.fast_weight_breakdown is not None
    assert alpha.fast_weight_breakdown.enabled is not None
    assert alpha.fast_weight_breakdown.disabled is not None
    assert alpha.fast_weight_breakdown.roi_uplift is not None

    totals = report.aggregates
    assert totals.trades == 3
    assert totals.wins == 2
    assert totals.losses == 1
    assert totals.total_pnl == pytest.approx(600.0)
    assert totals.roi == pytest.approx(600.0 / (10_000.0 + 8_000.0 + 5_000.0))


def test_tracker_reports_loop_metrics() -> None:
    tracker = StrategyPerformanceTracker(initial_capital=50_000.0)

    tracker.record_trade(
        "gamma",
        pnl=100.0,
        notional=2_000.0,
        return_pct=0.05,
        timestamp=_timestamp(0),
        fast_weights_enabled=True,
    )

    tracker.record_regime_evaluation(predicted="calm", actual="calm")
    tracker.record_regime_evaluation(predicted="storm", actual="calm")

    tracker.record_drift_evaluation(triggered=True, drift_present=False)
    tracker.record_drift_evaluation(triggered=False, drift_present=True)
    tracker.record_drift_evaluation(triggered=False, drift_present=False)

    metrics = tracker.generate_report(as_of=_timestamp(2)).loop_metrics

    assert isinstance(metrics, LoopKpiMetrics)
    assert metrics.total_regime_evaluations == 2
    assert metrics.regime_accuracy == pytest.approx(0.5)
    assert metrics.drift_false_positive_rate == pytest.approx(0.5)
    assert metrics.drift_false_negative_rate == pytest.approx(1.0)
    assert metrics.drift_counts["total"] == 3


def test_tracker_emits_roi_snapshot_even_without_trades() -> None:
    tracker = StrategyPerformanceTracker(initial_capital=25_000.0)
    report = tracker.generate_report()

    assert report.roi_snapshot is not None
    roi_snapshot = report.roi_snapshot
    assert roi_snapshot.net_pnl == pytest.approx(0.0)
    assert roi_snapshot.executed_trades == 0


def test_tracker_markdown_contains_headline_sections() -> None:
    tracker = StrategyPerformanceTracker(initial_capital=75_000.0)
    tracker.record_trade(
        "delta",
        pnl=250.0,
        notional=5_000.0,
        return_pct=0.05,
        timestamp=_timestamp(0),
        fast_weights_enabled=True,
    )

    markdown = tracker.generate_report(as_of=_timestamp(1)).to_markdown()

    assert "**Strategy KPIs**" in markdown
    assert "**Portfolio totals**" in markdown
    assert "**ROI posture**" in markdown
