from datetime import UTC, datetime, timedelta

import pytest

from src.operations import (
    LoopKpiMetrics,
    aggregate_fast_weight_metadata,
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


def test_tracker_collects_fast_weight_metrics_and_hebbian_samples() -> None:
    tracker = StrategyPerformanceTracker(initial_capital=120_000.0)

    tracker.record_trade(
        "alpha",
        pnl=400.0,
        notional=12_000.0,
        return_pct=0.033,
        timestamp=_timestamp(0),
        fast_weights_enabled=True,
        fast_weight_metrics={
            "total": 5,
            "active": 2,
            "dormant": 3,
            "inhibitory": 1,
            "suppressed_inhibitory": 0,
            "active_percentage": 40.0,
            "sparsity": 0.6,
            "max_multiplier": 1.6,
            "min_multiplier": 0.85,
        },
        fast_weight_summary={
            "momentum_boost": {
                "current_multiplier": 1.6,
                "previous_multiplier": 1.4,
                "feature_value": 0.9,
            }
        },
    )

    tracker.record_trade(
        "alpha",
        pnl=250.0,
        notional=9_000.0,
        return_pct=0.027,
        timestamp=_timestamp(1),
        fast_weights_enabled=False,
        fast_weight_metrics={
            "total": 4,
            "active": 1,
            "dormant": 3,
            "inhibitory": 0,
            "suppressed_inhibitory": 1,
            "active_percentage": 25.0,
            "sparsity": 0.75,
            "max_multiplier": 1.25,
            "min_multiplier": 0.7,
        },
        fast_weight_summary={
            "momentum_boost": {
                "current_multiplier": 1.25,
                "previous_multiplier": 1.5,
                "feature_value": 0.8,
            },
            "mean_reversion_guard": {
                "current_multiplier": 0.95,
                "previous_multiplier": 1.05,
            },
        },
    )

    report = tracker.generate_report(as_of=_timestamp(2))

    alpha = next(strategy for strategy in report.strategies if strategy.strategy_id == "alpha")
    fast_weight_meta = alpha.metadata.get("fast_weight")
    assert fast_weight_meta is not None

    toggle_counts = fast_weight_meta.get("toggle_counts", {})
    assert toggle_counts["enabled"] == 1
    assert toggle_counts["disabled"] == 1

    metrics_summary = fast_weight_meta.get("metrics", {})
    assert metrics_summary["samples"] == 2
    assert metrics_summary["active_mean"] == pytest.approx((2 + 1) / 2)
    assert metrics_summary["sparsity_mean"] == pytest.approx((0.6 + 0.75) / 2)
    assert metrics_summary["max_multiplier"] == pytest.approx(1.6)
    assert metrics_summary["min_multiplier"] == pytest.approx(0.7)

    hebbian_adapters = fast_weight_meta.get("hebbian_adapters", {})
    assert "momentum_boost" in hebbian_adapters
    momentum_stats = hebbian_adapters["momentum_boost"]
    assert momentum_stats["samples"] == 2
    assert momentum_stats["current_multiplier_mean"] == pytest.approx((1.6 + 1.25) / 2)
    assert momentum_stats["delta_mean"] == pytest.approx(((1.6 - 1.4) + (1.25 - 1.5)) / 2)
    assert momentum_stats["feature_value_mean"] == pytest.approx((0.9 + 0.8) / 2)

    assert "mean_reversion_guard" in hebbian_adapters
    guard_stats = hebbian_adapters["mean_reversion_guard"]
    assert guard_stats["samples"] == 1
    assert guard_stats["current_multiplier_mean"] == pytest.approx(0.95)

    aggregates_meta = report.aggregates.metadata.get("fast_weight")
    assert aggregates_meta is not None
    assert aggregates_meta["toggle_counts"]["enabled"] == 1
    assert aggregates_meta["metrics"]["samples"] == 2
    assert aggregates_meta["metrics"]["active_percentage_mean"] == pytest.approx((40.0 + 25.0) / 2)
    assert "momentum_boost" in aggregates_meta["hebbian_adapters"]


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


def test_aggregate_fast_weight_metadata_rolls_up_events() -> None:
    events = [
        {
            "metadata": {
                "fast_weight": {
                    "enabled": True,
                    "metrics": {
                        "total": 5,
                        "active": 2,
                        "dormant": 3,
                        "inhibitory": 1,
                        "suppressed_inhibitory": 0,
                        "active_percentage": 40.0,
                        "sparsity": 0.6,
                        "max_multiplier": 1.5,
                        "min_multiplier": 0.9,
                    },
                    "summary": {
                        "momentum": {
                            "current_multiplier": 1.5,
                            "previous_multiplier": 1.3,
                            "feature_value": 0.85,
                        }
                    },
                }
            }
        },
        {
            "metadata": {
                "fast_weight": {
                    "enabled": False,
                    "metrics": {
                        "total": 5,
                        "active": 1,
                        "dormant": 4,
                        "inhibitory": 0,
                        "suppressed_inhibitory": 1,
                        "active_percentage": 20.0,
                        "sparsity": 0.8,
                        "max_multiplier": 1.3,
                        "min_multiplier": 0.7,
                    },
                    "summary": {
                        "momentum": {
                            "current_multiplier": 1.3,
                            "previous_multiplier": 1.6,
                            "feature_value": 0.8,
                        },
                        "mean_reversion": {
                            "current_multiplier": 0.95,
                            "previous_multiplier": 1.05,
                        },
                    },
                }
            }
        },
    ]

    rollup = aggregate_fast_weight_metadata(events)

    toggle_counts = rollup.get("toggle_counts", {})
    assert toggle_counts.get("enabled") == 1
    assert toggle_counts.get("disabled") == 1

    metrics = rollup.get("metrics", {})
    assert metrics.get("active_mean") == pytest.approx(1.5)
    assert metrics.get("sparsity_mean") == pytest.approx(0.7)
    assert metrics.get("max_multiplier") == pytest.approx(1.5)
    assert metrics.get("min_multiplier") == pytest.approx(0.7)

    adapters = rollup.get("hebbian_adapters", {})
    momentum = adapters.get("momentum")
    assert momentum is not None
    assert momentum.get("samples") == 2
    assert momentum.get("current_multiplier_mean") == pytest.approx((1.5 + 1.3) / 2)
    assert momentum.get("feature_value_mean") == pytest.approx((0.85 + 0.8) / 2)

    mean_reversion = adapters.get("mean_reversion")
    assert mean_reversion is not None
    assert mean_reversion.get("samples") == 1
