from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.trading.monitoring.performance_metrics import (
    PerformanceMetrics,
    calculate_annualized_return,
    calculate_correlation_matrix,
    calculate_regime_performance,
    calculate_strategy_performance,
    calculate_trading_metrics,
    calculate_var_cvar,
    create_empty_metrics,
)
from src.trading.monitoring.performance_tracker import PerformanceTracker


def test_create_empty_metrics_uses_timestamp() -> None:
    timestamp = datetime(2024, 1, 2, 15, 30)

    metrics = create_empty_metrics(timestamp)

    assert metrics.start_date == timestamp
    assert metrics.end_date == timestamp
    assert metrics.total_trades == 0
    assert metrics.correlation_matrix.empty


def test_calculate_annualized_return_matches_expected_growth() -> None:
    df = pd.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 4)],
            "equity": [100_000.0, 110_000.0],
        }
    )

    annualized = calculate_annualized_return(df, 100_000.0)
    expected = (1 + 0.10) ** (365 / 3) - 1

    assert annualized == pytest.approx(expected)


def test_calculate_trading_metrics_handles_duration_and_profit_factor() -> None:
    trades = [
        {
            "pnl": 120.0,
            "entry_time": datetime(2024, 1, 1, 9),
            "exit_time": datetime(2024, 1, 1, 10),
        },
        {
            "pnl": -60.0,
            "entry_time": datetime(2024, 1, 2, 9),
            "exit_time": datetime(2024, 1, 2, 11),
        },
    ]

    summary = calculate_trading_metrics(trades)

    assert summary["total_trades"] == 2
    assert summary["winning_trades"] == 1
    assert summary["losing_trades"] == 1
    assert summary["profit_factor"] == pytest.approx(2.0)
    assert summary["avg_trade_duration"] == pytest.approx(1.5)  # hours


def test_calculate_strategy_and_regime_performance_filters_empty_entries() -> None:
    strategy = {
        "alpha": {"trades": 2, "wins": 1, "total_return": 0.06, "total_pnl": 300.0},
        "beta": {"trades": 0, "wins": 0, "total_return": 0.0},
    }
    regime = {
        "bull": {"trades": 1, "avg_return": 0.03, "total_return": 0.03},
        "flat": {"trades": 0, "avg_return": 0.0, "total_return": 0.0},
    }

    strategy_summary = calculate_strategy_performance(strategy)
    regime_summary = calculate_regime_performance(regime)

    assert list(strategy_summary) == ["alpha"]
    assert strategy_summary["alpha"]["trade_count"] == 2
    assert list(regime_summary) == ["bull"]
    assert regime_summary["bull"]["trade_count"] == 1


def test_calculate_correlation_matrix_requires_multiple_series() -> None:
    strategies = {
        "alpha": {"trades": 2, "total_return": 0.08},
        "beta": {"trades": 3, "total_return": 0.12},
    }

    matrix = calculate_correlation_matrix(strategies)

    assert set(matrix.columns) == {"alpha", "beta"}
    assert set(matrix.index) == {"alpha", "beta"}


def test_calculate_var_cvar_returns_positive_risk_numbers() -> None:
    returns = [-0.03, 0.01, -0.015, 0.02, -0.025]

    var, cvar = calculate_var_cvar(returns)

    assert var > 0
    assert cvar > 0


def test_performance_tracker_integration_uses_shared_helpers() -> None:
    tracker = PerformanceTracker(initial_balance=100_000.0)
    start = datetime(2024, 1, 1)

    tracker.update_daily_equity(100_000.0, date=start)
    tracker.update_daily_equity(102_000.0, date=start + timedelta(days=1))
    tracker.update_daily_equity(103_000.0, date=start + timedelta(days=2))

    tracker.record_trade(
        {
            "entry_price": 100.0,
            "exit_price": 110.0,
            "size": 1.0,
            "strategy": "alpha",
        }
    )
    tracker.record_trade(
        {
            "entry_price": 200.0,
            "exit_price": 195.0,
            "size": 1.0,
            "strategy": "alpha",
        }
    )
    tracker.update_regime_performance("bull", 0.02)

    metrics = tracker.calculate_metrics(force_recalculate=True)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == 2
    assert metrics.strategy_performance["alpha"]["trade_count"] == 2
    assert metrics.regime_performance["bull"]["trade_count"] == 1

    cached = tracker.calculate_metrics()
    assert cached is metrics


def _seed_equity_curve(tracker: PerformanceTracker, start: datetime) -> None:
    tracker.update_daily_equity(100_000.0, date=start)
    tracker.update_daily_equity(100_060.0, date=start + timedelta(days=1))
    tracker.update_daily_equity(100_130.0, date=start + timedelta(days=2))


def test_baseline_alert_requires_sustained_series() -> None:
    tracker = PerformanceTracker(initial_balance=100_000.0)
    start = datetime(2024, 1, 1)
    _seed_equity_curve(tracker, start)

    for idx in range(3):
        tracker.record_trade(
            {
                "entry_price": 100.0,
                "exit_price": 100.01,
                "size": 1.0,
                "strategy": "alpha",
                "spread": 0.02,
                "timestamp": start + timedelta(minutes=idx),
            }
        )

    alerts = tracker.get_performance_alerts()
    baseline_alerts = [alert for alert in alerts if alert["type"] == "baseline"]

    assert baseline_alerts == []


def test_baseline_alert_triggers_on_sustained_underperformance() -> None:
    tracker = PerformanceTracker(initial_balance=100_000.0)
    start = datetime(2024, 1, 1)
    _seed_equity_curve(tracker, start)

    for idx in range(5):
        tracker.record_trade(
            {
                "entry_price": 100.0,
                "exit_price": 100.005,
                "size": 1.0,
                "strategy": "alpha",
                "spread": 0.02,
                "timestamp": start + timedelta(minutes=idx),
            }
        )

    alerts = tracker.get_performance_alerts()
    baseline_alerts = [alert for alert in alerts if alert["type"] == "baseline"]

    assert len(baseline_alerts) == 1
    baseline_alert = baseline_alerts[0]
    assert baseline_alert["metric"] == "baseline_performance_ratio"
    assert baseline_alert["value"] < 1.0
    assert baseline_alert["details"]["streak"] >= 5
