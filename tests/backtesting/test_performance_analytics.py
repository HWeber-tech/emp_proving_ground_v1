"""Tests for the performance analytics helpers."""

from __future__ import annotations

import math

import pytest

from src.backtesting import (
    compute_drawdown_events,
    compute_performance_metrics,
    compute_trade_attribution,
)


def test_compute_performance_metrics_end_to_end() -> None:
    equity_curve = [100, 105, 102, 110, 108, 120]
    trades = [
        {"pnl": 10.0, "duration": 1.5},
        {"pnl": -5.0, "duration": 2.0},
        {"pnl": 7.0},
        {"pnl": 0.0},
    ]

    metrics = compute_performance_metrics(
        equity_curve,
        trades=trades,
        periods_per_year=12,
    )

    assert metrics.total_return == pytest.approx(0.2, rel=1e-9)
    assert metrics.annualised_return == pytest.approx(0.5489414099829015, rel=1e-12)
    assert metrics.annualised_volatility == pytest.approx(0.18789694802321827, rel=1e-12)
    assert metrics.sharpe_ratio == pytest.approx(2.462489004980255, rel=1e-12)
    assert metrics.sortino_ratio == pytest.approx(8.8191082626893, rel=1e-12)
    assert metrics.calmar_ratio == pytest.approx(19.212949349401548, rel=1e-12)
    assert metrics.max_drawdown == pytest.approx(0.02857142857142858, rel=1e-12)
    assert metrics.max_drawdown_duration == 1
    assert metrics.average_drawdown == pytest.approx(0.023376623376623384, rel=1e-12)

    trade_stats = metrics.trade_attribution
    assert trade_stats.total_trades == 4
    assert trade_stats.winning_trades == 2
    assert trade_stats.losing_trades == 1
    assert trade_stats.breakeven_trades == 1
    assert trade_stats.win_rate == pytest.approx(0.5, rel=1e-12)
    assert trade_stats.average_win == pytest.approx(8.5, rel=1e-12)
    assert trade_stats.average_loss == pytest.approx(-5.0, rel=1e-12)
    assert trade_stats.expectancy == pytest.approx(3.0, rel=1e-12)
    assert trade_stats.profit_factor == pytest.approx(3.4, rel=1e-12)
    assert trade_stats.average_duration == pytest.approx(1.75, rel=1e-12)


def test_compute_drawdown_events_handles_unrecovered() -> None:
    events = compute_drawdown_events([100, 95, 94])

    assert len(events) == 1
    event = events[0]
    assert event.start_index == 0
    assert event.trough_index == 2
    assert event.recovery_index is None
    assert event.depth == pytest.approx(0.06, rel=1e-12)
    assert event.duration == 2


def test_compute_trade_attribution_empty_and_validation() -> None:
    empty_stats = compute_trade_attribution([])
    assert empty_stats.total_trades == 0
    assert math.isclose(empty_stats.win_rate, 0.0)
    assert empty_stats.average_duration is None

    with pytest.raises(KeyError):
        compute_trade_attribution([{"not_pnl": 1.0}])
