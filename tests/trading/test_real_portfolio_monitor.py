"""Integration-level tests for the real portfolio monitor."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.trading.models.position import Position
from src.trading.portfolio.config import PortfolioMonitorConfig
from src.trading.portfolio.real_portfolio_monitor import RealPortfolioMonitor


@pytest.fixture()
def portfolio_monitor(tmp_path: Path) -> RealPortfolioMonitor:
    """Return a portfolio monitor backed by a temporary SQLite database."""

    db_dir = tmp_path / "portfolio"
    db_dir.mkdir()
    db_path = db_dir / "portfolio.db"
    config = PortfolioMonitorConfig(database_path=db_path, initial_balance=10_000.0)
    return RealPortfolioMonitor(config)


def test_real_portfolio_monitor_end_to_end(portfolio_monitor: RealPortfolioMonitor) -> None:
    """Exercise add/update/close flows and the derived analytics helpers."""

    monitor = portfolio_monitor
    position = Position(symbol="AAPL", size=2.0, entry_price=100.0, position_id="pos-1")

    assert monitor.add_position(position) is True

    open_positions = monitor.get_positions()
    assert len(open_positions) == 1
    assert open_positions[0].status == "OPEN"
    assert open_positions[0].unrealized_pnl == pytest.approx(0.0)

    assert monitor.update_position_price("pos-1", 110.0) is True

    # Refresh positions to confirm the database write round-trips correctly.
    open_positions = monitor.get_positions()
    assert open_positions[0].current_price == pytest.approx(110.0)
    assert open_positions[0].unrealized_pnl == pytest.approx(20.0)

    snapshot = monitor.get_portfolio_snapshot()
    assert snapshot.unrealized_pnl == pytest.approx(20.0)
    assert snapshot.total_value == pytest.approx(monitor.initial_balance + 20.0)

    exit_time = datetime.now()
    assert monitor.close_position("pos-1", 120.0, exit_time=exit_time) is True

    # Closed positions should no longer appear in the open position list.
    assert monitor.get_positions() == []

    # Store a post-close snapshot so metrics can read the final state.
    monitor.get_portfolio_snapshot()

    history = monitor.get_position_history(days=7)
    assert len(history) == 1
    assert history[0].status == "CLOSED"

    daily_pnl = monitor.get_daily_pnl(days=7)
    assert len(daily_pnl) == 1
    assert daily_pnl[0]["trades"] == 1
    assert daily_pnl[0]["pnl"] == pytest.approx(40.0)

    metrics = monitor.get_performance_metrics()
    assert metrics.total_trades == 1
    assert metrics.winning_trades == 1
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.profit_factor >= 0.0
