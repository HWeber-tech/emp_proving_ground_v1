"""Runtime portfolio monitor backed by SQLite storage.

The original implementation relied on implicit commits, inline SQL literals,
and blanket ``except Exception`` blocks that obscured the root cause when
database operations failed.  This revision adopts structured connection
management, parameterised SQL statements, and narrower exception handling so
the security hardening sprint in the roadmap can show tangible progress.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, cast

import pandas as pd

from ...config.portfolio_config import PortfolioConfig
from ..models import Position
from ..models.portfolio_snapshot import PortfolioSnapshot
from ..monitoring.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class PortfolioMonitorError(RuntimeError):
    """Raised when the portfolio monitor fails to persist or retrieve data."""


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Best-effort conversion from SQLite payloads to ``datetime`` objects."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class RealPortfolioMonitor:
    """
    Real implementation of portfolio monitoring
    Replaces the mock with functional portfolio tracking
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.db_path = config.database_path
        self.initial_balance = config.initial_balance

        # Initialize database
        self._init_database()

        # Cache for performance
        self._position_cache: Dict[str, Position] = {}
        self._last_update = datetime.now()

        logger.info(
            "RealPortfolioMonitor initialised with initial balance %.2f",
            self.initial_balance,
        )

    @contextmanager
    def _managed_connection(self) -> Iterator[sqlite3.Connection]:
        """Return a managed SQLite connection with consistent settings."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - wrapped downstream
            conn.rollback()
            raise PortfolioMonitorError("SQLite operation failed") from exc
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialise portfolio tables if they do not exist."""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS positions (
                        position_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        status TEXT NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        exit_time DATETIME,
                        realized_pnl REAL DEFAULT 0.0,
                        unrealized_pnl REAL DEFAULT 0.0
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_value REAL,
                        cash_balance REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        position_count INTEGER
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id TEXT,
                        symbol TEXT,
                        action TEXT,
                        size REAL,
                        price REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        fees REAL DEFAULT 0.0
                    )
                """
                )
        except sqlite3.Error as exc:
            logger.exception("Failed to initialise portfolio database")
            raise PortfolioMonitorError("Failed to initialise portfolio database") from exc

    def get_balance(self) -> float:
        """Get current cash balance."""

        return float(self.initial_balance)

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """Convert a SQLite row into a ``Position`` instance."""

        entry_time = _parse_timestamp(row["entry_time"]) or datetime.now()
        exit_time = _parse_timestamp(row["exit_time"])
        status_value = row["status"]
        P = cast(Any, Position)
        position = P(
            symbol=row["symbol"],
            size=row["size"],
            entry_price=row["entry_price"],
            position_id=row["position_id"],
            current_price=row["current_price"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
        )
        position.status = status_value
        position.stop_loss = row["stop_loss"]
        position.take_profit = row["take_profit"]
        position.entry_time = entry_time
        position.exit_time = exit_time
        return position

    def get_positions(self) -> List[Position]:
        """Return all open positions from the backing store."""

        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM positions WHERE status = ?",
                    ("OPEN",),
                )
                rows = cursor.fetchall()
        except PortfolioMonitorError:
            logger.exception("Failed to read open positions")
            return []

        positions = [self._row_to_position(row) for row in rows]
        return positions

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions = self.get_positions()
        position_value = sum(pos.value for pos in positions)
        return float(self.initial_balance + position_value)

    def _normalise_status(self, status: object | None) -> str:
        """Return a canonical uppercase status string."""

        if status is None:
            return "OPEN"
        value = getattr(status, "value", status)
        return str(value).upper()

    def add_position(self, position: Position) -> bool:
        """Add a new position."""

        try:
            with self._managed_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (
                        position_id, symbol, size, entry_price, current_price, status,
                        stop_loss, take_profit, entry_time, realized_pnl, unrealized_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        position.position_id,
                        position.symbol,
                        position.size,
                        position.entry_price,
                        position.current_price,
                        self._normalise_status(position.status),
                        position.stop_loss,
                        position.take_profit,
                        (
                            position.entry_time.isoformat()
                            if isinstance(position.entry_time, datetime)
                            else datetime.now().isoformat()
                        ),
                        position.realized_pnl,
                        position.unrealized_pnl,
                    ),
                )
        except PortfolioMonitorError:
            logger.exception("Failed to add position %s", position.position_id)
            return False

        # Update cache
        self._position_cache[str(position.position_id or position.symbol)] = position
        logger.info("Added position %s for %s", position.position_id, position.symbol)
        return True

    def update_position_price(self, position_id: str, new_price: float) -> bool:
        """Update position price and recalculate P&L"""
        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    "SELECT size, entry_price FROM positions WHERE position_id = ?",
                    (position_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    logger.warning("Position not found: %s", position_id)
                    return False

                size = float(row["size"])
                entry_price = float(row["entry_price"])
                unrealized_pnl = (new_price - entry_price) * size

                conn.execute(
                    """
                    UPDATE positions
                    SET current_price = ?, unrealized_pnl = ?
                    WHERE position_id = ?
                """,
                    (new_price, unrealized_pnl, position_id),
                )
        except PortfolioMonitorError:
            logger.exception("Failed to update price for %s", position_id)
            return False

        if position_id in self._position_cache:
            cached = self._position_cache[position_id]
            cached.current_price = new_price
            cached.unrealized_pnl = unrealized_pnl

        return True

    def close_position(
        self, position_id: str, exit_price: float, exit_time: Optional[datetime] = None
    ) -> bool:
        """Close a position and calculate final P&L"""
        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    "SELECT size, entry_price FROM positions WHERE position_id = ?",
                    (position_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    logger.warning("Position not found: %s", position_id)
                    return False

                size = float(row["size"])
                entry_price = float(row["entry_price"])
                realized_pnl = (exit_price - entry_price) * size
                conn.execute(
                    """
                    UPDATE positions
                    SET status = ?, exit_time = ?, realized_pnl = ?, current_price = ?
                    WHERE position_id = ?
                """,
                    (
                        "CLOSED",
                        (exit_time or datetime.now()).isoformat(),
                        realized_pnl,
                        exit_price,
                        position_id,
                    ),
                )
        except PortfolioMonitorError:
            logger.exception("Failed to close position %s", position_id)
            return False

        if position_id in self._position_cache:
            self._position_cache[position_id].close(exit_price, exit_time)

        logger.info("Closed position %s with P&L %.4f", position_id, realized_pnl)
        return True

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot"""
        positions = self.get_positions()

        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        realized_pnl = sum(pos.realized_pnl for pos in positions)

        position_value = sum(pos.value for pos in positions)
        cash_balance = self.initial_balance - position_value

        snapshot = PortfolioSnapshot(
            total_value=self.initial_balance + unrealized_pnl,
            cash_balance=cash_balance,
            positions=positions,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

        self._store_snapshot(snapshot)
        return snapshot

    def _store_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Store portfolio snapshot in database"""
        try:
            with self._managed_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO portfolio_snapshots (
                        total_value, cash_balance, unrealized_pnl, realized_pnl, position_count
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        snapshot.total_value,
                        snapshot.cash_balance,
                        snapshot.unrealized_pnl,
                        snapshot.realized_pnl,
                        len(snapshot.positions),
                    ),
                )
        except PortfolioMonitorError:
            logger.exception("Failed to persist portfolio snapshot")

    @staticmethod
    def _make_performance_metrics(
        *,
        total_return: float = 0.0,
        annualized_return: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        win_rate: float = 0.0,
        profit_factor: float = 0.0,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
    ) -> PerformanceMetrics:
        """Create a PerformanceMetrics instance satisfying required fields with safe defaults."""
        now = datetime.now()
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0.0,
            max_drawdown=max_drawdown,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=profit_factor,
            avg_trade_duration=0.0,
            strategy_performance={},
            regime_performance={},
            correlation_matrix=pd.DataFrame(),
            start_date=now,
            end_date=now,
            last_updated=now,
        )

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        snapshots: List[sqlite3.Row]
        gross_profit = 0.0
        gross_loss = 0.0

        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, total_value, cash_balance, unrealized_pnl,
                           realized_pnl, position_count
                    FROM portfolio_snapshots
                    ORDER BY timestamp DESC LIMIT 100
                """
                )
                snapshots = cursor.fetchall()

                if not snapshots:
                    return self._make_performance_metrics(
                        total_return=0.0,
                        annualized_return=0.0,
                        sharpe_ratio=0.0,
                        max_drawdown=0.0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        total_trades=0,
                        winning_trades=0,
                        losing_trades=0,
                    )

                initial_value = self.initial_balance
                current_value = float(snapshots[0]["total_value"])
                total_return = (current_value - initial_value) / initial_value

                oldest_ts = _parse_timestamp(snapshots[-1]["timestamp"]) or datetime.now()
                days_elapsed = (datetime.now() - oldest_ts).days
                annualized_return = total_return * (365 / days_elapsed) if days_elapsed > 0 else 0.0

                cursor = conn.execute(
                    """
                    SELECT COUNT(*) AS trades,
                           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins
                    FROM positions WHERE status = ?
                """,
                    ("CLOSED",),
                )
                result = cursor.fetchone()
                total_trades = int(result["trades"] or 0)
                winning_trades = int(result["wins"] or 0)
                losing_trades = total_trades - winning_trades
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

                cursor = conn.execute(
                    "SELECT SUM(realized_pnl) AS profit FROM positions WHERE realized_pnl > 0"
                )
                profit_row = cursor.fetchone()
                if profit_row and profit_row["profit"] is not None:
                    gross_profit = float(profit_row["profit"])

                cursor = conn.execute(
                    "SELECT ABS(SUM(realized_pnl)) AS loss FROM positions WHERE realized_pnl < 0"
                )
                loss_row = cursor.fetchone()
                if loss_row and loss_row["loss"] is not None:
                    gross_loss = float(loss_row["loss"])
        except PortfolioMonitorError:
            logger.exception("Failed to compute performance metrics")
            return self._make_performance_metrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
            )

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        max_drawdown = 0.0
        peak_value = self.initial_balance
        for snapshot in reversed(snapshots):
            value = float(snapshot["total_value"])
            if value > peak_value:
                peak_value = value
            elif peak_value > 0:
                drawdown = (peak_value - value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)

        # Calculate daily returns for Sharpe ratio
        values = [
            float(row["total_value"])
            for row in reversed(snapshots)
            if row["total_value"] is not None
        ]
        returns: List[float] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if prev != 0:
                returns.append((curr - prev) / prev)
        if len(returns) > 1:
            mu = sum(returns) / len(returns)
            var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
            sigma = var**0.5
            sharpe_ratio = (mu / sigma) * (252**0.5) if sigma > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return self._make_performance_metrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
        )

    def get_position_history(self, days: int = 30) -> List[Position]:
        """Get position history for the last N days"""
        interval = max(int(days), 0)

        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM positions
                    WHERE entry_time > datetime('now', ?)
                    ORDER BY entry_time DESC
                """,
                    (f"-{interval} days",),
                )
                rows = cursor.fetchall()
        except PortfolioMonitorError:
            logger.exception("Failed to read position history")
            return []

        return [self._row_to_position(row) for row in rows]

    def get_daily_pnl(self, days: int = 30) -> List[Dict[str, float]]:
        """Get daily P&L for the last N days"""
        interval = max(int(days), 0)

        try:
            with self._managed_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DATE(exit_time) as date,
                           SUM(realized_pnl) as daily_pnl,
                           COUNT(*) as trades
                    FROM positions
                    WHERE exit_time > datetime('now', ?)
                    GROUP BY DATE(exit_time)
                    ORDER BY date DESC
                """,
                    (f"-{interval} days",),
                )
                rows = cursor.fetchall()
        except PortfolioMonitorError:
            logger.exception("Failed to read daily P&L")
            return []

        return [
            {"date": row["date"], "pnl": float(row["daily_pnl"] or 0.0), "trades": int(row["trades"] or 0)}
            for row in rows
        ]
