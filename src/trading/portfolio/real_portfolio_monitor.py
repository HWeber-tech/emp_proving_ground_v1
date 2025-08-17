"""
Real Portfolio Monitor Implementation
Replaces the mock with functional portfolio tracking and P&L calculation
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ...config.portfolio_config import PortfolioConfig
from ..models import Position
from ..monitoring.performance_tracker import PerformanceMetrics as PerformanceMetrics

logger = logging.getLogger(__name__)

@dataclass
class PortfolioSnapshot:
    total_value: float
    cash_balance: float
    positions: List[Position]
    unrealized_pnl: float
    realized_pnl: float


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
        
        logger.info(f"RealPortfolioMonitor initialized with initial balance: {self.initial_balance}")
    
    def _init_database(self) -> None:
        """Initialize portfolio database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
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
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_value REAL,
                cash_balance REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                position_count INTEGER
            )
        ''')
        
        cursor.execute('''
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
        ''')
        
        conn.commit()
        conn.close()
    
    def get_balance(self) -> float:
        """Get current cash balance"""
        return self.initial_balance
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM positions WHERE status = 'OPEN'
            ''')
            
            positions = []
            for row in cursor.fetchall():
                position = Position(
                    row[1],
                    size=row[2],
                    entry_price=row[3],
                    position_id=row[0],
                    current_price=row[4],
                    realized_pnl=row[10],
                    unrealized_pnl=row[11]
                )
                positions.append(position)
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        positions = self.get_positions()
        position_value = sum(pos.value for pos in positions)
        return self.initial_balance + position_value
    
    def add_position(self, position: Position) -> bool:
        """Add a new position"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO positions (
                    position_id, symbol, size, entry_price, current_price, status,
                    stop_loss, take_profit, entry_time, realized_pnl, unrealized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.position_id,
                position.symbol,
                position.size,
                position.entry_price,
                position.current_price,
                'OPEN',
                None,
                None,
                datetime.now().isoformat(),
                position.realized_pnl,
                position.unrealized_pnl
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._position_cache[str(position.position_id)] = position
            
            logger.info(f"Added position: {position.position_id} for {position.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def update_position_price(self, position_id: str, new_price: float) -> bool:
        """Update position price and recalculate P&L"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current position
            cursor.execute('''
                SELECT * FROM positions WHERE position_id = ?
            ''', (position_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Position not found: {position_id}")
                return False
            
            # Calculate new unrealized P&L
            size = row[2]
            entry_price = row[3]
            unrealized_pnl = (new_price - entry_price) * size
            
            # Update position
            cursor.execute('''
                UPDATE positions 
                SET current_price = ?, unrealized_pnl = ?
                WHERE position_id = ?
            ''', (new_price, unrealized_pnl, position_id))
            
            conn.commit()
            conn.close()
            
            # Update cache
            if position_id in self._position_cache:
                self._position_cache[position_id].current_price = new_price
                self._position_cache[position_id].unrealized_pnl = unrealized_pnl
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating position price: {e}")
            return False
    
    def close_position(self, position_id: str, exit_price: float, exit_time: Optional[datetime] = None) -> bool:
        """Close a position and calculate final P&L"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get position
            cursor.execute('''
                SELECT * FROM positions WHERE position_id = ?
            ''', (position_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Position not found: {position_id}")
                return False
            
            # Calculate final P&L
            size = row[2]
            entry_price = row[3]
            realized_pnl = (exit_price - entry_price) * size
            
            # Update position
            cursor.execute('''
                UPDATE positions 
                SET status = 'CLOSED', exit_time = ?, realized_pnl = ?, current_price = ?
                WHERE position_id = ?
            ''', (
                (exit_time or datetime.now()).isoformat(),
                realized_pnl,
                exit_price,
                position_id
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            if position_id in self._position_cache:
                try:
                    self._position_cache[position_id].current_price = exit_price
                    self._position_cache[position_id].realized_pnl = realized_pnl
                except Exception:
                    pass
            
            logger.info(f"Closed position: {position_id} with P&L: {realized_pnl:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot"""
        try:
            positions = self.get_positions()
            
            # Calculate P&L
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            realized_pnl = sum(pos.realized_pnl for pos in positions)
            
            # Calculate position value
            position_value = sum(pos.value for pos in positions)
            cash_balance = self.initial_balance - position_value
            
            snapshot = PortfolioSnapshot(
                total_value=self.initial_balance + unrealized_pnl,
                cash_balance=cash_balance,
                positions=positions,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl
            )
            
            # Store snapshot
            self._store_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting portfolio snapshot: {e}")
            return PortfolioSnapshot(
                total_value=self.initial_balance,
                cash_balance=self.initial_balance,
                positions=[],
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
    
    def _store_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Store portfolio snapshot in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots (
                    total_value, cash_balance, unrealized_pnl, realized_pnl, position_count
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                snapshot.total_value,
                snapshot.cash_balance,
                snapshot.unrealized_pnl,
                snapshot.realized_pnl,
                len(snapshot.positions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Fetch recent snapshots (DESC, limit 100)
            snapshots = self._fetch_recent_snapshots(cursor)
            if not snapshots:
                conn.close()
                return self._build_empty_performance_metrics()

            # Returns
            initial_value = self.initial_balance
            current_value = snapshots[0][2]  # Most recent total_value
            total_return = self._calc_total_return(initial_value, current_value)
            annualized_return = self._calc_annualized_return(total_return, snapshots[-1][1])

            # Trading metrics
            total_trades, winning_trades, losing_trades = self._fetch_trade_counts(cursor)
            win_rate = self._calc_win_rate(total_trades, winning_trades)
            gross_profit, gross_loss = self._fetch_gross_profit_loss(cursor)
            profit_factor = self._calc_profit_factor(gross_profit, gross_loss)
            avg_win, avg_loss, avg_trade_duration = self._compute_trade_aggregates(cursor)

            # Risk metrics from snapshots
            sharpe_ratio, daily_returns, volatility, sortino_ratio, var_95, cvar_95 = (
                self._compute_risk_metrics_from_snapshots(snapshots)
            )

            # Max drawdown (preserve simplified method)
            max_drawdown = self._calc_max_drawdown(initial_value, snapshots)

            # Period timestamps from snapshots
            start_date = datetime.fromisoformat(snapshots[-1][1]) if snapshots[-1][1] else datetime.now()
            end_date = datetime.fromisoformat(snapshots[0][1]) if snapshots[0][1] else datetime.now()

            conn.close()

            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                daily_returns=daily_returns,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                strategy_performance={},
                regime_performance={},
                correlation_matrix=pd.DataFrame(),
                start_date=start_date,
                end_date=end_date,
                last_updated=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._build_empty_performance_metrics()

    # --- Internal helpers for performance metrics --------------------------------
    def _build_metric_inputs(self, cursor, snapshots):
        """Prepare shared inputs for metric calculators."""
        initial_value = self.initial_balance
        current_value = snapshots[0][2]  # Most recent total_value
        oldest_timestamp = snapshots[-1][1]
        return {
            "cursor": cursor,
            "snapshots": snapshots,
            "initial_value": initial_value,
            "current_value": current_value,
            "oldest_timestamp": oldest_timestamp,
        }

    def _calc_returns_metrics(self, data):
        """Calculate return-related metrics."""
        total_return = self._calc_total_return(data["initial_value"], data["current_value"])
        annualized_return = self._calc_annualized_return(total_return, data["oldest_timestamp"])
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
        }

    def _calc_trading_metrics(self, data):
        """Calculate trading-related aggregates and ratios."""
        cursor = data["cursor"]
        total_trades, winning_trades, losing_trades = self._fetch_trade_counts(cursor)
        win_rate = self._calc_win_rate(total_trades, winning_trades)
        gross_profit, gross_loss = self._fetch_gross_profit_loss(cursor)
        profit_factor = self._calc_profit_factor(gross_profit, gross_loss)
        avg_win, avg_loss, avg_trade_duration = self._compute_trade_aggregates(cursor)
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_trade_duration,
        }

    def _calc_risk_metrics(self, data):
        """Calculate risk-related metrics from snapshots."""
        sharpe_ratio, daily_returns, volatility, sortino_ratio, var_95, cvar_95 = (
            self._compute_risk_metrics_from_snapshots(data["snapshots"])
        )
        return {
            "sharpe_ratio": sharpe_ratio,
            "daily_returns": daily_returns,
            "volatility": volatility,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def _calc_drawdown_metrics(self, data):
        """Calculate drawdown-related metrics."""
        return {
            "max_drawdown": self._calc_max_drawdown(data["initial_value"], data["snapshots"])
        }

    def _extract_period_timestamps(self, snapshots):
        """Extract start and end dates from snapshots (oldest to newest)."""
        start_date = datetime.fromisoformat(snapshots[-1][1]) if snapshots[-1][1] else datetime.now()
        end_date = datetime.fromisoformat(snapshots[0][1]) if snapshots[0][1] else datetime.now()
        return start_date, end_date

    def _assemble_performance_report(self, groups, start_date, end_date) -> PerformanceMetrics:
        """Assemble the PerformanceMetrics dataclass from computed groups."""
        return PerformanceMetrics(
            total_return=groups["total_return"],
            annualized_return=groups["annualized_return"],
            daily_returns=groups["daily_returns"],
            volatility=groups["volatility"],
            sharpe_ratio=groups["sharpe_ratio"],
            sortino_ratio=groups["sortino_ratio"],
            max_drawdown=groups["max_drawdown"],
            var_95=groups["var_95"],
            cvar_95=groups["cvar_95"],
            total_trades=groups["total_trades"],
            winning_trades=groups["winning_trades"],
            losing_trades=groups["losing_trades"],
            win_rate=groups["win_rate"],
            avg_win=groups["avg_win"],
            avg_loss=groups["avg_loss"],
            profit_factor=groups["profit_factor"],
            avg_trade_duration=groups["avg_trade_duration"],
            strategy_performance={},
            regime_performance={},
            correlation_matrix=pd.DataFrame(),
            start_date=start_date,
            end_date=end_date,
            last_updated=datetime.now(),
        )
    def _fetch_recent_snapshots(self, cursor):
        """Fetch recent portfolio snapshots (DESC, limit 100)."""
        cursor.execute('''
            SELECT * FROM portfolio_snapshots
            ORDER BY timestamp DESC LIMIT 100
        ''')
        return cursor.fetchall()

    def _build_empty_performance_metrics(self) -> PerformanceMetrics:
        """Build default zeroed PerformanceMetrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            avg_trade_duration=0.0,
            strategy_performance={},
            regime_performance={},
            correlation_matrix=pd.DataFrame(),
            start_date=datetime.now(),
            end_date=datetime.now(),
            last_updated=datetime.now(),
        )

    def _calc_total_return(self, initial_value: float, current_value: float) -> float:
        """Compute total return as (current - initial) / initial."""
        return (current_value - initial_value) / initial_value if initial_value != 0 else 0.0

    def _calc_annualized_return(self, total_return: float, oldest_timestamp: str) -> float:
        """Compute simplified annualized return based on days elapsed."""
        days_elapsed = (datetime.now() - datetime.fromisoformat(oldest_timestamp)).days
        if days_elapsed > 0:
            return total_return * (365 / days_elapsed)
        return 0.0

    def _fetch_trade_counts(self, cursor):
        """Fetch total/winning/losing trade counts for CLOSED positions."""
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
            FROM positions WHERE status = 'CLOSED'
        ''')
        result = cursor.fetchone()
        total_trades = result[0] if result and result[0] else 0
        winning_trades = result[1] if result and result[1] else 0
        losing_trades = total_trades - winning_trades
        return total_trades, winning_trades, losing_trades

    def _calc_win_rate(self, total_trades: int, winning_trades: int) -> float:
        """Compute win rate as winning_trades / total_trades (guarded)."""
        return winning_trades / total_trades if total_trades > 0 else 0.0

    def _fetch_gross_profit_loss(self, cursor):
        """Fetch gross profit and gross loss aggregates."""
        cursor.execute('''
            SELECT SUM(realized_pnl) FROM positions WHERE realized_pnl > 0
        ''')
        gp_row = cursor.fetchone()
        gross_profit = (gp_row[0] if gp_row and gp_row[0] is not None else 0.0)

        cursor.execute('''
            SELECT ABS(SUM(realized_pnl)) FROM positions WHERE realized_pnl < 0
        ''')
        gl_row = cursor.fetchone()
        gross_loss = (gl_row[0] if gl_row and gl_row[0] is not None else 0.0)
        return gross_profit, gross_loss

    def _calc_profit_factor(self, gross_profit: float, gross_loss: float) -> float:
        """Compute profit factor as gross_profit / gross_loss (guarded)."""
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def _calc_max_drawdown(self, initial_value: float, snapshots) -> float:
        """Compute simplified max drawdown using snapshot total_value field."""
        max_drawdown = 0.0
        peak_value = initial_value
        for snapshot in reversed(snapshots):
            value = snapshot[2]
            if value > peak_value:
                peak_value = value
            else:
                drawdown = (peak_value - value) / peak_value if peak_value != 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    def _compute_risk_metrics_from_snapshots(self, snapshots):
        """Compute Sharpe, daily returns, volatility, Sortino, VaR, CVaR from snapshots."""
        try:
            values = [row[2] for row in reversed(snapshots) if row and len(row) > 2 and row[2] is not None]
            returns: List[float] = []
            for i in range(1, len(values)):
                prev = values[i - 1]
                curr = values[i]
                if prev and prev != 0:
                    returns.append((curr - prev) / prev)

            daily_returns = returns
            if len(returns) > 1:
                mu = sum(returns) / len(returns)
                var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
                sigma = var ** 0.5
                volatility = sigma * (252 ** 0.5)
                sharpe_ratio = (mu / sigma) * (252 ** 0.5) if sigma > 0 else 0.0

                negatives = [r for r in returns if r < 0]
                if len(negatives) == 0:
                    sortino_ratio = float('inf') if mu > 0 else 0.0
                else:
                    downside_var = sum((r - 0.0) ** 2 for r in negatives) / len(negatives)
                    downside_sigma = downside_var ** 0.5
                    sortino_ratio = (mu / downside_sigma) * (252 ** 0.5) if downside_sigma > 0 else 0.0

                # VaR/CVaR at 95%
                sorted_returns = sorted(returns)
                idx = int(0.05 * len(sorted_returns))
                if idx >= len(sorted_returns):
                    idx = len(sorted_returns) - 1
                var_val = sorted_returns[idx]
                cvar_vals = [r for r in returns if r <= var_val]
                cvar_val = (sum(cvar_vals) / len(cvar_vals)) if cvar_vals else 0.0
                var_95 = abs(var_val)
                cvar_95 = abs(cvar_val)
            else:
                sharpe_ratio = 0.0
                volatility = 0.0
                sortino_ratio = 0.0
                var_95 = 0.0
                cvar_95 = 0.0

            return sharpe_ratio, daily_returns, volatility, sortino_ratio, var_95, cvar_95
        except Exception:
            return 0.0, [], 0.0, 0.0, 0.0, 0.0

    def _compute_trade_aggregates(self, cursor):
        """Compute avg_win, avg_loss, and avg_trade_duration (hours) from positions table."""
        # Average win
        cursor.execute('''
            SELECT AVG(realized_pnl) FROM positions WHERE realized_pnl > 0
        ''')
        row = cursor.fetchone()
        avg_win = row[0] if row and row[0] is not None else 0.0

        # Average loss (absolute)
        cursor.execute('''
            SELECT AVG(realized_pnl) FROM positions WHERE realized_pnl < 0
        ''')
        row = cursor.fetchone()
        avg_loss = abs(row[0]) if row and row[0] is not None else 0.0

        # Average trade duration in hours for closed positions
        cursor.execute('''
            SELECT entry_time, exit_time FROM positions
            WHERE status = 'CLOSED' AND entry_time IS NOT NULL AND exit_time IS NOT NULL
        ''')
        durations = []
        for et, xt in cursor.fetchall():
            try:
                start = datetime.fromisoformat(et)
                end = datetime.fromisoformat(xt)
                durations.append((end - start).total_seconds() / 3600.0)
            except Exception:
                continue
        avg_trade_duration = (sum(durations) / len(durations)) if durations else 0.0

        return avg_win, avg_loss, avg_trade_duration
    
    def get_position_history(self, days: int = 30) -> List[Position]:
        """Get position history for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Bandit B608: parameterized query to avoid SQL injection
            cursor.execute('''
                SELECT * FROM positions
                WHERE entry_time > datetime('now', ?)
                ORDER BY entry_time DESC
            ''', (f'-{int(days)} days',))
            
            positions = []
            for row in cursor.fetchall():
                position = Position(
                    row[1],
                    size=row[2],
                    entry_price=row[3],
                    position_id=row[0],
                    current_price=row[4],
                    realized_pnl=row[10],
                    unrealized_pnl=row[11]
                )
                positions.append(position)
            
            conn.close()
            return positions
            
        except Exception as e:
            logger.error(f"Error getting position history: {e}")
            return []
    
    def get_daily_pnl(self, days: int = 30) -> List[Dict[str, float]]:
        """Get daily P&L for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Bandit B608: parameterized query to avoid SQL injection
            cursor.execute('''
                SELECT DATE(timestamp) as date,
                       SUM(realized_pnl) as daily_pnl,
                       COUNT(*) as trades
                FROM positions
                WHERE exit_time > datetime('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (f'-{int(days)} days',))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'date': row[0],
                    'pnl': row[1] or 0.0,
                    'trades': row[2] or 0
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return []
