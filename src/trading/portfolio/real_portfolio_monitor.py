"""
Real Portfolio Monitor Implementation
Replaces the mock with functional portfolio tracking and P&L calculation
"""

import asyncio
import sqlite3
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..models import Position, PortfolioSnapshot
from ...config.portfolio_config import PortfolioConfig

logger = logging.getLogger(__name__)

from ..monitoring.performance_tracker import PerformanceMetrics as PerformanceMetrics


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
                    position_id=row[0],
                    symbol=row[1],
                    size=row[2],
                    entry_price=row[3],
                    current_price=row[4],
                    status=row[5],
                    stop_loss=row[6],
                    take_profit=row[7],
                    entry_time=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                    exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
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
                position.status.value,
                position.stop_loss,
                position.take_profit,
                position.entry_time.isoformat(),
                position.realized_pnl,
                position.unrealized_pnl
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._position_cache[position.position_id] = position
            
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
                self._position_cache[position_id].close(exit_price, exit_time)
            
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
            
            # Get historical snapshots
            cursor.execute('''
                SELECT * FROM portfolio_snapshots 
                ORDER BY timestamp DESC LIMIT 100
            ''')
            
            snapshots = cursor.fetchall()
            if not snapshots:
                return PerformanceMetrics(
                    total_return=0.0,
                    annualized_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0
                )
            
            # Calculate returns
            initial_value = self.initial_balance
            current_value = snapshots[0][2]  # Most recent total_value
            
            total_return = (current_value - initial_value) / initial_value
            
            # Calculate annualized return (simplified)
            days_elapsed = (datetime.now() - datetime.fromisoformat(snapshots[-1][1])).days
            if days_elapsed > 0:
                annualized_return = total_return * (365 / days_elapsed)
            else:
                annualized_return = 0.0
            
            # Get closed positions for win rate calculation
            cursor.execute('''
                SELECT COUNT(*), SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)
                FROM positions WHERE status = 'CLOSED'
            ''')
            
            result = cursor.fetchone()
            total_trades = result[0] if result[0] else 0
            winning_trades = result[1] if result[1] else 0
            losing_trades = total_trades - winning_trades
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit factor
            cursor.execute('''
                SELECT SUM(realized_pnl) FROM positions WHERE realized_pnl > 0
            ''')
            gross_profit = cursor.fetchone()[0] or 0.0
            
            cursor.execute('''
                SELECT ABS(SUM(realized_pnl)) FROM positions WHERE realized_pnl < 0
            ''')
            gross_loss = cursor.fetchone()[0] or 0.0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Calculate max drawdown (simplified)
            max_drawdown = 0.0
            peak_value = initial_value
            for snapshot in reversed(snapshots):
                value = snapshot[2]
                if value > peak_value:
                    peak_value = value
                else:
                    drawdown = (peak_value - value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            # Derive daily returns from snapshots (ordered oldest -> newest)
            try:
                values = [row[2] for row in reversed(snapshots) if row and len(row) > 2 and row[2] is not None]
                returns: List[float] = []
                for i in range(1, len(values)):
                    prev = values[i - 1]
                    curr = values[i]
                    if prev and prev != 0:
                        returns.append((curr - prev) / prev)
                if len(returns) > 1:
                    mu = sum(returns) / len(returns)
                    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
                    sigma = var ** 0.5
                    sharpe_ratio = (mu / sigma) * (252 ** 0.5) if sigma > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
            except Exception:
                sharpe_ratio = 0.0
            
            conn.close()
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0
            )
    
    def get_position_history(self, days: int = 30) -> List[Position]:
        """Get position history for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM positions 
                WHERE entry_time > datetime('now', '-{} days')
                ORDER BY entry_time DESC
            '''.format(days))
            
            positions = []
            for row in cursor.fetchall():
                position = Position(
                    position_id=row[0],
                    symbol=row[1],
                    size=row[2],
                    entry_price=row[3],
                    current_price=row[4],
                    status=row[5],
                    stop_loss=row[6],
                    take_profit=row[7],
                    entry_time=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                    exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
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
            
            cursor.execute('''
                SELECT DATE(timestamp) as date,
                       SUM(realized_pnl) as daily_pnl,
                       COUNT(*) as trades
                FROM positions
                WHERE exit_time > datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            '''.format(days))
            
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
