"""
Backtest Engine

Specialized backtesting engine for historical strategy simulation.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.sensory.core.base import MarketData
from ..base_strategy import BaseStrategy, StrategySignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of backtest simulation"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_history: List[Dict[str, Any]]
    execution_time: float


class BacktestEngine:
    """
    Advanced Backtesting Engine
    
    Implements comprehensive backtesting with:
    - Historical data processing
    - Strategy simulation
    - Realistic execution modeling
    - Transaction costs
    - Slippage modeling
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Backtest state
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [initial_capital]
        self.drawdown_curve = [0.0]
        
        logger.info(f"BacktestEngine initialized: capital={initial_capital}")
    
    async def run_backtest(self, strategy: BaseStrategy, 
                    historical_data: Dict[str, List[MarketData]],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResult:
        """Run complete backtest simulation"""
        
        import time
        start_time = time.time()
        
        # Reset backtest state
        self._reset_state()
        
        # Filter data by date range
        filtered_data = self._filter_data_by_date(historical_data, start_date, end_date)
        
        # Sort data by timestamp
        sorted_data = self._sort_data_by_timestamp(filtered_data)
        
        # Run simulation
        for timestamp, data_batch in sorted_data.items():
            await self._process_timestep(strategy, data_batch, timestamp)
        
        # Calculate final results
        result = self._calculate_results(strategy)
        result.execution_time = time.time() - start_time
        
        logger.info(f"Backtest completed: {result.total_trades} trades, {result.total_return:.2%} return")
        
        return result
    
    def _reset_state(self) -> None:
        """Reset backtest state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdown_curve = [0.0]
    
    def _filter_data_by_date(self, historical_data: Dict[str, List[MarketData]],
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> Dict[str, List[MarketData]]:
        """Filter historical data by date range"""
        filtered_data = {}
        
        for symbol, data in historical_data.items():
            filtered_symbol_data = []
            
            for market_data in data:
                if start_date and market_data.timestamp < start_date:
                    continue
                if end_date and market_data.timestamp > end_date:
                    continue
                filtered_symbol_data.append(market_data)
            
            if filtered_symbol_data:
                filtered_data[symbol] = filtered_symbol_data
        
        return filtered_data
    
    def _sort_data_by_timestamp(self, historical_data: Dict[str, List[MarketData]]) -> Dict[datetime, Dict[str, MarketData]]:
        """Sort data by timestamp for sequential processing"""
        timestamp_data = {}
        
        for symbol, data_list in historical_data.items():
            for market_data in data_list:
                timestamp = market_data.timestamp
                if timestamp not in timestamp_data:
                    timestamp_data[timestamp] = {}
                timestamp_data[timestamp][symbol] = market_data
        
        # Sort by timestamp
        sorted_timestamps = sorted(timestamp_data.keys())
        sorted_data = {timestamp: timestamp_data[timestamp] for timestamp in sorted_timestamps}
        
        return sorted_data
    
    async def _process_timestep(self, strategy: BaseStrategy, 
                         data_batch: Dict[str, MarketData], 
                         timestamp: datetime) -> None:
        """Process single timestep in backtest"""
        
        # Update strategy with current market data
        for symbol, market_data in data_batch.items():
            if symbol in strategy.symbols:
                # Generate signals
                signal = await strategy.generate_signal([market_data], symbol)
                
                if signal:
                    self._execute_signal(strategy, signal, market_data)
        
        # Update equity curve
        self._update_equity_curve(timestamp)
    
    def _execute_signal(self, strategy: BaseStrategy, signal: StrategySignal, market_data: MarketData) -> None:
        """Execute trading signal with realistic modeling"""
        
        symbol = signal.symbol
        current_price = market_data.close
        
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            execution_price = current_price * (1 + self.slippage)
        elif signal.signal_type == SignalType.SELL:
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price
        
        # Calculate position size
        position_size = strategy.calculate_position_size(signal, self.current_capital)
        
        # Calculate transaction costs
        transaction_value = abs(position_size * execution_price)
        transaction_cost = transaction_value * self.transaction_cost
        
        # Execute trade
        if signal.signal_type == SignalType.BUY:
            if symbol not in self.positions:
                self.positions[symbol] = 0.0
            
            # Check if we have enough capital
            required_capital = position_size * execution_price + transaction_cost
            if required_capital <= self.current_capital:
                self.positions[symbol] += position_size
                self.current_capital -= required_capital
                
                # Record trade
                self._record_trade(signal, execution_price, position_size, transaction_cost)
        
        elif signal.signal_type == SignalType.SELL:
            if symbol in self.positions and self.positions[symbol] > 0:
                # Close position
                close_size = min(position_size, self.positions[symbol])
                self.positions[symbol] -= close_size
                
                # Calculate proceeds
                proceeds = close_size * execution_price - transaction_cost
                self.current_capital += proceeds
                
                # Record trade
                self._record_trade(signal, execution_price, -close_size, transaction_cost)
    
    def _record_trade(self, signal: StrategySignal, execution_price: float, 
                     quantity: float, transaction_cost: float) -> None:
        """Record trade in history"""
        trade = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value,
            'execution_price': execution_price,
            'quantity': quantity,
            'transaction_cost': transaction_cost,
            'capital': self.current_capital,
            'confidence': signal.confidence,
            'metadata': signal.metadata
        }
        
        self.trades.append(trade)
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update equity curve with current portfolio value"""
        portfolio_value = self.current_capital
        
        # Add value of open positions
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                # This is simplified - in practice you'd get current market price
                # For now, we'll use the last known price
        
        self.equity_curve.append(portfolio_value)
        
        # Calculate drawdown
        peak_value = max(self.equity_curve)
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
        self.drawdown_curve.append(current_drawdown)
    
    def _calculate_results(self, strategy: BaseStrategy) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Basic metrics
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        total_trades = len(self.trades)
        
        # Calculate trade statistics
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        largest_win = 0.0
        largest_loss = 0.0
        
        for trade in self.trades:
            # Simplified P&L calculation
            if trade['quantity'] > 0:  # Buy trade
                # This would need to be paired with a sell trade for actual P&L
                pass
            else:  # Sell trade
                # This would calculate actual P&L
                pass
        
        # Calculate Sharpe ratio
        returns = self._calculate_returns()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Calculate max drawdown
        max_drawdown = max(self.drawdown_curve)
        
        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
        
        return BacktestResult(
            strategy_name=strategy.strategy_id,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=total_profit / winning_trades if winning_trades > 0 else 0.0,
            average_loss=total_loss / losing_trades if losing_trades > 0 else 0.0,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=0,  # Would need to calculate
            consecutive_losses=0,  # Would need to calculate
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
            trade_history=self.trades,
            execution_time=0.0
        )
    
    def _calculate_returns(self) -> List[float]:
        """Calculate daily returns from equity curve"""
        returns = []
        for i in range(1, len(self.equity_curve)):
            daily_return = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(daily_return)
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe_ratio 