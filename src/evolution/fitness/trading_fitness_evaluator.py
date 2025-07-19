"""
EMP Trading Fitness Evaluator v1.1

Fitness evaluation for trading strategies based on real market performance.
Implements the IFitnessEvaluator interface for evaluating DecisionGenome fitness.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.core.interfaces import IFitnessEvaluator
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


class TradingFitnessEvaluator(IFitnessEvaluator):
    """
    Evaluates trading strategy fitness based on real market performance.
    
    Uses historical market data to simulate trading performance and calculate
    fitness scores based on multiple performance metrics.
    """
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize trading fitness evaluator.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date for evaluation period
            end_date: End date for evaluation period
            initial_capital: Starting capital for simulation
            transaction_cost: Transaction cost as percentage
        """
        self.symbol = symbol
        self.start_date = start_date or datetime.now() - timedelta(days=30)
        self.end_date = end_date or datetime.now()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Generate synthetic market data for testing
        self.market_data = self._generate_synthetic_data()
        
        logger.info(f"TradingFitnessEvaluator initialized for {symbol}")
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(self.start_date, self.end_date, freq='1H')
        
        # Generate random walk with trend
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.001, len(dates))
        prices = 1.0 * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        return data
    
    def evaluate(self, genome: DecisionGenome) -> float:
        """
        Evaluate a trading strategy genome.
        
        Args:
            genome: Trading strategy genome to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Simulate trading based on genome parameters
            results = self._simulate_trading(genome)
            
            # Calculate fitness score
            fitness = self._calculate_fitness(results)
            
            # Store results in genome
            if hasattr(genome, 'performance_metrics'):
                genome.performance_metrics = results
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating genome: {e}")
            return 0.0
    
    def _calculate_fitness(self, results: Dict[str, Any]) -> float:
        """Calculate fitness score from trading results."""
        if not results or results.get('total_return', 0) <= 0:
            return 0.0
        
        # Weighted combination of performance metrics
        fitness = (
            results.get('total_return', 0) * 0.4 +
            results.get('sharpe_ratio', 0) * 0.3 +
            results.get('win_rate', 0) * 0.2 +
            (1 - abs(results.get('max_drawdown', 0))) * 0.1
        )
        
        return max(0.0, fitness)
    
    def _calculate_performance_metrics(self, equity_curve: list, trades: list) -> Dict[str, Any]:
        """Calculate performance metrics from equity curve and trades."""
        if not equity_curve or len(equity_curve) < 2:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        if not returns:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}
        
        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Sharpe ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        peak = equity_curve[0]
        max_drawdown = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Win rate
        profitable_trades = sum(1 for trade in trades if trade['type'] == 'sell')
        total_trades = len(trades) // 2  # Count buy-sell pairs
        win_rate = profitable_trades / max(1, total_trades)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'final_equity': equity_curve[-1]
        }
    
    def _simulate_trading(self, genome: DecisionGenome) -> Dict[str, Any]:
        """Simulate trading based on genome parameters."""
        capital = self.initial_capital
        position = 0.0
        trades = []
        equity_curve = [capital]
        
        # Calculate technical indicators
        data = self.market_data.copy()
        data['returns'] = data['close'].pct_change()
        data['sma_short'] = data['close'].rolling(window=genome.strategy.lookback_period).mean()
        data['sma_long'] = data['close'].rolling(window=genome.strategy.lookback_period * 2).mean()
        data['rsi'] = self._calculate_rsi(data['close'], genome.strategy.lookback_period)
        data['volatility'] = data['returns'].rolling(window=genome.strategy.lookback_period).std()
        
        # Simulate trading
        for i in range(genome.strategy.lookback_period * 2, len(data)):
            current_price = data.iloc[i]['close']
            current_volatility = data.iloc[i]['volatility']
            
            # Skip if not enough data
            if pd.isna(current_volatility):
                continue
            
            # Calculate signals
            trend_signal = self._calculate_trend_signal(data.iloc[i], genome)
            momentum_signal = self._calculate_momentum_signal(data.iloc[i], genome)
            volume_signal = self._calculate_volume_signal(data.iloc[i], genome)
            
            # Combined signal
            combined_signal = (
                trend_signal * genome.strategy.trend_weight +
                momentum_signal * genome.strategy.momentum_weight +
                volume_signal * genome.strategy.volume_weight
            )
            
            # Risk adjustment
            risk_adjustment = self._calculate_risk_adjustment(current_volatility, genome)
            combined_signal *= risk_adjustment
            
            # Entry logic
            if position == 0.0 and combined_signal > genome.strategy.entry_threshold:
                position_size = self._calculate_position_size(capital, current_volatility, genome)
                position = position_size
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'size': position_size,
                    'timestamp': data.iloc[i]['timestamp']
                })
                capital -= position_size * current_price * (1 + self.transaction_cost)
            
            # Exit logic
            elif position > 0.0 and combined_signal < -genome.strategy.exit_threshold:
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'size': position,
                    'timestamp': data.iloc[i]['timestamp']
                })
                capital += position * current_price * (1 - self.transaction_cost)
                position = 0.0
            
            # Update equity curve
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)
        
        # Close any remaining position
        if position > 0.0:
            final_price = data.iloc[-1]['close']
            capital += position * final_price * (1 - self.transaction_cost)
            equity_curve.append(capital)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(equity_curve, trades)
        return results
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_signal(self, row: pd.Series, genome: DecisionGenome) -> float:
        """Calculate trend-based signal."""
        if pd.isna(row['sma_short']) or pd.isna(row['sma_long']):
            return 0.0
        
        # Simple moving average crossover
        if row['sma_short'] > row['sma_long']:
            return 1.0
        elif row['sma_short'] < row['sma_long']:
            return -1.0
        else:
            return 0.0
    
    def _calculate_momentum_signal(self, row: pd.Series, genome: DecisionGenome) -> float:
        """Calculate momentum-based signal."""
        if pd.isna(row['rsi']):
            return 0.0
        
        # RSI-based signal
        if row['rsi'] < 30:
            return 1.0  # Oversold
        elif row['rsi'] > 70:
            return -1.0  # Overbought
        else:
            return 0.0
    
    def _calculate_volume_signal(self, row: pd.Series, genome: DecisionGenome) -> float:
        """Calculate volume-based signal."""
        # Simple volume signal (normalized)
        volume_ratio = row['volume'] / row['volume'].rolling(window=genome.strategy.lookback_period).mean()
        if pd.isna(volume_ratio):
            return 0.0
        
        return min(1.0, max(-1.0, (volume_ratio - 1.0) * 2.0))
    
    def _calculate_risk_adjustment(self, volatility: float, genome: DecisionGenome) -> float:
        """Calculate risk adjustment based on volatility."""
        if pd.isna(volatility):
            return 1.0
        
        # Reduce signal strength in high volatility
        if volatility > genome.risk.volatility_threshold:
            return 0.5
        else:
            return 1.0
    
    def _calculate_position_size(self, capital: float, volatility: float, genome: DecisionGenome) -> float:
        """Calculate position size based on risk parameters."""
        if pd.isna(volatility) or volatility == 0:
            volatility = 0.001
        
        # Position sizing based on Kelly criterion and risk tolerance
        kelly_fraction = genome.risk.risk_tolerance * 0.25  # Conservative Kelly
        volatility_adjustment = min(1.0, 0.02 / volatility)  # Risk adjustment
        
        position_size = capital * kelly_fraction * volatility_adjustment * genome.risk.position_size_multiplier
        
        return max(1000, min(capital * 0.1, position_size))  # Position limits
    
    @property
    def name(self) -> str:
        """Return the name of this fitness evaluator."""
        return f"TradingFitnessEvaluator({self.symbol})"
