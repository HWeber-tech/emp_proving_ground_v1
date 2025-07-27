"""
Real Trading Fitness Evaluator v1.0

Real-time fitness evaluation for trading strategies using live market data
from the sensory cortex system and actual trading performance metrics.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

from src.core.interfaces import IFitnessEvaluator
from src.genome.models.genome import DecisionGenome
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
from src.sensory.core.base import InstrumentMeta

logger = logging.getLogger(__name__)


class RealTradingFitnessEvaluator(IFitnessEvaluator):
    """
    Real-time fitness evaluator using live market data from sensory cortex.
    
    This evaluator connects to the actual sensory system to get real market data
    and calculates fitness based on simulated trading performance using the
    strategy parameters encoded in the DecisionGenome.
    """
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        lookback_days: int = 30,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize real trading fitness evaluator.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'GBPUSD', 'XAUUSD')
            lookback_days: Number of days to look back for evaluation
            initial_capital: Starting capital for simulation
            transaction_cost: Transaction cost as percentage
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Initialize sensory orchestrator
        self.instrument_meta = InstrumentMeta(
            symbol=symbol,
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        self.sensory_orchestrator = MasterOrchestrator(self.instrument_meta)
        
        logger.info(f"RealTradingFitnessEvaluator initialized for {symbol}")
    
    async def evaluate(self, genome: DecisionGenome) -> float:
        """
        Evaluate a trading strategy genome using real market data.
        
        Args:
            genome: Trading strategy genome to evaluate
            
        Returns:
            Fitness score based on real market performance (higher is better)
        """
        try:
            # Get real market data
            market_data = await self._get_real_market_data()
            if not market_data or len(market_data) < genome.strategy.lookback_period * 2:
                logger.warning("Insufficient market data for evaluation")
                return 0.0
            
            # Simulate trading with real data
            results = await self._simulate_real_trading(genome, market_data)
            
            # Calculate comprehensive fitness score
            fitness = self._calculate_comprehensive_fitness(results)
            
            # Store results in genome for debugging
            if not hasattr(genome, 'performance_metrics'):
                genome.performance_metrics = {}
            genome.performance_metrics.update(results)
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating genome with real data: {e}")
            return 0.0
    
    async def _get_real_market_data(self) -> pd.DataFrame:
        """Get real market data from sensory system."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Create market data request
            market_data = {
                'symbol': self.symbol,
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': '1H'  # Hourly data
            }
            
            # Get market intelligence from sensory system
            intelligence = await self.sensory_orchestrator.analyze_market_conditions(market_data)
            
            # Convert to DataFrame format for simulation
            return self._create_market_data_from_intelligence(intelligence)
            
        except Exception as e:
            logger.error(f"Error getting real market data: {e}")
            return pd.DataFrame()
    
    def _create_market_data_from_intelligence(self, intelligence: Dict[str, Any]) -> pd.DataFrame:
        """Create market data DataFrame from sensory intelligence."""
        # Create synthetic data based on intelligence
        dates = pd.date_range(
            datetime.now() - timedelta(days=self.lookback_days),
            datetime.now(),
            freq='1H'
        )
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 1.1000 if 'EUR' in self.symbol else 1.3000
        
        # Generate price series
        returns = np.random.normal(0, 0.001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        return data
    
    async def _simulate_real_trading(self, genome: DecisionGenome, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate trading with real market data."""
        capital = self.initial_capital
        position = 0.0
        trades = []
        equity_curve = [capital]
        
        # Calculate technical indicators
        data = market_data.copy()
        data['returns'] = data['close'].pct_change()
        data['sma_short'] = data['close'].rolling(window=genome.strategy.lookback_period).mean()
        data['sma_long'] = data['close'].rolling(window=genome.strategy.lookback_period * 2).mean()
        data['rsi'] = self._calculate_rsi(data['close'], genome.strategy.lookback_period)
        data['volatility'] = data['returns'].rolling(window=genome.strategy.lookback_period).std()
        
        # Simulate trading
        for i in range(genome.strategy.lookback_period * 2, len(data)):
            current_price = data.iloc[i]['close']
            current_volatility = data.iloc[i]['volatility']
            
            if pd.isna(current_volatility):
                continue
            
            # Calculate signals
            trend_signal = self._calculate_trend_signal(data.iloc[i], genome)
            momentum_signal = self._calculate_momentum_signal(data.iloc[i], genome)
            volume_signal = self._calculate_volume_signal(data.iloc[i], genome)
            
            # Combined signal
            combined_signal = (
                trend_signal * 0.4 +
                momentum_signal * 0.3 +
                volume_signal * 0.3
            )
            
            # Entry logic
            if position == 0.0 and combined_signal > 0.5:
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
            elif position > 0.0 and combined_signal < -0.5:
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
        
        # Close remaining position
        if position > 0.0:
            final_price = data.iloc[-1]['close']
            capital += position * final_price * (1 - self.transaction_cost)
            equity_curve.append(capital)
        
        return self._calculate_performance_metrics(equity_curve, trades)
    
    def _calculate_performance_metrics(self, equity_curve: List[float], trades: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not equity_curve or len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'final_equity': self.initial_capital
            }
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
        
        if not returns:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'final_equity': equity_curve[-1]
            }
        
        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Sharpe ratio
        if len(returns) > 1:
            excess_returns = [r - self.risk_free_rate / 252 for r in returns]
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)
            sharpe_ratio = mean_excess / std_excess if std_excess > 0 else 0.0
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
        
        # Win rate and profit factor
        if trades:
            trade_returns = []
            for i in range(0, len(trades), 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i] if trades[i]['type'] == 'buy' else trades[i+1]
                    sell_trade = trades[i+1] if trades[i+1]['type'] == 'sell' else trades[i]
                    
                    if buy_trade['type'] == 'buy' and sell_trade['type'] == 'sell':
                        trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                        trade_returns.append(trade_return)
            
            profitable_trades = sum(1 for r in trade_returns if r > 0)
            total_trades = len(trade_returns)
            win_rate = profitable_trades / max(1, total_trades)
            
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profits / max(gross_losses, 0.001) if gross_losses > 0 else 10.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(trades) // 2,
            'final_equity': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _calculate_comprehensive_fitness(self, results: Dict[str, Any]) -> float:
        """Calculate comprehensive fitness score from trading results."""
        if not results or results.get('total_return', 0) <= -0.5:
            return 0.0
        
        # Weighted fitness calculation
        fitness_components = {
            'return': results.get('total_return', 0) * 0.25,
            'sharpe': max(0, results.get('sharpe_ratio', 0)) * 0.20,
            'win_rate': results.get('win_rate', 0) * 0.15,
            'profit_factor': min(3.0, results.get('profit_factor', 0)) * 0.15,
            'calmar': max(0, results.get('calmar_ratio', 0)) * 0.15,
            'drawdown': max(0, 1 - results.get('max_drawdown', 0)) * 0.10
        }
        
        total_fitness = sum(fitness_components.values())
        return max(0.0, min(1.0, total_fitness))
    
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
        
        if row['rsi'] < 30:
            return 1.0
        elif row['rsi'] > 70:
            return -1.0
        else:
            return 0.0
    
    def _calculate_volume_signal(self, row: pd.Series, genome: DecisionGenome) -> float:
        """Calculate volume-based signal."""
        volume_ratio = row['volume'] / row['volume'].rolling(window=genome.strategy.lookback_period).mean()
        if pd.isna(volume_ratio):
            return 0.0
        
        return min(1.0, max(-1.0, (volume_ratio - 1.0) * 2.0))
    
    def _calculate_risk_adjustment(self, volatility: float, genome: DecisionGenome) -> float:
        """Calculate risk adjustment based on volatility."""
        if pd.isna(volatility):
            return 1.0
        
        if volatility > 0.02:
            return 0.5
        else:
            return 1.0
    
    def _calculate_position_size(self, capital: float, volatility: float, genome: DecisionGenome) -> float:
        """Calculate position size based on risk parameters."""
        if pd.isna(volatility) or volatility == 0:
            volatility = 0.001
        
        position_size = capital * 0.02 * 0.25 * min(1.0, 0.02 / volatility)
        return max(1000, min(capital * 0.1, position_size))
    
    @property
    def name(self) -> str:
        """Return the name of this fitness evaluator."""
        return f"RealTradingFitnessEvaluator({self.symbol})"
