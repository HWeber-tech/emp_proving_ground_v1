#!/usr/bin/env python3
"""
Advanced Risk Management System - Phase 2.2

This module implements sophisticated risk management that integrates with evolved strategies
and live trading, providing portfolio-level risk controls, correlation analysis, and
dynamic position sizing based on market conditions and strategy performance.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_UP
import asyncio

from src.trading.strategy_manager import StrategyManager, StrategySignal
from src.trading.mock_ctrader_interface import MarketData, Position, Order

# Configure decimal precision
getcontext().prec = 12
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for portfolio analysis."""
    total_exposure: float = 0.0
    leverage_ratio: float = 0.0
    correlation_score: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    recovery_time: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RiskLimits:
    """Configurable risk limits for the trading system."""
    max_total_exposure: float = 0.3  # 30% of account
    max_leverage: float = 5.0  # 5:1 leverage
    max_drawdown: float = 0.15  # 15% max drawdown
    max_correlation: float = 0.7  # 70% correlation limit
    max_var_95: float = 0.02  # 2% daily VaR
    max_position_size: float = 0.05  # 5% per position
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_consecutive_losses: int = 5
    min_risk_reward_ratio: float = 1.5
    max_volatility: float = 0.25  # 25% annualized volatility
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility_adjusted
    correlation_lookback_days: int = 30
    volatility_lookback_days: int = 20


@dataclass
class PortfolioState:
    """Current portfolio state for risk analysis."""
    total_equity: float = 0.0
    total_margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    open_orders: List[Order] = field(default_factory=list)
    last_reset: datetime = field(default_factory=datetime.now)


class AdvancedRiskManager:
    """
    Advanced risk management system with portfolio-level controls.
    
    This class provides:
    - Portfolio correlation analysis
    - Dynamic position sizing
    - Real-time risk monitoring
    - Strategy-specific risk controls
    - Market regime adaptation
    """
    
    def __init__(self, risk_limits: RiskLimits, strategy_manager: StrategyManager):
        """
        Initialize the advanced risk manager.
        
        Args:
            risk_limits: Risk limit configuration
            strategy_manager: Strategy manager for performance data
        """
        self.risk_limits = risk_limits
        self.strategy_manager = strategy_manager
        self.portfolio_state = PortfolioState()
        self.risk_metrics = RiskMetrics()
        self.position_history = []
        self.pnl_history = []
        self.correlation_matrix = pd.DataFrame()
        self.volatility_cache = {}
        
        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info("Advanced risk manager initialized")
    
    def update_portfolio_state(self, positions: List[Position], equity: float, 
                              margin: float, orders: Optional[List[Order]] = None):
        """
        Update current portfolio state.
        
        Args:
            positions: Current open positions
            equity: Account equity
            margin: Used margin
            orders: Pending orders
        """
        self.portfolio_state.total_equity = equity
        self.portfolio_state.total_margin = margin
        self.portfolio_state.free_margin = equity - margin
        self.portfolio_state.margin_level = (equity / margin) if margin > 0 else float('inf')
        
        # Update positions
        self.portfolio_state.positions = {pos.position_id: pos for pos in positions}
        
        # Update orders
        if orders:
            self.portfolio_state.open_orders = orders
        else:
            self.portfolio_state.open_orders = []
        
        # Reset daily stats if new day
        current_date = datetime.now().date()
        if current_date > self.portfolio_state.last_reset.date():
            self._reset_daily_stats()
            self.portfolio_state.last_reset = datetime.now()
    
    def validate_signal(self, signal: StrategySignal, market_data: Dict[str, MarketData]) -> Tuple[bool, str, Dict]:
        """
        Validate a trading signal against risk management rules.
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            Tuple of (is_valid, reason, risk_metadata)
        """
        try:
            # Check basic signal validity
            if not self._validate_signal_basics(signal):
                return False, "Invalid signal parameters", {}
            
            # Check portfolio-level limits
            portfolio_check = self._validate_portfolio_limits(signal)
            if not portfolio_check[0]:
                return False, portfolio_check[1], {}
            
            # Check correlation limits
            correlation_check = self._validate_correlation_limits(signal)
            if not correlation_check[0]:
                return False, correlation_check[1], {}
            
            # Check volatility limits
            volatility_check = self._validate_volatility_limits(signal, market_data)
            if not volatility_check[0]:
                return False, volatility_check[1], {}
            
            # Check strategy-specific limits
            strategy_check = self._validate_strategy_limits(signal)
            if not strategy_check[0]:
                return False, strategy_check[1], {}
            
            # Calculate risk metadata
            risk_metadata = self._calculate_risk_metadata(signal, market_data)
            
            return True, "Signal approved", risk_metadata
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    def calculate_position_size(self, signal: StrategySignal, account_equity: float, 
                               market_data: Dict[str, MarketData]) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            signal: Trading signal
            account_equity: Account equity
            market_data: Market data for volatility calculation
            
        Returns:
            Position size in lots
        """
        try:
            # Get base position size from signal
            base_size = signal.volume
            
            # Apply Kelly criterion if enabled
            if self.risk_limits.position_sizing_method == "kelly":
                kelly_size = self._calculate_kelly_position_size(signal, account_equity)
                base_size = min(base_size, kelly_size)
            
            # Apply volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(signal.symbol, market_data)
            base_size *= volatility_adjustment
            
            # Apply correlation adjustment
            correlation_adjustment = self._calculate_correlation_adjustment(signal.symbol)
            base_size *= correlation_adjustment
            
            # Apply strategy performance adjustment
            strategy_adjustment = self._calculate_strategy_adjustment(signal.strategy_id)
            base_size *= strategy_adjustment
            
            # Apply risk limits
            max_size = account_equity * self.risk_limits.max_position_size
            base_size = min(base_size, max_size)
            
            # Ensure minimum size
            base_size = max(base_size, 0.01)  # Minimum 0.01 lots
            
            logger.debug(f"Calculated position size: {base_size:.4f} lots for {signal.symbol}")
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum size on error
    
    def update_risk_metrics(self, positions: List[Position], market_data: Dict[str, MarketData]):
        """
        Update comprehensive risk metrics.
        
        Args:
            positions: Current positions
            market_data: Market data for calculations
        """
        try:
            # Calculate basic metrics
            self._calculate_exposure_metrics(positions)
            self._calculate_performance_metrics()
            self._calculate_correlation_metrics(positions, market_data)
            self._calculate_volatility_metrics(market_data)
            self._calculate_var_metrics()
            
            # Update timestamp
            self.risk_metrics.last_updated = datetime.now()
            
            logger.debug("Risk metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'portfolio_state': {
                'total_equity': self.portfolio_state.total_equity,
                'total_margin': self.portfolio_state.total_margin,
                'free_margin': self.portfolio_state.free_margin,
                'margin_level': self.portfolio_state.margin_level,
                'position_count': len(self.portfolio_state.positions),
                'daily_pnl': self.portfolio_state.daily_pnl
            },
            'risk_metrics': {
                'total_exposure': f"{self.risk_metrics.total_exposure:.2%}",
                'leverage_ratio': f"{self.risk_metrics.leverage_ratio:.2f}",
                'correlation_score': f"{self.risk_metrics.correlation_score:.2f}",
                'var_95': f"{self.risk_metrics.var_95:.2%}",
                'max_drawdown': f"{self.risk_metrics.max_drawdown:.2%}",
                'sharpe_ratio': f"{self.risk_metrics.sharpe_ratio:.2f}",
                'volatility': f"{self.risk_metrics.volatility:.2%}",
                'win_rate': f"{self.risk_metrics.win_rate:.2%}"
            },
            'risk_limits': {
                'max_exposure': f"{self.risk_limits.max_total_exposure:.2%}",
                'max_leverage': f"{self.risk_limits.max_leverage:.1f}",
                'max_drawdown': f"{self.risk_limits.max_drawdown:.2%}",
                'max_var': f"{self.risk_limits.max_var_95:.2%}"
            },
            'alerts': self._generate_risk_alerts()
        }
    
    def _validate_signal_basics(self, signal: StrategySignal) -> bool:
        """Validate basic signal parameters."""
        if not signal or not signal.symbol:
            return False
        
        if signal.confidence < 0.1 or signal.confidence > 1.0:
            return False
        
        if signal.action not in ['buy', 'sell', 'hold']:
            return False
        
        if signal.volume <= 0:
            return False
        
        return True
    
    def _validate_portfolio_limits(self, signal: StrategySignal) -> Tuple[bool, str]:
        """Validate portfolio-level risk limits."""
        # Check total exposure
        if self.risk_metrics.total_exposure >= self.risk_limits.max_total_exposure:
            return False, "Maximum total exposure exceeded"
        
        # Check leverage
        if self.risk_metrics.leverage_ratio >= self.risk_limits.max_leverage:
            return False, "Maximum leverage exceeded"
        
        # Check daily loss limit
        if self.portfolio_state.daily_pnl <= -self.portfolio_state.total_equity * self.risk_limits.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        # Check drawdown
        if self.risk_metrics.max_drawdown >= self.risk_limits.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        return True, "Portfolio limits OK"
    
    def _validate_correlation_limits(self, signal: StrategySignal) -> Tuple[bool, str]:
        """Validate correlation limits."""
        if self.risk_metrics.correlation_score >= self.risk_limits.max_correlation:
            return False, "Portfolio correlation too high"
        
        return True, "Correlation limits OK"
    
    def _validate_volatility_limits(self, signal: StrategySignal, market_data: Dict[str, MarketData]) -> Tuple[bool, str]:
        """Validate volatility limits."""
        if signal.symbol in market_data:
            # Calculate symbol volatility
            symbol_volatility = self._calculate_symbol_volatility(signal.symbol, market_data)
            
            if symbol_volatility >= self.risk_limits.max_volatility:
                return False, f"Symbol volatility too high: {symbol_volatility:.2%}"
        
        return True, "Volatility limits OK"
    
    def _validate_strategy_limits(self, signal: StrategySignal) -> Tuple[bool, str]:
        """Validate strategy-specific limits."""
        # Get strategy performance
        strategy_perf = self.strategy_manager.get_strategy_performance(signal.strategy_id)
        
        if strategy_perf:
            # Check consecutive losses
            if strategy_perf.losing_trades >= self.risk_limits.max_consecutive_losses:
                return False, "Strategy consecutive loss limit exceeded"
            
            # Check win rate
            if strategy_perf.win_rate < 0.3:  # Minimum 30% win rate
                return False, "Strategy win rate too low"
        
        return True, "Strategy limits OK"
    
    def _calculate_risk_metadata(self, signal: StrategySignal, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Calculate risk metadata for the signal."""
        return {
            'position_size': signal.volume,
            'confidence': signal.confidence,
            'risk_reward_ratio': self._calculate_risk_reward_ratio(signal),
            'volatility': self._calculate_symbol_volatility(signal.symbol, market_data),
            'correlation_impact': self._calculate_correlation_impact(signal.symbol),
            'strategy_performance': self._get_strategy_performance_metrics(signal.strategy_id)
        }
    
    def _calculate_kelly_position_size(self, signal: StrategySignal, account_equity: float) -> float:
        """Calculate position size using Kelly criterion."""
        strategy_perf = self.strategy_manager.get_strategy_performance(signal.strategy_id)
        
        if not strategy_perf or strategy_perf.total_trades < 10:
            return 0.01  # Default to minimum size
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        win_rate = strategy_perf.win_rate
        avg_win = strategy_perf.avg_win
        avg_loss = abs(strategy_perf.avg_loss)
        
        if avg_loss == 0:
            return 0.01
        
        # Calculate odds received (b)
        b = avg_win / avg_loss
        
        # Kelly fraction
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        
        # Apply conservative Kelly (half Kelly)
        conservative_kelly = kelly_fraction * 0.5
        
        # Convert to position size
        position_size = max(0.01, conservative_kelly * account_equity * 0.01)
        
        return position_size
    
    def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict[str, MarketData]) -> float:
        """Calculate volatility-based position size adjustment."""
        if symbol not in market_data:
            return 1.0
        
        # Get historical volatility
        volatility = self._calculate_symbol_volatility(symbol, market_data)
        
        # Adjust position size inversely to volatility
        if volatility > 0:
            adjustment = min(2.0, max(0.5, 0.15 / volatility))  # Cap between 0.5x and 2x
            return adjustment
        
        return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based position size adjustment."""
        if self.risk_metrics.correlation_score > 0.5:
            # Reduce position size for high correlation
            adjustment = 1.0 - (self.risk_metrics.correlation_score - 0.5)
            return max(0.5, adjustment)
        
        return 1.0
    
    def _calculate_strategy_adjustment(self, strategy_id: str) -> float:
        """Calculate strategy performance-based position size adjustment."""
        strategy_perf = self.strategy_manager.get_strategy_performance(strategy_id)
        
        if not strategy_perf:
            return 0.5  # Conservative adjustment for unknown strategy
        
        # Adjust based on win rate and profit factor
        win_rate_adjustment = min(1.5, max(0.5, strategy_perf.win_rate * 2))
        profit_factor_adjustment = min(1.5, max(0.5, strategy_perf.profit_factor / 2))
        
        return (win_rate_adjustment + profit_factor_adjustment) / 2
    
    def _calculate_exposure_metrics(self, positions: List[Position]):
        """Calculate exposure and leverage metrics."""
        total_exposure = sum(abs(pos.volume * pos.entry_price) for pos in positions)
        self.risk_metrics.total_exposure = total_exposure / self.portfolio_state.total_equity if self.portfolio_state.total_equity > 0 else 0
        
        self.risk_metrics.leverage_ratio = self.portfolio_state.total_margin / self.portfolio_state.total_equity if self.portfolio_state.total_equity > 0 else 0
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        if len(self.pnl_history) < 2:
            return
        
        returns = np.diff(self.pnl_history)
        
        if len(returns) > 0:
            self.risk_metrics.volatility = np.std(returns) * np.sqrt(252)
            self.risk_metrics.sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                self.risk_metrics.sortino_ratio = (np.mean(returns) * 252) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_correlation_metrics(self, positions: List[Position], market_data: Dict[str, MarketData]):
        """Calculate correlation metrics."""
        if len(positions) < 2:
            self.risk_metrics.correlation_score = 0.0
            return
        
        # Calculate correlation between positions
        position_returns = []
        for pos in positions:
            if pos.symbol_id in market_data:
                # Calculate position return
                current_price = market_data[pos.symbol_id].bid if pos.side.value == 'buy' else market_data[pos.symbol_id].ask
                position_return = (current_price - pos.entry_price) / pos.entry_price
                position_returns.append(position_return)
        
        if len(position_returns) >= 2:
            correlation_matrix = np.corrcoef(position_returns)
            # Average correlation (excluding diagonal)
            avg_correlation = (np.sum(correlation_matrix) - len(correlation_matrix)) / (len(correlation_matrix) ** 2 - len(correlation_matrix))
            self.risk_metrics.correlation_score = float(abs(avg_correlation))
    
    def _calculate_volatility_metrics(self, market_data: Dict[str, MarketData]):
        """Calculate volatility metrics."""
        # This would use historical data in a real implementation
        # For now, use a simplified calculation
        self.risk_metrics.volatility = 0.15  # 15% annualized volatility
    
    def _calculate_var_metrics(self):
        """Calculate Value at Risk metrics."""
        if len(self.pnl_history) < 10:
            self.risk_metrics.var_95 = 0.02  # Default 2% VaR
            return
        
        returns = np.diff(self.pnl_history)
        self.risk_metrics.var_95 = float(np.percentile(returns, 5))  # 5th percentile
    
    def _calculate_symbol_volatility(self, symbol: str, market_data: Dict[str, MarketData]) -> float:
        """Calculate symbol volatility."""
        # This would use historical data in a real implementation
        # For now, return a default value
        return 0.15  # 15% annualized volatility
    
    def _calculate_risk_reward_ratio(self, signal: StrategySignal) -> float:
        """Calculate risk-reward ratio for the signal."""
        if signal.stop_loss and signal.take_profit and signal.entry_price:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            return reward / risk if risk > 0 else 0
        
        return 1.5  # Default ratio
    
    def _calculate_correlation_impact(self, symbol: str) -> float:
        """Calculate correlation impact of adding this symbol."""
        # This would calculate the impact on portfolio correlation
        # For now, return a default value
        return 0.1  # 10% correlation impact
    
    def _get_strategy_performance_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        strategy_perf = self.strategy_manager.get_strategy_performance(strategy_id)
        
        if strategy_perf:
            return {
                'win_rate': strategy_perf.win_rate,
                'profit_factor': strategy_perf.profit_factor,
                'total_trades': strategy_perf.total_trades,
                'avg_win': strategy_perf.avg_win,
                'avg_loss': strategy_perf.avg_loss
            }
        
        return {}
    
    def _generate_risk_alerts(self) -> List[str]:
        """Generate risk alerts based on current metrics."""
        alerts = []
        
        if self.risk_metrics.total_exposure > self.risk_limits.max_total_exposure * 0.8:
            alerts.append(f"High exposure warning: {self.risk_metrics.total_exposure:.2%}")
        
        if self.risk_metrics.leverage_ratio > self.risk_limits.max_leverage * 0.8:
            alerts.append(f"High leverage warning: {self.risk_metrics.leverage_ratio:.2f}")
        
        if self.risk_metrics.correlation_score > self.risk_limits.max_correlation * 0.8:
            alerts.append(f"High correlation warning: {self.risk_metrics.correlation_score:.2f}")
        
        if self.risk_metrics.var_95 > self.risk_limits.max_var_95 * 0.8:
            alerts.append(f"High VaR warning: {self.risk_metrics.var_95:.2%}")
        
        if self.portfolio_state.daily_pnl < -self.portfolio_state.total_equity * self.risk_limits.max_daily_loss * 0.8:
            alerts.append(f"Daily loss warning: {self.portfolio_state.daily_pnl:.2f}")
        
        return alerts
    
    def _reset_daily_stats(self):
        """Reset daily statistics."""
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'max_drawdown': 0.0
        }
        self.portfolio_state.daily_pnl = 0.0
        self.portfolio_state.daily_trades = 0


def main():
    """Test the advanced risk manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test advanced risk manager")
    parser.add_argument("--test-validation", action="store_true", help="Test signal validation")
    parser.add_argument("--test-position-sizing", action="store_true", help="Test position sizing")
    parser.add_argument("--test-risk-metrics", action="store_true", help="Test risk metrics calculation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test components
    risk_limits = RiskLimits()
    strategy_manager = StrategyManager()
    
    # Create risk manager
    risk_manager = AdvancedRiskManager(risk_limits, strategy_manager)
    
    if args.test_validation:
        print("Testing signal validation...")
        
        # Create test signal
        from src.trading.strategy_manager import StrategySignal
        
        test_signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            action="buy",
            confidence=0.7,
            entry_price=1.1000,
            stop_loss=1.0980,
            take_profit=1.1040,
            volume=0.01
        )
        
        # Test validation
        is_valid, reason, metadata = risk_manager.validate_signal(test_signal, {})
        print(f"Signal valid: {is_valid}")
        print(f"Reason: {reason}")
        print(f"Metadata: {metadata}")
    
    if args.test_position_sizing:
        print("Testing position sizing...")
        
        # Create test signal
        test_signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="EURUSD",
            action="buy",
            confidence=0.7,
            entry_price=1.1000,
            stop_loss=1.0980,
            take_profit=1.1040,
            volume=0.01
        )
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(test_signal, 10000.0, {})
        print(f"Calculated position size: {position_size:.4f} lots")
    
    if args.test_risk_metrics:
        print("Testing risk metrics calculation...")
        
        # Create test positions
        from src.trading.mock_ctrader_interface import Position, OrderSide
        
        test_positions = [
            Position(
                position_id="pos1",
                symbol_id=1,
                side=OrderSide.BUY,
                volume=0.01,
                entry_price=1.1000,
                current_price=1.1010,
                profit_loss=10.0
            )
        ]
        
        # Update risk metrics
        risk_manager.update_risk_metrics(test_positions, {})
        
        # Get risk report
        report = risk_manager.get_risk_report()
        print("Risk Report:")
        print(f"Total Exposure: {report['risk_metrics']['total_exposure']}")
        print(f"Leverage Ratio: {report['risk_metrics']['leverage_ratio']}")
        print(f"Correlation Score: {report['risk_metrics']['correlation_score']}")
        print(f"VaR (95%): {report['risk_metrics']['var_95']}")
        print(f"Alerts: {report['alerts']}")


if __name__ == "__main__":
    main() 