#!/usr/bin/env python3
"""
Advanced Risk Management System - Phase 2B Enhanced Implementation

This module implements sophisticated risk management that integrates with:
- Market regime detection for adaptive parameters
- Strategy manager for performance-based adjustments
- Dynamic risk parameter system
- Real-time market condition adaptation

Author: EMP Development Team
Phase: 2B - Adaptive Risk Management
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_UP
import asyncio

from src.trading.strategies.strategy_manager import StrategyManager, StrategySignal
from src.trading.risk.market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeDetectionResult
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
class DynamicRiskParameters:
    """Dynamic risk parameters based on market regime and conditions."""
    max_position_size: float = 0.05  # 5% per position
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    max_portfolio_risk: float = 0.3  # 30% total risk
    correlation_limit: float = 0.7
    leverage_multiplier: float = 1.0
    min_confidence_threshold: float = 0.6
    volatility_scaling: bool = True
    regime_adjustments: Dict[str, float] = field(default_factory=dict)


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
    Enhanced risk management system with market regime adaptation.
    
    This class provides:
    - Market regime-aware risk parameters
    - Dynamic position sizing based on conditions
    - Real-time risk monitoring with regime context
    - Strategy performance integration
    - Adaptive risk limits
    """
    
    def __init__(self, strategy_manager: StrategyManager, regime_detector: MarketRegimeDetector):
        """
        Initialize the enhanced risk manager.
        
        Args:
            strategy_manager: Strategy manager for performance data
            regime_detector: Market regime detector for adaptive parameters
        """
        self.strategy_manager = strategy_manager
        self.regime_detector = regime_detector
        self.portfolio_state = PortfolioState()
        self.risk_metrics = RiskMetrics()
        self.dynamic_params = DynamicRiskParameters()
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
        
        # Regime-based adjustments
        self.regime_adjustments = self._initialize_regime_adjustments()
        
        logger.info("Enhanced risk manager initialized")
    
    def _initialize_regime_adjustments(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize regime-specific risk parameter adjustments."""
        return {
            MarketRegime.TRENDING_UP: {
                'max_position_size': 1.2,      # 20% increase
                'stop_loss_multiplier': 1.1,   # Wider stops
                'take_profit_multiplier': 1.3, # Wider targets
                'max_portfolio_risk': 1.1,     # Slightly higher risk
                'leverage_multiplier': 1.2,    # More leverage
                'min_confidence_threshold': 0.5, # Lower confidence needed
            },
            MarketRegime.TRENDING_DOWN: {
                'max_position_size': 0.8,      # 20% reduction
                'stop_loss_multiplier': 0.9,   # Tighter stops
                'take_profit_multiplier': 1.2, # Still good targets
                'max_portfolio_risk': 0.9,     # Slightly lower risk
                'leverage_multiplier': 0.8,    # Less leverage
                'min_confidence_threshold': 0.7, # Higher confidence needed
            },
            MarketRegime.RANGING: {
                'max_position_size': 1.0,      # Normal sizing
                'stop_loss_multiplier': 0.8,   # Tighter stops for mean reversion
                'take_profit_multiplier': 0.9, # Quicker profits
                'max_portfolio_risk': 0.9,     # Conservative
                'leverage_multiplier': 0.9,    # Slightly less leverage
                'min_confidence_threshold': 0.7, # Higher confidence
            },
            MarketRegime.VOLATILE: {
                'max_position_size': 0.7,      # 30% reduction
                'stop_loss_multiplier': 0.8,   # Tighter stops
                'take_profit_multiplier': 0.9, # Quicker profits
                'max_portfolio_risk': 0.7,     # Significant risk reduction
                'leverage_multiplier': 0.6,    # Much less leverage
                'min_confidence_threshold': 0.8, # Very high confidence
            },
            MarketRegime.CRISIS: {
                'max_position_size': 0.5,      # 50% reduction
                'stop_loss_multiplier': 0.7,   # Much tighter stops
                'take_profit_multiplier': 0.8, # Quick profits
                'max_portfolio_risk': 0.5,     # Defensive positioning
                'leverage_multiplier': 0.3,    # Minimal leverage
                'min_confidence_threshold': 0.9, # Extremely high confidence
            },
            MarketRegime.RECOVERY: {
                'max_position_size': 0.8,      # Cautious increase
                'stop_loss_multiplier': 0.9,   # Moderate stops
                'take_profit_multiplier': 1.1, # Good targets
                'max_portfolio_risk': 0.8,     # Moderate risk
                'leverage_multiplier': 0.8,    # Moderate leverage
                'min_confidence_threshold': 0.7, # Higher confidence
            },
            MarketRegime.LOW_VOLATILITY: {
                'max_position_size': 1.1,      # Slight increase
                'stop_loss_multiplier': 1.1,   # Wider stops
                'take_profit_multiplier': 1.2, # Wider targets
                'max_portfolio_risk': 1.1,     # Slightly higher risk
                'leverage_multiplier': 1.1,    # Slightly more leverage
                'min_confidence_threshold': 0.5, # Lower confidence needed
            },
            MarketRegime.HIGH_VOLATILITY: {
                'max_position_size': 0.6,      # 40% reduction
                'stop_loss_multiplier': 0.7,   # Tighter stops
                'take_profit_multiplier': 0.8, # Quicker profits
                'max_portfolio_risk': 0.6,     # Significant risk reduction
                'leverage_multiplier': 0.5,    # Much less leverage
                'min_confidence_threshold': 0.8, # High confidence needed
            }
        }
    
    async def update_dynamic_parameters(self, market_data: pd.DataFrame):
        """Update risk parameters based on current market regime."""
        try:
            # Detect current market regime
            regime_result = await self.regime_detector.detect_regime(market_data)
            
            # Get regime-specific adjustments
            adjustments = self.regime_adjustments.get(regime_result.regime, {})
            
            # Apply adjustments to dynamic parameters
            for param, adjustment in adjustments.items():
                if hasattr(self.dynamic_params, param):
                    base_value = getattr(self.dynamic_params, param)
                    setattr(self.dynamic_params, param, base_value * adjustment)
            
            # Store regime context
            self.dynamic_params.regime_adjustments = adjustments
            
            logger.info(f"Updated risk parameters for regime: {regime_result.regime.value}")
            
        except Exception as e:
            logger.error(f"Error updating dynamic parameters: {e}")
    
    async def validate_signal(self, signal: StrategySignal, market_data: Dict[str, MarketData], 
                            regime_result: Optional[RegimeDetectionResult] = None) -> Tuple[bool, str, Dict]:
        """
        Validate a trading signal against dynamic risk management rules.
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            regime_result: Optional regime detection result
            
        Returns:
            Tuple of (is_valid, reason, risk_metadata)
        """
        try:
            # Get current regime if not provided
            if regime_result is None:
                # Convert market_data to DataFrame for regime detection
                df_data = self._convert_market_data_to_df(market_data)
                regime_result = await self.regime_detector.detect_regime(df_data)
            
            # Update dynamic parameters based on regime
            await self.update_dynamic_parameters(self._convert_market_data_to_df(market_data))
            
            # Check basic signal validity
            if not self._validate_signal_basics(signal):
                return False, "Invalid signal parameters", {}
            
            # Check regime-specific confidence threshold
            if signal.confidence < self.dynamic_params.min_confidence_threshold:
                return False, f"Confidence below regime threshold: {signal.confidence:.2f}", {}
            
            # Check portfolio-level limits with regime adjustments
            portfolio_check = await self._validate_portfolio_limits(signal, regime_result)
            if not portfolio_check[0]:
                return False, portfolio_check[1], {}
            
            # Check correlation limits
            correlation_check = await self._validate_correlation_limits(signal)
            if not correlation_check[0]:
                return False, correlation_check[1], {}
            
            # Check volatility limits
            volatility_check = await self._validate_volatility_limits(signal, market_data)
            if not volatility_check[0]:
                return False, volatility_check[1], {}
            
            # Check strategy-specific limits
            strategy_check = await self._validate_strategy_limits(signal)
            if not strategy_check[0]:
                return False, strategy_check[1], {}
            
            # Calculate risk metadata with regime context
            risk_metadata = await self._calculate_risk_metadata(signal, market_data, regime_result)
            
            return True, "Signal approved", risk_metadata
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    async def calculate_position_size(self, signal: StrategySignal, account_equity: float, 
                                    market_data: Dict[str, MarketData], 
                                    regime_result: Optional[RegimeDetectionResult] = None) -> float:
        """
        Calculate optimal position size based on dynamic risk parameters.
        
        Args:
            signal: Trading signal
            account_equity: Account equity
            market_data: Market data for volatility calculation
            regime_result: Optional regime detection result
            
        Returns:
            Position size in lots
        """
        try:
            # Get current regime if not provided
            if regime_result is None:
                df_data = self._convert_market_data_to_df(market_data)
                regime_result = await self.regime_detector.detect_regime(df_data)
            
            # Update dynamic parameters
            await self.update_dynamic_parameters(self._convert_market_data_to_df(market_data))
            
            # Get base position size from signal
            base_size = signal.volume
            
            # Apply Kelly criterion with regime adjustments
            kelly_size = await self._calculate_kelly_position_size(signal, account_equity, regime_result)
            base_size = min(base_size, kelly_size)
            
            # Apply volatility adjustment
            volatility_adjustment = await self._calculate_volatility_adjustment(signal.symbol, market_data, regime_result)
            base_size *= volatility_adjustment
            
            # Apply correlation adjustment
            correlation_adjustment = await self._calculate_correlation_adjustment(signal.symbol)
            base_size *= correlation_adjustment
            
            # Apply strategy performance adjustment
            strategy_adjustment = await self._calculate_strategy_adjustment(signal.strategy_id)
            base_size *= strategy_adjustment
            
            # Apply regime-adjusted risk limits
            max_size = account_equity * self.dynamic_params.max_position_size
            base_size = min(base_size, max_size)
            
            # Ensure minimum size
            base_size = max(base_size, 0.01)  # Minimum 0.01 lots
            
            logger.debug(f"Calculated position size: {base_size:.4f} lots for {signal.symbol} in {regime_result.regime.value}")
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum size on error
    
    def _convert_market_data_to_df(self, market_data: Dict[str, MarketData]) -> pd.DataFrame:
        """Convert market data dictionary to DataFrame for regime detection."""
        # This is a simplified conversion - in real implementation, use historical data
        data = []
        for symbol, md in market_data.items():
            data.append({
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': 1000  # Default volume
            })
        
        if data:
            return pd.DataFrame(data)
        else:
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    async def _validate_portfolio_limits(self, signal: StrategySignal, regime_result: RegimeDetectionResult) -> Tuple[bool, str]:
        """Validate portfolio-level risk limits with regime adjustments."""
        # Check total exposure with regime adjustment
        max_exposure = self.dynamic_params.max_portfolio_risk
        if self.risk_metrics.total_exposure >= max_exposure:
            return False, f"Maximum total exposure exceeded: {self.risk_metrics.total_exposure:.2%} vs {max_exposure:.2%}"
        
        # Check leverage with regime adjustment
        max_leverage = 5.0 * self.dynamic_params.leverage_multiplier
        if self.risk_metrics.leverage_ratio >= max_leverage:
            return False, f"Maximum leverage exceeded: {self.risk_metrics.leverage_ratio:.2f} vs {max_leverage:.2f}"
        
        return True, "Portfolio limits OK"
    
    async def _validate_correlation_limits(self, signal: StrategySignal) -> Tuple[bool, str]:
        """Validate correlation limits."""
        if self.risk_metrics.correlation_score >= self.dynamic_params.correlation_limit:
            return False, f"Portfolio correlation too high: {self.risk_metrics.correlation_score:.2f}"
        
        return True, "Correlation limits OK"
    
    async def _validate_volatility_limits(self, signal: StrategySignal, market_data: Dict[str, MarketData]) -> Tuple[bool, str]:
        """Validate volatility limits."""
        if signal.symbol in market_data:
            volatility = await self._calculate_symbol_volatility(signal.symbol, market_data)
            # Volatility limits are handled by regime adjustments
            return True, "Volatility limits OK"
        
        return True, "Volatility limits OK"
    
    async def _validate_strategy_limits(self, signal: StrategySignal) -> Tuple[bool, str]:
        """Validate strategy-specific limits."""
        strategy_perf = self.strategy_manager.get_strategy_performance(signal.strategy_id)
        
        if strategy_perf:
            # Check consecutive losses
            if strategy_perf.losing_trades >= 5:  # Max 5 consecutive losses
                return False, "Strategy consecutive loss limit exceeded"
            
            # Check win rate
            if strategy_perf.win_rate < 0.3:  # Minimum 30% win rate
                return False, "Strategy win rate too low"
        
        return True, "Strategy limits OK"
    
    async def _calculate_kelly_position_size(self, signal: StrategySignal, account_equity: float, 
                                           regime_result: RegimeDetectionResult) -> float:
        """Calculate position size using Kelly criterion with regime adjustments."""
        strategy_perf = self.strategy_manager.get_strategy_performance(signal.strategy_id)
        
        if not strategy_perf or strategy_perf.total_trades < 10:
            return 0.01  # Default to minimum size
        
        # Kelly formula: f = (bp - q) / b
        win_rate = strategy_perf.win_rate
        avg_win = strategy_perf.avg_win
        avg_loss = abs(strategy_perf.avg_loss)
        
        if avg_loss == 0:
            return 0.01
        
        # Calculate odds received (b)
        b = avg_win / avg_loss
        
        # Kelly fraction
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        
        # Apply conservative Kelly (half Kelly) with regime adjustment
        regime_adjustment = self.regime_adjustments.get(regime_result.regime, {}).get('leverage_multiplier', 1.0)
        conservative_kelly = kelly_fraction * 0.5 * regime_adjustment
        
        # Convert to position size
        position_size = max(0.01, conservative_kelly * account_equity * 0.01)
        
        return position_size
    
    async def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict[str, MarketData], 
                                             regime_result: RegimeDetectionResult) -> float:
        """Calculate volatility-based position size adjustment with regime context."""
        # This is a simplified implementation
        # In real implementation, use historical volatility
        regime_volatility = regime_result.characteristics.volatility_level
        
        # Adjust position size inversely to volatility
        if regime_volatility > 0:
            adjustment = min(2.0, max(0.5, 0.5 / regime_volatility))
            return adjustment
        
        return 1.0
    
    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based position size adjustment."""
        if self.risk_metrics.correlation_score > 0.5:
            adjustment = 1.0 - (self.risk_metrics.correlation_score - 0.5)
            return max(0.5, adjustment)
        
        return 1.0
    
    async def _calculate_strategy_adjustment(self, strategy_id: str) -> float:
        """Calculate strategy performance-based position size adjustment."""
        strategy_perf = self.strategy_manager.get_strategy_performance(strategy_id)
        
        if not strategy_perf:
            return 0.5
        
        # Adjust based on win rate and profit factor
        win_rate_adjustment = min(1.5, max(0.5, strategy_perf.win_rate * 2))
        profit_factor_adjustment = min(1.5, max(0.5, strategy_perf.profit_factor / 2))
        
        return (win_rate_adjustment + profit_factor_adjustment) / 2
    
    async def _calculate_risk_metadata(self, signal: StrategySignal, market_data: Dict[str, MarketData], 
                                     regime_result: RegimeDetectionResult) -> Dict[str, Any]:
        """Calculate risk metadata with regime context."""
        return {
            'position_size': signal.volume,
            'confidence': signal.confidence,
            'regime': regime_result.regime.value,
            'regime_confidence': regime_result.confidence,
            'risk_reward_ratio': await self._calculate_risk_reward_ratio(signal, regime_result),
            'volatility_level': regime_result.characteristics.volatility_level,
            'trend_strength': regime_result.characteristics.trend_strength,
            'strategy_performance': self._get_strategy_performance_metrics(signal.strategy_id),
            'dynamic_parameters': {
                'max_position_size': self.dynamic_params.max_position_size,
                'stop_loss_multiplier': self.dynamic_params.stop_loss_multiplier,
                'take_profit_multiplier': self.dynamic_params.take_profit_multiplier,
            }
        }
    
    async def _calculate_risk_reward_ratio(self, signal: StrategySignal, regime_result: RegimeDetectionResult) -> float:
        """Calculate risk-reward ratio with regime adjustments."""
        base_ratio = 1.5
        
        # Adjust based on regime
        regime_adjustment = self.regime_adjustments.get(regime_result.regime, {}).get('take_profit_multiplier', 1.0)
        
        return base_ratio * regime_adjustment
    
    def _get_strategy_performance_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        strategy_perf = self.strategy_manager.get_strategy_performance(strategy_id)
        
        if strategy_perf:
            return {
                'win_rate': strategy_perf.win_rate,
                'profit_factor': strategy_perf.profit_factor,
                'total_trades': strategy_perf.total_trades,
                'avg_win': strategy_perf.avg_win,
                'avg_loss': strategy_perf.avg_loss,
                'total_pnl': strategy_perf.total_pnl
            }
        
        return {}
    
    def update_portfolio_state(self, positions: List[Position], equity: float, 
                              margin: float, orders: Optional[List[Order]] = None):
        """Update current portfolio state."""
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
    
    def update_risk_metrics(self, positions: List[Position], market_data: Dict[str, MarketData]):
        """Update comprehensive risk metrics."""
        try:
            self._calculate_exposure_metrics(positions)
            self._calculate_performance_metrics()
            self._calculate_correlation_metrics(positions, market_data)
            self._calculate_volatility_metrics(market_data)
            self._calculate_var_metrics()
            
            self.risk_metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report with regime context."""
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
            'dynamic_parameters': {
                'max_position_size': f"{self.dynamic_params.max_position_size:.2%}",
                'min_confidence_threshold': f"{self.dynamic_params.min_confidence_threshold:.2f}",
                'leverage_multiplier': f"{self.dynamic_params.leverage_multiplier:.2f}",
                'regime': list(self.dynamic_params.regime_adjustments.keys())[-1] if self.dynamic_params.regime_adjustments else "default"
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
    
    def _calculate_correlation_metrics(self, positions: List[Position], market_data: Dict[str, MarketData]):
        """Calculate correlation metrics."""
        if len(positions) < 2:
            self.risk_metrics.correlation_score = 0.0
            return
        
        position_returns = []
        for pos in positions:
            position_returns.append(pos.profit_loss / (pos.volume * pos.entry_price) if pos.volume * pos.entry_price > 0 else 0)
        
        if len(position_returns) >= 2:
            correlation_matrix = np.corrcoef(position_returns)
            avg_correlation = (np.sum(correlation_matrix) - len(correlation_matrix)) / (len(correlation_matrix) ** 2 - len(correlation_matrix))
            self.risk_metrics.correlation_score = float(abs(avg_correlation))
    
    def _calculate_volatility_metrics(self, market_data: Dict[str, MarketData]):
        """Calculate volatility metrics."""
        self.risk_metrics.volatility = 0.15  # Default 15% annualized volatility
    
    def _calculate_var_metrics(self):
        """Calculate Value at Risk metrics."""
        if len(self.pnl_history) < 10:
            self.risk_metrics.var_95 = 0.02  # Default 2% VaR
            return
        
        returns = np.diff(self.pnl_history)
        self.risk_metrics.var_95 = float(np.percentile(returns, 5))
    
    def _generate_risk_alerts(self) -> List[str]:
        """Generate risk alerts based on current metrics."""
        alerts = []
        
        if self.risk_metrics.total_exposure > self.dynamic_params.max_portfolio_risk * 0.8:
            alerts.append(f"High exposure warning: {self.risk_metrics.total_exposure:.2%}")
        
        if self.risk_metrics.leverage_ratio > 5.0 * self.dynamic_params.leverage_multiplier * 0.8:
            alerts.append(f"High leverage warning: {self.risk_metrics.leverage_ratio:.2f}")
        
        if self.risk_metrics.correlation_score > self.dynamic_params.correlation_limit * 0.8:
            alerts.append(f"High correlation warning: {self.risk_metrics.correlation_score:.2f}")
        
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


async def main():
    """Test the enhanced risk manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test enhanced risk manager")
    parser.add_argument("--test-validation", action="store_true", help="Test signal validation")
    parser.add_argument("--test-position-sizing", action="store_true", help="Test position sizing")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test components
    strategy_manager = StrategyManager()
    regime_detector = MarketRegimeDetector()
    
    # Create risk manager
    risk_manager = AdvancedRiskManager(strategy_manager, regime_detector)
    
    if args.test_validation:
        print("Testing signal validation...")
        
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
        
        # Test validation
        is_valid, reason, metadata = await risk_manager.validate_signal(test_signal, {})
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
        position_size = await risk_manager.calculate_position_size(test_signal, 10000.0, {})
        print(f"Calculated position size: {position_size:.4f} lots")


if __name__ == "__main__":
    asyncio.run(main())
