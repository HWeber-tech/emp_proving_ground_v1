"""
Enhanced Risk Management System for EMP

This module implements comprehensive risk management including dynamic risk assessment,
portfolio risk management, advanced position sizing, drawdown protection, and
risk-adjusted returns calculation.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from src.sensory.core.base import MarketData
from src.trading.strategy_engine import StrategySignal, StrategyPerformance

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for position sizing"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskMetric(Enum):
    """Types of risk metrics"""
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    CALMAR = "calmar_ratio"


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: float
    take_profit: float
    strategy_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioRisk:
    """Portfolio risk metrics"""
    total_value: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_portfolio_value: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.15      # 15% max drawdown
    max_var: float = 0.03           # 3% VaR limit
    max_correlation: float = 0.7    # Maximum correlation between positions
    min_sharpe_ratio: float = 0.5   # Minimum Sharpe ratio
    max_leverage: float = 2.0       # Maximum leverage
    stop_loss_threshold: float = 0.02  # 2% stop loss
    take_profit_threshold: float = 0.04  # 4% take profit


class DynamicRiskAssessment:
    """Real-time risk assessment and calculation"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.risk_history: List[PortfolioRisk] = []
        self.market_data_history: Dict[str, List[MarketData]] = {}
        
        logger.info("DynamicRiskAssessment initialized")
    
    def calculate_portfolio_risk(self, positions: List[Position], 
                               market_data: Dict[str, List[MarketData]],
                               portfolio_value: float) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        
        risk = PortfolioRisk()
        risk.total_value = portfolio_value
        
        if not positions:
            return risk
        
        # Calculate position values and PnL
        position_values = []
        position_weights = []
        returns_history = []
        
        for position in positions:
            position_value = abs(position.quantity * position.current_price)
            position_values.append(position_value)
            position_weights.append(position_value / portfolio_value)
            risk.unrealized_pnl += position.unrealized_pnl
            risk.realized_pnl += position.realized_pnl
        
        risk.total_pnl = risk.unrealized_pnl + risk.realized_pnl
        
        # Calculate returns history for risk metrics
        if len(positions) > 1:
            returns_history = self._calculate_returns_history(positions, market_data)
        
        if returns_history:
            # Calculate risk metrics
            risk.volatility = np.std(returns_history) * np.sqrt(252)  # Annualized
            risk.var_95 = np.percentile(returns_history, 5)
            risk.var_99 = np.percentile(returns_history, 1)
            risk.cvar_95 = np.mean(returns_history[returns_history <= risk.var_95])
            risk.cvar_99 = np.mean(returns_history[returns_history <= risk.var_99])
            
            # Calculate Sharpe ratio
            if risk.volatility > 0:
                risk.sharpe_ratio = (np.mean(returns_history) * 252) / risk.volatility
            
            # Calculate Sortino ratio
            negative_returns = returns_history[returns_history < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    risk.sortino_ratio = (np.mean(returns_history) * 252) / downside_deviation
        
        # Calculate correlation matrix
        if len(positions) > 1:
            risk.correlation_matrix = self._calculate_correlation_matrix(positions, market_data)
            risk.covariance_matrix = self._calculate_covariance_matrix(positions, market_data)
        
        # Calculate drawdown
        risk.current_drawdown = self._calculate_current_drawdown(portfolio_value)
        risk.max_drawdown = max(risk.max_drawdown, risk.current_drawdown)
        
        # Store risk history
        self.risk_history.append(risk)
        if len(self.risk_history) > self.lookback_period:
            self.risk_history.pop(0)
        
        return risk
    
    def _calculate_returns_history(self, positions: List[Position], 
                                 market_data: Dict[str, List[MarketData]]) -> List[float]:
        """Calculate historical returns for portfolio"""
        returns = []
        
        for position in positions:
            if position.symbol in market_data and len(market_data[position.symbol]) > 1:
                prices = [md.close for md in market_data[position.symbol]]
                position_returns = np.diff(prices) / prices[:-1]
                returns.extend(position_returns)
        
        return returns if returns else [0.0]
    
    def _calculate_correlation_matrix(self, positions: List[Position], 
                                   market_data: Dict[str, List[MarketData]]) -> np.ndarray:
        """Calculate correlation matrix for positions"""
        returns_data = []
        
        for position in positions:
            if position.symbol in market_data and len(market_data[position.symbol]) > 1:
                prices = [md.close for md in market_data[position.symbol]]
                returns = np.diff(prices) / prices[:-1]
                returns_data.append(returns)
        
        if len(returns_data) > 1:
            # Pad shorter series with zeros
            max_length = max(len(r) for r in returns_data)
            padded_returns = []
            for returns in returns_data:
                padded = np.pad(returns, (0, max_length - len(returns)), 'constant')
                padded_returns.append(padded)
            
            return np.corrcoef(padded_returns)
        
        return np.array([[1.0]])
    
    def _calculate_covariance_matrix(self, positions: List[Position], 
                                   market_data: Dict[str, List[MarketData]]) -> np.ndarray:
        """Calculate covariance matrix for positions"""
        returns_data = []
        
        for position in positions:
            if position.symbol in market_data and len(market_data[position.symbol]) > 1:
                prices = [md.close for md in market_data[position.symbol]]
                returns = np.diff(prices) / prices[:-1]
                returns_data.append(returns)
        
        if len(returns_data) > 1:
            # Pad shorter series with zeros
            max_length = max(len(r) for r in returns_data)
            padded_returns = []
            for returns in returns_data:
                padded = np.pad(returns, (0, max_length - len(returns)), 'constant')
                padded_returns.append(padded)
            
            return np.cov(padded_returns)
        
        return np.array([[0.0]])
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown"""
        if not self.risk_history:
            return 0.0
        
        peak_value = max(risk.total_value for risk in self.risk_history)
        if peak_value > 0:
            return (peak_value - current_value) / peak_value
        
        return 0.0
    
    def get_risk_alert(self, risk: PortfolioRisk, limits: RiskLimits) -> List[str]:
        """Get risk alerts based on limits"""
        alerts = []
        
        if risk.current_drawdown > limits.max_drawdown:
            alerts.append(f"Drawdown limit exceeded: {risk.current_drawdown:.2%}")
        
        if abs(risk.var_95) > limits.max_var:
            alerts.append(f"VaR limit exceeded: {abs(risk.var_95):.2%}")
        
        if risk.sharpe_ratio < limits.min_sharpe_ratio:
            alerts.append(f"Sharpe ratio below limit: {risk.sharpe_ratio:.2f}")
        
        if risk.total_pnl < -limits.max_daily_loss * risk.total_value:
            alerts.append(f"Daily loss limit exceeded: {risk.total_pnl:.2f}")
        
        return alerts


class PositionSizingEngine:
    """Advanced position sizing algorithms"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.kelly_multiplier = self._get_kelly_multiplier()
        
        logger.info(f"PositionSizingEngine initialized with {risk_level.value} risk level")
    
    def _get_kelly_multiplier(self) -> float:
        """Get Kelly criterion multiplier based on risk level"""
        multipliers = {
            RiskLevel.CONSERVATIVE: 0.25,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.AGGRESSIVE: 0.75
        }
        return multipliers.get(self.risk_level, 0.5)
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate position size using Kelly criterion"""
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply risk level multiplier
        kelly_fraction *= self.kelly_multiplier
        
        # Ensure reasonable bounds
        return max(0.0, min(kelly_fraction, 0.25))
    
    def risk_parity_sizing(self, positions: List[Position], 
                          target_risk: float, portfolio_value: float) -> Dict[str, float]:
        """Calculate position sizes using risk parity approach"""
        if not positions:
            return {}
        
        # Calculate current risk contribution
        risk_contributions = []
        for position in positions:
            position_value = abs(position.quantity * position.current_price)
            risk_contrib = position_value / portfolio_value
            risk_contributions.append(risk_contrib)
        
        # Target equal risk contribution
        target_contribution = target_risk / len(positions)
        
        # Calculate new position sizes
        new_sizes = {}
        for i, position in enumerate(positions):
            current_contribution = risk_contributions[i]
            if current_contribution > 0:
                adjustment_factor = target_contribution / current_contribution
                new_size = position.quantity * adjustment_factor
                new_sizes[position.symbol] = new_size
        
        return new_sizes
    
    def volatility_based_sizing(self, signal: StrategySignal, 
                              volatility: float, portfolio_value: float,
                              risk_per_trade: float) -> float:
        """Calculate position size based on volatility"""
        if volatility <= 0:
            return 0.0
        
        # Volatility-adjusted position size
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / (signal.price * volatility)
        
        # Apply maximum position size limit
        max_size = portfolio_value * 0.1  # 10% max position size
        return min(position_size, max_size)
    
    def optimal_position_size(self, signal: StrategySignal, 
                            strategy_performance: StrategyPerformance,
                            portfolio_value: float, risk_limits: RiskLimits) -> float:
        """Calculate optimal position size using multiple methods"""
        
        # Kelly criterion
        kelly_size = self.kelly_criterion(
            strategy_performance.win_rate,
            strategy_performance.average_win,
            strategy_performance.average_loss
        )
        
        # Volatility-based sizing
        volatility = strategy_performance.volatility if hasattr(strategy_performance, 'volatility') else 0.02
        vol_size = self.volatility_based_sizing(signal, volatility, portfolio_value, risk_limits.stop_loss_threshold)
        
        # Risk-based sizing
        risk_size = portfolio_value * risk_limits.max_position_size
        
        # Combine methods (weighted average)
        optimal_size = (kelly_size * 0.4 + vol_size * 0.4 + risk_size * 0.2)
        
        # Apply limits
        optimal_size = min(optimal_size, risk_limits.max_position_size * portfolio_value)
        optimal_size = max(optimal_size, 0.0)
        
        return optimal_size


class DrawdownProtection:
    """Drawdown protection and risk controls"""
    
    def __init__(self, max_drawdown: float = 0.15):
        self.max_drawdown = max_drawdown
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.protection_triggered = False
        
        logger.info(f"DrawdownProtection initialized with {max_drawdown:.1%} max drawdown")
    
    def update_drawdown(self, current_value: float) -> float:
        """Update current drawdown calculation"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0
        
        return self.current_drawdown
    
    def should_trigger_protection(self) -> bool:
        """Check if drawdown protection should be triggered"""
        if self.current_drawdown >= self.max_drawdown and not self.protection_triggered:
            self.protection_triggered = True
            logger.warning(f"Drawdown protection triggered: {self.current_drawdown:.2%}")
            return True
        
        return False
    
    def get_position_reduction_factor(self) -> float:
        """Get position reduction factor when protection is triggered"""
        if not self.protection_triggered:
            return 1.0
        
        # Reduce positions by 50% when protection is triggered
        return 0.5
    
    def reset_protection(self, current_value: float) -> None:
        """Reset protection when drawdown improves"""
        if self.current_drawdown < self.max_drawdown * 0.5:  # 50% of max drawdown
            self.protection_triggered = False
            logger.info("Drawdown protection reset")


class RiskAnalytics:
    """Advanced risk analytics and calculations"""
    
    def __init__(self):
        self.var_confidence_levels = [0.95, 0.99]
        self.stress_scenarios = self._create_stress_scenarios()
        
        logger.info("RiskAnalytics initialized")
    
    def _create_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Create stress test scenarios"""
        return {
            "market_crash": {
                "equity_shock": -0.20,
                "volatility_shock": 2.0,
                "correlation_shock": 0.3
            },
            "flash_crash": {
                "equity_shock": -0.10,
                "volatility_shock": 3.0,
                "correlation_shock": 0.5
            },
            "volatility_spike": {
                "equity_shock": -0.05,
                "volatility_shock": 2.5,
                "correlation_shock": 0.2
            },
            "liquidity_crisis": {
                "equity_shock": -0.15,
                "volatility_shock": 2.0,
                "correlation_shock": 0.4
            }
        }
    
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        tail_returns = [r for r in returns if r <= var]
        
        return np.mean(tail_returns) if tail_returns else var
    
    def calculate_beta(self, asset_returns: List[float], 
                      market_returns: List[float]) -> float:
        """Calculate beta relative to market"""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    def stress_test(self, positions: List[Position], 
                   market_data: Dict[str, List[MarketData]]) -> Dict[str, float]:
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        for scenario_name, scenario_params in self.stress_scenarios.items():
            # Calculate stress impact
            stress_pnl = 0.0
            
            for position in positions:
                if position.symbol in market_data:
                    # Apply equity shock
                    price_shock = position.current_price * scenario_params["equity_shock"]
                    stress_pnl += position.quantity * price_shock
            
            stress_results[scenario_name] = stress_pnl
        
        return stress_results
    
    def calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        if not returns:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['mean_return'] = np.mean(returns)
        metrics['volatility'] = np.std(returns)
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # VaR and CVaR
        for confidence in self.var_confidence_levels:
            metrics[f'var_{int(confidence*100)}'] = self.calculate_var(returns, confidence)
            metrics[f'cvar_{int(confidence*100)}'] = self.calculate_cvar(returns, confidence)
        
        # Risk-adjusted returns
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
        
        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns)
            if downside_deviation > 0:
                metrics['sortino_ratio'] = metrics['mean_return'] / downside_deviation
        
        return metrics


class RiskManager:
    """Main risk management orchestrator"""
    
    def __init__(self, risk_limits: RiskLimits, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_limits = risk_limits
        self.risk_assessment = DynamicRiskAssessment()
        self.position_sizing = PositionSizingEngine(risk_level)
        self.drawdown_protection = DrawdownProtection(risk_limits.max_drawdown)
        self.risk_analytics = RiskAnalytics()
        
        # Risk state
        self.positions: List[Position] = []
        self.portfolio_value = 0.0
        self.risk_alerts: List[str] = []
        
        logger.info("RiskManager initialized")
    
    async def assess_risk(self, market_data: Dict[str, List[MarketData]]) -> PortfolioRisk:
        """Assess current portfolio risk"""
        portfolio_risk = self.risk_assessment.calculate_portfolio_risk(
            self.positions, market_data, self.portfolio_value
        )
        
        # Update drawdown protection
        self.drawdown_protection.update_drawdown(self.portfolio_value)
        
        # Check for risk alerts
        self.risk_alerts = self.risk_assessment.get_risk_alert(portfolio_risk, self.risk_limits)
        
        return portfolio_risk
    
    def calculate_position_size(self, signal: StrategySignal, 
                              strategy_performance: StrategyPerformance) -> float:
        """Calculate optimal position size for signal"""
        return self.position_sizing.optimal_position_size(
            signal, strategy_performance, self.portfolio_value, self.risk_limits
        )
    
    def should_allow_trade(self, signal: StrategySignal, 
                          portfolio_risk: PortfolioRisk) -> bool:
        """Determine if trade should be allowed based on risk limits"""
        
        # Check drawdown protection
        if self.drawdown_protection.should_trigger_protection():
            logger.warning("Trade blocked due to drawdown protection")
            return False
        
        # Check VaR limits
        if abs(portfolio_risk.var_95) > self.risk_limits.max_var:
            logger.warning("Trade blocked due to VaR limit")
            return False
        
        # Check daily loss limit
        if portfolio_risk.total_pnl < -self.risk_limits.max_daily_loss * self.portfolio_value:
            logger.warning("Trade blocked due to daily loss limit")
            return False
        
        # Check position size limits
        position_value = signal.quantity * signal.price
        if position_value > self.risk_limits.max_position_size * self.portfolio_value:
            logger.warning("Trade blocked due to position size limit")
            return False
        
        return True
    
    def get_risk_adjustments(self) -> Dict[str, float]:
        """Get risk adjustments for current market conditions"""
        adjustments = {
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0
        }
        
        # Apply drawdown protection
        if self.drawdown_protection.protection_triggered:
            adjustments['position_size_multiplier'] = self.drawdown_protection.get_position_reduction_factor()
        
        # Apply volatility adjustments
        if hasattr(self, 'current_volatility') and self.current_volatility > 0.03:  # High volatility
            adjustments['stop_loss_multiplier'] = 1.5
            adjustments['take_profit_multiplier'] = 1.2
        
        return adjustments
    
    def update_position(self, position: Position) -> None:
        """Update position in risk manager"""
        # Remove existing position if exists
        self.positions = [p for p in self.positions if p.symbol != position.symbol]
        
        # Add new position
        if position.quantity != 0:
            self.positions.append(position)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            'portfolio_risk': {
                'total_value': self.portfolio_value,
                'total_positions': len(self.positions),
                'current_drawdown': self.drawdown_protection.current_drawdown,
                'protection_triggered': self.drawdown_protection.protection_triggered
            },
            'risk_alerts': self.risk_alerts,
            'risk_limits': {
                'max_drawdown': self.risk_limits.max_drawdown,
                'max_var': self.risk_limits.max_var,
                'max_position_size': self.risk_limits.max_position_size,
                'max_daily_loss': self.risk_limits.max_daily_loss
            },
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'value': p.quantity * p.current_price,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for p in self.positions
            ]
        }


# Example usage
async def main():
    """Example usage of the risk management system"""
    
    # Create risk limits
    risk_limits = RiskLimits(
        max_portfolio_value=100000.0,
        max_position_size=0.1,
        max_daily_loss=0.02,
        max_drawdown=0.15,
        max_var=0.03
    )
    
    # Create risk manager
    risk_manager = RiskManager(risk_limits, RiskLevel.MODERATE)
    
    # Create sample position
    position = Position(
        symbol="EURUSD",
        quantity=10000.0,
        entry_price=1.1000,
        entry_time=datetime.utcnow(),
        current_price=1.1050,
        unrealized_pnl=500.0,
        realized_pnl=0.0,
        stop_loss=1.0900,
        take_profit=1.1200,
        strategy_id="trend_strategy_1"
    )
    
    risk_manager.update_position(position)
    risk_manager.portfolio_value = 100000.0
    
    print("Risk management system initialized and ready")


if __name__ == "__main__":
    asyncio.run(main()) 