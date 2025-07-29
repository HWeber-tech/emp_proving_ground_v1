"""
Kelly Criterion Position Sizing

Specialized Kelly criterion implementation for optimal position sizing.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly criterion calculation result"""
    kelly_fraction: float
    optimal_position_size: float
    win_rate: float
    average_win: float
    average_loss: float
    risk_adjusted_fraction: float
    confidence_interval: Tuple[float, float]


class KellyCriterion:
    """
    Kelly Criterion Position Sizing
    
    Implements Kelly criterion with:
    - Win rate calculation
    - Average win/loss analysis
    - Risk-adjusted Kelly fraction
    - Confidence intervals
    - Fractional Kelly
    """
    
    def __init__(self, max_kelly_fraction: float = 0.25, 
                 confidence_level: float = 0.95,
                 min_trades: int = 30):
        self.max_kelly_fraction = max_kelly_fraction
        self.confidence_level = confidence_level
        self.min_trades = min_trades
        
        # Kelly state
        self.trade_history = []
        self.kelly_history = []
        
        logger.info(f"KellyCriterion initialized: max_fraction={max_kelly_fraction}")
    
    def calculate_position_size(self, trade_history: List[Dict[str, Any]], 
                              available_capital: float,
                              current_win_rate: Optional[float] = None,
                              current_avg_win: Optional[float] = None,
                              current_avg_loss: Optional[float] = None) -> KellyResult:
        """Calculate optimal position size using Kelly criterion"""
        
        if len(trade_history) < self.min_trades:
            return self._create_default_result(available_capital)
        
        # Calculate win rate and average win/loss
        win_rate = current_win_rate or self._calculate_win_rate(trade_history)
        avg_win = current_avg_win or self._calculate_average_win(trade_history)
        avg_loss = current_avg_loss or self._calculate_average_loss(trade_history)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Apply risk adjustments
        risk_adjusted_fraction = self._apply_risk_adjustments(kelly_fraction, trade_history)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            win_rate, avg_win, avg_loss, trade_history
        )
        
        # Calculate optimal position size
        optimal_position_size = available_capital * risk_adjusted_fraction
        
        result = KellyResult(
            kelly_fraction=kelly_fraction,
            optimal_position_size=optimal_position_size,
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            risk_adjusted_fraction=risk_adjusted_fraction,
            confidence_interval=confidence_interval
        )
        
        self.kelly_history.append(result)
        
        return result
    
    def _calculate_win_rate(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade history"""
        if not trade_history:
            return 0.5
        
        winning_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
        total_trades = len(trade_history)
        
        return winning_trades / total_trades if total_trades > 0 else 0.5
    
    def _calculate_average_win(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate average winning trade"""
        winning_trades = [trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0]
        
        if not winning_trades:
            return 1.0  # Default average win
        
        return np.mean(winning_trades)
    
    def _calculate_average_loss(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate average losing trade"""
        losing_trades = [abs(trade.get('pnl', 0)) for trade in trade_history if trade.get('pnl', 0) < 0]
        
        if not losing_trades:
            return 1.0  # Default average loss
        
        return np.mean(losing_trades)
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction"""
        if avg_loss == 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        b = avg_win / avg_loss  # odds received
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Ensure Kelly fraction is within reasonable bounds
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_kelly_fraction))
        
        return kelly_fraction
    
    def _apply_risk_adjustments(self, kelly_fraction: float, 
                               trade_history: List[Dict[str, Any]]) -> float:
        """Apply risk adjustments to Kelly fraction"""
        
        # Fractional Kelly (use half Kelly for safety)
        fractional_kelly = kelly_fraction * 0.5
        
        # Volatility adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(trade_history)
        
        # Drawdown adjustment
        drawdown_adjustment = self._calculate_drawdown_adjustment(trade_history)
        
        # Correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(trade_history)
        
        # Combine adjustments
        adjusted_fraction = fractional_kelly * volatility_adjustment * drawdown_adjustment * correlation_adjustment
        
        # Ensure within bounds
        adjusted_fraction = max(0.0, min(adjusted_fraction, self.max_kelly_fraction))
        
        return adjusted_fraction
    
    def _calculate_volatility_adjustment(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate volatility-based adjustment"""
        if len(trade_history) < 10:
            return 1.0
        
        # Calculate PnL volatility
        pnl_values = [trade.get('pnl', 0) for trade in trade_history[-50:]]
        volatility = np.std(pnl_values)
        
        # Adjust based on volatility
        if volatility > np.mean(pnl_values) * 2:
            return 0.7  # Reduce position size for high volatility
        elif volatility < np.mean(pnl_values) * 0.5:
            return 1.1  # Increase position size for low volatility
        else:
            return 1.0
    
    def _calculate_drawdown_adjustment(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate drawdown-based adjustment"""
        if len(trade_history) < 20:
            return 1.0
        
        # Calculate recent drawdown
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for trade in trade_history[-20:]:
            cumulative_pnl += trade.get('pnl', 0)
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = (peak_pnl - cumulative_pnl) / peak_pnl if peak_pnl > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Adjust based on drawdown
        if max_drawdown > 0.1:  # 10% drawdown
            return 0.8
        elif max_drawdown > 0.05:  # 5% drawdown
            return 0.9
        else:
            return 1.0
    
    def _calculate_correlation_adjustment(self, trade_history: List[Dict[str, Any]]) -> float:
        """Calculate correlation-based adjustment"""
        if len(trade_history) < 10:
            return 1.0
        
        # Calculate autocorrelation of PnL
        pnl_values = [trade.get('pnl', 0) for trade in trade_history[-20:]]
        
        if len(pnl_values) < 2:
            return 1.0
        
        # Calculate lag-1 autocorrelation
        autocorr = np.corrcoef(pnl_values[:-1], pnl_values[1:])[0, 1]
        
        if np.isnan(autocorr):
            return 1.0
        
        # Adjust based on autocorrelation
        if abs(autocorr) > 0.3:  # High autocorrelation
            return 0.8  # Reduce position size
        else:
            return 1.0
    
    def _calculate_confidence_interval(self, win_rate: float, avg_win: float, avg_loss: float,
                                     trade_history: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction"""
        if len(trade_history) < 30:
            return (0.0, self.max_kelly_fraction)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        kelly_samples = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_trades = np.random.choice(trade_history, size=len(trade_history), replace=True)
            
            # Calculate Kelly fraction for bootstrap sample
            boot_win_rate = self._calculate_win_rate(bootstrap_trades)
            boot_avg_win = self._calculate_average_win(bootstrap_trades)
            boot_avg_loss = self._calculate_average_loss(bootstrap_trades)
            
            boot_kelly = self._calculate_kelly_fraction(boot_win_rate, boot_avg_win, boot_avg_loss)
            kelly_samples.append(boot_kelly)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(kelly_samples, lower_percentile)
        upper_bound = np.percentile(kelly_samples, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _create_default_result(self, available_capital: float) -> KellyResult:
        """Create default Kelly result when insufficient data"""
        return KellyResult(
            kelly_fraction=0.0,
            optimal_position_size=available_capital * 0.02,  # 2% default
            win_rate=0.5,
            average_win=1.0,
            average_loss=1.0,
            risk_adjusted_fraction=0.02,
            confidence_interval=(0.0, 0.05)
        )
    
    def get_kelly_summary(self) -> Dict[str, Any]:
        """Get Kelly criterion summary"""
        if not self.kelly_history:
            return {}
        
        latest_result = self.kelly_history[-1]
        
        return {
            'current_kelly_fraction': latest_result.kelly_fraction,
            'risk_adjusted_fraction': latest_result.risk_adjusted_fraction,
            'win_rate': latest_result.win_rate,
            'average_win': latest_result.average_win,
            'average_loss': latest_result.average_loss,
            'confidence_interval': latest_result.confidence_interval,
            'kelly_trend': self._calculate_kelly_trend()
        }
    
    def _calculate_kelly_trend(self) -> str:
        """Calculate Kelly fraction trend"""
        if len(self.kelly_history) < 5:
            return "stable"
        
        recent_fractions = [r.kelly_fraction for r in self.kelly_history[-5:]]
        trend = np.polyfit(range(len(recent_fractions)), recent_fractions, 1)[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable" 
