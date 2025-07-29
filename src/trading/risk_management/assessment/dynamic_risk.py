"""
Dynamic Risk Assessment

Real-time risk assessment with market regime detection and volatility analysis.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.sensory.core.base import MarketData

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Dynamic risk metrics"""
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    correlation: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    risk_score: float


class DynamicRiskAssessor:
    """
    Dynamic Risk Assessment System
    
    Implements real-time risk assessment with:
    - Volatility analysis
    - Value at Risk (VaR) calculation
    - Conditional VaR (CVaR) calculation
    - Market regime detection
    - Correlation analysis
    - Beta calculation
    """
    
    def __init__(self, lookback_period: int = 252, confidence_level: float = 0.95):
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        
        # Risk state
        self.risk_history = []
        self.market_regime = "normal"
        self.volatility_regime = "normal"
        
        logger.info(f"DynamicRiskAssessor initialized: lookback={lookback_period}")
    
    def assess_risk(self, market_data: List[MarketData], 
                   portfolio_positions: Dict[str, float],
                   benchmark_data: Optional[List[MarketData]] = None) -> RiskMetrics:
        """Assess current risk metrics"""
        
        if len(market_data) < self.lookback_period:
            return self._create_default_metrics()
        
        # Calculate returns
        returns = self._calculate_returns(market_data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(returns)
        
        # Calculate VaR and CVaR
        var_95, var_99 = self._calculate_var(returns)
        cvar_95, cvar_99 = self._calculate_cvar(returns)
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(market_data)
        
        # Calculate correlation
        correlation = self._calculate_correlation(returns, benchmark_data)
        
        # Calculate beta
        beta = self._calculate_beta(returns, benchmark_data)
        
        # Calculate risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            volatility, var_95, max_drawdown, correlation
        )
        
        # Update market regime
        self._update_market_regime(volatility, var_95)
        
        metrics = RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            correlation=correlation,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            risk_score=risk_score
        )
        
        self.risk_history.append(metrics)
        
        return metrics
    
    def _calculate_returns(self, market_data: List[MarketData]) -> List[float]:
        """Calculate price returns"""
        returns = []
        for i in range(1, len(market_data)):
            if market_data[i-1].close > 0:
                ret = (market_data[i].close - market_data[i-1].close) / market_data[i-1].close
                returns.append(ret)
        return returns
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility"""
        if not returns:
            return 0.0
        
        # Calculate rolling volatility
        if len(returns) >= 30:
            rolling_vol = np.std(returns[-30:])
        else:
            rolling_vol = np.std(returns)
        
        # Annualize (assuming daily returns)
        annualized_vol = rolling_vol * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_var(self, returns: List[float]) -> tuple:
        """Calculate Value at Risk"""
        if not returns:
            return 0.0, 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate VaR at 95% and 99% confidence
        var_95_idx = int(0.05 * len(sorted_returns))
        var_99_idx = int(0.01 * len(sorted_returns))
        
        var_95 = -sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else 0.0
        var_99 = -sorted_returns[var_99_idx] if var_99_idx < len(sorted_returns) else 0.0
        
        return var_95, var_99
    
    def _calculate_cvar(self, returns: List[float]) -> tuple:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return 0.0, 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate CVaR at 95% and 99% confidence
        var_95_idx = int(0.05 * len(sorted_returns))
        var_99_idx = int(0.01 * len(sorted_returns))
        
        if var_95_idx < len(sorted_returns):
            cvar_95 = -np.mean(sorted_returns[:var_95_idx])
        else:
            cvar_95 = 0.0
        
        if var_99_idx < len(sorted_returns):
            cvar_99 = -np.mean(sorted_returns[:var_99_idx])
        else:
            cvar_99 = 0.0
        
        return cvar_95, cvar_99
    
    def _calculate_max_drawdown(self, market_data: List[MarketData]) -> float:
        """Calculate maximum drawdown"""
        if not market_data:
            return 0.0
        
        prices = [md.close for md in market_data]
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_correlation(self, returns: List[float], 
                             benchmark_data: Optional[List[MarketData]]) -> float:
        """Calculate correlation with benchmark"""
        if not benchmark_data or len(benchmark_data) < len(returns) + 1:
            return 0.0
        
        benchmark_returns = self._calculate_returns(benchmark_data)
        
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
        
        if len(returns) < 2:
            return 0.0
        
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_beta(self, returns: List[float], 
                       benchmark_data: Optional[List[MarketData]]) -> float:
        """Calculate beta relative to benchmark"""
        if not benchmark_data or len(benchmark_data) < len(returns) + 1:
            return 1.0
        
        benchmark_returns = self._calculate_returns(benchmark_data)
        
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
        
        if len(returns) < 2:
            return 1.0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 1.0
        
        beta = covariance / benchmark_variance
        return beta if not np.isnan(beta) else 1.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualized Sortino ratio
        sortino_ratio = (mean_return / downside_deviation) * np.sqrt(252)
        return sortino_ratio
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        annualized_return = mean_return * 252
        
        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio
    
    def _calculate_risk_score(self, volatility: float, var_95: float, 
                            max_drawdown: float, correlation: float) -> float:
        """Calculate overall risk score (0-100)"""
        # Normalize components
        vol_score = min(volatility * 100, 100)  # Volatility score
        var_score = min(var_95 * 100, 100)      # VaR score
        dd_score = min(max_drawdown * 100, 100)  # Drawdown score
        corr_score = abs(correlation) * 50       # Correlation score
        
        # Weighted average
        risk_score = (vol_score * 0.3 + var_score * 0.3 + 
                     dd_score * 0.3 + corr_score * 0.1)
        
        return min(risk_score, 100)
    
    def _update_market_regime(self, volatility: float, var_95: float) -> None:
        """Update market regime based on risk metrics"""
        # Volatility regime
        if volatility > 0.3:  # 30% annualized volatility
            self.volatility_regime = "high"
        elif volatility < 0.1:  # 10% annualized volatility
            self.volatility_regime = "low"
        else:
            self.volatility_regime = "normal"
        
        # Market regime
        if var_95 > 0.05:  # 5% daily VaR
            self.market_regime = "stress"
        elif var_95 < 0.01:  # 1% daily VaR
            self.market_regime = "calm"
        else:
            self.market_regime = "normal"
    
    def _create_default_metrics(self) -> RiskMetrics:
        """Create default risk metrics when insufficient data"""
        return RiskMetrics(
            volatility=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            max_drawdown=0.0,
            correlation=0.0,
            beta=1.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            risk_score=0.0
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk assessment summary"""
        if not self.risk_history:
            return {}
        
        latest_metrics = self.risk_history[-1]
        
        return {
            'current_risk_score': latest_metrics.risk_score,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'volatility': latest_metrics.volatility,
            'var_95': latest_metrics.var_95,
            'max_drawdown': latest_metrics.max_drawdown,
            'sharpe_ratio': latest_metrics.sharpe_ratio,
            'risk_trend': self._calculate_risk_trend()
        }
    
    def _calculate_risk_trend(self) -> str:
        """Calculate risk trend over time"""
        if len(self.risk_history) < 10:
            return "stable"
        
        recent_scores = [m.risk_score for m in self.risk_history[-10:]]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 1:
            return "increasing"
        elif trend < -1:
            return "decreasing"
        else:
            return "stable" 
