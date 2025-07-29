"""
EMP Risk Analyzer v1.1

Risk analysis and assessment for the thinking layer.
Provides comprehensive risk metrics and analysis for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

from ...core.events import TradeIntent, RiskMetrics, AnalysisResult

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Risk analyzer for cognitive assessment of trading risks."""
    
    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 252):
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.risk_history: List[Dict[str, Any]] = []
        
        logger.info(f"Risk Analyzer initialized with {confidence_level:.0%} confidence level")
        
    def analyze_risk(self, trade_history: List[TradeIntent], 
                    market_data: Optional[List[Any]] = None) -> AnalysisResult:
        """Analyze trading risk and generate risk metrics."""
        try:
            if not trade_history:
                logger.warning("Empty trade history provided for risk analysis")
                return self._create_default_analysis()
                
            # Convert trade history to risk data
            risk_data = self._convert_trades_to_risk_data(trade_history)
            
            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(risk_data, market_data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_risk_confidence(risk_data)
            
            # Create analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="risk_analysis",
                result={
                    "risk_metrics": metrics.__dict__,
                    "trade_count": len(trade_history),
                    "analysis_period": self._calculate_analysis_period(trade_history),
                    "confidence_level": self.confidence_level,
                    "risk_assessment": self._assess_risk_level(metrics)
                },
                confidence=confidence,
                metadata={
                    "analyzer_version": "1.1.0",
                    "method": "comprehensive_risk_analysis",
                    "confidence_level": self.confidence_level,
                    "lookback_period": self.lookback_period
                }
            )
            
            # Store in history
            self.risk_history.append({
                "timestamp": result.timestamp,
                "trade_count": len(trade_history),
                "var_95": metrics.var_95,
                "cvar_95": metrics.cvar_95,
                "max_drawdown": metrics.max_drawdown,
                "risk_score": metrics.risk_score,
                "confidence": confidence
            })
            
            logger.debug(f"Risk analyzed: VaR95={metrics.var_95:.2%}, CVaR95={metrics.cvar_95:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing risk: {e}")
            return self._create_default_analysis()
            
    def _convert_trades_to_risk_data(self, trade_history: List[TradeIntent]) -> Dict[str, Any]:
        """Convert trade history to risk data structure."""
        # Sort trades by timestamp
        sorted_trades = sorted(trade_history, key=lambda x: x.timestamp)
        
        # Extract trade returns and positions
        trade_returns = []
        position_sizes = []
        timestamps = []
        
        for trade in sorted_trades:
            if trade.action == "HOLD":
                continue
                
            # Calculate trade return (simplified)
            if trade.price and trade.quantity:
                # Simplified P&L calculation
                if trade.action == "SELL":
                    # Assume we're closing a position
                    trade_return = 0.02  # Simplified 2% return per trade
                    trade_returns.append(trade_return)
                    position_sizes.append(trade.quantity * trade.price)
                    timestamps.append(trade.timestamp)
                    
        return {
            'trade_returns': pd.Series(trade_returns),
            'position_sizes': position_sizes,
            'timestamps': timestamps,
            'total_trades': len(trade_returns)
        }
        
    def _calculate_risk_metrics(self, risk_data: Dict[str, Any], 
                              market_data: Optional[List[Any]] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        trade_returns = risk_data['trade_returns']
        position_sizes = risk_data['position_sizes']
        
        # Value at Risk (VaR)
        var_95 = self._calculate_var(trade_returns, 0.95)
        var_99 = self._calculate_var(trade_returns, 0.99)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = self._calculate_cvar(trade_returns, 0.95)
        cvar_99 = self._calculate_cvar(trade_returns, 0.99)
        
        # Beta calculation (simplified)
        beta = self._calculate_beta(trade_returns, market_data)
        
        # Correlation calculation
        correlation = self._calculate_correlation(trade_returns, market_data)
        
        # Current drawdown
        current_drawdown = self._calculate_current_drawdown(trade_returns)
        
        # Risk score
        risk_score = self._calculate_risk_score(var_95, cvar_95, current_drawdown, beta)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            correlation=correlation,
            current_drawdown=current_drawdown,
            risk_score=risk_score,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(trade_returns),
                "confidence_level": self.confidence_level
            }
        )
        
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0.0
            
        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return abs(var)
        
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0.0
            
        # Historical CVaR
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # Calculate expected value of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) > 0:
            cvar = tail_returns.mean()
        else:
            cvar = var_threshold
            
        return abs(cvar)
        
    def _calculate_beta(self, returns: pd.Series, market_data: Optional[List[Any]]) -> float:
        """Calculate beta relative to market."""
        if not market_data or len(returns) < 2:
            return 1.0  # Default to market beta
            
        try:
            # Simplified market return calculation
            # In a real implementation, you'd use actual market data
            market_returns = pd.Series([0.001] * len(returns))  # Simplified 0.1% daily return
            
            # Calculate covariance and variance
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = 1.0
                
            return beta
            
        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0
            
    def _calculate_correlation(self, returns: pd.Series, market_data: Optional[List[Any]]) -> float:
        """Calculate correlation with market."""
        if not market_data or len(returns) < 2:
            return 0.0
            
        try:
            # Simplified market return calculation
            market_returns = pd.Series([0.001] * len(returns))  # Simplified
            
            correlation = np.corrcoef(returns, market_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0
            
    def _calculate_current_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown from returns."""
        if len(returns) < 2:
            return 0.0
            
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate current drawdown
        current_drawdown = (cumulative_returns.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        
        return abs(current_drawdown)
        
    def _calculate_risk_score(self, var_95: float, cvar_95: float, 
                            current_drawdown: float, beta: float) -> float:
        """Calculate composite risk score."""
        # Normalize risk metrics to [0, 1] scale
        var_score = min(var_95 / 0.1, 1.0)  # Normalize to 10% VaR
        cvar_score = min(cvar_95 / 0.15, 1.0)  # Normalize to 15% CVaR
        drawdown_score = min(current_drawdown / 0.2, 1.0)  # Normalize to 20% drawdown
        beta_score = min(abs(beta - 1.0) / 0.5, 1.0)  # Normalize beta deviation
        
        # Weighted risk score
        weights = [0.3, 0.3, 0.25, 0.15]  # VaR, CVaR, Drawdown, Beta
        risk_score = (var_score * weights[0] + 
                     cvar_score * weights[1] + 
                     drawdown_score * weights[2] + 
                     beta_score * weights[3])
        
        return min(risk_score, 1.0)
        
    def _assess_risk_level(self, metrics: RiskMetrics) -> str:
        """Assess overall risk level."""
        risk_score = metrics.risk_score
        
        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.4:
            return "LOW_MEDIUM"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "MEDIUM_HIGH"
        else:
            return "HIGH"
            
    def _calculate_risk_confidence(self, risk_data: Dict[str, Any]) -> float:
        """Calculate confidence in risk analysis."""
        confidence_factors = []
        
        # Data sufficiency
        trade_count = risk_data['total_trades']
        if trade_count >= 100:
            confidence_factors.append(1.0)
        elif trade_count >= 50:
            confidence_factors.append(0.8)
        elif trade_count >= 20:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Data quality
        returns = risk_data['trade_returns']
        if len(returns) > 0:
            # Check for extreme outliers
            q1 = returns.quantile(0.25)
            q3 = returns.quantile(0.75)
            iqr = q3 - q1
            outliers = returns[(returns < q1 - 1.5 * iqr) | (returns > q3 + 1.5 * iqr)]
            
            outlier_ratio = len(outliers) / len(returns)
            if outlier_ratio < 0.05:
                confidence_factors.append(1.0)
            elif outlier_ratio < 0.1:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Time span
        if len(risk_data['timestamps']) > 1:
            time_span = (risk_data['timestamps'][-1] - risk_data['timestamps'][0]).days
            if time_span >= 252:  # 1 year
                confidence_factors.append(1.0)
            elif time_span >= 126:  # 6 months
                confidence_factors.append(0.8)
            elif time_span >= 63:   # 3 months
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.1)
            
        return np.mean(confidence_factors)
        
    def _calculate_analysis_period(self, trade_history: List[TradeIntent]) -> Dict[str, Any]:
        """Calculate the analysis period from trade history."""
        if not trade_history:
            return {"start": None, "end": None, "duration_days": 0}
            
        start_time = min(trade.timestamp for trade in trade_history)
        end_time = max(trade.timestamp for trade in trade_history)
        duration_days = (end_time - start_time).days
        
        return {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_days": duration_days
        }
        
    def _create_default_analysis(self) -> AnalysisResult:
        """Create default analysis when risk analysis fails."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="risk_analysis",
            result={
                "risk_metrics": RiskMetrics().__dict__,
                "trade_count": 0,
                "analysis_period": {"start": None, "end": None, "duration_days": 0},
                "confidence_level": self.confidence_level,
                "risk_assessment": "UNKNOWN"
            },
            confidence=0.1,
            metadata={
                "analyzer_version": "1.1.0",
                "method": "default_fallback",
                "error": "Insufficient data for analysis"
            }
        )
        
    def get_risk_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get risk analysis history."""
        if limit:
            return self.risk_history[-limit:]
        return self.risk_history.copy()
        
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get statistics about risk analyses."""
        if not self.risk_history:
            return {}
            
        var_95s = [h['var_95'] for h in self.risk_history]
        cvar_95s = [h['cvar_95'] for h in self.risk_history]
        drawdowns = [h['max_drawdown'] for h in self.risk_history]
        risk_scores = [h['risk_score'] for h in self.risk_history]
        confidences = [h['confidence'] for h in self.risk_history]
        
        return {
            'total_analyses': len(self.risk_history),
            'average_var_95': np.mean(var_95s),
            'average_cvar_95': np.mean(cvar_95s),
            'average_drawdown': np.mean(drawdowns),
            'average_risk_score': np.mean(risk_scores),
            'average_confidence': np.mean(confidences),
            'max_var_95': max(var_95s) if var_95s else 0,
            'max_cvar_95': max(cvar_95s) if cvar_95s else 0,
            'max_risk_score': max(risk_scores) if risk_scores else 0
        } 
