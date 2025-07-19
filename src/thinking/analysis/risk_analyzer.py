"""
EMP Risk Analyzer v1.1

Comprehensive risk analysis module that processes sensory signals
and pattern detection results to assess market risk levels.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from src.core.interfaces import SensorySignal, AnalysisResult
from src.core.exceptions import ThinkingException

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: float
    correlation: float
    risk_score: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    metadata: Dict[str, Any]


class RiskAnalyzer:
    """Analyzes risk from sensory signals and market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', 252)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self._signal_history: List[SensorySignal] = []
        
    def analyze(self, signals: List[SensorySignal], 
                market_data: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze risk from sensory signals."""
        try:
            # Update signal history
            self._update_signal_history(signals)
            
            # Extract risk-relevant signals
            risk_signals = self._extract_risk_signals(signals)
            
            if not risk_signals:
                return self._create_neutral_result()
                
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(risk_signals, market_data)
            
            # Create analysis result
            return AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="risk_analysis",
                result={
                    'volatility': risk_metrics.volatility,
                    'var_95': risk_metrics.var_95,
                    'cvar_95': risk_metrics.cvar_95,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'beta': risk_metrics.beta,
                    'correlation': risk_metrics.correlation,
                    'risk_score': risk_metrics.risk_score,
                    'risk_level': risk_metrics.risk_level,
                    'metadata': risk_metrics.metadata
                },
                confidence=self._calculate_confidence(risk_signals),
                metadata={
                    'signal_count': len(risk_signals),
                    'analysis_method': 'comprehensive_risk'
                }
            )
            
        except Exception as e:
            raise ThinkingException(f"Error in risk analysis: {e}")
            
    def _update_signal_history(self, signals: List[SensorySignal]):
        """Update the signal history."""
        self._signal_history.extend(signals)
        
        # Keep only recent signals
        if len(self._signal_history) > self.lookback_periods:
            self._signal_history = self._signal_history[-self.lookback_periods:]
            
    def _extract_risk_signals(self, signals: List[SensorySignal]) -> List[SensorySignal]:
        """Extract signals relevant to risk analysis."""
        risk_signals = []
        
        for signal in signals:
            if signal.signal_type in ['price_composite', 'volatility', 'volume_composite']:
                risk_signals.append(signal)
                
        return risk_signals
        
    def _calculate_risk_metrics(self, signals: List[SensorySignal], 
                               market_data: Optional[Dict[str, Any]]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # Extract price movements from signals
        price_signals = [s for s in signals if s.signal_type == 'price_composite']
        
        if not price_signals:
            return self._create_neutral_risk_metrics()
            
        # Calculate returns from price signals
        returns = []
        for i in range(1, len(price_signals)):
            prev_value = price_signals[i-1].value
            curr_value = price_signals[i].value
            if prev_value != 0:
                returns.append((curr_value - prev_value) / prev_value)
                
        if not returns:
            return self._create_neutral_risk_metrics()
            
        returns_array = np.array(returns)
        
        # Calculate volatility
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        # Calculate VaR and CVaR
        var_95 = -np.percentile(returns_array, 5)
        cvar_95 = -np.mean(returns_array[returns_array <= -var_95])
        
        # Calculate maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate beta and correlation (if market data available)
        beta = 1.0  # Default beta
        correlation = 0.0  # Default correlation
        
        if market_data and 'market_returns' in market_data:
            market_returns = np.array(market_data['market_returns'])
            if len(market_returns) == len(returns_array):
                # Calculate correlation
                correlation = np.corrcoef(returns_array, market_returns)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                    
                # Calculate beta
                market_variance = np.var(market_returns)
                if market_variance > 0:
                    covariance = np.cov(returns_array, market_returns)[0, 1]
                    beta = covariance / market_variance
                else:
                    beta = 1.0
                    
        # Calculate composite risk score
        risk_score = self._calculate_risk_score(
            volatility, var_95, max_drawdown, beta
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            beta=beta,
            correlation=correlation,
            risk_score=risk_score,
            risk_level=risk_level,
            metadata={
                'return_count': len(returns),
                'signal_count': len(signals),
                'analysis_periods': self.lookback_periods
            }
        )
        
    def _calculate_risk_score(self, volatility: float, var_95: float, 
                             max_drawdown: float, beta: float) -> float:
        """Calculate composite risk score."""
        # Normalize components to 0-1 range
        vol_score = min(volatility / 0.5, 1.0)  # 50% volatility = max score
        var_score = min(var_95 / 0.1, 1.0)      # 10% VaR = max score
        dd_score = min(max_drawdown / 0.3, 1.0)  # 30% drawdown = max score
        beta_score = min(abs(beta - 1.0) / 2.0, 1.0)  # Beta deviation from 1
        
        # Weighted average
        weights = [0.3, 0.3, 0.3, 0.1]  # volatility, VaR, drawdown, beta
        risk_score = (
            vol_score * weights[0] +
            var_score * weights[1] +
            dd_score * weights[2] +
            beta_score * weights[3]
        )
        
        return min(risk_score, 1.0)
        
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on risk score."""
        if risk_score < 0.25:
            return 'LOW'
        elif risk_score < 0.5:
            return 'MEDIUM'
        elif risk_score < 0.75:
            return 'HIGH'
        else:
            return 'EXTREME'
            
    def _calculate_confidence(self, signals: List[SensorySignal]) -> float:
        """Calculate confidence in risk analysis."""
        if not signals:
            return 0.0
            
        confidences = [s.confidence for s in signals]
        return np.mean(confidences)
        
    def _create_neutral_risk_metrics(self) -> RiskMetrics:
        """Create neutral risk metrics when no data is available."""
        return RiskMetrics(
            volatility=0.0,
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            beta=1.0,
            correlation=0.0,
            risk_score=0.0,
            risk_level='LOW',
            metadata={
                'return_count': 0,
                'signal_count': 0,
                'analysis_periods': 0
            }
        )
        
    def _create_neutral_result(self) -> AnalysisResult:
        """Create a neutral analysis result when no signals are available."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="risk_analysis",
            result={
                'volatility': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0,
                'beta': 1.0,
                'correlation': 0.0,
                'risk_score': 0.0,
                'risk_level': 'LOW',
                'metadata': {}
            },
            confidence=0.0,
            metadata={
                'signal_count': 0,
                'analysis_method': 'neutral_fallback'
            }
        ) 