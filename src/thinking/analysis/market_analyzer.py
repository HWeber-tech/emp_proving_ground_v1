"""
EMP Market Analyzer v1.1

Comprehensive market analysis for the thinking layer.
Combines multiple analysis components into unified market insights.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from src.core.interfaces import ThinkingPattern, SensorySignal, AnalysisResult  # legacy
except Exception:  # pragma: no cover
    ThinkingPattern = SensorySignal = AnalysisResult = object  # type: ignore
from src.core.exceptions import ThinkingException
from .performance_analyzer import PerformanceAnalyzer
from .risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)


class MarketAnalyzer(ThinkingPattern):
    """Comprehensive market analyzer combining multiple analysis components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
    def analyze(self, signals: List[SensorySignal]) -> AnalysisResult:
        """Analyze market using multiple analysis components."""
        try:
            # Perform performance analysis
            performance_result = self.performance_analyzer.analyze_performance(
                [],  # Empty trade history for signal-based analysis
                initial_capital=100000.0
            )
            
            # Perform risk analysis
            risk_result = self.risk_analyzer.analyze_risk(
                [],  # Empty trade history for signal-based analysis
                market_data=None
            )
            
            # Combine analysis results
            combined_analysis = self._combine_analysis_results(
                performance_result, risk_result, signals
            )
            
            # Create unified market analysis result
            return AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="market_analysis",
                result=combined_analysis,
                confidence=self._calculate_overall_confidence(performance_result, risk_result),
                metadata={
                    'analysis_components': ['performance', 'risk'],
                    'signal_count': len(signals),
                    'analysis_method': 'comprehensive_market_analysis'
                }
            )
            
        except Exception as e:
            raise ThinkingException(f"Error in market analysis: {e}")
            
    def learn(self, feedback: Dict[str, Any]) -> bool:
        """Learn from feedback to improve market analysis."""
        try:
            # Delegate learning to component analyzers
            performance_learned = self.performance_analyzer.learn(feedback)
            risk_learned = self.risk_analyzer.learn(feedback)
            
            logger.info("Market analyzer learned from feedback")
            return performance_learned and risk_learned
            
        except Exception as e:
            logger.error(f"Error in market analyzer learning: {e}")
            return False
            
    def _combine_analysis_results(self, performance_result: AnalysisResult,
                                risk_result: AnalysisResult,
                                signals: List[SensorySignal]) -> Dict[str, Any]:
        """Combine analysis results into unified market insights."""
        
        # Extract key metrics
        performance_metrics = performance_result.result.get('performance_metrics', {})
        risk_metrics = risk_result.result.get('risk_metrics', {})
        
        # Calculate market sentiment from signals
        market_sentiment = self._calculate_market_sentiment(signals)
        
        # Create combined analysis
        combined_analysis = {
            'market_sentiment': market_sentiment,
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'market_health': self._calculate_market_health(performance_metrics, risk_metrics),
            'trading_opportunity': self._assess_trading_opportunity(performance_metrics, risk_metrics, market_sentiment),
            'risk_level': self._assess_risk_level(risk_metrics),
            'signal_quality': self._assess_signal_quality(signals)
        }
        
        return combined_analysis
        
    def _calculate_market_sentiment(self, signals: List[SensorySignal]) -> Dict[str, Any]:
        """Calculate market sentiment from sensory signals."""
        if not signals:
            return {
                'overall_sentiment': 0.5,
                'confidence': 0.0,
                'signal_count': 0
            }
            
        # Calculate sentiment from signal values
        sentiment_values = []
        confidences = []
        
        for signal in signals:
            if signal.signal_type in ['sentiment', 'momentum', 'price_composite']:
                sentiment_values.append(signal.value)
                confidences.append(signal.confidence)
                
        if sentiment_values:
            overall_sentiment = sum(v * c for v, c in zip(sentiment_values, confidences)) / sum(confidences)
            confidence = np.mean(confidences)
        else:
            overall_sentiment = 0.5
            confidence = 0.0
            
        return {
            'overall_sentiment': max(0.0, min(1.0, overall_sentiment)),
            'confidence': confidence,
            'signal_count': len(signals)
        }
        
    def _calculate_market_health(self, performance_metrics: Dict[str, Any],
                               risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market health score."""
        
        # Extract key metrics
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = performance_metrics.get('max_drawdown', 0.0)
        risk_score = risk_metrics.get('risk_score', 0.5)
        
        # Calculate health components
        performance_health = min(max(sharpe_ratio / 2.0, 0.0), 1.0)  # Normalize Sharpe ratio
        drawdown_health = max(0.0, 1.0 - max_drawdown)  # Lower drawdown = better health
        risk_health = max(0.0, 1.0 - risk_score)  # Lower risk = better health
        
        # Calculate overall health
        overall_health = (performance_health + drawdown_health + risk_health) / 3.0
        
        return {
            'overall_health': overall_health,
            'performance_health': performance_health,
            'drawdown_health': drawdown_health,
            'risk_health': risk_health,
            'health_status': self._classify_health_status(overall_health)
        }
        
    def _assess_trading_opportunity(self, performance_metrics: Dict[str, Any],
                                  risk_metrics: Dict[str, Any],
                                  market_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess trading opportunity based on analysis."""
        
        # Extract metrics
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        risk_score = risk_metrics.get('risk_score', 0.5)
        sentiment = market_sentiment.get('overall_sentiment', 0.5)
        
        # Calculate opportunity score
        opportunity_score = 0.0
        
        # Performance component
        if sharpe_ratio > 1.0:
            opportunity_score += 0.4
        elif sharpe_ratio > 0.5:
            opportunity_score += 0.2
            
        # Risk component
        if risk_score < 0.3:
            opportunity_score += 0.3
        elif risk_score < 0.5:
            opportunity_score += 0.15
            
        # Sentiment component
        if sentiment > 0.7:
            opportunity_score += 0.3
        elif sentiment > 0.5:
            opportunity_score += 0.15
            
        opportunity_score = min(opportunity_score, 1.0)
        
        return {
            'opportunity_score': opportunity_score,
            'opportunity_level': self._classify_opportunity_level(opportunity_score),
            'recommended_action': self._recommend_action(opportunity_score, sentiment)
        }
        
    def _assess_risk_level(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current risk level."""
        risk_score = risk_metrics.get('risk_score', 0.5)
        var_95 = risk_metrics.get('var_95', 0.0)
        current_drawdown = risk_metrics.get('current_drawdown', 0.0)
        
        # Classify risk level
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'var_95': var_95,
            'current_drawdown': current_drawdown,
            'risk_warnings': self._generate_risk_warnings(risk_metrics)
        }
        
    def _assess_signal_quality(self, signals: List[SensorySignal]) -> Dict[str, Any]:
        """Assess the quality of sensory signals."""
        if not signals:
            return {
                'quality_score': 0.0,
                'signal_count': 0,
                'average_confidence': 0.0,
                'quality_status': 'poor'
            }
            
        # Calculate quality metrics
        confidences = [s.confidence for s in signals]
        average_confidence = np.mean(confidences)
        
        # Calculate quality score
        quality_score = average_confidence * min(len(signals) / 10.0, 1.0)
        
        # Classify quality
        if quality_score > 0.7:
            quality_status = 'excellent'
        elif quality_score > 0.5:
            quality_status = 'good'
        elif quality_score > 0.3:
            quality_status = 'fair'
        else:
            quality_status = 'poor'
            
        return {
            'quality_score': quality_score,
            'signal_count': len(signals),
            'average_confidence': average_confidence,
            'quality_status': quality_status
        }
        
    def _calculate_overall_confidence(self, performance_result: AnalysisResult,
                                    risk_result: AnalysisResult) -> float:
        """Calculate overall confidence from component analyses."""
        performance_confidence = performance_result.confidence
        risk_confidence = risk_result.confidence
        
        return (performance_confidence + risk_confidence) / 2.0
        
    def _classify_health_status(self, health_score: float) -> str:
        """Classify market health status."""
        if health_score > 0.7:
            return 'excellent'
        elif health_score > 0.5:
            return 'good'
        elif health_score > 0.3:
            return 'fair'
        else:
            return 'poor'
            
    def _classify_opportunity_level(self, opportunity_score: float) -> str:
        """Classify trading opportunity level."""
        if opportunity_score > 0.7:
            return 'high'
        elif opportunity_score > 0.4:
            return 'medium'
        elif opportunity_score > 0.2:
            return 'low'
        else:
            return 'none'
            
    def _recommend_action(self, opportunity_score: float, sentiment: float) -> str:
        """Recommend trading action based on opportunity and sentiment."""
        if opportunity_score > 0.7 and sentiment > 0.6:
            return 'strong_buy'
        elif opportunity_score > 0.5 and sentiment > 0.5:
            return 'buy'
        elif opportunity_score < 0.2 or sentiment < 0.3:
            return 'sell'
        else:
            return 'hold'
            
    def _generate_risk_warnings(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on metrics."""
        warnings = []
        
        risk_score = risk_metrics.get('risk_score', 0.5)
        var_95 = risk_metrics.get('var_95', 0.0)
        current_drawdown = risk_metrics.get('current_drawdown', 0.0)
        
        if risk_score > 0.7:
            warnings.append("High risk score detected")
        if var_95 > 0.05:
            warnings.append("High Value at Risk (VaR)")
        if current_drawdown > 0.1:
            warnings.append("Significant current drawdown")
            
        return warnings 
