"""
Dynamic Adjustment

Real-time strategy parameter adjustment based on market conditions.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdjustmentRecommendation:
    """Strategy adjustment recommendation"""
    strategy_id: str
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    reason: str
    priority: str  # low, medium, high, critical


class DynamicAdjustment:
    """
    Dynamic Strategy Adjustment System
    
    Implements real-time parameter adjustment with:
    - Market regime detection
    - Performance-based adjustment
    - Risk-based adjustment
    - Adaptive parameter tuning
    """
    
    def __init__(self):
        self.adjustment_history: Dict[str, List[AdjustmentRecommendation]] = {}
        self.market_regime = "normal"
        
        logger.info("DynamicAdjustment initialized")
    
    def analyze_strategy_performance(self, strategy_id: str, 
                                   performance_metrics: Dict[str, Any],
                                   current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Analyze strategy performance and recommend adjustments"""
        
        recommendations = []
        
        # Performance-based adjustments
        if 'sharpe_ratio' in performance_metrics:
            sharpe = performance_metrics['sharpe_ratio']
            if sharpe < 0.5:
                recommendations.extend(self._recommend_risk_reduction(strategy_id, current_parameters))
            elif sharpe > 2.0:
                recommendations.extend(self._recommend_aggressive_adjustment(strategy_id, current_parameters))
        
        # Drawdown-based adjustments
        if 'max_drawdown' in performance_metrics:
            drawdown = performance_metrics['max_drawdown']
            if drawdown > 0.15:  # 15% drawdown
                recommendations.extend(self._recommend_drawdown_protection(strategy_id, current_parameters))
        
        # Win rate adjustments
        if 'win_rate' in performance_metrics:
            win_rate = performance_metrics['win_rate']
            if win_rate < 0.4:
                recommendations.extend(self._recommend_win_rate_improvement(strategy_id, current_parameters))
        
        # Market regime adjustments
        recommendations.extend(self._recommend_regime_adjustments(strategy_id, current_parameters))
        
        # Store recommendations
        if strategy_id not in self.adjustment_history:
            self.adjustment_history[strategy_id] = []
        
        self.adjustment_history[strategy_id].extend(recommendations)
        
        return recommendations
    
    def _recommend_risk_reduction(self, strategy_id: str, 
                                current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Recommend risk reduction adjustments"""
        recommendations = []
        
        # Reduce position size
        if 'risk_per_trade' in current_parameters:
            current_risk = current_parameters['risk_per_trade']
            if current_risk > 0.02:  # 2%
                recommendations.append(AdjustmentRecommendation(
                    strategy_id=strategy_id,
                    parameter='risk_per_trade',
                    current_value=current_risk,
                    recommended_value=current_risk * 0.7,  # Reduce by 30%
                    confidence=0.8,
                    reason="Poor performance - reduce risk exposure",
                    priority="high"
                ))
        
        # Tighten stop loss
        if 'stop_loss' in current_parameters:
            current_stop = current_parameters['stop_loss']
            if current_stop > 0.03:  # 3%
                recommendations.append(AdjustmentRecommendation(
                    strategy_id=strategy_id,
                    parameter='stop_loss',
                    current_value=current_stop,
                    recommended_value=current_stop * 0.8,  # Tighten by 20%
                    confidence=0.7,
                    reason="Reduce potential losses",
                    priority="medium"
                ))
        
        return recommendations
    
    def _recommend_aggressive_adjustment(self, strategy_id: str, 
                                       current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Recommend aggressive adjustments for high performance"""
        recommendations = []
        
        # Increase position size
        if 'risk_per_trade' in current_parameters:
            current_risk = current_parameters['risk_per_trade']
            if current_risk < 0.05:  # 5%
                recommendations.append(AdjustmentRecommendation(
                    strategy_id=strategy_id,
                    parameter='risk_per_trade',
                    current_value=current_risk,
                    recommended_value=min(current_risk * 1.3, 0.05),  # Increase by 30%
                    confidence=0.6,
                    reason="High performance - increase exposure",
                    priority="medium"
                ))
        
        return recommendations
    
    def _recommend_drawdown_protection(self, strategy_id: str, 
                                     current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Recommend drawdown protection adjustments"""
        recommendations = []
        
        # Emergency stop loss
        if 'stop_loss' in current_parameters:
            current_stop = current_parameters['stop_loss']
            recommendations.append(AdjustmentRecommendation(
                strategy_id=strategy_id,
                parameter='stop_loss',
                current_value=current_stop,
                recommended_value=min(current_stop * 0.5, 0.02),  # Tighten significantly
                confidence=0.9,
                reason="High drawdown - emergency protection",
                priority="critical"
            ))
        
        # Reduce position size
        if 'risk_per_trade' in current_parameters:
            current_risk = current_parameters['risk_per_trade']
            recommendations.append(AdjustmentRecommendation(
                strategy_id=strategy_id,
                parameter='risk_per_trade',
                current_value=current_risk,
                recommended_value=current_risk * 0.5,  # Reduce by 50%
                confidence=0.9,
                reason="High drawdown - reduce exposure",
                priority="critical"
            ))
        
        return recommendations
    
    def _recommend_win_rate_improvement(self, strategy_id: str, 
                                      current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Recommend win rate improvement adjustments"""
        recommendations = []
        
        # Adjust signal thresholds
        if 'signal_threshold' in current_parameters:
            current_threshold = current_parameters['signal_threshold']
            recommendations.append(AdjustmentRecommendation(
                strategy_id=strategy_id,
                parameter='signal_threshold',
                current_value=current_threshold,
                recommended_value=current_threshold * 1.2,  # Increase threshold
                confidence=0.7,
                reason="Low win rate - be more selective",
                priority="high"
            ))
        
        return recommendations
    
    def _recommend_regime_adjustments(self, strategy_id: str, 
                                    current_parameters: Dict[str, Any]) -> List[AdjustmentRecommendation]:
        """Recommend market regime-based adjustments"""
        recommendations = []
        
        # Adjust based on current market regime
        if self.market_regime == "volatile":
            # Reduce position size in volatile markets
            if 'risk_per_trade' in current_parameters:
                current_risk = current_parameters['risk_per_trade']
                recommendations.append(AdjustmentRecommendation(
                    strategy_id=strategy_id,
                    parameter='risk_per_trade',
                    current_value=current_risk,
                    recommended_value=current_risk * 0.8,
                    confidence=0.8,
                    reason="Volatile market - reduce exposure",
                    priority="high"
                ))
        
        elif self.market_regime == "trending":
            # Increase position size in trending markets
            if 'risk_per_trade' in current_parameters:
                current_risk = current_parameters['risk_per_trade']
                recommendations.append(AdjustmentRecommendation(
                    strategy_id=strategy_id,
                    parameter='risk_per_trade',
                    current_value=current_risk,
                    recommended_value=min(current_risk * 1.2, 0.05),
                    confidence=0.7,
                    reason="Trending market - increase exposure",
                    priority="medium"
                ))
        
        return recommendations
    
    def update_market_regime(self, regime: str) -> None:
        """Update current market regime"""
        self.market_regime = regime
        logger.info(f"Market regime updated to: {regime}")
    
    def get_adjustment_history(self, strategy_id: str) -> List[AdjustmentRecommendation]:
        """Get adjustment history for a strategy"""
        return self.adjustment_history.get(strategy_id, [])
    
    def apply_adjustment(self, strategy_id: str, 
                        parameter: str, 
                        new_value: Any) -> bool:
        """Apply an adjustment to a strategy"""
        try:
            # This would interface with the actual strategy to apply the adjustment
            logger.info(f"Applied adjustment to {strategy_id}: {parameter} = {new_value}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply adjustment: {e}")
            return False
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of all adjustments"""
        total_adjustments = sum(len(history) for history in self.adjustment_history.values())
        critical_adjustments = sum(
            len([r for r in history if r.priority == "critical"])
            for history in self.adjustment_history.values()
        )
        
        return {
            'total_adjustments': total_adjustments,
            'critical_adjustments': critical_adjustments,
            'market_regime': self.market_regime,
            'strategies_monitored': len(self.adjustment_history)
        } 