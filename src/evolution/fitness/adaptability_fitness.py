"""
Adaptability fitness dimension for strategy evaluation.

Measures performance consistency across different market regimes.
"""

import numpy as np
from typing import Dict, Any, List
from .base_fitness import BaseFitness, FitnessResult


class AdaptabilityFitness(BaseFitness):
    """Adaptability optimization through cross-regime performance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_regimes = self.config.get('min_regimes', 3)
        self.consistency_threshold = self.config.get('consistency_threshold', 0.7)
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate adaptability fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract regime performance data
            regime_performance = strategy_performance.get('regime_performance', {})
            if not regime_performance:
                return FitnessResult(
                    score=0.5,  # Neutral score if no regime data
                    raw_value=0.5,
                    confidence=0.3,
                    metadata={'error': 'No regime performance data'},
                    weight=self.weight
                )
            
            # Calculate adaptability metrics
            regimes = list(regime_performance.keys())
            performances = [regime_performance[r].get('return', 0.0) for r in regimes]
            
            if len(regimes) < self.min_regimes:
                return FitnessResult(
                    score=0.3,
                    raw_value=0.3,
                    confidence=0.5,
                    metadata={'error': f'Insufficient regimes: {len(regimes)}'},
                    weight=self.weight
                )
            
            # Performance consistency across regimes
            consistency_score = self._calculate_consistency_score(performances)
            
            # Regime transition handling
            transition_score = self._calculate_transition_score(strategy_performance)
            
            # Parameter stability
            stability_score = self._calculate_parameter_stability(strategy_performance)
            
            # Composite adaptability score
            composite_score = (
                consistency_score * 0.5 +
                transition_score * 0.3 +
                stability_score * 0.2
            )
            
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(regimes), self.min_regimes)
            
            metadata = {
                'consistency_score': consistency_score,
                'transition_score': transition_score,
                'stability_score': stability_score,
                'regimes_tested': len(regimes),
                'performance_std': np.std(performances) if performances else 0.0
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=consistency_score,
                confidence=confidence,
                metadata=metadata,
                weight=self.weight
            )
            
        except Exception as e:
            return FitnessResult(
                score=0.0,
                raw_value=0.0,
                confidence=0.0,
                metadata={'error': str(e)},
                weight=self.weight
            )
    
    def get_optimal_weight(self, market_regime: str) -> float:
        """Get optimal weight based on market regime."""
        weights = {
            'TRENDING_UP': 0.9,      # Less critical in stable trends
            'TRENDING_DOWN': 1.2,    # Important for adapting to downtrends
            'RANGING': 1.3,          # Critical for ranging markets
            'VOLATILE': 1.4,         # High importance in volatile conditions
            'CRISIS': 1.5,           # Maximum importance during crisis
            'RECOVERY': 1.1,         # Moderate importance during recovery
            'LOW_VOLATILITY': 0.8,   # Standard importance
            'HIGH_VOLATILITY': 1.4   # High importance in high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['regime_performance', 'transition_data', 'parameter_stability']
    
    def _calculate_consistency_score(self, performances: List[float]) -> float:
        """Calculate performance consistency across regimes."""
        if not performances:
            return 0.0
        
        # Normalize performances to 0-1 range
        min_perf, max_perf = min(performances), max(performances)
        if max_perf == min_perf:
            return 1.0
        
        normalized = [(p - min_perf) / (max_perf - min_perf) for p in performances]
        
        # Calculate coefficient of variation
        mean_norm = np.mean(normalized)
        std_norm = np.std(normalized)
        
        if mean_norm == 0:
            return 0.5
        
        cv = std_norm / mean_norm
        
        # Consistency score (lower CV = higher consistency)
        max_acceptable_cv = 2.0
        consistency_score = max(0.0, 1.0 - min(cv / max_acceptable_cv, 1.0))
        
        return consistency_score
    
    def _calculate_transition_score(self, strategy_performance: Dict[str, Any]) -> float:
        """Calculate regime transition handling score."""
        transition_data = strategy_performance.get('transition_data', {})
        if not transition_data:
            return 0.5
        
        # Score based on transition performance
        successful_transitions = transition_data.get('successful_transitions', 0)
        total_transitions = transition_data.get('total_transitions', 1)
        
        transition_success_rate = successful_transitions / total_transitions
        
        # Bonus for quick adaptation
        avg_adaptation_time = transition_data.get('avg_adaptation_time', 10)
        adaptation_bonus = max(0.0, 1.0 - avg_adaptation_time / 20)
        
        transition_score = (transition_success_rate * 0.7 + adaptation_bonus * 0.3)
        
        return transition_score
    
    def _calculate_parameter_stability(self, strategy_performance: Dict[str, Any]) -> float:
        """Calculate parameter stability score."""
        stability_data = strategy_performance.get('parameter_stability', {})
        if not stability_data:
            return 0.5
        
        # Score based on parameter variance
        param_variance = stability_data.get('parameter_variance', 0.0)
        max_variance = stability_data.get('max_variance', 1.0)
        
        if max_variance == 0:
            return 1.0
        
        stability_score = max(0.0, 1.0 - min(param_variance / max_variance, 1.0))
        
        return stability_score
