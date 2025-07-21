"""
Innovation fitness dimension for strategy evaluation.

Measures strategy uniqueness and creative solution generation.
"""

import numpy as np
from typing import Dict, Any, List
from .base_fitness import BaseFitness, FitnessResult


class InnovationFitness(BaseFitness):
    """Innovation optimization through uniqueness and creative solutions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.uniqueness_threshold = self.config.get('uniqueness_threshold', 0.7)
        self.novelty_weight = self.config.get('novelty_weight', 0.5)
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate innovation fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract innovation metrics
            strategy_signature = strategy_performance.get('strategy_signature', {})
            parameter_novelty = strategy_performance.get('parameter_novelty', 0.0)
            behavior_patterns = strategy_performance.get('behavior_patterns', [])
            correlation_with_others = strategy_performance.get('correlation_with_others', [])
            
            # Strategy uniqueness score
            uniqueness_score = self._calculate_uniqueness(strategy_signature)
            
            # Parameter novelty score
            novelty_score = self._calculate_parameter_novelty(parameter_novelty)
            
            # Behavior pattern innovation
            pattern_score = self._calculate_pattern_innovation(behavior_patterns)
            
            # Correlation diversity (low correlation = high innovation)
            diversity_score = self._calculate_diversity_score(correlation_with_others)
            
            # Creative solution identification
            creative_score = self._calculate_creative_score(strategy_performance)
            
            # Composite innovation score
            composite_score = (
                uniqueness_score * 0.3 +
                novelty_score * 0.25 +
                pattern_score * 0.2 +
                diversity_score * 0.15 +
                creative_score * 0.1
            )
            
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(behavior_patterns) if behavior_patterns else 0)
            
            metadata = {
                'uniqueness_score': uniqueness_score,
                'novelty_score': novelty_score,
                'pattern_score': pattern_score,
                'diversity_score': diversity_score,
                'creative_score': creative_score,
                'strategy_signature': strategy_signature
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=uniqueness_score,
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
            'TRENDING_UP': 1.0,      # Standard importance
            'TRENDING_DOWN': 1.1,    # Important for new solutions
            'RANGING': 1.2,          # Critical for ranging innovation
            'VOLATILE': 1.3,         # High importance in volatility
            'CRISIS': 1.4,           # Important for crisis innovation
            'RECOVERY': 1.2,         # Important for recovery innovation
            'LOW_VOLATILITY': 0.9,   # Standard importance
            'HIGH_VOLATILITY': 1.3   # High importance in high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['strategy_signature', 'parameter_novelty', 'behavior_patterns', 'correlation_with_others']
    
    def _calculate_uniqueness(self, strategy_signature: Dict[str, Any]) -> float:
        """Calculate strategy uniqueness score."""
        if not strategy_signature:
            return 0.0
        
        # Uniqueness based on signature complexity
        complexity = strategy_signature.get('complexity', 0)
        max_complexity = 100
        
        # Normalize complexity
        complexity_score = min(1.0, complexity / max_complexity)
        
        # Uniqueness based on parameter count
        parameter_count = strategy_signature.get('parameter_count', 0)
        max_parameters = 50
        
        parameter_score = min(1.0, parameter_count / max_parameters)
        
        # Combined uniqueness
        uniqueness = (complexity_score + parameter_score) / 2.0
        
        return uniqueness
    
    def _calculate_parameter_novelty(self, parameter_novelty: float) -> float:
        """Calculate parameter novelty score."""
        # Normalize novelty score
        max_novelty = 1.0
        novelty_score = min(1.0, parameter_novelty / max_novelty)
        
        return novelty_score
    
    def _calculate_pattern_innovation(self, behavior_patterns: List[str]) -> float:
        """Calculate behavior pattern innovation score."""
        if not behavior_patterns:
            return 0.0
        
        # Innovation based on pattern diversity
        unique_patterns = len(set(behavior_patterns))
        total_patterns = len(behavior_patterns)
        
        diversity = unique_patterns / total_patterns if total_patterns > 0 else 0.0
        
        # Innovation based on pattern complexity
        complex_patterns = sum(1 for p in behavior_patterns if len(p) > 10)
        complexity_ratio = complex_patterns / total_patterns if total_patterns > 0 else 0.0
        
        # Combined pattern innovation
        pattern_score = (diversity + complexity_ratio) / 2.0
        
        return pattern_score
    
    def _calculate_diversity_score(self, correlations: List[float]) -> float:
        """Calculate correlation diversity score."""
        if not correlations:
            return 1.0
        
        # Average correlation
        avg_correlation = np.mean([abs(c) for c in correlations])
        
        # Diversity score (lower correlation = higher diversity)
        diversity_score = max(0.0, 1.0 - avg_correlation)
        
        return diversity_score
    
    def _calculate_creative_score(self, strategy_performance: Dict[str, Any]) -> float:
        """Calculate creative solution identification score."""
        creative_indicators = strategy_performance.get('creative_indicators', {})
        
        # Score based on creative indicators
        adaptive_behavior = creative_indicators.get('adaptive_behavior', 0.0)
        emergent_patterns = creative_indicators.get('emergent_patterns', 0.0)
        novel_solutions = creative_indicators.get('novel_solutions', 0.0)
        
        # Combined creative score
        creative_score = (adaptive_behavior + emergent_patterns + novel_solutions) / 3.0
        
        return min(1.0, creative_score)
