"""
Antifragility fitness dimension for strategy evaluation.

Measures performance improvement during stress and volatility.
"""

import numpy as np
from typing import Dict, Any
from .base_fitness import BaseFitness, FitnessResult


class AntifragilityFitness(BaseFitness):
    """Antifragility optimization through stress performance improvement."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.stress_threshold = self.config.get('stress_threshold', 2.0)  # 2 std dev
        self.volatility_threshold = self.config.get('volatility_threshold', 0.25)  # 25%
        
    def calculate_fitness(self, 
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> FitnessResult:
        """Calculate antifragility fitness score."""
        try:
            if not self.validate_inputs(strategy_performance):
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'Invalid inputs'},
                    weight=self.weight
                )
            
            # Extract antifragility metrics
            returns = strategy_performance.get('returns', [])
            stress_returns = strategy_performance.get('stress_returns', [])
            normal_returns = strategy_performance.get('normal_returns', [])
            volatility_harvesting = strategy_performance.get('volatility_harvesting', 0.0)
            
            if not returns:
                return FitnessResult(
                    score=0.0,
                    raw_value=0.0,
                    confidence=0.0,
                    metadata={'error': 'No returns data'},
                    weight=self.weight
                )
            
            # Stress vs normal performance
            stress_performance = self._calculate_stress_performance(stress_returns, normal_returns)
            
            # Volatility harvesting score
            volatility_score = self._calculate_volatility_harvesting(volatility_harvesting)
            
            # Crisis alpha generation
            crisis_alpha = self._calculate_crisis_alpha(strategy_performance)
            
            # Tail risk exploitation
            tail_exploitation = self._calculate_tail_exploitation(returns)
            
            # Composite antifragility score
            composite_score = (
                stress_performance * 0.35 +
                volatility_score * 0.25 +
                crisis_alpha * 0.25 +
                tail_exploitation * 0.15
            )
            
            final_score = max(0.0, min(1.0, composite_score))
            
            # Calculate confidence
            confidence = self.calculate_confidence(len(returns))
            
            metadata = {
                'stress_performance': stress_performance,
                'volatility_score': volatility_score,
                'crisis_alpha': crisis_alpha,
                'tail_exploitation': tail_exploitation,
                'stress_returns_count': len(stress_returns),
                'normal_returns_count': len(normal_returns)
            }
            
            return FitnessResult(
                score=final_score,
                raw_value=stress_performance,
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
            'TRENDING_UP': 0.7,      # Less critical in stable trends
            'TRENDING_DOWN': 1.3,    # Important for crisis alpha
            'RANGING': 1.0,          # Moderate importance
            'VOLATILE': 1.5,         # Maximum importance in volatility
            'CRISIS': 2.0,           # Maximum importance during crisis
            'RECOVERY': 1.2,         # Important for recovery gains
            'LOW_VOLATILITY': 0.8,   # Standard importance
            'HIGH_VOLATILITY': 1.5   # Maximum importance in high vol
        }
        return weights.get(market_regime, 1.0)
    
    def get_required_keys(self) -> list:
        """Return required data keys."""
        return ['returns', 'stress_returns', 'normal_returns', 'volatility_harvesting', 'crisis_alpha']
    
    def _calculate_stress_performance(self, stress_returns: list, normal_returns: list) -> float:
        """Calculate performance improvement during stress periods."""
        if not stress_returns or not normal_returns:
            return 0.5
        
        stress_mean = np.mean(stress_returns) if stress_returns else 0.0
        normal_mean = np.mean(normal_returns) if normal_returns else 0.0
        
        # Performance improvement during stress
        if normal_mean == 0:
            return 0.5
        
        improvement = (stress_mean - normal_mean) / abs(normal_mean)
        
        # Normalize improvement score
        improvement_score = max(0.0, min(1.0, (improvement + 1.0) / 2.0))
        
        return improvement_score
    
    def _calculate_volatility_harvesting(self, volatility_harvesting: float) -> float:
        """Calculate volatility harvesting effectiveness."""
        # Normalize volatility harvesting score
        max_harvesting = 0.5  # Maximum expected harvesting
        harvesting_score = max(0.0, min(1.0, volatility_harvesting / max_harvesting))
        
        return harvesting_score
    
    def _calculate_crisis_alpha(self, strategy_performance: Dict[str, Any]) -> float:
        """Calculate crisis alpha generation."""
        crisis_performance = strategy_performance.get('crisis_performance', {})
        if not crisis_performance:
            return 0.0
        
        crisis_return = crisis_performance.get('return', 0.0)
        market_return = crisis_performance.get('market_return', 0.0)
        
        # Crisis alpha = strategy return - market return
        alpha = crisis_return - market_return
        
        # Normalize alpha score
        max_alpha = 0.5  # Maximum expected alpha
        alpha_score = max(0.0, min(1.0, (alpha + max_alpha) / (2 * max_alpha)))
        
        return alpha_score
    
    def _calculate_tail_exploitation(self, returns: list) -> float:
        """Calculate tail risk exploitation."""
        if len(returns) < 10:
            return 0.0
        
        # Calculate tail statistics
        sorted_returns = sorted(returns)
        tail_size = max(1, len(returns) // 10)
        
        left_tail = sorted_returns[:tail_size]
        right_tail = sorted_returns[-tail_size:]
        
        # Tail exploitation score
        left_tail_mean = np.mean(left_tail)
        right_tail_mean = np.mean(right_tail)
        
        # Positive skew indicates tail exploitation
        tail_skew = (right_tail_mean + left_tail_mean) / 2.0
        
        # Normalize tail exploitation
        tail_score = max(0.0, min(1.0, (tail_skew + 0.1) / 0.2))
        
        return tail_score
