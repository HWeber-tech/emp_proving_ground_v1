"""
Multi-dimensional fitness evaluator combining all 7 fitness dimensions.

Provides comprehensive strategy evaluation across profit, survival, adaptability,
robustness, antifragility, efficiency, and innovation dimensions.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from .base_fitness import FitnessResult
from .profit_fitness import ProfitFitness
from .survival_fitness import SurvivalFitness
from .adaptability_fitness import AdaptabilityFitness
from .robustness_fitness import RobustnessFitness
from .antifragility_fitness import AntifragilityFitness
from .efficiency_fitness import EfficiencyFitness
from .innovation_fitness import InnovationFitness

logger = logging.getLogger(__name__)


@dataclass
class MultiDimensionalFitness:
    """Container for multi-dimensional fitness results."""
    profit: FitnessResult
    survival: FitnessResult
    adaptability: FitnessResult
    robustness: FitnessResult
    antifragility: FitnessResult
    efficiency: FitnessResult
    innovation: FitnessResult
    overall_score: float
    confidence: float
    regime_weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'profit': self.profit.to_dict(),
            'survival': self.survival.to_dict(),
            'adaptability': self.adaptability.to_dict(),
            'robustness': self.robustness.to_dict(),
            'antifragility': self.antifragility.to_dict(),
            'efficiency': self.efficiency.to_dict(),
            'innovation': self.innovation.to_dict(),
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'regime_weights': self.regime_weights
        }


class MultiDimensionalFitnessEvaluator:
    """Comprehensive fitness evaluator using all 7 dimensions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-dimensional fitness evaluator."""
        self.config = config or {}
        
        # Initialize all fitness dimensions
        self.fitness_dimensions = {
            'profit': ProfitFitness(self.config.get('profit', {})),
            'survival': SurvivalFitness(self.config.get('survival', {})),
            'adaptability': AdaptabilityFitness(self.config.get('adaptability', {})),
            'robustness': RobustnessFitness(self.config.get('robustness', {})),
            'antifragility': AntifragilityFitness(self.config.get('antifragility', {})),
            'efficiency': EfficiencyFitness(self.config.get('efficiency', {})),
            'innovation': InnovationFitness(self.config.get('innovation', {}))
        }
        
        # Default base weights
        self.base_weights = {
            'profit': 0.25,
            'survival': 0.20,
            'adaptability': 0.15,
            'robustness': 0.15,
            'antifragility': 0.10,
            'efficiency': 0.10,
            'innovation': 0.05
        }
        
        # Dynamic weight adjustment based on regime
        self.regime_weights = self.config.get('regime_weights', {})
        
        logger.info("MultiDimensionalFitnessEvaluator initialized with 7 dimensions")
    
    def evaluate_strategy(self, 
                         strategy_id: str,
                         strategy_performance: Dict[str, Any],
                         market_data: Dict[str, Any],
                         regime_data: Dict[str, Any]) -> MultiDimensionalFitness:
        """Evaluate strategy across all 7 fitness dimensions."""
        try:
            # Get current market regime
            current_regime = regime_data.get('current_regime', 'RANGING')
            
            # Calculate dynamic weights based on regime
            dynamic_weights = self._calculate_dynamic_weights(current_regime)
            
            # Evaluate each fitness dimension
            fitness_results = {}
            for name, dimension in self.fitness_dimensions.items():
                # Update dimension weight based on regime
                dimension.weight = dynamic_weights[name]
                
                # Calculate fitness
                result = dimension.calculate_fitness(
                    strategy_performance,
                    market_data,
                    regime_data
                )
                fitness_results[name] = result
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(fitness_results)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(fitness_results)
            
            return MultiDimensionalFitness(
                profit=fitness_results['profit'],
                survival=fitness_results['survival'],
                adaptability=fitness_results['adaptability'],
                robustness=fitness_results['robustness'],
                antifragility=fitness_results['antifragility'],
                efficiency=fitness_results['efficiency'],
                innovation=fitness_results['innovation'],
                overall_score=overall_score,
                confidence=overall_confidence,
                regime_weights=dynamic_weights
            )
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_id}: {e}")
            return self._create_error_result(str(e))
    
    def evaluate_population(self, 
                           population: List[Dict[str, Any]],
                           market_data: Dict[str, Any],
                           regime_data: Dict[str, Any]) -> List[MultiDimensionalFitness]:
        """Evaluate entire population of strategies."""
        results = []
        
        for strategy in population:
            strategy_id = strategy.get('id', 'unknown')
            performance = strategy.get('performance', {})
            
            result = self.evaluate_strategy(
                strategy_id,
                performance,
                market_data,
                regime_data
            )
            results.append(result)
        
        return results
    
    def get_fitness_summary(self, fitness: MultiDimensionalFitness) -> Dict[str, Any]:
        """Get human-readable summary of fitness evaluation."""
        return {
            'overall_score': f"{fitness.overall_score:.3f}",
            'confidence': f"{fitness.confidence:.3f}",
            'regime': fitness.regime_weights,
            'dimensions': {
                'profit': {
                    'score': f"{fitness.profit.score:.3f}",
                    'confidence': f"{fitness.profit.confidence:.3f}"
                },
                'survival': {
                    'score': f"{fitness.survival.score:.3f}",
                    'confidence': f"{fitness.survival.confidence:.3f}"
                },
                'adaptability': {
                    'score': f"{fitness.adaptability.score:.3f}",
                    'confidence': f"{fitness.adaptability.confidence:.3f}"
                },
                'robustness': {
                    'score': f"{fitness.robustness.score:.3f}",
                    'confidence': f"{fitness.robustness.confidence:.3f}"
                },
                'antifragility': {
                    'score': f"{fitness.antifragility.score:.3f}",
                    'confidence': f"{fitness.antifragility.confidence:.3f}"
                },
                'efficiency': {
                    'score': f"{fitness.efficiency.score:.3f}",
                    'confidence': f"{fitness.efficiency.confidence:.3f}"
                },
                'innovation': {
                    'score': f"{fitness.innovation.score:.3f}",
                    'confidence': f"{fitness.innovation.confidence:.3f}"
                }
            }
        }
    
    def _calculate_dynamic_weights(self, regime: str) -> Dict[str, float]:
        """Calculate dynamic weights based on market regime."""
        weights = self.base_weights.copy()
        
        # Apply regime-specific adjustments
        regime_adjustments = {
            'TRENDING_UP': {
                'profit': 1.3, 'survival': 0.8, 'adaptability': 0.9,
                'robustness': 0.8, 'antifragility': 0.7, 'efficiency': 1.1, 'innovation': 1.0
            },
            'TRENDING_DOWN': {
                'profit': 1.1, 'survival': 1.5, 'adaptability': 1.2,
                'robustness': 1.2, 'antifragility': 1.3, 'efficiency': 1.0, 'innovation': 1.1
            },
            'RANGING': {
                'profit': 0.9, 'survival': 1.0, 'adaptability': 1.3,
                'robustness': 1.4, 'antifragility': 1.0, 'efficiency': 1.3, 'innovation': 1.2
            },
            'VOLATILE': {
                'profit': 1.2, 'survival': 1.3, 'adaptability': 1.4,
                'robustness': 1.5, 'antifragility': 1.5, 'efficiency': 1.2, 'innovation': 1.3
            },
            'CRISIS': {
                'profit': 0.7, 'survival': 2.0, 'adaptability': 1.5,
                'robustness': 1.3, 'antifragility': 2.0, 'efficiency': 0.9, 'innovation': 1.4
            },
            'RECOVERY': {
                'profit': 1.4, 'survival': 1.2, 'adaptability': 1.1,
                'robustness': 1.1, 'antifragility': 1.2, 'efficiency': 1.1, 'innovation': 1.2
            },
            'LOW_VOLATILITY': {
                'profit': 1.0, 'survival': 0.9, 'adaptability': 0.8,
                'robustness': 1.0, 'antifragility': 0.8, 'efficiency': 1.2, 'innovation': 0.9
            },
            'HIGH_VOLATILITY': {
                'profit': 1.2, 'survival': 1.4, 'adaptability': 1.4,
                'robustness': 1.5, 'antifragility': 1.5, 'efficiency': 1.1, 'innovation': 1.3
            }
        }
        
        adjustments = regime_adjustments.get(regime, {})
        
        # Apply adjustments
        for dimension, adjustment in adjustments.items():
            if dimension in weights:
                weights[dimension] *= adjustment
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_overall_score(self, fitness_results: Dict[str, FitnessResult]) -> float:
        """Calculate weighted overall fitness score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, result in fitness_results.items():
            weighted_sum += result.score * result.weight
            total_weight += result.weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_overall_confidence(self, fitness_results: Dict[str, FitnessResult]) -> float:
        """Calculate overall confidence based on individual confidences."""
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for name, result in fitness_results.items():
            weighted_confidence += result.confidence * result.weight
            total_weight += result.weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_confidence / total_weight
    
    def _create_error_result(self, error_msg: str) -> MultiDimensionalFitness:
        """Create error result with default values."""
        error_result = FitnessResult(
            score=0.0,
            raw_value=0.0,
            confidence=0.0,
            metadata={'error': error_msg},
            weight=0.0
        )
        
        return MultiDimensionalFitness(
            profit=error_result,
            survival=error_result,
            adaptability=error_result,
            robustness=error_result,
            antifragility=error_result,
            efficiency=error_result,
            innovation=error_result,
            overall_score=0.0,
            confidence=0.0,
            regime_weights=self.base_weights
        )


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    evaluator = MultiDimensionalFitnessEvaluator()
    
    # Example strategy performance
    strategy_performance = {
        'total_return': 0.15,
        'returns': [0.01, -0.005, 0.02, -0.01, 0.015] * 50,
        'volatility': 0.12,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'total_trades': 250,
        'avg_slippage': 0.0005,
        'transaction_costs': 0.02,
        'regime_performance': {
            'TRENDING_UP': {'return': 0.20},
            'RANGING': {'return': 0.10},
            'VOLATILE': {'return': 0.12}
        }
    }
    
    market_data = {
        'current_volatility': 0.15,
        'market_regime': 'TRENDING_UP'
    }
    
    regime_data = {
        'current_regime': 'TRENDING_UP',
        'regime_history': ['TRENDING_UP', 'RANGING', 'VOLATILE']
    }
    
    # Evaluate strategy
    result = evaluator.evaluate_strategy(
        'test_strategy',
        strategy_performance,
        market_data,
        regime_data
    )
    
    # Print summary
    summary = evaluator.get_fitness_summary(result)
    print("Fitness Evaluation Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
