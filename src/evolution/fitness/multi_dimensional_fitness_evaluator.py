#!/usr/bin/env python3
"""
Multi-Dimensional Fitness Evaluator
=================================

This module implements the 7 fitness classes required for Phase 2:
1. Profit - Financial performance metrics
2. Survival - Risk management and drawdown control
3. Adaptability - Performance across market regimes
4. Robustness - Consistency under stress
5. Antifragility - Benefit from disorder
6. Efficiency - Resource utilization
7. Innovation - Novelty and uniqueness

Each class implements calculate_fitness() method returning 0-1 score.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ProfitFitness:
    """Profit-based fitness evaluation"""
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Calculate profit fitness score (0-1)"""
        try:
            total_return = performance_data.get('total_return', 0.0)
            sharpe_ratio = performance_data.get('sharpe_ratio', 0.0)
            
            # Normalize returns (0-1 scale)
            return_score = min(max(total_return / 0.5, 0.0), 1.0)  # 50% return = 1.0
            
            # Normalize Sharpe ratio (0-1 scale)
            sharpe_score = min(max(sharpe_ratio / 3.0, 0.0), 1.0)  # Sharpe 3.0 = 1.0
            
            # Combined profit fitness
            profit_fitness = (return_score * 0.6) + (sharpe_score * 0.4)
            
            return profit_fitness
            
        except Exception as e:
            logger.error(f"Error calculating profit fitness: {e}")
            return 0.0


class SurvivalFitness:
    """Survival-based fitness evaluation"""
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Calculate survival fitness score (0-1)"""
        try:
            max_drawdown = performance_data.get('max_drawdown', 1.0)
            win_rate = performance_data.get('win_rate', 0.0)
            
            # Survival based on drawdown control (lower drawdown = higher survival)
            drawdown_score = max(1.0 - (max_drawdown / 0.2), 0.0)  # 20% drawdown = 0.0
            
            # Survival based on win rate
            win_rate_score = min(win_rate / 0.6, 1.0)  # 60% win rate = 1.0
            
            # Combined survival fitness
            survival_fitness = (drawdown_score * 0.7) + (win_rate_score * 0.3)
            
            return survival_fitness
            
        except Exception as e:
            logger.error(f"Error calculating survival fitness: {e}")
            return 0.0


class AdaptabilityFitness:
    """Adaptability-based fitness evaluation"""
    
    def __init__(self):
        self.performance_history = {}
    
    def calculate_fitness(self, strategy_id: str, market_regimes: List[str], performance_data: Dict[str, Any]) -> float:
        """Calculate adaptability fitness score (0-1)"""
        try:
            # Count unique regimes the strategy has encountered
            unique_regimes = len(set(market_regimes))
            regime_diversity = min(unique_regimes / 5.0, 1.0)  # 5 regimes = 1.0
            
            # Performance consistency across regimes
            consistency_score = 0.8  # Placeholder
            
            # Combined adaptability fitness
            adaptability_fitness = (regime_diversity * 0.6) + (consistency_score * 0.4)
            
            return adaptability_fitness
            
        except Exception as e:
            logger.error(f"Error calculating adaptability fitness: {e}")
            return 0.0


class RobustnessFitness:
    """Robustness-based fitness evaluation"""
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Calculate robustness fitness score (0-1)"""
        try:
            # Robustness based on consistency across different market conditions
            volatility = performance_data.get('volatility', 0.0)
            max_drawdown = performance_data.get('max_drawdown', 0.0)
            
            # Lower volatility and drawdown indicate higher robustness
            volatility_score = max(1.0 - (volatility / 0.5), 0.0)  # 50% volatility = 0.0
            drawdown_score = max(1.0 - (max_drawdown / 0.3), 0.0)  # 30% drawdown = 0.0
            
            # Combined robustness fitness
            robustness_fitness = (volatility_score * 0.6) + (drawdown_score * 0.4)
            
            return robustness_fitness
            
        except Exception as e:
            logger.error(f"Error calculating robustness fitness: {e}")
            return 0.0


class AntifragilityFitness:
    """Antifragility-based fitness evaluation"""
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Calculate antifragility fitness score (0-1)"""
        try:
            # Antifragility based on performance during market stress
            stress_performance = performance_data.get('stress_performance', 0.0)
            crisis_alpha = performance_data.get('crisis_alpha', 0.0)
            
            # Higher performance during stress indicates antifragility
            stress_score = min(max(stress_performance / 0.2, 0.0), 1.0)  # 20% = 1.0
            crisis_score = min(max(crisis_alpha / 0.1, 0.0), 1.0)  # 10% = 1.0
            
            # Combined antifragility fitness
            antifragility_fitness = (stress_score * 0.7) + (crisis_score * 0.3)
            
            return antifragility_fitness
            
        except Exception as e:
            logger.error(f"Error calculating antifragility fitness: {e}")
            return 0.0


class EfficiencyFitness:
    """Efficiency-based fitness evaluation"""
    
    def calculate_fitness(self, performance_data: Dict[str, Any]) -> float:
        """Calculate efficiency fitness score (0-1)"""
        try:
            # Efficiency based on resource utilization
            trades_per_day = performance_data.get('trades_per_day', 0.0)
            avg_holding_period = performance_data.get('avg_holding_period', 1.0)
            
            # Higher efficiency with optimal trading frequency
            frequency_score = min(max(trades_per_day / 10.0, 0.0), 1.0)  # 10 trades/day = 1.0
            holding_score = min(max(1.0 / avg_holding_period, 0.0), 1.0)  # Shorter holding = higher efficiency
            
            # Combined efficiency fitness
            efficiency_fitness = (frequency_score * 0.6) + (holding_score * 0.4)
            
            return efficiency_fitness
            
        except Exception as e:
            logger.error(f"Error calculating efficiency fitness: {e}")
            return 0.0


class InnovationFitness:
    """Innovation-based fitness evaluation"""
    
    def calculate_fitness(self, strategy_id: str, performance_data: Dict[str, Any]) -> float:
        """Calculate innovation fitness score (0-1)"""
        try:
            # Innovation based on strategy uniqueness and novelty
            # Placeholder - would implement actual innovation scoring
            # Based on strategy complexity, uniqueness, and novel approaches
            innovation_score = 0.7  # Default innovation score
            
            return innovation_score
            
        except Exception as e:
            logger.error(f"Error calculating innovation fitness: {e}")
            return 0.0


class MultiDimensionalFitnessEvaluator:
    """Main evaluator that combines all 7 fitness dimensions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = {
            'profit': 0.25,
            'survival': 0.20,
            'adaptability': 0.15,
            'robustness': 0.15,
            'antifragility': 0.10,
            'efficiency': 0.10,
            'innovation': 0.05
        }
        
        # Initialize individual fitness evaluators
        self.profit_evaluator = ProfitFitness()
        self.survival_evaluator = SurvivalFitness()
        self.adaptability_evaluator = AdaptabilityFitness()
        self.robustness_evaluator = RobustnessFitness()
        self.antifragility_evaluator = AntifragilityFitness()
        self.efficiency_evaluator = EfficiencyFitness()
        self.innovation_evaluator = InnovationFitness()
        
        self.performance_history = {}
        
        logger.info("MultiDimensionalFitnessEvaluator initialized")
    
    def evaluate_strategy_fitness(
        self,
        strategy_id: str,
        performance_data: Dict[str, Any],
        market_regimes: List[str]
    ) -> Dict[str, float]:
        """Evaluate strategy fitness across all 7 dimensions"""
        try:
            # Calculate individual fitness dimensions
            profit_score = self.profit_evaluator.calculate_fitness(performance_data)
            survival_score = self.survival_evaluator.calculate_fitness(performance_data)
            adaptability_score = self.adaptability_evaluator.calculate_fitness(
                strategy_id, market_regimes, performance_data
            )
            robustness_score = self.robustness_evaluator.calculate_fitness(performance_data)
            antifragility_score = self.antifragility_evaluator.calculate_fitness(performance_data)
            efficiency_score = self.efficiency_evaluator.calculate_fitness(performance_data)
            innovation_score = self.innovation_evaluator.calculate_fitness(strategy_id, performance_data)
            
            # Calculate overall fitness with weights
            overall_fitness = (
                profit_score * self.weights['profit'] +
                survival_score * self.weights['survival'] +
                adaptability_score * self.weights['adaptability'] +
                robustness_score * self.weights['robustness'] +
                antifragility_score * self.weights['antifragility'] +
                efficiency_score * self.weights['efficiency'] +
                innovation_score * self.weights['innovation']
            )
            
            # Calculate confidence
            confidence = min(performance_data.get('sample_size', 0) / 30.0, 1.0)
            
            return {
                'profit': profit_score,
                'survival': survival_score,
                'adaptability': adaptability_score,
                'robustness': robustness_score,
                'antifragility': antifragility_score,
                'efficiency': efficiency_score,
                'innovation': innovation_score,
                'overall': overall_fitness,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error evaluating fitness for strategy {strategy_id}: {e}")
            return {
                'profit': 0.0,
                'survival': 0.0,
                'adaptability': 0.0,
                'robustness': 0.0,
                'antifragility': 0.0,
                'efficiency': 0.0,
                'innovation': 0.0,
                'overall': 0.0,
                'confidence': 0.0
            }


if __name__ == "__main__":
    """Test the multi-dimensional fitness evaluator"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    evaluator = MultiDimensionalFitnessEvaluator()
    
    # Test data
    performance_data = {
        'total_return': 0.25,
        'sharpe_ratio': 2.5,
        'max_drawdown': 0.08,
        'win_rate': 0.65,
        'volatility': 0.15,
        'trades_per_day': 5.0,
        'avg_holding_period': 2.0,
        'sample_size': 100,
        'stress_performance': 0.1,
        'crisis_alpha': 0.05
    }
    
    market_regimes = ["TRENDING_UP", "LOW_VOLATILITY"]
    
    fitness_scores = evaluator.evaluate_strategy_fitness("test_strategy", performance_data, market_regimes)
    
    print("Fitness Scores:")
    for dimension, score in fitness_scores.items():
        print(f"  {dimension}: {score:.3f}")
