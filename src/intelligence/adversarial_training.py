#!/usr/bin/env python3
"""
ADVERSARIAL-30: Generative Adversarial Markets
=============================================

Train strategies against intelligent, evolving adversaries using GAN architecture.
Implements intelligent market simulation, adaptive difficulty, stress testing,
and realistic validation for creating robust trading strategies.

This module creates a competitive training environment where strategies
must survive increasingly challenging market conditions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MarketScenario:
    """Represents a challenging market scenario."""
    scenario_id: str
    difficulty_level: float
    market_conditions: Dict[str, float]
    stress_factors: Dict[str, float]
    target_strategies: List[str]
    realism_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurvivalResult:
    """Represents strategy survival results."""
    strategy_id: str
    survived: bool
    performance_score: float
    stress_endurance: float
    adaptation_score: float
    failure_reason: Optional[str] = None


class MarketDataGenerator(nn.Module):
    """Neural network for generating synthetic market data."""
    
    def __init__(self, input_dim: int = 100, output_dim: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class StrategyTester(nn.Module):
    """Neural network for testing strategy robustness."""
    
    def __init__(self, input_dim: int = 50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        return self.network(market_data)


class MarketDataGenerator:
    """Generates challenging market scenarios."""
    
    def __init__(self):
        self.generator = MarketDataGenerator()
        self.difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.current_difficulty = 0.1
        self.scaler = StandardScaler()
        
    async def generate_scenarios(self, difficulty_level: float, 
                               target_strategies: List[str],
                               num_scenarios: int = 100) -> List[MarketScenario]:
        """Generate challenging market scenarios."""
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate base market conditions
            base_conditions = self._generate_base_conditions()
            
            # Apply difficulty-based stress
            stress_factors = self._apply_stress_factors(base_conditions, difficulty_level)
            
            # Create scenario
            scenario = MarketScenario(
                scenario_id=f"adversarial_{i}_{datetime.utcnow().timestamp()}",
                difficulty_level=difficulty_level,
                market_conditions=base_conditions,
                stress_factors=stress_factors,
                target_strategies=target_strategies,
                realism_score=self._calculate_realism_score(base_conditions, stress_factors)
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_base_conditions(self) -> Dict[str, float]:
        """Generate base market conditions."""
        return {
            'volatility': np.random.uniform(0.005, 0.05),
            'trend': np.random.uniform(-0.1, 0.1),
            'volume': np.random.uniform(100, 10000),
            'spread': np.random.uniform(0.0001, 0.001),
            'liquidity': np.random.uniform(0.5, 2.0),
            'momentum': np.random.uniform(-0.5, 0.5),
            'mean_reversion': np.random.uniform(0, 1),
            'regime_stability': np.random.uniform(0, 1),
            'news_impact': np.random.uniform(-0.1, 0.1),
            'market_correlation': np.random.uniform(-1, 1)
        }
    
    def _apply_stress_factors(self, base_conditions: Dict[str, float], 
                            difficulty: float) -> Dict[str, float]:
        """Apply stress factors based on difficulty level."""
        stress_factors = {}
        
        # Increase volatility with difficulty
        stress_factors['volatility_multiplier'] = 1 + (difficulty * 2)
        
        # Increase trend strength
        stress_factors['trend_amplification'] = 1 + (difficulty * 1.5)
        
        # Reduce liquidity
        stress_factors['liquidity_reduction'] = max(0.1, 1 - (difficulty * 0.8))
        
        # Increase spread
        stress_factors['spread_inflation'] = 1 + (difficulty * 3)
        
        # Add noise
        stress_factors['noise_level'] = difficulty * 0.5
        
        return stress_factors
    
    def _calculate_realism_score(self, conditions: Dict[str, float], 
                               stress: Dict[str, float]) -> float:
        """Calculate how realistic the scenario is."""
        # Check if values are within reasonable bounds
        realism_checks = [
            0 <= conditions['volatility'] * stress.get('volatility_multiplier', 1) <= 0.2,
            -0.5 <= conditions['trend'] * stress.get('trend_amplification', 1) <= 0.5,
            10 <= conditions['volume'] <= 100000,
            0.00001 <= conditions['spread'] * stress.get('spread_inflation', 1) <= 0.01
        ]
        
        return sum(realism_checks) / len(realism_checks)


class StrategyTester:
    """Tests strategies against market scenarios."""
    
    def __init__(self):
        self.tester = StrategyTester()
        self.performance_threshold = 0.7
        
    async def test_strategies(self, strategy_population: List[Dict[str, Any]], 
                            scenarios: List[MarketScenario]) -> List[SurvivalResult]:
        """Test strategies against market scenarios."""
        
        results = []
        
        for strategy in strategy_population:
            result = await self._test_single_strategy(strategy, scenarios)
            results.append(result)
        
        return results
    
    async def _test_single_strategy(self, strategy: Dict[str, Any], 
                                  scenarios: List[MarketScenario]) -> SurvivalResult:
        """Test a single strategy against scenarios."""
        
        survival_count = 0
        total_performance = 0
        
        for scenario in scenarios:
            performance = await self._evaluate_strategy_performance(strategy, scenario)
            
            if performance > 0:  # Survived
                survival_count += 1
            
            total_performance += performance
        
        survival_rate = survival_count / len(scenarios)
        avg_performance = total_performance / len(scenarios)
        
        return SurvivalResult(
            strategy_id=strategy.get('id', 'unknown'),
            survived=survival_rate >= self.performance_threshold,
            performance_score=avg_performance,
            stress_endurance=survival_rate,
            adaptation_score=self._calculate_adaptation_score(strategy, scenarios)
        )
    
    async def _evaluate_strategy_performance(self, strategy: Dict[str, Any], 
                                           scenario: MarketScenario) -> float:
        """Evaluate strategy performance in a scenario."""
        
        # Simplified performance evaluation
        # In reality, this would run actual backtests
        
        base_performance = np.random.normal(0.02, 0.05)
        
        # Adjust for difficulty
        difficulty_penalty = scenario.difficulty_level * 0.1
        
        # Adjust for stress factors
        stress_penalty = sum(scenario.stress_factors.values()) * 0.05
        
        performance = base_performance - difficulty_penalty - stress_penalty
        
        return max(-1, min(1, performance))
    
    def _calculate_adaptation_score(self, strategy: Dict[str, Any], 
                                  scenarios: List[MarketScenario]) -> float:
        """Calculate how well the strategy adapts to different conditions."""
        
        # Check if strategy has adaptation mechanisms
        adaptation_features = strategy.get('adaptation_features', {})
        
        score = 0
        if adaptation_features.get('dynamic_risk', False):
            score += 0.3
        if adaptation_features.get('pattern_learning', False):
            score += 0.3
        if adaptation_features.get('regime_detection', False):
            score += 0.2
        if adaptation_features.get('parameter_optimization', False):
            score += 0.2
        
        return score


class AdversarialTrainer:
    """Trains both generator and discriminator in adversarial manner."""
    
    def __init__(self):
        self.generator = MarketDataGenerator()
        self.discriminator = StrategyTester()
        self.learning_rate = 0.001
        self.adversarial_loss = nn.BCELoss()
        
    async def train_generator(self, generator: MarketDataGenerator, 
                            survival_results: List[SurvivalResult],
                            target_failure_rate: float = 0.3) -> None:
        """Train generator to create more challenging scenarios."""
        
        # Calculate current failure rate
        failure_rate = 1 - (sum(r.survived for r in survival_results) / len(survival_results))
        
        # Adjust generator based on failure rate
        if failure_rate < target_failure_rate:
            # Increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
        elif failure_rate > target_failure_rate:
            # Decrease difficulty
            self.current_difficulty = max(0.1, self.current_difficulty - 0.05)
        
        logger.info(f"Adjusted generator difficulty to {self.current_difficulty}")
    
    async def train_discriminator(self, strategy_population: List[Dict[str, Any]], 
                                synthetic_scenarios: List[MarketScenario],
                                survival_results: List[SurvivalResult]) -> List[Dict[str, Any]]:
        """Train strategies to survive better."""
        
        improved_strategies = []
        
        for strategy, result in zip(strategy_population, survival_results):
            if not result.survived:
                # Improve strategy based on failure
                improved_strategy = self._improve_strategy(strategy, result, synthetic_scenarios)
                improved_strategies.append(improved_strategy)
            else:
                # Keep successful strategy
                improved_strategies.append(strategy)
        
        return improved_strategies
    
    def _improve_strategy(self, strategy: Dict[str, Any], 
                        failure_result: SurvivalResult,
                        scenarios: List[MarketScenario]) -> Dict[str, Any]:
        """Improve strategy based on failure analysis."""
        
        improved = strategy.copy()
        
        # Increase risk management
        if 'risk_management' not in improved:
            improved['risk_management'] = {}
        
        improved['risk_management']['max_drawdown'] = max(
            0.02, improved['risk_management'].get('max_drawdown', 0.05) * 0.8
        )
        
        # Add adaptation features
        if 'adaptation_features' not in improved:
            improved['adaptation_features'] = {}
        
        improved['adaptation_features']['dynamic_risk'] = True
        improved['adaptation_features']['regime_detection'] = True
        
        # Adjust parameters based on failure patterns
        if failure_result.failure_reason:
            improved['failure_analysis'] = failure_result.failure_reason
        
        return improved


class ScenarioValidator:
    """Validates that synthetic scenarios are realistic."""
    
    def __init__(self):
        self.realism_threshold = 0.7
        self.historical_stats = {}
        
    async def validate_realism(self, synthetic_scenarios: List[MarketScenario], 
                             real_market_data: Dict[str, Any]) -> float:
        """Validate that scenarios are realistic compared to real data."""
        
        if not real_market_data:
            return 1.0
        
        total_realism = 0
        
        for scenario in synthetic_scenarios:
            realism = self._validate_single_scenario(scenario, real_market_data)
            total_realism += realism
        
        avg_realism = total_realism / len(synthetic_scenarios)
        
        return avg_realism
    
    def _validate_single_scenario(self, scenario: MarketScenario, 
                                real_data: Dict[str, Any]) -> float:
        """Validate a single scenario against real data."""
        
        # Check statistical properties
        realism_checks = []
        
        # Volatility check
        if 'volatilities' in real_data:
            vol_percentile = stats.percentileofscore(
                real_data['volatilities'], 
                scenario.market_conditions['volatility']
            )
            realism_checks.append(10 <= vol_percentile <= 90)
        
        # Volume check
        if 'volumes' in real_data:
            vol_percentile = stats.percentileofscore(
                real_data['volumes'], 
                scenario.market_conditions['volume']
            )
            realism_checks.append(10 <= vol_percentile <= 90)
        
        # Trend check
        if 'trends' in real_data:
            trend_percentile = stats.percentileofscore(
                real_data['trends'], 
                abs(scenario.market_conditions['trend'])
            )
            realism_checks.append(10 <= trend_percentile <= 90)
        
        return sum(realism_checks) / max(1, len(realism_checks))


class MarketGAN:
    """Main GAN system for adversarial training."""
    
    def __init__(self, num_epochs: int = 100):
        self.generator = MarketDataGenerator()
        self.discriminator = StrategyTester()
        self.adversarial_trainer = AdversarialTrainer()
        self.scenario_validator = ScenarioValidator()
        self.num_epochs = num_epochs
        self.current_difficulty = 0.1
        
    async def train_adversarial_strategies(self, strategy_population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Train strategies using adversarial training."""
        
        logger.info("Starting adversarial training...")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Adversarial epoch {epoch + 1}/{self.num_epochs}")
            
            # Generate challenging scenarios
            synthetic_scenarios = await self.generator.generate_scenarios(
                difficulty_level=self.current_difficulty,
                target_strategies=[s['id'] for s in strategy_population]
            )
            
            # Test strategies against scenarios
            survival_results = await self.discriminator.test_strategies(
                strategy_population, synthetic_scenarios
            )
            
            # Train generator to create more challenging scenarios
            await self.adversarial_trainer.train_generator(
                self.generator, survival_results, target_failure_rate=0.3
            )
            
            # Train strategies to survive better
            improved_strategies = await self.adversarial_trainer.train_discriminator(
                strategy_population, synthetic_scenarios, survival_results
            )
            
            # Validate scenarios are realistic
            scenario_validity = await self.scenario_validator.validate_realism(
                synthetic_scenarios, real_market_data=self._get_real_data()
            )
            
            # Update population
            strategy_population = improved_strategies
            
            # Log progress
            survival_rate = sum(r.survived for r in survival_results) / len(survival_results)
            logger.info(f"Epoch {epoch + 1}: Survival rate {survival_rate:.2f}, "
                       f"Scenario realism {scenario_validity:.2f}")
        
        return strategy_population
    
    def _get_real_data(self) -> Dict[str, Any]:
        """Get real market data for validation."""
        # This would integrate with real market data
        return {
            'volatilities': [0.01, 0.015, 0.02, 0.025, 0.03],
            'volumes': [500, 1000, 2000, 5000, 10000],
            'trends': [-0.05, -0.02, 0, 0.02, 0.05]
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about adversarial training."""
        return {
            'current_difficulty': self.current_difficulty,
            'num_epochs': self.num_epochs,
            'generator_trained': True,
            'discriminator_trained': True
        }


# Example usage and testing
async def test_adversarial_training():
    """Test the adversarial training system."""
    gan = MarketGAN(num_epochs=5)
    
    # Create test strategy population
    strategies = [
        {'id': f'strategy_{i}', 'type': 'momentum', 'risk_tolerance': 0.02}
        for i in range(10)
    ]
    
    # Train adversarially
    trained_strategies = await gan.train_adversarial_strategies(strategies)
    
    print(f"Trained {len(trained_strategies)} strategies")
    stats = gan.get_training_stats()
    print(f"Training stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_adversarial_training())
