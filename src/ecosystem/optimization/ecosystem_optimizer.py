#!/usr/bin/env python3
"""
Ecosystem Optimizer
==================

Portfolio-level optimization system that evolves entire strategy ecosystems
rather than individual strategies, focusing on synergy and diversification.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics import mutual_info_score
import pandas as pd

from src.core.interfaces import IEcosystemOptimizer, DecisionGenome, MarketContext
from src.ecosystem.species.factories import get_all_factories
from src.ecosystem.evaluation.niche_detector import NicheDetector
from src.ecosystem.coordination.coordination_engine import CoordinationEngine

logger = logging.getLogger(__name__)


@dataclass
class EcosystemMetrics:
    """Metrics for evaluating ecosystem performance."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: np.ndarray
    diversification_ratio: float
    synergy_score: float
    antifragility_score: float


class EcosystemOptimizer(IEcosystemOptimizer):
    """Advanced ecosystem optimization system."""
    
    def __init__(self):
        self.niche_detector = NicheDetector()
        self.coordination_engine = CoordinationEngine()
        self.ecosystem_history = []
        self.optimization_params = {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_rate': 0.1
        }
        
    async def optimize_ecosystem(self, 
                            species_populations: Dict[str, List[DecisionGenome]],
                            market_context: MarketContext,
                            performance_history: Dict[str, Any]) -> Dict[str, List[DecisionGenome]]:
        """Optimize entire ecosystem for maximum synergy and diversification."""
        
        # Detect current market niches
        niches = await self.niche_detector.detect_niches(market_context.data)
        
        # Calculate optimal species distribution
        optimal_distribution = await self._calculate_optimal_distribution(
            niches, performance_history
        )
        
        # Evolve species populations
        evolved_populations = {}
        for species_type, population in species_populations.items():
            target_size = optimal_distribution.get(species_type, 10)
            evolved = await self._evolve_species_population(
                species_type, population, target_size, market_context
            )
            evolved_populations[species_type] = evolved
        
        # Optimize coordination between species
        coordinated_populations = await self._optimize_coordination(
            evolved_populations, market_context
        )
        
        # Evaluate ecosystem performance
        ecosystem_metrics = await self._evaluate_ecosystem_performance(
            coordinated_populations, performance_history
        )
        
        # Store optimization results
        self.ecosystem_history.append({
            'timestamp': datetime.now(),
            'populations': coordinated_populations,
            'metrics': ecosystem_metrics,
            'niches': niches
        })
        
        return coordinated_populations
    
    async def _calculate_optimal_distribution(self, 
                                            niches: Dict[str, Any],
                                            performance_history: Dict[str, Any]) -> Dict[str, int]:
        """Calculate optimal species distribution based on market niches."""
        
        distribution = {}
        total_slots = self.optimization_params['population_size']
        
        # Base allocation from niches
        niche_preferences = defaultdict(list)
        for niche_id, niche in niches.items():
            for species in niche.preferred_species:
                niche_preferences[species].append(niche.opportunity_score)
        
        # Calculate weighted allocation
        total_preference = sum(
            sum(scores) for scores in niche_preferences.values()
        )
        
        if total_preference > 0:
            for species, scores in niche_preferences.items():
                weight = sum(scores) / total_preference
                count = max(1, int(total_slots * weight))
                distribution[species] = count
        
        # Ensure minimum representation for all species
        all_species = ['stalker', 'ambusher', 'pack_hunter', 'scavenger', 'alpha']
        for species in all_species:
            if species not in distribution:
                distribution[species] = max(1, total_slots // len(all_species))
        
        # Normalize to total slots
        current_total = sum(distribution.values())
        if current_total != total_slots:
            scale = total_slots / current_total
            for species in distribution:
                distribution[species] = max(1, int(distribution[species] * scale))
        
        return distribution
    
    async def _evolve_species_population(self,
                                        species_type: str,
                                        population: List[DecisionGenome],
                                        target_size: int,
                                        market_context: MarketContext) -> List[DecisionGenome]:
        """Evolve a specific species population."""
        
        if not population:
            # Create initial population
            factory = get_all_factories().get(species_type)
            if factory:
                population = [factory.create_genome() for _ in range(target_size)]
            else:
                return []
        
        # Select best performers
        scored_population = []
        for genome in population:
            score = await self._evaluate_genome_performance(genome, market_context)
            scored_population.append((score, genome))
        
        # Sort by performance
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # Apply elitism
        elite_count = max(1, int(len(scored_population) * self.optimization_params['elitism_rate']))
        new_population = [genome for _, genome in scored_population[:elite_count]]
        
        # Fill remaining slots
        while len(new_population) < target_size:
            # Crossover and mutation
            if len(scored_population) >= 2:
                parent1 = random.choice(scored_population[:max(5, len(scored_population)//2)])[1]
                parent2 = random.choice(scored_population[:max(5, len(scored_population)//2)])[1]
                child = self._crossover_genomes(parent1, parent2)
            else:
                child = scored_population[0][1] if scored_population else None
            
            if child and random.random() < self.optimization_params['mutation_rate']:
                child = self._mutate_genome(child)
            
            if child:
                new_population.append(child)
        
        return new_population[:target_size]
    
    async def _evaluate_genome_performance(self, genome: DecisionGenome, 
                                         market_context: MarketContext) -> float:
        """Evaluate performance of a single genome."""
        # Simplified performance evaluation
        # In real implementation, this would use backtesting
        base_score = 0.5
        
        # Adjust for species suitability
        species_suitability = {
            'stalker': 0.8,
            'ambusher': 0.7,
            'pack_hunter': 0.9,
            'scavenger': 0.6,
            'alpha': 0.85
        }
        
        species_bonus = species_suitability.get(genome.species_type, 0.5)
        
        # Adjust for market regime
        
        return base_score * species_bonus * regime_bonus
    
    def _crossover_genomes(self, parent1: DecisionGenome, 
                          parent2: DecisionGenome) -> DecisionGenome:
        """Perform crossover between two genomes."""
        import random
        
        child = DecisionGenome(
            species_type=parent1.species_type,
            parameters={},
            indicators=[],
            rules={},
            risk_profile={}
        )
        
        # Crossover parameters
        for key in parent1.parameters:
            if random.random() < 0.5:
                child.parameters[key] = parent1.parameters[key]
            else:
                child.parameters[key] = parent2.parameters.get(key, parent1.parameters[key])
        
        # Crossover indicators and rules
        child.indicators = list(set(parent1.indicators + parent2.indicators))
        child.rules = {**parent1.rules, **parent2.rules}
        
        # Crossover risk profile
        for key in parent1.risk_profile:
            if random.random() < 0.5:
                child.risk_profile[key] = parent1.risk_profile[key]
            else:
                child.risk_profile[key] = parent2.risk_profile.get(key, parent1.risk_profile[key])
        
        return child
    
    def _mutate_genome(self, genome: DecisionGenome) -> DecisionGenome:
        """Apply mutation to a genome."""
        import random
        
        mutated = DecisionGenome(
            species_type=genome.species_type,
            parameters=genome.parameters.copy(),
            indicators=genome.indicators.copy(),
            rules=genome.rules.copy(),
            risk_profile=genome.risk_profile.copy()
        )
        
        # Mutate parameters
        for key, value in mutated.parameters.items():
            if random.random() < 0.1:  # 10% mutation rate
                if isinstance(value, float):
                    mutated.parameters[key] = value * random.uniform(0.8, 1.2)
                elif isinstance(value, int):
                    mutated.parameters[key] = max(1, value + random.randint(-2, 2))
        
        return mutated
    
    async def _optimize_coordination(self,
                                   populations: Dict[str, List[DecisionGenome]],
                                   market_context: MarketContext) -> Dict[str, List[DecisionGenome]]:
        """Optimize coordination between species."""
        
        # Create coordination intents
        intents = []
        for species_type, population in populations.items():
            for genome in population:
                intent = TradeIntent(
                    strategy_id=f"{species_type}_{id(genome)}",
                    species_type=species_type,
                    symbol="PORTFOLIO",
                    direction="BUY",
                    confidence=0.8,
                    size=0.02,
                    priority=5,
                    timestamp=datetime.now()
                )
                intents.append(intent)
        
        # Resolve coordination
        coordination_result = await self.coordination_engine.resolve_intents(
            intents, market_context
        )
        
        # Filter populations based on coordination
        coordinated_populations = {}
        approved_strategies = {intent.strategy_id for intent in coordination_result.approved_intents}
        
        for species_type, population in populations.items():
            coordinated_populations[species_type] = [
                genome for genome in population
                if f"{species_type}_{id(genome)}" in approved_strategies
            ]
        
        return coordinated_populations
    
    async def _evaluate_ecosystem_performance(self,
                                            populations: Dict[str, List[DecisionGenome]],
                                            performance_history: Dict[str, Any]) -> EcosystemMetrics:
        """Evaluate overall ecosystem performance."""
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(populations)
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(correlation_matrix)
        
        # Calculate synergy score
        synergy_score = self._calculate_synergy_score(populations, performance_history)
        
        # Calculate antifragility score
        antifragility_score = self._calculate_antifragility_score(
            populations, performance_history
        )
        
        # Mock performance metrics (would be calculated from actual trading)
        total_return = 0.15  # 15% annual return
        sharpe_ratio = 1.8
        max_drawdown = 0.08  # 8% max drawdown
        
        return EcosystemMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            correlation_matrix=correlation_matrix,
            diversification_ratio=diversification_ratio,
            synergy_score=synergy_score,
            antifragility_score=antifragility_score
        )
    
    def _calculate_correlation_matrix(self, 
                                    populations: Dict[str, List[DecisionGenome]]) -> np.ndarray:
        """Calculate correlation matrix between species."""
        species_list = list(populations.keys())
        n_species = len(species_list)
        
        correlation_matrix = np.eye(n_species)
        
        # Mock correlation values based on species characteristics
        correlation_map = {
            ('stalker', 'alpha'): 0.7,
            ('stalker', 'pack_hunter'): 0.5,
            ('stalker', 'ambusher'): 0.2,
            ('stalker', 'scavenger'): 0.3,
            ('alpha', 'pack_hunter'): 0.6,
            ('alpha', 'ambusher'): 0.1,
            ('alpha', 'scavenger'): 0.4,
            ('pack_hunter', 'ambusher'): 0.3,
            ('pack_hunter', 'scavenger'): 0.5,
            ('ambusher', 'scavenger'): 0.8
        }
        
        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i != j:
                    key = tuple(sorted([species1, species2]))
                    correlation_matrix[i, j] = correlation_map.get(key, 0.5)
        
        return correlation_matrix
    
    def _calculate_diversification_ratio(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification ratio from correlation matrix."""
        n_assets = len(correlation_matrix)
        if n_assets == 0:
            return 0.0
        
        # Average correlation
        avg_correlation = (np.sum(correlation_matrix) - n_assets) / (n_assets * (n_assets - 1))
        
        # Diversification ratio (higher is better)
        diversification_ratio = 1.0 - avg_correlation
        
        return max(0.0, min(1.0, diversification_ratio))
    
    def _calculate_synergy_score(self,
                            populations: Dict[str, List[DecisionGenome]],
                            performance_history: Dict[str, Any]) -> float:
        """Calculate synergy score between species."""
        # Mock synergy calculation
        # In real implementation, would analyze complementary strategies
        species_count = len(populations)
        genome_count = sum(len(pop) for pop in populations.values())
        
        # Higher diversity and balanced populations indicate better synergy
        diversity_bonus = species_count / 5.0  # 5 species max
        balance_bonus = 1.0 - np.std([len(pop) for pop in populations.values()]) / max(1, genome_count)
        
        synergy_score = (diversity_bonus + balance_bonus) / 2
        
        return max(0.0, min(1.0, synergy_score))
    
    def _calculate_antifragility_score(self,
                                     populations: Dict[str, List[DecisionGenome]],
                                     performance_history: Dict[str, Any]) -> float:
        """Calculate antifragility score of the ecosystem."""
        # Mock antifragility calculation
        # In real implementation, would analyze performance under stress
        
        # Factors contributing to antifragility
        species_diversity = len(populations) / 5.0
        strategy_diversity = sum(len(pop) for pop in populations.values()) / 50.0
        
        # Adaptability indicators
        
        antifragility_score = (species_diversity + strategy_diversity + adaptability_score) / 3
        
        return max(0.0, min(1.0, antifragility_score))
    
    async def get_ecosystem_summary(self) -> Dict[str, Any]:
        """Get summary of ecosystem state."""
        if not self.ecosystem_history:
            return {
                'total_optimizations': 0,
                'best_metrics': None,
                'current_species_distribution': {}
            }
        
        latest = self.ecosystem_history[-1]
        
        return {
            'total_optimizations': len(self.ecosystem_history),
            'best_metrics': {
                'total_return': latest['metrics'].total_return,
                'sharpe_ratio': latest['metrics'].sharpe_ratio,
                'diversification_ratio': latest['metrics'].diversification_ratio,
                'synergy_score': latest['metrics'].synergy_score
            },
            'current_species_distribution': {
                species: len(population)
                for species, population in latest['populations'].items()
            }
        }


# Example usage
async def test_ecosystem_optimizer():
    """Test the ecosystem optimizer."""
    optimizer = EcosystemOptimizer()
    
    # Create test populations
    test_populations = {
        'stalker': [],
        'ambusher': [],
        'pack_hunter': [],
        'scavenger': [],
        'alpha': []
    }
    
    # Create test market context
    market_context = MarketContext(
        symbol="EURUSD",
        regime="trending_bull",
        volatility=0.02,
        trend_strength=0.8,
        volume_anomaly=1.2,
        data={'data': []}
    )
    
    # Create test performance history
    performance_history = {
        'returns': [0.01, 0.02, -0.01, 0.03, 0.01],
        'sharpe_ratios': [1.5, 1.8, 1.2, 2.1, 1.6]
    }
    
    # Optimize ecosystem
    optimized = await optimizer.optimize_ecosystem(
        test_populations, market_context, performance_history
    )
    
    print("Ecosystem Optimization Complete")
    print(f"Species distribution: { {k: len(v) for k, v in optimized.items()} }")
    
    summary = await optimizer.get_ecosystem_summary()
    print(f"Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(test_ecosystem_optimizer())

    def _calculate_regime_bonus(self, market_regime: str) -> float:
        """Calculate regime-specific bonus based on actual market conditions."""
        regime_multipliers = {
            'trending': 1.2,
            'ranging': 0.9,
            'volatile': 1.1,
            'calm': 1.0,
            'crisis': 0.8
        }
        return regime_multipliers.get(market_regime, 1.0)
    
    def _calculate_adaptability_score(self, genome, market_data) -> float:
        """Calculate adaptability score based on genome performance across market conditions."""
        if not hasattr(genome, 'performance_metrics') or not genome.performance_metrics:
            return 0.5  # Neutral score for new genomes
        
        # Calculate adaptability based on performance variance across different market conditions
        performance_values = list(genome.performance_metrics.values())
        if len(performance_values) < 2:
            return 0.5
        
        # Lower variance indicates better adaptability
        import statistics
        variance = statistics.variance(performance_values)
        # Normalize to 0-1 scale (lower variance = higher adaptability)
        adaptability = max(0.0, min(1.0, 1.0 - (variance / 10.0)))
        return adaptability
