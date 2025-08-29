#!/usr/bin/env python3
"""
Ecosystem Optimizer
==================

Portfolio-level optimization system that evolves entire strategy ecosystems
rather than individual strategies, focusing on synergy and diversification.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from src.core.interfaces import IEcosystemOptimizer, MarketContext, TradeIntent
from src.core.types import JSONObject
from src.ecosystem.coordination.coordination_engine import CoordinationEngine
from src.ecosystem.evaluation.niche_detector import MarketNiche, NicheDetector
from src.ecosystem.species.factories import get_all_factories
from src.genome.models.adapters import from_legacy as adapt_to_canonical
from src.genome.models.genome import DecisionGenome as CanonDecisionGenome
from src.genome.models.genome import mutate as genome_mutate

logger = logging.getLogger(__name__)

__all__ = ["EcosystemOptimizer", "EcosystemMetrics", "EcosystemSummary"]

if TYPE_CHECKING:
    from src.core.interfaces import DecisionGenome as DecisionGenome


@dataclass
class EcosystemMetrics:
    """Metrics for evaluating ecosystem performance."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: NDArray[np.float64]
    diversification_ratio: float
    synergy_score: float
    antifragility_score: float


class MetricsSummary(TypedDict):
    total_return: float
    sharpe_ratio: float
    diversification_ratio: float
    synergy_score: float


class EcosystemSummary(TypedDict):
    total_optimizations: int
    best_metrics: Optional[MetricsSummary]
    current_species_distribution: Dict[str, int]


class EcosystemOptimizer(IEcosystemOptimizer):
    """Advanced ecosystem optimization system."""

    def __init__(self) -> None:
        self.niche_detector = NicheDetector()
        self.coordination_engine = CoordinationEngine()
        self.ecosystem_history: List[Dict[str, object]] = []
        self.optimization_params: Dict[str, float] = {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.1,
        }

    async def optimize_ecosystem(
        self,
        species_populations: Mapping[str, Sequence["DecisionGenome"]],
        market_context: MarketContext,
        performance_history: JSONObject,
    ) -> Mapping[str, Sequence["DecisionGenome"]]:
        """Optimize entire ecosystem for maximum synergy and diversification."""

        # Normalize/convert input populations to canonical genomes
        canonical_populations: Dict[str, List[CanonDecisionGenome]] = {}
        for species_type, population in species_populations.items():
            canonical_populations[species_type] = [
                self._ensure_canonical(g) for g in list(population)
            ]

        # Detect current market niches
        market_data_jo: JSONObject = (
            market_context.data if isinstance(market_context.data, dict) else {"data": []}
        )
        niches = await self.niche_detector.detect_niches(market_data_jo)

        # Calculate optimal species distribution
        optimal_distribution = await self._calculate_optimal_distribution(
            niches, performance_history
        )

        # Evolve species populations
        evolved_populations: Dict[str, List[CanonDecisionGenome]] = {}
        for species_type, population in canonical_populations.items():
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
        self.ecosystem_history.append(
            {
                "timestamp": datetime.now(),
                "populations": coordinated_populations,
                "metrics": ecosystem_metrics,
                "niches": niches,
            }
        )

        # Narrow to the interface type expected by IEcosystemOptimizer while preserving runtime object
        return cast(Mapping[str, Sequence["DecisionGenome"]], coordinated_populations)

    async def _calculate_optimal_distribution(
        self, niches: Dict[str, MarketNiche], performance_history: JSONObject
    ) -> Dict[str, int]:
        """Calculate optimal species distribution based on market niches."""

        distribution: Dict[str, int] = {}
        total_slots = int(self.optimization_params["population_size"])

        # Base allocation from niches
        niche_preferences: DefaultDict[str, List[float]] = defaultdict(list)
        for _niche_id, niche in niches.items():
            for species in niche.preferred_species:
                niche_preferences[species].append(float(niche.opportunity_score))

        # Calculate weighted allocation
        total_preference = sum(sum(scores) for scores in niche_preferences.values())

        if total_preference > 0:
            for species, scores in niche_preferences.items():
                weight = sum(scores) / total_preference
                count = max(1, int(total_slots * weight))
                distribution[species] = count

        # Ensure minimum representation for all species
        all_species = ["stalker", "ambusher", "pack_hunter", "scavenger", "alpha"]
        for species in all_species:
            if species not in distribution:
                distribution[species] = max(1, total_slots // len(all_species))

        # Normalize to total slots
        current_total = sum(distribution.values())
        if current_total != total_slots and current_total > 0:
            scale = total_slots / current_total
            for species in list(distribution.keys()):
                distribution[species] = max(1, int(distribution[species] * scale))

        return distribution

    async def _evolve_species_population(
        self,
        species_type: str,
        population: List[CanonDecisionGenome],
        target_size: int,
        market_context: MarketContext,
    ) -> List[CanonDecisionGenome]:
        """Evolve a specific species population."""

        # Ensure canonical genomes
        population = [self._ensure_canonical(g) for g in population]

        if not population:
            # Create initial population using species factory and adapt to canonical
            factory = get_all_factories().get(species_type)
            if factory:
                legacy_population = [factory.create_genome() for _ in range(target_size)]
                population = [adapt_to_canonical(g) for g in legacy_population]
            else:
                return []

        # Select best performers
        scored_population: List[Tuple[float, CanonDecisionGenome]] = []
        for genome in population:
            score = await self._evaluate_genome_performance(genome, market_context)
            scored_population.append((score, genome))

        # Sort by performance
        scored_population.sort(key=lambda x: x[0], reverse=True)

        # Apply elitism
        elite_count = max(
            1, int(len(scored_population) * float(self.optimization_params["elitism_rate"]))
        )
        new_population: List[CanonDecisionGenome] = [
            genome for _, genome in scored_population[:elite_count]
        ]

        # Fill remaining slots
        child: Optional[CanonDecisionGenome] = None
        while len(new_population) < target_size and scored_population:
            # Crossover and mutation
            if len(scored_population) >= 2:
                parent1 = random.choice(scored_population[: max(5, len(scored_population) // 2)])[1]
                parent2 = random.choice(scored_population[: max(5, len(scored_population) // 2)])[1]
                child = self._crossover_genomes(parent1, parent2)
            else:
                child = scored_population[0][1] if scored_population else None

            if child is not None and random.random() < float(
                self.optimization_params["mutation_rate"]
            ):
                child = self._mutate_genome(child)

            if child is not None:
                new_population.append(child)

        return new_population[:target_size]

    async def _evaluate_genome_performance(
        self, genome: CanonDecisionGenome, market_context: MarketContext
    ) -> float:
        """Evaluate performance of a single genome."""
        # Simplified performance evaluation
        # In real implementation, this would use backtesting
        base_score = 0.5

        # Adjust for species suitability
        species_suitability: Dict[str, float] = {
            "stalker": 0.8,
            "ambusher": 0.7,
            "pack_hunter": 0.9,
            "scavenger": 0.6,
            "alpha": 0.85,
        }

        species = (genome.species_type or "").lower()
        species_bonus = species_suitability.get(species, 0.5)

        # Adjust for market regime
        regime_bonus = 1.0
        try:
            regime_str = str(getattr(market_context, "regime", "")).lower()
            if "trend" in regime_str:
                regime_bonus = 1.1
            elif "volatile" in regime_str or "crisis" in regime_str:
                regime_bonus = 0.9
        except Exception:
            regime_bonus = 1.0

        return float(base_score * species_bonus * regime_bonus)

    def _crossover_genomes(
        self, parent1: CanonDecisionGenome, parent2: CanonDecisionGenome
    ) -> CanonDecisionGenome:
        """Perform crossover between two genomes (canonical model)."""
        child_params: Dict[str, float] = {}

        # Union of keys from both parents
        all_keys = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        for key in all_keys:
            if random.random() < 0.5:
                val = parent1.parameters.get(key, parent2.parameters.get(key, 0.0))
            else:
                val = parent2.parameters.get(key, parent1.parameters.get(key, 0.0))
            try:
                child_params[str(key)] = float(val)
            except Exception:
                # Skip non-coercible values
                continue

        species_type = parent1.species_type or parent2.species_type
        generation = max(int(parent1.generation), int(parent2.generation))
        parent_ids = [parent1.id, parent2.id]

        # Use canonical constructor with coercions via from_dict
        child = CanonDecisionGenome.from_dict(
            {
                "parameters": child_params,
                "species_type": species_type,
                "generation": generation,
                "parent_ids": parent_ids,
            }
        )
        return child

    def _mutate_genome(self, genome: CanonDecisionGenome) -> CanonDecisionGenome:
        """Apply mutation to a genome (canonical model)."""
        new_params: Dict[str, float] = {}
        for key, value in genome.parameters.items():
            if random.random() < 0.1:
                try:
                    # Apply a mild random multiplicative perturbation
                    mutated_val = float(value) * random.uniform(0.8, 1.2)
                    new_params[str(key)] = float(mutated_val)
                except Exception:
                    continue

        if not new_params:
            return genome

        # Use canonical mutate builder to produce a new instance with history
        return genome_mutate(genome, "optimizer_mutation", new_params)

    async def _optimize_coordination(
        self, populations: Dict[str, List[CanonDecisionGenome]], market_context: MarketContext
    ) -> Dict[str, List[CanonDecisionGenome]]:
        """Optimize coordination between species."""

        # Create coordination intents
        intents: List[TradeIntent] = []
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
                    timestamp=datetime.now(),
                )
                intents.append(intent)

        # Resolve coordination
        coordination_result = await self.coordination_engine.resolve_intents(
            intents, market_context
        )

        # Filter populations based on coordination
        coordinated_populations: Dict[str, List[CanonDecisionGenome]] = {}
        approved_strategies = {
            intent.strategy_id for intent in coordination_result.approved_intents
        }

        for species_type, population in populations.items():
            coordinated_populations[species_type] = [
                genome
                for genome in population
                if f"{species_type}_{id(genome)}" in approved_strategies
            ]

        return coordinated_populations

    async def _evaluate_ecosystem_performance(
        self, populations: Dict[str, List[CanonDecisionGenome]], performance_history: JSONObject
    ) -> EcosystemMetrics:
        """Evaluate overall ecosystem performance."""

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(populations)

        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(correlation_matrix)

        # Calculate synergy score
        synergy_score = self._calculate_synergy_score(populations, performance_history)

        # Calculate antifragility score
        antifragility_score = self._calculate_antifragility_score(populations, performance_history)

        # Mock performance metrics (would be calculated from actual trading)
        total_return = 0.15  # 15% annual return
        sharpe_ratio = 1.8
        max_drawdown = 0.08  # 8% max drawdown

        return EcosystemMetrics(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            correlation_matrix=correlation_matrix,
            diversification_ratio=float(diversification_ratio),
            synergy_score=float(synergy_score),
            antifragility_score=float(antifragility_score),
        )

    def _calculate_correlation_matrix(
        self, populations: Dict[str, List[CanonDecisionGenome]]
    ) -> NDArray[np.float64]:
        """Calculate correlation matrix between species."""
        species_list = list(populations.keys())
        n_species = len(species_list)

        correlation_matrix: NDArray[np.float64] = np.eye(n_species, dtype=np.float64)

        # Mock correlation values based on species characteristics
        correlation_map: Dict[Tuple[str, str], float] = {
            ("stalker", "alpha"): 0.7,
            ("stalker", "pack_hunter"): 0.5,
            ("stalker", "ambusher"): 0.2,
            ("stalker", "scavenger"): 0.3,
            ("alpha", "pack_hunter"): 0.6,
            ("alpha", "ambusher"): 0.1,
            ("alpha", "scavenger"): 0.4,
            ("pack_hunter", "ambusher"): 0.3,
            ("pack_hunter", "scavenger"): 0.5,
            ("ambusher", "scavenger"): 0.8,
        }

        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i != j:
                    pair: Tuple[str, str] = (
                        (species1, species2) if species1 < species2 else (species2, species1)
                    )
                    correlation_matrix[i, j] = float(correlation_map.get(pair, 0.5))

        return correlation_matrix

    def _calculate_diversification_ratio(self, correlation_matrix: NDArray[np.float64]) -> float:
        """Calculate diversification ratio from correlation matrix."""
        n_assets = int(len(correlation_matrix))
        if n_assets <= 1:
            return 0.0

        # Average correlation
        avg_correlation = (float(np.sum(correlation_matrix)) - n_assets) / (
            n_assets * (n_assets - 1)
        )

        # Diversification ratio (higher is better)
        diversification_ratio = 1.0 - float(avg_correlation)

        return max(0.0, min(1.0, float(diversification_ratio)))

    def _calculate_synergy_score(
        self, populations: Dict[str, List[CanonDecisionGenome]], performance_history: JSONObject
    ) -> float:
        """Calculate synergy score between species."""
        # Mock synergy calculation
        # In real implementation, would analyze complementary strategies
        species_count = len(populations)
        genome_count = sum(len(pop) for pop in populations.values())

        # Higher diversity and balanced populations indicate better synergy
        diversity_bonus = species_count / 5.0  # 5 species max
        std_val = (
            float(np.std([len(pop) for pop in populations.values()])) if genome_count > 0 else 0.0
        )
        denom = float(max(1, genome_count))
        balance_bonus = float(1.0 - std_val / denom)

        synergy_score = float((diversity_bonus + balance_bonus) / 2.0)

        return max(0.0, min(1.0, synergy_score))

    def _calculate_antifragility_score(
        self, populations: Dict[str, List[CanonDecisionGenome]], performance_history: JSONObject
    ) -> float:
        """Calculate antifragility score of the ecosystem."""
        # Mock antifragility calculation
        # In real implementation, would analyze performance under stress

        # Factors contributing to antifragility
        species_diversity = len(populations) / 5.0
        strategy_diversity = sum(len(pop) for pop in populations.values()) / 50.0

        # Adaptability indicators
        adaptability_score = 0.5

        antifragility_score = (species_diversity + strategy_diversity + adaptability_score) / 3

        return max(0.0, min(1.0, float(antifragility_score)))

    def _ensure_canonical(self, genome: object) -> CanonDecisionGenome:
        """Ensure any incoming genome is in canonical form."""
        if isinstance(genome, CanonDecisionGenome):
            return genome
        try:
            return cast(CanonDecisionGenome, adapt_to_canonical(genome))
        except Exception:
            # Minimal fallback canonical genome
            return CanonDecisionGenome.from_dict(
                {
                    "parameters": {},
                    "species_type": getattr(genome, "species_type", None),
                }
            )

    async def get_ecosystem_summary(self) -> EcosystemSummary:
        """Get summary of ecosystem state."""
        if not self.ecosystem_history:
            return {
                "total_optimizations": 0,
                "best_metrics": None,
                "current_species_distribution": {},
            }

        latest = self.ecosystem_history[-1]
        metrics = latest["metrics"]
        assert isinstance(metrics, EcosystemMetrics)
        populations = latest["populations"]
        assert isinstance(populations, dict)

        return {
            "total_optimizations": len(self.ecosystem_history),
            "best_metrics": {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "diversification_ratio": metrics.diversification_ratio,
                "synergy_score": metrics.synergy_score,
            },
            "current_species_distribution": {
                species: len(population) for species, population in populations.items()
            },
        }

    def _calculate_regime_bonus(self, market_regime: str) -> float:
        """Calculate regime-specific bonus based on actual market conditions."""
        regime_multipliers = {
            "trending": 1.2,
            "ranging": 0.9,
            "volatile": 1.1,
            "calm": 1.0,
            "crisis": 0.8,
        }
        return float(regime_multipliers.get(market_regime, 1.0))

    def _calculate_adaptability_score(
        self, genome: CanonDecisionGenome, market_data: Mapping[str, object]
    ) -> float:
        """Calculate adaptability score based on genome performance across market conditions."""
        if not hasattr(genome, "performance_metrics") or not genome.performance_metrics:
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
        return float(adaptability)


# Example usage
async def test_ecosystem_optimizer() -> None:
    """Test the ecosystem optimizer."""
    optimizer = EcosystemOptimizer()

    # Create test populations
    test_populations: Dict[str, List[CanonDecisionGenome]] = {
        "stalker": [],
        "ambusher": [],
        "pack_hunter": [],
        "scavenger": [],
        "alpha": [],
    }

    # Create test market context
    market_context = MarketContext(
        symbol="EURUSD",
        regime="trending_bull",
        volatility=0.02,
        trend_strength=0.8,
        volume_anomaly=1.2,
        data={"data": []},
    )

    # Create test performance history
    performance_history: JSONObject = {
        "returns": [0.01, 0.02, -0.01, 0.03, 0.01],
        "sharpe_ratios": [1.5, 1.8, 1.2, 2.1, 1.6],
    }

    # Optimize ecosystem
    optimized = await optimizer.optimize_ecosystem(
        cast(Mapping[str, Sequence["DecisionGenome"]], test_populations),
        market_context,
        performance_history,
    )

    print("Ecosystem Optimization Complete")
    print(f"Species distribution: { {k: len(v) for k, v in optimized.items()} }")

    summary = await optimizer.get_ecosystem_summary()
    print(f"Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(test_ecosystem_optimizer())
