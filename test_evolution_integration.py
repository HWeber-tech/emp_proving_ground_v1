#!/usr/bin/env python3
"""
Test script to verify evolution engine integration
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evolution.engine.genetic_engine import GeneticEngine
from src.evolution.engine.population_manager import PopulationManager
from src.evolution.selection.tournament_selection import TournamentSelection
from src.evolution.crossover.uniform_crossover import UniformCrossover
from src.evolution.mutation.gaussian_mutation import GaussianMutation
from src.genome.models.genome import DecisionGenome
from src.genome.factories.genome_factory import GenomeFactory


class MockFitnessEvaluator:
    """Mock fitness evaluator for testing"""
    
    def evaluate(self, genome, market_data=None):
        """Simple fitness evaluation based on genome parameters"""
        try:
            # Simple fitness based on strategy parameters
            fitness = 0.5  # Base fitness
            
            # Add some variation based on parameters
            if hasattr(genome, 'strategy') and genome.strategy:
                fitness += genome.strategy.entry_threshold * 0.1
                fitness += genome.strategy.momentum_weight * 0.2
                fitness += (100 - genome.strategy.lookback_period) * 0.001
            
            # Add some randomness
            import random
            fitness += random.uniform(-0.1, 0.1)
            
            return max(0.0, min(1.0, fitness))
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return 0.5


async def test_evolution_engine():
    """Test the complete evolution engine"""
    print("🧬 Testing Evolution Engine Integration...")
    
    try:
        # Create components
        population_manager = PopulationManager(population_size=10)
        selection_strategy = TournamentSelection(tournament_size=3)
        crossover_strategy = UniformCrossover(crossover_rate=0.8)
        mutation_strategy = GaussianMutation(mutation_strength=0.1)
        fitness_evaluator = MockFitnessEvaluator()
        genome_factory = GenomeFactory()
        
        # Create evolution engine
        engine = GeneticEngine(
            population_manager=population_manager,
            selection_strategy=selection_strategy,
            crossover_strategy=crossover_strategy,
            mutation_strategy=mutation_strategy,
            fitness_evaluator=fitness_evaluator,
            genome_factory=genome_factory,
            population_size=10,
            mutation_rate=0.1,
            elitism_count=2
        )
        
        print("✅ All components created successfully")
        
        # Initialize population
        await engine.initialize_population()
        print("✅ Population initialized")
        
        # Run one generation
        await engine.evolve_generation()
        print("✅ First generation evolved")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"✅ Statistics: {stats}")
        
        # Get best genome
        best_genome = engine.get_best_genome()
        print(f"✅ Best genome fitness: {best_genome.fitness if hasattr(best_genome, 'fitness') else 'N/A'}")
        
        print("\n🎉 Evolution Engine Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Evolution Engine Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_population_manager():
    """Test the population manager"""
    print("\n📊 Testing Population Manager...")
    
    try:
        pm = PopulationManager(population_size=5)
        
        # Test initialization
        class MockGenome:
            def __init__(self):
                self.fitness = 0.5
        
        pm.initialize_population(lambda: MockGenome())
        print(f"✅ Population initialized with {len(pm)} genomes")
        
        # Test statistics
        stats = pm.get_population_statistics()
        print(f"✅ Statistics: {stats}")
        
        # Test best genomes
        best = pm.get_best_genomes(2)
        print(f"✅ Best genomes retrieved: {len(best)}")
        
        print("✅ Population Manager Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Population Manager Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting Evolution System Integration Tests...\n")
    
    # Test population manager
    pm_result = await test_population_manager()
    
    # Test evolution engine
    engine_result = await test_evolution_engine()
    
    if pm_result and engine_result:
        print("\n🎊 ALL EVOLUTION TESTS PASSED!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
