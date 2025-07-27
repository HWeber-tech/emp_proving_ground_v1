#!/usr/bin/env python3
"""
Simple test script to verify evolution engine integration
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


class MockFitnessEvaluator:
    """Mock fitness evaluator for testing"""
    
    def evaluate(self, genome, market_data=None):
        """Simple fitness evaluation based on genome parameters"""
        try:
            # Simple fitness based on genome ID hash
            import hashlib
            fitness = (int(hashlib.md5(genome.genome_id.encode()).hexdigest(), 16) % 1000) / 1000.0
            return fitness
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return 0.5


class MockGenomeFactory:
    """Mock genome factory for testing"""
    
    def create_genome(self):
        """Create a new mock genome"""
        import uuid
        genome = DecisionGenome(genome_id=str(uuid.uuid4()))
        genome.fitness_score = 0.0  # Will be set by evaluator
        return genome


async def test_evolution_engine():
    """Test the complete evolution engine"""
    print("🧬 Testing Evolution Engine Integration...")
    
    try:
        # Create components
        population_manager = PopulationManager(population_size=5)
        selection_strategy = TournamentSelection(tournament_size=3)
        crossover_strategy = UniformCrossover(crossover_rate=0.8)
        mutation_strategy = GaussianMutation(mutation_strength=0.1)
        fitness_evaluator = MockFitnessEvaluator()
        genome_factory = MockGenomeFactory()
        
        # Create evolution engine
        from src.evolution.engine.genetic_engine import EvolutionConfig
        
        config = EvolutionConfig(
            population_size=5,
            elite_count=1,
            crossover_rate=0.8,
            mutation_rate=0.1,
            max_generations=10
        )
        
        engine = GeneticEngine(
            population_manager=population_manager,
            selection_strategy=selection_strategy,
            crossover_strategy=crossover_strategy,
            mutation_strategy=mutation_strategy,
            fitness_evaluator=fitness_evaluator,
            genome_factory=genome_factory,
            config=config
        )
        
        print("✅ All components created successfully")
        
        # Initialize population
        engine.initialize_population()
        print("✅ Population initialized")
        
        # Run one generation
        stats = engine.evolve_generation()
        print(f"✅ First generation evolved: {stats}")
        
        # Get best genome
        best_genome = engine.get_best_genome()
        print(f"✅ Best genome fitness: {engine.get_best_fitness():.4f}")
        
        print("\n🎉 Evolution Engine Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Evolution Engine Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting Evolution System Integration Tests...\n")
    
    # Test evolution engine
    engine_result = await test_evolution_engine()
    
    if engine_result:
        print("\n🎊 ALL EVOLUTION TESTS PASSED!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
