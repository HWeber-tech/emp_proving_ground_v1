#!/usr/bin/env python3
"""
Test script to verify the evolution system fixes.
"""

import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evolution import (
    DecisionGenome, 
    EvolutionConfig, 
    FitnessEvaluator, 
    EvolutionEngine
)

# Mock data storage for testing
class MockDataStorage:
    """Mock data storage that returns None to trigger synthetic data creation."""
    def __init__(self):
        self.data = {}
    
    def get_data_range(self, instrument, start_date, end_date):
        """Return None to trigger synthetic data creation."""
        return None

def test_evolution_system():
    """Test the evolution system with fixes."""
    print("Testing Evolution System Fixes...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create mock data storage
        mock_storage = MockDataStorage()
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(mock_storage, instrument="EURUSD")
        
        # Create evolution config
        config = EvolutionConfig(
            population_size=10,  # Small for testing
            elite_ratio=0.2,
            crossover_ratio=0.5,
            mutation_ratio=0.3,
            tournament_size=2,
            max_generations=3,  # Small for testing
            convergence_threshold=0.001,
            stagnation_limit=5
        )
        
        # Create evolution engine
        engine = EvolutionEngine(config, fitness_evaluator)
        
        # Initialize population
        print("Initializing population...")
        success = engine.initialize_population(seed=42)
        if not success:
            print("❌ Failed to initialize population")
            return False
        
        print(f"✅ Population initialized with {len(engine.population)} genomes")
        
        # Evolve for a few generations
        print("\nEvolving population...")
        for generation in range(3):
            print(f"Generation {generation + 1}:")
            stats = engine.evolve_generation()
            print(f"  Best fitness: {stats.best_fitness:.4f}")
            print(f"  Avg fitness: {stats.avg_fitness:.4f}")
            print(f"  Diversity: {stats.diversity:.4f}")
        
        # Get best genomes
        best_genomes = engine.get_best_genomes(3)
        print(f"\n✅ Top 3 genomes:")
        for i, genome in enumerate(best_genomes):
            print(f"  {i+1}. {genome.genome_id}: fitness={genome.fitness_score:.4f}")
        
        # Get evolution summary
        summary = engine.get_evolution_summary()
        print(f"\n✅ Evolution summary:")
        print(f"  Total generations: {summary['total_generations']}")
        print(f"  Best fitness: {summary['best_fitness']:.4f}")
        print(f"  Population size: {summary['population_size']}")
        
        print("\n🎉 Evolution system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evolution_system()
    sys.exit(0 if success else 1) 