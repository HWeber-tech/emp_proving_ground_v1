#!/usr/bin/env python3
"""
Integration test for EMP Proving Ground v1.0

This script tests that all components can be imported and initialized correctly.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all components can be imported."""
    print("Testing component imports...")
    
    try:
        # Test data pipeline imports
        from emp.data.ingestion import DukascopyIngestor
        from emp.data.cleaning import TickDataCleaner
        from emp.data.storage import TickDataStorage
        from emp.data.regimes import MarketRegimeIdentifier
        print("✓ Data pipeline components imported")
        
        # Test simulation imports
        from emp.simulation.market import MarketSimulator
        from emp.simulation.adversary import AdversarialEngine
        print("✓ Simulation components imported")
        
        # Test agent imports
        from emp.agent.sensory import SensoryCortex
        from emp.agent.genome import DecisionGenome
        print("✓ Agent components imported")
        
        # Test evolution imports
        from emp.evolution.fitness import FitnessEvaluator, EvolutionConfig
        from emp.evolution.engine import EvolutionEngine
        print("✓ Evolution components imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_initialization():
    """Test that all components can be initialized."""
    print("\nTesting component initialization...")
    
    try:
        # Initialize data components
        storage = TickDataStorage()
        cleaner = TickDataCleaner()
        ingestor = DukascopyIngestor(storage, cleaner)
        regime_identifier = MarketRegimeIdentifier(storage)
        print("✓ Data components initialized")
        
        # Initialize simulation components
        simulator = MarketSimulator(storage)
        adversary = AdversarialEngine()
        print("✓ Simulation components initialized")
        
        # Initialize agent components
        sensory_cortex = SensoryCortex("EURUSD", storage)
        genome = DecisionGenome()
        print("✓ Agent components initialized")
        
        # Initialize evolution components
        fitness_evaluator = FitnessEvaluator(storage)
        config = EvolutionConfig(population_size=10)
        evolution_engine = EvolutionEngine(config, fitness_evaluator)
        print("✓ Evolution components initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test data storage
        storage = TickDataStorage()
        print("✓ Data storage created")
        
        # Test genome creation and mutation
        genome = DecisionGenome()
        mutated_genome = genome.mutate(0.1)
        print("✓ Genome mutation works")
        
        # Test genome crossover
        genome2 = DecisionGenome()
        offspring1, offspring2 = genome.crossover(genome2)
        print("✓ Genome crossover works")
        
        # Test genome complexity calculation
        complexity = genome.get_complexity()
        print(f"✓ Genome complexity: {complexity}")
        
        # Test sensory cortex
        sensory_cortex = SensoryCortex("EURUSD", storage)
        print("✓ Sensory cortex created")
        
        # Test adversarial engine
        adversary = AdversarialEngine()
        print("✓ Adversarial engine created")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("EMP Proving Ground v1.0 - Integration Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed")
        return False
    
    # Test initialization
    if not test_initialization():
        print("\n❌ Initialization test failed")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality test failed")
        return False
    
    print("\n✅ All integration tests passed!")
    print("\nThe EMP Proving Ground v1.0 system is ready to use.")
    print("\nNext steps:")
    print("1. Download market data: python run_evolution.py --mode download")
    print("2. Run evolution: python run_evolution.py --mode evolution")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 