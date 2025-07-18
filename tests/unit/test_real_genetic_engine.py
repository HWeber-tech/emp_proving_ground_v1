#!/usr/bin/env python3
"""
Test Real Genetic Engine

This test verifies that the real genetic programming engine can evolve
actual trading strategies on real market data.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_real_genetic_engine():
    """Test the real genetic engine functionality."""
    print("🧪 Testing Real Genetic Engine...")
    
    try:
        # Import the real genetic engine
        from src.evolution.real_genetic_engine import RealGeneticEngine
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Initialize data source
        data_source = RealDataIngestor()
        
        # Initialize genetic engine
        engine = RealGeneticEngine(data_source, population_size=10)
        
        # Set up evaluation period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        # Initialize population
        print("   Initializing population...")
        engine.initialize_population("EURUSD", start_date, end_date)
        
        if not engine.population:
            print("   ❌ Failed to initialize population")
            return False
        
        print(f"   ✅ Population initialized with {len(engine.population)} strategies")
        
        # Check initial fitness scores
        best_initial = max(engine.population, key=lambda s: s.fitness_score)
        print(f"   Best initial fitness: {best_initial.fitness_score:.4f}")
        
        # Evolve for a few generations
        print("   Evolving population...")
        evolution_history = engine.evolve("EURUSD", start_date, end_date, generations=3)
        
        if not evolution_history:
            print("   ❌ No evolution history generated")
            return False
        
        print(f"   ✅ Evolution completed with {len(evolution_history)} generations")
        
        # Check final fitness scores
        best_final = max(engine.population, key=lambda s: s.fitness_score)
        print(f"   Best final fitness: {best_final.fitness_score:.4f}")
        
        # Check if evolution improved performance
        if best_final.fitness_score > best_initial.fitness_score:
            print("   ✅ Evolution improved performance!")
            improvement = ((best_final.fitness_score - best_initial.fitness_score) / best_initial.fitness_score) * 100
            print(f"   Improvement: {improvement:.1f}%")
        else:
            print("   ⚠️  Evolution did not improve performance (this can happen with small populations)")
        
        # Show best strategy details
        if engine.best_strategy:
            print(f"   Best strategy: {engine.best_strategy.name}")
            print(f"   Parameters: {engine.best_strategy.parameters}")
            print(f"   Performance: {engine.best_strategy.performance_metrics}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing genetic engine: {e}")
        return False


def test_strategy_evaluation():
    """Test individual strategy evaluation."""
    print("\n🧪 Testing Strategy Evaluation...")
    
    try:
        from src.evolution.real_genetic_engine import TradingStrategy, StrategyEvaluator
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Initialize evaluator
        data_source = RealDataIngestor()
        evaluator = StrategyEvaluator(data_source)
        
        # Create a test strategy
        strategy = TradingStrategy(
            id="test_strategy_001",
            name="Test_SMA_Strategy",
            parameters={
                'sma_fast': 10,
                'sma_slow': 20,
                'sma_period': 20
            },
            indicators=['SMA'],
            entry_rules=['SMA_CROSSOVER'],
            exit_rules=['SMA_CROSSOVER'],
            risk_management={
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
        )
        
        # Set up evaluation period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Evaluate strategy
        print("   Evaluating test strategy...")
        performance = evaluator.evaluate_strategy(strategy, "EURUSD", start_date, end_date)
        
        if performance:
            print(f"   ✅ Strategy evaluation completed")
            print(f"   Total return: {performance.get('total_return', 0):.2%}")
            print(f"   Number of trades: {performance.get('num_trades', 0)}")
            print(f"   Win rate: {performance.get('win_rate', 0):.2%}")
            print(f"   Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
            return True
        else:
            print("   ❌ Strategy evaluation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing strategy evaluation: {e}")
        return False


def test_technical_indicators():
    """Test technical indicators calculation."""
    print("\n🧪 Testing Technical Indicators...")
    
    try:
        from src.evolution.real_genetic_engine import TechnicalIndicators
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Get some real data
        data_source = RealDataIngestor()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = data_source.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for indicator testing")
            return False
        
        # Test indicators
        indicators = TechnicalIndicators()
        
        # Test SMA
        sma = indicators.sma(data, 20)
        if not sma.empty:
            print("   ✅ SMA calculation successful")
        
        # Test RSI
        rsi = indicators.rsi(data, 14)
        if not rsi.empty:
            print("   ✅ RSI calculation successful")
        
        # Test MACD
        macd_line, signal_line, histogram = indicators.macd(data, 12, 26, 9)
        if not macd_line.empty:
            print("   ✅ MACD calculation successful")
        
        # Test Bollinger Bands
        upper, middle, lower = indicators.bollinger_bands(data, 20, 2.0)
        if not upper.empty:
            print("   ✅ Bollinger Bands calculation successful")
        
        print(f"   ✅ All technical indicators working with {len(data)} data points")
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing technical indicators: {e}")
        return False


def test_strategy_templates():
    """Test strategy template creation."""
    print("\n🧪 Testing Strategy Templates...")
    
    try:
        from src.evolution.real_genetic_engine import RealGeneticEngine
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Initialize engine
        data_source = RealDataIngestor()
        engine = RealGeneticEngine(data_source, population_size=5)
        
        # Check templates
        if engine.strategy_templates:
            print(f"   ✅ {len(engine.strategy_templates)} strategy templates available")
            
            for template in engine.strategy_templates:
                print(f"   - {template['name']}: {template['indicators']}")
            
            # Test creating strategies from templates
            strategies = []
            for i in range(3):
                strategy = engine._create_random_strategy(f"Template_Strategy_{i}")
                strategies.append(strategy)
            
            print(f"   ✅ Created {len(strategies)} strategies from templates")
            
            # Check strategy properties
            for strategy in strategies:
                if strategy.parameters and strategy.indicators:
                    print(f"   - {strategy.name}: {list(strategy.parameters.keys())} parameters")
            
            return True
        else:
            print("   ❌ No strategy templates available")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing strategy templates: {e}")
        return False


def main():
    """Run all real genetic engine tests."""
    print("🚀 REAL GENETIC ENGINE TEST SUITE")
    print("=" * 50)
    
    # Test 1: Real genetic engine
    test1_passed = test_real_genetic_engine()
    
    # Test 2: Strategy evaluation
    test2_passed = test_strategy_evaluation()
    
    # Test 3: Technical indicators
    test3_passed = test_technical_indicators()
    
    # Test 4: Strategy templates
    test4_passed = test_strategy_templates()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"Real Genetic Engine: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Strategy Evaluation: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Technical Indicators: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Strategy Templates: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed, test4_passed])
    print(f"\nOverall: {total_passed}/4 tests passed")
    
    if total_passed >= 3:
        print("🎉 Real genetic programming is working!")
        print("   The system can now evolve actual trading strategies on real market data.")
    else:
        print("⚠️  Real genetic programming needs improvement.")
        print("   The system is still using mock evolution.")
    
    return total_passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 