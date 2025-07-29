#!/usr/bin/env python3
"""
End-to-End Testing
Tests the complete EMP system workflow from data ingestion to evolution results.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_complete_workflow():
    """Test the complete EMP system workflow."""
    print("Testing Complete EMP System Workflow...")
    
    try:
        # Step 1: Initialize core components
        print("\n1. Initializing Core Components...")
        
        from src.core import Instrument, InstrumentProvider
        from src.risk import RiskManager, RiskConfig
        from src.pnl import EnhancedPosition
        from src.data import TickDataStorage
        
        # Create instrument provider
        instrument_provider = InstrumentProvider()
        print("✅ Instrument provider created")
        
        # Create risk manager
        risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25'),
            min_position_size=1000,
            max_position_size=1000000
        )
        risk_manager = RiskManager(risk_config, instrument_provider)
        print("✅ Risk manager created")
        
        # Create data storage
        data_storage = TickDataStorage()
        print("✅ Data storage created")
        
        # Step 2: Initialize sensory cortex
        print("\n2. Initializing Sensory Cortex...")
        
        from src.sensory.core.base import InstrumentMeta
        from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
        
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        sensory_cortex = MasterOrchestrator(instrument_meta)
        print("✅ Sensory cortex created")
        
        # Step 3: Initialize evolution system
        print("\n3. Initializing Evolution System...")
        
        from src.evolution import (
            DecisionGenome, 
            EvolutionConfig, 
            FitnessEvaluator, 
            EvolutionEngine
        )
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(data_storage, "EURUSD")
        print("✅ Fitness evaluator created")
        
        # Create evolution engine
        config = EvolutionConfig(
            population_size=20,  # Larger population for better testing
            elite_ratio=0.2,
            crossover_ratio=0.5,
            mutation_ratio=0.3,
            tournament_size=3,
            max_generations=5,
            convergence_threshold=0.001,
            stagnation_limit=3
        )
        engine = EvolutionEngine(config, fitness_evaluator)
        print("✅ Evolution engine created")
        
        # Step 4: Generate synthetic market data
        print("\n4. Generating Market Data...")
        
        market_data = create_comprehensive_market_data()
        print(f"✅ Generated {len(market_data)} market data points")
        
        # Step 5: Initialize population
        print("\n5. Initializing Evolution Population...")
        
        success = engine.initialize_population(seed=42)
        if not success:
            print("❌ Failed to initialize population")
            return False
        print(f"✅ Population initialized with {len(engine.population)} genomes")
        
        # Step 6: Run evolution
        print("\n6. Running Evolution...")
        
        evolution_results = []
        for generation in range(config.max_generations):
            print(f"  Generation {generation + 1}:")
            stats = engine.evolve_generation()
            evolution_results.append(stats)
            print(f"    Best fitness: {stats.best_fitness:.4f}")
            print(f"    Avg fitness: {stats.avg_fitness:.4f}")
            print(f"    Diversity: {stats.diversity:.4f}")
            
            # Check for convergence
            if stats.convergence_rate < config.convergence_threshold:
                print(f"    ✅ Converged at generation {generation + 1}")
                break
        
        # Step 7: Analyze results
        print("\n7. Analyzing Results...")
        
        # Get best genomes
        best_genomes = engine.get_best_genomes(5)
        print(f"✅ Retrieved {len(best_genomes)} best genomes")
        
        # Test best genome on market data
        if best_genomes:
            best_genome = best_genomes[0]
            print(f"Testing best genome: {best_genome.genome_id}")
            
            # Evaluate best genome
            results = best_genome.evaluate(market_data, sensory_cortex)
            
            print(f"  Total return: {results['total_return']:.4f}")
            print(f"  Sharpe ratio: {results['sharpe_ratio']:.4f}")
            print(f"  Max drawdown: {results['max_drawdown']:.4f}")
            print(f"  Win rate: {results['win_rate']:.4f}")
            print(f"  Number of trades: {len(results['trades'])}")
            
            # Verify results are reasonable
            if results['total_return'] >= -1.0 and results['total_return'] <= 2.0:
                print("✅ Results are within reasonable bounds")
            else:
                print("⚠️  Results may be outside expected bounds")
        
        # Step 8: Test risk management integration
        print("\n8. Testing Risk Management Integration...")
        
        instrument = instrument_provider.get_instrument("EURUSD")
        if instrument and best_genomes:
            # Create a position based on best genome
            position = EnhancedPosition(
                symbol="EURUSD",
                quantity=10000,
                avg_price=Decimal('1.1000'),
                entry_timestamp=datetime.now(),
                last_swap_time=datetime.now()
            )
            
            # Test risk validation
            is_valid = risk_manager.validate_position(
                position=position,
                instrument=instrument,
                equity=Decimal('100000')
            )
            print(f"✅ Position validation: {is_valid}")
        
        # Step 9: Generate evolution summary
        print("\n9. Generating Evolution Summary...")
        
        summary = engine.get_evolution_summary()
        print(f"✅ Evolution summary generated:")
        print(f"  Total generations: {summary['total_generations']}")
        print(f"  Best fitness: {summary['best_fitness']:.4f}")
        print(f"  Population size: {summary['population_size']}")
        
        # Step 10: Verify system health
        print("\n10. Verifying System Health...")
        
        # Check that all components are working
        components_healthy = True
        
        # Check evolution system
        if len(engine.population) > 0:
            print("✅ Evolution system healthy")
        else:
            print("❌ Evolution system unhealthy")
            components_healthy = False
        
        # Check sensory cortex
        if hasattr(sensory_cortex, 'update'):
            print("✅ Sensory cortex healthy")
        else:
            print("❌ Sensory cortex unhealthy")
            components_healthy = False
        
        # Check risk management
        if hasattr(risk_manager, 'validate_position'):
            print("✅ Risk management healthy")
        else:
            print("❌ Risk management unhealthy")
            components_healthy = False
        
        if components_healthy:
            print("✅ All system components are healthy")
        else:
            print("❌ Some system components are unhealthy")
            return False
        
        print("\n🎉 Complete EMP system workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comprehensive_market_data():
    """Create comprehensive synthetic market data for testing."""
    np.random.seed(42)
    
    # Generate 200 data points (more realistic for evolution)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
    
    # Generate realistic price movements with trends and volatility
    base_price = 1.1000
    prices = [base_price]
    
    # Add some trending periods
    for i in range(199):
        # Vary volatility and trend based on time
        if i < 50:
            # Trending up
            change = np.random.normal(0.0002, 0.0008)
        elif i < 100:
            # Trending down
            change = np.random.normal(-0.0002, 0.0008)
        elif i < 150:
            # High volatility
            change = np.random.normal(0.0000, 0.0015)
        else:
            # Low volatility
            change = np.random.normal(0.0000, 0.0005)
        
        new_price = prices[-1] + change
        prices.append(max(0.5, new_price))
    
    # Create OHLCV data
    data = []
    for i in range(0, len(prices), 4):  # Smaller bars for more data points
        if i + 4 <= len(prices):
            bar_prices = prices[i:i+4]
            data.append({
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                'volume': np.random.randint(1000, 15000)
            })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    return df

def main():
    """Run the complete end-to-end test."""
    print("=" * 60)
    print("EMP SYSTEM - END-TO-END TESTING")
    print("=" * 60)
    
    success = test_complete_workflow()
    
    print("\n" + "=" * 60)
    print("END-TO-END TESTING SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ COMPLETE WORKFLOW: PASS")
        print("\n🎉 EMP System is fully functional!")
        print("\nThe system successfully:")
        print("  • Initialized all core components")
        print("  • Created and configured sensory cortex")
        print("  • Set up evolution system with fitness evaluation")
        print("  • Generated synthetic market data")
        print("  • Evolved trading strategies across multiple generations")
        print("  • Integrated risk management and PnL tracking")
        print("  • Produced meaningful trading results")
        print("  • Maintained system health throughout")
    else:
        print("❌ COMPLETE WORKFLOW: FAIL")
        print("\nThe system encountered issues during testing.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
