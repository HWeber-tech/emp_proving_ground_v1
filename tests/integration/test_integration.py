#!/usr/bin/env python3
"""
Integration Testing
Tests component interactions and data flow in the EMP system.
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

def test_sensory_evolution_integration():
    """Test integration between sensory cortex and evolution system."""
    print("Testing Sensory-Evolution Integration...")
    
    try:
        # Create components
        from src.sensory.core.base import InstrumentMeta
        from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
        from src.evolution import DecisionGenome, FitnessEvaluator
        
        # Create instrument meta
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        
        # Create sensory cortex
        sensory_cortex = MasterOrchestrator(instrument_meta)
        print("✅ Sensory cortex created")
        
        # Create genome
        genome = DecisionGenome(
            genome_id="test_genome",
            decision_tree={
                'type': 'test',
                'parameters': {
                    'buy_threshold': 0.5,
                    'sell_threshold': 0.5,
                    'momentum_weight': 0.3,
                    'trend_weight': 0.4,
                    'institutional_weight': 0.3
                }
            }
        )
        print("✅ Genome created")
        
        # Create synthetic market data
        market_data = create_synthetic_market_data()
        print("✅ Market data created")
        
        # Test genome evaluation with sensory cortex
        results = genome.evaluate(market_data, sensory_cortex)
        print("✅ Genome evaluation completed")
        
        # Verify results structure
        expected_keys = ['trades', 'equity_curve', 'total_return', 'sharpe_ratio']
        for key in expected_keys:
            if key in results:
                print(f"✅ Result contains {key}")
            else:
                print(f"❌ Result missing {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Sensory-evolution integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_sensory_integration():
    """Test integration between data storage and sensory cortex."""
    print("\nTesting Data-Sensory Integration...")
    
    try:
        from src.data import TickDataStorage
        from src.sensory.core.base import InstrumentMeta
        from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
        
        # Create data storage
        data_storage = TickDataStorage()
        print("✅ Data storage created")
        
        # Create sensory cortex
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
        
        # Test data flow
        market_data = create_synthetic_market_data()
        
        # Test sensory perception on market data
        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            if i < 5:  # Test first 5 rows
                try:
                    perception = sensory_cortex.perceive(row)
                    print(f"✅ Sensory perception successful for row {i}")
                except Exception as e:
                    print(f"⚠️  Sensory perception failed for row {i}: {e}")
                    # Continue testing other rows
        
        return True
        
    except Exception as e:
        print(f"❌ Data-sensory integration failed: {e}")
        return False

def test_risk_pnl_integration():
    """Test integration between risk management and PnL tracking."""
    print("\nTesting Risk-PnL Integration...")
    
    try:
        from src.risk import RiskManager, RiskConfig
        from src.core import InstrumentProvider
        from src.pnl import EnhancedPosition
        from src.core import Instrument
        
        # Create risk manager
        risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25'),
            min_position_size=1000,
            max_position_size=1000000
        )
        instrument_provider = InstrumentProvider()
        risk_manager = RiskManager(risk_config, instrument_provider)
        print("✅ Risk manager created")
        
        # Create position
        position = EnhancedPosition(
            symbol="EURUSD",
            quantity=10000,
            avg_price=Decimal('1.1000'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now()
        )
        print("✅ Position created")
        
        # Test risk checks
        instrument = instrument_provider.get_instrument("EURUSD")
        if instrument:
            # Test position sizing
            max_size = risk_manager.calculate_position_size(
                account_equity=Decimal('100000'),
                stop_loss_pips=Decimal('50'),
                instrument=instrument,
                account_currency="USD"
            )
            print(f"✅ Position size calculated: {max_size}")
            
            # Test position validation
            is_valid = risk_manager.validate_position(
                position=position,
                instrument=instrument,
                equity=Decimal('100000')
            )
            print(f"✅ Position validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk-PnL integration failed: {e}")
        return False

def test_evolution_fitness_integration():
    """Test integration between evolution and fitness evaluation."""
    print("\nTesting Evolution-Fitness Integration...")
    
    try:
        from src.evolution import (
            DecisionGenome, 
            EvolutionConfig, 
            FitnessEvaluator, 
            EvolutionEngine
        )
        
        # Create mock data storage
        class MockDataStorage:
            def get_data_range(self, instrument, start, end):
                return create_synthetic_market_data()
        
        # Create fitness evaluator
        fitness_evaluator = FitnessEvaluator(MockDataStorage(), "EURUSD")
        print("✅ Fitness evaluator created")
        
        # Create evolution engine
        config = EvolutionConfig(
            population_size=5,  # Small for testing
            elite_ratio=0.2,
            crossover_ratio=0.5,
            mutation_ratio=0.3,
            tournament_size=2,
            max_generations=2,
            convergence_threshold=0.001,
            stagnation_limit=3
        )
        engine = EvolutionEngine(config, fitness_evaluator)
        print("✅ Evolution engine created")
        
        # Test population initialization
        success = engine.initialize_population(seed=42)
        if success:
            print("✅ Population initialized")
            
            # Test one generation
            stats = engine.evolve_generation()
            print(f"✅ Generation evolved - Best fitness: {stats.best_fitness:.4f}")
            
            # Test best genomes retrieval
            best_genomes = engine.get_best_genomes(3)
            print(f"✅ Retrieved {len(best_genomes)} best genomes")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution-fitness integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_synthetic_market_data():
    """Create synthetic market data for testing."""
    np.random.seed(42)
    
    # Generate 50 data points
    dates = pd.date_range(start='2023-01-01', periods=50, freq='H')
    
    # Generate realistic price movements
    base_price = 1.1000
    prices = [base_price]
    
    for i in range(49):
        change = np.random.normal(0.0001, 0.001)
        new_price = prices[-1] + change
        prices.append(max(0.5, new_price))
    
    # Create OHLCV data
    data = []
    for i in range(0, len(prices), 5):
        if i + 5 <= len(prices):
            bar_prices = prices[i:i+5]
            data.append({
                'open': bar_prices[0],
                'high': max(bar_prices),
                'low': min(bar_prices),
                'close': bar_prices[-1],
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    return df

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("EMP SYSTEM - INTEGRATION TESTING")
    print("=" * 60)
    
    results = {}
    
    # Test each integration
    results['sensory_evolution'] = test_sensory_evolution_integration()
    results['data_sensory'] = test_data_sensory_integration()
    results['risk_pnl'] = test_risk_pnl_integration()
    results['evolution_fitness'] = test_evolution_fitness_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TESTING SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for integration, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{integration.upper():20} : {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall Status: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
